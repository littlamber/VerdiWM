#!/usr/bin/env python3
"""Screen typed anchor-repair primitives on frozen Ctrl-World replay contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from scripts.run_ctrl_world_local_fingerprint_probe import (
        LocalFingerprintProbeError,
        _load_rollout_module,
        _metrics,
        _outcome_difference,
        _outcome_vector,
        _paired_grid,
        _runtime_args,
        _set_seed,
        load_contexts,
    )
except ModuleNotFoundError:  # Direct execution places ``scripts`` on sys.path.
    from run_ctrl_world_local_fingerprint_probe import (  # type: ignore[no-redef]
        LocalFingerprintProbeError,
        _load_rollout_module,
        _metrics,
        _outcome_difference,
        _outcome_vector,
        _paired_grid,
        _runtime_args,
        _set_seed,
        load_contexts,
    )


OUTCOME_NAMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class AnchorRepairError(LocalFingerprintProbeError):
    """A typed anchor repair could not be executed faithfully."""


class AnchorRepairPrimitive:
    """Base contract for a bounded, inference-only Ctrl-World repair."""

    primitive_id = "abstract"
    target_hook = "abstract"

    def __init__(self, strength: float) -> None:
        if not math.isfinite(strength) or strength < 0.0 or strength >= 1.0:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_STRENGTH_INVALID")
        self.strength = float(strength)
        self._invocations = 0
        self._active_invocations = 0
        self._delta_sum = 0.0
        self._maximum_delta = 0.0
        self._effective_strengths: list[float] = []

    def transform_conditions(
        self,
        *,
        history: torch.Tensor,
        current: torch.Tensor,
        first: torch.Tensor,
        interaction: int,
        interaction_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del first, interaction, interaction_count
        return history, self._record(current, current, 0.0)

    def _record(
        self, before: torch.Tensor, after: torch.Tensor, effective_strength: float
    ) -> torch.Tensor:
        if before.shape != after.shape:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_SHAPE_CHANGED")
        delta = (after - before).abs()
        maximum = float(delta.max().item())
        self._invocations += 1
        self._active_invocations += int(maximum > 0.0)
        self._delta_sum += float(delta.mean().item())
        self._maximum_delta = max(self._maximum_delta, maximum)
        self._effective_strengths.append(float(effective_strength))
        return after

    def result(self) -> dict[str, object]:
        count = max(self._invocations, 1)
        return {
            "invocation_count": self._invocations,
            "active_invocation_count": self._active_invocations,
            "mean_abs_tensor_delta": self._delta_sum / count,
            "maximum_abs_tensor_delta": self._maximum_delta,
            "effective_strengths": self._effective_strengths,
        }


class FixedNegativeAnchorControl(AnchorRepairPrimitive):
    """Positive control: apply the already observed negative anchor direction."""

    primitive_id = "fixed_negative_anchor_control"
    target_hook = "current_image_condition"

    def transform_conditions(
        self,
        *,
        history: torch.Tensor,
        current: torch.Tensor,
        first: torch.Tensor,
        interaction: int,
        interaction_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del interaction, interaction_count
        if current.shape != first.shape:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_FIRST_SHAPE_INVALID")
        transformed = current + self.strength * (current - first)
        return history, self._record(current, transformed, self.strength)


class FirstFrameConditioningDecay(AnchorRepairPrimitive):
    """Increase first-frame attenuation as autoregressive interaction age grows."""

    primitive_id = "first_frame_conditioning_decay"
    target_hook = "current_image_condition"

    def transform_conditions(
        self,
        *,
        history: torch.Tensor,
        current: torch.Tensor,
        first: torch.Tensor,
        interaction: int,
        interaction_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current.shape != first.shape or interaction_count < 2:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_FIRST_SHAPE_INVALID")
        progress = float(interaction) / float(interaction_count - 1)
        effective = self.strength * progress
        transformed = current + effective * (current - first)
        return history, self._record(current, transformed, effective)


class RecencyAnchorBalance(AnchorRepairPrimitive):
    """Move older history anchors toward current state with an age-weighted blend."""

    primitive_id = "recency_anchor_balance"
    target_hook = "history_condition"

    def transform_conditions(
        self,
        *,
        history: torch.Tensor,
        current: torch.Tensor,
        first: torch.Tensor,
        interaction: int,
        interaction_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del first, interaction, interaction_count
        if history.ndim != 5 or current.ndim != 4 or history.shape[0] != current.shape[0]:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_HISTORY_SHAPE_INVALID")
        if history.shape[2:] != current.shape[1:] or history.shape[1] < 2:
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_HISTORY_SHAPE_INVALID")
        age_weight = torch.linspace(
            1.0, 0.0, history.shape[1], device=history.device, dtype=history.dtype
        ).view(1, history.shape[1], 1, 1, 1)
        transformed = history + self.strength * age_weight * (current.unsqueeze(1) - history)
        return self._record(history, transformed, self.strength), current


PRIMITIVES = {
    cls.primitive_id: cls
    for cls in (FixedNegativeAnchorControl, FirstFrameConditioningDecay, RecencyAnchorBalance)
}


def _run_context(
    *,
    rollout_agent: object,
    runtime_args: object,
    context: Mapping[str, object],
    primitive_id: str | None,
    strength: float,
) -> tuple[dict[str, object], np.ndarray]:
    _set_seed(int(context["seed"]))
    pred_step = int(getattr(runtime_args, "pred_step"))
    interact_num = int(getattr(runtime_args, "interact_num"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    eef_gt, _joint_pos_gt, _video_dict, video_latents, instruction = rollout_agent.get_traj_info(
        str(context["episode_id"]),
        start_idx=int(context["start_idx"]),
        steps=int(pred_step * interact_num + 8),
    )
    first_latent = torch.cat([value[0] for value in video_latents], dim=1).unsqueeze(0)
    if tuple(first_latent.shape) != (1, 4, 72, 40):
        raise AnchorRepairError(f"CTRL_WORLD_ANCHOR_REPAIR_INITIAL_SHAPE:{tuple(first_latent.shape)}")
    history_latents = [first_latent for _ in range(num_history * 4)]
    history_eef = [eef_gt[0:1] for _ in range(num_history * 4)]
    primitive_cls = PRIMITIVES.get(primitive_id) if primitive_id else AnchorRepairPrimitive
    primitive = primitive_cls(strength)
    all_true: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    interactions: list[dict[str, object]] = []
    for interaction in range(interact_num):
        start_id = int(interaction * (pred_step - 1))
        end_id = start_id + pred_step
        target_latents = [value[start_id:end_id] for value in video_latents]
        cartesian_pose = eef_gt[start_id:end_id]
        if cartesian_pose.shape != (pred_step, 7):
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_ACTION_WINDOW_INVALID")
        history_idx = [0, 0, -8, -6, -4, -2]
        history_pose = np.concatenate([history_eef[index] for index in history_idx], axis=0)
        action_cond = np.concatenate((history_pose, cartesian_pose), axis=0)
        history_input = torch.cat([history_latents[index] for index in history_idx], dim=0).unsqueeze(0)
        current_latent = history_latents[-1]
        if action_cond.shape != (num_history + num_frames, 7):
            raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_CONDITION_WINDOW_INVALID")
        history_input, current_latent = primitive.transform_conditions(
            history=history_input,
            current=current_latent,
            first=first_latent,
            interaction=interaction,
            interaction_count=interact_num,
        )
        videos_cat, true_video, prediction, predicted_latents = rollout_agent.forward_wm(
            action_cond,
            target_latents,
            current_latent,
            his_cond=history_input,
            text=instruction if bool(getattr(runtime_args, "text_cond")) else None,
        )
        del videos_cat
        metric = _metrics(true_video, prediction)
        metric["interaction"] = interaction
        interactions.append(metric)
        all_true.append(true_video)
        all_prediction.append(prediction)
        history_eef.append(cartesian_pose[pred_step - 1 : pred_step])
        history_latents.append(
            torch.cat([value[pred_step - 1] for value in predicted_latents], dim=1).unsqueeze(0)
        )
    outcome = _outcome_vector(interactions)
    if not all(math.isfinite(value) for value in outcome.values()):
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_OUTCOME_NONFINITE")
    row = {
        "identity": {
            "context_id": str(context["context_id"]),
            "episode_id": str(context["episode_id"]),
            "start_idx": int(context["start_idx"]),
            "seed": int(context["seed"]),
        },
        "interactions": interactions,
        "outcomes": outcome,
        "hook_audit": primitive.result(),
    }
    visual = _paired_grid(np.concatenate(all_true, axis=1), np.concatenate(all_prediction, axis=1))
    return row, visual


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_OUTPUT_EXISTS")
    if args.primitive_id not in PRIMITIVES or int(args.interact_num) < 2:
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_ARGUMENT_INVALID")
    strengths = tuple(float(value) for value in args.strengths)
    if 0.0 not in strengths or len(strengths) < 2 or len(strengths) != len(set(strengths)):
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_STRENGTH_GRID_INVALID")
    if any(not math.isfinite(value) or value < 0.0 or value >= 1.0 for value in strengths):
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_STRENGTH_GRID_INVALID")

    ctrl_world_root = Path(args.ctrl_world_root).resolve(strict=True)
    contexts_path = Path(args.contexts_json).resolve(strict=True)
    contexts = load_contexts(contexts_path, args.seeds, args.context_ids)
    module = _load_rollout_module(ctrl_world_root)
    runtime_args = _runtime_args(
        dataset_root=Path(args.dataset_root).resolve(strict=True),
        data_stat=Path(args.data_stat).resolve(strict=True),
        svd_model_path=Path(args.svd_model_path).resolve(strict=True),
        clip_model_path=Path(args.clip_model_path).resolve(strict=True),
        ckpt_path=Path(args.ckpt_path).resolve(strict=True),
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
        enable_signed_history_correction=False,
        unsigned_history_gate=False,
        enable_multiscale_history_adapter=False,
        multiscale_history_always_on=False,
    )
    output_root.mkdir(mode=0o700, parents=True)
    rollout_agent = module.agent(runtime_args)

    references: dict[tuple[str, int], Mapping[str, object]] = {}
    for context in contexts:
        row, _visual = _run_context(
            rollout_agent=rollout_agent,
            runtime_args=runtime_args,
            context=context,
            primitive_id=None,
            strength=0.0,
        )
        references[(str(context["context_id"]), int(context["seed"]))] = row

    measurements: list[dict[str, object]] = []
    zero_identity_checks: list[dict[str, object]] = []
    visual_sample: np.ndarray | None = None
    for strength in (0.0,) + tuple(value for value in strengths if value != 0.0):
        for context in contexts:
            row, visual = _run_context(
                rollout_agent=rollout_agent,
                runtime_args=runtime_args,
                context=context,
                primitive_id=str(args.primitive_id),
                strength=strength,
            )
            row["strength"] = strength
            measurements.append(row)
            if strength == 0.0:
                key = (str(context["context_id"]), int(context["seed"]))
                difference = _outcome_difference(
                    references[key]["outcomes"], row["outcomes"]  # type: ignore[arg-type]
                )
                audit = row["hook_audit"]
                passed = (
                    difference <= float(args.zero_identity_tolerance)
                    and int(audit["invocation_count"]) == int(args.interact_num)
                    and float(audit["maximum_abs_tensor_delta"]) == 0.0
                )
                zero_identity_checks.append(
                    {
                        "identity": row["identity"],
                        "maximum_outcome_abs_difference": difference,
                        "tolerance": float(args.zero_identity_tolerance),
                        "state": "passed" if passed else "failed",
                    }
                )
                if visual_sample is None:
                    visual_sample = visual

    nonzero = [row for row in measurements if float(row["strength"]) > 0.0]
    hook_active = all(
        int(row["hook_audit"]["invocation_count"]) == int(args.interact_num)
        and int(row["hook_audit"]["active_invocation_count"]) > 0
        and float(row["hook_audit"]["maximum_abs_tensor_delta"]) > 0.0
        for row in nonzero
    )
    zero_passed = all(row["state"] == "passed" for row in zero_identity_checks)
    if visual_sample is None:
        raise AnchorRepairError("CTRL_WORLD_ANCHOR_REPAIR_VIDEO_MISSING")
    primitive_cls = PRIMITIVES[str(args.primitive_id)]
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-result",
        "state": "ready" if hook_active and zero_passed else "failed",
        "campaign_id": str(args.campaign_id),
        "primitive_id": str(args.primitive_id),
        "primitive_type": "inference_conditioning",
        "target_hook": primitive_cls.target_hook,
        "outcome_names": list(OUTCOME_NAMES),
        "input": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "contexts_json": str(contexts_path),
            "contexts_sha256": hashlib.sha256(contexts_path.read_bytes()).hexdigest(),
            "strengths": [0.0] + [value for value in strengths if value != 0.0],
            "interact_num": int(args.interact_num),
            "num_inference_steps": int(args.num_inference_steps),
        },
        "unwrapped_references": list(references.values()),
        "measurements": measurements,
        "zero_identity_checks": zero_identity_checks,
        "hook_activation": {"state": "passed" if hook_active else "failed"},
        "artifacts": {"zero_strength_rollout": "zero-strength-rollout.mp4"},
        "claim_boundary": (
            "Frozen-checkpoint inference-only repair screen. A development effect does not "
            "establish independent repair confirmation, training benefit, or RSI."
        ),
    }
    import mediapy

    mediapy.write_video(output_root / "zero-strength-rollout.mp4", visual_sample, fps=4)
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--primitive-id", choices=tuple(sorted(PRIMITIVES)), required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--contexts-json", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--context-ids", type=str, nargs="*")
    parser.add_argument("--interact-num", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--zero-identity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
