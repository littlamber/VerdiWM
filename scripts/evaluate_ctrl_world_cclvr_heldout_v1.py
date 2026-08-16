#!/usr/bin/env python3
"""Evaluate cached CCLVR routes on frozen held-out Ctrl-World replay contexts."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from scripts import run_ctrl_world_local_fingerprint_probe as fingerprint
except ImportError:  # Direct script execution places this directory on sys.path.
    import run_ctrl_world_local_fingerprint_probe as fingerprint


ARM_DOSES = (-0.99, 0.0, 0.99)


class CCLVRHeldoutEvaluationError(RuntimeError):
    """A CCLVR held-out shard could not be evaluated faithfully."""


@dataclass(frozen=True)
class PreparedContext:
    context: Mapping[str, object]
    eef_gt: np.ndarray
    video_latents: Sequence[torch.Tensor]
    instruction: str
    python_rng_state: object
    numpy_rng_state: tuple[object, ...]
    torch_rng_state: torch.Tensor
    cuda_rng_states: list[torch.Tensor]


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_rng(prepared: PreparedContext) -> None:
    random.setstate(prepared.python_rng_state)
    np.random.set_state(prepared.numpy_rng_state)  # type: ignore[arg-type]
    torch.set_rng_state(prepared.torch_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(prepared.cuda_rng_states)


def _prepare_context(
    rollout_agent: object,
    runtime_args: object,
    context: Mapping[str, object],
) -> PreparedContext:
    fingerprint._set_seed(int(context["seed"]))
    pred_step = int(getattr(runtime_args, "pred_step"))
    interact_num = int(getattr(runtime_args, "interact_num"))
    eef_gt, _joint_pos_gt, _video_dict, video_latents, instruction = rollout_agent.get_traj_info(
        str(context["episode_id"]),
        start_idx=int(context["start_idx"]),
        steps=int(pred_step * interact_num + 8),
    )
    return PreparedContext(
        context=context,
        eef_gt=eef_gt,
        video_latents=video_latents,
        instruction=str(instruction),
        python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


class CCLVRRouteProbe:
    """Apply exact fixed endpoints or a first-invocation cached hard CCLVR route."""

    def __init__(
        self,
        model: object,
        *,
        mode: str,
        route_scope: str,
        fixed_dose: float | None = None,
        target_interaction: int | None = None,
    ) -> None:
        if mode not in {"fixed", "learned_cached"}:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_MODE_INVALID")
        if route_scope not in {"episode", "interaction"}:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_SCOPE_INVALID")
        if mode == "fixed":
            if fixed_dose is None or not any(
                math.isclose(float(fixed_dose), dose, abs_tol=1e-12) for dose in ARM_DOSES
            ):
                raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_FIXED_DOSE_INVALID")
        elif fixed_dose is not None or target_interaction is not None:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_LEARNED_ROUTE_FRAME_INVALID")
        self.model = model
        self.mode = mode
        self.route_scope = route_scope
        self.fixed_dose = None if fixed_dose is None else float(fixed_dose)
        self.target_interaction = target_interaction
        self.current_interaction: int | None = None
        self._cache: dict[int, float] = {}
        self._decision_records: dict[int, dict[str, object]] = {}
        self._invocations: list[dict[str, object]] = []
        self._runtime_features: list[list[list[float]]] = []
        self._maximum_abs_residual = 0.0

    def __enter__(self) -> "CCLVRRouteProbe":
        corrector = getattr(self.model, "history_corrector", None)
        original = getattr(corrector, "forward", None)
        if (
            corrector is None
            or not callable(original)
            or getattr(corrector, "multiscale_side_adapter", None) is None
        ):
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_HOOK_MISSING")
        if self.mode == "learned_cached" and getattr(corrector, "local_arm_value_head", None) is None:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_HOOK_MISSING")
        normalized_dose = float(getattr(corrector, "local_arm_normalized_dose", math.nan))
        if not math.isclose(normalized_dose, ARM_DOSES[2], abs_tol=1e-9):
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_DOSE_MISMATCH")
        self._corrector = corrector
        self._original = original

        def wrapped(_module: object, *args: object, **kwargs: object) -> object:
            if self.current_interaction is None:
                raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_INTERACTION_UNSET")
            if kwargs.get("adapter_scale_override") is not None:
                raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_OVERRIDE_CONFLICT")
            overrides = dict(kwargs)
            interaction = self.current_interaction
            cache_key = 0 if self.route_scope == "episode" else interaction
            first_decision = False
            if self.mode == "fixed":
                applied_dose = (
                    self.fixed_dose
                    if self.target_interaction is None or self.target_interaction == interaction
                    else 0.0
                )
                overrides["adapter_scale_override"] = applied_dose
                output = original(*args, **overrides)
            elif cache_key in self._cache:
                applied_dose = self._cache[cache_key]
                overrides["adapter_scale_override"] = applied_dose
                output = original(*args, **overrides)
            else:
                output = original(*args, **overrides)
                selected = getattr(output, "selected_arm", None)
                features = getattr(output, "runtime_features", None)
                normalized = getattr(output, "normalized_adapter_dose", None)
                if (
                    not isinstance(selected, torch.Tensor)
                    or selected.numel() != 1
                    or not isinstance(features, torch.Tensor)
                    or not isinstance(normalized, torch.Tensor)
                    or normalized.numel() != 1
                ):
                    raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_OUTPUT_INVALID")
                arm = int(selected.item())
                if arm not in (0, 1, 2) or not normalized.is_floating_point():
                    raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_SELECTION_INVALID")
                observed_dose = float(normalized.item())
                expected_dose = ARM_DOSES[arm]
                representable_expected = float(
                    torch.as_tensor(
                        expected_dose,
                        device=normalized.device,
                        dtype=normalized.dtype,
                    ).item()
                )
                if not math.isclose(observed_dose, representable_expected, abs_tol=1e-6):
                    raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_SELECTION_INVALID")
                applied_dose = expected_dose
                soft = corrector.predict_local_arm_policy(features, hard=False)
                probabilities = soft.probabilities[0].detach().float().cpu().tolist()
                values = soft.values[0].detach().float().cpu().tolist()
                adjusted = soft.adjusted_values[0].detach().float().cpu().tolist()
                if len(probabilities) != 3 or len(values) != 3 or len(adjusted) != 3:
                    raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_POLICY_INVALID")
                self._cache[cache_key] = applied_dose
                self._decision_records[cache_key] = {
                    "cache_key": cache_key,
                    "decision_interaction": interaction,
                    "selected_arm": arm,
                    "selected_dose": applied_dose,
                    "soft_probabilities": probabilities,
                    "arm_values": values,
                    "adjusted_arm_values": adjusted,
                }
                first_decision = True
            residuals = getattr(output, "down_block_additional_residuals", None)
            mid_residual = getattr(output, "mid_block_additional_residual", None)
            runtime_features = getattr(output, "runtime_features", None)
            if (
                not isinstance(residuals, tuple)
                or not isinstance(mid_residual, torch.Tensor)
                or not isinstance(runtime_features, torch.Tensor)
                or runtime_features.ndim != 3
                or runtime_features.shape[0] != 1
                or not torch.isfinite(runtime_features).all()
            ):
                raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_RESIDUAL_INVALID")
            self._runtime_features.append(runtime_features[0].detach().float().cpu().tolist())
            maximum = float(
                torch.stack(
                    [value.detach().float().abs().max() for value in (*residuals, mid_residual)]
                ).max().item()
            )
            self._maximum_abs_residual = max(self._maximum_abs_residual, maximum)
            self._invocations.append(
                {
                    "interaction": interaction,
                    "cache_key": cache_key,
                    "applied_dose": float(applied_dose),
                    "first_decision": first_decision,
                    "maximum_abs_residual": maximum,
                }
            )
            return output

        corrector.forward = MethodType(wrapped, corrector)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._corrector.forward = self._original
        return False

    def set_interaction(self, interaction: int) -> None:
        if interaction < 0:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_INTERACTION_INVALID")
        self.current_interaction = int(interaction)

    def result(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "route_scope": self.route_scope,
            "fixed_dose": self.fixed_dose,
            "target_interaction": self.target_interaction,
            "invocation_count": len(self._invocations),
            "maximum_abs_residual": self._maximum_abs_residual,
            "decision_records": [self._decision_records[key] for key in sorted(self._decision_records)],
            "invocations": list(self._invocations),
            "runtime_feature_records": list(self._runtime_features),
        }


def _runtime_args(args: argparse.Namespace) -> object:
    runtime = fingerprint._runtime_args(
        dataset_root=Path(args.dataset_root).resolve(strict=True),
        data_stat=Path(args.data_stat).resolve(strict=True),
        svd_model_path=Path(args.svd_model_path).resolve(strict=True),
        clip_model_path=Path(args.clip_model_path).resolve(strict=True),
        ckpt_path=Path(args.ckpt_path).resolve(strict=True),
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
        enable_signed_history_correction=True,
        unsigned_history_gate=False,
        enable_multiscale_history_adapter=True,
        multiscale_history_always_on=False,
    )
    runtime.enable_cclvr = True
    runtime.cclvr_supervision_path = str(Path(args.cclvr_supervision_path).resolve(strict=True))
    runtime.cclvr_supervision_sha256 = str(args.cclvr_supervision_sha256)
    runtime.cclvr_supervision_variant = str(args.cclvr_supervision_variant)
    runtime.cclvr_value_hidden_dim = int(args.cclvr_value_hidden_dim)
    runtime.cclvr_policy_temperature = float(args.cclvr_policy_temperature)
    runtime.cclvr_normalized_dose = ARM_DOSES[2]
    return runtime


def _run_prepared_context(
    *,
    rollout_agent: object,
    runtime_args: object,
    prepared: PreparedContext,
    route_mode: str,
    route_scope: str,
    fixed_dose: float | None = None,
    target_interaction: int | None = None,
    action_dose: float | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    _restore_rng(prepared)
    context = prepared.context
    pred_step = int(getattr(runtime_args, "pred_step"))
    interact_num = int(getattr(runtime_args, "interact_num"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    first_latent = torch.cat([value[0] for value in prepared.video_latents], dim=1).unsqueeze(0)
    if tuple(first_latent.shape) != (1, 4, 72, 40):
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_INITIAL_LATENT_INVALID")
    history_latents = [first_latent for _ in range(num_history * 4)]
    history_eef = [prepared.eef_gt[0:1] for _ in range(num_history * 4)]
    route = CCLVRRouteProbe(
        getattr(rollout_agent, "model", None),
        mode=route_mode,
        route_scope=route_scope,
        fixed_dose=fixed_dose,
        target_interaction=target_interaction,
    )
    action_probe = (
        fingerprint.ActionConditioningScale(getattr(rollout_agent, "model", None), action_dose)
        if action_dose is not None
        else None
    )
    all_true: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    interactions: list[dict[str, object]] = []
    with ExitStack() as stack:
        stack.enter_context(route)
        if action_probe is not None:
            stack.enter_context(action_probe)
        for interaction in range(interact_num):
            route.set_interaction(interaction)
            start_id = int(interaction * (pred_step - 1))
            end_id = start_id + pred_step
            target_latents = [value[start_id:end_id] for value in prepared.video_latents]
            cartesian_pose = prepared.eef_gt[start_id:end_id]
            history_idx = [0, 0, -8, -6, -4, -2]
            history_pose = np.concatenate([history_eef[index] for index in history_idx], axis=0)
            action_cond = np.concatenate((history_pose, cartesian_pose), axis=0)
            history_input = torch.cat([history_latents[index] for index in history_idx], dim=0).unsqueeze(0)
            current_latent = history_latents[-1]
            if action_cond.shape != (num_history + num_frames, 7):
                raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ACTION_WINDOW_INVALID")
            videos_cat, true_video, prediction, predicted_latents = rollout_agent.forward_wm(
                action_cond,
                target_latents,
                current_latent,
                his_cond=history_input,
                text=prepared.instruction if bool(getattr(runtime_args, "text_cond")) else None,
            )
            del videos_cat
            metric = fingerprint._metrics(true_video, prediction)
            metric["interaction"] = interaction
            interactions.append(metric)
            all_true.append(true_video)
            all_prediction.append(prediction)
            history_eef.append(cartesian_pose[pred_step - 1 : pred_step])
            history_latents.append(
                torch.cat([value[pred_step - 1] for value in predicted_latents], dim=1).unsqueeze(0)
            )
    predicted_video = np.concatenate(all_prediction, axis=1)
    route_audit = route.result()
    row = {
        "identity": {
            "context_id": str(context["context_id"]),
            "episode_id": str(context["episode_id"]),
            "start_idx": int(context["start_idx"]),
            "seed": int(context["seed"]),
        },
        "interactions": interactions,
        "outcomes": fingerprint._outcome_vector(interactions),
        "route_audit": route_audit,
        "runtime_feature_records": route_audit["runtime_feature_records"],
        "action_hook_audit": action_probe.result() if action_probe is not None else None,
        "prediction_response": fingerprint._prediction_response(predicted_video),
    }
    visual = fingerprint._paired_grid(np.concatenate(all_true, axis=1), predicted_video)
    return row, visual


def _validate_route_row(
    row: Mapping[str, object],
    *,
    interact_num: int,
    inference_steps: int,
    route_scope: str,
    fixed_dose: float | None,
    target_interaction: int | None,
) -> None:
    audit = row.get("route_audit")
    if not isinstance(audit, Mapping) or int(audit.get("invocation_count", -1)) != interact_num * inference_steps:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_AUDIT_INVALID")
    invocations = audit.get("invocations")
    decisions = audit.get("decision_records")
    if not isinstance(invocations, list) or not isinstance(decisions, list):
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_AUDIT_INVALID")
    if fixed_dose is not None:
        if decisions:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_FIXED_ROUTE_DECISION_INVALID")
        active = [
            record
            for record in invocations
            if not math.isclose(float(record["applied_dose"]), 0.0, abs_tol=1e-12)
        ]
        expected_active = 0 if math.isclose(fixed_dose, 0.0, abs_tol=1e-12) else inference_steps
        if len(active) != expected_active:
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_FIXED_ROUTE_ACTIVATION_INVALID")
        if expected_active and any(int(record["interaction"]) != target_interaction for record in active):
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_FIXED_ROUTE_LOCALITY_INVALID")
        maximum = float(audit.get("maximum_abs_residual", math.nan))
        if (expected_active == 0 and maximum != 0.0) or (expected_active > 0 and maximum <= 0.0):
            raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_FIXED_ROUTE_RESIDUAL_INVALID")
        return
    expected_decisions = 1 if route_scope == "episode" else interact_num
    if len(decisions) != expected_decisions:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_CACHED_ROUTE_DECISION_COUNT_INVALID")
    by_key: dict[int, set[float]] = {}
    for record in invocations:
        by_key.setdefault(int(record["cache_key"]), set()).add(float(record["applied_dose"]))
    if len(by_key) != expected_decisions or any(len(values) != 1 for values in by_key.values()):
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_CACHED_ROUTE_UNSTABLE")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_OUTPUT_EXISTS")
    if args.mode not in {"routing", "action_sensitivity"}:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_MODE_INVALID")
    if args.route_scope not in {"episode", "interaction"}:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ROUTE_SCOPE_INVALID")
    contexts_path = Path(args.contexts_json).resolve(strict=True)
    contexts = fingerprint.load_contexts(contexts_path, args.seeds, [str(args.context_id)])
    if any(str(context["episode_id"]) != "1799" for context in contexts):
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_PROMOTION_CONTEXT_REQUIRED")
    if int(args.interact_num) != 4 or int(args.num_inference_steps) != 4:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_PROTOCOL_INVALID")
    action_doses = tuple(float(value) for value in args.action_doses)
    if args.mode == "action_sensitivity" and action_doses != (-0.1, 0.0, 0.1):
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ACTION_GRID_INVALID")
    if args.mode == "routing" and action_doses:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ACTION_GRID_UNEXPECTED")

    ctrl_world_root = Path(args.ctrl_world_root).resolve(strict=True)
    module = fingerprint._load_rollout_module(ctrl_world_root)
    runtime_args = _runtime_args(args)
    output_root.mkdir(mode=0o700, parents=True)
    rollout_agent = module.agent(runtime_args)
    measurements: list[dict[str, object]] = []
    sample_visual: np.ndarray | None = None
    for context in contexts:
        prepared = _prepare_context(rollout_agent, runtime_args, context)
        if args.mode == "routing":
            zero, visual = _run_prepared_context(
                rollout_agent=rollout_agent,
                runtime_args=runtime_args,
                prepared=prepared,
                route_mode="fixed",
                route_scope=str(args.route_scope),
                fixed_dose=0.0,
            )
            zero.update({"kind": "fixed_zero", "dose": 0.0, "target_interaction": None})
            _validate_route_row(
                zero,
                interact_num=4,
                inference_steps=4,
                route_scope=str(args.route_scope),
                fixed_dose=0.0,
                target_interaction=None,
            )
            measurements.append(zero)
            sample_visual = visual if sample_visual is None else sample_visual
            for interaction in range(4):
                for dose in (ARM_DOSES[0], ARM_DOSES[2]):
                    endpoint, _visual = _run_prepared_context(
                        rollout_agent=rollout_agent,
                        runtime_args=runtime_args,
                        prepared=prepared,
                        route_mode="fixed",
                        route_scope=str(args.route_scope),
                        fixed_dose=dose,
                        target_interaction=interaction,
                    )
                    endpoint.update(
                        {
                            "kind": "fixed_endpoint",
                            "dose": dose,
                            "target_interaction": interaction,
                        }
                    )
                    _validate_route_row(
                        endpoint,
                        interact_num=4,
                        inference_steps=4,
                        route_scope=str(args.route_scope),
                        fixed_dose=dose,
                        target_interaction=interaction,
                    )
                    measurements.append(endpoint)
            learned, _visual = _run_prepared_context(
                rollout_agent=rollout_agent,
                runtime_args=runtime_args,
                prepared=prepared,
                route_mode="learned_cached",
                route_scope=str(args.route_scope),
            )
            learned.update({"kind": "learned_cached", "dose": None, "target_interaction": None})
            _validate_route_row(
                learned,
                interact_num=4,
                inference_steps=4,
                route_scope=str(args.route_scope),
                fixed_dose=None,
                target_interaction=None,
            )
            measurements.append(learned)
        else:
            for dose in action_doses:
                learned, visual = _run_prepared_context(
                    rollout_agent=rollout_agent,
                    runtime_args=runtime_args,
                    prepared=prepared,
                    route_mode="learned_cached",
                    route_scope=str(args.route_scope),
                    action_dose=dose,
                )
                learned.update(
                    {
                        "kind": "action_sensitivity",
                        "action_dose": dose,
                        "dose": None,
                        "target_interaction": None,
                    }
                )
                _validate_route_row(
                    learned,
                    interact_num=4,
                    inference_steps=4,
                    route_scope=str(args.route_scope),
                    fixed_dose=None,
                    target_interaction=None,
                )
                audit = learned["action_hook_audit"]
                if (
                    not isinstance(audit, Mapping)
                    or float(audit["invocation_count"]) < 4.0
                    or (
                        math.isclose(dose, 0.0, abs_tol=1e-12)
                        and float(audit["maximum_abs_tensor_delta"]) != 0.0
                    )
                    or (
                        not math.isclose(dose, 0.0, abs_tol=1e-12)
                        and float(audit["maximum_abs_tensor_delta"]) <= 0.0
                    )
                ):
                    raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_ACTION_HOOK_INVALID")
                measurements.append(learned)
                sample_visual = visual if sample_visual is None else sample_visual

    if sample_visual is None:
        raise CCLVRHeldoutEvaluationError("CCLVR_HELDOUT_VISUAL_MISSING")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-shard-v1",
        "state": "ready",
        "campaign_id": str(args.campaign_id),
        "cell_id": str(args.cell_id),
        "mode": str(args.mode),
        "route_scope": str(args.route_scope),
        "input": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "checkpoint_sha256": _sha256(Path(args.ckpt_path)),
            "contexts_json": str(contexts_path),
            "contexts_sha256": _sha256(contexts_path),
            "selected_seeds": [int(value) for value in args.seeds],
            "interact_num": 4,
            "num_inference_steps": 4,
            "arm_doses": list(ARM_DOSES),
            "action_doses": list(action_doses),
            "cclvr_supervision_variant": str(args.cclvr_supervision_variant),
            "cclvr_supervision_sha256": str(args.cclvr_supervision_sha256),
        },
        "runtime": fingerprint._runtime_receipt(),
        "measurements": measurements,
        "artifacts": {"sample_rollout": "sample-rollout.mp4"},
        "claim_boundary": (
            "Promotion-only episode 1799 evidence for the frozen 64-step CCLVR screen. "
            "It cannot be reused for training, labels, coverage control, or calibration fitting."
        ),
    }
    import mediapy

    try:
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    mediapy.write_video(output_root / "sample-rollout.mp4", sample_visual, fps=4)
    _atomic_json(output_root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--mode", choices=("routing", "action_sensitivity"), required=True)
    parser.add_argument("--route-scope", choices=("episode", "interaction"), required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--contexts-json", type=Path, required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--interact-num", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--action-doses", type=float, nargs="*", default=())
    parser.add_argument("--cclvr-supervision-path", type=Path, required=True)
    parser.add_argument("--cclvr-supervision-sha256", required=True)
    parser.add_argument("--cclvr-supervision-variant", required=True)
    parser.add_argument("--cclvr-value-hidden-dim", type=int, default=128)
    parser.add_argument("--cclvr-policy-temperature", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps({"state": result["state"], "cell_id": result["cell_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
