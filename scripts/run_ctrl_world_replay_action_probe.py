#!/usr/bin/env python3
"""Measure a reversible Ctrl-World action-conditioning response on frozen replay.

This runner intentionally lives outside the upstream Ctrl-World checkout.  It
uses the published replay path, freezes checkpoint/trajectory/seed per dose,
and applies a temporary transformation only to ``action_encoder.forward``.
The output is diagnostic evidence, not a model-improvement claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

import numpy as np
import torch


class ActionProbeError(RuntimeError):
    """The frozen action-conditioning probe cannot be executed safely."""


class ActionEmbeddingDose:
    """A reversible, audited perturbation of Ctrl-World action embeddings."""

    def __init__(self, model: object, *, probe_id: str, dose: float) -> None:
        if probe_id not in {"action_conditioning_scale", "action_embedding_temporal_mix"}:
            raise ActionProbeError(f"ACTION_PROBE_UNKNOWN:{probe_id}")
        if not math.isfinite(dose) or abs(dose) >= 1.0:
            raise ActionProbeError("ACTION_PROBE_DOSE_INVALID")
        encoder = getattr(model, "action_encoder", None)
        forward = getattr(encoder, "forward", None)
        if encoder is None or not callable(forward):
            raise ActionProbeError("ACTION_PROBE_HOOK_MISSING")
        self.encoder = encoder
        self.original = forward
        self.probe_id = probe_id
        self.dose = float(dose)
        self.audit: dict[str, float] = {
            "invocation_count": 0.0,
            "mean_abs_embedding_delta": 0.0,
            "maximum_temporal_mean_abs_error": 0.0,
        }

    def __enter__(self) -> "ActionEmbeddingDose":
        original = self.original
        dose = self.dose
        probe_id = self.probe_id
        audit = self.audit

        def wrapped_forward(_module: object, *args: object, **kwargs: object) -> object:
            embedding = original(*args, **kwargs)
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3:
                raise ActionProbeError("ACTION_PROBE_EMBEDDING_SHAPE_INVALID")
            if probe_id == "action_conditioning_scale":
                output = embedding if dose == 0.0 else embedding * (1.0 + dose)
            else:
                temporal_mean = embedding.mean(dim=1, keepdim=True)
                output = embedding + dose * (temporal_mean - embedding)
                mean_error = float(
                    (output.mean(dim=1, keepdim=True) - temporal_mean).abs().max().item()
                )
                audit["maximum_temporal_mean_abs_error"] = max(
                    audit["maximum_temporal_mean_abs_error"], mean_error
                )
            audit["invocation_count"] += 1.0
            audit["mean_abs_embedding_delta"] += float((output - embedding).abs().mean().item())
            return output

        self.encoder.forward = MethodType(wrapped_forward, self.encoder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.encoder.forward = self.original
        return False

    def result(self) -> dict[str, float]:
        count = max(self.audit["invocation_count"], 1.0)
        return {
            "invocation_count": self.audit["invocation_count"],
            "mean_abs_embedding_delta": self.audit["mean_abs_embedding_delta"] / count,
            "maximum_temporal_mean_abs_error": self.audit[
                "maximum_temporal_mean_abs_error"
            ],
        }


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionProbeError(f"ACTION_PROBE_JSON_INVALID:{path}") from exc


def _load_episodes(path: Path) -> list[dict[str, object]]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ActionProbeError("ACTION_PROBE_EPISODES_INVALID")
    episodes: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ActionProbeError("ACTION_PROBE_EPISODES_INVALID")
        episode_id = str(item.get("episode_id", ""))
        if not episode_id or episode_id in seen:
            raise ActionProbeError("ACTION_PROBE_EPISODES_INVALID")
        start_idx = item.get("start_idx")
        seed = item.get("seed")
        if not isinstance(start_idx, int) or start_idx < 0 or not isinstance(seed, int):
            raise ActionProbeError("ACTION_PROBE_EPISODES_INVALID")
        seen.add(episode_id)
        episodes.append({"episode_id": episode_id, "start_idx": start_idx, "seed": seed})
    return episodes


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rollout_module(ctrl_world_root: Path) -> Any:
    root = ctrl_world_root.resolve(strict=True)
    for value in (str(root), str(root / "scripts")):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    source = root / "scripts" / "rollout_replay_traj.py"
    if not source.is_file():
        raise ActionProbeError("ACTION_PROBE_REPLAY_ENTRYPOINT_MISSING")
    spec = importlib.util.spec_from_file_location("verdiwm_ctrl_world_replay", source)
    if spec is None or spec.loader is None:
        raise ActionProbeError("ACTION_PROBE_REPLAY_IMPORT_SPEC_INVALID")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_args(
    *,
    ctrl_world_root: Path,
    dataset_root: Path,
    data_stat: Path,
    svd_model_path: Path,
    clip_model_path: Path,
    ckpt_path: Path,
    interact_num: int,
    num_inference_steps: int,
) -> object:
    del ctrl_world_root
    from config import wm_args

    args = wm_args(task_type="replay")
    args.svd_model_path = str(svd_model_path)
    args.clip_model_path = str(clip_model_path)
    args.ckpt_path = str(ckpt_path)
    args.val_model_path = str(ckpt_path)
    args.val_dataset_dir = str(dataset_root)
    args.data_stat_path = str(data_stat)
    args.interact_num = int(interact_num)
    args.num_inference_steps = int(num_inference_steps)
    args.save_dir = "unused-by-action-probe"
    return args


def _metrics(true_video: np.ndarray, predicted_video: np.ndarray) -> dict[str, object]:
    true = true_video.astype(np.float32)
    predicted = predicted_video.astype(np.float32)
    absolute = np.abs(true - predicted).mean(axis=(0, 2, 3, 4))
    mse = np.square(true - predicted).mean(axis=(0, 2, 3, 4))
    psnr = 10.0 * np.log10((255.0**2) / np.maximum(mse, 1e-9))
    return {
        "l1_per_frame": [float(value) for value in absolute],
        "psnr_per_frame": [float(value) for value in psnr],
        "mean_l1": float(absolute.mean()),
        "final_l1": float(absolute[-1]),
        "mean_psnr": float(psnr.mean()),
        "final_psnr": float(psnr[-1]),
    }


def _paired_grid(true_video: np.ndarray, predicted_video: np.ndarray) -> np.ndarray:
    """Return [GT all views | prediction all views] in a true vertical layout."""

    if true_video.shape != predicted_video.shape or true_video.ndim != 5:
        raise ActionProbeError("ACTION_PROBE_VIDEO_SHAPE_INVALID")
    ground_truth = np.concatenate([true_video[index] for index in range(true_video.shape[0])], axis=1)
    prediction = np.concatenate(
        [predicted_video[index] for index in range(predicted_video.shape[0])], axis=1
    )
    return np.concatenate((ground_truth, prediction), axis=1)


def _run_episode(
    *,
    rollout_agent: object,
    runtime_args: object,
    episode: dict[str, object],
    probe_id: str,
    dose: float,
) -> tuple[dict[str, object], np.ndarray]:
    _set_seed(int(episode["seed"]))
    pred_step = int(getattr(runtime_args, "pred_step"))
    interact_num = int(getattr(runtime_args, "interact_num"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    eef_gt, joint_pos_gt, video_dict, video_latents, instruction = rollout_agent.get_traj_info(
        str(episode["episode_id"]),
        start_idx=int(episode["start_idx"]),
        steps=int(pred_step * interact_num + 8),
    )
    first_latent = torch.cat([value[0] for value in video_latents], dim=1).unsqueeze(0)
    expected_shape = (1, 4, 72, 40)
    if tuple(first_latent.shape) != expected_shape:
        raise ActionProbeError(f"ACTION_PROBE_INITIAL_LATENT_SHAPE:{tuple(first_latent.shape)}")
    history_latents = [first_latent for _ in range(num_history * 4)]
    history_joint = [joint_pos_gt[0:1] for _ in range(num_history * 4)]
    history_eef = [eef_gt[0:1] for _ in range(num_history * 4)]
    all_true: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    interactions: list[dict[str, object]] = []
    model = getattr(rollout_agent, "model", None)
    with ActionEmbeddingDose(model, probe_id=probe_id, dose=dose) as hook:
        for interaction in range(interact_num):
            start_id = int(interaction * (pred_step - 1))
            end_id = start_id + pred_step
            target_latents = [value[start_id:end_id] for value in video_latents]
            cartesian_pose = eef_gt[start_id:end_id]
            if cartesian_pose.shape != (pred_step, 7):
                raise ActionProbeError("ACTION_PROBE_ACTION_WINDOW_INVALID")
            history_idx = [0, 0, -8, -6, -4, -2]
            history_pose = np.concatenate([history_eef[index] for index in history_idx], axis=0)
            action_cond = np.concatenate((history_pose, cartesian_pose), axis=0)
            history_input = torch.cat(
                [history_latents[index] for index in history_idx], dim=0
            ).unsqueeze(0)
            current_latent = history_latents[-1]
            if action_cond.shape != (num_history + num_frames, 7):
                raise ActionProbeError("ACTION_PROBE_CONDITION_WINDOW_INVALID")
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
            history_joint.append(joint_pos_gt[start_id + pred_step - 1 : start_id + pred_step])
            history_latents.append(
                torch.cat([value[pred_step - 1] for value in predicted_latents], dim=1).unsqueeze(0)
            )
    l1_series = [float(value["mean_l1"]) for value in interactions]
    slope = float(np.polyfit(np.arange(len(l1_series), dtype=np.float64), l1_series, 1)[0]) if len(l1_series) > 1 else 0.0
    episode_result = {
        "episode_id": str(episode["episode_id"]),
        "start_idx": int(episode["start_idx"]),
        "seed": int(episode["seed"]),
        "interactions": interactions,
        "mean_l1": float(np.mean(l1_series)),
        "final_interaction_l1": float(l1_series[-1]),
        "horizon_l1_slope": slope,
        "hook_audit": hook.result(),
    }
    return episode_result, _paired_grid(
        np.concatenate(all_true, axis=1), np.concatenate(all_prediction, axis=1)
    )


def _summarize_dose(episodes: Sequence[dict[str, object]]) -> dict[str, float]:
    return {
        "mean_l1": float(np.mean([float(row["mean_l1"]) for row in episodes])),
        "final_interaction_l1": float(
            np.mean([float(row["final_interaction_l1"]) for row in episodes])
        ),
        "horizon_l1_slope": float(
            np.mean([float(row["horizon_l1_slope"]) for row in episodes])
        ),
        "mean_hook_delta": float(
            np.mean([float(row["hook_audit"]["mean_abs_embedding_delta"]) for row in episodes])
        ),
        "total_hook_invocations": float(
            np.sum([float(row["hook_audit"]["invocation_count"]) for row in episodes])
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ActionProbeError("ACTION_PROBE_OUTPUT_EXISTS")
    ctrl_world_root = Path(args.ctrl_world_root).resolve(strict=True)
    dataset_root = Path(args.dataset_root).resolve(strict=True)
    data_stat = Path(args.data_stat).resolve(strict=True)
    episodes = _load_episodes(Path(args.episodes_json).resolve(strict=True))
    doses = tuple(float(value) for value in args.doses)
    if 0.0 not in doses or len(doses) != len(set(doses)):
        raise ActionProbeError("ACTION_PROBE_DOSE_GRID_INVALID")
    module = _load_rollout_module(ctrl_world_root)
    runtime_args = _runtime_args(
        ctrl_world_root=ctrl_world_root,
        dataset_root=dataset_root,
        data_stat=data_stat,
        svd_model_path=Path(args.svd_model_path).resolve(strict=True),
        clip_model_path=Path(args.clip_model_path).resolve(strict=True),
        ckpt_path=Path(args.ckpt_path).resolve(strict=True),
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
    )
    output_root.mkdir(mode=0o700, parents=True)
    rollout_agent = module.agent(runtime_args)
    rows_by_dose: dict[float, list[dict[str, object]]] = {}
    visual_sample: np.ndarray | None = None
    ordered_doses = (0.0,) + tuple(value for value in doses if value != 0.0)
    for dose in ordered_doses:
        episode_rows: list[dict[str, object]] = []
        for episode in episodes:
            row, visual = _run_episode(
                rollout_agent=rollout_agent,
                runtime_args=runtime_args,
                episode=episode,
                probe_id=str(args.probe_id),
                dose=dose,
            )
            episode_rows.append(row)
            if dose == 0.0 and visual_sample is None:
                visual_sample = visual
        rows_by_dose[dose] = episode_rows
    baseline = _summarize_dose(rows_by_dose[0.0])
    dose_rows = []
    for dose in ordered_doses:
        summary = _summarize_dose(rows_by_dose[dose])
        dose_rows.append(
            {
                "dose": dose,
                "summary": summary,
                "delta_from_zero": {
                    key: float(summary[key] - baseline[key])
                    for key in ("mean_l1", "final_interaction_l1", "horizon_l1_slope")
                },
                "episodes": rows_by_dose[dose],
            }
        )
    if visual_sample is None:
        raise ActionProbeError("ACTION_PROBE_VIDEO_MISSING")
    import mediapy

    mediapy.write_video(output_root / "rollout.mp4", visual_sample, fps=4)
    metrics = {
        "probe_ready": 1.0,
        "episode_count": float(len(episodes)),
        "dose_count": float(len(doses)),
        "baseline_mean_l1": baseline["mean_l1"],
        "baseline_final_interaction_l1": baseline["final_interaction_l1"],
        "baseline_horizon_l1_slope": baseline["horizon_l1_slope"],
        "baseline_hook_invocations": baseline["total_hook_invocations"],
    }
    for row in dose_rows:
        dose = float(row["dose"])
        if dose == 0.0:
            continue
        tag = ("plus" if dose > 0 else "minus") + str(abs(dose)).replace(".", "d")
        delta = row["delta_from_zero"]
        metrics[f"{tag}_delta_mean_l1"] = float(delta["mean_l1"])
        metrics[f"{tag}_delta_final_interaction_l1"] = float(delta["final_interaction_l1"])
        metrics[f"{tag}_delta_horizon_l1_slope"] = float(delta["horizon_l1_slope"])
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-action-probe-result",
        "state": "ready",
        "probe_id": str(args.probe_id),
        "model_family": "ctrl-world",
        "runtime_capability": "action-conditioned-predictive-video",
        "metrics": metrics,
        "input": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "dataset_root": str(dataset_root),
            "episodes": episodes,
            "doses": list(doses),
            "interact_num": int(args.interact_num),
            "num_inference_steps": int(args.num_inference_steps),
        },
        "dose_responses": dose_rows,
        "artifacts": {"paired_video": "rollout.mp4", "video_layout": "vertical_gt_then_prediction"},
        "claim_boundary": (
            "This is an inference-only target-local response measurement. It does not "
            "establish a repair benefit, a training result, or cross-backbone transfer."
        ),
    }
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--episodes-json", type=Path, required=True)
    parser.add_argument(
        "--probe-id",
        choices=("action_conditioning_scale", "action_embedding_temporal_mix"),
        default="action_conditioning_scale",
    )
    parser.add_argument("--doses", type=float, nargs="+", default=(-0.025, 0.0, 0.025))
    parser.add_argument("--interact-num", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
