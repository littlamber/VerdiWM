#!/usr/bin/env python3
"""Measure Ctrl-World against the frozen two-surface ACWM protocol.

The harness lives outside the Ctrl-World checkout and never changes its source,
checkpoint, or dataset.  It has no VLA policy dependency and emits only the
four metrics admitted by the ACWM dual-evaluation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class CtrlWorldACWMEvaluationError(RuntimeError):
    """The external model or frozen ACWM protocol could not be evaluated."""


_METRIC_IDS = (
    "long_horizon_prediction_error",
    "action_conditioning_error",
    "trajectory_fidelity_error",
    "horizon_drift_slope",
)
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    contract = _load_mapping(args.contract)
    _validate_contract(contract)
    candidate = _candidate_metadata(args.candidate_id, args.guidance_scale)
    protocol = _protocol_for_stage(contract, args.stage)
    paths = _validate_paths(args)
    numpy, torch = _runtime_dependencies()
    _seed_everything(torch, int(contract["protocol"]["seed"]))
    rollout_module = _load_rollout_module(paths["ctrl_world_root"])
    runtime_args = _build_runtime_args(
        ctrl_world_root=paths["ctrl_world_root"],
        dataset_root=paths["dataset_root"],
        data_stat=paths["data_stat"],
        checkpoint=paths["checkpoint"],
        svd_model=paths["svd_model"],
        clip_model=paths["clip_model"],
        interactions=max(
            int(protocol["paired_prediction_interactions"]),
            int(protocol["rollout_interactions"]),
        ),
        inference_steps=int(protocol["num_inference_steps"]),
        guidance_scale=float(candidate["parameters"]["guidance_scale"]),
    )
    agent = rollout_module.agent(runtime_args)

    paired_rows: list[dict[str, object]] = []
    for offset, context in enumerate(protocol["paired_prediction_contexts"]):
        _seed_everything(torch, int(contract["protocol"]["seed"]) + offset)
        paired_rows.append(
            _run_context(
                agent=agent,
                runtime_args=runtime_args,
                context=context,
                interactions=int(protocol["paired_prediction_interactions"]),
                generated_history=False,
                numpy=numpy,
                torch=torch,
            )
        )

    rollout_rows: list[dict[str, object]] = []
    for offset, context in enumerate(protocol["rollout_contexts"]):
        _seed_everything(torch, int(contract["protocol"]["seed"]) + 10_000 + offset)
        rollout_rows.append(
            _run_context(
                agent=agent,
                runtime_args=runtime_args,
                context=context,
                interactions=int(protocol["rollout_interactions"]),
                generated_history=True,
                numpy=numpy,
                torch=torch,
            )
        )

    metrics = _aggregate_metrics(paired_rows, rollout_rows, numpy=numpy)
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-acwm-measurement",
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": args.stage,
        "scope": "acwm_only",
        "evidence_source": contract["evidence_source"],
        "candidate": candidate,
        "metrics": metrics,
        "protocol": protocol,
        "asset_fingerprints": {
            name: _fingerprint(path)
            for name, path in sorted(paths.items())
            if name != "ctrl_world_root"
        },
        "source_revision": _source_revision(paths["ctrl_world_root"]),
        "paired_prediction": paired_rows,
        "giga_style_rollout": rollout_rows,
        "runtime": _runtime_receipt(torch),
        "claim_boundary": (
            "Paired predictive ACWM measurement only; no policy, task-success, "
            "task-progress, or safety-event signal was executed."
        ),
    }
    _validate_metrics(result["metrics"])
    _write_result(Path(args.output_root), result)
    return result


def _candidate_metadata(candidate_id: str, guidance_scale: object) -> dict[str, object]:
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise CtrlWorldACWMEvaluationError("ACWM_CANDIDATE_ID_INVALID")
    if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)):
        raise CtrlWorldACWMEvaluationError("ACWM_GUIDANCE_SCALE_INVALID")
    normalized = float(guidance_scale)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise CtrlWorldACWMEvaluationError("ACWM_GUIDANCE_SCALE_INVALID")
    return {
        "candidate_id": candidate_id,
        "candidate_kind": "inference_guidance_scale",
        "parameters": {"guidance_scale": normalized},
    }


def _validate_contract(contract: Mapping[str, object]) -> None:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from wmloop.control.acwm_dual_evaluation import validate_acwm_dual_evaluation_contract

    validate_acwm_dual_evaluation_contract(contract, root=Path(project_root))


def _protocol_for_stage(contract: Mapping[str, object], stage: str) -> Mapping[str, object]:
    protocol = contract.get("protocol")
    if not isinstance(protocol, Mapping):
        raise CtrlWorldACWMEvaluationError("ACWM_PROTOCOL_MISSING")
    stages = protocol.get("stages")
    if not isinstance(stages, list):
        raise CtrlWorldACWMEvaluationError("ACWM_PROTOCOL_STAGES_INVALID")
    matches = [row for row in stages if isinstance(row, Mapping) and row.get("stage") == stage]
    if len(matches) != 1:
        raise CtrlWorldACWMEvaluationError("ACWM_PROTOCOL_STAGE_INVALID")
    return matches[0]


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    raw = {
        "ctrl_world_root": args.ctrl_world_root,
        "dataset_root": args.dataset_root,
        "data_stat": args.data_stat,
        "checkpoint": args.checkpoint,
        "svd_model": args.svd_model,
        "clip_model": args.clip_model,
    }
    paths: dict[str, Path] = {}
    for name, value in raw.items():
        path = Path(value).expanduser().resolve()
        if not path.exists() or path.is_symlink():
            raise CtrlWorldACWMEvaluationError(f"ACWM_ASSET_INVALID:{name}")
        paths[name] = path
    if not (paths["ctrl_world_root"] / "scripts" / "rollout_replay_traj.py").is_file():
        raise CtrlWorldACWMEvaluationError("ACWM_CTRL_WORLD_ENTRYPOINT_MISSING")
    if not (paths["dataset_root"] / "droid_subset" / "annotation" / "val").is_dir():
        raise CtrlWorldACWMEvaluationError("ACWM_DROID_SUBSET_MISSING")
    return paths


def _runtime_dependencies() -> tuple[Any, Any]:
    try:
        import numpy
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in external runtime
        raise CtrlWorldACWMEvaluationError("ACWM_RUNTIME_DEPENDENCY_MISSING") from exc
    if not torch.cuda.is_available():
        raise CtrlWorldACWMEvaluationError("ACWM_CUDA_REQUIRED")
    return numpy, torch


def _load_rollout_module(root: Path) -> Any:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    source = root / "scripts" / "rollout_replay_traj.py"
    spec = importlib.util.spec_from_file_location("verdiwm_ctrl_world_rollout", source)
    if spec is None or spec.loader is None:
        raise CtrlWorldACWMEvaluationError("ACWM_ROLLOUT_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_runtime_args(
    *,
    ctrl_world_root: Path,
    dataset_root: Path,
    data_stat: Path,
    checkpoint: Path,
    svd_model: Path,
    clip_model: Path,
    interactions: int,
    inference_steps: int,
    guidance_scale: float,
) -> object:
    del ctrl_world_root
    config = importlib.import_module("config")
    runtime_args = config.wm_args(task_type="replay")
    runtime_args.dataset_root_path = str(dataset_root)
    runtime_args.dataset_meta_info_path = str(data_stat.parent.parent)
    runtime_args.dataset_names = "droid_subset"
    runtime_args.val_dataset_dir = str(dataset_root / "droid_subset")
    runtime_args.data_stat_path = str(data_stat)
    runtime_args.svd_model_path = str(svd_model)
    runtime_args.clip_model_path = str(clip_model)
    runtime_args.ckpt_path = str(checkpoint)
    runtime_args.val_model_path = str(checkpoint)
    runtime_args.interact_num = interactions
    runtime_args.num_inference_steps = inference_steps
    runtime_args.guidance_scale = guidance_scale
    runtime_args.save_dir = "unused-by-acwm-evaluator"
    return runtime_args


def _run_context(
    *,
    agent: object,
    runtime_args: object,
    context: Mapping[str, object],
    interactions: int,
    generated_history: bool,
    numpy: Any,
    torch: Any,
) -> dict[str, object]:
    pred_step = int(getattr(runtime_args, "pred_step"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    episode_id = str(context["episode_id"])
    start_idx = int(context["start_idx"])
    eef_gt, _joint_pos, _videos, video_latents, instruction = agent.get_traj_info(
        episode_id, start_idx=start_idx, steps=int(pred_step * interactions + 8)
    )
    first_latent = torch.cat([value[0] for value in video_latents], dim=1).unsqueeze(0)
    if tuple(first_latent.shape) != (1, 4, 72, 40):
        raise CtrlWorldACWMEvaluationError("ACWM_INITIAL_LATENT_SHAPE_INVALID")
    latent_history = [first_latent for _ in range(num_history * 4)]
    eef_history = [eef_gt[0:1] for _ in range(num_history * 4)]
    true_batches: list[Any] = []
    prediction_batches: list[Any] = []
    interaction_errors: list[float] = []

    for interaction in range(interactions):
        start = interaction * (pred_step - 1)
        end = start + pred_step
        target_latents = [value[start:end] for value in video_latents]
        action = eef_gt[start:end]
        if tuple(action.shape) != (pred_step, 7):
            raise CtrlWorldACWMEvaluationError("ACWM_ACTION_WINDOW_INVALID")
        history_indices = [0, 0, -8, -6, -4, -2]
        history_pose = numpy.concatenate([eef_history[index] for index in history_indices], axis=0)
        action_cond = numpy.concatenate((history_pose, action), axis=0)
        history_cond = torch.cat([latent_history[index] for index in history_indices], dim=0).unsqueeze(0)
        current = latent_history[-1]
        if tuple(action_cond.shape) != (num_history + num_frames, 7):
            raise CtrlWorldACWMEvaluationError("ACWM_CONDITION_WINDOW_INVALID")
        _visual, true_video, prediction, predicted_latents = agent.forward_wm(
            action_cond,
            target_latents,
            current,
            his_cond=history_cond,
            text=instruction if bool(getattr(runtime_args, "text_cond")) else None,
        )
        error = _rgb_l1(true_video, prediction, numpy=numpy)
        interaction_errors.append(error)
        true_batches.append(true_video)
        prediction_batches.append(prediction)
        eef_history.append(action[pred_step - 1 : pred_step])
        if generated_history:
            next_latent = torch.cat(
                [value[pred_step - 1] for value in predicted_latents], dim=1
            ).unsqueeze(0)
        else:
            next_latent = torch.cat(
                [value[end - 1] for value in video_latents], dim=1
            ).unsqueeze(0)
        latent_history.append(next_latent)

    true_all = numpy.concatenate(true_batches, axis=1)
    prediction_all = numpy.concatenate(prediction_batches, axis=1)
    return {
        "context": {"episode_id": episode_id, "start_idx": start_idx},
        "interactions": interactions,
        "mean_rgb_l1": _rgb_l1(true_all, prediction_all, numpy=numpy),
        "motion_rgb_l1": _motion_l1(true_all, prediction_all, numpy=numpy),
        "interaction_rgb_l1": interaction_errors,
        "horizon_drift_slope": _slope(interaction_errors, numpy=numpy),
    }


def _rgb_l1(true_video: Any, prediction: Any, *, numpy: Any) -> float:
    if true_video.shape != prediction.shape or true_video.ndim != 5:
        raise CtrlWorldACWMEvaluationError("ACWM_VIDEO_SHAPE_INVALID")
    value = float(numpy.abs(true_video.astype(numpy.float32) - prediction.astype(numpy.float32)).mean())
    if not math.isfinite(value):
        raise CtrlWorldACWMEvaluationError("ACWM_METRIC_NONFINITE")
    return value


def _motion_l1(true_video: Any, prediction: Any, *, numpy: Any) -> float:
    if true_video.shape[1] < 2:
        raise CtrlWorldACWMEvaluationError("ACWM_MOTION_HORIZON_INVALID")
    true_delta = numpy.diff(true_video.astype(numpy.float32), axis=1)
    prediction_delta = numpy.diff(prediction.astype(numpy.float32), axis=1)
    value = float(numpy.abs(true_delta - prediction_delta).mean())
    if not math.isfinite(value):
        raise CtrlWorldACWMEvaluationError("ACWM_METRIC_NONFINITE")
    return value


def _slope(values: Sequence[float], *, numpy: Any) -> float:
    if len(values) < 2:
        raise CtrlWorldACWMEvaluationError("ACWM_SLOPE_HORIZON_INVALID")
    slope = float(numpy.polyfit(numpy.arange(len(values), dtype=numpy.float64), values, 1)[0])
    if not math.isfinite(slope):
        raise CtrlWorldACWMEvaluationError("ACWM_METRIC_NONFINITE")
    return slope


def _aggregate_metrics(
    paired_rows: Sequence[Mapping[str, object]],
    rollout_rows: Sequence[Mapping[str, object]],
    *,
    numpy: Any,
) -> dict[str, float]:
    if not paired_rows or not rollout_rows:
        raise CtrlWorldACWMEvaluationError("ACWM_MEASUREMENT_ROWS_EMPTY")
    metrics = {
        "long_horizon_prediction_error": float(
            numpy.mean([float(row["mean_rgb_l1"]) for row in paired_rows])
        ),
        "action_conditioning_error": float(
            numpy.mean([float(row["motion_rgb_l1"]) for row in paired_rows])
        ),
        "trajectory_fidelity_error": float(
            numpy.mean([float(row["mean_rgb_l1"]) for row in rollout_rows])
        ),
        "horizon_drift_slope": float(
            numpy.mean([float(row["horizon_drift_slope"]) for row in rollout_rows])
        ),
    }
    _validate_metrics(metrics)
    return metrics


def _validate_metrics(metrics: object) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRIC_IDS):
        raise CtrlWorldACWMEvaluationError("ACWM_METRIC_SET_INVALID")
    for name in _METRIC_IDS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CtrlWorldACWMEvaluationError(f"ACWM_METRIC_NONFINITE:{name}")


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a runtime dependency
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(path.rglob("*")):
        if not child.is_file() or child.is_symlink():
            continue
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_fingerprint(child).encode("ascii"))
    return digest.hexdigest()


def _source_revision(root: Path) -> Mapping[str, str]:
    head = root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return {"state": "unbound"}
    if value.startswith("ref: "):
        try:
            value = (root / ".git" / value[5:]).read_text(encoding="utf-8").strip()
        except OSError:
            return {"state": "unbound"}
    return {"state": "bound", "revision": value}


def _runtime_receipt(torch: Any) -> dict[str, object]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {"index": index, "name": properties.name, "total_memory_bytes": int(properties.total_memory)}
        )
    return {
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldACWMEvaluationError("ACWM_CONTRACT_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise CtrlWorldACWMEvaluationError("ACWM_CONTRACT_FILE_INVALID")
    return payload


def _write_result(destination: Path, result: Mapping[str, object]) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise CtrlWorldACWMEvaluationError("ACWM_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        payload = json.dumps(result, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
        (temporary / "measurement.json").write_text(payload, encoding="utf-8")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "verdiwm-ctrl-world-acwm-measurement-manifest",
                    "measurement_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "stage": result["stage"],
                    "contract_digest": result["contract_digest"],
                    "candidate": result["candidate"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--svd-model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", default="baseline")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = run_evaluation(args)
    print(json.dumps({"metrics": result["metrics"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
