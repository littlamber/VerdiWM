"""Raw-frame long-horizon ACWM probe runtime.

The upstream ACWM evaluator writes only aggregate metrics, so M1 needs a
separate runtime that keeps raw prediction/GT alignment visible without editing
the frozen vendor checkout.  Heavy imports are intentionally delayed so normal
repo tests can run in the lightweight uv environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.diagnoser import summarize_horizon_curve, summarize_segment_drift
from wmloop.diagnose.inverse_dynamics_train import spec_action_dim
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.vendor import verify_vendor_checkout


class HorizonProbeError(RuntimeError):
    """A long-horizon probe input or output failed closed."""


@dataclass(frozen=True)
class HorizonProbeConfig:
    repo_root: Path
    data_root: Path
    checkpoint_root: Path
    environment: str
    split: str
    output_root: Path
    vendor_root: Path | None = None
    horizons: tuple[int, ...] = (16, 32, 48, 64)
    max_trajectories: int = 2
    num_inference_steps: int = 50
    device: str = "cuda"
    seed: int = 7
    mode: str = "autoregressive"
    max_evidence: int = 1
    max_video_evidence: int = 0
    archive_db: Path | None = None
    cas_root: Path | None = None
    checkpoint_path: Path | None = None
    gpu_index: int | None = None
    gpu_exclusivity_audit_manifest: Path | None = None
    gpu_exclusivity_max_age_seconds: float | None = 300.0


def run_horizon_probe(config: HorizonProbeConfig) -> dict[str, object]:
    """Run one environment/split raw-frame horizon probe and write a manifest."""

    _validate_config(config)
    gpu_exclusivity = _verify_gpu_launch_preflight(
        device=config.device,
        gpu_index=config.gpu_index,
        manifest_path=config.gpu_exclusivity_audit_manifest,
        max_age_seconds=config.gpu_exclusivity_max_age_seconds,
    )
    started = time.monotonic()
    repo_root = Path(config.repo_root).resolve(strict=True)
    vendor_root = (
        Path(config.vendor_root).resolve(strict=True)
        if config.vendor_root is not None
        else repo_root / "vendor" / "ACWM-Phys"
    )
    if not vendor_root.is_dir():
        raise HorizonProbeError("HORIZON_VENDOR_ROOT_MISSING")
    vendor_runtime_sha256 = runtime_tree_sha256(vendor_root)
    source_revision = verify_vendor_checkout(repo_root)
    spec = _environment_spec(config.environment)
    data_root = Path(config.data_root).resolve(strict=True)
    checkpoint_root = Path(config.checkpoint_root).resolve(strict=True)
    checkpoint_path = _regular_file(
        Path(config.checkpoint_path) if config.checkpoint_path is not None else checkpoint_root / spec.checkpoint_relative_path,
        "HORIZON_CHECKPOINT_MISSING",
    )
    vae_path = _regular_file(checkpoint_root / "Wan2.1_VAE.pth", "HORIZON_VAE_MISSING")
    split_root = _trusted_directory(data_root / spec.dataset_relative_path / config.split, "HORIZON_SPLIT_MISSING")
    metadata_path = _regular_file(split_root / "metadata.pt", "HORIZON_METADATA_MISSING")
    output_root = Path(config.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise HorizonProbeError("HORIZON_OUTPUT_EXISTS")

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import yaml  # type: ignore[import-not-found]
    from skimage.metrics import structural_similarity  # type: ignore[import-not-found]

    device = _runtime_device(config.device, gpu_index=config.gpu_index, torch=torch)
    metadata_digest = _sha256_file(metadata_path)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, list) or not metadata:
        raise HorizonProbeError("HORIZON_METADATA_INVALID")
    vendor_env = _vendor_environment_name(config.environment)
    cfg_path = _regular_file(vendor_root / "configs" / "envs" / f"{vendor_env}.yaml", "HORIZON_CONFIG_MISSING")
    with _vendor_context(vendor_root=vendor_root, data_root=data_root, vae_path=vae_path):
        from eval import load_checkpoint, load_model  # type: ignore[import-not-found]

        model_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(model_config, dict):
            raise HorizonProbeError("HORIZON_CONFIG_INVALID")
        model_config.setdefault("model_config", {})["vae_config"] = [str(vae_path)]
        target_h, target_w = _obs_hw(model_config)
        temporal_rate = int(model_config.get("model_config", {}).get("temporal_compress_rate", 4))
        model = load_model(model_config, device)
        checkpoint_step = load_checkpoint(model, str(checkpoint_path), device)

    candidate_indices = _candidate_trajectory_indices(len(metadata), config.seed)
    trajectory_results: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    evidence_paths: list[Path] = []
    video_evidence_paths: list[Path] = []
    used_trajectory_count = 0
    attempted_trajectory_count = 0
    for trajectory_index in candidate_indices:
        if used_trajectory_count >= config.max_trajectories:
            break
        attempted_trajectory_count += 1
        entry = metadata[trajectory_index]
        if not isinstance(entry, Mapping):
            skipped.append({"trajectory_index": trajectory_index, "reason": "metadata_entry_invalid"})
            continue
        video_rel = entry.get("video_path")
        if not isinstance(video_rel, str) or not video_rel:
            skipped.append({"trajectory_index": trajectory_index, "reason": "video_path_missing"})
            continue
        actions = _action_array(entry, expected_dim=spec_action_dim(config.environment), np=np)
        if actions is None or actions.shape[0] < 2:
            skipped.append({"trajectory_index": trajectory_index, "reason": "actions_invalid"})
            continue
        usable_horizons = _usable_horizons(
            requested=config.horizons,
            available_steps=min(_entry_length(entry), int(actions.shape[0])),
            temporal_compress_rate=temporal_rate,
        )
        if not usable_horizons:
            skipped.append({"trajectory_index": trajectory_index, "reason": "horizon_not_available"})
            continue
        output_length = _model_output_length_for_horizon(max(usable_horizons), temporal_rate)
        video_path = _resolve_video_path(split_root, video_rel)
        try:
            frames = _load_rgb_video(video_path, target_hw=(target_h, target_w), max_frames=output_length, cv2=cv2, np=np)
        except HorizonProbeError as exc:
            skipped.append({"trajectory_index": trajectory_index, "reason": str(exc)})
            continue
        output_length = min(output_length, int(frames.shape[0]), int(actions.shape[0]))
        usable_horizons = tuple(horizon for horizon in usable_horizons if horizon <= output_length)
        if not usable_horizons:
            skipped.append({"trajectory_index": trajectory_index, "reason": "video_too_short"})
            continue
        output_length = _model_output_length_for_horizon(max(usable_horizons), temporal_rate)
        output_length = min(output_length, int(frames.shape[0]), int(actions.shape[0]))

        torch.manual_seed(config.seed + trajectory_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + trajectory_index)
        o_0 = torch.from_numpy(frames[0]).to(device=device, dtype=torch.float32).unsqueeze(0)
        action = torch.from_numpy(actions[:output_length]).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model.generate(
                o_0,
                action,
                num_inference_steps=config.num_inference_steps,
                noise_level=0.0,
                mode=config.mode,
            )
        prediction = prediction.detach().float().cpu().clamp(0, 1)
        gt = torch.from_numpy(frames[: min(output_length, int(prediction.shape[1]))]).float().unsqueeze(0)
        actual_generated_frames = int(min(prediction.shape[1], gt.shape[1]))
        metrics_by_horizon: dict[str, object] = {}
        per_frame_psnr: dict[int, float] = {}
        for horizon in usable_horizons:
            if actual_generated_frames < horizon:
                continue
            metrics = _compute_metrics(prediction[:, :horizon], gt[:, :horizon], np=np, structural_similarity=structural_similarity)
            metrics_by_horizon[str(horizon)] = metrics["summary"]
            for frame, value in metrics["per_frame_psnr"].items():
                per_frame_psnr[frame] = value
            if len(evidence_paths) < config.max_evidence:
                evidence_paths.append(
                    _write_evidence_frame(
                        root=output_root,
                        trajectory_index=trajectory_index,
                        horizon=horizon,
                        gt_frame=gt[0, horizon - 1].numpy(),
                        pred_frame=prediction[0, horizon - 1].numpy(),
                        cv2=cv2,
                        np=np,
                    )
                )
        if not metrics_by_horizon:
            skipped.append({"trajectory_index": trajectory_index, "reason": "generated_too_short"})
            continue
        retained_video_path: str | None = None
        if len(video_evidence_paths) < config.max_video_evidence:
            video_horizon = max(int(horizon) for horizon in metrics_by_horizon)
            temporary_video = _write_evidence_video(
                root=output_root,
                trajectory_index=trajectory_index,
                horizon=video_horizon,
                gt_frames=gt[0, :video_horizon].numpy(),
                pred_frames=prediction[0, :video_horizon].numpy(),
                cv2=cv2,
                np=np,
            )
            video_evidence_paths.append(temporary_video)
            retained_video_path = str(output_root / "videos" / temporary_video.name)
        trajectory_results.append(
            {
                "trajectory_index": trajectory_index,
                "video_path": str(video_path),
                "requested_video_path": video_rel,
                "entry_length": _entry_length(entry),
                "action_count": int(actions.shape[0]),
                "requested_output_length": _model_output_length_for_horizon(max(usable_horizons), temporal_rate),
                "generated_frame_count": actual_generated_frames,
                "usable_horizons": list(usable_horizons),
                "metrics_by_horizon": metrics_by_horizon,
                "per_frame_psnr": {str(frame): value for frame, value in sorted(per_frame_psnr.items())},
                "rollout_video_path": retained_video_path,
            }
        )
        used_trajectory_count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not trajectory_results:
        raise HorizonProbeError("HORIZON_NO_TRAJECTORIES_READY")
    aggregate = aggregate_probe_results(trajectory_results, horizons=config.horizons)
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-probe-run",
        "state": "ready",
        "environment": config.environment,
        "vendor_environment": vendor_env,
        "split": config.split,
        "goal_id": "g1_long_horizon",
        "probe_id": "horizon_curve",
        "source_revision": source_revision,
        "vendor_root": str(vendor_root),
        "vendor_runtime_sha256": vendor_runtime_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "config_path": str(cfg_path),
        "vae_path": str(vae_path),
        "horizons": list(config.horizons),
        "temporal_compress_rate": temporal_rate,
        "mode": config.mode,
        "num_inference_steps": config.num_inference_steps,
        "device": str(device),
        "gpu_index": config.gpu_index,
        "gpu_binding": _gpu_binding_receipt(device=device, gpu_index=config.gpu_index),
        "gpu_exclusivity_audit": gpu_exclusivity,
        "seed": config.seed,
        "max_trajectories": config.max_trajectories,
        "used_trajectory_count": used_trajectory_count,
        "attempted_trajectory_count": attempted_trajectory_count,
        "skipped_trajectories": skipped,
        "aggregate": aggregate,
        "trajectory_results": trajectory_results,
        "evidence_paths": [str(path) for path in evidence_paths],
        "video_evidence_paths": [str(output_root / "videos" / path.name) for path in video_evidence_paths],
        "duration_seconds": time.monotonic() - started,
    }
    if config.archive_db is not None:
        manifest["archive_refs_path"] = str(output_root / "archive-refs.json")
    temporary = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        evidence_target = temporary / "evidence"
        if evidence_paths:
            evidence_target.mkdir(mode=0o700)
            for path in evidence_paths:
                os.replace(path, evidence_target / path.name)
            manifest["evidence_paths"] = [str(output_root / "evidence" / path.name) for path in evidence_paths]
        if video_evidence_paths:
            video_target = temporary / "videos"
            video_target.mkdir(mode=0o700)
            for path in video_evidence_paths:
                os.replace(path, video_target / path.name)
        metrics_payload = {
            "schema_version": 1,
            "artifact_type": "wmloop-horizon-probe-metrics",
            "environment": config.environment,
            "split": config.split,
            "aggregate": aggregate,
            "trajectory_results": trajectory_results,
        }
        _write_json_atomic(temporary / "metrics.json", metrics_payload)
        _write_json_atomic(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        for path in evidence_paths:
            if path.exists() or path.is_symlink():
                path.unlink()
        for path in video_evidence_paths:
            if path.exists() or path.is_symlink():
                path.unlink()
        raise
    refs = _register_artifacts(output_root=output_root, archive_db=config.archive_db, cas_root=config.cas_root)
    if refs:
        _write_json_atomic(output_root / "archive-refs.json", refs)
        return {**manifest, "archive_refs": refs}
    return manifest


def aggregate_probe_results(
    trajectory_results: Sequence[Mapping[str, object]], *, horizons: Sequence[int]
) -> dict[str, object]:
    """Aggregate per-trajectory probe outputs into horizon and per-frame curves."""

    if not trajectory_results:
        raise HorizonProbeError("HORIZON_AGGREGATE_EMPTY")
    horizon_metrics: dict[str, dict[str, object]] = {}
    for horizon in horizons:
        samples: list[Mapping[str, object]] = []
        for item in trajectory_results:
            raw = item.get("metrics_by_horizon")
            if isinstance(raw, Mapping) and str(horizon) in raw and isinstance(raw[str(horizon)], Mapping):
                samples.append(raw[str(horizon)])  # type: ignore[arg-type]
        if not samples:
            continue
        horizon_metrics[str(horizon)] = {
            "sample_count": len(samples),
            "mse": _mean_metric(samples, "mse"),
            "masked_mse": _mean_metric(samples, "masked_mse"),
            "psnr": _mean_metric(samples, "psnr"),
            "ssim": _mean_metric(samples, "ssim"),
        }
    psnr_by_horizon = {
        int(horizon): float(metrics["psnr"])
        for horizon, metrics in horizon_metrics.items()
        if isinstance(metrics, Mapping) and "psnr" in metrics
    }
    per_frame_samples: dict[int, list[float]] = {}
    for item in trajectory_results:
        raw = item.get("per_frame_psnr")
        if not isinstance(raw, Mapping):
            continue
        for frame, value in raw.items():
            frame_index = int(frame)
            metric_value = float(value)
            if math.isfinite(metric_value):
                per_frame_samples.setdefault(frame_index, []).append(metric_value)
    per_frame_psnr = {
        frame: sum(values) / len(values)
        for frame, values in sorted(per_frame_samples.items())
        if values
    }
    curve: dict[str, object] = {}
    if len(psnr_by_horizon) >= 2:
        curve = summarize_horizon_curve(metric="psnr", observations=psnr_by_horizon)
    segment_drift: dict[str, object] | None = None
    if len(per_frame_psnr) >= 12:
        segment_drift = summarize_segment_drift(observations=per_frame_psnr, window=12)
    return {
        "horizon_metrics": horizon_metrics,
        "horizon_curve": curve,
        "per_frame_psnr": {str(frame): value for frame, value in per_frame_psnr.items()},
        "segment_drift": segment_drift,
    }


def summarize_horizon_probe_runs(
    *,
    runs_root: Path,
    output_path: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    environments: Sequence[str] | None = None,
) -> dict[str, object]:
    """Summarize the latest ready raw-frame horizon probe for each environment."""

    root = Path(runs_root).resolve()
    expected = tuple(environments or [spec.environment for spec in CANONICAL_ACWM_ENVIRONMENTS])
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    incomplete_horizons: list[dict[str, object]] = []
    archive: ArchiveStore | None = ArchiveStore(archive_db) if archive_db is not None else None
    cas: ContentAddressedStore | None = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent) if archive_db is not None else None
    for environment in expected:
        manifest_path = _latest_ready_manifest(root, environment)
        if manifest_path is None:
            missing.append(environment)
            continue
        manifest = _read_json(manifest_path)
        horizon_metrics = manifest["aggregate"]["horizon_metrics"]
        requested_horizons = [str(horizon) for horizon in manifest.get("horizons", sorted(horizon_metrics))]
        available_horizons = sorted(str(horizon) for horizon in horizon_metrics)
        missing_horizons = [horizon for horizon in requested_horizons if horizon not in horizon_metrics]
        if missing_horizons:
            incomplete_horizons.append(
                {
                    "environment": environment,
                    "available_horizons": available_horizons,
                    "missing_horizons": missing_horizons,
                }
            )
        row: dict[str, object] = {
            "environment": environment,
            "split": manifest["split"],
            "manifest_path": str(manifest_path),
            "checkpoint_path": manifest["checkpoint_path"],
            "metadata_sha256": manifest["metadata_sha256"],
            "used_trajectory_count": manifest["used_trajectory_count"],
            "attempted_trajectory_count": manifest["attempted_trajectory_count"],
            "available_horizons": available_horizons,
            "missing_horizons": missing_horizons,
            "horizon_metrics": horizon_metrics,
            "horizon_curve": manifest["aggregate"]["horizon_curve"],
            "segment_drift": manifest["aggregate"]["segment_drift"],
        }
        if archive is not None and cas is not None:
            refs = {
                "manifest_ref": _put_file(cas, manifest_path),
                "metrics_ref": _put_file(cas, manifest_path.parent / "metrics.json"),
            }
            for evidence in manifest.get("evidence_paths", []):
                if isinstance(evidence, str):
                    refs.setdefault("evidence_refs", []).append(_put_file(cas, Path(evidence)))  # type: ignore[union-attr]
            for value in refs.values():
                if isinstance(value, list):
                    for uri in value:
                        archive.record_artifact_reference(uri)
                else:
                    archive.record_artifact_reference(value)
            row.update(refs)
        rows.append(row)
    summary = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-probe-summary",
        "state": "ready" if not missing and not incomplete_horizons else "incomplete",
        "runs_root": str(root),
        "ready_count": len(rows),
        "environment_count": len(expected),
        "missing_environments": missing,
        "incomplete_horizon_environments": incomplete_horizons,
        "records": rows,
    }
    if output_path is not None:
        target = Path(output_path).resolve()
        if target.exists() or target.is_symlink():
            raise HorizonProbeError("HORIZON_SUMMARY_OUTPUT_EXISTS")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json_atomic(target, summary)
    return summary


def _validate_config(config: HorizonProbeConfig) -> None:
    _environment_spec(config.environment)
    if (
        not config.split
        or not config.horizons
        or len(set(config.horizons)) != len(config.horizons)
        or any(horizon < 2 for horizon in config.horizons)
        or config.max_trajectories < 1
        or config.num_inference_steps < 1
        or config.seed < 0
        or config.mode not in {"parallel", "autoregressive"}
        or config.max_evidence < 0
        or config.max_video_evidence < 0
        or not config.device
    ):
        raise HorizonProbeError("HORIZON_CONFIG_INVALID")
    if config.archive_db is None and config.cas_root is not None:
        raise HorizonProbeError("HORIZON_CAS_WITHOUT_ARCHIVE_INVALID")


def _verify_gpu_launch_preflight(
    *,
    device: str,
    gpu_index: int | None,
    manifest_path: Path | None,
    max_age_seconds: float | None,
) -> dict[str, object] | None:
    if not _device_requires_gpu(device):
        return None
    if gpu_index is None:
        raise HorizonProbeError("HORIZON_GPU_INDEX_REQUIRED")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise HorizonProbeError("HORIZON_GPU_INDEX_INVALID")
    return verify_gpu_exclusivity_ready(
        manifest_path,
        gpu_index=gpu_index,
        max_age_seconds=max_age_seconds,
    )


def _device_requires_gpu(device: str) -> bool:
    return device.strip().lower().startswith("cuda")


def _environment_spec(environment: str) -> Any:
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise HorizonProbeError(f"HORIZON_ENVIRONMENT_UNKNOWN:{environment}")


def _vendor_environment_name(environment: str) -> str:
    return "clothmove" if environment == "cloth_move" else environment


def _model_output_length_for_horizon(horizon: int, temporal_compress_rate: int) -> int:
    if horizon < 1 or temporal_compress_rate < 1:
        raise HorizonProbeError("HORIZON_LENGTH_INVALID")
    if temporal_compress_rate == 1:
        return horizon
    remainder = (horizon - 1) % temporal_compress_rate
    if remainder == 0:
        return horizon
    return horizon + temporal_compress_rate - remainder


def _usable_horizons(
    *, requested: Sequence[int], available_steps: int, temporal_compress_rate: int
) -> tuple[int, ...]:
    return tuple(
        horizon
        for horizon in sorted(requested)
        if _model_output_length_for_horizon(horizon, temporal_compress_rate) <= available_steps
    )


def _candidate_trajectory_indices(total: int, seed: int) -> tuple[int, ...]:
    if total < 1:
        raise HorizonProbeError("HORIZON_SELECTION_INVALID")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return tuple(indices)


def _resolve_video_path(split_root: Path, video_path: str) -> Path:
    raw = Path(video_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.append(split_root / raw.name)
    else:
        candidates.append(split_root / raw)
        candidates.append(split_root / raw.name)
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return candidates[-1]


def _action_array(entry: Mapping[str, Any], *, expected_dim: int, np: Any) -> Any | None:
    raw = entry.get("actions")
    if raw is None:
        raw = entry.get("commands")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        lin = raw.get("linear_velocity")
        ang = raw.get("angular_velocity")
        if lin is None or ang is None:
            return None
        array = np.stack([_to_numpy(lin, np), _to_numpy(ang, np)], axis=-1)
    else:
        array = _to_numpy(raw, np)
    if len(array.shape) != 2 or array.shape[1] != expected_dim:
        return None
    return array.astype("float32")


def _to_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype="float32")


def _entry_length(entry: Mapping[str, Any]) -> int:
    value = entry.get("length")
    if isinstance(value, int) and value > 0:
        return value
    action = entry.get("actions")
    if hasattr(action, "shape") and action.shape:
        return int(action.shape[0])
    commands = entry.get("commands")
    if hasattr(commands, "shape") and commands.shape:
        return int(commands.shape[0])
    if isinstance(commands, Mapping):
        lin = commands.get("linear_velocity")
        if hasattr(lin, "shape") and lin.shape:
            return int(lin.shape[0])
    return 0


def _load_rgb_video(path: Path, *, target_hw: tuple[int, int], max_frames: int, cv2: Any, np: Any) -> Any:
    if max_frames < 2:
        raise HorizonProbeError("HORIZON_VIDEO_LENGTH_INVALID")
    if not path.is_file() or path.is_symlink():
        raise HorizonProbeError("video_missing")
    capture = cv2.VideoCapture(str(path))
    frames = []
    target_h, target_w = target_hw
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype("float32") / 255.0)
    finally:
        capture.release()
    if len(frames) < 2:
        raise HorizonProbeError("video_too_short")
    return np.stack(frames)


def _compute_metrics(pred_video: Any, gt_video: Any, *, np: Any, structural_similarity: Any) -> dict[str, object]:
    """Compute vendor-compatible aggregate metrics plus per-frame PSNR."""

    import torch  # type: ignore[import-not-found]

    min_len = min(int(pred_video.shape[1]), int(gt_video.shape[1]))
    if min_len < 2:
        raise HorizonProbeError("HORIZON_METRIC_LENGTH_INVALID")
    pred = pred_video[:, :min_len]
    gt = gt_video[:, :min_len]
    mse_per_batch = ((pred - gt) ** 2).mean(dim=(1, 2, 3, 4))
    mse = float(mse_per_batch.mean().item())
    motion_diff = torch.abs(gt - gt[:, :1])
    motion_mask = motion_diff.max(dim=4, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    weight = (0.01 + motion_mask).expand_as(gt)
    masked_mse = float(((weight * (pred - gt) ** 2).sum() / (weight.sum() + 1e-8)).item())
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))
    pred_np = pred.numpy()
    gt_np = gt.numpy()
    ssim_values = []
    for batch in range(pred_np.shape[0]):
        for frame in range(pred_np.shape[1]):
            ssim_values.append(
                float(
                    structural_similarity(
                        pred_np[batch, frame],
                        gt_np[batch, frame],
                        data_range=1.0,
                        channel_axis=-1,
                    )
                )
            )
    per_frame_mse = ((pred - gt) ** 2).mean(dim=(0, 2, 3, 4))
    per_frame_psnr = {
        int(frame): float(10 * np.log10(1.0 / (float(value.item()) + 1e-8)))
        for frame, value in enumerate(per_frame_mse)
    }
    return {
        "summary": {
            "mse": mse,
            "masked_mse": masked_mse,
            "psnr": psnr,
            "ssim": float(np.mean(ssim_values)),
            "frame_count": min_len,
        },
        "per_frame_psnr": per_frame_psnr,
    }


def _write_evidence_frame(
    *,
    root: Path,
    trajectory_index: int,
    horizon: int,
    gt_frame: Any,
    pred_frame: Any,
    cv2: Any,
    np: Any,
) -> Path:
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root.parent / f".{root.name}.evidence.traj{trajectory_index}.h{horizon}.png"
    combined = np.concatenate([gt_frame, pred_frame], axis=1)
    payload = (np.clip(combined, 0, 1) * 255).astype("uint8")
    ok = cv2.imwrite(str(target), cv2.cvtColor(payload, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HorizonProbeError("HORIZON_EVIDENCE_WRITE_FAILED")
    return target


def _write_evidence_video(
    *,
    root: Path,
    trajectory_index: int,
    horizon: int,
    gt_frames: Any,
    pred_frames: Any,
    cv2: Any,
    np: Any,
) -> Path:
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root.parent / f".{root.name}.rollout.traj{trajectory_index}.h{horizon}.mp4"
    frame_count = min(len(gt_frames), len(pred_frames), horizon)
    if frame_count < 1:
        raise HorizonProbeError("HORIZON_VIDEO_EVIDENCE_EMPTY")
    height, width = gt_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (int(width * 2), int(height)),
    )
    if not writer.isOpened():
        raise HorizonProbeError("HORIZON_VIDEO_EVIDENCE_WRITE_FAILED")
    try:
        for frame_index in range(frame_count):
            combined = np.concatenate([gt_frames[frame_index], pred_frames[frame_index]], axis=1)
            payload = (np.clip(combined, 0, 1) * 255).astype("uint8")
            writer.write(cv2.cvtColor(payload, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not target.is_file() or target.stat().st_size < 1:
        raise HorizonProbeError("HORIZON_VIDEO_EVIDENCE_WRITE_FAILED")
    return target


def _obs_hw(config: Mapping[str, Any]) -> tuple[int, int]:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise HorizonProbeError("HORIZON_CONFIG_DATASET_INVALID")
    shape = dataset.get("obs_shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise HorizonProbeError("HORIZON_CONFIG_OBS_SHAPE_INVALID")
    channels, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    if channels != 3 or height < 8 or width < 8:
        raise HorizonProbeError("HORIZON_CONFIG_OBS_SHAPE_INVALID")
    return height, width


def _mean_metric(samples: Sequence[Mapping[str, object]], name: str) -> float:
    values = [float(sample[name]) for sample in samples if name in sample]
    if not values or any(not math.isfinite(value) for value in values):
        raise HorizonProbeError(f"HORIZON_METRIC_INVALID:{name}")
    return sum(values) / len(values)


def _latest_ready_manifest(root: Path, environment: str) -> Path | None:
    candidates: list[Path] = []
    for path in root.glob("*/manifest.json"):
        try:
            payload = _read_json(path)
        except HorizonProbeError:
            continue
        if payload.get("state") == "ready" and payload.get("environment") == environment:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_manifest_sort_key)


def _manifest_sort_key(path: Path) -> tuple[int, float, str]:
    name = path.parent.name
    try:
        suffix = int(name.rsplit("-r", 1)[1])
    except (IndexError, ValueError):
        suffix = -1
    return suffix, path.stat().st_mtime, name


def _runtime_device(device: str, *, gpu_index: int | None, torch: Any) -> Any:
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        return torch.device(device)
    if not torch.cuda.is_available():
        raise HorizonProbeError("HORIZON_CUDA_UNAVAILABLE")
    if gpu_index is None:
        raise HorizonProbeError("HORIZON_GPU_INDEX_REQUIRED")
    logical_index = _logical_cuda_index(
        gpu_index=gpu_index,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if logical_index >= int(torch.cuda.device_count()):
        raise HorizonProbeError("HORIZON_GPU_INDEX_UNAVAILABLE")
    if normalized != "cuda":
        match = re.fullmatch(r"cuda:(\d+)", normalized)
        if match is None or int(match.group(1)) != logical_index:
            raise HorizonProbeError("HORIZON_GPU_DEVICE_MISMATCH")
    torch.cuda.set_device(logical_index)
    return torch.device(f"cuda:{logical_index}")


def _logical_cuda_index(*, gpu_index: int, cuda_visible_devices: str | None) -> int:
    if cuda_visible_devices is None:
        return gpu_index
    visible = [value.strip() for value in cuda_visible_devices.split(",") if value.strip()]
    if not visible or any(not value.isdigit() for value in visible):
        raise HorizonProbeError("HORIZON_CUDA_VISIBLE_DEVICES_UNSUPPORTED")
    requested = str(gpu_index)
    if requested not in visible:
        raise HorizonProbeError("HORIZON_GPU_INDEX_NOT_VISIBLE")
    return visible.index(requested)


def _gpu_binding_receipt(*, device: Any, gpu_index: int | None) -> dict[str, object] | None:
    if gpu_index is None or not str(device).startswith("cuda:"):
        return None
    return {
        "physical_gpu_index": gpu_index,
        "logical_device_index": int(str(device).split(":", 1)[1]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@contextmanager
def _vendor_context(*, vendor_root: Path, data_root: Path, vae_path: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_data = os.environ.get("ACWM_DATA_ROOT")
    previous_vae = os.environ.get("WAN_VAE_PATH")
    previous_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(vendor_root))
    sys.dont_write_bytecode = True
    os.environ["ACWM_DATA_ROOT"] = str(data_root)
    os.environ["WAN_VAE_PATH"] = str(vae_path)
    os.chdir(vendor_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        _restore_env("ACWM_DATA_ROOT", previous_data)
        _restore_env("WAN_VAE_PATH", previous_vae)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _register_artifacts(*, output_root: Path, archive_db: Path | None, cas_root: Path | None) -> dict[str, object]:
    if archive_db is None:
        return {}
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    refs: dict[str, object] = {
        "manifest_ref": _put_file(cas, output_root / "manifest.json"),
        "metrics_ref": _put_file(cas, output_root / "metrics.json"),
        "evidence_refs": [],
        "video_evidence_refs": [],
    }
    for path in sorted((output_root / "evidence").glob("*.png")):
        refs["evidence_refs"].append(_put_file(cas, path))  # type: ignore[union-attr]
    for path in sorted((output_root / "videos").glob("*.mp4")):
        refs["video_evidence_refs"].append(_put_file(cas, path))  # type: ignore[union-attr]
    archive.record_artifact_reference(refs["manifest_ref"])  # type: ignore[arg-type]
    archive.record_artifact_reference(refs["metrics_ref"])  # type: ignore[arg-type]
    for uri in refs["evidence_refs"]:  # type: ignore[union-attr]
        archive.record_artifact_reference(uri)
    for uri in refs["video_evidence_refs"]:  # type: ignore[union-attr]
        archive.record_artifact_reference(uri)
    return refs


def _put_file(cas: ContentAddressedStore, path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HorizonProbeError("HORIZON_ARTIFACT_INVALID")
    media_type = {".png": "image/png", ".mp4": "video/mp4"}.get(path.suffix, "application/json")
    return cas.put_bytes(path.read_bytes(), media_type=media_type).uri


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HorizonProbeError(code)
    return candidate.resolve()


def _trusted_directory(path: Path, code: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise HorizonProbeError(code)
    return candidate.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HorizonProbeError("HORIZON_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise HorizonProbeError("HORIZON_MANIFEST_INVALID")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one raw-frame long-horizon probe")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--vendor-root", type=Path)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--checkpoint-path", type=Path)
    run.add_argument("--environment", required=True)
    run.add_argument("--split", default="ind_test")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48, 64])
    run.add_argument("--max-trajectories", type=int, default=2)
    run.add_argument("--num-inference-steps", type=int, default=50)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--mode", choices=("parallel", "autoregressive"), default="autoregressive")
    run.add_argument("--max-evidence", type=int, default=1)
    run.add_argument("--max-video-evidence", type=int, default=0)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--gpu-index", type=int)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    summarize = commands.add_parser("summarize", help="summarize latest ready horizon probe manifests")
    summarize.add_argument("--runs-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    summarize.add_argument("--archive-db", type=Path)
    summarize.add_argument("--cas-root", type=Path)
    summarize.add_argument("--environments", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_horizon_probe(
            HorizonProbeConfig(
                repo_root=args.repo_root,
                data_root=args.data_root,
                checkpoint_root=args.checkpoint_root,
                environment=args.environment,
                split=args.split,
                output_root=args.output_root,
                vendor_root=args.vendor_root,
                horizons=tuple(args.horizons),
                max_trajectories=args.max_trajectories,
                num_inference_steps=args.num_inference_steps,
                device=args.device,
                seed=args.seed,
                mode=args.mode,
                max_evidence=args.max_evidence,
                max_video_evidence=args.max_video_evidence,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                checkpoint_path=args.checkpoint_path,
                gpu_index=args.gpu_index,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            )
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    if args.command == "summarize":
        summary = summarize_horizon_probe_runs(
            runs_root=args.runs_root,
            output_path=args.output,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            environments=args.environments,
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
