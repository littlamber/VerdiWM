"""Run measured M1 appearance/action/OOD raw probes with ACWM inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.diagnoser import summarize_horizon_curve
from wmloop.diagnose.horizon_runtime import (
    HorizonProbeError,
    _action_array,
    _candidate_trajectory_indices,
    _device_requires_gpu,
    _compute_metrics,
    _entry_length,
    _environment_spec,
    _load_rgb_video,
    _model_output_length_for_horizon,
    _obs_hw,
    _regular_file,
    _resolve_video_path,
    _runtime_device,
    _sha256_file,
    _trusted_directory,
    _usable_horizons,
    _vendor_context,
    _vendor_environment_name,
)
from wmloop.diagnose.inverse_dynamics_train import _InverseDynamicsMLP, spec_action_dim
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.vendor import verify_vendor_checkout


class RawProbeMeasureRuntimeError(RuntimeError):
    """Raw-probe measured runtime failed closed."""


@dataclass(frozen=True)
class RawProbeMeasureConfig:
    repo_root: Path
    data_root: Path
    checkpoint_root: Path
    inverse_summary_path: Path
    environment: str
    output_root: Path
    vendor_root: Path | None = None
    ind_split: str = "ind_test"
    ood_split: str = "ood_test"
    horizons: tuple[int, ...] = (16, 32, 48, 64)
    primary_horizon: int = 64
    max_trajectories: int = 1
    num_inference_steps: int = 4
    device: str = "cuda"
    seed: int = 7
    mode: str = "autoregressive"
    action_tolerance: float = 0.1
    archive_db: Path | None = None
    cas_root: Path | None = None
    checkpoint_path: Path | None = None
    action_only: bool = False
    gpu_index: int | None = None
    gpu_exclusivity_audit_manifest: Path | None = None
    gpu_exclusivity_max_age_seconds: float | None = 300.0


def run_raw_probe_measurements(config: RawProbeMeasureConfig) -> dict[str, object]:
    """Run ACWM action/no-action and InD/OoD measurements for one environment."""

    _validate_config(config)
    gpu_exclusivity = _verify_gpu_launch_preflight(
        device=config.device,
        gpu_index=config.gpu_index,
        manifest_path=config.gpu_exclusivity_audit_manifest,
        max_age_seconds=config.gpu_exclusivity_max_age_seconds,
    )
    started = time.monotonic()
    repo_root = Path(config.repo_root).resolve(strict=True)
    data_root = Path(config.data_root).resolve(strict=True)
    checkpoint_root = Path(config.checkpoint_root).resolve(strict=True)
    output_root = Path(config.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_OUTPUT_EXISTS")
    source_revision = verify_vendor_checkout(repo_root)
    spec = _environment_spec(config.environment)
    inverse = _inverse_record(config.inverse_summary_path, config.environment)
    checkpoint_path = _regular_file(
        Path(config.checkpoint_path) if config.checkpoint_path is not None else checkpoint_root / spec.checkpoint_relative_path,
        "RAW_MEASURE_CHECKPOINT_MISSING",
    )
    vae_path = _regular_file(checkpoint_root / "Wan2.1_VAE.pth", "RAW_MEASURE_VAE_MISSING")
    vendor_root = (
        Path(config.vendor_root).resolve(strict=True)
        if config.vendor_root is not None
        else repo_root / "vendor" / "ACWM-Phys"
    )
    if not vendor_root.is_dir():
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_VENDOR_ROOT_MISSING")
    vendor_runtime_sha256 = runtime_tree_sha256(vendor_root)
    vendor_env = _vendor_environment_name(config.environment)
    cfg_path = _regular_file(vendor_root / "configs" / "envs" / f"{vendor_env}.yaml", "RAW_MEASURE_CONFIG_MISSING")

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import yaml  # type: ignore[import-not-found]
    from skimage.metrics import structural_similarity  # type: ignore[import-not-found]

    device = _runtime_device(config.device, gpu_index=config.gpu_index, torch=torch)
    with _vendor_context(vendor_root=vendor_root, data_root=data_root, vae_path=vae_path):
        from eval import load_checkpoint, load_model  # type: ignore[import-not-found]

        model_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(model_config, dict):
            raise RawProbeMeasureRuntimeError("RAW_MEASURE_CONFIG_INVALID")
        model_config.setdefault("model_config", {})["vae_config"] = [str(vae_path)]
        target_h, target_w = _obs_hw(model_config)
        temporal_rate = int(model_config.get("model_config", {}).get("temporal_compress_rate", 4))
        model = load_model(model_config, device)
        checkpoint_step = load_checkpoint(model, str(checkpoint_path), device)
    inverse_model = _load_inverse_model(
        checkpoint_path=Path(str(inverse["checkpoint_path"])),
        environment=config.environment,
        device=device,
        torch=torch,
    )
    ind_result = _run_split(
        config=config,
        split=config.ind_split,
        spec=spec,
        data_root=data_root,
        model=model,
        inverse_model=inverse_model,
        target_hw=(target_h, target_w),
        temporal_rate=temporal_rate,
        device=device,
        cv2=cv2,
        np=np,
        torch=torch,
        structural_similarity=structural_similarity,
        run_no_action=True,
    )
    if config.action_only:
        record = build_action_measurement_record(
            environment=config.environment,
            split=config.ind_split,
            predicted_actions=ind_result["predicted_actions"],
            target_actions=ind_result["target_actions"],
            action_conditioned_psnr=float(ind_result["primary_action_psnr"]),
            no_action_psnr=float(ind_result["primary_no_action_psnr"]),
            inverse_dynamics_r2=float(inverse["gt_r2"]),
            low_confidence=bool(inverse["low_confidence"]),
            action_tolerance=config.action_tolerance,
        )
        measurements = {
            "schema_version": 1,
            "artifact_type": "wmloop-action-following-measurement-input",
            "source_kind": "measured",
            "records": [record],
        }
        ood_result = None
    else:
        ood_result = _run_split(
            config=config,
            split=config.ood_split,
            spec=spec,
            data_root=data_root,
            model=model,
            inverse_model=inverse_model,
            target_hw=(target_h, target_w),
            temporal_rate=temporal_rate,
            device=device,
            cv2=cv2,
            np=np,
            torch=torch,
            structural_similarity=structural_similarity,
            run_no_action=False,
        )
        record = build_measurement_record(
            environment=config.environment,
            split=config.ind_split,
            motion_magnitudes=ind_result["motion_magnitudes"],
            ssim_scores=ind_result["ssim_scores"],
            predicted_actions=ind_result["predicted_actions"],
            target_actions=ind_result["target_actions"],
            action_conditioned_psnr=float(ind_result["primary_action_psnr"]),
            no_action_psnr=float(ind_result["primary_no_action_psnr"]),
            inverse_dynamics_r2=float(inverse["gt_r2"]),
            low_confidence=bool(inverse["low_confidence"]),
            ind_auc=float(ind_result["auc_psnr"]),
            ood_auc_by_condition={config.ood_split: float(ood_result["auc_psnr"])},
            action_tolerance=config.action_tolerance,
        )
        measurements = {
            "schema_version": 1,
            "artifact_type": "wmloop-raw-probe-measurement-input",
            "source_kind": "measured",
            "records": [record],
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-action-following-measure-runtime-manifest"
        if config.action_only
        else "wmloop-m1-raw-probe-measure-runtime-manifest",
        "state": "ready",
        "environment": config.environment,
        "source_revision": source_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "vendor_root": str(vendor_root),
        "vendor_runtime_sha256": vendor_runtime_sha256,
        "inverse_summary_path": str(Path(config.inverse_summary_path).resolve()),
        "inverse_checkpoint_path": inverse["checkpoint_path"],
        "inverse_dynamics_r2": float(inverse["gt_r2"]),
        "inverse_dynamics_low_confidence": bool(inverse["low_confidence"]),
        "ind_split_result": _split_public_result(ind_result),
        "ood_split_result": None if ood_result is None else _split_public_result(ood_result),
        "action_only": config.action_only,
        "primary_horizon": config.primary_horizon,
        "horizons": list(config.horizons),
        "num_inference_steps": config.num_inference_steps,
        "max_trajectories": config.max_trajectories,
        "device": str(device),
        "gpu_exclusivity_audit": gpu_exclusivity,
        "duration_seconds": time.monotonic() - started,
    }
    return _write_bundle(measurements=measurements, manifest=manifest, output_root=output_root, archive_db=config.archive_db, cas_root=config.cas_root)


def build_measurement_record(
    *,
    environment: str,
    split: str,
    motion_magnitudes: Sequence[float],
    ssim_scores: Sequence[float],
    predicted_actions: Sequence[Sequence[float]],
    target_actions: Sequence[Sequence[float]],
    action_conditioned_psnr: float,
    no_action_psnr: float,
    inverse_dynamics_r2: float,
    low_confidence: bool,
    ind_auc: float,
    ood_auc_by_condition: Mapping[str, float],
    action_tolerance: float,
) -> dict[str, object]:
    """Build the measurement-input record consumed by raw_probe_evidence."""

    if (
        not environment
        or not split
        or len(motion_magnitudes) != len(ssim_scores)
        or not motion_magnitudes
        or len(predicted_actions) != len(target_actions)
        or not predicted_actions
        or not all(_finite(value) for value in (action_conditioned_psnr, no_action_psnr, inverse_dynamics_r2, ind_auc, action_tolerance))
        or action_tolerance < 0
        or not isinstance(low_confidence, bool)
        or not ood_auc_by_condition
    ):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_RECORD_INVALID")
    return {
        "environment": environment,
        "split": split,
        "appearance_drift": {
            "motion_magnitudes": [float(value) for value in motion_magnitudes],
            "ssim_scores": [float(value) for value in ssim_scores],
            "low_motion_fraction": 0.25,
        },
        "action_following": {
            "predicted_actions": [[float(value) for value in row] for row in predicted_actions],
            "target_actions": [[float(value) for value in row] for row in target_actions],
            "tolerance": float(action_tolerance),
            "action_conditioned_psnr": float(action_conditioned_psnr),
            "no_action_psnr": float(no_action_psnr),
            "inverse_dynamics_r2": float(inverse_dynamics_r2),
            "low_confidence": low_confidence,
        },
        "ood_profile": {
            "ind_auc": float(ind_auc),
            "ood_auc_by_condition": {str(key): float(value) for key, value in ood_auc_by_condition.items()},
        },
    }


def build_action_measurement_record(
    *,
    environment: str,
    split: str,
    predicted_actions: Sequence[Sequence[float]],
    target_actions: Sequence[Sequence[float]],
    action_conditioned_psnr: float,
    no_action_psnr: float,
    inverse_dynamics_r2: float,
    low_confidence: bool,
    action_tolerance: float,
) -> dict[str, object]:
    """Build an action-following-only measurement record for verifier gating."""

    if (
        not environment
        or not split
        or len(predicted_actions) != len(target_actions)
        or not predicted_actions
        or not all(
            _finite(value)
            for value in (action_conditioned_psnr, no_action_psnr, inverse_dynamics_r2, action_tolerance)
        )
        or action_tolerance < 0
        or not isinstance(low_confidence, bool)
    ):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_ACTION_RECORD_INVALID")
    return {
        "environment": environment,
        "split": split,
        "action_following": {
            "predicted_actions": [[float(value) for value in row] for row in predicted_actions],
            "target_actions": [[float(value) for value in row] for row in target_actions],
            "tolerance": float(action_tolerance),
            "action_conditioned_psnr": float(action_conditioned_psnr),
            "no_action_psnr": float(no_action_psnr),
            "inverse_dynamics_r2": float(inverse_dynamics_r2),
            "low_confidence": low_confidence,
        },
    }


def _run_split(
    *,
    config: RawProbeMeasureConfig,
    split: str,
    spec: Any,
    data_root: Path,
    model: Any,
    inverse_model: Mapping[str, Any],
    target_hw: tuple[int, int],
    temporal_rate: int,
    device: Any,
    cv2: Any,
    np: Any,
    torch: Any,
    structural_similarity: Any,
    run_no_action: bool,
) -> dict[str, object]:
    split_root = _trusted_directory(data_root / spec.dataset_relative_path / split, "RAW_MEASURE_SPLIT_MISSING")
    metadata_path = _regular_file(split_root / "metadata.pt", "RAW_MEASURE_METADATA_MISSING")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, list) or not metadata:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_METADATA_INVALID")
    selected = _candidate_trajectory_indices(len(metadata), config.seed)
    trajectory_results: list[dict[str, object]] = []
    motion_magnitudes: list[float] = []
    ssim_scores: list[float] = []
    predicted_actions: list[list[float]] = []
    target_actions: list[list[float]] = []
    skipped: list[dict[str, object]] = []
    attempted = 0
    for trajectory_index in selected:
        if len(trajectory_results) >= config.max_trajectories:
            break
        attempted += 1
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
        available_steps = min(_entry_length(entry), int(actions.shape[0]))
        usable_horizons = _usable_horizons(requested=config.horizons, available_steps=available_steps, temporal_compress_rate=temporal_rate)
        if config.primary_horizon not in usable_horizons:
            skipped.append({"trajectory_index": trajectory_index, "reason": "primary_horizon_not_available"})
            continue
        output_length = _model_output_length_for_horizon(max(usable_horizons), temporal_rate)
        video_path = _resolve_video_path(split_root, video_rel)
        try:
            frames = _load_rgb_video(video_path, target_hw=target_hw, max_frames=output_length, cv2=cv2, np=np)
        except HorizonProbeError as exc:
            skipped.append({"trajectory_index": trajectory_index, "reason": str(exc)})
            continue
        output_length = min(output_length, int(frames.shape[0]), int(actions.shape[0]))
        usable_horizons = tuple(horizon for horizon in usable_horizons if horizon <= output_length)
        if config.primary_horizon not in usable_horizons:
            skipped.append({"trajectory_index": trajectory_index, "reason": "video_too_short_for_primary"})
            continue
        torch.manual_seed(config.seed + trajectory_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + trajectory_index)
        o_0 = torch.from_numpy(frames[0]).to(device=device, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.from_numpy(actions[:output_length]).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model.generate(
                o_0,
                action_tensor,
                num_inference_steps=config.num_inference_steps,
                noise_level=0.0,
                mode=config.mode,
            ).detach().float().cpu().clamp(0, 1)
            no_action_prediction = None
            if run_no_action:
                no_action_prediction = model.generate(
                    o_0,
                    torch.zeros_like(action_tensor),
                    num_inference_steps=config.num_inference_steps,
                    noise_level=0.0,
                    mode=config.mode,
                ).detach().float().cpu().clamp(0, 1)
        gt = torch.from_numpy(frames[: min(output_length, int(prediction.shape[1]))]).float().unsqueeze(0)
        metrics_by_horizon: dict[str, object] = {}
        for horizon in usable_horizons:
            if int(prediction.shape[1]) < horizon or int(gt.shape[1]) < horizon:
                continue
            metrics_by_horizon[str(horizon)] = _compute_metrics(
                prediction[:, :horizon],
                gt[:, :horizon],
                np=np,
                structural_similarity=structural_similarity,
            )["summary"]
        if str(config.primary_horizon) not in metrics_by_horizon:
            skipped.append({"trajectory_index": trajectory_index, "reason": "primary_metric_missing"})
            continue
        primary_action_psnr = float(metrics_by_horizon[str(config.primary_horizon)]["psnr"])  # type: ignore[index]
        primary_no_action_psnr = primary_action_psnr
        if no_action_prediction is not None:
            no_action_metrics = _compute_metrics(
                no_action_prediction[:, : config.primary_horizon],
                gt[:, : config.primary_horizon],
                np=np,
                structural_similarity=structural_similarity,
            )["summary"]
            primary_no_action_psnr = float(no_action_metrics["psnr"])  # type: ignore[index]
        if run_no_action:
            _extend_appearance_arrays(
                gt=gt[0, : config.primary_horizon].numpy(),
                prediction=prediction[0, : config.primary_horizon].numpy(),
                motion_magnitudes=motion_magnitudes,
                ssim_scores=ssim_scores,
                np=np,
                structural_similarity=structural_similarity,
            )
            inv_pred, inv_target = _inverse_action_predictions(
                prediction=prediction[0, : config.primary_horizon].numpy(),
                target_actions=actions[: config.primary_horizon],
                inverse_model=inverse_model,
                cv2=cv2,
                np=np,
                torch=torch,
            )
            predicted_actions.extend(inv_pred)
            target_actions.extend(inv_target)
        trajectory_results.append(
            {
                "trajectory_index": trajectory_index,
                "usable_horizons": list(usable_horizons),
                "metrics_by_horizon": metrics_by_horizon,
                "primary_action_psnr": primary_action_psnr,
                "primary_no_action_psnr": primary_no_action_psnr,
            }
        )
    if not trajectory_results:
        raise RawProbeMeasureRuntimeError(f"RAW_MEASURE_NO_TRAJECTORIES_READY:{split}")
    auc = _auc_from_trajectory_metrics(trajectory_results, config.horizons)
    return {
        "split": split,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "used_trajectory_count": len(trajectory_results),
        "attempted_trajectory_count": attempted,
        "skipped_trajectories": skipped,
        "trajectory_results": trajectory_results,
        "auc_psnr": auc,
        "motion_magnitudes": motion_magnitudes,
        "ssim_scores": ssim_scores,
        "predicted_actions": predicted_actions,
        "target_actions": target_actions,
        "primary_action_psnr": _mean([float(item["primary_action_psnr"]) for item in trajectory_results]),
        "primary_no_action_psnr": _mean([float(item["primary_no_action_psnr"]) for item in trajectory_results]),
    }


def _extend_appearance_arrays(
    *,
    gt: Any,
    prediction: Any,
    motion_magnitudes: list[float],
    ssim_scores: list[float],
    np: Any,
    structural_similarity: Any,
) -> None:
    for frame in range(1, min(int(gt.shape[0]), int(prediction.shape[0]))):
        motion_magnitudes.append(float(np.mean(np.abs(gt[frame] - gt[frame - 1]))))
        ssim_scores.append(
            float(
                structural_similarity(
                    prediction[frame],
                    gt[frame],
                    data_range=1.0,
                    channel_axis=-1,
                )
            )
        )


def _inverse_action_predictions(
    *,
    prediction: Any,
    target_actions: Any,
    inverse_model: Mapping[str, Any],
    cv2: Any,
    np: Any,
    torch: Any,
) -> tuple[list[list[float]], list[list[float]]]:
    model = inverse_model["model"]
    image_size = int(inverse_model["image_size"])
    target_mean = inverse_model["target_mean"]
    target_std = inverse_model["target_std"]
    device = inverse_model["device"]
    features = []
    targets = []
    limit = min(int(prediction.shape[0]) - 1, int(target_actions.shape[0]) - 1)
    for frame in range(max(0, limit)):
        before = _gray_frame(prediction[frame], image_size=image_size, cv2=cv2, np=np)
        after = _gray_frame(prediction[frame + 1], image_size=image_size, cv2=cv2, np=np)
        features.append(np.stack([before, after, after - before], axis=0).astype("float32").reshape(-1))
        targets.append([float(value) for value in target_actions[frame]])
    if not features:
        return [], []
    with torch.no_grad():
        x = torch.from_numpy(np.stack(features).astype("float32")).to(device)
        pred = (model(x).cpu() * target_std + target_mean).numpy()
    return [[float(value) for value in row] for row in pred], targets


def _gray_frame(frame: Any, *, image_size: int, cv2: Any, np: Any) -> Any:
    payload = (np.clip(frame, 0, 1) * 255).astype("uint8")
    gray = cv2.cvtColor(payload, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return gray.astype("float32") / 255.0


def _auc_from_trajectory_metrics(trajectory_results: Sequence[Mapping[str, object]], horizons: Sequence[int]) -> float:
    samples: dict[int, list[float]] = {}
    for item in trajectory_results:
        raw = item.get("metrics_by_horizon")
        if not isinstance(raw, Mapping):
            continue
        for horizon in horizons:
            metrics = raw.get(str(horizon))
            if isinstance(metrics, Mapping) and _finite(metrics.get("psnr")):
                samples.setdefault(horizon, []).append(float(metrics["psnr"]))
    means = {horizon: _mean(values) for horizon, values in samples.items() if values}
    if len(means) < 2:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_AUC_INSUFFICIENT")
    curve = summarize_horizon_curve(metric="psnr", observations=means)
    key = f"auc_psnr_{min(means)}_{max(means)}"
    value = curve.get(key)
    if not _finite(value):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_AUC_INVALID")
    return float(value)


def _load_inverse_model(*, checkpoint_path: Path, environment: str, device: Any, torch: Any) -> Mapping[str, Any]:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_CHECKPOINT_MISSING")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-inverse-dynamics-head":
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_CHECKPOINT_INVALID")
    image_size = int(payload.get("image_size", 32))
    hidden_dim = _hidden_dim(str(payload.get("architecture", "")))
    action_dim = spec_action_dim(environment)
    model = _InverseDynamicsMLP(3 * image_size * image_size, action_dim, hidden_dim)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "image_size": image_size,
        "target_mean": payload["target_mean"],
        "target_std": payload["target_std"],
        "device": device,
    }


def _hidden_dim(architecture: str) -> int:
    match = re.search(r"3x([0-9]+)-gray", architecture)
    if match is None:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_ARCH_INVALID")
    return int(match.group(1))


def _inverse_record(summary_path: Path, environment: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_SUMMARY_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-inverse-dynamics-cache-summary":
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_SUMMARY_INVALID")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_INVERSE_SUMMARY_INVALID")
    for record in records:
        if isinstance(record, Mapping) and record.get("environment") == environment:
            return record
    raise RawProbeMeasureRuntimeError(f"RAW_MEASURE_INVERSE_RECORD_MISSING:{environment}")


def _split_public_result(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "split": result["split"],
        "metadata_path": result["metadata_path"],
        "metadata_sha256": result["metadata_sha256"],
        "used_trajectory_count": result["used_trajectory_count"],
        "attempted_trajectory_count": result["attempted_trajectory_count"],
        "auc_psnr": result["auc_psnr"],
        "primary_action_psnr": result["primary_action_psnr"],
        "primary_no_action_psnr": result["primary_no_action_psnr"],
        "skipped_trajectories": result["skipped_trajectories"],
    }


def _write_bundle(
    *,
    measurements: Mapping[str, object],
    manifest: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    measurements_bytes = _canonical_json_bytes(measurements)
    manifest_bytes = _canonical_json_bytes(manifest)
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_bytes_atomic(temporary / "measurements.json", measurements_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload in (("raw_probe_measurements", measurements_bytes), ("raw_probe_measure_runtime_manifest", manifest_bytes)):
                ref = cas.put_bytes(payload, media_type="application/json").uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        final_manifest = {**manifest, "cas_refs": cas_refs, "measurements_path": str(destination / "measurements.json")}
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(final_manifest))
        os.replace(temporary, destination)
        return final_manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _validate_config(config: RawProbeMeasureConfig) -> None:
    _environment_spec(config.environment)
    if (
        not config.ind_split
        or not config.ood_split
        or not config.horizons
        or config.primary_horizon not in config.horizons
        or config.max_trajectories < 1
        or config.num_inference_steps < 1
        or config.seed < 0
        or config.mode not in {"parallel", "autoregressive"}
        or not _finite(config.action_tolerance)
        or config.action_tolerance < 0
        or not config.device
        or not isinstance(config.action_only, bool)
    ):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_CONFIG_INVALID")
    if config.archive_db is None and config.cas_root is not None:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_CAS_WITHOUT_ARCHIVE_INVALID")


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
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_GPU_INDEX_REQUIRED")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_GPU_INDEX_INVALID")
    return verify_gpu_exclusivity_ready(
        manifest_path,
        gpu_index=gpu_index,
        max_age_seconds=max_age_seconds,
    )


def _mean(values: Sequence[float]) -> float:
    if not values or any(not _finite(value) for value in values):
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_MEAN_INVALID")
    return sum(values) / len(values)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RawProbeMeasureRuntimeError("RAW_MEASURE_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one measured raw-probe bundle")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--vendor-root", type=Path)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--inverse-summary", type=Path, required=True)
    run.add_argument("--environment", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--ind-split", default="ind_test")
    run.add_argument("--ood-split", default="ood_test")
    run.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48, 64])
    run.add_argument("--primary-horizon", type=int, default=64)
    run.add_argument("--max-trajectories", type=int, default=1)
    run.add_argument("--num-inference-steps", type=int, default=4)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--mode", choices=("parallel", "autoregressive"), default="autoregressive")
    run.add_argument("--action-tolerance", type=float, default=0.1)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--checkpoint-path", type=Path)
    run.add_argument("--action-only", action="store_true")
    run.add_argument("--gpu-index", type=int)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_raw_probe_measurements(
            RawProbeMeasureConfig(
                repo_root=args.repo_root,
                data_root=args.data_root,
                checkpoint_root=args.checkpoint_root,
                inverse_summary_path=args.inverse_summary,
                environment=args.environment,
                output_root=args.output_root,
                vendor_root=args.vendor_root,
                ind_split=args.ind_split,
                ood_split=args.ood_split,
                horizons=tuple(args.horizons),
                primary_horizon=args.primary_horizon,
                max_trajectories=args.max_trajectories,
                num_inference_steps=args.num_inference_steps,
                device=args.device,
                seed=args.seed,
                mode=args.mode,
                action_tolerance=args.action_tolerance,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                checkpoint_path=args.checkpoint_path,
                action_only=args.action_only,
                gpu_index=args.gpu_index,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            )
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
