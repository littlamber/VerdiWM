#!/usr/bin/env python3
"""Measure pour-water event completion from paired GT/prediction rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.diagnose.pour_water_video_adapter import diagnose_paired_rollout_video


class PourWaterEventGateError(RuntimeError):
    """The paired rollout could not support a defensible event measurement."""


def classify_event_measurements(
    rows: Sequence[Mapping[str, object]],
    *,
    min_completion_ratio: float = 0.25,
    min_completion_uplift: float = 0.05,
    max_onset_delay_frames: int = 30,
) -> dict[str, object]:
    if not rows or not 0.0 <= min_completion_ratio <= 1.0 or min_completion_uplift < 0.0:
        raise PourWaterEventGateError("POUR_WATER_EVENT_CLASSIFICATION_INVALID")
    if max_onset_delay_frames < 0:
        raise PourWaterEventGateError("POUR_WATER_EVENT_CLASSIFICATION_INVALID")
    normalized: list[dict[str, object]] = []
    for row in rows:
        baseline = _finite(row.get("baseline_completion_ratio"), "baseline_completion_ratio")
        candidate = _finite(row.get("candidate_completion_ratio"), "candidate_completion_ratio")
        baseline_delay = _nonnegative_int(row.get("baseline_onset_delay_frames"), "baseline_onset_delay_frames")
        candidate_delay = _nonnegative_int(row.get("candidate_onset_delay_frames"), "candidate_onset_delay_frames")
        normalized.append(
            {
                "trajectory_index": _nonnegative_int(row.get("trajectory_index"), "trajectory_index"),
                "baseline_completion_ratio": baseline,
                "candidate_completion_ratio": candidate,
                "completion_uplift": candidate - baseline,
                "baseline_event_pass": baseline >= min_completion_ratio and baseline_delay <= max_onset_delay_frames,
                "candidate_event_pass": candidate >= min_completion_ratio and candidate_delay <= max_onset_delay_frames,
                "baseline_onset_delay_frames": baseline_delay,
                "candidate_onset_delay_frames": candidate_delay,
            }
        )
    baseline_mean = _mean(float(row["baseline_completion_ratio"]) for row in normalized)
    candidate_mean = _mean(float(row["candidate_completion_ratio"]) for row in normalized)
    uplift = candidate_mean - baseline_mean
    candidate_pass = all(bool(row["candidate_event_pass"]) for row in normalized)
    baseline_pass = all(bool(row["baseline_event_pass"]) for row in normalized)
    if candidate_pass and uplift >= min_completion_uplift:
        classification = "event_positive"
    elif candidate_pass and baseline_pass:
        classification = "event_equivalent"
    elif candidate_mean + min_completion_uplift < baseline_mean:
        classification = "event_regression"
    else:
        classification = "event_failure"
    return {
        "classification": classification,
        "candidate_event_pass": candidate_pass,
        "baseline_event_pass": baseline_pass,
        "baseline_mean_completion_ratio": baseline_mean,
        "candidate_mean_completion_ratio": candidate_mean,
        "mean_completion_uplift": uplift,
        "thresholds": {
            "min_completion_ratio": min_completion_ratio,
            "min_completion_uplift": min_completion_uplift,
            "max_onset_delay_frames": max_onset_delay_frames,
        },
        "trajectory_results": normalized,
    }


def build_pour_water_event_gate(
    *,
    baseline_manifest: Path,
    candidate_manifest: Path,
    output_root: Path,
    min_completion_ratio: float = 0.25,
    min_completion_uplift: float = 0.05,
    max_onset_delay_frames: int = 30,
    max_gt_panel_mse: float = 1e-4,
    primitive: str = "",
    seed: int | None = None,
) -> dict[str, object]:
    baseline_path = Path(baseline_manifest).resolve(strict=True)
    candidate_path = Path(candidate_manifest).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PourWaterEventGateError("POUR_WATER_EVENT_OUTPUT_EXISTS")
    if primitive and (not primitive.strip() or any(character.isspace() for character in primitive)):
        raise PourWaterEventGateError("POUR_WATER_EVENT_PRIMITIVE_INVALID")
    if seed is not None and (isinstance(seed, bool) or seed < 0):
        raise PourWaterEventGateError("POUR_WATER_EVENT_SEED_INVALID")
    baseline = _load_manifest(baseline_path)
    candidate = _load_manifest(candidate_path)
    for field in ("environment", "split", "metadata_sha256", "mode"):
        if baseline.get(field) != candidate.get(field):
            raise PourWaterEventGateError(f"POUR_WATER_EVENT_PAIR_MISMATCH:{field}")
    if baseline.get("environment") != "pour_water":
        raise PourWaterEventGateError("POUR_WATER_EVENT_ENVIRONMENT_INVALID")
    baseline_videos = _trajectory_videos(baseline)
    candidate_videos = _trajectory_videos(candidate)
    indices = sorted(set(baseline_videos) & set(candidate_videos))
    if not indices:
        raise PourWaterEventGateError("POUR_WATER_EVENT_NO_PAIRED_TRAJECTORIES")
    measurements = [
        _measure_video_pair(
            trajectory_index=index,
            baseline_video=baseline_videos[index],
            candidate_video=candidate_videos[index],
            max_gt_panel_mse=max_gt_panel_mse,
        )
        for index in indices
    ]
    classification = classify_event_measurements(
        measurements,
        min_completion_ratio=min_completion_ratio,
        min_completion_uplift=min_completion_uplift,
        max_onset_delay_frames=max_onset_delay_frames,
    )
    diagnostics = [
        diagnose_paired_rollout_video(video_path=candidate_videos[index], trajectory_index=index)
        for index in indices
    ]
    routed_failure_families = list(
        dict.fromkeys(
            signature
            for diagnostic in diagnostics
            for signature in diagnostic["routed_failure_families"]
            if isinstance(signature, str)
        )
    )
    if classification["classification"] in {"event_failure", "event_regression"}:
        routed_failure_families = list(
            dict.fromkeys(("fluid_volume_transport", "sparse_event_undercoverage", *routed_failure_families))
        )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-pour-water-event-gate",
        "state": "ready",
        "environment": "pour_water",
        "split": baseline.get("split"),
        "primitive": primitive,
        "seed": seed,
        **classification,
        "event_improvement_pass": classification["classification"] == "event_positive",
        "measurements": measurements,
        "diagnostic_routing": {
            "state": "measured",
            "routed_failure_families": routed_failure_families,
            "video_diagnostics": diagnostics,
            "verdict_exposure_allowed": False,
        },
        "source": {
            "baseline_manifest": str(baseline_path),
            "baseline_manifest_sha256": _sha256(baseline_path),
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": _sha256(candidate_path),
        },
        "claim_boundary": (
            "This deterministic water-mask gate is diagnostic event evidence. It can prevent a pixel-metric "
            "gain from being treated as a solved pouring event, but it does not replace the frozen official gate."
        ),
    }
    return _write_bundle(destination, report)


def _measure_video_pair(
    *,
    trajectory_index: int,
    baseline_video: Path,
    candidate_video: Path,
    max_gt_panel_mse: float,
) -> dict[str, object]:
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        raise PourWaterEventGateError("POUR_WATER_EVENT_MEDIA_DEPENDENCY_MISSING") from exc
    baseline_frames = np.stack(imageio.mimread(baseline_video))[:, :, :, :3]
    candidate_frames = np.stack(imageio.mimread(candidate_video))[:, :, :, :3]
    frame_count = min(len(baseline_frames), len(candidate_frames))
    if frame_count < 32:
        raise PourWaterEventGateError("POUR_WATER_EVENT_FRAME_COUNT_INSUFFICIENT")
    baseline_frames = baseline_frames[:frame_count]
    candidate_frames = candidate_frames[:frame_count]
    width = min(baseline_frames.shape[2], candidate_frames.shape[2])
    half = width // 2
    if half < 1:
        raise PourWaterEventGateError("POUR_WATER_EVENT_FRAME_SHAPE_INVALID")
    baseline_gt = baseline_frames[:, :, :half]
    candidate_gt = candidate_frames[:, :, :half]
    gt_mse = float(np.square((baseline_gt.astype(np.float32) - candidate_gt.astype(np.float32)) / 255.0).mean())
    if gt_mse > max_gt_panel_mse:
        raise PourWaterEventGateError("POUR_WATER_EVENT_GT_ALIGNMENT_FAILED")
    gt_masks = _water_masks(baseline_gt, np)
    baseline_masks = _water_masks(baseline_frames[:, :, half : half * 2], np)
    candidate_masks = _water_masks(candidate_frames[:, :, half : half * 2], np)
    initial_window = max(1, min(10, frame_count // 8))
    late_window = max(5, frame_count // 5)
    initial_occupancy = gt_masks[:initial_window].mean(axis=0)
    late_occupancy = gt_masks[-late_window:].mean(axis=0)
    target_mask = (late_occupancy >= 0.15) & (initial_occupancy < 0.10)
    target_pixels = int(target_mask.sum())
    if target_pixels < 100:
        raise PourWaterEventGateError("POUR_WATER_EVENT_TARGET_SUPPORT_INSUFFICIENT")
    gt_curve = (gt_masks & target_mask).sum(axis=(1, 2)).astype(float)
    baseline_curve = (baseline_masks & target_mask).sum(axis=(1, 2)).astype(float)
    candidate_curve = (candidate_masks & target_mask).sum(axis=(1, 2)).astype(float)
    gt_late_mean = float(gt_curve[-late_window:].mean())
    if gt_late_mean <= 0.0:
        raise PourWaterEventGateError("POUR_WATER_EVENT_GT_COMPLETION_MISSING")
    onset_threshold = max(20.0, 0.10 * gt_late_mean)
    gt_onset = _onset_frame(gt_curve, onset_threshold, frame_count)
    baseline_onset = _onset_frame(baseline_curve, onset_threshold, frame_count)
    candidate_onset = _onset_frame(candidate_curve, onset_threshold, frame_count)
    baseline_completion = float(baseline_curve[-late_window:].mean() / gt_late_mean)
    candidate_completion = float(candidate_curve[-late_window:].mean() / gt_late_mean)
    return {
        "trajectory_index": trajectory_index,
        "frame_count": frame_count,
        "target_support_pixels": target_pixels,
        "gt_panel_mse_between_sources": gt_mse,
        "gt_late_target_area_mean": gt_late_mean,
        "baseline_late_target_area_mean": float(baseline_curve[-late_window:].mean()),
        "candidate_late_target_area_mean": float(candidate_curve[-late_window:].mean()),
        "baseline_completion_ratio": baseline_completion,
        "candidate_completion_ratio": candidate_completion,
        "completion_uplift": candidate_completion - baseline_completion,
        "gt_event_onset_frame": gt_onset,
        "baseline_event_onset_frame": None if baseline_onset == frame_count else baseline_onset,
        "candidate_event_onset_frame": None if candidate_onset == frame_count else candidate_onset,
        "baseline_onset_delay_frames": max(0, baseline_onset - gt_onset),
        "candidate_onset_delay_frames": max(0, candidate_onset - gt_onset),
        "baseline_video": str(baseline_video),
        "candidate_video": str(candidate_video),
    }


def _water_masks(frames: Any, np: Any) -> Any:
    values = frames.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    return (blue > 100) & (blue - red > 25) & (green - red > 10) & (blue >= green - 5)


def _onset_frame(curve: Any, threshold: float, fallback: int) -> int:
    for index, value in enumerate(curve):
        if float(value) >= threshold:
            return index
    return fallback


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PourWaterEventGateError("POUR_WATER_EVENT_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("state") != "ready":
        raise PourWaterEventGateError("POUR_WATER_EVENT_MANIFEST_NOT_READY")
    if payload.get("artifact_type") != "wmloop-horizon-probe-run":
        raise PourWaterEventGateError("POUR_WATER_EVENT_MANIFEST_TYPE_INVALID")
    return payload


def _trajectory_videos(manifest: Mapping[str, Any]) -> dict[int, Path]:
    rows = manifest.get("trajectory_results")
    if not isinstance(rows, list):
        return {}
    result: dict[int, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index, value = row.get("trajectory_index"), row.get("rollout_video_path")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(value, str):
            path = Path(value).resolve()
            if path.is_file() and not path.is_symlink():
                result[index] = path
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PourWaterEventGateError(f"POUR_WATER_EVENT_VALUE_INVALID:{field}")
    return float(value)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PourWaterEventGateError(f"POUR_WATER_EVENT_VALUE_INVALID:{field}")
    return value


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        report_path = temporary / "event-gate.json"
        report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-pour-water-event-gate-manifest",
            "state": "ready",
            "environment": "pour_water",
            "classification": report["classification"],
            "candidate_event_pass": report["candidate_event_pass"],
            "event_improvement_pass": report["event_improvement_pass"],
            "primitive": report["primitive"],
            "seed": report["seed"],
            "report_path": str(destination / "event-gate.json"),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return {**dict(report), "manifest": manifest}
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-completion-ratio", type=float, default=0.25)
    parser.add_argument("--min-completion-uplift", type=float, default=0.05)
    parser.add_argument("--max-onset-delay-frames", type=int, default=30)
    parser.add_argument("--primitive", default="")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_pour_water_event_gate(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        output_root=args.output_root,
        min_completion_ratio=args.min_completion_ratio,
        min_completion_uplift=args.min_completion_uplift,
        max_onset_delay_frames=args.max_onset_delay_frames,
        primitive=args.primitive,
        seed=args.seed,
    )
    print(json.dumps(report["manifest"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
