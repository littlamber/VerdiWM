"""Run a declared workload and derive bounded diagnostic video signatures.

This module is intentionally model agnostic.  The external command owns model
loading and rollout generation; VerdiWM only measures the declared video
layout, emits a typed diagnostic result, and returns a non-zero status when
the workload or its evidence is invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import imageio.v2 as imageio
import numpy as np


class ExternalVideoProbeError(RuntimeError):
    """A declared external video probe cannot produce valid evidence."""


def run_external_video_probe(
    *,
    command: Sequence[str],
    scratch_root: Path,
    video_pattern: str,
    output_path: str,
    model_family: str,
    runtime_capability: str,
    layout: str = "vertical",
    max_frames: int = 64,
    min_frames: int = 1,
    max_pair_l1: float = 1.0,
    min_pred_temporal_change: float = 0.0,
    max_horizon_drift_slope: float = float("inf"),
    fallback_signature: str = "no_declared_failure_detected",
) -> int:
    """Run ``command`` and write one normalized diagnostic result."""

    scratch = _safe_root(scratch_root, "EXTERNAL_VIDEO_PROBE_SCRATCH_INVALID")
    if not command or any("\x00" in token for token in command):
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_COMMAND_INVALID")
    pattern_path = Path(video_pattern)
    if not video_pattern or pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_PATTERN_INVALID")
    if layout not in {"vertical", "horizontal"}:
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_LAYOUT_INVALID")
    if max_frames < 1 or min_frames < 1 or min_frames > max_frames:
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_FRAME_LIMIT_INVALID")
    for value, code in (
        (max_pair_l1, "EXTERNAL_VIDEO_PROBE_L1_THRESHOLD_INVALID"),
        (min_pred_temporal_change, "EXTERNAL_VIDEO_PROBE_TEMPORAL_THRESHOLD_INVALID"),
        (max_horizon_drift_slope, "EXTERNAL_VIDEO_PROBE_DRIFT_THRESHOLD_INVALID"),
    ):
        if value < 0 or (not np.isfinite(value) and value != float("inf")):
            raise ExternalVideoProbeError(code)
    probe_id = os.environ.get("VERDIWM_PROBE_ID", "")
    if any(
        _IDENTIFIER.fullmatch(value) is None
        for value in (probe_id, model_family, runtime_capability, fallback_signature)
    ):
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_METADATA_INVALID")

    environment = dict(os.environ)
    completed = subprocess.run(
        tuple(str(token) for token in command),
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        check=False,
    )
    matches = _video_matches(scratch, video_pattern)
    output = _inside(scratch, output_path)
    if completed.returncode != 0 or len(matches) != 1:
        result = _invalid_result(
            probe_id=probe_id,
            model_family=model_family,
            runtime_capability=runtime_capability,
            external_exit_code=int(completed.returncode),
            video_match_count=len(matches),
        )
        _write_json(output, result)
        return int(completed.returncode) or 2

    try:
        frames = _read_video(matches[0], max_frames=max_frames)
        metrics, signatures = _diagnose_frames(
            frames,
            layout=layout,
            min_frames=min_frames,
            max_pair_l1=float(max_pair_l1),
            min_pred_temporal_change=float(min_pred_temporal_change),
            max_horizon_drift_slope=float(max_horizon_drift_slope),
            fallback_signature=fallback_signature,
        )
    except (
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        ExternalVideoProbeError,
    ) as exc:
        result = _invalid_result(
            probe_id=probe_id,
            model_family=model_family,
            runtime_capability=runtime_capability,
            external_exit_code=int(completed.returncode),
            video_match_count=len(matches),
            error=f"{type(exc).__name__}:{str(exc)[:300]}",
        )
        _write_json(output, result)
        return 2

    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-result",
        "state": "ready",
        "probe_id": probe_id,
        "model_family": model_family,
        "runtime_capability": runtime_capability,
        "failure_signatures": signatures,
        "metrics": {
            **metrics,
            "probe_ready": 1.0,
            "external_exit_code": float(completed.returncode),
        },
        "video": {
            "path": matches[0].relative_to(scratch).as_posix(),
            "layout": layout,
        },
        "claim_boundary": (
            "Video diagnostics are exploratory routing evidence; they cannot "
            "establish model-quality gains or formal verdicts."
        ),
    }
    _write_json(output, result)
    return 0


def _diagnose_frames(
    frames: np.ndarray,
    *,
    layout: str,
    min_frames: int,
    max_pair_l1: float,
    min_pred_temporal_change: float,
    max_horizon_drift_slope: float,
    fallback_signature: str,
) -> tuple[dict[str, float], list[str]]:
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[3] not in {1, 3, 4}:
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_FRAMES_INVALID")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if layout == "vertical":
        if height < 2 or height % 2:
            raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_VERTICAL_LAYOUT_INVALID")
        reference, prediction = np.split(frames[..., :3], 2, axis=1)
    else:
        if width < 2 or width % 2:
            raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_HORIZONTAL_LAYOUT_INVALID")
        reference, prediction = np.split(frames[..., :3], 2, axis=2)
    reference = reference.astype(np.float32)
    prediction = prediction.astype(np.float32)
    error = np.abs(reference - prediction).mean(axis=(1, 2, 3))
    mse = np.square(reference - prediction).mean(axis=(1, 2, 3))
    prediction_delta = np.abs(np.diff(prediction, axis=0)).mean(axis=(1, 2, 3))
    reference_delta = np.abs(np.diff(reference, axis=0)).mean(axis=(1, 2, 3))
    slope = (
        float(np.polyfit(np.arange(len(error), dtype=np.float64), error, 1)[0])
        if len(error) > 1
        else 0.0
    )
    mean_l1 = float(error.mean())
    mean_psnr = float(np.mean(10.0 * np.log10((255.0**2) / np.maximum(mse, 1e-9))))
    pred_temporal = float(prediction_delta.mean()) if len(prediction_delta) else 0.0
    ref_temporal = float(reference_delta.mean()) if len(reference_delta) else 0.0
    metrics = {
        "frame_count": float(frames.shape[0]),
        "paired_l1_mean": mean_l1,
        "paired_l1_max": float(error.max()),
        "paired_psnr_mean": mean_psnr,
        "predicted_temporal_change_mean": pred_temporal,
        "reference_temporal_change_mean": ref_temporal,
        "horizon_error_slope": slope,
    }
    signatures: list[str] = []
    if mean_l1 > max_pair_l1:
        signatures.append("paired_rollout_error_high")
    if frames.shape[0] < min_frames:
        signatures.append("short_horizon_observed")
    if pred_temporal < min_pred_temporal_change:
        signatures.append("predicted_temporal_collapse")
    if slope > max_horizon_drift_slope:
        signatures.append("horizon_drift")
    if not signatures:
        signatures.append(fallback_signature)
    return metrics, signatures


def _read_video(path: Path, *, max_frames: int) -> np.ndarray:
    reader = imageio.get_reader(path, format="ffmpeg")
    try:
        frame_count = int(reader.count_frames())
        if frame_count < 1:
            raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_VIDEO_EMPTY")
        indices = (
            np.linspace(0, frame_count - 1, max_frames, dtype=int)
            if frame_count > max_frames
            else np.arange(frame_count, dtype=int)
        )
        frames = np.asarray([reader.get_data(int(index)) for index in indices])
    finally:
        reader.close()
    if frames.ndim == 3:
        frames = frames[..., None]
    return frames


def _video_matches(scratch: Path, pattern: str) -> list[Path]:
    matches = []
    for candidate in scratch.glob(pattern):
        resolved = candidate.resolve()
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and _is_inside(scratch, resolved)
        ):
            matches.append(resolved)
    return sorted(set(matches), key=lambda path: path.relative_to(scratch).as_posix())


def _safe_root(path: Path, code: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ExternalVideoProbeError(code)
    root = source.resolve()
    if not root.is_dir():
        raise ExternalVideoProbeError(code)
    return root


def _inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_OUTPUT_PATH_INVALID")
    resolved = (root / candidate).resolve()
    if not _is_inside(root, resolved):
        raise ExternalVideoProbeError("EXTERNAL_VIDEO_PROBE_OUTPUT_PATH_ESCAPE")
    return resolved


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _invalid_result(
    *,
    probe_id: str,
    model_family: str,
    runtime_capability: str,
    external_exit_code: int,
    video_match_count: int,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-result",
        "state": "invalid",
        "probe_id": probe_id,
        "model_family": model_family,
        "runtime_capability": runtime_capability,
        "failure_signatures": [],
        "metrics": {
            "probe_ready": 0.0,
            "external_exit_code": float(external_exit_code),
            "video_match_count": float(video_match_count),
        },
        "error": error or "EXTERNAL_VIDEO_PROBE_WORKLOAD_OR_VIDEO_INVALID",
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-pattern", required=True)
    parser.add_argument("--output-path", default="diagnostic-output.json")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--runtime-capability", required=True)
    parser.add_argument(
        "--layout", choices=("vertical", "horizontal"), default="vertical"
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-pair-l1", type=float, default=1.0)
    parser.add_argument("--min-pred-temporal-change", type=float, default=0.0)
    parser.add_argument("--max-horizon-drift-slope", type=float, default=float("inf"))
    parser.add_argument("--fallback-signature", default="no_declared_failure_detected")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    scratch = os.environ.get("VERDIWM_TRIAL_SCRATCH")
    if not scratch:
        print("EXTERNAL_VIDEO_PROBE_SCRATCH_REQUIRED", file=sys.stderr)
        return 2
    try:
        return run_external_video_probe(
            command=command,
            scratch_root=Path(scratch),
            video_pattern=args.video_pattern,
            output_path=args.output_path,
            model_family=args.model_family,
            runtime_capability=args.runtime_capability,
            layout=args.layout,
            max_frames=args.max_frames,
            min_frames=args.min_frames,
            max_pair_l1=args.max_pair_l1,
            min_pred_temporal_change=args.min_pred_temporal_change,
            max_horizon_drift_slope=args.max_horizon_drift_slope,
            fallback_signature=args.fallback_signature,
        )
    except ExternalVideoProbeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
