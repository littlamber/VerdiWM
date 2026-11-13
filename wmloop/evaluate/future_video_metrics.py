"""CPU replay of the supported frozen future-frame diagnostic recipe.

Recipe ``wan-future-rgb-v1`` uses frames 1..12 of 13 RGB uint8 frames at
512x768. PSNR uses aggregate MSE; SSIM is global RGB, not windowed SSIM.
Decoding runs in a bounded subprocess and never executes archived Python.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock, sha256_bytes
from wmloop.process_lifecycle import LocalCommandSignals, retire_process_group
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate import future_video_recipe
from wmloop.evaluate.future_video_recipe import (
    METRIC_DIRECTIONS, RECIPE_ID, FutureVideoMetricError, future_frame_metrics, global_rgb_ssim,
)
# These identities describe the already frozen algorithm, not a model, data
# location, evaluator label, target threshold or effect authority.
FROZEN_ALGORITHM_SHA256 = {
    "evaluate_future_only.py": "a51130f52e74a634602ce3918afcc8086a1c6bc629882b1c95da009ace950d58",
    "base_frame_math.py": "a0d05006fb76cfacdf59da306a25d904626b831f6868ea7256709f54eff55e00",
}
_METRICS = {
    "psnr_db": {"direction": "higher", "definition": "10*log10(255^2 / mean(all future frame MSE)); zero MSE is nonfinite"},
    "global_ssim_rgb": {"direction": "higher", "definition": "global RGB population-moment SSIM averaged over future frames"},
    "aggregate_mse": {"direction": "lower", "definition": "mean full RGB future frame MSE"},
}


def supported_frozen_algorithm(files: Mapping[str, str], freeze: Mapping[str, object]) -> bool:
    """Accept the historical freeze or the stable public numerical recipe."""
    public = {"future_video_recipe.py": sha256_bytes(Path(future_video_recipe.__file__).read_bytes())}
    return (dict(files) == FROZEN_ALGORITHM_SHA256
            or (dict(files) == public and freeze.get("numeric_recipe") == RECIPE_ID))


def freeze_future_video_protocol(*, protocol: Mapping[str, object], output_root: Path) -> Path:
    """Freeze a user's declared comparison with the installed numeric recipe.

    Call before candidate generation. The output is a reviewable bundle;
    measurements and candidate runs use it unchanged.
    """
    try:
        document = json.loads(canonical_bytes(protocol))
    except (ValueError, TypeError, OverflowError) as exc:
        raise FutureVideoMetricError("FUTURE_VIDEO_PROTOCOL_INVALID") from exc
    if (not isinstance(document, dict) or not isinstance(document.get("evaluator_id"), str)
        or not document["evaluator_id"] or not isinstance(document.get("comparison"), dict)
        or document.get("formal_acceptance", False) is not False
        or document.get("ranking_primary_metric") not in METRIC_DIRECTIONS
        or type(document.get("metric_tie_epsilon")) not in (int, float)
        or document["metric_tie_epsilon"] < 0):
        raise FutureVideoMetricError("FUTURE_VIDEO_PROTOCOL_INVALID")
    document.update(
        formal_acceptance=False, numeric_recipe=RECIPE_ID, metrics=_METRICS,
        crop="exclude video_frame_index=0; retain indices 1 through 12 inclusive",
        decode="imageio.v2 / ffmpeg RGB uint8; exactly 13 frames of shape 512x768x3",
        claim_boundary="Matched single-context diagnostic ranking evidence only; no formal effect, promotion or transfer authority.",
    )
    try:
        validate_document("diagnostic_paired_protocol", document)
    except ContractValidationError as exc:
        raise FutureVideoMetricError("FUTURE_VIDEO_PROTOCOL_INVALID") from exc
    payloads = {
        "protocol.json": canonical_bytes(document) + b"\n",
        "future_video_recipe.py": Path(future_video_recipe.__file__).read_bytes(),
    }
    freeze = {
        "schema_version": 1, "artifact_type": "verdiwm-frozen-future-video-recipe",
        "identity": document["evaluator_id"], "numeric_recipe": RECIPE_ID, "diagnostic_only": True,
        "files": {name: sha256_bytes(payload) for name, payload in payloads.items()},
        "claim_boundary": document["claim_boundary"],
    }
    validate_document("frozen_future_video_recipe", freeze)
    output = checked_path(output_root, code="FUTURE_VIDEO_OUTPUT_INVALID", error=FutureVideoMetricError)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(output.parent / f".{output.name}.lock"):
        if output.exists():
            raise FutureVideoMetricError("FUTURE_VIDEO_OUTPUT_EXISTS")
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
            staging = Path(temporary)
            for name, payload in payloads.items():
                atomic_write(staging / name, payload)
            atomic_write(staging / "freeze-record.json", canonical_bytes(freeze) + b"\n")
            os.replace(staging, output)
    return output


def measure_frozen_future_video(
    *, frozen_root: Path, prediction: Path, ground_truth: Path, output: Path,
) -> dict[str, object]:
    """Measure new video bytes against an unchanged supported frozen bundle."""
    root = checked_path(frozen_root, code="FUTURE_VIDEO_FREEZE_INVALID", error=FutureVideoMetricError)
    freeze_path = checked_path(root / "freeze-record.json", code="FUTURE_VIDEO_FREEZE_INVALID", error=FutureVideoMetricError)
    freeze_bytes = freeze_path.read_bytes()
    freeze = _load_object(freeze_bytes, "FUTURE_VIDEO_FREEZE_INVALID")
    if not isinstance(freeze.get("files"), dict) or "protocol.json" not in freeze["files"]:
        raise FutureVideoMetricError("FUTURE_VIDEO_FREEZE_INVALID")
    if freeze.get("artifact_type") == "verdiwm-frozen-future-video-recipe":
        try:
            validate_document("frozen_future_video_recipe", freeze)
        except ContractValidationError as exc:
            raise FutureVideoMetricError("FUTURE_VIDEO_FREEZE_INVALID") from exc
    files = freeze["files"]
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise FutureVideoMetricError("FUTURE_VIDEO_FREEZE_INVALID")
        source = checked_path(root / name, code="FUTURE_VIDEO_FREEZE_INVALID", error=FutureVideoMetricError)
        if sha256_bytes(source.read_bytes()) != digest:
            raise FutureVideoMetricError("FUTURE_VIDEO_FREEZE_DRIFT")
    protocol = _load_object((root / "protocol.json").read_bytes(), "FUTURE_VIDEO_PROTOCOL_INVALID")
    algorithm = {name: digest for name, digest in files.items() if name != "protocol.json"}
    if (not supported_frozen_algorithm(algorithm, freeze) or freeze.get("diagnostic_only") is not True
        or freeze.get("identity") != protocol.get("evaluator_id") or protocol.get("formal_acceptance") is not False):
        raise FutureVideoMetricError("FUTURE_VIDEO_FREEZE_INVALID")
    try:
        validate_document("diagnostic_paired_protocol", protocol)
    except ContractValidationError as exc:
        raise FutureVideoMetricError("FUTURE_VIDEO_PROTOCOL_INVALID") from exc
    videos = {}
    for label, path in (("prediction", prediction), ("ground_truth", ground_truth)):
        source = checked_path(path, code="FUTURE_VIDEO_INPUT_INVALID", error=FutureVideoMetricError)
        videos[label] = source.read_bytes()
    measurement = {
        "schema_version": 1, "artifact_type": "verdiwm-future-frame-measurement",
        "evaluator_id": protocol["evaluator_id"], "freeze_record_sha256": sha256_bytes(freeze_bytes),
        "input_videos_sha256": {label: sha256_bytes(payload) for label, payload in videos.items()},
        **recompute_future_video_metrics(prediction=videos["prediction"], ground_truth=videos["ground_truth"]),
        "diagnostic_only": True, "promotion_allowed": False,
        "claim_boundary": protocol.get("claim_boundary", "Diagnostic ranking only."),
    }
    validate_document("future_frame_measurement", measurement)
    destination = checked_path(output, code="FUTURE_VIDEO_OUTPUT_INVALID", error=FutureVideoMetricError)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(destination.parent / f".{destination.name}.lock"):
        if destination.exists():
            raise FutureVideoMetricError("FUTURE_VIDEO_OUTPUT_EXISTS")
        atomic_write(destination, canonical_bytes(measurement) + b"\n")
    return measurement


def _load_object(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("object required")
        canonical_bytes(value)
        return value
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise FutureVideoMetricError(code) from exc


def _decode(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path, format="ffmpeg")
    frames = []
    try:
        if tuple(reader.get_meta_data().get("size", ())) != (768, 512):
            raise FutureVideoMetricError("FUTURE_VIDEO_FRAME_CONTRACT_INVALID")
        for frame in reader:
            if len(frames) == 13 or frame.shape != (512, 768, 3) or frame.dtype != np.uint8:
                raise FutureVideoMetricError("FUTURE_VIDEO_FRAME_CONTRACT_INVALID")
            frames.append(frame)
    finally:
        reader.close()
    if len(frames) != 13:
        raise FutureVideoMetricError("FUTURE_VIDEO_FRAME_CONTRACT_INVALID")
    return np.asarray(frames)


def recompute_future_video_metrics(*, prediction: bytes, ground_truth: bytes) -> dict[str, object]:
    """Decode archived bytes with a 45-second deadline and bounded input size."""
    if any(not payload or len(payload) > 64 * 1024 * 1024 for payload in (prediction, ground_truth)):
        raise FutureVideoMetricError("FUTURE_VIDEO_BYTES_INVALID")
    with tempfile.TemporaryDirectory(prefix="verdiwm-future-video-") as temporary:
        root = Path(temporary)
        (root / "prediction.mp4").write_bytes(prediction)
        (root / "ground-truth.mp4").write_bytes(ground_truth)
        try:
            scope = LocalCommandSignals() if threading.current_thread() is threading.main_thread() else nullcontext(None)
            with scope as signals:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-m", "wmloop.evaluate.future_video_metrics", "--worker-root", str(root)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 45.0
                try:
                    while True:
                        if signals is not None:
                            signals.check()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(process.args, 45.0)
                        try:
                            stdout, _stderr = process.communicate(timeout=min(0.2, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    retire_process_group(process)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FutureVideoMetricError("FUTURE_VIDEO_REPLAY_FAILED") from exc
        if process.returncode != 0:
            raise FutureVideoMetricError("FUTURE_VIDEO_REPLAY_FAILED")
        try:
            result = json.loads(stdout)
            canonical_bytes(result)
            if not isinstance(result, dict) or set(result) != {"metrics", "per_frame"}:
                raise ValueError("invalid numeric replay")
            return result
        except (ValueError, TypeError) as exc:
            raise FutureVideoMetricError("FUTURE_VIDEO_REPLAY_FAILED") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")
    freeze_parser = subparsers.add_parser("freeze", help="Freeze a declared comparison before generation")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    measure_parser = subparsers.add_parser("measure", help="Measure videos with a frozen supported recipe")
    measure_parser.add_argument("--frozen-root", type=Path, required=True)
    measure_parser.add_argument("--prediction", type=Path, required=True)
    measure_parser.add_argument("--ground-truth", type=Path, required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        protocol_path = checked_path(args.protocol, code="FUTURE_VIDEO_PROTOCOL_INVALID", error=FutureVideoMetricError)
        freeze_future_video_protocol(protocol=_load_object(protocol_path.read_bytes(), "FUTURE_VIDEO_PROTOCOL_INVALID"), output_root=args.output_root)
        return 0
    if args.command == "measure":
        measure_frozen_future_video(frozen_root=args.frozen_root, prediction=args.prediction, ground_truth=args.ground_truth, output=args.output)
        return 0
    if args.worker_root is None:
        parser.error("a freeze or measure command is required")
    result = future_frame_metrics(_decode(args.worker_root / "prediction.mp4"), _decode(args.worker_root / "ground-truth.mp4"))
    # A private worker's stdout is a JSON protocol, never a progress log.
    sys.stdout.buffer.write(canonical_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
