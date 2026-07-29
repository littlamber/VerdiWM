"""Validate and summarize a Cosmos3 forward-dynamics GPU smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


class Cosmos3GpuRuntimeReceiptError(ValueError):
    """The runtime evidence is incomplete or internally inconsistent."""


def build_cosmos3_gpu_runtime_receipt(
    *,
    run_root: Path,
    gpu_samples_path: Path,
    output_root: Path,
    expected_gpu_uuid: str | None = None,
) -> dict[str, object]:
    """Fail closed unless the official run and physical-GPU evidence agree."""

    run = Path(run_root).resolve(strict=True)
    samples_path = Path(gpu_samples_path).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_OUTPUT_EXISTS")

    sample_outputs_path = run / "robotics_action_cond_chunk_00/sample_outputs.json"
    video_path = run / "robotics_action_cond_chunk_00/vision.mp4"
    action_path = run / "inputs/robotics_droid_action_chunk_00.json"
    benchmark_path = run / "benchmark.json"
    for path in (sample_outputs_path, video_path, action_path, benchmark_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise Cosmos3GpuRuntimeReceiptError(f"COSMOS3_GPU_RECEIPT_ARTIFACT_MISSING:{path.name}")

    sample_outputs = _load_mapping(sample_outputs_path)
    if sample_outputs.get("status") != "success":
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_INFERENCE_NOT_SUCCESSFUL")
    args = sample_outputs.get("args")
    if not isinstance(args, Mapping) or args.get("model_mode") != "forward_dynamics":
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_WRONG_MODEL_MODE")

    actions = json.loads(action_path.read_text(encoding="utf-8"))
    if not isinstance(actions, list) or not actions or not all(isinstance(row, list) for row in actions):
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_ACTIONS_INVALID")
    widths = {len(row) for row in actions}
    if len(widths) != 1:
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_ACTIONS_RAGGED")

    gpu_samples = _load_gpu_samples(samples_path)
    active = [row for row in gpu_samples if int(row.get("used_memory_mib", 0)) > 0]
    if not active:
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_NO_ACTIVE_GPU_SAMPLE")
    uuids = sorted({str(row["gpu_uuid"]) for row in active})
    if expected_gpu_uuid is not None and expected_gpu_uuid not in uuids:
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_GPU_UUID_MISMATCH")

    video = _probe_video(video_path)
    if int(video["frame_count"]) < 2 or float(video["duration_seconds"]) <= 0:
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_VIDEO_INVALID")
    benchmark = _load_mapping(benchmark_path)
    average = benchmark.get("average")
    if not isinstance(average, Mapping):
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_BENCHMARK_INVALID")

    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-forward-dynamics-gpu-runtime-receipt",
        "state": "ready",
        "model_family": "cosmos3",
        "model_mode": "forward_dynamics",
        "inference_status": "success",
        "identity": {"sample_index": 0, "seed": int(args.get("seed", 0))},
        "action_shape": [len(actions), next(iter(widths))],
        "physical_gpu": {
            "observed_gpu_uuids": uuids,
            "sample_count": len(gpu_samples),
            "active_sample_count": len(active),
            "peak_used_memory_mib": max(int(row["used_memory_mib"]) for row in active),
        },
        "video": video,
        "runtime_seconds": float(average["OmniInference.generate_batch"]),
        "sha256": {
            "action_input": _sha256(action_path),
            "sample_outputs": _sha256(sample_outputs_path),
            "vision_video": _sha256(video_path),
        },
        "claim_boundary": (
            "This receipt proves one official Cosmos3-Nano ACWM forward-dynamics GPU execution. "
            "It does not establish predictive quality, primitive benefit, or cross-backbone transfer."
        ),
    }
    _write_bundle(destination=destination, receipt=receipt)
    return receipt


def _load_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3GpuRuntimeReceiptError(f"COSMOS3_GPU_RECEIPT_JSON_INVALID:{path.name}")
    return payload


def _load_gpu_samples(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or not row.get("gpu_uuid") or "used_memory_mib" not in row:
            raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_GPU_SAMPLE_INVALID")
        rows.append(row)
    return rows


def _probe_video(path: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration,size",
                "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Cosmos3GpuRuntimeReceiptError("COSMOS3_GPU_RECEIPT_FFPROBE_FAILED") from exc
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    fmt = payload["format"]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_seconds": float(fmt["duration"]),
        "size_bytes": int(fmt["size"]),
    }


def _write_bundle(*, destination: Path, receipt: Mapping[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        receipt_path = temporary / "runtime-receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (temporary / "MANIFEST.sha256").write_text(
            f"{_sha256(receipt_path)}  runtime-receipt.json\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid")
    args = parser.parse_args(argv)
    receipt = build_cosmos3_gpu_runtime_receipt(
        run_root=args.run_root, gpu_samples_path=args.gpu_samples, output_root=args.output_root,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
