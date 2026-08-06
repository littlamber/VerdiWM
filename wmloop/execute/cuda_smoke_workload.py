"""Small real-CUDA workload used to prove the auto-experiment runtime."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA_SMOKE_TORCH_REQUIRED") from exc

    scratch_value = os.environ.get("VERDIWM_TRIAL_SCRATCH")
    if not scratch_value:
        raise RuntimeError("CUDA_SMOKE_SCRATCH_REQUIRED")
    scratch = Path(scratch_value).resolve()
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_SMOKE_CUDA_UNAVAILABLE")

    matrix_size = _positive_int("VERDIWM_CUDA_SMOKE_MATRIX_SIZE", default=4096)
    minimum_iterations = _positive_int("VERDIWM_CUDA_SMOKE_ITERATIONS", default=8)
    minimum_seconds = _positive_float("VERDIWM_CUDA_SMOKE_MIN_SECONDS", default=2.0)
    torch.manual_seed(20260806)
    torch.cuda.manual_seed_all(20260806)
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda:0")
    left = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    right = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    started = time.monotonic()
    iterations = 0
    product = left @ right
    while iterations < minimum_iterations or time.monotonic() - started < minimum_seconds:
        product = left @ right
        iterations += 1
    torch.cuda.synchronize()
    elapsed_seconds = time.monotonic() - started

    observed = product[:16, :16].double().cpu()
    reference = left[:16].double().cpu() @ right[:, :16].double().cpu()
    max_abs_error = float((observed - reference).abs().max().item())
    finite_fraction = float(torch.isfinite(product).float().mean().item())
    checksum = float(product[:16, :16].double().sum().item())
    physical_index = _required_int("VERDIWM_PHYSICAL_GPU_INDEX")
    expected_uuid = os.environ.get("VERDIWM_PHYSICAL_GPU_UUID", "")
    observed_gpu = _physical_gpu_identity(physical_index)
    if expected_uuid and observed_gpu["gpu_uuid"] != expected_uuid:
        raise RuntimeError("CUDA_SMOKE_GPU_UUID_MISMATCH")
    properties = torch.cuda.get_device_properties(device)

    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-result",
        "state": "ready" if math.isfinite(checksum) else "invalid",
        "device": {
            "type": "cuda",
            "logical_index": 0,
            "physical_index": physical_index,
            "gpu_uuid": observed_gpu["gpu_uuid"],
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "torch_cuda_version": torch.version.cuda,
        },
        "metrics": {
            "finite_fraction": finite_fraction,
            "max_abs_error": max_abs_error,
            "elapsed_seconds": elapsed_seconds,
            "iterations": float(iterations),
            "checksum": checksum,
        },
        "workload": {
            "matrix_size": matrix_size,
            "dtype": "float32",
            "minimum_seconds": minimum_seconds,
        },
    }
    _write_json_atomic(scratch / "result.json", result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


def _physical_gpu_identity(index: int) -> dict[str, str]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--id",
            str(index),
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",", 1)]
    if completed.returncode != 0 or len(fields) != 2 or not fields[0]:
        raise RuntimeError("CUDA_SMOKE_GPU_IDENTITY_UNAVAILABLE")
    return {"gpu_uuid": fields[0], "name": fields[1]}


def _positive_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"CUDA_SMOKE_INTEGER_INVALID:{name}") from exc
    if value <= 0:
        raise RuntimeError(f"CUDA_SMOKE_INTEGER_INVALID:{name}")
    return value


def _required_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"CUDA_SMOKE_INTEGER_MISSING:{name}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"CUDA_SMOKE_INTEGER_INVALID:{name}") from exc
    if value < 0:
        raise RuntimeError(f"CUDA_SMOKE_INTEGER_INVALID:{name}")
    return value


def _positive_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"CUDA_SMOKE_FLOAT_INVALID:{name}") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"CUDA_SMOKE_FLOAT_INVALID:{name}")
    return value


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
