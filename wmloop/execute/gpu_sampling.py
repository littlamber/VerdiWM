"""Lightweight GPU sampling for execution receipts."""

from __future__ import annotations

import datetime as _dt
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any, Mapping, TypeVar


T = TypeVar("T")


class GpuSamplingRecorder:
    """Collect an auditable nvidia-smi curve around blocking commands."""

    def __init__(
        self,
        *,
        gpu_index: int,
        sample_interval_seconds: float = 5.0,
        sample_provider: Callable[[int], Mapping[str, object]] | None = None,
    ) -> None:
        if gpu_index < 0:
            raise ValueError("GPU_SAMPLING_GPU_INDEX_INVALID")
        if sample_interval_seconds <= 0:
            raise ValueError("GPU_SAMPLING_INTERVAL_INVALID")
        self._gpu_index = gpu_index
        self._interval = sample_interval_seconds
        self._sample_provider = sample_provider or _nvidia_smi_sample
        self._started = time.monotonic()
        self._samples: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def capture(self, label: str, callback: Callable[[], T]) -> T:
        """Run ``callback`` while sampling the configured GPU."""

        self.sample(label=label, phase="start")
        stop = threading.Event()
        thread = threading.Thread(target=self._sample_until_stopped, args=(label, stop), daemon=True)
        thread.start()
        try:
            return callback()
        finally:
            stop.set()
            thread.join(timeout=max(self._interval * 2, 1.0))
            self.sample(label=label, phase="end")

    def sample(self, *, label: str, phase: str) -> None:
        payload = dict(self._sample_provider(self._gpu_index))
        sample = {
            "ordinal": 0,
            "label": label,
            "phase": phase,
            "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - self._started,
            "gpu_index": self._gpu_index,
            **payload,
        }
        with self._lock:
            sample["ordinal"] = len(self._samples) + 1
            self._samples.append(sample)

    def to_document(self) -> dict[str, object]:
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
        labels = []
        for sample in samples:
            label = str(sample.get("label"))
            if label not in labels:
                labels.append(label)
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-gpu-sampling-curve",
            "state": "ready" if samples else "empty",
            "gpu_index": self._gpu_index,
            "sample_interval_seconds": self._interval,
            "sample_count": len(samples),
            "available_sample_count": sum(1 for sample in samples if sample.get("status") == "ready"),
            "command_labels": labels,
            "samples": samples,
        }

    def _sample_until_stopped(self, label: str, stop: threading.Event) -> None:
        while not stop.wait(self._interval):
            self.sample(label=label, phase="during")


def _nvidia_smi_sample(gpu_index: int) -> Mapping[str, object]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "error": exc.__class__.__name__}
    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "error": completed.stderr.strip()[:200] or f"nvidia-smi exited {completed.returncode}",
        }
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        if index != gpu_index:
            continue
        return {
            "status": "ready",
            "memory_used_mib": _parse_int(fields[1]),
            "utilization_gpu_percent": _parse_int(fields[2]),
            "temperature_gpu_c": _parse_int(fields[3]),
        }
    return {"status": "unavailable", "error": f"gpu index {gpu_index} not found"}


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
