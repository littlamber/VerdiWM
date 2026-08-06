"""Exclusive, observable GPU leases for bounded local experiments.

The lease coordinates VerdiWM workers on one host. It is deliberately not a
replacement for a cluster scheduler: the lease is acquired only after a
fresh ``nvidia-smi`` snapshot shows that the requested physical GPU is below
the configured admission thresholds and has no compute application.
"""

from __future__ import annotations

import csv
import datetime as _dt
import fcntl
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


class GpuLeaseError(RuntimeError):
    """A GPU lease could not be acquired or used safely."""


SnapshotProvider = Callable[[], Mapping[str, object]]


@dataclass(frozen=True)
class GpuLease:
    """A held host-local lease for one physical GPU."""

    index: int
    uuid: str
    name: str
    lock_path: Path
    _handle: object

    def environment(self) -> dict[str, str]:
        """Return the environment values that bind a child to this GPU."""

        return {
            "CUDA_VISIBLE_DEVICES": str(self.index),
            "VERDIWM_PHYSICAL_GPU_INDEX": str(self.index),
            "VERDIWM_PHYSICAL_GPU_UUID": self.uuid,
        }

    def to_document(self) -> dict[str, object]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "lock_path": str(self.lock_path),
        }

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            object.__setattr__(self, "_handle", None)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class GpuLeaseManager:
    """Acquire a fresh lease from a deterministic set of GPU candidates."""

    def __init__(
        self,
        *,
        lock_root: Path = Path("/tmp/verdiwm-gpu-leases"),
        memory_used_threshold_mib: int = 1024,
        utilization_threshold_percent: int = 10,
        snapshot_provider: SnapshotProvider | None = None,
    ) -> None:
        if memory_used_threshold_mib < 0:
            raise ValueError("GPU_LEASE_MEMORY_THRESHOLD_INVALID")
        if utilization_threshold_percent < 0 or utilization_threshold_percent > 100:
            raise ValueError("GPU_LEASE_UTILIZATION_THRESHOLD_INVALID")
        self._lock_root = Path(lock_root).resolve()
        self._memory_threshold = memory_used_threshold_mib
        self._utilization_threshold = utilization_threshold_percent
        self._snapshot_provider = snapshot_provider or _query_snapshot

    def acquire(
        self,
        allowed_indices: Sequence[int],
        *,
        wait_seconds: float = 0.0,
        poll_seconds: float = 2.0,
    ) -> GpuLease:
        candidates = _normalize_indices(allowed_indices)
        if not candidates:
            raise GpuLeaseError("GPU_LEASE_CANDIDATES_EMPTY")
        if wait_seconds < 0 or poll_seconds <= 0:
            raise GpuLeaseError("GPU_LEASE_WAIT_POLICY_INVALID")
        deadline = time.monotonic() + wait_seconds
        last_blockers: list[str] = []
        while True:
            lease, blockers = self._try_acquire(candidates)
            if lease is not None:
                return lease
            last_blockers = blockers
            if time.monotonic() >= deadline:
                suffix = ":" + ",".join(last_blockers[:8]) if last_blockers else ""
                raise GpuLeaseError(f"GPU_LEASE_UNAVAILABLE{suffix}")
            time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))

    def _try_acquire(self, candidates: Sequence[int]) -> tuple[GpuLease | None, list[str]]:
        self._lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        snapshot = self._snapshot_provider()
        by_index = {
            int(item["index"]): item
            for item in _mapping_sequence(snapshot.get("gpus"))
            if _is_int(item.get("index"))
        }
        apps_by_uuid: dict[str, list[Mapping[str, object]]] = {}
        for app in _mapping_sequence(snapshot.get("compute_apps")):
            apps_by_uuid.setdefault(str(app.get("gpu_uuid") or ""), []).append(app)
        blockers: list[str] = []
        for index in candidates:
            lock_path = self._lock_root / f"gpu-{index}.lock"
            handle = lock_path.open("a+", encoding="ascii")
            os.chmod(lock_path, 0o600)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                blockers.append(f"{index}:LEASE_HELD")
                continue
            gpu = by_index.get(index)
            if gpu is None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                blockers.append(f"{index}:GPU_NOT_FOUND")
                continue
            uuid = str(gpu.get("uuid") or "")
            memory = _number(gpu.get("memory_used_mib"))
            utilization = _number(gpu.get("utilization_gpu_percent"))
            apps = apps_by_uuid.get(uuid, [])
            if memory is None or utilization is None or not uuid:
                blockers.append(f"{index}:GPU_SNAPSHOT_INVALID")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                continue
            if memory > self._memory_threshold:
                blockers.append(f"{index}:MEMORY_BUSY:{memory}")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                continue
            if utilization > self._utilization_threshold:
                blockers.append(f"{index}:UTILIZATION_BUSY:{utilization}")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                continue
            if apps:
                blockers.append(f"{index}:COMPUTE_APP_PRESENT")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                continue
            return GpuLease(
                index=index,
                uuid=uuid,
                name=str(gpu.get("name") or "unknown"),
                lock_path=lock_path,
                _handle=handle,
            ), blockers
        return None, blockers


def _query_snapshot() -> dict[str, object]:
    gpu_result = _run_nvidia_smi(
        (
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    apps_result = _run_nvidia_smi(
        (
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    gpus: list[dict[str, object]] = []
    if gpu_result["returncode"] == 0:
        for row in csv.reader(str(gpu_result["stdout"]).splitlines()):
            fields = [field.strip() for field in row]
            if len(fields) != 6:
                continue
            index = _parse_int(fields[0])
            if index is None:
                continue
            gpus.append(
                {
                    "index": index,
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_used_mib": _parse_int(fields[3]),
                    "memory_total_mib": _parse_int(fields[4]),
                    "utilization_gpu_percent": _parse_int(fields[5]),
                }
            )
    apps: list[dict[str, object]] = []
    if apps_result["returncode"] == 0:
        for row in csv.reader(str(apps_result["stdout"]).splitlines()):
            fields = [field.strip() for field in row]
            if len(fields) != 4:
                continue
            apps.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": _parse_int(fields[1]),
                    "process_name": fields[2],
                    "used_memory_mib": _parse_int(fields[3]),
                }
            )
    return {
        "timestamp_utc": _utc_now(),
        "gpus": sorted(gpus, key=lambda item: int(item["index"])),
        "compute_apps": apps,
        "gpu_query_returncode": gpu_result["returncode"],
        "apps_query_returncode": apps_result["returncode"],
    }


def _run_nvidia_smi(arguments: Sequence[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ("nvidia-smi", *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}:{exc}"}
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_indices(values: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GpuLeaseError("GPU_LEASE_INDEX_INVALID")
        if value not in result:
            result.append(value)
    return tuple(sorted(result))


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
