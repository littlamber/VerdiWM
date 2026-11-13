"""Exclusive, observable GPU leases for bounded local experiments.

The lease coordinates VerdiWM workers on one host. It is deliberately not a
replacement for a cluster scheduler: the lease is acquired only after a
fresh ``nvidia-smi`` snapshot shows that the requested physical GPU is below
the configured admission thresholds and has no compute application.
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from wmloop.runtime_env import default_lock_root

if os.name == "nt":
    import msvcrt

    def _lock_exclusive_nb(handle: object) -> None:
        """Windows equivalent of flock(LOCK_EX | LOCK_NB) via msvcrt."""

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(exc.errno or 11, exc.strerror or "lease held") from exc

    def _lock_unlock(handle: object) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_exclusive_nb(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _lock_unlock(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
            _lock_unlock(handle)
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
        lock_root: Path | None = None,
        memory_used_threshold_mib: int = 1024,
        utilization_threshold_percent: int = 10,
        snapshot_provider: SnapshotProvider | None = None,
    ) -> None:
        memory_threshold = _number(memory_used_threshold_mib)
        utilization_threshold = _number(utilization_threshold_percent)
        if memory_threshold is None or memory_threshold < 0:
            raise ValueError("GPU_LEASE_MEMORY_THRESHOLD_INVALID")
        if utilization_threshold is None or not 0 <= utilization_threshold <= 100:
            raise ValueError("GPU_LEASE_UTILIZATION_THRESHOLD_INVALID")
        self._lock_root = Path(lock_root if lock_root is not None else default_lock_root()).resolve()
        self._memory_threshold = memory_threshold
        self._utilization_threshold = utilization_threshold
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
        _validate_wait_policy(wait_seconds, poll_seconds)
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

    def acquire_many(
        self,
        allowed_indices: Sequence[int],
        count: int,
        *,
        wait_seconds: float = 0.0,
        poll_seconds: float = 2.0,
    ) -> list[GpuLease]:
        """Atomically acquire ``count`` leases from one admitted GPU set.

        A distributed worker must never start with a partial world.  We therefore
        try every allowed candidate in one attempt and release every partial
        lease if fewer than ``count`` GPUs can be admitted.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise GpuLeaseError("GPU_LEASE_COUNT_INVALID")
        candidates = _normalize_indices(allowed_indices)
        if count > len(candidates):
            raise GpuLeaseError("GPU_LEASE_COUNT_EXCEEDS_CANDIDATES")
        _validate_wait_policy(wait_seconds, poll_seconds)
        deadline = time.monotonic() + wait_seconds
        last_blockers: list[str] = []
        while True:
            leases, blockers = self._try_acquire_many(candidates, count)
            if leases is not None:
                return leases
            last_blockers = blockers
            if time.monotonic() >= deadline:
                suffix = ":" + ",".join(last_blockers[:8]) if last_blockers else ""
                raise GpuLeaseError(f"GPU_LEASE_UNAVAILABLE{suffix}")
            time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))

    def _try_acquire_many(
        self, candidates: Sequence[int], count: int
    ) -> tuple[list[GpuLease] | None, list[str]]:
        """Choose a complete available subset, rolling back on any failure."""
        snapshot = self._snapshot_provider()
        by_index, apps_by_uuid = _validated_snapshot(snapshot)
        self._lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        leases: list[GpuLease] = []
        blockers: list[str] = []
        complete = False
        try:
            for index in candidates:
                lock_path = self._lock_root / f"gpu-{index}.lock"
                handle = lock_path.open("a+", encoding="ascii")
                locked = False
                retained = False
                try:
                    os.chmod(lock_path, 0o600)
                    try:
                        _lock_exclusive_nb(handle)
                    except BlockingIOError:
                        blockers.append(f"{index}:LEASE_HELD")
                        continue
                    locked = True
                    gpu = by_index.get(index)
                    reason = self._admission_blocker(gpu, apps_by_uuid)
                    if reason is not None:
                        blockers.append(f"{index}:{reason}")
                        continue
                    leases.append(GpuLease(
                        index=index, uuid=str(gpu["uuid"]),
                        name=str(gpu.get("name") or "unknown"),
                        lock_path=lock_path, _handle=handle,
                    ))
                    retained = True
                    if len(leases) == count:
                        complete = True
                        return leases, blockers
                finally:
                    if not retained:
                        try:
                            if locked:
                                _lock_unlock(handle)
                        finally:
                            handle.close()
            return None, blockers
        finally:
            if not complete:
                for lease in leases:
                    lease.release()

    def _admission_blocker(
        self, gpu: Mapping[str, object] | None,
        apps_by_uuid: Mapping[str, list[Mapping[str, object]]],
    ) -> str | None:
        if gpu is None:
            return "GPU_NOT_FOUND"
        uuid = gpu.get("uuid")
        memory = _number(gpu.get("memory_used_mib"))
        utilization = _number(gpu.get("utilization_gpu_percent"))
        if (not isinstance(uuid, str) or not uuid.strip()
            or memory is None or memory < 0
            or utilization is None or not 0 <= utilization <= 100):
            return "GPU_SNAPSHOT_INVALID"
        if memory > self._memory_threshold:
            return f"MEMORY_BUSY:{memory}"
        if utilization > self._utilization_threshold:
            return f"UTILIZATION_BUSY:{utilization}"
        if apps_by_uuid.get(uuid):
            return "COMPUTE_APP_PRESENT"
        return None

    def _try_acquire(self, candidates: Sequence[int]) -> tuple[GpuLease | None, list[str]]:
        leases, blockers = self._try_acquire_many(candidates, 1)
        return (leases[0] if leases is not None else None), blockers


def _validated_snapshot(
    snapshot: Mapping[str, object],
) -> tuple[dict[int, Mapping[str, object]], dict[str, list[Mapping[str, object]]]]:
    if not isinstance(snapshot, Mapping):
        raise GpuLeaseError("GPU_LEASE_SNAPSHOT_INVALID")
    # Explicit providers may omit query metadata; the real nvidia-smi provider
    # always supplies it. A failed query cannot prove the absence of other jobs.
    for key in ("gpu_query_returncode", "apps_query_returncode"):
        if key in snapshot and (not _is_int(snapshot[key]) or snapshot[key] != 0):
            raise GpuLeaseError(f"GPU_LEASE_QUERY_FAILED:{key}")
    gpus = _mapping_sequence(snapshot.get("gpus"))
    apps = _mapping_sequence(snapshot.get("compute_apps"))
    by_index: dict[int, Mapping[str, object]] = {}
    for gpu in gpus:
        index = gpu.get("index")
        if not _is_int(index) or index < 0 or index in by_index:
            raise GpuLeaseError("GPU_LEASE_SNAPSHOT_INVALID")
        by_index[index] = gpu
    apps_by_uuid: dict[str, list[Mapping[str, object]]] = {}
    for app in apps:
        uuid = app.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise GpuLeaseError("GPU_LEASE_SNAPSHOT_INVALID")
        apps_by_uuid.setdefault(uuid, []).append(app)
    return by_index, apps_by_uuid


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
                raise GpuLeaseError("GPU_LEASE_GPU_QUERY_MALFORMED")
            index = _parse_int(fields[0])
            if index is None:
                raise GpuLeaseError("GPU_LEASE_GPU_QUERY_MALFORMED")
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
                raise GpuLeaseError("GPU_LEASE_APPS_QUERY_MALFORMED")
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
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise GpuLeaseError("GPU_LEASE_SNAPSHOT_INVALID")
    return tuple(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _validate_wait_policy(wait_seconds: float, poll_seconds: float) -> None:
    wait = _number(wait_seconds)
    poll = _number(poll_seconds)
    if wait is None or wait < 0 or poll is None or poll <= 0:
        raise GpuLeaseError("GPU_LEASE_WAIT_POLICY_INVALID")


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
