"""CPU fault injection for fail-closed admission and complete GPU lease sets."""

from __future__ import annotations

import pytest

from wmloop.execute import gpu_lease
from wmloop.execute.gpu_lease import GpuLeaseError, GpuLeaseManager


def _snapshot(count=3):
    return {
        "gpus": [
            {"index": index, "uuid": f"fixture-{index}", "name": "fixture",
             "memory_used_mib": 0, "utilization_gpu_percent": 0}
            for index in range(count)
        ],
        "compute_apps": [],
    }


def _manager(tmp_path, snapshot):
    return GpuLeaseManager(lock_root=tmp_path / "locks", snapshot_provider=lambda: snapshot)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("key", ["gpu_query_returncode", "apps_query_returncode"])
@pytest.mark.parametrize("returncode", [-1, 1, False, None])
def test_query_failure_never_grants_leases(tmp_path, count, key, returncode):
    snapshot = _snapshot()
    snapshot[key] = returncode
    manager = _manager(tmp_path, snapshot)
    with pytest.raises(GpuLeaseError, match="GPU_LEASE_QUERY_FAILED"):
        manager.acquire_many([0, 1, 2], count)
    snapshot[key] = 0
    leases = manager.acquire_many([0, 1, 2], count)
    for lease in leases:
        lease.release()


@pytest.mark.parametrize("field,value", [
    ("memory_used_mib", float("nan")),
    ("memory_used_mib", float("inf")),
    ("memory_used_mib", -1),
    ("utilization_gpu_percent", float("nan")),
    ("utilization_gpu_percent", float("-inf")),
    ("utilization_gpu_percent", -1),
    ("utilization_gpu_percent", 101),
    ("utilization_gpu_percent", True),
])
def test_invalid_numeric_snapshot_cannot_admit_single_or_group(tmp_path, field, value):
    snapshot = _snapshot(1)
    snapshot["gpus"][0][field] = value
    manager = _manager(tmp_path, snapshot)
    for acquire in (lambda: manager.acquire([0]), lambda: manager.acquire_many([0], 1)):
        with pytest.raises(GpuLeaseError, match="GPU_SNAPSHOT_INVALID"):
            acquire()
    snapshot["gpus"][0][field] = 0
    with manager.acquire([0]):
        pass


def test_multi_gpu_uses_later_idle_candidates(tmp_path):
    snapshot = _snapshot()
    snapshot["gpus"][0]["memory_used_mib"] = 2048
    manager = _manager(tmp_path, snapshot)
    leases = manager.acquire_many([0, 1, 2], 2)
    try:
        assert [lease.index for lease in leases] == [1, 2]
    finally:
        for lease in leases:
            lease.release()


def test_multi_gpu_skips_held_lock_and_rolls_back_incomplete_set(tmp_path):
    manager = _manager(tmp_path, _snapshot())
    with manager.acquire([0]):
        leases = manager.acquire_many([0, 1, 2], 2)
        assert [lease.index for lease in leases] == [1, 2]
        for lease in leases:
            lease.release()
        with pytest.raises(GpuLeaseError, match="GPU_LEASE_UNAVAILABLE"):
            manager.acquire_many([0, 1, 2], 3)
        with manager.acquire([1]), manager.acquire([2]):
            pass


def test_partial_locks_released_if_later_lock_operation_raises(tmp_path, monkeypatch):
    manager = _manager(tmp_path, _snapshot())
    original_lock = gpu_lease._lock_exclusive_nb
    calls = 0

    def fail_second(handle):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected lock failure")
        original_lock(handle)

    monkeypatch.setattr(gpu_lease, "_lock_exclusive_nb", fail_second)
    with pytest.raises(OSError, match="injected lock failure"):
        manager.acquire_many([0, 1], 2)
    monkeypatch.setattr(gpu_lease, "_lock_exclusive_nb", original_lock)
    with manager.acquire([0]), manager.acquire([1]):
        pass


def test_real_query_metadata_rejects_failed_compute_app_inspection(tmp_path, monkeypatch):
    def query(arguments):
        if "--query-gpu=" in arguments[0]:
            return {"returncode": 0, "stdout": "0, fixture-0, fixture, 0, 1000, 0\n"}
        return {"returncode": 1, "stdout": "", "stderr": "injected unavailable query"}

    monkeypatch.setattr(gpu_lease, "_run_nvidia_smi", query)
    manager = GpuLeaseManager(lock_root=tmp_path / "locks")
    with pytest.raises(GpuLeaseError, match="GPU_LEASE_QUERY_FAILED:apps_query_returncode"):
        manager.acquire([0])


@pytest.mark.parametrize("key,value", [
    ("compute_apps", None), ("compute_apps", ["malformed"]),
    ("compute_apps", [{"pid": 123}]),
    ("gpus", None),
])
def test_unknown_snapshot_shape_cannot_prove_gpu_idle(tmp_path, key, value):
    snapshot = _snapshot()
    snapshot[key] = value
    with pytest.raises(GpuLeaseError, match="GPU_LEASE_SNAPSHOT_INVALID"):
        _manager(tmp_path, snapshot).acquire([0])


@pytest.mark.parametrize("wait,poll", [(float("nan"), 1), (0, float("inf")), (True, 1)])
def test_nonfinite_wait_policy_is_rejected(tmp_path, wait, poll):
    manager = _manager(tmp_path, _snapshot())
    with pytest.raises(GpuLeaseError, match="GPU_LEASE_WAIT_POLICY_INVALID"):
        manager.acquire([0], wait_seconds=wait, poll_seconds=poll)


@pytest.mark.parametrize("option,value,error", [
    ("memory_used_threshold_mib", float("nan"), "MEMORY"),
    ("utilization_threshold_percent", float("nan"), "UTILIZATION"),
])
def test_invalid_threshold_cannot_disable_admission_checks(option, value, error):
    with pytest.raises(ValueError, match=f"GPU_LEASE_{error}_THRESHOLD_INVALID"):
        GpuLeaseManager(**{option: value})
