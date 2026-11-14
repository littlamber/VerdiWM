"""Cancellation during GPU admission uses fake snapshots and real CPU signals."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from wmloop.control.model_run import compile_model_run
from wmloop.execute.gpu_lease import GpuLeaseCancelled, GpuLeaseManager
from wmloop.execute.model_runner import execute_model_run
from wmloop.experiments.training_scale import build_training_scale_plan


def _snapshot(*, busy: bool = False) -> dict[str, object]:
    return {
        "gpus": [{"index": index, "uuid": f"cpu-{index}", "name": "CPU fixture",
                  "memory_used_mib": 2048 if busy and index == 1 else 0,
                  "utilization_gpu_percent": 0} for index in range(2)],
        "compute_apps": [],
    }


def _manager(root: Path, snapshot: dict[str, object]) -> GpuLeaseManager:
    return GpuLeaseManager(lock_root=root / "locks", snapshot_provider=lambda: snapshot)


def _manifest(root: Path) -> dict[str, object]:
    model, data = root / "model", root / "data"
    model.mkdir()
    data.mkdir()
    (model / "worker.py").write_text(
        "from pathlib import Path\nimport json, sys\n"
        "manifest = json.loads(Path(sys.argv[1]).read_text())\n"
        "(Path(manifest['output']['root']).parent / 'launched').touch()\n"
    )
    evaluator = root / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}')
    for split in ("train", "val"):
        (data / f"{split}.json").write_text(json.dumps({"records": [
            {"episode_id": f"{split}-{index}", "sample_id": f"{split}-{index}:0:150",
             "horizon_frames": 150} for index in range(20)
        ]}))
    scale = build_training_scale_plan(
        train_manifest=data / "train.json", val_manifest=data / "val.json", stage="probe",
    )
    command = ["worker.py", "{model_run_manifest}"]
    manifest = compile_model_run(
        campaign_id="lease-wait-fixture", model=model, data=data,
        training_scale_plan=scale,
        execution={
            "kind": "pipeline", "repo_root": str(model), "output_root": str(root / "output"),
            "evaluator_contract": str(evaluator), "runtime_python": sys.executable,
            "budget_total_gpu_hours": 1, "asset_bindings": {"--checkpoint": str(model)},
            "model_run_adapter": {
                "profile_id": "cpu-admission", "model_family": "cpu", "capability_level": "L2",
                "runner": {"requires_training": True, "protocol_version": 1, "adapter_id": None,
                           "train": command, "evaluate": command,
                           "checkpoint_receipt": "checkpoint.json", "evaluation_receipt": "evidence.json"},
            },
        },
        resource_allocation={
            "artifact_type": "verdiwm-gpu-resource-allocation", "state": "ready",
            "roles": [{"role": "autonomous_candidate_evaluation", "gpu_indices": [1]}],
        },
    )
    output = root / "output"
    output.mkdir()
    (output / "model-run.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("method", ["single", "many"])
def test_wait_cancels_without_waiting_for_long_poll_interval(tmp_path, method):
    manager = _manager(tmp_path, _snapshot(busy=True))
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(GpuLeaseCancelled, match="GPU_LEASE_CANCELLED"):
            if method == "single":
                manager.acquire([1], wait_seconds=60, poll_seconds=30, cancel_requested=stop.is_set)
            else:
                manager.acquire_many([1], 1, wait_seconds=60, poll_seconds=30,
                                     cancel_requested=stop.is_set)
    finally:
        timer.cancel()
        timer.join()
    assert time.monotonic() - started < 1


def test_cancelled_partial_world_releases_all_admitted_locks(tmp_path):
    snapshot = _snapshot(busy=True)
    manager = _manager(tmp_path, snapshot)
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    try:
        with pytest.raises(GpuLeaseCancelled):
            manager.acquire_many([0, 1], 2, wait_seconds=60, cancel_requested=stop.is_set)
    finally:
        timer.cancel()
        timer.join()
    snapshot["gpus"][1]["memory_used_mib"] = 0
    leases = manager.acquire_many([0, 1], 2)
    for lease in leases:
        lease.release()


def test_cancel_racing_with_successful_admission_releases_complete_world(tmp_path):
    stop = threading.Event()

    def snapshot():
        stop.set()  # Cancellation arrives while the admission attempt is in flight.
        return _snapshot()

    manager = GpuLeaseManager(lock_root=tmp_path / "locks", snapshot_provider=snapshot)
    with pytest.raises(GpuLeaseCancelled):
        manager.acquire_many([0, 1], 2, cancel_requested=stop.is_set)
    leases = manager.acquire_many([0, 1], 2)
    for lease in leases:
        lease.release()


def test_cancellation_callback_failure_after_admission_does_not_leak_locks(tmp_path):
    admitted = False

    def snapshot():
        nonlocal admitted
        admitted = True
        return _snapshot()

    def cancellation():
        if admitted:
            raise RuntimeError("injected cancellation callback failure")
        return False

    manager = GpuLeaseManager(lock_root=tmp_path / "locks", snapshot_provider=snapshot)
    with pytest.raises(RuntimeError, match="injected cancellation callback failure"):
        manager.acquire_many([0, 1], 2, cancel_requested=cancellation)
    leases = manager.acquire_many([0, 1], 2)
    for lease in leases:
        lease.release()


def _assert_unstarted_receipt(root: Path, receipt: dict[str, object]) -> None:
    assert receipt["state"] == "cancelled"
    assert receipt["cancelled"] is True
    assert receipt["leased_gpus"] == []
    assert [stage["stage"] for stage in receipt["stages"]] == ["train", "evaluate"]
    assert all(stage["state"] == "not_started" and stage["returncode"] is None
               for stage in receipt["stages"])
    assert "settled_cost" not in receipt
    assert not (root / "launched").exists()
    assert not (root / "output/checkpoint.json").exists()
    assert not (root / "output/evidence.json").exists()
    assert json.loads((root / "output/execution-receipt.json").read_text()) == receipt


def test_model_run_cancelled_before_admission_does_not_query_or_launch(tmp_path):
    manifest = _manifest(tmp_path)

    def forbidden_snapshot():
        pytest.fail("pre-cancelled model run queried GPU availability")

    manager = GpuLeaseManager(lock_root=tmp_path / "locks", snapshot_provider=forbidden_snapshot)
    result = execute_model_run(manifest, lease_manager=manager, lease_wait_seconds=60,
                               cancel_requested=lambda: True)
    _assert_unstarted_receipt(tmp_path, result)


def test_model_run_wait_cancellation_writes_unstarted_receipt(tmp_path):
    manifest = _manifest(tmp_path)
    manager = _manager(tmp_path, _snapshot(busy=True))
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        result = execute_model_run(manifest, lease_manager=manager, lease_wait_seconds=60,
                                   cancel_requested=stop.is_set)
    finally:
        timer.cancel()
        timer.join()
    assert time.monotonic() - started < 1
    assert result["elapsed_seconds"] > 0
    _assert_unstarted_receipt(tmp_path, result)


@pytest.mark.parametrize("kind", ["duck", "subclass"])
def test_custom_manager_retains_legacy_signature_and_releases_late_cancel(tmp_path, kind):
    manifest = _manifest(tmp_path)
    delegate = _manager(tmp_path, _snapshot())
    stop = threading.Event()
    calls = []

    class LegacyManager(GpuLeaseManager if kind == "subclass" else object):
        def acquire_many(self, allowed_indices, count, *, wait_seconds=0.0):
            calls.append((allowed_indices, count, wait_seconds))
            leases = delegate.acquire_many(allowed_indices, count)
            stop.set()
            return leases

    result = execute_model_run(manifest, lease_manager=LegacyManager(), lease_wait_seconds=0.7,
                               cancel_requested=stop.is_set)
    assert calls == [([1], 1, 0.7)]
    assert result["state"] == "cancelled"
    assert result["stages"][0]["state"] == "not_started"
    assert result["leased_gpus"][0]["uuid"] == "cpu-1"
    assert not (tmp_path / "launched").exists()
    with delegate.acquire([1]):
        pass


_SIGNAL_CONTROLLER = """
import json, sys
from pathlib import Path
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.model_runner import execute_model_run
root = Path(sys.argv[1])
def snapshot():
    (root / 'waiting').touch()
    return {'gpus': [{'index': 1, 'uuid': 'cpu-1', 'name': 'CPU fixture',
                     'memory_used_mib': 2048, 'utilization_gpu_percent': 0}], 'compute_apps': []}
manager = GpuLeaseManager(lock_root=root / 'locks', snapshot_provider=snapshot)
manifest = json.loads((root / 'output/model-run.json').read_text())
try:
    execute_model_run(manifest, lease_manager=manager, lease_wait_seconds=60)
except KeyboardInterrupt:
    raise SystemExit(130)
"""


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_real_signal_interrupts_model_run_lease_wait_and_saves_receipt(tmp_path, signum):
    _manifest(tmp_path)
    controller = subprocess.Popen(
        [sys.executable, "-c", _SIGNAL_CONTROLLER, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "waiting").exists() and time.monotonic() < deadline:
            assert controller.poll() is None
            time.sleep(0.01)
        assert (tmp_path / "waiting").exists()
        started = time.monotonic()
        os.kill(controller.pid, signum)
        _, stderr = controller.communicate(timeout=3)
        assert controller.returncode == 128 + signum, stderr.decode()
        assert time.monotonic() - started < 2
        receipt = json.loads((tmp_path / "output/execution-receipt.json").read_text())
        _assert_unstarted_receipt(tmp_path, receipt)
        with _manager(tmp_path, _snapshot()).acquire([1]):
            pass
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)
