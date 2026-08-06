from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import pytest

from wmloop.execute import campaign_daemon
from wmloop.execute import experiment_scheduler
from wmloop.execute.campaign_daemon import (
    CampaignDaemonError,
    CampaignDaemonOptions,
)
from wmloop.execute.gpu_lease import GpuLeaseError


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "configs" / "smoke" / "auto_experiment_candidate_batch_cuda_v1.json"


def _write_queue(
    root: Path,
    *,
    candidate_ids: tuple[str, ...],
) -> Path:
    root.mkdir(parents=True)
    plans = root / "plans"
    plans.mkdir()
    selected = []
    for candidate_id in candidate_ids:
        plan = plans / f"{candidate_id}-screen.json"
        plan.write_text(
            json.dumps({"candidate_id": candidate_id, "stage": "screen"}),
            encoding="utf-8",
        )
        selected.append(
            {
                "candidate_id": candidate_id,
                "stages": [
                    {
                        "stage": "screen",
                        "plan_path": f"plans/{plan.name}",
                        "plan_sha256": campaign_daemon._sha256(plan.read_bytes()),
                    }
                ],
            }
        )
    queue = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-queue",
        "state": "ready",
        "total_budget_gpu_hours": 1.0,
        "selected": selected,
        "deferred": [],
    }
    path = root / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")
    return path


def _options(
    tmp_path: Path,
    queues: tuple[Path, ...],
    **overrides: object,
) -> CampaignDaemonOptions:
    values: dict[str, object] = {
        "queue_paths": queues,
        "output_root": tmp_path / "daemon",
        "workspace_root": ROOT,
        "archive_db": tmp_path / "archive.db",
        "cas_root": tmp_path / "cas",
        "lock_root": tmp_path / "leases",
        "poll_seconds": 0.0,
        "max_cycles": 2,
        "max_parallel": 2,
        "max_attempts_per_candidate": 3,
        "cleanup_enabled": False,
    }
    values.update(overrides)
    return CampaignDaemonOptions(**values)  # type: ignore[arg-type]


def _candidate_id(queue_path: Path) -> str:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    return str(queue["selected"][0]["candidate_id"])


def test_workers_are_bounded_and_share_one_budget(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a", "candidate-b"))
    active = 0
    max_active = 0
    calls: list[tuple[Path, Path, float]] = []
    lock = threading.Lock()

    def runner(**kwargs: object) -> dict[str, object]:
        nonlocal active, max_active
        worker_queue = Path(kwargs["queue_path"])
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(
                (
                    worker_queue,
                    Path(kwargs["budget_db"]),
                    float(kwargs["budget_total_gpu_hours"]),
                )
            )
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"candidate_states": {_candidate_id(worker_queue): "completed"}}

    manifest = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (queue,)),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert manifest["launch_count"] == 2
    assert max_active == 2
    assert {budget for _, budget, _ in calls} == {
        tmp_path / "daemon" / "budget.db"
    }
    assert {total for _, _, total in calls} == {1.0}
    assert all("cycle-" not in str(worker_queue) for worker_queue, _, _ in calls)


def test_projected_queue_runs_through_real_scheduler_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue_root = tmp_path / "queue"
    experiment_scheduler.plan_candidate_batch(
        batch_path=BATCH,
        output_root=queue_root,
        workspace_root=ROOT,
    )
    stages: list[str] = []

    def fake_experiment(**kwargs: object) -> dict[str, object]:
        stages.append(Path(kwargs["output_root"]).name)
        return {"verdict": "PASS", "evidence_level": "runtime_verified"}

    monkeypatch.setattr(experiment_scheduler, "run_auto_experiment", fake_experiment)
    manifest = campaign_daemon.run_campaign_daemon(
        _options(
            tmp_path,
            (queue_root / "queue.json",),
            max_parallel=1,
        ),
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert stages == ["screen", "gate", "confirm"]
    worker_executions = list((tmp_path / "daemon" / "workers").rglob("execution.json"))
    assert len(worker_executions) == 1


def test_same_candidate_id_in_distinct_queues_has_distinct_state(tmp_path: Path) -> None:
    first_queue = _write_queue(tmp_path / "first", candidate_ids=("shared-id",))
    second_queue = _write_queue(tmp_path / "second", candidate_ids=("shared-id",))

    def runner(**kwargs: object) -> dict[str, object]:
        candidate_id = _candidate_id(Path(kwargs["queue_path"]))
        return {"candidate_states": {candidate_id: "completed"}}

    manifest = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (first_queue, second_queue)),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert len(manifest["candidate_states"]) == 2
    assert all(key.endswith("::shared-id") for key in manifest["candidate_states"])
    assert manifest["budget_total_gpu_hours"] == 2.0


def test_restart_skips_completed_candidate_and_reuses_worker_root(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a", "candidate-b"))
    first_calls: list[Path] = []

    def first_runner(**kwargs: object) -> dict[str, object]:
        worker_queue = Path(kwargs["queue_path"])
        first_calls.append(worker_queue)
        candidate_id = _candidate_id(worker_queue)
        if candidate_id == "candidate-b":
            raise RuntimeError("synthetic worker interruption")
        return {"candidate_states": {candidate_id: "completed"}}

    first = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (queue,), max_cycles=1),
        queue_runner=first_runner,
        sleeper=lambda _: None,
    )
    assert first["state"] == "blocked"

    second_calls: list[Path] = []

    def second_runner(**kwargs: object) -> dict[str, object]:
        worker_queue = Path(kwargs["queue_path"])
        second_calls.append(worker_queue)
        candidate_id = _candidate_id(worker_queue)
        return {"candidate_states": {candidate_id: "completed"}}

    second = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (queue,), max_cycles=2),
        queue_runner=second_runner,
        sleeper=lambda _: None,
    )

    assert second["state"] == "completed"
    assert len(second_calls) == 1
    assert _candidate_id(second_calls[0]) == "candidate-b"
    first_b = next(path for path in first_calls if _candidate_id(path) == "candidate-b")
    assert second_calls[0] == first_b


def test_max_attempts_blocks_candidate_without_unbounded_retry(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))
    calls = 0

    def runner(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("always fails")

    manifest = campaign_daemon.run_campaign_daemon(
        _options(
            tmp_path,
            (queue,),
            max_cycles=8,
            max_parallel=1,
            max_attempts_per_candidate=2,
        ),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "blocked"
    assert calls == 2
    record = next(iter(manifest["candidate_states"].values()))
    assert record["state"] == "blocked"
    assert record["errors"] == 2


def test_gpu_capacity_deferral_does_not_consume_failure_attempts(
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))
    calls = 0

    def runner(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GpuLeaseError(
                "GPU_LEASE_UNAVAILABLE:4:COMPUTE_APP_PRESENT"
            )
        candidate_id = _candidate_id(Path(kwargs["queue_path"]))
        return {"candidate_states": {candidate_id: "completed"}}

    manifest = campaign_daemon.run_campaign_daemon(
        _options(
            tmp_path,
            (queue,),
            max_cycles=2,
            max_parallel=1,
            max_attempts_per_candidate=1,
        ),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert calls == 2
    record = next(iter(manifest["candidate_states"].values()))
    assert record["errors"] == 0
    assert record["deferrals"] == 1
    first_cycle = json.loads(
        (tmp_path / "daemon" / "cycles" / "cycle-000001.json").read_text()
    )
    assert first_cycle["workers"][0]["state"] == "deferred"


def test_cycle_limit_blocks_still_retryable_work(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))

    def runner(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("transient")

    manifest = campaign_daemon.run_campaign_daemon(
        _options(
            tmp_path,
            (queue,),
            max_cycles=1,
            max_parallel=1,
            max_attempts_per_candidate=3,
        ),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "blocked"
    assert manifest["cycle"] == 1


def test_cleanup_failure_is_recorded_without_losing_completed_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))

    def cleanup(**_kwargs: object) -> dict[str, object]:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(campaign_daemon, "cleanup_auto_experiment_scratch", cleanup)

    def runner(**kwargs: object) -> dict[str, object]:
        candidate_id = _candidate_id(Path(kwargs["queue_path"]))
        return {"candidate_states": {candidate_id: "completed"}}

    manifest = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (queue,), cleanup_enabled=True, max_parallel=1),
        queue_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert manifest["cleanup"]["state"] == "error"
    assert manifest["cleanup"]["deleted_count"] == 0


def test_stop_signal_is_persisted_after_in_flight_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))
    installed: dict[str, object] = {}

    def install(handler: object) -> dict[int, object]:
        installed["handler"] = handler
        return {}

    monkeypatch.setattr(campaign_daemon, "_install_signal_handlers", install)

    def runner(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("retryable")

    def sleeper(_seconds: float) -> None:
        handler = installed["handler"]
        assert callable(handler)
        handler(signal.SIGTERM, None)

    manifest = campaign_daemon.run_campaign_daemon(
        _options(tmp_path, (queue,), max_cycles=3, max_parallel=1),
        queue_runner=runner,
        sleeper=sleeper,
    )

    assert manifest["state"] == "stopped"
    status = json.loads((tmp_path / "daemon" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopped"


def test_daemon_lock_rejects_live_pid_and_takes_over_stale_pid(tmp_path: Path) -> None:
    destination = tmp_path / "daemon"
    destination.mkdir()
    lock_path = destination / "daemon.lock"
    lock_path.write_text(json.dumps({"pid": __import__("os").getpid()}), encoding="utf-8")
    with pytest.raises(CampaignDaemonError, match="ALREADY_RUNNING"):
        campaign_daemon._acquire_daemon_lock(destination)

    lock_path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    campaign_daemon._acquire_daemon_lock(destination)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["pid"] == __import__("os").getpid()
    campaign_daemon._release_daemon_lock(destination)
    assert not lock_path.exists()


def test_worker_plan_path_cannot_escape_source_queue(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["selected"][0]["stages"][0]["plan_path"] = "../outside.json"
    payload["selected"][0]["stages"][0]["plan_sha256"] = campaign_daemon._sha256(
        outside.read_bytes()
    )
    queue.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignDaemonError, match="PLAN_PATH_INVALID"):
        campaign_daemon.run_campaign_daemon(
            _options(tmp_path, (queue,), max_cycles=1, max_parallel=1),
            queue_runner=lambda **_: {},
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize("resource_inside_output", [False, True])
def test_output_and_durable_resources_must_not_overlap(
    tmp_path: Path,
    resource_inside_output: bool,
) -> None:
    queue = _write_queue(tmp_path / "queue", candidate_ids=("candidate-a",))
    if resource_inside_output:
        chosen_output = tmp_path / "daemon"
        cas_root = chosen_output / "cas"
    else:
        cas_root = tmp_path / "durable-cas"
        chosen_output = cas_root / "daemon"

    with pytest.raises(CampaignDaemonError, match="OUTPUT_OVERLAP_INVALID"):
        campaign_daemon.run_campaign_daemon(
            _options(
                tmp_path,
                (queue,),
                output_root=chosen_output,
                cas_root=cas_root,
                max_cycles=1,
                max_parallel=1,
            ),
            queue_runner=lambda **_: {},
            sleeper=lambda _: None,
        )
