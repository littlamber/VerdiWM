"""Bounded CPU concurrency and restart checks using independent campaign stores."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
from wmloop.control import campaign_repository


def _draft(root: Path) -> CampaignStore:
    model = root / "model"
    model.mkdir()
    data = root / "data"
    data.mkdir()
    evaluator = root / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}')
    store = CampaignStore(root / "state", project_root=root)
    store.create({
        "campaign_id": "fixture", "goal": "improve score", "model": str(model),
        "dataset": str(data), "budget": 1, "resource_request": {"gpu_indices": []},
        "execution": {"kind": "pipeline", "repo_root": str(model), "output_root": str(root / "output"),
                      "evaluator_contract": str(evaluator), "budget_total_gpu_hours": 1,
                      "asset_bindings": {"--checkpoint": str(model)}},
    })
    return store


def _assert_queue_count(store: CampaignStore, expected: int):
    assert len(list((store.root / "dispatch/pending").glob("*.json"))) == expected


def test_independent_thread_stores_confirm_once(tmp_path):
    original = _draft(tmp_path)
    barrier = Barrier(4)
    def confirm(_):
        store = CampaignStore(original.root)
        barrier.wait(timeout=5)
        return store.confirm("fixture")
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(confirm, range(4)))
    assert {row["status"] for row in rows} == {"queued"}
    assert len({row["state_version"] for row in rows}) == 1
    _assert_queue_count(original, 1)
    calls = []
    run_dispatcher(DispatcherOptions(original.root), runner=lambda execution: calls.append(execution) or {})
    assert len(calls) == 1


def test_independent_process_stores_confirm_once(tmp_path):
    store = _draft(tmp_path)
    code = "from pathlib import Path; import sys; from wmloop.control.campaign_api import CampaignStore; CampaignStore(Path(sys.argv[1])).confirm('fixture')"
    children = [subprocess.Popen([sys.executable, "-c", code, str(store.root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    try:
        for child in children:
            output, error = child.communicate(timeout=15)
            assert child.returncode == 0, error.decode()
        _assert_queue_count(store, 1)
        assert store.get("fixture")["status"] == "queued"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_concurrent_cancel_and_claim_preserve_cancelled_terminal_state(tmp_path):
    store = _draft(tmp_path)
    row = store.confirm("fixture")
    pending = Path(row["dispatch_ref"])
    active = store.root / "dispatch/running/fixture.json"
    barrier = Barrier(2)
    def operation(action):
        independent = CampaignStore(store.root)
        barrier.wait(timeout=5)
        return independent.cancel("fixture") if action == "cancel" else independent.claim_dispatch("fixture", pending, active)
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(operation, ("claim", "cancel")))
    assert store.get("fixture")["status"] == "cancelled"
    calls = []
    run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {})
    assert calls == []
    _assert_queue_count(store, 0)
    assert not active.exists()
    assert (store.root / "dispatch/cancelled/fixture.json").is_file()


def test_process_exit_after_commit_recovers_pending_work_once(tmp_path):
    store = _draft(tmp_path)
    code = """
from pathlib import Path
import os
import sys
from wmloop.control.campaign_api import CampaignStore
from wmloop.control import campaign_repository
store = CampaignStore(Path(sys.argv[1]))
def stop_before_projection(*args, **kwargs):
    os._exit(17)
campaign_repository.atomic_write = stop_before_projection
store.confirm('fixture')
"""
    child = subprocess.run([sys.executable, "-c", code, str(store.root)], capture_output=True, timeout=15)
    assert child.returncode == 17
    assert store.get("fixture")["status"] == "queued"
    _assert_queue_count(store, 0)
    recovered = CampaignStore(store.root)
    _assert_queue_count(recovered, 1)
    calls = []
    for _ in range(2):
        run_dispatcher(DispatcherOptions(recovered.root), runner=lambda execution: calls.append(execution) or {})
    assert len(calls) == 1
    assert recovered.get("fixture")["status"] == "completed"


@pytest.mark.parametrize("phase", ["confirm", "cancel", "settle"])
def test_restart_flushes_committed_outbox_without_duplicate_work(tmp_path, monkeypatch, phase):
    store = _draft(tmp_path)
    dispatch = None
    if phase != "confirm":
        row = store.confirm("fixture")
        if phase == "settle":
            dispatch = store.claim_dispatch("fixture", Path(row["dispatch_ref"]), store.root / "dispatch/running/fixture.json")
    expected = {"confirm": "queued", "cancel": "cancelled", "settle": "completed"}[phase]
    def unavailable(*args, **kwargs):
        raise OSError("fixture temporary projection failure")
    with monkeypatch.context() as fault:
        fault.setattr(campaign_repository, "atomic_write", unavailable)
        with pytest.raises(OSError, match="fixture temporary"):
            if phase == "confirm":
                store.confirm("fixture")
            elif phase == "cancel":
                store.cancel("fixture")
            else:
                store.settle_dispatch("fixture", dispatch=dispatch, status="completed", result={"test_fixture": True})
    assert store.get("fixture")["status"] == expected
    recovered = CampaignStore(store.root)
    assert json.loads((store.root / "fixture.json").read_text())["status"] == expected
    with sqlite3.connect(store.repository.database) as connection:
        assert connection.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
    calls = []
    for _ in range(2):
        run_dispatcher(DispatcherOptions(recovered.root), runner=lambda execution: calls.append(execution) or {})
    assert len(calls) == (1 if phase == "confirm" else 0)
    _assert_queue_count(recovered, 0)
    assert not (store.root / "dispatch/running/fixture.json").exists()
