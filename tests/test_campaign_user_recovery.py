"""CPU user flows with real SQLite, dispatch files and process cancellation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.campaign_dispatcher import (
    CampaignDispatchError,
    DispatcherOptions,
    _dispatcher_lock,
    _run_subprocess,
    run_dispatcher,
)


@pytest.fixture
def campaign(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}')
    store = CampaignStore(tmp_path / "state", project_root=tmp_path)
    payload = {
        "campaign_id": "fixture", "goal": "improve score",
        "model": str(model), "dataset": str(data), "budget": 1,
        "resource_request": {"gpu_indices": []},
        "execution": {
            "kind": "pipeline", "repo_root": str(model),
            "output_root": str(tmp_path / "output"),
            "evaluator_contract": str(evaluator), "budget_total_gpu_hours": 1,
            "asset_bindings": {"--checkpoint": str(model)},
        },
    }
    store.create(payload)
    return store, payload


def test_duplicate_submission_and_confirmation_do_not_execute_twice(campaign):
    store, payload = campaign
    assert store.create(payload)["status"] == "created"
    store.confirm("fixture")
    store.confirm("fixture")
    assert store.create(payload)["status"] == "queued"
    calls = []
    for _ in range(2):
        run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {"outcome": "completed"})
    assert len(calls) == 1
    assert store.get("fixture")["status"] == "completed"


@pytest.mark.parametrize("field", ["execution", "model_run", "campaign_id", "execution_hash", "resource_allocation"])
def test_edited_dispatch_projection_never_changes_confirmed_execution(campaign, field):
    store, _ = campaign
    record = store.confirm("fixture")
    pending = Path(record["dispatch_ref"])
    dispatch = json.loads(pending.read_text())
    dispatch[field] = {"unconfirmed": True} if field in {"execution", "model_run", "resource_allocation"} else "changed"
    pending.write_text(json.dumps(dispatch))
    calls = []
    report = run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {})
    assert calls == []
    assert report["failed_campaign_ids"] == ["fixture"]
    assert store.get("fixture")["execution_error"]["message"] == "DISPATCH_EXECUTION_MISMATCH"
    assert not pending.exists()


def test_dispatch_uses_verified_snapshot_after_claim(campaign, monkeypatch):
    store, payload = campaign
    store.confirm("fixture")
    original = CampaignStore.claim_dispatch
    def change_projection(self, campaign_id, source, active):
        claimed = original(self, campaign_id, source, active)
        corrupted = json.loads(active.read_text())
        corrupted["execution"]["output_root"] = str(store.root / "unconfirmed")
        active.write_text(json.dumps(corrupted))
        return claimed
    monkeypatch.setattr(CampaignStore, "claim_dispatch", change_projection)
    calls = []
    run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {})
    assert calls[0]["output_root"] == payload["execution"]["output_root"]


def test_cancelled_pending_work_is_not_launched(campaign):
    store, _ = campaign
    store.confirm("fixture")
    store.cancel("fixture")
    store.cancel("fixture")
    calls = []
    run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {})
    assert calls == []
    assert store.get("fixture")["status"] == "cancelled"
    with pytest.raises(CampaignAPIError, match="STATUS_TRANSITION_INVALID"):
        store.confirm("fixture")


def test_interrupted_claim_is_failed_without_automatic_rerun(campaign):
    store, _ = campaign
    record = store.confirm("fixture")
    pending = Path(record["dispatch_ref"])
    store.claim_dispatch("fixture", pending, store.root / "dispatch/running/fixture.json")
    calls = []
    report = run_dispatcher(DispatcherOptions(store.root), runner=lambda execution: calls.append(execution) or {})
    assert calls == []
    assert report["failed_campaign_ids"] == ["fixture"]
    assert store.get("fixture")["execution_error"]["type"] == "DISPATCH_INTERRUPTED"


def test_lock_excludes_second_process_and_recovers_after_owner_exit(tmp_path):
    lock = tmp_path / "dispatcher.lock"
    ready = tmp_path / "ready"
    code = "from pathlib import Path; import sys, time; from wmloop.control.campaign_dispatcher import _dispatcher_lock\nwith _dispatcher_lock(Path(sys.argv[1])):\n Path(sys.argv[2]).touch()\n time.sleep(15)\n"
    process = subprocess.Popen([sys.executable, "-c", code, str(lock), str(ready)])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(CampaignDispatchError, match="DISPATCHER_ALREADY_RUNNING"):
            with _dispatcher_lock(lock):
                pytest.fail("two owners acquired the dispatcher lock")
        inode = lock.stat().st_ino
        process.terminate()
        process.wait(timeout=5)
        with _dispatcher_lock(lock):
            assert lock.stat().st_ino == inode
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_cancel_terminates_real_cpu_child(monkeypatch):
    monkeypatch.setattr("wmloop.control.campaign_dispatcher._build_command", lambda execution: [sys.executable, "-c", "import time; time.sleep(30)"])
    result = _run_subprocess({}, cancel_requested=lambda: True, terminate_grace_seconds=1)
    assert result["cancelled"] is True
    assert result["termination"] == "SIGTERM"
    assert result["returncode"] != 0
