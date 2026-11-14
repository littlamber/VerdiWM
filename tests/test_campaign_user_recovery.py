"""CPU user flows with real SQLite, dispatch files and process cancellation."""
from __future__ import annotations

import json
import hashlib
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
from wmloop.storage import canonical_bytes


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


@pytest.mark.parametrize("requested_status", ["completed", "failed"])
def test_late_result_preserves_cancellation_intent_and_controller_evidence(campaign, requested_status):
    store, _ = campaign
    record = store.confirm("fixture")
    source = Path(record["dispatch_ref"])
    dispatch = store.claim_dispatch("fixture", source, store.root / "dispatch/running/fixture.json")
    cancelled = store.cancel("fixture")
    result = {"stop_reason": "SIGNAL_RECEIVED", "rounds_completed": 1, "cancelled": True}
    error = {"type": "ControllerStopped", "message": "worker stopped during cancellation"}
    final = store.settle_dispatch("fixture", dispatch=dispatch, status=requested_status,
                                  result=result, error=error)
    record = store.get("fixture")
    assert final == record["status"] == "cancelled"
    assert record["cancellation_requested_at"] == cancelled["cancellation_requested_at"]
    assert record["execution_result"] == result
    assert record["execution_error"] == error
    assert record["execution_result_hash"] == hashlib.sha256(canonical_bytes(result)).hexdigest()
    terminal = json.loads(Path(record["dispatch_ref"]).read_text())
    assert terminal["state"] == "cancelled"
    assert terminal["result"] == result
    assert terminal["error"] == error
    assert not (store.root / "dispatch/running/fixture.json").exists()
    assert not (store.root / f"dispatch/{requested_status}/fixture.json").exists()


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


def test_blocked_revision_can_be_replaced_after_source_fix_with_history_preserved(campaign, monkeypatch):
    store, payload = campaign
    from wmloop.control.automatic_campaign import compile_revision
    monkeypatch.setattr("wmloop.control.campaign_api.compile_revision", lambda **kwargs: compile_revision(**kwargs, project_root=store.project_root))
    store.create(payload)
    record = store.confirm("fixture")
    run_dispatcher(DispatcherOptions(store.root), runner=lambda _: {
        "outcome": "blocked", "blocked_stage": "fixture_gate",
        "error": {"type": "FixtureBoundary", "message": "FIXTURE_REQUIRES_REPAIR"},
    })
    blocked = store.get("fixture")
    old_revision = Path(blocked["revision_ref"])
    old_dispatch = Path(blocked["dispatch_ref"])
    frozen_revision, frozen_dispatch = old_revision.read_bytes(), old_dispatch.read_bytes()
    # A real source snapshot change can create a new internal revision. No
    # evaluator, archive or prior receipt is edited to recover this attempt.
    subprocess.run(["git", "init", "-q", str(store.project_root)], check=True)
    (store.project_root / "wmloop").mkdir()
    (store.project_root / "wmloop/fixed.py").write_text("# repaired source\n")
    replacement = store.create(payload)
    assert replacement["status"] == "created"
    assert replacement["revision_id"] != blocked["revision_id"]
    assert replacement["prior_revision_id"] == blocked["revision_id"]
    previous = replacement["revision_history"][-1]
    assert previous["status"] == "blocked"
    terminal = json.loads(Path(previous["terminal_record_ref"]).read_text())
    assert terminal["execution_result"] == blocked["execution_result"]
    assert terminal["execution_result_hash"] == blocked["execution_result_hash"]
    assert "revision_history" not in terminal
    assert Path(previous["dispatch_ref"]).read_bytes() == frozen_dispatch
    assert old_dispatch.read_bytes() == frozen_dispatch
    assert old_revision.read_bytes() == frozen_revision
    store.confirm("fixture")
    run_dispatcher(DispatcherOptions(store.root), runner=lambda _: {"outcome": "blocked", "reason": "new attempt"})
    assert store.get("fixture")["status"] == "blocked"
    assert Path(previous["dispatch_ref"]).read_bytes() == frozen_dispatch
    assert old_revision.read_bytes() == frozen_revision


def test_blocked_revision_with_live_dispatch_claim_cannot_be_replaced(campaign, monkeypatch):
    store, payload = campaign
    from wmloop.control.automatic_campaign import compile_revision
    monkeypatch.setattr("wmloop.control.campaign_api.compile_revision", lambda **kwargs: compile_revision(**kwargs, project_root=store.project_root))
    store.create(payload)
    store.confirm("fixture")
    run_dispatcher(DispatcherOptions(store.root), runner=lambda _: {"outcome": "blocked"})
    subprocess.run(["git", "init", "-q", str(store.project_root)], check=True)
    active = store.root / "dispatch/running/fixture.json"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text("{}")
    with pytest.raises(CampaignAPIError, match="^CAMPAIGN_REVISION_ACTIVE$"):
        store.create(payload)


@pytest.mark.parametrize("failure", ["transaction", "campaign_id", "revision_id"])
def test_terminal_revision_snapshot_failure_keeps_old_record_and_evidence(campaign, monkeypatch, failure):
    from wmloop.control.automatic_campaign import compile_revision

    store, payload = campaign
    monkeypatch.setattr("wmloop.control.campaign_api.compile_revision", lambda **kwargs: compile_revision(**kwargs, project_root=store.project_root))
    store.create(payload)
    store.confirm("fixture")
    run_dispatcher(DispatcherOptions(store.root), runner=lambda _: {"outcome": "blocked"})
    previous = store.get("fixture")
    dispatch = Path(previous["dispatch_ref"])
    if failure != "transaction":
        projection = json.loads(dispatch.read_text())
        projection[failure] = "another-attempt"
        dispatch.write_text(json.dumps(projection))
    frozen_dispatch = dispatch.read_bytes()
    subprocess.run(["git", "init", "-q", str(store.project_root)], check=True)
    original_write = store._write
    if failure == "transaction":
        def fail_on_replacement(path, value):
            if path == store.root / "fixture.json":
                raise OSError("injected transaction failure")
            original_write(path, value)
        monkeypatch.setattr(store, "_write", fail_on_replacement)
        expected, message = OSError, "injected transaction failure"
    else:
        expected, message = CampaignAPIError, "DISPATCH_REVISION_MISMATCH"
    with pytest.raises(expected, match=message):
        store.create(payload)
    assert store.get("fixture") == previous
    assert dispatch.read_bytes() == frozen_dispatch
    assert not list((store.root / "revisions/fixture").glob("*.terminal.json"))
    assert not list((store.root / "revisions/fixture").glob("*.dispatch.json"))
