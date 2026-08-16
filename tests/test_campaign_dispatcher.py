import json
from pathlib import Path
import sys
import time

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control import campaign_dispatcher
from wmloop.control.campaign_dispatcher import (
    DispatcherOptions,
    _build_command,
    _run_subprocess,
    run_dispatcher,
)
from wmloop.execute import autonomous_pipeline


def _user_fields():
    return {
        "model": "/share/project/model",
        "dataset": "/share/project/data",
        "budget": 0.1,
    }


def test_dispatcher_consumes_pending_manifest_and_settles_campaign(tmp_path: Path):
    api_root = tmp_path / "campaigns"
    store = CampaignStore(api_root)
    record = store.create(
        {
            "campaign_id": "dispatch-1",
            "goal": "verify",
            **_user_fields(),
            "execution": {
                "kind": "evolution",
                "repo_root": "/share/project/model",
                "output_root": "/share/project/runs/dispatch-1",
                "state_root": "/share/project/state/dispatch-1",
                "evaluator_contract": "/share/project/evaluator.json",
                "total_budget_gpu_hours": 0.1,
            },
        }
    )
    assert store.confirm(record["campaign_id"])["status"] == "queued"

    def runner(execution):
        assert execution["kind"] == "evolution"
        return {"verdict": "PASS", "state": "completed"}

    result = run_dispatcher(
        DispatcherOptions(state_root=api_root, max_cycles=1),
        runner=runner,
    )
    assert result["settled_campaign_ids"] == ["dispatch-1"]
    assert store.get("dispatch-1")["status"] == "completed"
    assert (api_root / "dispatch" / "completed" / "dispatch-1.json").is_file()


def test_dispatcher_fails_closed_on_interrupted_running_manifest(tmp_path: Path):
    store = CampaignStore(tmp_path)
    store.create(
        {
            "campaign_id": "interrupted-1",
            "goal": "verify",
            **_user_fields(),
            "execution": {
                "kind": "evolution",
                "repo_root": "/share/project/model",
                "output_root": "/share/project/runs/interrupted-1",
                "state_root": "/share/project/state/interrupted-1",
                "evaluator_contract": "/share/project/evaluator.json",
                "total_budget_gpu_hours": 0.1,
            },
        }
    )
    store.confirm("interrupted-1")
    pending = tmp_path / "dispatch" / "pending" / "interrupted-1.json"
    running = tmp_path / "dispatch" / "running" / "interrupted-1.json"
    running.parent.mkdir(parents=True)
    pending.replace(running)
    store.record_dispatch_result("interrupted-1", status="running")

    result = run_dispatcher(DispatcherOptions(state_root=tmp_path))

    assert result["failed_campaign_ids"] == ["interrupted-1"]
    assert store.get("interrupted-1")["status"] == "failed"
    assert (tmp_path / "dispatch" / "failed" / "interrupted-1.json").is_file()


def test_pipeline_command_forwards_assets_probe_and_budget_controls():
    command = _build_command(
        {
            "kind": "pipeline",
            "repo_root": "/model",
            "output_root": "/runs/c1",
            "evaluator_contract": "/contracts/evaluator.json",
            "probe_contract": "/contracts/probe.json",
            "candidate_catalog": "/contracts/methods.json",
            "settlement_manifest": "/evidence/settlements.json",
            "budget_total_gpu_hours": 2.0,
            "runtime_python": "/runtime/python",
            "asset_bindings": {"--checkpoint": "/assets/model.pt"},
            "probe_imports": False,
            "budget_require_high_cost_approval": False,
        }
    )

    assert command[:3] == [sys.executable, "-m", "wmloop.execute.autonomous_pipeline"]
    assert "--asset=--checkpoint=/assets/model.pt" in command
    assert "--asset" not in command
    assert "--probe-contract" in command
    assert "--candidate-catalog" in command
    assert "--settlement-manifest" in command
    assert "--no-import-probe" in command
    assert "--auto-approve-high-cost" in command


def test_pipeline_command_asset_flags_are_accepted_by_pipeline_parser(
    monkeypatch: pytest.MonkeyPatch,
):
    command = _build_command(
        {
            "kind": "pipeline",
            "repo_root": "/model",
            "output_root": "/runs/c1",
            "evaluator_contract": "/contracts/evaluator.json",
            "budget_total_gpu_hours": 0.2,
            "asset_bindings": {"--checkpoint": "/assets/model.pt"},
        }
    )
    captured = {}

    def fake_run(options):
        captured["asset_bindings"] = options.asset_bindings
        return {"verdict": "PASS"}

    monkeypatch.setattr(autonomous_pipeline, "run_autonomous_pipeline", fake_run)

    assert autonomous_pipeline.main(command[3:]) == 0
    assert captured["asset_bindings"] == (
        ("--checkpoint", Path("/assets/model.pt")),
    )


def test_subprocess_cancellation_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        campaign_dispatcher,
        "_build_command",
        lambda _execution: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    started = time.monotonic()

    result = _run_subprocess(
        {"kind": "pipeline"},
        cancel_requested=lambda: time.monotonic() - started > 0.2,
        poll_seconds=0.02,
        terminate_grace_seconds=0.2,
    )

    assert result["cancelled"] is True
    assert result["termination"] in {"SIGTERM", "SIGKILL"}
    assert time.monotonic() - started < 3


def test_blocked_pipeline_is_a_completed_scientific_outcome(
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = {
        "artifact_type": "verdiwm-autonomous-pipeline-manifest",
        "state": "blocked",
        "verdict": "BLOCKED",
    }
    monkeypatch.setattr(
        campaign_dispatcher,
        "_build_command",
        lambda _execution: [
            sys.executable,
            "-c",
            f"import json,sys; print(json.dumps({manifest!r})); sys.exit(2)",
        ],
    )

    result = _run_subprocess({"kind": "pipeline"})

    assert result["returncode"] == 2
    assert result["outcome"] == "blocked"
    assert result["pipeline_manifest"]["verdict"] == "BLOCKED"


def test_dispatcher_honors_parallel_batch(tmp_path: Path):
    store = CampaignStore(tmp_path)
    for index in range(2):
        campaign_id = f"parallel-{index}"
        store.create(
            {
                "campaign_id": campaign_id,
                "goal": "verify",
                **_user_fields(),
                "execution": {
                    "kind": "evolution",
                    "repo_root": "/share/project/model",
                    "output_root": f"/share/project/runs/{campaign_id}",
                    "state_root": f"/share/project/state/{campaign_id}",
                    "evaluator_contract": "/share/project/evaluator.json",
                    "total_budget_gpu_hours": 0.1,
                },
            }
        )
        store.confirm(campaign_id)

    started = time.monotonic()
    result = run_dispatcher(
        DispatcherOptions(state_root=tmp_path, max_parallel=2),
        runner=lambda _execution: (time.sleep(0.25) or {"state": "completed"}),
    )

    assert sorted(result["settled_campaign_ids"]) == ["parallel-0", "parallel-1"]
    assert time.monotonic() - started < 0.45


def test_dispatcher_can_target_one_campaign_without_consuming_older_work(
    tmp_path: Path,
):
    store = CampaignStore(tmp_path)
    for campaign_id in ("older", "target"):
        store.create(
            {
                "campaign_id": campaign_id,
                "goal": "verify",
                **_user_fields(),
                "execution": {
                    "kind": "evolution",
                    "repo_root": "/share/project/model",
                    "output_root": f"/share/project/runs/{campaign_id}",
                    "state_root": f"/share/project/state/{campaign_id}",
                    "evaluator_contract": "/share/project/evaluator.json",
                    "total_budget_gpu_hours": 0.1,
                },
            }
        )
        store.confirm(campaign_id)

    result = run_dispatcher(
        DispatcherOptions(state_root=tmp_path, campaign_ids=("target",)),
        runner=lambda _execution: {"state": "completed"},
    )

    assert result["settled_campaign_ids"] == ["target"]
    assert store.get("target")["status"] == "completed"
    assert store.get("older")["status"] == "queued"
