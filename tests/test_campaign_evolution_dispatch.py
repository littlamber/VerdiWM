"""Public campaign states must reflect a real controller child process exit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from wmloop.control import campaign_dispatcher
from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher


def _manifest(tmp_path: Path, state: str, reason: str) -> dict:
    return {
        "schema_version": 1, "artifact_type": "verdiwm-evolution-daemon-manifest",
        "state": state, "input_hash": "a" * 64,
        "output_root": str(tmp_path / "output"), "state_root": str(tmp_path / "evolution"),
        "iteration": 1, "max_iterations": 2, "poll_seconds": 0,
        "total_budget_gpu_hours": 1, "max_failures": 3, "max_no_information": 3,
        "failure_count": 0, "no_information_count": 0, "settled_iterations": 1,
        "deferral_count": 0, "last_outcome": None, "last_error": None,
        "last_signatures": [], "last_signature_hash": None,
        "current_iteration_path": None, "stop_reason": reason,
        "claim_boundary": "Controller execution state confers no scientific verdict authority.",
    }


@pytest.mark.parametrize(("state", "reason", "expected"), [
    ("completed", "MAX_ITERATIONS_REACHED", "completed"),
    ("completed", "GOAL_CONFIRMED", "completed"),
    ("blocked", "CONSECUTIVE_FAILURE_LIMIT", "blocked"),
    ("exhausted", "GLOBAL_BUDGET_EXHAUSTED", "blocked"),
    ("interrupted", "CONTROLLER_ERROR", "failed"),
    ("stopped", "SIGNAL_RECEIVED", "cancelled"),
])
def test_child_terminal_state_survives_campaign_settlement(tmp_path, monkeypatch, state, reason, expected):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text('{"metrics":["quality"]}')
    store = CampaignStore(tmp_path / "campaigns", project_root=tmp_path)
    created = store.create({
        "goal": "improve quality", "model": str(model), "dataset": str(data), "budget": 1,
        "resource_request": {"gpu_indices": []},
        "execution": {
            "kind": "evolution", "repo_root": str(model), "output_root": str(tmp_path / "output"),
            "state_root": str(tmp_path / "evolution"), "evaluator_contract": str(evaluator),
            "total_budget_gpu_hours": 1,
        },
    })
    campaign_id = created["campaign_id"]
    store.confirm(campaign_id)
    payload = json.dumps(_manifest(tmp_path, state, reason))
    monkeypatch.setattr(campaign_dispatcher, "_build_command", lambda _: [
        sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload,
    ])
    report = run_dispatcher(DispatcherOptions(store.root))
    record = store.get(campaign_id)
    assert record["status"] == expected
    assert record["execution_result"]["stop_reason"] == reason
    assert record["execution_result"]["rounds_completed"] == 1
    collection = "settled_campaign_ids" if expected == "completed" else f"{expected}_campaign_ids"
    assert report[collection] == [campaign_id]


@pytest.mark.parametrize("payload,error", [
    ("", "DISPATCH_EVOLUTION_MANIFEST_MISSING"),
    ('{"artifact_type":"verdiwm-evolution-daemon-manifest"}', "DISPATCH_EVOLUTION_MANIFEST_INVALID"),
])
def test_zero_exit_without_valid_controller_manifest_fails(tmp_path, monkeypatch, payload, error):
    monkeypatch.setattr(campaign_dispatcher, "_build_command", lambda _: [
        sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload,
    ])
    with pytest.raises(campaign_dispatcher.CampaignProcessError, match=error):
        campaign_dispatcher._run_subprocess({"kind": "evolution"})


def test_evolution_command_preserves_frozen_pipeline_inputs(tmp_path):
    command = campaign_dispatcher._build_command({
        "kind": "evolution", "repo_root": tmp_path, "output_root": tmp_path / "run",
        "state_root": tmp_path / "state", "evaluator_contract": tmp_path / "evaluator.json",
        "total_budget_gpu_hours": 0.5, "max_iterations": 2, "max_no_information": 0,
        "stop_on_success": True, "candidate_catalog": tmp_path / "catalog.json",
        "settlement_manifest": tmp_path / "settlement.json", "model_irg_path": tmp_path / "irg.json",
        "irg_protected_metrics": ["action_following"], "adapter_contract_ready": True,
        "adapter_asset_parameters": ["--checkpoint"], "asset_bindings": {"--checkpoint": tmp_path / "weights"},
        "runtime_timeout_seconds": 17,
    })
    assert "--stop-on-success" in command
    assert command[command.index("--max-iterations") + 1] == "2"
    assert command[command.index("--max-no-information") + 1] == "0"
    assert command[command.index("--total-budget-gpu-hours") + 1] == "0.5"
    assert command[command.index("--irg-protected-metric") + 1] == "action_following"
    assert "--adapter-asset-parameter=--checkpoint" in command
    assert "--adapter-contract-ready" in command
    assert command[command.index("--candidate-catalog") + 1] == str(tmp_path / "catalog.json")
    assert command[command.index("--settlement-manifest") + 1] == str(tmp_path / "settlement.json")
