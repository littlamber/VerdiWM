"""Public dispatch settlement for recovered model-run terminal receipts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control import campaign_dispatcher
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
from wmloop.execute.budget import BudgetError
from wmloop.execute.model_run_budget import (
    ModelRunBudget,
    ModelRunBudgetError,
    compile_model_run_budget_binding,
)


def _budget_manifest(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    plan = {"training_plan": {"steps": 3, "seed": 17}}
    execution = {
        "budget_db": str(tmp_path / "ledger.sqlite"),
        "cas_root": str(tmp_path / "cas-root"),
        "budget_total_gpu_hours": 2.0,
        "budget_max_trial_gpu_hours": 2.0,
        "budget_high_trial_limit": 2,
    }
    binding = compile_model_run_budget_binding(
        execution, plan_document=plan, estimated_gpu_hours=0.5
    )
    return {**plan, "budget_binding": binding}, binding


def _terminal_receipt(
    binding: dict[str, object], *, state: str, launched: bool
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "state": state,
        "claim_boundary": "Engineering execution provenance only.",
        "launch_attempted": launched,
        "estimated_gpu_hours": binding["estimated_gpu_hours"],
        "budget_state": "settled" if launched else "released",
        "budget_trial_id": binding["trial_id"],
        "budget_reservation": {
            "trial_id": binding["trial_id"],
            "fencing_token": 1,
            "plan_sha256": binding["plan_sha256"],
        },
    }
    if launched:
        receipt.update(allocated_gpu_count=1, allocated_gpu_hours=0.25)
    return receipt


@pytest.mark.parametrize(
    ("terminal_state", "launched"),
    [
        ("failed", True),
        ("timed_out", True),
        ("cancelled", True),
        ("failed_before_launch", False),
        ("cancelled_before_launch", False),
    ],
)
def test_non_success_budget_receipt_recovers_after_interrupted_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_state: str, launched: bool
) -> None:
    manifest, binding = _budget_manifest(tmp_path)
    budget = ModelRunBudget.admit(manifest)
    receipt_path = tmp_path / "output" / "execution-receipt.json"
    receipt_path.parent.mkdir()
    receipt = _terminal_receipt(binding, state=terminal_state, launched=launched)

    if launched:
        original = budget.ledger.settle

        def interrupt_settlement(*args: object, **kwargs: object):
            monkeypatch.setattr(budget.ledger, "settle", original)
            raise BudgetError("injected interruption after receipt write")

        monkeypatch.setattr(budget.ledger, "settle", interrupt_settlement)
    else:
        original = budget.ledger.release

        def interrupt_release(*args: object, **kwargs: object):
            monkeypatch.setattr(budget.ledger, "release", original)
            raise BudgetError("injected interruption after receipt write")

        monkeypatch.setattr(budget.ledger, "release", interrupt_release)

    with pytest.raises(ModelRunBudgetError, match="SETTLEMENT_INCOMPLETE"):
        budget.persist_terminal(receipt_path, receipt)
    assert json.loads(receipt_path.read_text())["state"] == terminal_state

    recovered = ModelRunBudget.admit(manifest).recover_terminal(receipt_path)
    assert recovered is not None
    assert recovered["state"] == terminal_state
    entry = budget.ledger.get(str(binding["trial_id"]))
    assert entry is not None
    assert entry.state == ("settled" if launched else "released")
    assert ModelRunBudget.admit(manifest).recover_terminal(receipt_path) == recovered


@pytest.mark.parametrize(
    ("receipt_state", "expected_status", "report_key"),
    [
        ("completed", "completed", "settled_campaign_ids"),
        ("failed", "failed", "failed_campaign_ids"),
        ("failed_before_launch", "failed", "failed_campaign_ids"),
        ("timed_out", "failed", "failed_campaign_ids"),
        ("unexpected_terminal_state", "failed", "failed_campaign_ids"),
        ("cancelled", "cancelled", "cancelled_campaign_ids"),
        ("cancelled_before_launch", "cancelled", "cancelled_campaign_ids"),
    ],
)
def test_dispatch_replays_model_run_receipt_to_matching_campaign_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_state: str,
    expected_status: str,
    report_key: str,
) -> None:
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text('{"metrics":["quality"]}', encoding="utf-8")
    store = CampaignStore(tmp_path / "campaigns", project_root=tmp_path)
    record = store.create(
        {
            "goal": "improve quality",
            "model": str(model),
            "dataset": str(data),
            "budget": 1,
            "resource_request": {"gpu_indices": []},
            "execution": {
                "kind": "pipeline",
                "repo_root": str(model),
                "output_root": str(tmp_path / "output"),
                "evaluator_contract": str(evaluator),
                "budget_total_gpu_hours": 1.0,
                "asset_bindings": {"--checkpoint": str(model)},
            },
        }
    )
    # The manifest is opaque to the dispatcher; the production v3 executor
    # receives this frozen snapshot after claim_dispatch verifies it.
    record["model_run"] = {"model_run_id": "recovered-run"}
    store._write(store._path(record["campaign_id"]), record)
    store.confirm(record["campaign_id"])
    terminal = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-run-execution-receipt",
        "state": receipt_state,
        "budget_state": "settled",
        "claim_boundary": "Already-settled execution receipt replayed after dispatcher restart.",
    }
    monkeypatch.setattr(
        campaign_dispatcher,
        "execute_model_run",
        lambda manifest, cancel_requested: terminal,
    )

    report = run_dispatcher(DispatcherOptions(store.root))
    final = store.get(record["campaign_id"])
    assert final["status"] == expected_status
    assert final["execution_result"] == terminal
    assert report[report_key] == [record["campaign_id"]]
    assert json.loads(Path(final["dispatch_ref"]).read_text())["state"] == expected_status
