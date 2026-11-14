"""CPU-only tests for model-run budget admission and receipt settlement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy
from wmloop.execute.model_run_budget import (
    ModelRunBudget,
    ModelRunBudgetError,
    compile_model_run_budget_binding,
    estimate_model_run_gpu_hours,
    model_run_budget_plan_sha256,
    validate_model_run_budget_binding,
    validate_model_run_budget_limit,
)


def _manifest(
    tmp_path: Path,
    *,
    estimate: float = 0.5,
    total: float = 2.0,
    max_trial: float = 2.0,
):
    plan = {"training_plan": {"steps": 3, "seed": 17}}
    execution = {
        "budget_db": str(tmp_path / "ledger.sqlite"),
        "cas_root": str(tmp_path / "cas-root"),
        "budget_total_gpu_hours": total,
        "budget_max_trial_gpu_hours": max_trial,
        "budget_high_trial_limit": 2,
    }
    binding = compile_model_run_budget_binding(
        execution, plan_document=plan, estimated_gpu_hours=estimate
    )
    return {**plan, "budget_binding": binding}, binding


def _terminal(binding, *, state="completed", gpu_hours=0.25, gpu_count=1):
    return {
        "state": state,
        "claim_boundary": "Engineering execution provenance only.",
        "launch_attempted": True,
        "allocated_gpu_count": gpu_count,
        "allocated_gpu_hours": gpu_hours,
        "estimated_gpu_hours": binding["estimated_gpu_hours"],
        "budget_state": "settled",
        "budget_trial_id": binding["trial_id"],
        "budget_reservation": {
            "trial_id": binding["trial_id"],
            "fencing_token": 1,
            "plan_sha256": binding["plan_sha256"],
        },
    }


def test_gpu_hour_estimate_sums_stage_budgets_per_gpu():
    assert estimate_model_run_gpu_hours({"train": 3600, "eval": 1800}, 2) == 3.0


def test_model_run_budget_limit_uses_the_frozen_estimate():
    validate_model_run_budget_limit(1.0, 1.0)
    with pytest.raises(
        ModelRunBudgetError,
        match=r"MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL:estimated=25:budget=1",
    ):
        validate_model_run_budget_limit(25.0, 1.0)


def test_budget_binding_rejects_estimate_above_total_at_compile_time(tmp_path):
    execution = {
        "budget_db": str(tmp_path / "ledger.sqlite"),
        "cas_root": str(tmp_path / "cas"),
        "budget_total_gpu_hours": 1.0,
    }

    with pytest.raises(
        ModelRunBudgetError,
        match=r"MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL:estimated=25:budget=1",
    ):
        compile_model_run_budget_binding(
            execution, plan_document={"plan": 1}, estimated_gpu_hours=25.0
        )


def _reseal_budget_binding(manifest):
    binding = manifest["budget_binding"]
    plan_document = {
        key: value for key, value in manifest.items()
        if key not in {"model_run_id", "budget_binding"}
    }
    digest = model_run_budget_plan_sha256(
        plan_document,
        ledger_path=binding["ledger_path"],
        cas_root=binding["cas_root"],
        estimate=binding["estimated_gpu_hours"],
        cost_class=binding["cost_class"],
        human_approved=binding["human_approved"],
        policy=binding["policy"],
    )
    binding["plan_sha256"] = digest
    binding["trial_id"] = "model-run-trial-" + digest[:24]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda binding: binding["policy"].update(total_gpu_hours=0.25),
            "MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL",
        ),
        (
            lambda binding: binding.update(cost_class="low"),
            "MODEL_RUN_BUDGET_COST_CLASS_INVALID",
        ),
    ],
)
def test_resealed_budget_binding_cannot_weaken_frozen_admission_policy(
    tmp_path, mutate, error
):
    estimate = 50.0 if error.endswith("COST_CLASS_INVALID") else 0.5
    total = 200.0 if error.endswith("COST_CLASS_INVALID") else 2.0
    manifest, _ = _manifest(tmp_path, estimate=estimate, total=total, max_trial=200.0)
    mutate(manifest["budget_binding"])
    _reseal_budget_binding(manifest)

    with pytest.raises(ModelRunBudgetError, match=error):
        validate_model_run_budget_binding(manifest)


def test_budget_rejects_understated_cost_class_and_unapproved_high_cost(tmp_path):
    with pytest.raises(ModelRunBudgetError, match="COST_CLASS_INVALID"):
        compile_model_run_budget_binding(
            {
                "budget_db": str(tmp_path / "ledger.sqlite"),
                "cas_root": str(tmp_path / "cas"),
                "budget_total_gpu_hours": 200.0,
                "budget_max_trial_gpu_hours": 200.0,
                "budget_cost_class": "low",
            },
            plan_document={"plan": 1},
            estimated_gpu_hours=50.0,
        )

    manifest, _ = _manifest(tmp_path, estimate=121.0, total=200.0, max_trial=200.0)
    with pytest.raises(ModelRunBudgetError, match="ADMISSION_FAILED"):
        ModelRunBudget.admit(manifest)


def test_budget_exhaustion_fails_before_admission(tmp_path):
    policy = BudgetPolicy(
        total_gpu_hours=1.0,
        high_trial_limit=2,
        max_trial_gpu_hours=2.0,
        require_high_cost_approval=True,
    )
    BudgetLedger(tmp_path / "ledger.sqlite", policy).admit(
        "existing", cost_class="very_low", estimated_gpu_hours=0.5
    )
    manifest, _ = _manifest(tmp_path, estimate=0.75, total=1.0, max_trial=2.0)
    with pytest.raises(ModelRunBudgetError, match="ADMISSION_FAILED"):
        ModelRunBudget.admit(manifest)


def test_concurrent_duplicate_admission_creates_one_reservation(tmp_path):
    manifest, binding = _manifest(tmp_path)

    def admit():
        try:
            return ModelRunBudget.admit(manifest)
        except ModelRunBudgetError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: admit(), range(4)))

    admitted = [result for result in results if result is not None]
    assert sum(result.is_new for result in admitted) == 1
    ledger_entry = admitted[0].ledger.get(str(binding["trial_id"]))
    assert ledger_entry is not None
    assert ledger_entry.state == "admitted"


def test_settlement_is_idempotent_and_recovery_finishes_after_crash(tmp_path, monkeypatch):
    manifest, binding = _manifest(tmp_path)
    budget = ModelRunBudget.admit(manifest)
    receipt_path = tmp_path / "output" / "execution-receipt.json"
    receipt_path.parent.mkdir()
    receipt = _terminal(binding)
    original_settle = budget.ledger.settle
    calls = 0

    def crash_after_cas(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BudgetError("injected settlement interruption")
        return original_settle(*args, **kwargs)

    monkeypatch.setattr(budget.ledger, "settle", crash_after_cas)
    with pytest.raises(ModelRunBudgetError, match="SETTLEMENT_INCOMPLETE"):
        budget.persist_terminal(receipt_path, receipt)
    assert receipt_path.is_file()
    assert budget.ledger.get(str(binding["trial_id"])).state == "admitted"

    recovered = budget.recover_terminal(receipt_path)
    assert recovered is not None
    assert budget.ledger.get(str(binding["trial_id"])).state == "settled"
    assert budget.recover_terminal(receipt_path) == recovered


def test_admitted_run_without_receipt_cannot_be_relaunched(tmp_path):
    manifest, _ = _manifest(tmp_path)
    first = ModelRunBudget.admit(manifest)
    assert first.is_new

    retry = ModelRunBudget.admit(manifest)
    assert not retry.is_new
    with pytest.raises(ModelRunBudgetError, match="RECOVERY_REQUIRED"):
        retry.recover_terminal(tmp_path / "missing-receipt.json")


def test_unlaunched_run_can_release_reservation(tmp_path):
    manifest, binding = _manifest(tmp_path)
    budget = ModelRunBudget.admit(manifest)
    budget.release_unlaunched()
    released = budget.ledger.get(str(binding["trial_id"]))
    assert released is not None
    assert released.state == "released"


def test_released_model_run_with_lost_receipt_cannot_be_readmitted(tmp_path):
    manifest, binding = _manifest(tmp_path)
    budget = ModelRunBudget.admit(manifest)
    receipt_path = tmp_path / "output" / "execution-receipt.json"
    receipt_path.parent.mkdir()
    receipt = _terminal(binding, state="cancelled_before_launch")
    receipt.update(
        {
            "launch_attempted": False,
            "budget_state": "released",
        }
    )
    receipt.pop("allocated_gpu_hours")
    receipt.pop("allocated_gpu_count")
    budget.persist_terminal(receipt_path, receipt)
    receipt_path.unlink()

    retry = ModelRunBudget.admit(manifest)
    assert not retry.is_new
    with pytest.raises(ModelRunBudgetError, match="RECOVERY_REQUIRED"):
        retry.recover_terminal(receipt_path)


def test_legacy_budget_ledger_migrates_state_check_and_preserves_rows(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-budget.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE trials (
            trial_id TEXT PRIMARY KEY NOT NULL,
            cost_class TEXT NOT NULL,
            estimated_gpu_hours REAL NOT NULL,
            human_approved INTEGER NOT NULL,
            fencing_token INTEGER NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('admitted', 'settled')),
            actual_gpu_hours REAL,
            receipt_ref TEXT,
            plan_sha256 TEXT
        );
        CREATE TABLE policy (
            singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
            total_gpu_hours REAL NOT NULL,
            high_trial_limit INTEGER NOT NULL
        );
        INSERT INTO policy VALUES (1, 4.0, 2);
        INSERT INTO trials VALUES (
            'legacy-admitted', 'low', 1.0, 0, 3, 'admitted', NULL, NULL,
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        );
        INSERT INTO trials VALUES (
            'legacy-settled', 'very_low', 0.5, 0, 2, 'settled', 0.25,
            'cas://sha256/legacy',
            'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        );
        """
    )
    connection.commit()
    connection.close()

    ledger = BudgetLedger(
        database,
        BudgetPolicy(
            total_gpu_hours=4.0,
            high_trial_limit=2,
            max_trial_gpu_hours=120.0,
            require_high_cost_approval=True,
        ),
    )
    admitted = ledger.get("legacy-admitted")
    settled = ledger.get("legacy-settled")
    assert admitted is not None and admitted.state == "admitted"
    assert admitted.fencing_token == 3
    assert settled is not None and settled.state == "settled"
    assert settled.actual_gpu_hours == 0.25
    assert settled.receipt_ref == "cas://sha256/legacy"


def test_terminal_receipt_must_bind_gpu_count_and_claim_boundary(tmp_path):
    manifest, binding = _manifest(tmp_path)
    budget = ModelRunBudget.admit(manifest)
    receipt_path = tmp_path / "receipt.json"
    receipt = _terminal(binding)
    del receipt["allocated_gpu_count"]
    with pytest.raises(ModelRunBudgetError, match="RECEIPT_INVALID"):
        budget.persist_terminal(receipt_path, receipt)
