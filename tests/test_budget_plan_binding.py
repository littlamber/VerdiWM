from concurrent.futures import ThreadPoolExecutor
import pytest

from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy


def ledger(tmp_path):
    return BudgetLedger(tmp_path / "budget.db", BudgetPolicy(total_gpu_hours=10))


def test_plan_binding_fences_takeover_and_competing_admission(tmp_path):
    budget = ledger(tmp_path)
    digest = "a" * 64
    first = budget.admit("trial", cost_class="low", estimated_gpu_hours=1,
                         plan_sha256=digest, reject_existing=True)
    for wrong in ("b" * 64, None):
        with pytest.raises(BudgetError, match="TRIAL_PLAN_BINDING_MISMATCH"):
            budget.takeover("trial", expected_fencing_token=first.fencing_token, plan_sha256=wrong)
    with pytest.raises(BudgetError, match="TRIAL_ALREADY_EXISTS"):
        budget.admit("trial", cost_class="low", estimated_gpu_hours=1,
                     plan_sha256=digest, reject_existing=True)
    second = budget.takeover("trial", expected_fencing_token=first.fencing_token, plan_sha256=digest)
    with pytest.raises(BudgetError, match="STALE_FENCING_TOKEN"):
        budget.takeover("trial", expected_fencing_token=first.fencing_token, plan_sha256=digest)
    assert second.fencing_token == 2


def test_legacy_unbound_active_fails_closed_but_settlement_remains_readable(tmp_path):
    budget = ledger(tmp_path)
    first = budget.admit("legacy", cost_class="low", estimated_gpu_hours=1)
    with pytest.raises(BudgetError, match="TRIAL_PLAN_BINDING_MISMATCH"):
        budget.takeover("legacy", expected_fencing_token=first.fencing_token, plan_sha256="a" * 64)
    budget.settle("legacy", fencing_token=first.fencing_token, actual_gpu_hours=1,
                  receipt_ref="cas://sha256/" + "b" * 64)
    assert budget.get("legacy").receipt_ref == "cas://sha256/" + "b" * 64


def test_released_attempt_can_retry_only_with_same_identity_and_new_fence(tmp_path):
    budget = ledger(tmp_path)
    digest = "a" * 64
    first = budget.admit(
        "retryable",
        cost_class="low",
        estimated_gpu_hours=1,
        plan_sha256=digest,
        reject_existing=True,
    )
    budget.release("retryable", fencing_token=first.fencing_token)

    with pytest.raises(BudgetError, match="TRIAL_PLAN_BINDING_MISMATCH"):
        budget.admit(
            "retryable",
            cost_class="low",
            estimated_gpu_hours=1,
            plan_sha256="b" * 64,
            reject_existing=True,
        )
    with pytest.raises(BudgetError, match="TRIAL_IDENTITY_CONFLICT"):
        budget.admit(
            "retryable",
            cost_class="medium",
            estimated_gpu_hours=1,
            plan_sha256=digest,
            reject_existing=True,
        )

    retry = budget.admit(
        "retryable",
        cost_class="low",
        estimated_gpu_hours=1,
        plan_sha256=digest,
        reject_existing=True,
    )
    assert retry.state == "admitted"
    assert retry.fencing_token == first.fencing_token + 1

    with pytest.raises(BudgetError, match="STALE_FENCING_TOKEN"):
        budget.settle(
            "retryable",
            fencing_token=first.fencing_token,
            actual_gpu_hours=0.5,
            receipt_ref="cas://sha256/" + "c" * 64,
            plan_sha256=digest,
        )
    settled = budget.settle(
        "retryable",
        fencing_token=retry.fencing_token,
        actual_gpu_hours=0.5,
        receipt_ref="cas://sha256/" + "c" * 64,
        plan_sha256=digest,
    )
    assert settled.state == "settled"


def test_concurrent_first_admissions_have_only_one_winner(tmp_path):
    budget = ledger(tmp_path)
    def admit():
        try:
            return budget.admit("trial", cost_class="low", estimated_gpu_hours=1,
                                plan_sha256="a" * 64, reject_existing=True).fencing_token
        except BudgetError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: admit(), range(2)))
    assert sorted(results, key=str) == [1, "TRIAL_ALREADY_EXISTS"]
