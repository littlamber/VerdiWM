"""Formal acceptance must retain regressions and honor frozen goal rules."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.archive.store import ContentAddressedStore
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.training_smoke.evidence import (
    _verification_evidence_from_probe_replications,
    _write_accept_replicate_inputs,
)
from wmloop.orchestrator import ExecutionOutcome, ResearchLoop
from wmloop.propose.generator import GeneratedProposal, ProposalContext
from wmloop.storage import canonical_bytes, sha256_bytes
from wmloop.verify.acceptance_policy import verification_policy_from_goal
from wmloop.verify.judge import VerificationEvidence, Verdict, judge
from wmloop.verify.statistics import student_t_mean_interval


def _evidence(**changes):
    return replace(VerificationEvidence(
        proposal_id="candidate", readonly_evaluator_verified=True,
        accept_split_verified=True, extended_horizon_verified=True,
        diff_audit_passed=True, evidence_complete=True,
        accept_metric_deltas={"utility": 1.0}, replication_deltas=(1.0, 1.0, 1.0),
        action_following_observed=1.0, action_following_threshold=0.5,
    ), **changes)


def _goal(*, minimum_effect=0.5, directions=None):
    return {
        "schema_version": 2, "artifact_type": "wmloop-goal-spec",
        "claim_boundary": "Frozen rules for a future target experiment.",
        "goal_id": "test_frozen_acceptance_v2", "metric_family": ["quality"],
        "horizons": [16, 32], "primary_objective": "utility",
        "action_following_gate": {"tau_af": 0.5},
        "eval_protocol": {
            "dev_split": "dev", "accept_split": "accept", "accept_horizon_extra": 16,
            "seeds_accept": [101, 202, 303],
            "acceptance_policy": {
                "metric_directions": directions or {"utility": "maximize"},
                "minimum_effect": minimum_effect,
                "confidence_interval": "student_t_two_sided_95",
            },
        },
        "budget": {"total_gpu_hours": 10, "per_trial_max_gpu_hours": 1},
        "envs": ["test"],
    }


def test_positive_utility_cannot_cancel_a_protected_regression():
    result = judge(_evidence(accept_metric_deltas={"utility": 1.0, "protected": -3.0}))
    assert result.verdict == Verdict.REJECT
    assert result.negative_metrics == {"protected": -3.0}
    assert result.delta_m_ver == {"utility": 1.0, "protected": -3.0}
    assert result.violation == "ACCEPT_METRIC_REGRESSION"


def test_zero_protected_delta_does_not_block_improvement():
    assert judge(_evidence(accept_metric_deltas={"utility": 1.0, "protected": 0.0})).verdict == Verdict.ACCEPT


def test_af_failure_retains_regressions_while_denying_metric_credit():
    result = judge(_evidence(
        accept_metric_deltas={"utility": 1.0, "protected": -3.0},
        action_following_observed=0.1,
    ))
    assert result.verdict == Verdict.REJECT
    assert result.violation == "AF_GATE_FAILED"
    assert result.delta_m_ver == {"utility": 0.0, "protected": 0.0}
    assert result.negative_metrics == {"protected": -3.0}


@pytest.mark.parametrize("deltas", [(1.0,), (1.0, 1.0)])
def test_compatibility_path_never_accepts_fewer_than_three_observations(deltas):
    result = judge(_evidence(replication_deltas=deltas))
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.gates["G4_replication"] == "pending"


def test_low_relative_variance_does_not_replace_a_small_sample_confidence_bound():
    result = judge(_evidence(replication_deltas=(0.5, 1.0, 1.5)))
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.replication_statistics["lower_bound"] == pytest.approx(1 - 4.3028 * 0.5 / 3**0.5)
    assert result.replication_statistics["lower_bound"] < 0


def test_many_independent_observations_can_establish_an_effect_with_high_variance():
    result = judge(_evidence(replication_deltas=(0.0, 2.0) * 50))
    assert result.verdict == Verdict.ACCEPT
    assert result.replication_statistics["lower_bound"] > 0
    assert result.replication_statistics["seed_binding"] == "unverified_compatibility"


@pytest.mark.parametrize("effect, expected", [(0.99, Verdict.INCONCLUSIVE), (1.0, Verdict.INCONCLUSIVE), (1.01, Verdict.ACCEPT)])
def test_nonzero_frozen_threshold_is_strict(effect, expected):
    policy = verification_policy_from_goal(_goal(minimum_effect=1.0))
    result = judge(_evidence(
        accept_metric_deltas={"utility": effect}, replication_deltas=(effect,) * 3,
        replication_seeds=(101, 202, 303),
    ), policy=policy)
    assert result.verdict == expected
    assert result.replication_statistics["threshold"] == 1.0
    assert result.replication_statistics["policy_sha256"] == policy.sha256
    assert result.replication_statistics["seed_binding"] == "frozen_goal"


@pytest.mark.parametrize("seeds", [None, (101, 102, 103), (101, 202, 404)])
def test_frozen_policy_requires_its_declared_seed_set(seeds):
    result = judge(_evidence(replication_seeds=seeds), policy=verification_policy_from_goal(_goal()))
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.replication_statistics["seed_binding"] in {"missing", "mismatch"}


@pytest.mark.parametrize("seeds", [(101, 101, 303), (True, 202, 303), (101, 202), (101, 202, -1)])
def test_invalid_or_duplicate_replication_seed_ids_are_rejected(seeds):
    with pytest.raises(ValueError, match="VERIFICATION_EVIDENCE_INVALID"):
        judge(_evidence(replication_seeds=seeds))


def test_policy_normalizes_primary_and_protected_metric_directions():
    policy = verification_policy_from_goal(_goal(directions={"utility": "minimize", "latency": "minimize"}))
    evidence = _evidence(
        accept_metric_deltas={"utility": -1.0, "latency": -2.0},
        replication_deltas=(-1.0,) * 3, replication_seeds=(101, 202, 303),
    )
    assert judge(evidence, policy=policy).verdict == Verdict.ACCEPT
    regressed = judge(replace(evidence, accept_metric_deltas={"utility": -1.0, "latency": 2.0}), policy=policy)
    assert regressed.verdict == Verdict.REJECT
    assert regressed.negative_metrics == {"latency": -2.0}


@pytest.mark.parametrize("enabled, threshold", [(False, None), (True, 0.0)])
def test_executor_cannot_disable_or_lower_frozen_action_following_gate(enabled, threshold):
    result = judge(_evidence(
        replication_seeds=(101, 202, 303), action_following_observed=0.1,
        action_following_gate_enabled=enabled, action_following_threshold=threshold,
    ), policy=verification_policy_from_goal(_goal()))
    assert result.verdict == Verdict.REJECT
    assert result.action_following_gate == {"enabled": True, "threshold": 0.5, "observed": 0.1, "pass": False}


def test_missing_protected_outcome_cannot_be_implicitly_passed():
    policy = verification_policy_from_goal(_goal(directions={"utility": "maximize", "protected": "maximize"}))
    result = judge(_evidence(replication_seeds=(101, 202, 303)), policy=policy)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.violation == "VERIFICATION_METRIC_EVIDENCE_MISSING"


def test_primary_delta_must_match_its_paired_replication_mean():
    result = judge(_evidence(
        accept_metric_deltas={"utility": 10.0}, replication_seeds=(101, 202, 303),
    ), policy=verification_policy_from_goal(_goal()))
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.violation == "VERIFICATION_PRIMARY_REPLICATION_MISMATCH"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 10**400, -0.1])
def test_frozen_policy_rejects_invalid_thresholds(value):
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_INVALID"):
        verification_policy_from_goal(_goal(minimum_effect=value))


@pytest.mark.parametrize("mutation", ["direction", "missing_primary", "duplicate_seed", "confidence"])
def test_frozen_policy_rejects_invalid_declarations(mutation):
    goal = _goal()
    rules = goal["eval_protocol"]["acceptance_policy"]
    if mutation == "direction":
        rules["metric_directions"]["utility"] = "up"
    elif mutation == "missing_primary":
        rules["metric_directions"] = {"other": "maximize"}
    elif mutation == "duplicate_seed":
        goal["eval_protocol"]["seeds_accept"] = [101, 101, 303]
    else:
        rules["confidence_interval"] = "normal_95"
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_INVALID"):
        verification_policy_from_goal(goal)


def test_historical_goal_and_verdict_remain_readable_without_new_policy_fields():
    goal = _goal()
    for key in ("schema_version", "artifact_type", "claim_boundary"):
        del goal[key]
    del goal["eval_protocol"]["acceptance_policy"]
    assert verification_policy_from_goal(goal) is None
    historical = judge(_evidence()).to_dict()
    for key in ("schema_version", "artifact_type", "claim_boundary", "replication_statistics"):
        del historical[key]
    validate_document("verdict", historical)


@pytest.mark.parametrize("version", [None, 1, 3, True, "2"])
def test_acceptance_policy_requires_explicit_supported_goal_version(version):
    goal = _goal()
    if version is None:
        for key in ("schema_version", "artifact_type", "claim_boundary"):
            del goal[key]
    else:
        goal["schema_version"] = version
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_INVALID"):
        verification_policy_from_goal(goal)


def test_v2_goal_cannot_omit_its_acceptance_policy():
    goal = _goal()
    del goal["eval_protocol"]["acceptance_policy"]
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_INVALID"):
        verification_policy_from_goal(goal)


def test_v2_verdict_round_trip_rejects_undeclared_statistics_fields():
    verdict = judge(_evidence()).to_dict()
    validate_document("verdict", verdict)
    verdict["replication_statistics"]["executor_declares_pass"] = True
    with pytest.raises(ContractValidationError):
        validate_document("verdict", verdict)


def test_goal_policy_and_verdict_cas_round_trip_preserves_binding(tmp_path):
    goal = _goal()
    policy = verification_policy_from_goal(goal)
    verdict = judge(_evidence(replication_seeds=(101, 202, 303)), policy=policy).to_dict()
    cas = ContentAddressedStore(tmp_path)
    refs = {}
    for schema, document in (("goal_spec", goal), ("verification_policy", policy.to_dict()), ("verdict", verdict)):
        validate_document(schema, document)
        ref = cas.put_bytes(canonical_bytes(document), media_type="application/json")
        reloaded = json.loads(ref.path.read_bytes())
        validate_document(schema, reloaded)
        assert reloaded == document
        refs[schema] = ref
    assert refs["goal_spec"].sha256 == policy.goal_sha256
    assert refs["verification_policy"].sha256 == policy.sha256
    assert verdict["replication_statistics"]["policy_sha256"] == policy.sha256


def test_historical_freeze_schema_entries_still_match():
    root = Path(__file__).resolve().parents[1]
    frozen_schemas = {"configs/schemas/goal_spec.schema.json", "configs/schemas/verdict.schema.json"}
    matched = 0
    for path in (root / "configs/constitution").glob("*.freeze.json"):
        freeze = json.loads(path.read_text())
        for entry in freeze["entries"]:
            if entry["path"] in frozen_schemas:
                payload = (root / entry["path"]).read_bytes()
                assert sha256_bytes(payload) == entry["sha256"]
                assert len(payload) == entry["size"]
                matched += 1
    assert matched > 0


def test_compiled_policy_owns_its_goal_snapshot():
    goal = _goal()
    policy = verification_policy_from_goal(goal)
    digest = sha256_bytes(canonical_bytes(goal))
    goal["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0.0
    goal["eval_protocol"]["seeds_accept"][0] = 404
    assert policy.minimum_effect == 0.5
    assert policy.seeds_accept == (101, 202, 303)
    assert policy.goal_sha256 == digest


def test_control_plane_freezes_threshold_before_the_provider_can_mutate_the_goal(tmp_path):
    goal = _goal(minimum_effect=2.0)
    frozen_goal_digest = sha256_bytes(canonical_bytes(goal))
    events = []

    class Generator:
        def generate(self, context):
            events.append("generate")
            context.goal_spec["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0.0
            return GeneratedProposal(
                {"proposal_id": "candidate", "budget_estimate_gpu_hours": 0.1,
                 "interventions": [{"primitive": "fake", "params": {}}]},
                "test", (), 1,
            )

    class Executor:
        def execute(self, proposal, fencing_token):
            events.append("execute")
            return ExecutionOutcome(0.1, "cas://sha256/" + "a" * 64,
                                    _evidence(replication_seeds=(101, 202, 303)))

    budget = BudgetLedger(tmp_path / "budget.db", BudgetPolicy(10))
    registry = SimpleNamespace(validate_combination=lambda _: [SimpleNamespace(cost_class="very_low")])
    context = ProposalContext({}, goal, {}, registry)
    result = ResearchLoop(
        proposal_generator=Generator(), budget_ledger=budget, executor=Executor(),
        vendor_verifier=lambda: "frozen-vendor",
    ).run_round(context)
    assert events == ["generate", "execute"]
    assert result.settlement.state == "settled"
    assert result.verdict.verdict == Verdict.INCONCLUSIVE
    assert result.verdict.replication_statistics["threshold"] == 2.0
    assert result.verdict.replication_statistics["policy"]["goal_sha256"] == frozen_goal_digest


def test_supplied_policy_cannot_change_rules_while_claiming_the_same_goal_digest(tmp_path):
    goal = _goal(minimum_effect=2.0)
    policy = replace(verification_policy_from_goal(goal), minimum_effect=0.0)
    loop = ResearchLoop(
        proposal_generator=None, executor=None, vendor_verifier=lambda: "unused",
        budget_ledger=BudgetLedger(tmp_path / "budget.db", BudgetPolicy(10)),
        verification_policy=policy,
    )
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_MISMATCH"):
        loop.run_round(ProposalContext({}, goal, {}, None))


def test_wrong_frozen_goal_is_rejected_before_any_execution(tmp_path):
    goal = _goal()
    policy = verification_policy_from_goal(goal)
    other_goal = copy.deepcopy(goal)
    other_goal["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0
    loop = ResearchLoop(
        proposal_generator=None, executor=None, vendor_verifier=lambda: "unused",
        budget_ledger=BudgetLedger(tmp_path / "budget.db", BudgetPolicy(10)),
        verification_policy=policy,
    )
    with pytest.raises(ValueError, match="VERIFICATION_POLICY_GOAL_MISMATCH"):
        loop.run_round(ProposalContext({}, other_goal, {}, None))


def test_training_evidence_keeps_measured_replication_seed_identity():
    evidence = _verification_evidence_from_probe_replications(
        proposal_id="candidate", diff_audit_passed=True, required_horizons=[16, 32],
        metric_name="utility", action_following_threshold=0.5,
        action_following={"candidate": {"inv_dyn_acc_perframe": 1.0}},
        replications=[{"delta_primary_metric": 1.0, "seed": seed, "required_horizons_complete": True}
                      for seed in (101, 202, 303)],
    )
    assert evidence.replication_seeds == [101, 202, 303]
    assert judge(evidence, policy=verification_policy_from_goal(_goal())).verdict == Verdict.ACCEPT


def test_training_cohort_uses_frozen_seed_ids(monkeypatch, tmp_path):
    selection = SimpleNamespace(trajectory_ids=("first", "second", "third"))
    plan = SimpleNamespace(selection=lambda *_: selection, dataset_freeze_sha256="a" * 64,
                           heldout_protocol_sha256="b" * 64)
    monkeypatch.setattr("wmloop.execute.training_smoke.evidence.build_baseline_evaluation_plan", lambda *_args, **_kwargs: plan)
    records = _write_accept_replicate_inputs(
        output_root=tmp_path, heldout_protocol={}, dataset_freeze={},
        environment="push_cube", max_accept_trajectories=1, replication_count=3,
        replication_seeds=(101, 202, 303),
    )
    assert [record["seed"] for record in records] == [101, 202, 303]
    assert [record["trajectory_ids"] for record in records] == [["first"], ["second"], ["third"]]


@pytest.mark.parametrize("values", [(), (1.0,), (True, 1.0), (float("nan"), 1.0), (10**400, 1.0), (1e308, 1e308)])
def test_shared_interval_rejects_unsupported_or_unrepresentable_samples(values):
    with pytest.raises(ValueError, match="VERIFICATION_STATISTICS_NUMERIC_INVALID"):
        student_t_mean_interval(values)
