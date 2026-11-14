"""Goal stops require fresh, settled verifier evidence, using isolated stores.

All measurements below are synthetic CPU fixture inputs. They exercise the real
Judge, SQLite archive and CAS; they do not represent model or GPU experiments.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.storage import canonical_bytes, sha256_bytes
from wmloop.verify.acceptance_policy import verification_policy_from_goal
from wmloop.verify.goal_success import (
    capture_settled_receipt_refs,
    evaluate_iteration_goal_success,
    resolve_frozen_goal,
)
from wmloop.verify.judge import VerificationEvidence, judge


@dataclass
class GoalCase:
    root: Path
    goal: dict
    goal_path: Path
    archive: ArchiveStore
    archive_db: Path
    cas: ContentAddressedStore
    cas_root: Path
    counter: int = 0

    def put(self, document):
        return self.cas.put_bytes(canonical_bytes(document), media_type="application/json").uri

    def settle(self, *, receipt, verdict, proposal_id, goal_id, exploratory=False):
        verdict_ref = self.put(verdict)
        context_ref = self.put({"synthetic_fixture": True, "proposal_id": proposal_id})
        receipt_ref = self.put(receipt)
        self.archive.record_settled_trial(SettledTrialRecord(
            trial_id=proposal_id, proposal_id=proposal_id, goal_id=goal_id,
            library_version="goal-stop-fixture-v1", failure_context_ref=context_ref,
            verdict_ref=verdict_ref, receipt_ref=receipt_ref, gpu_hours=0,
            hypothesis_hash="a" * 64, impl_diff_hash="b" * 64, evaluator_hash="c" * 64,
            settlement_state="settled", receipt_hash=receipt_ref.rsplit("/", 1)[-1],
            exploratory=exploratory,
        ))
        return receipt_ref

    def formal(self, *, effect=1.0, protected=0.0, rewrite=None, exploratory=False):
        self.counter += 1
        proposal = f"formal-{self.counter}"
        policy = verification_policy_from_goal(self.goal)
        evidence = VerificationEvidence(
            proposal_id=proposal, readonly_evaluator_verified=True,
            accept_split_verified=True, extended_horizon_verified=True,
            diff_audit_passed=True, evidence_complete=True,
            accept_metric_deltas={"utility": effect, "protected": protected},
            replication_deltas=(effect,) * 3, replication_seeds=(101, 202, 303),
            action_following_observed=1.0, action_following_threshold=0.5,
        )
        verdict = judge(evidence, policy=policy).to_dict()
        receipt = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-smoke-execution-receipt",
            "state": "ready", "proposal_id": proposal,
            "frozen_goal_ref": self.put(self.goal),
            "verification_policy_ref": self.put(policy.to_dict()),
            "training_config_ref": self.put({"synthetic_cpu_measurements": [effect] * 3}),
        }
        if rewrite is not None:
            rewrite(receipt, verdict)
        ref = self.settle(receipt=receipt, verdict=verdict, proposal_id=proposal,
                          goal_id=self.goal["goal_id"], exploratory=exploratory)
        return ref, receipt, verdict

    def confirm(self, formal_ref, *, campaign="current-iteration", rewrite=None):
        self.counter += 1
        trial = f"confirm-{self.counter}"
        queue = self.root / "pipeline/compiled/queue"
        for directory in (self.root / "inputs", queue / "plans", queue / "runs"):
            directory.mkdir(parents=True, exist_ok=True)
        plan = {"campaign_id": campaign, "stage": "confirm", "trial_id": trial}
        plan_path = queue / "plans/confirm.json"
        plan_path.write_bytes(canonical_bytes(plan))
        digest = sha256_bytes(plan_path.read_bytes())
        verdict = {"verdict": "PASS", "claim_boundary": "Exploratory execution only."}
        context = {"synthetic_fixture": True, "proposal_id": trial}
        receipt = {
            "schema_version": 1, "artifact_type": "verdiwm-auto-experiment-receipt",
            "state": "terminal", "settlement_state": "settled", "stage": "confirm",
            "campaign_id": campaign, "trial_id": trial, "archive_trial_id": trial,
            "plan_sha256": digest, "artifact_refs": {"result.json": self.put({"receipt_ref": formal_ref})},
            "support_refs": {}, "verdict": verdict, "verdict_ref": self.put(verdict),
            "failure_context_ref": self.put(context),
        }
        if rewrite is not None:
            rewrite(receipt)
        ref = self.settle(receipt=receipt, verdict=verdict, proposal_id=trial,
                          goal_id=campaign, exploratory=True)
        manifest = {**receipt, "receipt_ref": ref}
        manifest_path = queue / "runs/confirm.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        (self.root / "inputs/candidate-batch.json").write_bytes(canonical_bytes({"campaign_id": campaign}))
        (queue / "queue.json").write_bytes(canonical_bytes({
            "campaign_id": campaign,
            "selected": [{"candidate_id": "candidate", "stages": [{
                "stage": "confirm", "plan_path": "plans/confirm.json", "plan_sha256": digest,
            }]}],
        }))
        (queue / "execution.json").write_bytes(canonical_bytes({
            "campaign_id": campaign,
            "results": {"candidate:confirm": {"state": "completed", "stage": "confirm",
                                                "manifest_path": str(manifest_path)}},
        }))
        return manifest

    def evaluate(self, *, prior=()):
        return evaluate_iteration_goal_success(
            iteration_root=self.root, archive_db=self.archive_db, cas_root=self.cas_root,
            frozen_goal=self.goal_path, prior_receipt_refs=prior,
        )


@pytest.fixture
def goal_case(tmp_path):
    goal = {
        "schema_version": 2, "artifact_type": "wmloop-goal-spec",
        "claim_boundary": "Synthetic CPU fixture for the frozen goal evidence reader.",
        "goal_id": "goal-stop-fixture", "metric_family": ["quality"],
        "horizons": [16, 32], "primary_objective": "utility",
        "action_following_gate": {"tau_af": 0.5},
        "eval_protocol": {
            "dev_split": "dev", "accept_split": "accept", "accept_horizon_extra": 16,
            "seeds_accept": [101, 202, 303],
            "acceptance_policy": {
                "metric_directions": {"utility": "maximize", "protected": "maximize"},
                "minimum_effect": 0.5, "confidence_interval": "student_t_two_sided_95",
            },
        },
        "budget": {"total_gpu_hours": 1, "per_trial_max_gpu_hours": 0.1}, "envs": ["test"],
    }
    goal_path = tmp_path / "goal.json"
    goal_path.write_bytes(canonical_bytes(goal))
    archive_db = tmp_path / "archive.db"
    cas_root = tmp_path / "store"
    return GoalCase(tmp_path / "iteration", goal, goal_path, ArchiveStore(archive_db),
                    archive_db, ContentAddressedStore(cas_root), cas_root)


def test_current_settled_confirm_and_frozen_judge_accept_confirm_the_goal(goal_case):
    case = goal_case
    prior = capture_settled_receipt_refs(case.archive_db)
    formal_ref, receipt, verdict = case.formal()
    outer = case.confirm(formal_ref)
    result = case.evaluate(prior=prior)
    assert verdict["verdict"] == "ACCEPT"
    assert verdict["replication_statistics"]["lower_bound"] > 0.5
    assert result["state"] == "confirmed"
    assert result["reason"] == "FROZEN_GOAL_CONFIRMED"
    assert {formal_ref, outer["receipt_ref"], receipt["frozen_goal_ref"],
            receipt["verification_policy_ref"]} <= set(result["evidence_refs"])
    for ref in result["evidence_refs"]:
        assert case.cas.read_bytes(ref)


def test_execution_pass_without_formal_receipt_does_not_confirm_goal(goal_case):
    case = goal_case
    case.confirm(case.put({"state": "completed", "verdict": "PASS"}))
    assert case.evaluate()["state"] == "not_confirmed"


def test_old_positive_outer_receipt_cannot_be_repackaged_in_current_queue(goal_case):
    case = goal_case
    formal_ref, _, _ = case.formal()
    old = case.confirm(formal_ref, campaign="previous-iteration")
    case.confirm(formal_ref)
    (case.root / "pipeline/compiled/queue/runs/confirm.json").write_bytes(canonical_bytes(old))
    result = case.evaluate()
    assert result["state"] == "not_confirmed"
    assert result["reason"] == "CONFIRM_ITERATION_BINDING_INVALID"


def test_new_outer_receipt_cannot_promote_an_old_formal_positive(goal_case):
    case = goal_case
    formal_ref, _, _ = case.formal()
    prior = capture_settled_receipt_refs(case.archive_db)
    case.confirm(formal_ref)
    assert formal_ref in prior
    assert case.evaluate(prior=prior)["state"] == "not_confirmed"


@pytest.mark.parametrize("mutation", ["goal_id", "threshold"])
def test_a_different_frozen_goal_cannot_reuse_a_positive_verdict(goal_case, mutation):
    case = goal_case
    formal_ref, _, _ = case.formal()
    case.confirm(formal_ref)
    if mutation == "goal_id":
        case.goal["goal_id"] = "different-goal"
    else:
        case.goal["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0.1
    case.goal_path.write_bytes(canonical_bytes(case.goal))
    assert case.evaluate()["state"] == "not_confirmed"


@pytest.mark.parametrize("member", ["frozen_goal_ref", "verification_policy_ref", "training_config_ref", "verdict_ref", "failure_context_ref"])
def test_missing_declared_evidence_cannot_confirm_goal(goal_case, member):
    case = goal_case
    formal_ref, receipt, _ = case.formal()
    outer = case.confirm(formal_ref)
    ref = receipt.get(member, outer.get(member))
    # A corrupt/missing content object must not be excused by an ACCEPT label.
    digest = ref.rsplit("/", 1)[-1]
    (case.cas_root / "cas" / digest[:2] / digest).unlink()
    assert case.evaluate()["state"] == "not_confirmed"


def test_modified_confirm_plan_is_not_current_evidence(goal_case):
    case = goal_case
    formal_ref, _, _ = case.formal()
    case.confirm(formal_ref)
    plan_path = case.root / "pipeline/compiled/queue/plans/confirm.json"
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    assert case.evaluate() == {
        "state": "not_confirmed", "reason": "CONFIRM_PLAN_BINDING_INVALID", "evidence_refs": [],
    }


@pytest.mark.parametrize("member", ["archive_trial_id", "proposal_id", "verdict_ref", "failure_context_ref"])
def test_formal_receipt_cannot_claim_different_archive_identity_or_refs(goal_case, member):
    case = goal_case
    unrelated = case.put({"unrelated": member})
    value = unrelated if member.endswith("_ref") else "different-trial"
    formal_ref, _, _ = case.formal(rewrite=lambda receipt, _: receipt.update({member: value}))
    case.confirm(formal_ref)
    assert case.evaluate()["state"] == "not_confirmed"


def test_exploratory_archive_scope_cannot_be_promoted_to_goal_success(goal_case):
    case = goal_case
    formal_ref, _, _ = case.formal(exploratory=True)
    case.confirm(formal_ref)
    assert case.evaluate()["state"] == "not_confirmed"


@pytest.mark.parametrize("corruption", [
    "receipt_hash", "trial_receipt", "verdict_projection", "settlement",
    "proposal_goal", "proposal_id", "proposal_library", "missing_proposal",
    "settlement_receipt_hash",
])
def test_corrupt_archive_projections_do_not_grant_goal_authority(goal_case, corruption):
    case = goal_case
    formal_ref, _, _ = case.formal()
    case.confirm(formal_ref)
    with sqlite3.connect(case.archive_db) as connection:
        if corruption == "receipt_hash":
            connection.execute("UPDATE receipts SET receipt_hash=? WHERE receipt_ref=?", ("0" * 64, formal_ref))
        elif corruption == "trial_receipt":
            connection.execute("UPDATE trials SET receipt_ref=? WHERE receipt_ref=?", (case.put({"other": True}), formal_ref))
        elif corruption == "verdict_projection":
            connection.execute("UPDATE verdicts SET verdict_ref=? WHERE trial_id='formal-1'", (case.put({"other": True}),))
        elif corruption == "proposal_goal":
            connection.execute("UPDATE proposals SET goal_id='other' WHERE trial_id='formal-1'")
        elif corruption == "proposal_id":
            connection.execute("UPDATE proposals SET proposal_id='other' WHERE trial_id='formal-1'")
        elif corruption == "proposal_library":
            connection.execute("UPDATE proposals SET library_version='other' WHERE trial_id='formal-1'")
        elif corruption == "missing_proposal":
            connection.execute("DELETE FROM proposals WHERE trial_id='formal-1'")
        elif corruption == "settlement_receipt_hash":
            connection.execute(
                "UPDATE trials SET settlement_json=json_set(settlement_json,'$.receipt_hash',?) WHERE receipt_ref=?",
                ("0" * 64, formal_ref),
            )
        else:
            connection.execute("UPDATE trials SET settlement_json=? WHERE receipt_ref=?", ('{"state":"running"}', formal_ref))
    assert case.evaluate()["state"] == "not_confirmed"


@pytest.mark.parametrize(("effect", "protected"), [(0.1, 0.0), (1.0, -0.1)])
def test_forging_accept_cannot_bypass_effect_threshold_or_protected_metric(goal_case, effect, protected):
    case = goal_case

    def claim_accept(_receipt, verdict):
        verdict.update(verdict="ACCEPT", violation=None, negative_metrics={})
        verdict["gates"] = dict.fromkeys(verdict["gates"], "pass")

    formal_ref, _, _ = case.formal(effect=effect, protected=protected, rewrite=claim_accept)
    case.confirm(formal_ref)
    assert case.evaluate()["state"] == "not_confirmed"


def test_goal_resolution_is_relative_to_evaluator_and_requires_frozen_v2_policy(goal_case):
    case = goal_case
    evaluator = case.goal_path.parent / "evaluator.json"
    evaluator.write_bytes(canonical_bytes({"frozen_goal": case.goal_path.name}))
    assert resolve_frozen_goal(evaluator) == case.goal_path
    case.goal["schema_version"] = 1
    case.goal["eval_protocol"].pop("acceptance_policy")
    case.goal_path.write_bytes(canonical_bytes(case.goal))
    with pytest.raises(ValueError, match="EVOLUTION_FROZEN_GOAL_POLICY_REQUIRED"):
        resolve_frozen_goal(evaluator)


def test_snapshot_of_absent_archive_does_not_create_one(tmp_path):
    database = tmp_path / "absent.db"
    assert capture_settled_receipt_refs(database) == []
    assert not database.exists()


@pytest.mark.parametrize("corruption", ["invalid_sqlite", "missing_receipts_table"])
def test_corrupt_archive_is_an_error_instead_of_an_empty_freshness_baseline(tmp_path, corruption):
    database = tmp_path / "corrupt.db"
    if corruption == "invalid_sqlite":
        database.write_bytes(b"not a sqlite database")
    else:
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(ValueError, match="GOAL_SUCCESS_BASELINE_INVALID"):
        capture_settled_receipt_refs(database)
