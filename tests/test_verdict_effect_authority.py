"""Compatibility verdicts cannot become verified archive or audit gains."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.settlement import _primary_delta, _record_formal_settled_trial
from wmloop.storage import canonical_bytes
from wmloop.verify.acceptance_policy import VerificationPolicy, has_frozen_policy_binding
from wmloop.verify.judge import VerificationEvidence, judge
from wmloop.verify.m3_acceptance_audit import _model_quality_summary


@pytest.fixture
def bound(tmp_path):
    cas = ContentAddressedStore(tmp_path / "cas")
    goal = cas.put_bytes(canonical_bytes({"fixture": "frozen-goal"}), media_type="application/json")
    policy = VerificationPolicy(
        primary_metric="utility", metric_directions=(("protected", "maximize"), ("utility", "maximize")),
        minimum_effect=0.5, seeds_accept=(101, 202, 303), goal_sha256=goal.sha256,
        action_following_threshold=0.5,
    )
    policy_ref = cas.put_bytes(canonical_bytes(policy.to_dict()), media_type="application/json")
    evidence = VerificationEvidence(
        proposal_id="candidate", readonly_evaluator_verified=True,
        accept_split_verified=True, extended_horizon_verified=True,
        diff_audit_passed=True, evidence_complete=True,
        accept_metric_deltas={"protected": 0.0, "utility": 1.0},
        replication_deltas=(1.0, 1.0, 1.0), replication_seeds=(101, 202, 303),
        action_following_observed=1.0, action_following_threshold=0.5,
    )
    verdict = judge(evidence, policy=policy).to_dict()
    receipt = {
        "candidate": {"worktree_diff_sha256": "d" * 64},
        "frozen_goal_ref": goal.uri, "verification_policy_ref": policy_ref.uri,
    }
    return {"cas": cas, "archive": ArchiveStore(tmp_path / "archive.db"),
            "policy": policy, "evidence": evidence, "verdict": verdict, "receipt": receipt}


def _record(bound, *, verdict=None, receipt=None):
    verdict = bound["verdict"] if verdict is None else verdict
    receipt = bound["receipt"] if receipt is None else receipt
    cas = bound["cas"]
    refs = [cas.put_bytes(canonical_bytes(item), media_type="application/json").uri
            for item in (receipt, {"failure": "context"}, verdict)]
    _record_formal_settled_trial(
        archive=bound["archive"],
        proposal={"proposal_id": "candidate", "goal_id": "goal", "env": "test",
                  "interventions": [{"layer": "train", "primitive": "anchor", "params": {}}]},
        verdict=verdict, receipt=receipt, receipt_ref=refs[0], failure_context_ref=refs[1],
        verdict_ref=refs[2], gpu_hours=0.0, settlement_state="settled",
        round_start_verification={"evaluator_freeze": {"sha256": "e" * 64}},
    )
    return refs[2]


def _report(bound):
    verdict = bound["verdict"]
    return {**verdict, "state": "ready", "verdict_schema_version": 2, "receipt": bound["receipt"]}


def _damage(bound, damage):
    statistics = bound["verdict"]["replication_statistics"]
    if damage == "policy_missing":
        statistics["policy"] = None
    elif damage == "policy_hash":
        statistics["policy_sha256"] = "f" * 64
    elif damage == "policy_changed":
        statistics["policy"]["minimum_effect"] = 0.0
    elif damage == "seed_binding":
        statistics["seed_binding"] = "unverified_compatibility"
    elif damage == "seeds":
        statistics["replication_seeds"] = [101, 202, 999]
    elif damage == "boolean_seed":
        statistics["replication_seeds"] = [True, 202, 303]
    elif damage == "sample_count":
        statistics["sample_count"] = 4
    elif damage == "minimum_sample_count":
        statistics["minimum_sample_count"] = 4
    elif damage == "threshold":
        statistics["threshold"] = 0.0
    elif damage == "confidence_interval":
        statistics["confidence_interval"] = "unverified"
    elif damage == "goal_ref":
        bound["receipt"]["frozen_goal_ref"] = "cas://sha256/" + "f" * 64
    elif damage == "policy_ref":
        bound["receipt"]["verification_policy_ref"] = "cas://sha256/" + "f" * 64
    elif damage == "goal_ref_missing":
        bound["receipt"].pop("frozen_goal_ref")
    else:
        bound["receipt"].pop("verification_policy_ref")


@pytest.mark.parametrize("damage", [
    "policy_missing", "policy_hash", "policy_changed", "seed_binding", "seeds", "boolean_seed",
    "sample_count", "minimum_sample_count", "threshold", "confidence_interval",
    "goal_ref", "policy_ref", "goal_ref_missing", "policy_ref_missing",
])
def test_accept_without_frozen_binding_is_exploratory_and_fails_audit(bound, damage):
    _damage(bound, damage)
    assert not has_frozen_policy_binding(bound["verdict"]["replication_statistics"], receipt=bound["receipt"])
    _record(bound)
    assert bound["archive"].visible_settled_trials() == ("candidate",)
    assert not bound["archive"].list_cells()
    assert _model_quality_summary(_report(bound))["passed"] is False


def test_compatibility_accept_stays_archived_without_verified_gain(bound):
    compatibility = judge(bound["evidence"]).to_dict()
    assert compatibility["verdict"] == "ACCEPT"
    _record(bound, verdict=compatibility)
    assert bound["archive"].visible_settled_trials() == ("candidate",)
    assert not bound["archive"].list_cells()
    report = dict(_report(bound), replication_statistics=compatibility["replication_statistics"])
    assert _model_quality_summary(report)["passed"] is False


@pytest.mark.parametrize("gain,expected_verdict", [(1.0, "ACCEPT"), (0.0, "REJECT"), (-1.0, "REJECT")])
def test_bound_positive_null_and_negative_results_remain_settleable(bound, gain, expected_verdict):
    evidence = replace(bound["evidence"], accept_metric_deltas={"protected": 0.0, "utility": gain},
                       replication_deltas=(gain, gain, gain))
    verdict = judge(evidence, policy=bound["policy"]).to_dict()
    assert verdict["verdict"] == expected_verdict
    _record(bound, verdict=verdict)
    cells = bound["archive"].list_cells()
    assert len(cells) == 1
    # The protected metric deliberately precedes utility in the delta mapping.
    assert cells[0].stats.mean_verified_improvement == gain


def test_bound_but_inconclusive_evidence_does_not_update_formal_cells(bound):
    evidence = replace(bound["evidence"], replication_deltas=(-1.0, 1.0, 3.0))
    verdict = judge(evidence, policy=bound["policy"]).to_dict()
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert has_frozen_policy_binding(verdict["replication_statistics"], receipt=bound["receipt"])
    _record(bound, verdict=verdict)
    assert not bound["archive"].list_cells()


@pytest.mark.parametrize("gain", [1.0, 0.0, -1.0])
def test_protected_regression_rejects_without_positive_scheduler_credit(bound, gain):
    evidence = replace(bound["evidence"], accept_metric_deltas={"protected": -3.0, "utility": gain},
                       replication_deltas=(gain, gain, gain))
    verdict = judge(evidence, policy=bound["policy"]).to_dict()
    assert verdict["verdict"] == "REJECT"
    assert verdict["violation"] == "ACCEPT_METRIC_REGRESSION"
    original = deepcopy(verdict)
    verdict_ref = _record(bound, verdict=verdict)
    cells = bound["archive"].list_cells()
    assert len(cells) == 1
    assert cells[0].stats.mean_verified_improvement == min(0.0, gain)
    assert verdict == original
    archived = json.loads(bound["cas"].read_bytes(verdict_ref))
    assert archived == original
    assert archived["delta_m_ver"] == {"protected": -3.0, "utility": gain}
    assert archived["negative_metrics"]["protected"] == -3.0


@pytest.mark.parametrize("primary", [None, True, float("nan"), float("inf"), 10**400])
def test_primary_metric_cannot_fall_back_to_a_protected_value(bound, primary):
    verdict = deepcopy(bound["verdict"])
    verdict["delta_m_ver"] = {"protected": 99.0}
    if primary is not None:
        verdict["delta_m_ver"]["utility"] = primary
    with pytest.raises(OrchestratorTrainingEvalSmokeError, match="FORMAL_DELTA_INVALID"):
        _primary_delta(verdict)
    assert not bound["archive"].visible_settled_trials()


def test_bound_accept_passes_audit_while_version_metadata_cannot_be_removed(bound):
    report = _report(bound)
    assert _model_quality_summary(report)["passed"] is True
    report.pop("verdict_schema_version")
    assert _model_quality_summary(report)["passed"] is False


def test_unversioned_historical_settlement_and_report_keep_their_existing_semantics(bound):
    historical = {key: value for key, value in bound["verdict"].items()
                  if key not in {"schema_version", "artifact_type", "claim_boundary", "replication_statistics"}}
    _record(bound, verdict=historical, receipt={"candidate": {"worktree_diff_sha256": "d" * 64}})
    assert bound["archive"].list_cells()[0].stats.mean_verified_improvement == 0.0
    assert _model_quality_summary({"state": "ready", "verdict": "ACCEPT"})["passed"] is True


def test_unversioned_historical_reject_keeps_existing_scheduler_projection(bound):
    historical = {key: value for key, value in bound["verdict"].items()
                  if key not in {"schema_version", "artifact_type", "claim_boundary", "replication_statistics"}}
    historical.update(verdict="REJECT", delta_m_ver={"utility": 1.0, "protected": -3.0})
    _record(bound, verdict=historical, receipt={"candidate": {"worktree_diff_sha256": "d" * 64}})
    assert bound["archive"].list_cells()[0].stats.mean_verified_improvement == 1.0
