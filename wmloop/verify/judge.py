"""Fail-closed four-gate verifier for formal improvement claims."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.verify.acceptance_policy import CONFIDENCE_INTERVAL, VerificationPolicy
from wmloop.verify.statistics import student_t_mean_interval


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    VOID = "VOID"


@dataclass(frozen=True)
class VerificationEvidence:
    """Measured candidate-minus-control contrasts, before policy directions.

    Historical callers supply improvement-oriented deltas. An explicit frozen
    policy supplies each metric's direction and declares the primary utility.
    """

    proposal_id: str
    readonly_evaluator_verified: bool
    accept_split_verified: bool
    extended_horizon_verified: bool
    diff_audit_passed: bool
    evidence_complete: bool
    accept_metric_deltas: Mapping[str, float]
    replication_deltas: tuple[float, ...] | list[float]
    action_following_observed: float | None
    action_following_threshold: float | None
    action_following_gate_enabled: bool = True
    replication_seeds: tuple[int, ...] | list[int] | None = None


@dataclass(frozen=True)
class JudgeResult:
    proposal_id: str
    gates: Mapping[str, str]
    action_following_gate: Mapping[str, object]
    delta_m_ver: Mapping[str, float]
    negative_metrics: Mapping[str, float]
    verdict: Verdict
    violation: str | None
    replication_statistics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "artifact_type": "wmloop-verdict",
            "claim_boundary": (
                "Compatibility evidence does not certify a predeclared goal "
                "threshold or matched seeds; formal acceptance requires the "
                "recorded frozen goal policy and seed binding."
            ),
            "proposal_id": self.proposal_id,
            "gates": dict(self.gates),
            "action_following_gate": dict(self.action_following_gate),
            "delta_m_ver": dict(self.delta_m_ver),
            "negative_metrics": dict(self.negative_metrics),
            "verdict": self.verdict.value,
            "violation": self.violation,
            "replication_statistics": dict(self.replication_statistics),
        }


def judge(evidence: VerificationEvidence, *, policy: VerificationPolicy | None = None) -> JudgeResult:
    """Return the only verdict constructor permitted to certify improvement."""

    _validate_evidence(evidence)
    deltas = {name: float(value) for name, value in evidence.accept_metric_deltas.items()}
    replication_deltas = tuple(float(value) for value in evidence.replication_deltas)
    policy_metrics_complete = True
    if policy is not None:
        if not isinstance(policy, VerificationPolicy):
            raise ValueError("VERIFICATION_POLICY_INVALID")
        directions = dict(policy.metric_directions)
        if set(deltas) - set(directions):
            raise ValueError("VERIFICATION_METRIC_POLICY_MISMATCH")
        policy_metrics_complete = set(deltas) == set(directions)
        deltas = {name: value if directions[name] == "maximize" else -value for name, value in deltas.items()}
        if directions[policy.primary_metric] == "minimize":
            replication_deltas = tuple(-value for value in replication_deltas)
    negatives = {name: value for name, value in deltas.items() if value < 0}
    g1 = "pass" if evidence.readonly_evaluator_verified else "fail"
    g2 = "pass" if evidence.accept_split_verified and evidence.extended_horizon_verified else "fail"
    g3 = "pass" if evidence.diff_audit_passed else "fail"
    g4, replication_statistics = _replication_gate(
        replication_deltas, replication_seeds=evidence.replication_seeds, policy=policy,
    )
    gates = {"G1_readonly": g1, "G2_heldout": g2, "G3_audit": g3, "G4_replication": g4}
    action_gate = _action_following_gate(evidence, policy=policy)
    credited_deltas = deltas
    violation = None
    if g1 == "fail":
        verdict, violation = Verdict.VOID, "G1_READONLY_VIOLATION"
    elif g3 == "fail":
        verdict, violation = Verdict.VOID, "G3_DIFF_AUDIT_VIOLATION"
    elif action_gate["enabled"] and action_gate["observed"] is None:
        verdict, violation = Verdict.INCONCLUSIVE, "AF_GATE_EVIDENCE_MISSING"
    elif action_gate["enabled"] and not action_gate["pass"]:
        # A static or action-ignoring degeneration must never buy metric credit.
        credited_deltas = {name: 0.0 for name in deltas}
        verdict, violation = Verdict.REJECT, "AF_GATE_FAILED"
    elif not evidence.evidence_complete or g2 != "pass" or g4 == "pending":
        verdict = Verdict.INCONCLUSIVE
    elif not policy_metrics_complete:
        verdict, violation = Verdict.INCONCLUSIVE, "VERIFICATION_METRIC_EVIDENCE_MISSING"
    elif negatives:
        verdict, violation = Verdict.REJECT, "ACCEPT_METRIC_REGRESSION"
    elif not any(value > 0 for value in deltas.values()):
        verdict = Verdict.REJECT
    elif policy is not None and not _primary_mean_matches(deltas, replication_statistics, policy):
        verdict, violation = Verdict.INCONCLUSIVE, "VERIFICATION_PRIMARY_REPLICATION_MISMATCH"
    elif g4 != "pass":
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.ACCEPT
    result = JudgeResult(
        proposal_id=evidence.proposal_id,
        gates=gates,
        action_following_gate=action_gate,
        delta_m_ver=credited_deltas,
        negative_metrics=negatives,
        verdict=verdict,
        violation=violation,
        replication_statistics=replication_statistics,
    )
    try:
        validate_document("verdict", result.to_dict())
    except ContractValidationError as exc:
        raise ValueError(f"VERDICT_CONTRACT_INVALID:{exc}") from exc
    return result


def _replication_gate(
    deltas: tuple[float, ...], *, replication_seeds: tuple[int, ...] | list[int] | None,
    policy: VerificationPolicy | None,
) -> tuple[str, dict[str, object]]:
    threshold = policy.minimum_effect if policy is not None else 0.0
    summary: dict[str, object] = {
        "confidence_interval": CONFIDENCE_INTERVAL,
        "sample_count": len(deltas),
        "minimum_sample_count": len(policy.seeds_accept) if policy is not None else 3,
        "threshold": threshold,
        "replication_seeds": list(replication_seeds) if replication_seeds is not None else None,
        "seed_binding": "unverified_compatibility",
        "policy": policy.to_dict() if policy is not None else None,
        "policy_sha256": policy.sha256 if policy is not None else None,
    }
    if len(deltas) < 3:
        return "pending", summary
    try:
        summary.update(student_t_mean_interval(deltas))
    except ValueError as exc:
        raise ValueError("VERIFICATION_EVIDENCE_INVALID") from exc
    if policy is not None:
        if replication_seeds is None:
            summary["seed_binding"] = "missing"
            return "pending", summary
        if set(replication_seeds) != set(policy.seeds_accept):
            summary["seed_binding"] = "mismatch"
            return "fail", summary
        summary["seed_binding"] = "frozen_goal"
    return ("pass" if summary["lower_bound"] > threshold else "fail"), summary


def _primary_mean_matches(
    deltas: Mapping[str, float], summary: Mapping[str, object], policy: VerificationPolicy,
) -> bool:
    return math.isclose(deltas[policy.primary_metric], summary["mean"], rel_tol=1e-9, abs_tol=1e-12)


def _action_following_gate(
    evidence: VerificationEvidence, *, policy: VerificationPolicy | None,
) -> dict[str, object]:
    if policy is None and not evidence.action_following_gate_enabled:
        return {"enabled": False, "threshold": None, "observed": None, "pass": True}
    observed = evidence.action_following_observed
    threshold = policy.action_following_threshold if policy is not None else evidence.action_following_threshold
    if observed is None or threshold is None:
        return {"enabled": True, "threshold": threshold, "observed": observed, "pass": False}
    return {"enabled": True, "threshold": threshold, "observed": observed, "pass": observed >= threshold}


def _validate_evidence(evidence: VerificationEvidence) -> None:
    if not evidence.proposal_id or not evidence.accept_metric_deltas:
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    boolean_gates = (
        evidence.readonly_evaluator_verified,
        evidence.accept_split_verified,
        evidence.extended_horizon_verified,
        evidence.diff_audit_passed,
        evidence.evidence_complete,
        evidence.action_following_gate_enabled,
    )
    if any(type(value) is not bool for value in boolean_gates):
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    numeric = [*evidence.accept_metric_deltas.values(), *evidence.replication_deltas]
    try:
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in numeric):
            raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    except OverflowError as exc:
        raise ValueError("VERIFICATION_EVIDENCE_INVALID") from exc
    if evidence.replication_seeds is not None and (
        not isinstance(evidence.replication_seeds, (list, tuple))
        or len(evidence.replication_seeds) != len(evidence.replication_deltas)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in evidence.replication_seeds)
        or len(set(evidence.replication_seeds)) != len(evidence.replication_seeds)
    ):
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    for value in (evidence.action_following_observed, evidence.action_following_threshold):
        try:
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError("VERIFICATION_EVIDENCE_INVALID")
        except OverflowError as exc:
            raise ValueError("VERIFICATION_EVIDENCE_INVALID") from exc
