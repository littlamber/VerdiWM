"""Fail-closed four-gate verifier for formal improvement claims."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    VOID = "VOID"


@dataclass(frozen=True)
class VerificationEvidence:
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


@dataclass(frozen=True)
class JudgeResult:
    proposal_id: str
    gates: Mapping[str, str]
    action_following_gate: Mapping[str, object]
    delta_m_ver: Mapping[str, float]
    negative_metrics: Mapping[str, float]
    verdict: Verdict
    violation: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "gates": dict(self.gates),
            "action_following_gate": dict(self.action_following_gate),
            "delta_m_ver": dict(self.delta_m_ver),
            "negative_metrics": dict(self.negative_metrics),
            "verdict": self.verdict.value,
            "violation": self.violation,
        }


def judge(evidence: VerificationEvidence) -> JudgeResult:
    """Return the only verdict constructor permitted to certify improvement."""

    _validate_evidence(evidence)
    deltas = {name: float(value) for name, value in evidence.accept_metric_deltas.items()}
    g1 = "pass" if evidence.readonly_evaluator_verified else "fail"
    g2 = "pass" if evidence.accept_split_verified and evidence.extended_horizon_verified else "fail"
    g3 = "pass" if evidence.diff_audit_passed else "fail"
    g4 = _replication_gate(evidence.replication_deltas)
    gates = {"G1_readonly": g1, "G2_heldout": g2, "G3_audit": g3, "G4_replication": g4}
    action_gate = _action_following_gate(evidence)
    if g1 == "fail":
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.VOID, "G1_READONLY_VIOLATION")
    elif g3 == "fail":
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.VOID, "G3_DIFF_AUDIT_VIOLATION")
    elif action_gate["enabled"] and action_gate["observed"] is None:
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.INCONCLUSIVE, "AF_GATE_EVIDENCE_MISSING")
    elif action_gate["enabled"] and not action_gate["pass"]:
        # A static or action-ignoring degeneration must never buy metric credit.
        result = _result(
            evidence.proposal_id,
            gates,
            action_gate,
            {name: 0.0 for name in deltas},
            Verdict.REJECT,
            "AF_GATE_FAILED",
        )
    elif not evidence.evidence_complete or g2 != "pass" or g4 == "pending":
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.INCONCLUSIVE, None)
    elif not any(value > 0 for value in deltas.values()):
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.REJECT, None)
    elif g4 != "pass":
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.INCONCLUSIVE, None)
    else:
        result = _result(evidence.proposal_id, gates, action_gate, deltas, Verdict.ACCEPT, None)
    try:
        validate_document("verdict", result.to_dict())
    except ContractValidationError as exc:
        raise ValueError(f"VERDICT_CONTRACT_INVALID:{exc}") from exc
    return result


def _replication_gate(deltas: tuple[float, ...] | list[float]) -> str:
    values = tuple(float(value) for value in deltas)
    if len(values) < 3:
        return "pending"
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    return "pass" if mean > 0 and deviation <= mean * 0.5 else "fail"


def _action_following_gate(evidence: VerificationEvidence) -> dict[str, object]:
    if not evidence.action_following_gate_enabled:
        return {"enabled": False, "threshold": None, "observed": None, "pass": True}
    observed = evidence.action_following_observed
    threshold = evidence.action_following_threshold
    if observed is None or threshold is None:
        return {"enabled": True, "threshold": threshold, "observed": observed, "pass": False}
    return {"enabled": True, "threshold": threshold, "observed": observed, "pass": observed >= threshold}


def _validate_evidence(evidence: VerificationEvidence) -> None:
    if not evidence.proposal_id or not evidence.accept_metric_deltas:
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    numeric = [*evidence.accept_metric_deltas.values(), *evidence.replication_deltas]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in numeric):
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    if not isinstance(evidence.action_following_gate_enabled, bool):
        raise ValueError("VERIFICATION_EVIDENCE_INVALID")
    for value in (evidence.action_following_observed, evidence.action_following_threshold):
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError("VERIFICATION_EVIDENCE_INVALID")


def _result(
    proposal_id: str,
    gates: Mapping[str, str],
    action_following_gate: Mapping[str, object],
    deltas: Mapping[str, float],
    verdict: Verdict,
    violation: str | None,
) -> JudgeResult:
    return JudgeResult(
        proposal_id=proposal_id,
        gates=dict(gates),
        action_following_gate=dict(action_following_gate),
        delta_m_ver=dict(deltas),
        negative_metrics={},
        verdict=verdict,
        violation=violation,
    )
