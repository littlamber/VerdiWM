"""Explainable ranking of previously tested methods for a new model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .knowledge_graph import _portable


@dataclass(frozen=True)
class TransferAssessment:
    assessment_id: str
    source_method_id: str
    target_model_id: str
    state: str
    score: float
    architecture_similarity: float
    diagnostic_similarity: float
    capability_coverage: float
    reasons: tuple[str, ...]
    risks: tuple[str, ...]
    auto_queue: bool
    source_method_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-transfer-assessment",
            "assessment_id": self.assessment_id,
            "source_method_id": self.source_method_id,
            "source_method_key": self.source_method_key,
            "target_model_id": self.target_model_id,
            "state": self.state,
            "score": self.score,
            "architecture_similarity": self.architecture_similarity,
            "diagnostic_similarity": self.diagnostic_similarity,
            "capability_coverage": self.capability_coverage,
            "reasons": list(self.reasons),
            "risks": list(self.risks),
            "auto_queue": self.auto_queue,
        }


def _set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence):
        return {str(item) for item in value if str(item)}
    return set()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def rank_transfer_candidates(
    state: Any,
    *,
    target_model_id: str,
    architecture_facets: Sequence[str] = (),
    diagnostic_dimensions: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    hooks: Sequence[str] = (),
    limit: int = 50,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Rank methods by diagnosis first, then architecture and capabilities.

    A positive score is only a routing signal.  Every returned candidate still
    requires target-side materialization and held-out verification.
    """
    target_arch = _set(architecture_facets)
    target_diag = _set(diagnostic_dimensions)
    target_caps = _set(capabilities) | _set(hooks)
    assessments: list[TransferAssessment] = []
    for row in state.graph_nodes(kind="method", limit=100000):
        payload = _payload(row)
        method_arch = _set(payload.get("architecture_facets", []))
        method_diag = _set(payload.get("diagnostic_dimensions", payload.get("diagnoses", [])))
        required = _set(payload.get("required_capabilities", []))
        anti = _set(payload.get("anti_conditions", []))
        architecture = _jaccard(method_arch, target_arch)
        diagnosis = _jaccard(method_diag, target_diag)
        coverage = 1.0 if not required else len(required & target_caps) / len(required)
        score = round(0.60 * diagnosis + 0.20 * architecture + 0.20 * coverage, 6)
        reasons: list[str] = []
        risks: list[str] = []
        if diagnosis > 0:
            reasons.append(f"diagnostic overlap {diagnosis:.2f}")
        if architecture > 0:
            reasons.append(f"architecture overlap {architecture:.2f}")
        if coverage < 1.0:
            missing = sorted(required - target_caps)
            risks.append("missing capabilities: " + ", ".join(missing))
            reasons.append("AI materialization required before execution")
        if anti & target_diag:
            risks.append("anti-condition matches: " + ", ".join(sorted(anti & target_diag)))
        historical = str(row.get("status") or payload.get("status") or "unknown")
        if historical in {"harmful", "rejected"}:
            risks.append("historically harmful or rejected in at least one context")
        if historical == "confirmed_positive":
            reasons.append("historically replicated positive")
        state_name = "candidate" if diagnosis > 0 and score >= 0.20 else "abstained"
        auto_queue = state_name == "candidate"
        digest = hashlib.sha256(json.dumps({"method": row["node_id"], "target": target_model_id}, sort_keys=True).encode()).hexdigest()[:16]
        assessment = TransferAssessment(
            assessment_id="assessment-" + digest,
            source_method_id=row["node_id"],
            target_model_id=target_model_id,
            state=state_name,
            score=score,
            architecture_similarity=round(architecture, 6),
            diagnostic_similarity=round(diagnosis, 6),
            capability_coverage=round(coverage, 6),
            reasons=tuple(reasons),
            risks=tuple(risks),
            auto_queue=auto_queue,
            source_method_key=str(row["node_key"]),
        )
        assessments.append(assessment)
        if persist:
            state.put_transfer_assessment(
                assessment.assessment_id,
                source_method_id=assessment.source_method_id,
                target_model_id=target_model_id,
                state=assessment.state,
                score=assessment.score,
                payload=assessment.to_dict(),
            )
    assessments.sort(key=lambda item: (-item.score, item.source_method_id))
    return [item.to_dict() for item in assessments[:limit]]


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("payload_json", "{}")))
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}
