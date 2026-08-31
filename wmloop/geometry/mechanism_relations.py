"""Typed, evidence-gated relationships between reusable mechanisms.

The relation record is deliberately separate from an individual intervention
effect.  An interaction is a claim about a *pair* (or ordered composition),
so it needs the single-mechanism counterfactuals, the combined result, and its
own verification state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.evidence_ir import is_content_addressed
from wmloop.geometry.types import GeometryValidationError


RELATION_TYPES = {
    "positive_synergy",
    "antagonism",
    "redundancy",
    "conditional_compatibility",
    "sequential_dependency",
    "substitution",
}
COMPOSITION_OPERATORS = {"parallel", "sequential", "gated", "conditional"}
VERIFICATION_STATES = {"candidate", "screened", "confirmed", "rejected", "abstained"}
_AUTHORITATIVE_STATES = {"confirmed", "rejected"}


@dataclass(frozen=True)
class MechanismRelation:
    """A portable claim about the relationship between two mechanisms.

    Effect values use the normalized convention that larger means better.  The
    interaction effect is ``combined - source - target + baseline``.
    """

    relation_id: str
    source_mechanism_id: str
    target_mechanism_id: str
    relation_type: str
    composition_operator: str
    condition_set: tuple[str, ...]
    anti_conditions: tuple[str, ...]
    baseline_effect: float
    source_effect: float
    target_effect: float
    combined_effect: float
    interaction_effect: float
    uncertainty: float
    replication_count: int
    required_ablations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    claim_scope: str
    verification_state: str
    validity_gates: Mapping[str, bool]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relation_id or not self.source_mechanism_id or not self.target_mechanism_id:
            raise GeometryValidationError("MECHANISM_RELATION_IDENTITY_INVALID")
        if self.source_mechanism_id == self.target_mechanism_id:
            raise GeometryValidationError("MECHANISM_RELATION_SELF_REFERENCE")
        if self.relation_type not in RELATION_TYPES:
            raise GeometryValidationError("MECHANISM_RELATION_TYPE_INVALID")
        if self.composition_operator not in COMPOSITION_OPERATORS:
            raise GeometryValidationError("MECHANISM_RELATION_OPERATOR_INVALID")
        if self.verification_state not in VERIFICATION_STATES:
            raise GeometryValidationError("MECHANISM_RELATION_STATE_INVALID")
        if not self.claim_scope or not self.validity_gates or not self.evidence_refs:
            raise GeometryValidationError("MECHANISM_RELATION_EVIDENCE_MISSING")
        if self.uncertainty < 0 or self.replication_count < 1:
            raise GeometryValidationError("MECHANISM_RELATION_UNCERTAINTY_INVALID")
        for value in (
            self.baseline_effect,
            self.source_effect,
            self.target_effect,
            self.combined_effect,
            self.interaction_effect,
            self.uncertainty,
        ):
            if not math.isfinite(float(value)):
                raise GeometryValidationError("MECHANISM_RELATION_VALUE_INVALID")
        for value in (*self.condition_set, *self.anti_conditions, *self.required_ablations, *self.evidence_refs):
            if not isinstance(value, str) or not value.strip():
                raise GeometryValidationError("MECHANISM_RELATION_TEXT_INVALID")
        if any(not isinstance(value, bool) for value in self.validity_gates.values()):
            raise GeometryValidationError("MECHANISM_RELATION_GATE_INVALID")
        if self.verification_state in _AUTHORITATIVE_STATES:
            if not all(self.validity_gates.values()):
                raise GeometryValidationError("MECHANISM_RELATION_GATE_FAILED")
            if not self.required_ablations:
                raise GeometryValidationError("MECHANISM_RELATION_ABLATION_MISSING")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for field in ("condition_set", "anti_conditions", "required_ablations", "evidence_refs", "notes"):
            payload[field] = list(payload[field])
        payload["validity_gates"] = dict(self.validity_gates)
        payload.update({"schema_version": 1, "artifact_type": "verdiwm-mechanism-relation"})
        return payload


def interaction_effect(*, baseline: float, source: float, target: float, combined: float) -> float:
    """Return the additive interaction contrast on the normalized benefit scale."""

    values = (baseline, source, target, combined)
    if any(not math.isfinite(float(value)) for value in values):
        raise GeometryValidationError("MECHANISM_RELATION_VALUE_INVALID")
    return float(combined) - float(source) - float(target) + float(baseline)


def classify_interaction(
    *,
    baseline: float,
    source: float,
    target: float,
    combined: float,
    synergy_threshold: float = 0.0,
    uncertainty: float = 0.0,
) -> tuple[str, float]:
    """Classify a measured pair while accounting for uncertainty.

    The threshold is an application-level minimum practical effect.  A margin
    smaller than the uncertainty is intentionally returned as ``abstained``
    rather than being over-interpreted.
    """

    if synergy_threshold < 0 or uncertainty < 0:
        raise GeometryValidationError("MECHANISM_RELATION_THRESHOLD_INVALID")
    effect = interaction_effect(baseline=baseline, source=source, target=target, combined=combined)
    if abs(effect) <= uncertainty:
        return "abstained", effect
    if effect > synergy_threshold:
        return "positive_synergy", effect
    if effect < -synergy_threshold:
        return "antagonism", effect
    return "redundancy", effect


def build_mechanism_relation(
    *,
    source_mechanism_id: str,
    target_mechanism_id: str,
    relation_type: str,
    composition_operator: str,
    baseline_effect: float,
    source_effect: float,
    target_effect: float,
    combined_effect: float,
    uncertainty: float,
    replication_count: int,
    required_ablations: Sequence[str],
    evidence_refs: Sequence[str],
    condition_set: Sequence[str] = (),
    anti_conditions: Sequence[str] = (),
    claim_scope: str = "ranking_only",
    verification_state: str = "candidate",
    validity_gates: Mapping[str, bool] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, object]:
    """Build and validate a content-addressable relationship artifact."""

    computed = interaction_effect(
        baseline=baseline_effect,
        source=source_effect,
        target=target_effect,
        combined=combined_effect,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-mechanism-relation",
        "source_mechanism_id": source_mechanism_id,
        "target_mechanism_id": target_mechanism_id,
        "relation_type": relation_type,
        "composition_operator": composition_operator,
        "condition_set": list(condition_set),
        "anti_conditions": list(anti_conditions),
        "baseline_effect": baseline_effect,
        "source_effect": source_effect,
        "target_effect": target_effect,
        "combined_effect": combined_effect,
        "interaction_effect": computed,
        "uncertainty": uncertainty,
        "replication_count": replication_count,
        "required_ablations": list(required_ablations),
        "evidence_refs": list(evidence_refs),
        "claim_scope": claim_scope,
        "verification_state": verification_state,
        "validity_gates": dict(validity_gates or {"evidence": True}),
        "notes": list(notes),
    }
    body["relation_id"] = "mechanism-relation-" + hashlib.sha256(_canonical_json(body)).hexdigest()[:24]
    validate_mechanism_relation(body)
    return body


def propose_mechanism_relation(
    *,
    source_mechanism_id: str,
    target_mechanism_id: str,
    composition_operator: str,
    baseline_effect: float,
    source_effect: float,
    target_effect: float,
    combined_effect: float,
    uncertainty: float,
    replication_count: int,
    required_ablations: Sequence[str],
    evidence_refs: Sequence[str],
    synergy_threshold: float = 0.0,
    condition_set: Sequence[str] = (),
    anti_conditions: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> dict[str, object]:
    """Create a non-authoritative candidate from a four-cell comparison.

    This is the product-facing entry point for automatic relation discovery.
    It classifies the observed contrast but deliberately leaves the candidate
    in ``candidate`` state until the experiment/evaluator pipeline promotes it.
    """

    relation_type, _ = classify_interaction(
        baseline=baseline_effect,
        source=source_effect,
        target=target_effect,
        combined=combined_effect,
        synergy_threshold=synergy_threshold,
        uncertainty=uncertainty,
    )
    if relation_type == "abstained":
        relation_type = "conditional_compatibility"
    return build_mechanism_relation(
        source_mechanism_id=source_mechanism_id,
        target_mechanism_id=target_mechanism_id,
        relation_type=relation_type,
        composition_operator=composition_operator,
        baseline_effect=baseline_effect,
        source_effect=source_effect,
        target_effect=target_effect,
        combined_effect=combined_effect,
        uncertainty=uncertainty,
        replication_count=replication_count,
        required_ablations=required_ablations,
        evidence_refs=evidence_refs,
        condition_set=condition_set,
        anti_conditions=anti_conditions,
        claim_scope="ranking_only",
        verification_state="candidate",
        validity_gates={"four_cell_comparison": True, "frozen_verifier": False},
        notes=notes,
    )


def settle_mechanism_relation(
    *,
    baseline: object,
    source: object,
    target: object,
    combined: object,
    source_mechanism_id: str,
    target_mechanism_id: str,
    composition_operator: str,
    required_ablations: Sequence[str],
    evidence_refs: Sequence[str] = (),
    condition_set: Sequence[str] = (),
    anti_conditions: Sequence[str] = (),
    synergy_threshold: float = 0.0,
) -> dict[str, object]:
    """Settle four compatible effect records into one relation artifact.

    This is the receipt-to-knowledge boundary. It accepts the existing
    ``EffectRecord`` shape without coupling the geometry layer to a particular
    trainer or evaluator implementation.
    """

    from wmloop.geometry.memory import EffectRecord

    records = (baseline, source, target, combined)
    if any(not isinstance(record, EffectRecord) for record in records):
        raise GeometryValidationError("MECHANISM_RELATION_EFFECT_RECORD_INVALID")
    typed = tuple(record for record in records if isinstance(record, EffectRecord))
    context_fields = ("backbone_family", "capability_class", "goal_schema", "outcome_schema", "data_regime", "horizons")
    first = typed[0].context
    for record in typed[1:]:
        if any(getattr(record.context, field) != getattr(first, field) for field in context_fields):
            raise GeometryValidationError("MECHANISM_RELATION_CONTEXT_MISMATCH")
    if source.primitive != source_mechanism_id or target.primitive != target_mechanism_id:
        raise GeometryValidationError("MECHANISM_RELATION_PRIMITIVE_BINDING_MISMATCH")
    refs = tuple(dict.fromkeys(str(value) for record in typed for value in record.evidence_refs))
    refs = tuple(dict.fromkeys((*refs, *(str(value) for value in evidence_refs))))
    if not refs:
        raise GeometryValidationError("MECHANISM_RELATION_EVIDENCE_REF_INVALID")
    relation_type, effect = classify_interaction(
        baseline=baseline.mean_effect,
        source=source.mean_effect,
        target=target.mean_effect,
        combined=combined.mean_effect,
        synergy_threshold=synergy_threshold,
        uncertainty=math.sqrt(sum(record.standard_error**2 for record in typed)),
    )
    uncertainty = math.sqrt(sum(record.standard_error**2 for record in typed))
    all_confirmed = all(record.status == "confirmed" for record in typed)
    all_gates = all(all(record.validity_gates.values()) for record in typed)
    authoritative = all_confirmed and all_gates and bool(required_ablations)
    if relation_type == "abstained":
        relation_state = "abstained"
        relation_type = "conditional_compatibility"
    elif authoritative:
        relation_state = "confirmed" if relation_type != "antagonism" else "rejected"
    else:
        relation_state = "candidate"
    return build_mechanism_relation(
        source_mechanism_id=source_mechanism_id,
        target_mechanism_id=target_mechanism_id,
        relation_type=relation_type,
        composition_operator=composition_operator,
        baseline_effect=baseline.mean_effect,
        source_effect=source.mean_effect,
        target_effect=target.mean_effect,
        combined_effect=combined.mean_effect,
        uncertainty=uncertainty,
        replication_count=min(record.replication_count for record in typed),
        required_ablations=required_ablations,
        evidence_refs=refs,
        condition_set=condition_set,
        anti_conditions=anti_conditions,
        claim_scope="target_local" if relation_state in {"confirmed", "rejected"} else "ranking_only",
        verification_state=relation_state,
        validity_gates={
            "four_cell_comparison": True,
            "effect_records_confirmed": all_confirmed,
            "effect_record_gates": all_gates,
            "required_ablations_declared": bool(required_ablations),
        },
        notes=(f"interaction_effect={effect:.8g}",),
    )


def validate_mechanism_relation(document: Mapping[str, object]) -> None:
    if not isinstance(document, Mapping):
        raise GeometryValidationError("MECHANISM_RELATION_DOCUMENT_INVALID")
    if document.get("artifact_type") != "verdiwm-mechanism-relation":
        raise GeometryValidationError("MECHANISM_RELATION_ARTIFACT_INVALID")
    refs = document.get("evidence_refs")
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not is_content_addressed(ref) for ref in refs):
        raise GeometryValidationError("MECHANISM_RELATION_EVIDENCE_REF_INVALID")
    required = ("condition_set", "anti_conditions", "required_ablations", "notes")
    for field in required:
        value = document.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise GeometryValidationError(f"MECHANISM_RELATION_{field.upper()}_INVALID")
    try:
        expected = interaction_effect(
            baseline=float(document["baseline_effect"]),
            source=float(document["source_effect"]),
            target=float(document["target_effect"]),
            combined=float(document["combined_effect"]),
        )
        actual = float(document["interaction_effect"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryValidationError("MECHANISM_RELATION_VALUE_INVALID") from exc
    if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9):
        raise GeometryValidationError("MECHANISM_RELATION_EFFECT_MISMATCH")
    gates = document.get("validity_gates")
    if not isinstance(gates, Mapping) or not gates or any(not isinstance(value, bool) for value in gates.values()):
        raise GeometryValidationError("MECHANISM_RELATION_GATE_INVALID")
    state = document.get("verification_state")
    if state in _AUTHORITATIVE_STATES:
        if not all(gates.values()):
            raise GeometryValidationError("MECHANISM_RELATION_GATE_FAILED")
        if not document.get("required_ablations"):
            raise GeometryValidationError("MECHANISM_RELATION_ABLATION_MISSING")
    relation_id = document.get("relation_id")
    body = dict(document)
    body.pop("relation_id", None)
    expected_id = "mechanism-relation-" + hashlib.sha256(_canonical_json(body)).hexdigest()[:24]
    if relation_id != expected_id:
        raise GeometryValidationError("MECHANISM_RELATION_ID_MISMATCH")
    try:
        validate_document("mechanism_relation", document)
    except ContractValidationError as exc:
        raise GeometryValidationError(f"MECHANISM_RELATION_SCHEMA_INVALID:{exc}") from exc


def relation_from_dict(document: Mapping[str, object]) -> MechanismRelation:
    validate_mechanism_relation(document)
    values = dict(document)
    for field in ("condition_set", "anti_conditions", "required_ablations", "evidence_refs", "notes"):
        values[field] = tuple(values[field])
    return MechanismRelation(
        relation_id=str(values["relation_id"]),
        source_mechanism_id=str(values["source_mechanism_id"]),
        target_mechanism_id=str(values["target_mechanism_id"]),
        relation_type=str(values["relation_type"]),
        composition_operator=str(values["composition_operator"]),
        condition_set=values["condition_set"],
        anti_conditions=values["anti_conditions"],
        baseline_effect=float(values["baseline_effect"]),
        source_effect=float(values["source_effect"]),
        target_effect=float(values["target_effect"]),
        combined_effect=float(values["combined_effect"]),
        interaction_effect=float(values["interaction_effect"]),
        uncertainty=float(values["uncertainty"]),
        replication_count=int(values["replication_count"]),
        required_ablations=values["required_ablations"],
        evidence_refs=values["evidence_refs"],
        claim_scope=str(values["claim_scope"]),
        verification_state=str(values["verification_state"]),
        validity_gates=values["validity_gates"],
        notes=values["notes"],
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
