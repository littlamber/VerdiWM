"""Path-independent projections of settled intervention experience.

Runtime receipts may contain local paths, but reusable knowledge must be
addressed by semantic identities and content-addressed evidence.  This module
keeps that boundary explicit while delegating effect and transfer authority to
the existing :mod:`wmloop.geometry.memory` contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.evidence_ir import (
    build_evidence_ir,
    is_content_addressed,
    reject_runtime_bindings,
    validate_evidence_ir,
)
from wmloop.geometry.memory import EffectRecord, build_transferable_experience
from wmloop.geometry.types import GeometryValidationError


def build_portable_experience(
    effect: EffectRecord,
    *,
    transfer_certificate: object | None = None,
    applicability: Mapping[str, object] | None = None,
    anti_conditions: Sequence[str] = (),
    semantic_intent: str | None = None,
    target_hooks: Sequence[str] = (),
    goal_binding: str | None = None,
    evaluator_binding: str | None = None,
) -> dict[str, object]:
    """Build a semantic transfer projection with no runtime path bindings."""

    base = build_transferable_experience(
        effect,
        transfer_certificate=transfer_certificate,
        applicability=applicability,
        anti_conditions=anti_conditions,
    )
    evidence_refs = tuple(str(value) for value in effect.evidence_refs)
    if any(not is_content_addressed(value) for value in evidence_refs):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_EVIDENCE_REF_NOT_CONTENT_ADDRESSED")

    context = effect.context
    portable: dict[str, object] = {
        **base,
        "portable_knowledge": {
            "model_family": context.backbone_family,
            "capability_class": context.capability_class,
            "goal_protocol": context.goal_schema,
            "outcome_protocol": context.outcome_schema,
            "dataset_regime": context.data_regime,
            "horizons": list(context.horizons),
            "primitive": effect.primitive,
        },
        "portability": {
            "knowledge_scope": "semantic_effect_and_transfer_boundary",
            "runtime_bindings_excluded": True,
            "evidence_ref_schemes": ["cas", "urn", "sha256"],
        },
    }
    portable["evidence_refs"] = list(evidence_refs)
    portable["evidence_ir"] = build_evidence_ir(
        effect,
        transfer_state=str(base["transfer_state"]),
        applicability=dict(base["applicability"]),
        semantic_intent=semantic_intent,
        target_hooks=target_hooks,
        goal_binding=goal_binding,
        evaluator_binding=evaluator_binding,
    )
    validate_portable_experience(portable)
    portable["portable_experience_id"] = "portable-experience-" + hashlib.sha256(
        _canonical_json(portable)
    ).hexdigest()[:24]
    return portable


def validate_portable_experience(document: Mapping[str, object]) -> None:
    """Validate the portability boundary, failing closed on path coupling."""

    if not isinstance(document, Mapping):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_DOCUMENT_INVALID")
    if document.get("artifact_type") != "verdiwm-transferable-experience":
        raise GeometryValidationError("PORTABLE_EXPERIENCE_ARTIFACT_INVALID")
    required = (
        "portable_knowledge",
        "applicability",
        "anti_conditions",
        "evidence_refs",
        "portability",
    )
    missing = tuple(name for name in required if name not in document)
    if missing:
        raise GeometryValidationError("PORTABLE_EXPERIENCE_FIELDS_MISSING:" + ",".join(missing))
    reject_runtime_bindings(document)
    refs = document["evidence_refs"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_EVIDENCE_INVALID")
    if any(not is_content_addressed(ref) for ref in refs):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_EVIDENCE_REF_NOT_CONTENT_ADDRESSED")
    knowledge = document["portable_knowledge"]
    if not isinstance(knowledge, Mapping):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_KNOWLEDGE_INVALID")
    for name in (
        "model_family",
        "capability_class",
        "goal_protocol",
        "outcome_protocol",
        "dataset_regime",
        "primitive",
    ):
        value = knowledge.get(name)
        if not isinstance(value, str) or not value.strip():
            raise GeometryValidationError("PORTABLE_EXPERIENCE_KNOWLEDGE_NOT_SEMANTIC:" + name)
    horizons = knowledge.get("horizons")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or value <= 0 for value in horizons)
    ):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_HORIZONS_INVALID")
    applicability = document["applicability"]
    if not isinstance(applicability, Mapping) or not applicability:
        raise GeometryValidationError("PORTABLE_EXPERIENCE_APPLICABILITY_INVALID")
    reject_runtime_bindings(applicability)
    anti_conditions = document["anti_conditions"]
    if not isinstance(anti_conditions, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in anti_conditions
    ):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_ANTI_CONDITIONS_INVALID")
    evidence_ir = document.get("evidence_ir")
    if evidence_ir is not None:
        if not isinstance(evidence_ir, Mapping):
            raise GeometryValidationError("PORTABLE_EXPERIENCE_EVIDENCE_IR_INVALID")
        validate_evidence_ir(evidence_ir)
    try:
        validate_document("portable_experience", document)
    except ContractValidationError as exc:
        raise GeometryValidationError(f"PORTABLE_EXPERIENCE_SCHEMA_INVALID:{exc}") from exc
    if "portable_experience_id" in document:
        body = dict(document)
        body.pop("portable_experience_id", None)
        expected = "portable-experience-" + hashlib.sha256(
            _canonical_json(body)
        ).hexdigest()[:24]
        if document["portable_experience_id"] != expected:
            raise GeometryValidationError("PORTABLE_EXPERIENCE_ID_MISMATCH")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
