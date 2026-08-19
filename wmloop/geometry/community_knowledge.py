"""Portable contracts used only by the community knowledge projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.geometry.evidence_ir import is_content_addressed, reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class CommunityKnowledgeError(ValueError):
    """A path-free community contract failed semantic validation."""


_CAS_SHA256 = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


def build_portrait_transition(
    *,
    parent_portrait: Mapping[str, object],
    portrait: Mapping[str, object],
    embodiment_id: str,
    outcome_state: str,
    evaluator_binding: str | None = None,
    verdict_ref: str | None = None,
    evidence_refs: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Bind one append-only portrait update to its admitted embodiment."""

    try:
        validate_model_portrait(parent_portrait, root=root)
        validate_model_portrait(portrait, root=root)
    except ModelPortraitError as exc:
        raise CommunityKnowledgeError(
            f"PORTRAIT_TRANSITION_PORTRAIT_INVALID:{exc}"
        ) from exc
    transition_ref = portrait.get("transition_ref")
    if (
        portrait.get("parent_portrait_id") != parent_portrait.get("portrait_id")
        or not isinstance(transition_ref, str)
        or not is_content_addressed(transition_ref)
    ):
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_PARENT_BINDING_INVALID")
    state = _enum(
        outcome_state,
        {"admitted", "target_confirmed", "verified_negative_boundary"},
        "PORTRAIT_TRANSITION_OUTCOME_INVALID",
    )
    evaluator, verdict = _frozen_bindings(
        state=state,
        evaluator_binding=evaluator_binding,
        verdict_ref=verdict_ref,
    )
    refs = _refs(evidence_refs, "PORTRAIT_TRANSITION_EVIDENCE_INVALID")
    refs = sorted(
        {
            *refs,
            str(transition_ref),
            "sha256:" + _digest(parent_portrait),
            "sha256:" + _digest(portrait),
            *([evaluator] if evaluator is not None else []),
            *([verdict] if verdict is not None else []),
        }
    )
    if state != "admitted" and not any(_CAS_SHA256.fullmatch(value) for value in refs):
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_CAS_EVIDENCE_REQUIRED")
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-portrait-transition",
        "parent_portrait_id": parent_portrait["portrait_id"],
        "parent_portrait_digest": "sha256:" + _digest(parent_portrait),
        "portrait_id": portrait["portrait_id"],
        "portrait_digest": "sha256:" + _digest(portrait),
        "embodiment_id": _text(
            embodiment_id, "PORTRAIT_TRANSITION_EMBODIMENT_INVALID"
        ),
        "transition_ref": transition_ref,
        "outcome_state": state,
        "evaluator_binding": evaluator,
        "verdict_ref": verdict,
        "evidence_refs": refs,
        "claim_scope": "ranking_only" if state == "admitted" else "target_local",
    }
    body["transition_id"] = _stable_id("portrait-transition", body)
    validate_portrait_transition(body, root=root)
    return body


def validate_portrait_transition(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate("portrait_transition", document, root=root)
    _reject_runtime(document)
    state = str(document.get("outcome_state"))
    evaluator, verdict = _frozen_bindings(
        state=state,
        evaluator_binding=document.get("evaluator_binding"),
        verdict_ref=document.get("verdict_ref"),
    )
    expected_scope = "ranking_only" if state == "admitted" else "target_local"
    if document.get("claim_scope") != expected_scope:
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_SCOPE_INVALID")
    refs = _refs(document.get("evidence_refs"), "PORTRAIT_TRANSITION_EVIDENCE_INVALID")
    required = {
        str(document.get("transition_ref")),
        str(document.get("parent_portrait_digest")),
        str(document.get("portrait_digest")),
        *([evaluator] if evaluator is not None else []),
        *([verdict] if verdict is not None else []),
    }
    if not required.issubset(set(refs)):
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_EVIDENCE_BINDING_INVALID")
    if state != "admitted" and not any(_CAS_SHA256.fullmatch(value) for value in refs):
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_CAS_EVIDENCE_REQUIRED")
    _check_id(document, "transition_id", "portrait-transition")


def build_protocol_contract(
    *,
    protocol_kind: str,
    protocol_id: str,
    protocol_version: str,
    semantic_dimensions: Sequence[str],
    protected_fields: Sequence[str],
    contract_ref: str,
    evidence_refs: Sequence[str],
    license_spdx_id: str,
    claim_boundary: str,
    root: Path | None = None,
) -> dict[str, object]:
    """Describe a probe, metric, or evaluator protocol without runtime files."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-protocol-contract",
        "protocol_kind": _enum(
            protocol_kind,
            {"probe", "metric", "evaluator"},
            "PROTOCOL_CONTRACT_KIND_INVALID",
        ),
        "protocol_id": _text(protocol_id, "PROTOCOL_CONTRACT_ID_INVALID"),
        "protocol_version": _text(
            protocol_version, "PROTOCOL_CONTRACT_VERSION_INVALID"
        ),
        "semantic_dimensions": _texts(
            semantic_dimensions, "PROTOCOL_CONTRACT_DIMENSION_INVALID"
        ),
        "protected_fields": _texts(
            protected_fields, "PROTOCOL_CONTRACT_PROTECTED_FIELD_INVALID"
        ),
        "contract_ref": _ref(contract_ref, "PROTOCOL_CONTRACT_REF_INVALID"),
        "evidence_refs": _refs(
            evidence_refs, "PROTOCOL_CONTRACT_EVIDENCE_INVALID"
        ),
        "license_spdx_id": _text(
            license_spdx_id, "PROTOCOL_CONTRACT_LICENSE_INVALID"
        ),
        "redistribution_allowed": True,
        "claim_boundary": _text(
            claim_boundary, "PROTOCOL_CONTRACT_CLAIM_BOUNDARY_INVALID"
        ),
    }
    body["contract_id"] = _stable_id("protocol-contract", body)
    validate_protocol_contract(body, root=root)
    return body


def validate_protocol_contract(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate("protocol_contract", document, root=root)
    _reject_runtime(document)
    _ref(document.get("contract_ref"), "PROTOCOL_CONTRACT_REF_INVALID")
    _refs(document.get("evidence_refs"), "PROTOCOL_CONTRACT_EVIDENCE_INVALID")
    _check_id(document, "contract_id", "protocol-contract")


def build_transformation_contract(
    *,
    source_semantics: Sequence[str],
    target_semantics: Sequence[str],
    invariants: Sequence[str],
    loss_policy: str,
    implementation_ref: str,
    verification_ref: str,
    evidence_refs: Sequence[str],
    license_spdx_id: str,
    claim_boundary: str,
    root: Path | None = None,
) -> dict[str, object]:
    """Describe a semantic data conversion without local dataset locations."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-transformation-contract",
        "source_semantics": _texts(
            source_semantics, "TRANSFORMATION_SOURCE_SEMANTICS_INVALID"
        ),
        "target_semantics": _texts(
            target_semantics, "TRANSFORMATION_TARGET_SEMANTICS_INVALID"
        ),
        "invariants": _texts(invariants, "TRANSFORMATION_INVARIANT_INVALID"),
        "loss_policy": _enum(
            loss_policy,
            {"lossless", "bounded_loss"},
            "TRANSFORMATION_LOSS_POLICY_INVALID",
        ),
        "implementation_ref": _ref(
            implementation_ref, "TRANSFORMATION_IMPLEMENTATION_REF_INVALID"
        ),
        "verification_ref": _ref(
            verification_ref, "TRANSFORMATION_VERIFICATION_REF_INVALID"
        ),
        "evidence_refs": _refs(
            evidence_refs, "TRANSFORMATION_EVIDENCE_INVALID"
        ),
        "license_spdx_id": _text(
            license_spdx_id, "TRANSFORMATION_LICENSE_INVALID"
        ),
        "redistribution_allowed": True,
        "claim_boundary": _text(
            claim_boundary, "TRANSFORMATION_CLAIM_BOUNDARY_INVALID"
        ),
    }
    body["transform_id"] = _stable_id("transformation-contract", body)
    validate_transformation_contract(body, root=root)
    return body


def validate_transformation_contract(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate("transformation_contract", document, root=root)
    _reject_runtime(document)
    _ref(
        document.get("implementation_ref"),
        "TRANSFORMATION_IMPLEMENTATION_REF_INVALID",
    )
    _ref(
        document.get("verification_ref"),
        "TRANSFORMATION_VERIFICATION_REF_INVALID",
    )
    _refs(document.get("evidence_refs"), "TRANSFORMATION_EVIDENCE_INVALID")
    _check_id(document, "transform_id", "transformation-contract")


def build_knowledge_lifecycle_record(
    *,
    action: str,
    subject_kind: str,
    subject_id: str,
    reason: str,
    authority_ref: str,
    evidence_refs: Sequence[str],
    replacement_kind: str | None = None,
    replacement_id: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Record revocation, deprecation, or supersession as append-only knowledge."""

    normalized_action = _enum(
        action,
        {"revocation", "deprecation", "supersession"},
        "KNOWLEDGE_LIFECYCLE_ACTION_INVALID",
    )
    if normalized_action == "supersession":
        replacement = {
            "kind": _text(
                replacement_kind, "KNOWLEDGE_LIFECYCLE_REPLACEMENT_INVALID"
            ),
            "id": _text(
                replacement_id, "KNOWLEDGE_LIFECYCLE_REPLACEMENT_INVALID"
            ),
        }
    else:
        if replacement_kind is not None or replacement_id is not None:
            raise CommunityKnowledgeError(
                "KNOWLEDGE_LIFECYCLE_REPLACEMENT_FORBIDDEN"
            )
        replacement = None
    authority = _ref(authority_ref, "KNOWLEDGE_LIFECYCLE_AUTHORITY_INVALID")
    refs = _refs(evidence_refs, "KNOWLEDGE_LIFECYCLE_EVIDENCE_INVALID")
    if not _CAS_SHA256.fullmatch(authority) and not any(
        _CAS_SHA256.fullmatch(value) for value in refs
    ):
        raise CommunityKnowledgeError("KNOWLEDGE_LIFECYCLE_CAS_EVIDENCE_REQUIRED")
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-knowledge-lifecycle",
        "action": normalized_action,
        "subject": {
            "kind": _text(subject_kind, "KNOWLEDGE_LIFECYCLE_SUBJECT_INVALID"),
            "id": _text(subject_id, "KNOWLEDGE_LIFECYCLE_SUBJECT_INVALID"),
        },
        "replacement": replacement,
        "reason": _text(reason, "KNOWLEDGE_LIFECYCLE_REASON_INVALID"),
        "authority_ref": authority,
        "evidence_refs": sorted({*refs, authority}),
    }
    body["lifecycle_id"] = _stable_id("knowledge-lifecycle", body)
    validate_knowledge_lifecycle_record(body, root=root)
    return body


def validate_knowledge_lifecycle_record(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate("knowledge_lifecycle", document, root=root)
    _reject_runtime(document)
    action = str(document.get("action"))
    replacement = document.get("replacement")
    if (action == "supersession") != isinstance(replacement, Mapping):
        raise CommunityKnowledgeError("KNOWLEDGE_LIFECYCLE_REPLACEMENT_INVALID")
    authority = _ref(
        document.get("authority_ref"), "KNOWLEDGE_LIFECYCLE_AUTHORITY_INVALID"
    )
    refs = _refs(
        document.get("evidence_refs"), "KNOWLEDGE_LIFECYCLE_EVIDENCE_INVALID"
    )
    if authority not in refs:
        raise CommunityKnowledgeError("KNOWLEDGE_LIFECYCLE_AUTHORITY_UNBOUND")
    if not _CAS_SHA256.fullmatch(authority) and not any(
        _CAS_SHA256.fullmatch(value) for value in refs
    ):
        raise CommunityKnowledgeError("KNOWLEDGE_LIFECYCLE_CAS_EVIDENCE_REQUIRED")
    _check_id(document, "lifecycle_id", "knowledge-lifecycle")


def _frozen_bindings(
    *, state: str, evaluator_binding: object, verdict_ref: object
) -> tuple[str | None, str | None]:
    if state == "admitted":
        if evaluator_binding is not None or verdict_ref is not None:
            raise CommunityKnowledgeError(
                "PORTRAIT_TRANSITION_ADMITTED_AUTHORITY_FORBIDDEN"
            )
        return None, None
    evaluator = _ref(
        evaluator_binding, "PORTRAIT_TRANSITION_EVALUATOR_BINDING_INVALID"
    )
    verdict = _ref(verdict_ref, "PORTRAIT_TRANSITION_VERDICT_REF_INVALID")
    if not _CAS_SHA256.fullmatch(verdict):
        raise CommunityKnowledgeError("PORTRAIT_TRANSITION_FROZEN_VERDICT_REQUIRED")
    return evaluator, verdict


def _validate(
    schema: str, document: Mapping[str, object], *, root: Path | None
) -> None:
    if not isinstance(document, Mapping):
        raise CommunityKnowledgeError("COMMUNITY_KNOWLEDGE_DOCUMENT_INVALID")
    try:
        validate_document(schema, document, root=root)
    except ContractValidationError as exc:
        raise CommunityKnowledgeError(
            f"COMMUNITY_KNOWLEDGE_SCHEMA_INVALID:{schema}:{exc}"
        ) from exc


def _reject_runtime(value: object) -> None:
    try:
        reject_runtime_bindings(value)
    except GeometryValidationError as exc:
        raise CommunityKnowledgeError(
            f"COMMUNITY_KNOWLEDGE_RUNTIME_BINDING_FORBIDDEN:{exc}"
        ) from exc


def _check_id(document: Mapping[str, object], field: str, prefix: str) -> None:
    body = dict(document)
    received = body.pop(field, None)
    if received != _stable_id(prefix, body):
        raise CommunityKnowledgeError("COMMUNITY_KNOWLEDGE_ID_MISMATCH:" + field)


def _ref(value: object, code: str) -> str:
    if not isinstance(value, str) or not is_content_addressed(value):
        raise CommunityKnowledgeError(code)
    return value


def _refs(value: object, code: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise CommunityKnowledgeError(code)
    refs = [_ref(item, code) for item in value]
    return sorted(set(refs))


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommunityKnowledgeError(code)
    text = value.strip()
    _reject_runtime(text)
    return text


def _texts(value: object, code: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CommunityKnowledgeError(code)
    values = sorted({_text(item, code) for item in value})
    if not values:
        raise CommunityKnowledgeError(code)
    return values


def _enum(value: object, allowed: set[str], code: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise CommunityKnowledgeError(code)
    return value


def _stable_id(prefix: str, body: Mapping[str, object]) -> str:
    return prefix + "-" + _digest(body)[:24]


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommunityKnowledgeError("COMMUNITY_KNOWLEDGE_CANONICAL_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()
