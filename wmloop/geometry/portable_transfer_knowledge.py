"""Path-free contracts for sharing method-transfer knowledge.

These records are deliberately separate from local implementation receipts.
They describe a mechanism, one implementation embodiment, a behavioral
fingerprint, and a transfer boundary using semantic identifiers and
content-addressed evidence only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.evidence_ir import is_content_addressed, reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


_OUTCOME_SCOPES = {
    "exploratory": "ranking_only",
    "screen_only": "ranking_only",
    "target_confirmed": "target_local",
    "verified_negative_boundary": "target_local",
    "licensed_transfer": "transfer_prior",
}
_AUTHORITATIVE_OUTCOMES = {
    "target_confirmed",
    "verified_negative_boundary",
    "licensed_transfer",
}


def build_mechanism_contract(
    *,
    causal_claim: str,
    intervention_semantics: str,
    required_capabilities: Sequence[str],
    optional_capabilities: Sequence[str] = (),
    target_interface_requirements: Sequence[str] = (),
    prohibited_substitutions: Sequence[str] = (),
    required_ablations: Sequence[str],
    falsification_criterion: str,
    known_anti_conditions: Sequence[str] = (),
    source_evidence_refs: Sequence[str],
) -> dict[str, object]:
    """Build a semantic, source-grounded mechanism contract."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-mechanism-contract",
        "causal_claim": causal_claim,
        "intervention_semantics": intervention_semantics,
        "required_capabilities": list(required_capabilities),
        "optional_capabilities": list(optional_capabilities),
        "target_interface_requirements": list(target_interface_requirements),
        "prohibited_substitutions": list(prohibited_substitutions),
        "required_ablations": list(required_ablations),
        "falsification_criterion": falsification_criterion,
        "known_anti_conditions": list(known_anti_conditions),
        "source_evidence_refs": list(source_evidence_refs),
    }
    body["mechanism_id"] = _stable_id("mechanism", body)
    validate_mechanism_contract(body)
    return body


def build_method_embodiment(
    *,
    mechanism_id: str,
    materialization_class: str,
    implementation_revision: str,
    interface_contracts: Sequence[str],
    implementation_state: str,
    claim_boundary: str,
    evidence_refs: Sequence[str],
    executable_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one implementation embodiment without exposing source paths."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-embodiment",
        "mechanism_id": mechanism_id,
        "materialization_class": materialization_class,
        "implementation_revision": implementation_revision,
        "interface_contracts": list(interface_contracts),
        "implementation_state": implementation_state,
        "claim_boundary": claim_boundary,
        "evidence_refs": list(evidence_refs),
    }
    if executable_binding is not None:
        body["executable_binding"] = dict(executable_binding)
    body["embodiment_id"] = _stable_id("embodiment", body)
    validate_method_embodiment(body)
    return body


def build_probe_fingerprint_summary(
    *,
    model_capability_id: str,
    model_family: str,
    probe_protocol_id: str,
    probe_protocol_version: str,
    diagnostic_role: str,
    context_class: str,
    split: str,
    horizons: Sequence[int],
    dose_values: Sequence[float],
    replication_count: int,
    response_dimension: int,
    response_summary: str,
    response_digest: str,
    uncertainty_summary: str,
    evidence_refs: Sequence[str],
) -> dict[str, object]:
    """Build a compact behavioral fingerprint with raw data held in CAS."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-fingerprint-summary",
        "model_capability_id": model_capability_id,
        "model_family": model_family,
        "probe_protocol_id": probe_protocol_id,
        "probe_protocol_version": probe_protocol_version,
        "diagnostic_role": diagnostic_role,
        "context_class": context_class,
        "split": split,
        "horizons": list(horizons),
        "dose_values": list(dose_values),
        "replication_count": replication_count,
        "response_summary": {
            "dimension": response_dimension,
            "summary": response_summary,
        },
        "response_digest": response_digest,
        "uncertainty_summary": uncertainty_summary,
        "evidence_refs": list(evidence_refs),
    }
    body["fingerprint_id"] = _stable_id("fingerprint", body)
    validate_probe_fingerprint_summary(body)
    return body


def build_transfer_boundary(
    *,
    mechanism_id: str,
    embodiment_id: str,
    source_fingerprint_id: str,
    target_fingerprint_id: str,
    target_model_capability_id: str,
    required_capabilities: Sequence[str],
    anti_conditions: Sequence[str],
    outcome_state: str,
    boundary_statement: str,
    evidence_refs: Sequence[str],
    evaluator_binding: str | None = None,
    verdict_ref: str | None = None,
) -> dict[str, object]:
    """Build an evidence-bound statement of where transfer does or does not hold."""

    claim_scope = _OUTCOME_SCOPES.get(outcome_state)
    if claim_scope is None:
        raise GeometryValidationError("TRANSFER_BOUNDARY_OUTCOME_INVALID")
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-transfer-boundary",
        "mechanism_id": mechanism_id,
        "embodiment_id": embodiment_id,
        "source_fingerprint_id": source_fingerprint_id,
        "target_fingerprint_id": target_fingerprint_id,
        "target_model_capability_id": target_model_capability_id,
        "required_capabilities": list(required_capabilities),
        "anti_conditions": list(anti_conditions),
        "outcome_state": outcome_state,
        "claim_scope": claim_scope,
        "boundary_statement": boundary_statement,
        "evaluator_binding": evaluator_binding,
        "verdict_ref": verdict_ref,
        "evidence_refs": list(evidence_refs),
    }
    body["boundary_id"] = _stable_id("transfer-boundary", body)
    validate_transfer_boundary(body)
    return body


def validate_mechanism_contract(document: Mapping[str, object]) -> None:
    _validate_portable_document(
        document,
        schema_name="mechanism_contract",
        identifier="mechanism_id",
        prefix="mechanism",
        reference_fields=("source_evidence_refs",),
    )


def validate_method_embodiment(document: Mapping[str, object]) -> None:
    _validate_portable_document(
        document,
        schema_name="method_embodiment",
        identifier="embodiment_id",
        prefix="embodiment",
    )


def validate_probe_fingerprint_summary(document: Mapping[str, object]) -> None:
    _validate_portable_document(
        document,
        schema_name="probe_fingerprint_summary",
        identifier="fingerprint_id",
        prefix="fingerprint",
        reference_fields=("response_digest", "evidence_refs"),
    )


def validate_transfer_boundary(document: Mapping[str, object]) -> None:
    _validate_portable_document(
        document,
        schema_name="transfer_boundary",
        identifier="boundary_id",
        prefix="transfer-boundary",
        reference_fields=("evaluator_binding", "verdict_ref", "evidence_refs"),
    )
    outcome = document.get("outcome_state")
    claim_scope = document.get("claim_scope")
    if _OUTCOME_SCOPES.get(outcome) != claim_scope:
        raise GeometryValidationError("TRANSFER_BOUNDARY_SCOPE_INVALID")
    evaluator = document.get("evaluator_binding")
    verdict = document.get("verdict_ref")
    if outcome in _AUTHORITATIVE_OUTCOMES:
        if not isinstance(evaluator, str) or not isinstance(verdict, str):
            raise GeometryValidationError("TRANSFER_BOUNDARY_AUTHORITY_BINDING_REQUIRED")
    elif evaluator is not None or verdict is not None:
        raise GeometryValidationError("TRANSFER_BOUNDARY_RANKING_BINDING_FORBIDDEN")


def _validate_portable_document(
    document: Mapping[str, object],
    *,
    schema_name: str,
    identifier: str,
    prefix: str,
    reference_fields: Sequence[str] = ("evidence_refs",),
) -> None:
    if not isinstance(document, Mapping):
        raise GeometryValidationError("PORTABLE_TRANSFER_KNOWLEDGE_DOCUMENT_INVALID")
    reject_runtime_bindings(document)
    try:
        validate_document(schema_name, document)
    except ContractValidationError as exc:
        raise GeometryValidationError(
            f"PORTABLE_TRANSFER_KNOWLEDGE_SCHEMA_INVALID:{exc}"
        ) from exc
    for field in reference_fields:
        _validate_references(document.get(field), field=field)
    body = dict(document)
    received = body.pop(identifier, None)
    expected = _stable_id(prefix, body)
    if received != expected:
        raise GeometryValidationError("PORTABLE_TRANSFER_KNOWLEDGE_ID_MISMATCH:" + identifier)


def _validate_references(value: object, *, field: str) -> None:
    values: Sequence[object]
    if value is None:
        return
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise GeometryValidationError("PORTABLE_TRANSFER_KNOWLEDGE_REFERENCE_INVALID:" + field)
    if any(not isinstance(item, str) or not is_content_addressed(item) for item in values):
        raise GeometryValidationError("PORTABLE_TRANSFER_KNOWLEDGE_REFERENCE_INVALID:" + field)


def _stable_id(prefix: str, body: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError("PORTABLE_TRANSFER_KNOWLEDGE_PAYLOAD_INVALID") from exc
    return prefix + "-" + hashlib.sha256(encoded).hexdigest()[:24]
