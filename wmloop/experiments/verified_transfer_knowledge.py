"""Project verified materialized transfers into path-free community knowledge.

The materialized transfer evidence record intentionally retains a candidate
provenance envelope for local audit.  This module is the narrow boundary
between that local evidence and reusable community knowledge: it reads only
the verified semantic fields required to describe a mechanism and a target
embodiment, and never exports candidate provenance or local locations.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import canonical_json_bytes
from wmloop.geometry import (
    GeometryValidationError,
    build_mechanism_contract,
    build_method_embodiment,
    build_transfer_boundary,
    validate_probe_fingerprint_summary,
)
from wmloop.geometry.evidence_ir import reject_runtime_bindings


class VerifiedTransferKnowledgeError(ValueError):
    """Verified source and target evidence could not produce portable knowledge."""


def project_verified_transfer_knowledge(
    *,
    source_assessment: Mapping[str, object],
    transfer_evidence: Mapping[str, object],
    source_fingerprint: Mapping[str, object] | None = None,
    target_fingerprint: Mapping[str, object] | None = None,
    evaluator_binding: str | None = None,
    project_root: Path | None = None,
) -> list[dict[str, object]]:
    """Create path-free records from one verified transfer.

    A mechanism and target embodiment are always useful when source grounding
    and frozen target evidence agree.  A transfer boundary is deliberately
    absent unless both independently validated behavioral fingerprints and the
    frozen evaluator binding are supplied.  The function never invents either
    fingerprint from a candidate's local provenance.
    """

    schema_root = Path(project_root).resolve() if project_root is not None else None
    _validate("acwm_source_transfer_assessment_v2", source_assessment, schema_root)
    _validate("materialized_transfer_evidence", transfer_evidence, schema_root)
    _validate_source_assessment(source_assessment)
    _validate_transfer_binding(source_assessment, transfer_evidence)

    source_refs = _source_refs(source_assessment)
    evidence_refs = _evidence_refs(transfer_evidence, source_assessment)
    required_capabilities = _semantic_strings(
        source_assessment["required_target_capabilities"],
        code="VERIFIED_TRANSFER_KNOWLEDGE_CAPABILITIES_INVALID",
    )
    if not required_capabilities:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_CAPABILITIES_INVALID")
    forbidden = _semantic_strings(
        source_assessment["forbidden_substitutions"],
        code="VERIFIED_TRANSFER_KNOWLEDGE_FORBIDDEN_SUBSTITUTIONS_INVALID",
    )
    target_intervention = _semantic_string(
        source_assessment["target_intervention"],
        code="VERIFIED_TRANSFER_KNOWLEDGE_INTERVENTION_INVALID",
    )
    falsification = _semantic_string(
        source_assessment["falsification_criterion"],
        code="VERIFIED_TRANSFER_KNOWLEDGE_FALSIFICATION_INVALID",
    )

    mechanism = build_mechanism_contract(
        causal_claim=(
            "The source-grounded "
            f"{_semantic_string(source_assessment['profile_id'], code='VERIFIED_TRANSFER_KNOWLEDGE_PROFILE_INVALID')} "
            f"mechanism is represented on the target by: {target_intervention}"
        ),
        intervention_semantics=target_intervention,
        required_capabilities=required_capabilities,
        target_interface_requirements=tuple(f"capability:{value}" for value in required_capabilities),
        prohibited_substitutions=forbidden,
        required_ablations=("no_intervention",),
        falsification_criterion=falsification,
        known_anti_conditions=tuple(
            f"required target capability unavailable: {value}" for value in required_capabilities
        ),
        source_evidence_refs=source_refs,
    )
    outcome = str(transfer_evidence["outcome"])
    embodiment = build_method_embodiment(
        mechanism_id=str(mechanism["mechanism_id"]),
        materialization_class="derived_embodiment",
        implementation_revision=str(transfer_evidence["implementation_revision"]),
        interface_contracts=tuple(f"capability:{value}" for value in required_capabilities),
        implementation_state=_implementation_state(outcome),
        claim_boundary=_semantic_string(
            transfer_evidence["claim_boundary"],
            code="VERIFIED_TRANSFER_KNOWLEDGE_CLAIM_BOUNDARY_INVALID",
        ),
        evidence_refs=evidence_refs,
    )
    records: list[dict[str, object]] = [mechanism, embodiment]
    boundary = _build_boundary_if_bound(
        source_assessment=source_assessment,
        transfer_evidence=transfer_evidence,
        mechanism=mechanism,
        embodiment=embodiment,
        source_fingerprint=source_fingerprint,
        target_fingerprint=target_fingerprint,
        evaluator_binding=evaluator_binding,
        evidence_refs=evidence_refs,
        required_capabilities=required_capabilities,
    )
    if boundary is not None:
        records.append(boundary)
    return records


def stage_verified_transfer_knowledge(
    *,
    source_assessment: Mapping[str, object],
    transfer_evidence: Mapping[str, object],
    output_root: Path,
    source_fingerprint: Mapping[str, object] | None = None,
    target_fingerprint: Mapping[str, object] | None = None,
    evaluator_binding: str | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Idempotently stage the projection for a later portable-graph rebuild."""

    records = project_verified_transfer_knowledge(
        source_assessment=source_assessment,
        transfer_evidence=transfer_evidence,
        source_fingerprint=source_fingerprint,
        target_fingerprint=target_fingerprint,
        evaluator_binding=evaluator_binding,
        project_root=project_root,
    )
    root = _prepare_output_root(output_root)
    identifiers: list[str] = []
    for record in records:
        identifier = _record_identifier(record)
        _write_idempotent(root / f"{identifier}.json", canonical_json_bytes(record))
        identifiers.append(identifier)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-verified-transfer-knowledge-staging",
        "state": "ready",
        "record_count": len(records),
        "record_ids": sorted(identifiers),
        "boundary_staged": any(
            record["artifact_type"] == "verdiwm-transfer-boundary" for record in records
        ),
    }


def _build_boundary_if_bound(
    *,
    source_assessment: Mapping[str, object],
    transfer_evidence: Mapping[str, object],
    mechanism: Mapping[str, object],
    embodiment: Mapping[str, object],
    source_fingerprint: Mapping[str, object] | None,
    target_fingerprint: Mapping[str, object] | None,
    evaluator_binding: str | None,
    evidence_refs: Sequence[str],
    required_capabilities: Sequence[str],
) -> dict[str, object] | None:
    outcome = str(transfer_evidence["outcome"])
    if outcome == "operational_failure":
        return None
    provided = (source_fingerprint, target_fingerprint, evaluator_binding)
    if all(value is None for value in provided):
        return None
    if any(value is None for value in provided):
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_BOUNDARY_BINDING_INCOMPLETE")
    assert source_fingerprint is not None
    assert target_fingerprint is not None
    assert evaluator_binding is not None
    try:
        validate_probe_fingerprint_summary(source_fingerprint)
        validate_probe_fingerprint_summary(target_fingerprint)
    except GeometryValidationError as exc:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_FINGERPRINT_INVALID") from exc
    if target_fingerprint.get("model_family") != transfer_evidence.get("model_family"):
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_TARGET_FINGERPRINT_MISMATCH")
    expected_binding = "sha256:" + str(transfer_evidence["policy_digest"])
    if evaluator_binding != expected_binding:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_EVALUATOR_BINDING_MISMATCH")
    outcome_state = (
        "target_confirmed" if outcome == "confirmed_positive" else "verified_negative_boundary"
    )
    result_phrase = (
        "Frozen verification confirmed the target embodiment only under the declared "
        "target fingerprint and evaluator binding."
        if outcome_state == "target_confirmed"
        else "Frozen verification rejected this target embodiment under the declared target "
        "fingerprint; this is an applicability boundary, not a source-method erasure."
    )
    return build_transfer_boundary(
        mechanism_id=str(mechanism["mechanism_id"]),
        embodiment_id=str(embodiment["embodiment_id"]),
        source_fingerprint_id=str(source_fingerprint["fingerprint_id"]),
        target_fingerprint_id=str(target_fingerprint["fingerprint_id"]),
        target_model_capability_id=str(target_fingerprint["model_capability_id"]),
        required_capabilities=required_capabilities,
        anti_conditions=tuple(
            f"required target capability unavailable: {value}" for value in required_capabilities
        ),
        outcome_state=outcome_state,
        boundary_statement=result_phrase,
        evaluator_binding=evaluator_binding,
        verdict_ref=str(transfer_evidence["verdict_ref"]),
        evidence_refs=evidence_refs,
    )


def _validate_source_assessment(assessment: Mapping[str, object]) -> None:
    if assessment.get("source_role") != "transferable_optimizer":
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_SOURCE_ROLE_INVALID")
    if assessment.get("transfer_mode") != "mechanism_transfer":
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_TRANSFER_MODE_INVALID")
    if assessment.get("execution_state") != "materialization_required":
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_SOURCE_STATE_INVALID")
    expected = _assessment_digest(assessment)
    if assessment.get("assessment_digest") != expected:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_ASSESSMENT_DIGEST_MISMATCH")
    evidence = assessment.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_SOURCE_EVIDENCE_INVALID")


def _validate_transfer_binding(
    assessment: Mapping[str, object], transfer: Mapping[str, object]
) -> None:
    if (
        transfer.get("settlement_state") != "settled"
        or transfer.get("verification_state") != "verified"
        or transfer.get("model_family") != "ctrl-world"
    ):
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_VERIFICATION_INVALID")
    for field in ("source_id", "source_digest", "assessment_digest"):
        if transfer.get(field) != assessment.get(field):
            raise VerifiedTransferKnowledgeError(
                "VERIFIED_TRANSFER_KNOWLEDGE_SOURCE_BINDING_MISMATCH:" + field
            )
    refs = transfer.get("evidence_refs")
    if not isinstance(refs, list) or transfer.get("verdict_ref") not in refs:
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_VERDICT_REFERENCE_INVALID")


def _source_refs(assessment: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                "sha256:" + str(assessment["source_digest"]),
                "sha256:" + str(assessment["assessment_digest"]),
            }
        )
    )


def _evidence_refs(
    transfer: Mapping[str, object], assessment: Mapping[str, object]
) -> tuple[str, ...]:
    refs = transfer.get("evidence_refs")
    assert isinstance(refs, list)
    return tuple(sorted({*(str(value) for value in refs), "sha256:" + str(assessment["assessment_digest"])}))


def _semantic_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerifiedTransferKnowledgeError(code)
    try:
        reject_runtime_bindings(value)
    except GeometryValidationError as exc:
        raise VerifiedTransferKnowledgeError(code) from exc
    return value


def _semantic_strings(value: object, *, code: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise VerifiedTransferKnowledgeError(code)
    return tuple(_semantic_string(item, code=code) for item in value)


def _implementation_state(outcome: str) -> str:
    if outcome == "confirmed_positive":
        return "confirmed"
    if outcome == "operational_failure":
        return "blocked"
    return "screened"


def _assessment_digest(assessment: Mapping[str, object]) -> str:
    body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validate(schema: str, payload: Mapping[str, object], root: Path | None) -> None:
    try:
        validate_document(schema, payload, root=root)
    except ContractValidationError as exc:
        raise VerifiedTransferKnowledgeError(
            f"VERIFIED_TRANSFER_KNOWLEDGE_SCHEMA_INVALID:{schema}:{exc}"
        ) from exc


def _prepare_output_root(output_root: Path) -> Path:
    raw = Path(output_root).expanduser()
    if raw.is_symlink():
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_OUTPUT_INVALID")
    root = raw.resolve()
    if root.exists() and not root.is_dir():
        raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_OUTPUT_INVALID")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _record_identifier(record: Mapping[str, object]) -> str:
    primary_fields = {
        "verdiwm-mechanism-contract": "mechanism_id",
        "verdiwm-method-embodiment": "embodiment_id",
        "verdiwm-transfer-boundary": "boundary_id",
    }
    field = primary_fields.get(record.get("artifact_type"))
    value = record.get(field) if field is not None else None
    if isinstance(value, str):
        return value
    raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_RECORD_INVALID")


def _write_idempotent(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise VerifiedTransferKnowledgeError("VERIFIED_TRANSFER_KNOWLEDGE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
