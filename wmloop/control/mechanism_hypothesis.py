"""Evidence-bound mechanism hypotheses used before code generation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import canonical_bytes


class MechanismHypothesisError(ValueError):
    """A mechanism hypothesis lacks a valid, content-bound explanation."""


def hypothesis_digest(document: Mapping[str, object]) -> str:
    body = {key: value for key, value in document.items() if key != "hypothesis_id"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def build_mechanism_hypothesis(
    *,
    target_failure: str,
    causal_assumption: str,
    target_touchpoint: Sequence[str],
    predicted_observation: str,
    falsification_test: str,
    required_capabilities: Sequence[str] = (),
    anti_conditions: Sequence[str] = (),
    source_evidence: Sequence[str] = (),
    novelty_status: str = "novelty_unresolved",
    claim_boundary: str = "A mechanism hypothesis is a falsifiable research proposal, not evidence of improvement.",
) -> dict[str, object]:
    # A string is a Sequence too; reject it rather than silently treating each
    # character as a separate touchpoint, capability, or evidence reference.
    for values in (target_touchpoint, required_capabilities, anti_conditions, source_evidence):
        if isinstance(values, (str, bytes)):
            raise MechanismHypothesisError("MECHANISM_HYPOTHESIS_LIST_REQUIRED")
    body = {
        "schema_version": 1,
        "artifact_type": "verdiwm-mechanism-hypothesis",
        "hypothesis_id": "",
        "target_failure": target_failure,
        "causal_assumption": causal_assumption,
        "target_touchpoint": list(target_touchpoint),
        "predicted_observation": predicted_observation,
        "falsification_test": falsification_test,
        "required_capabilities": list(required_capabilities),
        "anti_conditions": list(anti_conditions),
        "source_evidence": list(source_evidence),
        "novelty_status": novelty_status,
        "claim_boundary": claim_boundary,
    }
    body["hypothesis_id"] = "mechanism-hypothesis-" + hypothesis_digest(body)[:24]
    validate_mechanism_hypothesis(body)
    return body


def validate_mechanism_hypothesis(document: Mapping[str, object], *, root: Path | None = None) -> None:
    try:
        validate_document("mechanism_hypothesis", document, root=root)
        expected = "mechanism-hypothesis-" + hypothesis_digest(document)[:24]
    except (ContractValidationError, ValueError, TypeError) as exc:
        raise MechanismHypothesisError(f"MECHANISM_HYPOTHESIS_SCHEMA_INVALID:{exc}") from exc
    if document.get("hypothesis_id") != expected:
        raise MechanismHypothesisError("MECHANISM_HYPOTHESIS_DIGEST_MISMATCH")
    for key in ("target_failure", "causal_assumption", "predicted_observation", "falsification_test"):
        if not str(document[key]).strip():
            raise MechanismHypothesisError("MECHANISM_HYPOTHESIS_EMPTY_EXPLANATION")
