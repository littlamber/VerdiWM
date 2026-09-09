"""Portable settlement summaries retain verifier decisions without inventing effects."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from wmloop.contracts import validate_document
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.storage import canonical_bytes


def validate_settled_evidence(document: Mapping[str, object]) -> None:
    validate_document("settled_evidence", document)
    reject_runtime_bindings(document)
    body = dict(document)
    identifier = body.pop("evidence_id")
    if identifier != "settled-evidence-" + hashlib.sha256(canonical_bytes(body)).hexdigest()[:24]:
        raise ValueError("SETTLED_EVIDENCE_ID_MISMATCH")


def build_settled_evidence(*, receipt_ref: str, verdict_ref: str, failure_context_ref: str,
                           evaluator_sha256: str, hypothesis_sha256: str, implementation_sha256: str,
                           decision: str, evidence_scope: str) -> dict:
    body = {
        "schema_version": 1,
        "artifact_type": "verdiwm-settled-evidence",
        "receipt_ref": receipt_ref,
        "verdict_ref": verdict_ref,
        "failure_context_ref": failure_context_ref,
        "evaluator_sha256": evaluator_sha256,
        "hypothesis_sha256": hypothesis_sha256,
        "implementation_sha256": implementation_sha256,
        "decision": decision,
        "evidence_scope": evidence_scope,
        "claim_scope": "settlement_only",
        "claim_boundary": "This is a settled verifier decision. It does not estimate an effect, uncertainty, model similarity, or license target transfer.",
    }
    body["evidence_id"] = "settled-evidence-" + hashlib.sha256(canonical_bytes(body)).hexdigest()[:24]
    validate_settled_evidence(body)
    return body
