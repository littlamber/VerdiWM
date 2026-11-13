"""Accumulate explicitly declared diagnostics from settled CAS evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ContentAddressedStore
from wmloop.retrieve.index import (
    ProbeRetrievalError,
    index_probe_experience,
    probe_asset_fingerprint,
)


def accumulate_diagnostic_result(
    *, plan: Mapping[str, object], manifest: Mapping[str, object],
    retrieval_db: Path, archive_db: Path, cas_root: Path,
) -> int:
    """Reuse index validation; execution success never supplies an effect label.

    The caller supplies the validated plan and the authoritative runner's
    terminal manifest. Replaying after interruption is idempotent in the index.
    """
    artifact_path = plan.get("diagnostic_result_path")
    if artifact_path is None:
        return 0
    if (
        plan.get("experiment_kind") != "diagnostic"
        or not isinstance(artifact_path, str)
        or artifact_path not in plan.get("artifacts", [])
    ):
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_DECLARATION_INVALID")
    ref = manifest.get("receipt_ref")
    if not isinstance(ref, str):
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_RECEIPT_MISSING")
    cas = ContentAddressedStore(cas_root)
    receipt = _object(cas.read_bytes(ref))
    if (
        receipt.get("artifact_type") != "verdiwm-auto-experiment-receipt"
        or not isinstance(manifest.get("plan_sha256"), str)
        or receipt.get("plan_sha256") != manifest.get("plan_sha256")
        or receipt.get("trial_id") != plan.get("trial_id")
        or receipt.get("campaign_id") != plan.get("campaign_id")
        or receipt.get("stage") != plan.get("stage")
        or receipt.get("settlement_state") != "settled"
    ):
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_RECEIPT_BINDING_MISMATCH")
    artifact_refs = receipt.get("artifact_refs")
    result_ref = artifact_refs.get(artifact_path) if isinstance(artifact_refs, Mapping) else None
    if not isinstance(result_ref, str):
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_RESULT_MISSING")
    result = _object(cas.read_bytes(result_ref))
    receipt = {**receipt, "receipt_ref": ref, "receipt_hash": ref.rsplit("/", 1)[-1]}
    return index_probe_experience(
        database_path=retrieval_db, result=result, receipt=receipt,
        archive_db=archive_db, cas_root=cas_root,
        asset_fingerprint=probe_asset_fingerprint(receipt, cas=cas),
        result_artifact_path=artifact_path,
    )


def _object(payload: bytes) -> dict:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_EVIDENCE_INVALID") from exc
    if not isinstance(value, dict):
        raise ProbeRetrievalError("DIAGNOSTIC_MEMORY_EVIDENCE_INVALID")
    return value
