"""Reconstruct frozen open-method inputs from a content-addressed receipt.

This is an input-integrity check, not a formal verdict or proof of execution.
Checkpoint bytes and raw evaluator observations require separate verification.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import canonical_bytes, sha256_bytes


class OpenMethodEvidenceError(ValueError):
    """The receipt or its frozen inputs cannot be independently reconstructed."""


def _read_object(cas: ContentAddressedStore, ref: object) -> dict[str, Any]:
    try:
        value = json.loads(cas.read_bytes(ref))
    except (ArchiveInvariantError, OSError, TypeError, ValueError) as exc:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_OBJECT_INVALID") from exc
    if not isinstance(value, dict):
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_OBJECT_INVALID")
    return value


def _safe_paths(mapping: object) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_MANIFEST_INVALID")
    for name, ref in mapping.items():
        if not isinstance(name, str) or not name or not isinstance(ref, str):
            raise OpenMethodEvidenceError("OPEN_EVIDENCE_MANIFEST_INVALID")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise OpenMethodEvidenceError("OPEN_EVIDENCE_MANIFEST_INVALID")
    return mapping


def _check_member(cas: ContentAddressedStore, ref: str, digest: str) -> None:
    if ref != f"cas://sha256/{digest}":
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_BINDING_MISMATCH")
    try:
        cas.read_bytes(ref)
    except (ArchiveInvariantError, OSError, ValueError) as exc:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_MEMBER_INVALID") from exc


def read_frozen_inputs(cas: ContentAddressedStore, execution_ref: str) -> dict[str, Any]:
    """Read and check receipt-bound snapshots, leaving effect claims undecided."""
    receipt = _read_object(cas, execution_ref)
    try:
        validate_document("open_method_study_execution", receipt)
    except ContractValidationError as exc:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_RECEIPT_INVALID") from exc
    identity = {k: v for k, v in receipt.items() if k not in {"execution_id", "resource_usage"}}
    if receipt["execution_id"] != "open-execution-" + sha256_bytes(canonical_bytes(identity))[:24]:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_RECEIPT_ID_MISMATCH")
    snapshot = _read_object(cas, receipt.get("frozen_inputs_manifest_ref"))
    if set(snapshot) != {"study_files", "inputs", "verifier_implementation", "checkpoint_digest"}:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_MANIFEST_INVALID")
    study_files = _safe_paths(snapshot["study_files"])
    inputs = snapshot["inputs"]
    implementation = _safe_paths(snapshot["verifier_implementation"])
    if not isinstance(inputs, dict) or set(inputs) != {"train_split", "selection_split", "confirmation_split", "verifier"}:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_MANIFEST_INVALID")
    study = _read_object(cas, study_files.get("study.json"))
    if study.get("artifact_type") != "verdiwm-open-method-study" or study.get("state") != "ready_for_calibration":
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_STUDY_INVALID")
    study_id = "open-study-" + sha256_bytes(canonical_bytes({k: v for k, v in study.items() if k != "study_id"}))[:24]
    if study.get("study_id") != study_id or receipt["study_id"] != study_id:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_STUDY_ID_MISMATCH")
    expected = _safe_paths(study.get("file_sha256"))
    if set(study_files) != set(expected) | {"study.json"}:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_STUDY_FILES_MISMATCH")
    for name, digest in expected.items():
        _check_member(cas, study_files[name], digest)
    binding = study.get("experiment_binding")
    if not isinstance(binding, dict) or receipt.get("input_binding") != binding or snapshot["checkpoint_digest"] != binding.get("checkpoint_digest"):
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_BINDING_MISMATCH")
    for name, ref in inputs.items():
        _check_member(cas, ref, binding.get(f"{name}_digest"))
    verifier = _read_object(cas, inputs["verifier"])
    try:
        validate_document("open_method_verifier", verifier)
    except ContractValidationError as exc:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_VERIFIER_INVALID") from exc
    verifier_id = "open-verifier-" + sha256_bytes(canonical_bytes({k: v for k, v in verifier.items() if k != "verifier_id"}))[:24]
    if verifier["verifier_id"] != verifier_id:
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_VERIFIER_ID_MISMATCH")
    files = {row["relative_path"]: row["sha256"] for row in verifier["implementation_files"]}
    if len(files) != len(verifier["implementation_files"]) or set(files) != set(implementation):
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_VERIFIER_FILES_MISMATCH")
    for name, digest in files.items():
        _check_member(cas, implementation[name], digest)
    splits = {name: _read_object(cas, inputs[name]) for name in ("train_split", "selection_split", "confirmation_split")}
    units = []
    for split in splits.values():
        ids = split.get("episode_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise OpenMethodEvidenceError("OPEN_EVIDENCE_SPLIT_INVALID")
        units.append(set(ids))
    if any(units[i] & units[j] for i in range(3) for j in range(i + 1, 3)):
        raise OpenMethodEvidenceError("OPEN_EVIDENCE_SPLIT_OVERLAP")
    return {"receipt": receipt, "study": study, "verifier": verifier, "splits": splits}
