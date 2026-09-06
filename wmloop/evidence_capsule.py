"""Portable, path-free Evidence Capsule exchange.

An Evidence Capsule is a deliberately lossy projection of a private receipt.
Archive, CAS, the Evidence Graph, and frozen verifiers remain authoritative;
an imported capsule is only a routing prior and can never create a trial or a
target-side verdict.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "verdiwm-evidence-capsule"
IMPORT_RECEIPT_ARTIFACT_TYPE = "verdiwm-evidence-capsule-import-receipt"
ROUTING_AUTHORITY = "prior_only"
STATES = frozenset({"verified", "exploratory", "null", "harmful", "abstained", "disputed"})
CLAIM_BOUNDARY = (
    "This capsule is a portable evidence projection and routing prior, not a "
    "target-side verdict or execution authorization."
)

__all__ = [
    "EvidenceCapsuleError",
    "export_evidence_capsule",
    "validate_evidence_capsule",
    "import_evidence_capsule",
    "export_capsule",
    "validate_capsule",
    "import_capsule",
    "write_json_atomic",
    "main",
]

_FORBIDDEN_KEYS = frozenset(
    {
        "command",
        "commands",
        "argv",
        "cwd",
        "environment",
        "environment_keys",
        "runtime_path",
        "runtime_paths",
        "checkpoint_path",
        "checkpoint_paths",
        "checkpoint_name",
        "checkpoint_names",
        "dataset_path",
        "dataset_paths",
        "raw_video",
        "raw_videos",
        "raw_media",
        "raw_media_path",
        "raw_media_paths",
        "media_path",
        "media_paths",
    }
)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_URI_PRIVATE = re.compile(r"^(?:file|ssh)://", re.IGNORECASE)


class EvidenceCapsuleError(ValueError):
    """Raised when a capsule cannot be safely exchanged."""


def export_evidence_capsule(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    publisher: str | None = None,
    license: str | None = None,
) -> dict[str, Any]:
    """Project a private JSON receipt into a portable capsule atomically."""

    source_path = Path(source).expanduser()
    if not source_path.is_file() or source_path.is_symlink():
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SOURCE_INVALID")
    raw = source_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SOURCE_NOT_JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SOURCE_OBJECT_REQUIRED")

    source_sha256 = hashlib.sha256(raw).hexdigest()
    projected = _project_source(payload)
    source_artifact_type = payload.get("artifact_type", "unknown")
    if not isinstance(source_artifact_type, str) or not source_artifact_type:
        source_artifact_type = "unknown"
    state = _source_state(payload)
    capsule: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "capsule_id": "capsule-" + source_sha256,
        "state": state,
        "publisher": _metadata_string(publisher, payload.get("publisher"), field="publisher"),
        "license": _metadata_string(license, payload.get("license"), field="license"),
        "source_artifact_type": source_artifact_type,
        "source_sha256": source_sha256,
        "evidence": projected,
        "relations": _project_value(payload.get("relations", []), "relations"),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    validate_evidence_capsule(capsule)
    write_json_atomic(output, capsule)
    return capsule


def validate_evidence_capsule(capsule: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    """Validate and return a capsule, failing closed on private/runtime data."""

    payload = _load_json(capsule)
    if not isinstance(payload, Mapping):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_OBJECT_REQUIRED")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SCHEMA_UNSUPPORTED")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_ARTIFACT_TYPE_INVALID")
    capsule_id = payload.get("capsule_id")
    if not isinstance(capsule_id, str) or not capsule_id:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_ID_INVALID")
    if payload.get("state") not in STATES:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_STATE_INVALID")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or not _HEX64.fullmatch(source_sha256):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_HASH_INVALID")
    source_type = payload.get("source_artifact_type")
    if not isinstance(source_type, str) or not source_type:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SOURCE_TYPE_INVALID")
    if not isinstance(payload.get("publisher"), str) or not payload["publisher"]:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PUBLISHER_INVALID")
    if not isinstance(payload.get("license"), str) or not payload["license"]:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_LICENSE_INVALID")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_CLAIM_BOUNDARY_INVALID")
    if not isinstance(payload.get("evidence"), Mapping):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_EVIDENCE_INVALID")
    if not isinstance(payload.get("relations"), list):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_RELATIONS_INVALID")
    _validate_value(payload)
    return copy.deepcopy(dict(payload))


def import_evidence_capsule(
    capsule: str | os.PathLike[str], destination_root: str | os.PathLike[str]
) -> dict[str, Any]:
    """Validate and index a capsule locally as a prior-only artifact."""

    validated = validate_evidence_capsule(capsule)
    root = Path(destination_root).expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    capsule_bytes = _canonical_json(validated)
    capsule_sha256 = hashlib.sha256(capsule_bytes).hexdigest()
    capsule_dir = root / "capsules"
    capsule_dir.mkdir(mode=0o700, exist_ok=True)
    capsule_path = capsule_dir / f"{validated['capsule_id']}.json"
    write_json_atomic(capsule_path, validated)

    index_path = root / "evidence-capsule-index.json"
    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "verdiwm-evidence-capsule-index",
        "routing_authority": ROUTING_AUTHORITY,
        "capsules": [],
    }
    if index_path.exists():
        existing = _load_json(index_path)
        if isinstance(existing, Mapping):
            index.update(existing)
    rows = index.get("capsules")
    if not isinstance(rows, list):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_INDEX_INVALID")
    row = {
        "capsule_id": validated["capsule_id"],
        "capsule_sha256": capsule_sha256,
        "state": validated["state"],
        "source_artifact_type": validated["source_artifact_type"],
        "capsule_ref": f"urn:verdiwm:evidence-capsule:{validated['capsule_id']}",
    }
    rows[:] = [item for item in rows if isinstance(item, Mapping) and item.get("capsule_id") != row["capsule_id"]]
    rows.append(row)
    index["routing_authority"] = ROUTING_AUTHORITY
    write_json_atomic(index_path, index)

    receipt = {
        "artifact_type": IMPORT_RECEIPT_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "capsule_id": validated["capsule_id"],
        "capsule_sha256": capsule_sha256,
        "capsule_ref": row["capsule_ref"],
        "routing_authority": ROUTING_AUTHORITY,
        "target_side_verdict": None,
        "trial_authorized": False,
        "indexed": True,
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = root / "verdiwm-evidence-capsule-import-receipt.json"
    write_json_atomic(receipt_path, receipt)
    return {"receipt": receipt, "receipt_path": str(receipt_path), "index_path": str(index_path)}


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write JSON with replace semantics and restrictive directory defaults."""

    destination = Path(path).expanduser().resolve()
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_OUTPUT_INVALID")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(_canonical_json(payload) + b"\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return destination


def _project_source(source: Mapping[str, Any]) -> dict[str, Any]:
    evidence = source.get("evidence", source)
    if not isinstance(evidence, Mapping):
        evidence = {"value": evidence}
    # Receipts in the wild place semantic fields either at the root or under
    # ``evidence``.  Merge both locations while excluding capsule metadata so
    # the projection keeps model family/IRG/probe fields without copying
    # runtime bindings.
    combined: dict[str, Any] = {
        key: value
        for key, value in source.items()
        if key not in {"artifact_type", "state", "verdict_state", "publisher", "license", "relations", "evidence"}
    }
    combined.update(evidence)
    projected = _project_value(combined, "evidence")
    if not isinstance(projected, dict):
        return {"value": projected}
    return projected


def _project_value(value: Any, key: str) -> Any:
    if _forbidden_key(key):
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                continue
            if _forbidden_key(child_key):
                continue
            result[child_key] = _project_value(child_value, child_key)
        return result
    if isinstance(value, list):
        return [_project_value(item, key) for item in value]
    if isinstance(value, tuple):
        return [_project_value(item, key) for item in value]
    return value


def _validate_value(value: Any, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise EvidenceCapsuleError("EVIDENCE_CAPSULE_KEY_INVALID")
            if _forbidden_key(child_key):
                raise EvidenceCapsuleError(f"EVIDENCE_CAPSULE_PRIVATE_FIELD:{child_key}")
            lowered = child_key.lower()
            if (lowered.endswith("_hash") or lowered.endswith("_sha256") or lowered in {"hash", "sha256", "digest"}) and child_value is not None:
                if not isinstance(child_value, str) or not _HEX64.fullmatch(child_value):
                    raise EvidenceCapsuleError(f"EVIDENCE_CAPSULE_HASH_INVALID:{child_key}")
            _validate_value(child_value, child_key)
    elif isinstance(value, list):
        for item in value:
            _validate_value(item, key)
    elif isinstance(value, str):
        if _URI_PRIVATE.match(value) or _is_absolute_path(value):
            raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PRIVATE_REFERENCE")


def _forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _FORBIDDEN_KEYS or lowered.endswith("_path") or lowered.endswith("_paths")


def _is_absolute_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("\\") or bool(_WINDOWS_ABSOLUTE.match(value))


def _source_state(source: Mapping[str, Any]) -> str:
    state = source.get("state", source.get("verdict_state", "exploratory"))
    if state in STATES:
        return str(state)
    if state in {"PASS", "pass", "success", "accepted"}:
        return "verified"
    if state in {"FAIL", "fail", "rejected"}:
        return "null"
    return "exploratory"


def _metadata_string(*values: Any, field: str) -> str:
    for value in values:
        if value is not None:
            if not isinstance(value, str) or not value or _is_absolute_path(value) or _URI_PRIVATE.match(value):
                raise EvidenceCapsuleError(f"EVIDENCE_CAPSULE_{field.upper()}_INVALID")
            return value
    return "unknown"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(value: Mapping[str, Any] | str | os.PathLike[str]) -> Any:
    if isinstance(value, Mapping):
        return value
    path = Path(value).expanduser()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_READ_FAILED") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verdiwm-evidence-capsule")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--publisher")
    export.add_argument("--license")
    validate = sub.add_parser("validate")
    validate.add_argument("--capsule", required=True, type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("--capsule", required=True, type=Path)
    imp.add_argument("--destination-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_evidence_capsule(args.source, args.output, publisher=args.publisher, license=args.license)
        elif args.command == "validate":
            result = validate_evidence_capsule(args.capsule)
        else:
            result = import_evidence_capsule(args.capsule, args.destination_root)
    except EvidenceCapsuleError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


# Short aliases keep the exchange API convenient for callers while the
# explicit names remain the documented surface.
export_capsule = export_evidence_capsule
validate_capsule = validate_evidence_capsule
import_capsule = import_evidence_capsule


if __name__ == "__main__":
    raise SystemExit(main())
