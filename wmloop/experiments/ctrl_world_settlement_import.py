"""Import terminal Ctrl-World mechanism screens into Archive and CAS."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import (
    ArchiveInvariantError,
    ArchiveStore,
    ContentAddressedStore,
    SettledTrialRecord,
)


class CtrlWorldSettlementImportError(RuntimeError):
    """A source settlement failed the conservative import boundary."""


_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_CAS_PREFIX = "cas://sha256/"


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    sha256: str
    payload: bytes
    role: str

    @property
    def uri(self) -> str:
        return f"{_CAS_PREFIX}{self.sha256}"


@dataclass(frozen=True)
class PreparedSettlement:
    source: BoundDocument
    report: BoundDocument
    documents: tuple[BoundDocument, ...]
    context_bytes: bytes
    receipt_bytes: bytes
    evidence_record: dict[str, Any]
    trial_record: SettledTrialRecord


def import_ctrl_world_settlements(
    *,
    input_root: Path,
    archive_db: Path,
    cas_root: Path,
    output_root: Path,
    allowed_roots: Sequence[Path] = (),
    goal_id: str = "ctrl_world_predictive_quality_pilot_v2",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all settlement evidence, then import it idempotently."""

    source_root = Path(input_root).expanduser().resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise CtrlWorldSettlementImportError("SETTLEMENT_INPUT_ROOT_INVALID")
    roots = tuple(
        dict.fromkeys(
            [source_root]
            + [Path(path).expanduser().resolve() for path in allowed_roots]
        )
    )
    settlement_paths = sorted(source_root.rglob("SETTLEMENT.json"))
    if not settlement_paths:
        raise CtrlWorldSettlementImportError("SETTLEMENTS_NOT_FOUND")
    prepared = [
        _prepare_settlement(path, allowed_roots=roots, goal_id=goal_id)
        for path in settlement_paths
    ]
    trial_ids = [item.trial_record.trial_id for item in prepared]
    if len(trial_ids) != len(set(trial_ids)):
        raise CtrlWorldSettlementImportError("SETTLEMENT_TRIAL_ID_COLLISION")

    destination = _prepare_output_root(output_root)
    imported: list[str] = []
    skipped: list[str] = []
    if not dry_run:
        archive = ArchiveStore(archive_db)
        cas = ContentAddressedStore(cas_root)
        existing = set(archive.visible_settled_trials())
        for item in prepared:
            trial_id = item.trial_record.trial_id
            if trial_id in existing:
                skipped.append(trial_id)
                continue
            for document in item.documents:
                ref = cas.put_bytes(document.payload, media_type="application/json")
                if ref.sha256 != document.sha256:
                    raise CtrlWorldSettlementImportError("SETTLEMENT_CAS_HASH_MISMATCH")
            context_ref = cas.put_bytes(
                item.context_bytes, media_type="application/json"
            )
            receipt_ref = cas.put_bytes(
                item.receipt_bytes, media_type="application/json"
            )
            if (
                context_ref.uri != item.trial_record.failure_context_ref
                or receipt_ref.uri != item.trial_record.receipt_ref
            ):
                raise CtrlWorldSettlementImportError("SETTLEMENT_GENERATED_REF_MISMATCH")
            try:
                archive.record_settled_trial(item.trial_record)
                for document in item.documents:
                    archive.record_artifact_reference(document.uri)
            except ArchiveInvariantError as exc:
                raise CtrlWorldSettlementImportError(
                    f"SETTLEMENT_ARCHIVE_REJECTED:{trial_id}:{exc}"
                ) from exc
            imported.append(trial_id)
            existing.add(trial_id)

    for item in prepared:
        record = dict(item.evidence_record)
        record["import_state"] = (
            "planned"
            if dry_run
            else "imported"
            if item.trial_record.trial_id in imported
            else "already_present"
        )
        _write_json_atomic(
            destination / "records" / f"{item.trial_record.trial_id}.json",
            record,
        )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-settlement-import-manifest",
        "state": "planned" if dry_run else "completed",
        "dry_run": dry_run,
        "input_root": str(source_root),
        "archive_db": str(Path(archive_db).expanduser().resolve()),
        "cas_root": str(Path(cas_root).expanduser().resolve()),
        "goal_id": goal_id,
        "settlement_count": len(prepared),
        "planned_trial_ids": trial_ids,
        "imported_trial_ids": imported,
        "already_present_trial_ids": skipped,
        "evidence_scope": "context_local_exploratory",
        "promotion_authorized": False,
        "claim_boundary": "Imported rows are terminal, context-local exploratory evidence. They record positive, null, and harmful mechanism boundaries but cannot populate promoted cell priors or authorize a larger run.",
    }
    _write_json_atomic(destination / "manifest.json", manifest)
    return manifest


def _prepare_settlement(
    path: Path, *, allowed_roots: Sequence[Path], goal_id: str
) -> PreparedSettlement:
    source = _bound_document(
        path,
        expected_sha256=None,
        role="settlement",
        allowed_roots=allowed_roots,
    )
    settlement = _decode_object(source, "SETTLEMENT_JSON_INVALID")
    experiment_id, candidate, source_state, reason = _settlement_identity(settlement)
    paired = _path_hash_pairs(settlement)
    report_path, report_hash = _report_binding(settlement)
    report = _bound_document(
        report_path,
        expected_sha256=report_hash,
        role="evaluation_report",
        allowed_roots=allowed_roots,
    )
    report_payload = _decode_object(report, "SETTLEMENT_REPORT_INVALID")
    if str(report_payload.get("experiment_id")) != experiment_id:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_REPORT_EXPERIMENT_MISMATCH:{experiment_id}"
        )

    documents: dict[Path, BoundDocument] = {source.path: source, report.path: report}
    for role, evidence_path, expected_hash in paired:
        document = _bound_document(
            evidence_path,
            expected_sha256=expected_hash,
            role=role,
            allowed_roots=allowed_roots,
        )
        existing = documents.get(document.path)
        if existing is not None and existing.sha256 != document.sha256:
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_EVIDENCE_HASH_CONFLICT:{document.path}"
            )
        documents[document.path] = document

    input_receipts = report_payload.get("input_receipts")
    if not isinstance(input_receipts, list) or not input_receipts:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_REPORT_RECEIPTS_MISSING:{experiment_id}"
        )
    for index, raw in enumerate(input_receipts):
        if not isinstance(raw, Mapping):
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_REPORT_RECEIPT_INVALID:{experiment_id}:{index}"
            )
        receipt_path = raw.get("path")
        receipt_hash = raw.get("sha256")
        if not isinstance(receipt_path, str) or not isinstance(receipt_hash, str):
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_REPORT_RECEIPT_INVALID:{experiment_id}:{index}"
            )
        document = _bound_document(
            Path(receipt_path),
            expected_sha256=receipt_hash,
            role="report_input_receipt",
            allowed_roots=allowed_roots,
        )
        documents[document.path] = document

    receipt_dir = source.path.parent / "receipts"
    if not receipt_dir.is_dir() or receipt_dir.is_symlink():
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_RUNTIME_RECEIPTS_MISSING:{experiment_id}"
        )
    runtime_paths = sorted(receipt_dir.glob("*.json"))
    if not runtime_paths:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_RUNTIME_RECEIPTS_MISSING:{experiment_id}"
        )
    for runtime_path in runtime_paths:
        document = _bound_document(
            runtime_path,
            expected_sha256=None,
            role="runtime_receipt",
            allowed_roots=allowed_roots,
        )
        _decode_object(document, "SETTLEMENT_RUNTIME_RECEIPT_INVALID")
        documents[document.path] = document

    trial_id = _trial_id(experiment_id, source.sha256)
    proposal_id = f"settlement-import:{experiment_id}:{source.sha256[:16]}"
    context = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-settlement-context",
        "trial_id": trial_id,
        "experiment_id": experiment_id,
        "candidate_id": candidate,
        "goal_id": goal_id,
        "model_family": "ctrl_world",
        "environment": "droid_replay_context_local",
        "source_state": source_state,
        "evidence_scope": "context_local_exploratory",
        "dominant_failure_mode": _dominant_failure_mode(settlement),
        "claim_boundary": _claim_boundary(settlement),
    }
    context_bytes = _canonical(context)
    context_hash = _sha256(context_bytes)
    ordered_documents = tuple(
        sorted(documents.values(), key=lambda document: str(document.path))
    )
    import_receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-settlement-import-receipt",
        "trial_id": trial_id,
        "proposal_id": proposal_id,
        "experiment_id": experiment_id,
        "candidate_id": candidate,
        "goal_id": goal_id,
        "source_state": source_state,
        "verdict": "NOT_PROMOTED",
        "promotion_authorized": False,
        "confirmation_authorized": False,
        "evidence_scope": "context_local_exploratory",
        "gpu_hours": None,
        "source_settlement": _document_row(source),
        "evaluation_report": _document_row(report),
        "bound_documents": [_document_row(document) for document in ordered_documents],
        "failure_context_ref": f"{_CAS_PREFIX}{context_hash}",
        "claim_boundary": "The source screen is terminal but not promoted. Unknown GPU cost is recorded as unavailable and archived as 0.0 only to satisfy the non-negative ledger field; it is not a measured cost.",
    }
    receipt_bytes = _canonical(import_receipt)
    receipt_hash = _sha256(receipt_bytes)
    plan_hash = next(
        (
            document.sha256
            for document in ordered_documents
            if "plan" in document.role
        ),
        source.sha256,
    )
    trial_record = SettledTrialRecord(
        trial_id=trial_id,
        proposal_id=proposal_id,
        goal_id=goal_id,
        library_version="ctrl-world-mechanism-settlements-v1",
        failure_context_ref=f"{_CAS_PREFIX}{context_hash}",
        verdict_ref=source.uri,
        receipt_ref=f"{_CAS_PREFIX}{receipt_hash}",
        gpu_hours=0.0,
        hypothesis_hash=_sha256(
            _canonical({"candidate": candidate, "decision_reason": reason})
        ),
        impl_diff_hash=plan_hash,
        evaluator_hash=report.sha256,
        settlement_state="settled",
        receipt_hash=receipt_hash,
        exploratory=True,
    )
    evidence_record = {
        "schema_version": 1,
        "artifact_type": "verdiwm-imported-settlement-evidence",
        "trial_id": trial_id,
        "proposal_id": proposal_id,
        "experiment_id": experiment_id,
        "candidate_id": candidate,
        "goal_id": goal_id,
        "model_family": "ctrl_world",
        "environment": "droid_replay_context_local",
        "settlement_state": "settled",
        "source_state": source_state,
        "verdict": "NOT_PROMOTED",
        "evidence_scope": "exploratory",
        "promotion_authorized": False,
        "receipt_ref": trial_record.receipt_ref,
        "verdict_ref": trial_record.verdict_ref,
        "failure_context_ref": trial_record.failure_context_ref,
        "evaluation_report_ref": report.uri,
        "source_settlement_path": str(source.path),
        "source_settlement_sha256": source.sha256,
        "claim_boundary": _claim_boundary(settlement),
    }
    return PreparedSettlement(
        source=source,
        report=report,
        documents=ordered_documents,
        context_bytes=context_bytes,
        receipt_bytes=receipt_bytes,
        evidence_record=evidence_record,
        trial_record=trial_record,
    )


def _settlement_identity(
    settlement: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    artifact_type = settlement.get("artifact_type")
    experiment_id = settlement.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CtrlWorldSettlementImportError("SETTLEMENT_EXPERIMENT_ID_INVALID")
    if artifact_type == "verdiwm-mechanism-experiment-settlement":
        if settlement.get("state") != "settled_not_promoted":
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_PROMOTION_BOUNDARY_INVALID:{experiment_id}"
            )
        candidate = settlement.get("candidate")
        reason = settlement.get("decision_reason")
        if not isinstance(candidate, str) or not isinstance(reason, str):
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_DECISION_INVALID:{experiment_id}"
            )
        return experiment_id, candidate, "settled_not_promoted", reason
    if artifact_type == "verdiwm-ctrl-world-cclvr-heldout-settlement-v1":
        promotion = settlement.get("promotion")
        if (
            settlement.get("state") != "completed"
            or settlement.get("confirmation_authorized") is not False
            or not isinstance(promotion, Mapping)
            or promotion.get("state") != "not_promoted"
            or promotion.get("confirmation_authorized") is not False
        ):
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_PROMOTION_BOUNDARY_INVALID:{experiment_id}"
            )
        candidate = promotion.get("candidate")
        if not isinstance(candidate, str):
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_DECISION_INVALID:{experiment_id}"
            )
        failed = promotion.get("failed_checks")
        reason = "failed promotion checks: " + ",".join(
            str(value) for value in failed if isinstance(value, str)
        ) if isinstance(failed, list) else "promotion checks failed"
        return experiment_id, candidate, "completed_not_promoted", reason
    raise CtrlWorldSettlementImportError(
        f"SETTLEMENT_ARTIFACT_UNSUPPORTED:{experiment_id}"
    )


def _report_binding(settlement: Mapping[str, Any]) -> tuple[Path, str]:
    if settlement.get("artifact_type") == "verdiwm-mechanism-experiment-settlement":
        evidence = settlement.get("evidence")
        if not isinstance(evidence, Mapping):
            raise CtrlWorldSettlementImportError("SETTLEMENT_EVIDENCE_INVALID")
        path = evidence.get("evaluation_report")
        digest = evidence.get("evaluation_report_sha256")
    else:
        path = settlement.get("report")
        digest = settlement.get("report_sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise CtrlWorldSettlementImportError("SETTLEMENT_REPORT_BINDING_INVALID")
    return Path(path), digest


def _path_hash_pairs(
    payload: Mapping[str, Any], *, prefix: str = "settlement"
) -> list[tuple[str, Path, str]]:
    pairs: list[tuple[str, Path, str]] = []
    for key, value in payload.items():
        role = f"{prefix}.{key}"
        if key.endswith("_sha256") and isinstance(value, str):
            path_value = payload.get(key[: -len("_sha256")])
            if isinstance(path_value, str):
                pairs.append((role, Path(path_value), value))
        elif isinstance(value, Mapping):
            pairs.extend(_path_hash_pairs(value, prefix=role))
    return pairs


def _bound_document(
    path: Path,
    *,
    expected_sha256: str | None,
    role: str,
    allowed_roots: Sequence[Path],
) -> BoundDocument:
    candidate = Path(path).expanduser()
    try:
        metadata_before = os.lstat(candidate)
    except OSError as exc:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_MISSING:{candidate}"
        ) from exc
    if stat.S_ISLNK(metadata_before.st_mode) or not stat.S_ISREG(metadata_before.st_mode):
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_UNSAFE:{candidate}"
        )
    resolved = candidate.resolve()
    if not any(root == resolved or root in resolved.parents for root in allowed_roots):
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_OUTSIDE_ALLOWED_ROOTS:{resolved}"
        )
    if metadata_before.st_size > _MAX_DOCUMENT_BYTES:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_TOO_LARGE:{resolved}"
        )
    payload = resolved.read_bytes()
    metadata_after = os.lstat(resolved)
    if (
        metadata_before.st_dev,
        metadata_before.st_ino,
        metadata_before.st_size,
        metadata_before.st_mtime_ns,
    ) != (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
        metadata_after.st_mtime_ns,
    ):
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_CHANGED:{resolved}"
        )
    digest = _sha256(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CtrlWorldSettlementImportError(
            f"SETTLEMENT_EVIDENCE_HASH_MISMATCH:{resolved}"
        )
    if resolved.suffix == ".json" or resolved.name == "SETTLEMENT.json":
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CtrlWorldSettlementImportError(
                f"SETTLEMENT_EVIDENCE_JSON_INVALID:{resolved}"
            ) from exc
    return BoundDocument(path=resolved, sha256=digest, payload=payload, role=role)


def _decode_object(document: BoundDocument, code: str) -> dict[str, Any]:
    try:
        value = json.loads(document.payload)
    except json.JSONDecodeError as exc:
        raise CtrlWorldSettlementImportError(code) from exc
    if not isinstance(value, dict):
        raise CtrlWorldSettlementImportError(code)
    return value


def _dominant_failure_mode(settlement: Mapping[str, Any]) -> str:
    causal = settlement.get("causal_interpretation")
    if isinstance(causal, Mapping) and isinstance(
        causal.get("dominant_failure_mode"), str
    ):
        return str(causal["dominant_failure_mode"])
    promotion = settlement.get("promotion")
    if isinstance(promotion, Mapping) and isinstance(
        promotion.get("failed_checks"), list
    ):
        return "promotion_gates_failed:" + ",".join(
            str(value) for value in promotion["failed_checks"]
        )
    return "not_promoted"


def _claim_boundary(settlement: Mapping[str, Any]) -> str:
    causal = settlement.get("causal_interpretation")
    if isinstance(causal, Mapping) and isinstance(causal.get("confidence_limit"), str):
        return str(causal["confidence_limit"])
    promotion = settlement.get("promotion")
    if isinstance(promotion, Mapping) and isinstance(promotion.get("claim_boundary"), str):
        return str(promotion["claim_boundary"])
    return "Context-local exploratory settlement; no promotion or confirmation is authorized."


def _trial_id(experiment_id: str, digest: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", experiment_id.casefold()).strip("-")
    return f"ctrl-world-settlement-{slug[:80]}-{digest[:12]}"


def _document_row(document: BoundDocument) -> dict[str, str]:
    return {
        "role": document.role,
        "path": str(document.path),
        "sha256": document.sha256,
        "cas_ref": document.uri,
    }


def _prepare_output_root(path: Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise CtrlWorldSettlementImportError("SETTLEMENT_IMPORT_OUTPUT_INVALID")
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()) and not manifest_path.is_file():
        raise CtrlWorldSettlementImportError("SETTLEMENT_IMPORT_OUTPUT_NOT_OWNED")
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CtrlWorldSettlementImportError(
                "SETTLEMENT_IMPORT_OUTPUT_MANIFEST_INVALID"
            ) from exc
        if existing.get("artifact_type") != "verdiwm-ctrl-world-settlement-import-manifest":
            raise CtrlWorldSettlementImportError(
                "SETTLEMENT_IMPORT_OUTPUT_MANIFEST_INVALID"
            )
    (destination / "records").mkdir(mode=0o700, parents=True, exist_ok=True)
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
