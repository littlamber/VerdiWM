"""Read-only Archive/CAS projection into existing community export records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from wmloop.contracts import validate_document
from wmloop.geometry.settled_evidence import build_settled_evidence
from wmloop.experiments.community_export import export_community_knowledge
from wmloop.experiments.community_files import bounded_read
from wmloop.storage import canonical_bytes, checked_path, atomic_write


class EvidenceProjectionError(ValueError):
    """An archive projection lacks complete, content-addressed settlement evidence."""


def project_archive_evidence(*, archive: Path, cas: Path, output_root: Path,
                             execution_manifest: Path | None = None, max_trials: int = 10_000,
                             max_artifact_bytes: int = 5 * 1024 * 1024) -> dict:
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or not 1 <= max_trials <= 100_000:
        raise EvidenceProjectionError("EVIDENCE_PROJECTION_LIMIT_INVALID")
    if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or not 1024 <= max_artifact_bytes <= 1024 ** 3:
        raise EvidenceProjectionError("EVIDENCE_PROJECTION_SIZE_INVALID")
    database = checked_path(archive, code="EVIDENCE_ARCHIVE_INVALID", error=EvidenceProjectionError)
    cas_root = checked_path(cas, code="EVIDENCE_CAS_INVALID", error=EvidenceProjectionError)
    if not database.is_file() or not cas_root.is_dir():
        raise EvidenceProjectionError("EVIDENCE_SOURCE_MISSING")

    def read_cas(reference):
        if not isinstance(reference, str) or re.fullmatch(r"cas://sha256/[0-9a-f]{64}", reference) is None:
            raise EvidenceProjectionError("EVIDENCE_CAS_REFERENCE_INVALID")
        digest = reference.rsplit("/", 1)[-1]
        path = checked_path(cas_root / "cas" / digest[:2] / digest, code="EVIDENCE_CAS_MEMBER_INVALID", error=EvidenceProjectionError)
        try:
            raw = bounded_read(path, max_artifact_bytes)
        except OSError as exc:
            raise EvidenceProjectionError("EVIDENCE_CAS_MEMBER_MISSING") from exc
        if len(raw) > max_artifact_bytes:
            raise EvidenceProjectionError("EVIDENCE_CAS_SIZE_LIMIT")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise EvidenceProjectionError("EVIDENCE_CAS_DIGEST_MISMATCH")
        return raw

    connection = None
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        # LEFT JOIN exposes incomplete settlements instead of silently omitting them.
        rows = connection.execute("""
            SELECT t.*, r.receipt_ref AS settled_receipt_ref, r.receipt_hash,
                   v.verdict_ref AS settled_verdict_ref, p.proposal_id AS settled_proposal_id,
                   p.goal_id AS settled_goal_id, p.library_version AS settled_library_version
            FROM trials t LEFT JOIN receipts r ON r.trial_id=t.trial_id
            LEFT JOIN verdicts v ON v.trial_id=t.trial_id
            LEFT JOIN proposals p ON p.trial_id=t.trial_id
            ORDER BY t.trial_id LIMIT ?
        """, (max_trials + 1,)).fetchall()
        if len(rows) > max_trials:
            raise EvidenceProjectionError("EVIDENCE_TRIAL_LIMIT_EXCEEDED")
        if not rows:
            raise EvidenceProjectionError("EVIDENCE_NO_SETTLED_TRIALS")
        records = []
        for row in rows:
            settlement = json.loads(row["settlement_json"])
            fingerprint = json.loads(row["fingerprint_json"])
            if (not isinstance(settlement, dict) or not isinstance(fingerprint, dict)
                or settlement.get("state") != "settled"
                or row["receipt_ref"] != row["settled_receipt_ref"]
                or row["verdict_ref"] != row["settled_verdict_ref"]
                or row["proposal_id"] != row["settled_proposal_id"]
                or row["goal_id"] != row["settled_goal_id"]
                or row["library_version"] != row["settled_library_version"]
                or settlement.get("receipt_hash") != row["receipt_hash"]
                or row["receipt_hash"] != str(row["receipt_ref"]).rsplit("/", 1)[-1]):
                raise EvidenceProjectionError("EVIDENCE_SETTLEMENT_BINDING_INVALID")
            read_cas(row["receipt_ref"])
            read_cas(row["failure_context_ref"])
            verdict_bytes = read_cas(row["verdict_ref"])
            try:
                verdict = json.loads(verdict_bytes)
            except (ValueError, UnicodeDecodeError):
                verdict = None
            decision = "UNINTERPRETED"
            if isinstance(verdict, dict) and "verdict" in verdict:
                validate_document("verdict", verdict)
                if verdict["proposal_id"] != row["proposal_id"]:
                    raise EvidenceProjectionError("EVIDENCE_VERDICT_BINDING_INVALID")
                decision = verdict["verdict"]
            records.append(build_settled_evidence(
                receipt_ref=row["receipt_ref"], verdict_ref=row["verdict_ref"], failure_context_ref=row["failure_context_ref"],
                evaluator_sha256=fingerprint["evaluator_hash"], hypothesis_sha256=fingerprint["hypothesis_hash"],
                implementation_sha256=fingerprint["impl_diff_hash"], decision=decision,
                evidence_scope=settlement.get("evidence_scope", "verified")))
    except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise EvidenceProjectionError("EVIDENCE_ARCHIVE_INVALID") from exc
    finally:
        if connection is not None:
            connection.close()
    with tempfile.TemporaryDirectory(prefix="verdiwm-evidence-projection-") as temporary:
        source = Path(temporary)
        for record in records:
            atomic_write(source / (record["evidence_id"] + ".json"), canonical_bytes(record))
        return export_community_knowledge(source_roots=[source], output_root=output_root,
                                         execution_manifest=execution_manifest, max_files=max_trials)
