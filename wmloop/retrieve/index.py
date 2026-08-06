"""Small, provenance-first retrieval index for diagnostic experiences.

The index is deliberately a projection, not a second archive.  Every row is
bound to a settled auto-experiment receipt and a CAS result reference.  A
query returns only rows whose receipt and result still resolve through CAS;
textual similarity or an unreceipted JSON file is never enough to influence
candidate ordering.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ProbeRetrievalError(ValueError):
    """A probe experience or retrieval index invariant failed closed."""


@dataclass(frozen=True)
class ProbeExperience:
    """One failure-signature projection with immutable artifact provenance."""

    probe_id: str
    model_family: str
    runtime_capability: str
    failure_signature: str
    asset_fingerprint: str | None
    primitive: str | None
    stage: str
    verdict: str
    metric_outcome: float | None
    archive_trial_id: str
    receipt_ref: str
    result_ref: str
    receipt_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "model_family": self.model_family,
            "runtime_capability": self.runtime_capability,
            "failure_signature": self.failure_signature,
            "asset_fingerprint": self.asset_fingerprint,
            "primitive": self.primitive,
            "stage": self.stage,
            "verdict": self.verdict,
            "metric_outcome": self.metric_outcome,
            "archive_trial_id": self.archive_trial_id,
            "receipt_ref": self.receipt_ref,
            "result_ref": self.result_ref,
            "receipt_hash": self.receipt_hash,
        }


def index_probe_experience(
    *,
    database_path: Path,
    result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    archive_db: Path,
    cas_root: Path,
    asset_fingerprint: str | None = None,
    result_artifact_path: str = "result.json",
) -> int:
    """Index every signature from one settled probe result.

    Returns the number of newly inserted signature rows.  Existing identical
    rows are idempotent and do not count as new observations.
    """

    _validate_result(result)
    _validate_receipt(receipt)
    archive = ArchiveStore(Path(archive_db))
    archive_trial_id = str(receipt["archive_trial_id"])
    if archive_trial_id not in archive.visible_settled_trials():
        raise ProbeRetrievalError("PROBE_RETRIEVAL_ARCHIVE_SETTLEMENT_MISSING")
    receipt_ref = str(receipt["receipt_ref"])
    result_ref = _result_ref(receipt, artifact_path=result_artifact_path)
    cas = ContentAddressedStore(Path(cas_root))
    receipt_bytes = cas.read_bytes(receipt_ref)
    if hashlib.sha256(receipt_bytes).hexdigest() != str(receipt["receipt_hash"]):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_HASH_MISMATCH")
    stored_receipt = _json_object(receipt_bytes, "PROBE_RETRIEVAL_RECEIPT_INVALID")
    if stored_receipt.get("archive_trial_id") != archive_trial_id:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_ID_MISMATCH")
    stored_result = _json_object(cas.read_bytes(result_ref), "PROBE_RETRIEVAL_RESULT_INVALID")
    if stored_result.get("probe_id") != result.get("probe_id"):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_ID_MISMATCH")

    signatures = result["failure_signatures"]
    metrics = result.get("metrics")
    metric_outcome = _metric_outcome(metrics)
    primitive = _optional_string(result.get("primitive"))
    rows = [
        ProbeExperience(
            probe_id=str(result["probe_id"]),
            model_family=str(result["model_family"]),
            runtime_capability=str(result["runtime_capability"]),
            failure_signature=signature,
            asset_fingerprint=asset_fingerprint,
            primitive=primitive,
            stage=str(receipt["stage"]),
            verdict=str(receipt["verdict"]["verdict"]),
            metric_outcome=metric_outcome,
            archive_trial_id=archive_trial_id,
            receipt_ref=receipt_ref,
            result_ref=result_ref,
            receipt_hash=str(receipt["receipt_hash"]),
        )
        for signature in signatures
    ]
    connection = _connect(Path(database_path))
    inserted = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        for row in rows:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO experiences(
                    probe_id, model_family, runtime_capability, failure_signature,
                    asset_fingerprint, primitive, stage, verdict, metric_outcome,
                    archive_trial_id, receipt_ref, result_ref, receipt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.probe_id,
                    row.model_family,
                    row.runtime_capability,
                    row.failure_signature,
                    row.asset_fingerprint,
                    row.primitive,
                    row.stage,
                    row.verdict,
                    row.metric_outcome,
                    row.archive_trial_id,
                    row.receipt_ref,
                    row.result_ref,
                    row.receipt_hash,
                ),
            )
            inserted += int(cursor.rowcount == 1)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return inserted


def retrieve_probe_experiences(
    *,
    database_path: Path,
    model_family: str,
    runtime_capability: str,
    failure_signatures: Sequence[str],
    asset_fingerprint: str | None = None,
    primitive: str | None = None,
    stage: str | None = None,
    verdict: str | None = None,
    exclude_archive_trial_id: str | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    limit: int = 20,
) -> tuple[ProbeExperience, ...]:
    """Retrieve compatible evidence, rechecking receipt/CAS provenance."""

    if not model_family or not runtime_capability or limit < 1 or limit > 500:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_QUERY_INVALID")
    signatures = tuple(dict.fromkeys(value for value in failure_signatures if value))
    if not signatures:
        return ()
    path = Path(database_path).resolve()
    if not path.is_file():
        return ()
    placeholders = ",".join("?" for _ in signatures)
    clauses = [
        "model_family = ?",
        "runtime_capability = ?",
        f"failure_signature IN ({placeholders})",
    ]
    parameters: list[object] = [model_family, runtime_capability, *signatures]
    for column, value in (
        ("asset_fingerprint", asset_fingerprint),
        ("primitive", primitive),
        ("stage", stage),
        ("verdict", verdict),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    if exclude_archive_trial_id is not None:
        clauses.append("archive_trial_id != ?")
        parameters.append(exclude_archive_trial_id)
    query = (
        "SELECT probe_id, model_family, runtime_capability, failure_signature, "
        "asset_fingerprint, primitive, stage, verdict, metric_outcome, "
        "archive_trial_id, receipt_ref, result_ref, receipt_hash "
        "FROM experiences WHERE "
        + " AND ".join(clauses)
        + " ORDER BY CASE verdict WHEN 'PASS' THEN 0 ELSE 1 END, "
        "COALESCE(metric_outcome, -1e300) DESC, id ASC LIMIT ?"
    )
    parameters.append(limit)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        records = [ProbeExperience(**dict(row)) for row in connection.execute(query, parameters).fetchall()]
    finally:
        connection.close()
    if archive_db is None or cas_root is None:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_DURABLE_STORES_REQUIRED")
    archive = ArchiveStore(Path(archive_db))
    cas = ContentAddressedStore(Path(cas_root))
    verified: list[ProbeExperience] = []
    for row in records:
        if row.archive_trial_id not in archive.visible_settled_trials():
            continue
        try:
            receipt_bytes = cas.read_bytes(row.receipt_ref)
            result_bytes = cas.read_bytes(row.result_ref)
        except Exception:
            continue
        if hashlib.sha256(receipt_bytes).hexdigest() != row.receipt_hash:
            continue
        receipt = _json_object(receipt_bytes, "PROBE_RETRIEVAL_RECEIPT_INVALID")
        result = _json_object(result_bytes, "PROBE_RETRIEVAL_RESULT_INVALID")
        if receipt.get("settlement_state") != "settled" or result.get("probe_id") != row.probe_id:
            continue
        verified.append(row)
    return tuple(verified)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            probe_id TEXT NOT NULL,
            model_family TEXT NOT NULL,
            runtime_capability TEXT NOT NULL,
            failure_signature TEXT NOT NULL,
            asset_fingerprint TEXT,
            primitive TEXT,
            stage TEXT NOT NULL,
            verdict TEXT NOT NULL,
            metric_outcome REAL,
            archive_trial_id TEXT NOT NULL,
            receipt_ref TEXT NOT NULL,
            result_ref TEXT NOT NULL,
            receipt_hash TEXT NOT NULL,
            UNIQUE(archive_trial_id, failure_signature)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS experiences_query_idx ON experiences(" 
        "model_family, runtime_capability, failure_signature)"
    )
    connection.commit()
    return connection


def _validate_result(result: Mapping[str, Any]) -> None:
    required = ("schema_version", "artifact_type", "probe_id", "model_family", "runtime_capability", "failure_signatures")
    if result.get("schema_version") != 1 or result.get("artifact_type") != "verdiwm-diagnostic-probe-result":
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_CONTRACT_INVALID")
    for field in required[2:5]:
        if not isinstance(result.get(field), str) or not str(result[field]):
            raise ProbeRetrievalError(f"PROBE_RETRIEVAL_RESULT_FIELD_INVALID:{field}")
    signatures = result.get("failure_signatures")
    if not isinstance(signatures, list) or not signatures or any(not isinstance(item, str) or not item for item in signatures):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_FAILURE_SIGNATURES_INVALID")


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("settlement_state") != "settled" or receipt.get("verdict", {}).get("verdict") not in {"PASS", "VOID"}:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_NOT_SETTLED")
    for field in ("archive_trial_id", "receipt_ref", "receipt_hash", "stage"):
        if not isinstance(receipt.get(field), str) or not str(receipt[field]):
            raise ProbeRetrievalError(f"PROBE_RETRIEVAL_RECEIPT_FIELD_INVALID:{field}")


def _result_ref(
    receipt: Mapping[str, Any], *, artifact_path: str = "result.json"
) -> str:
    refs = receipt.get("artifact_refs")
    if (
        not artifact_path
        or not isinstance(refs, Mapping)
        or not isinstance(refs.get(artifact_path), str)
    ):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_REF_MISSING")
    return str(refs[artifact_path])


def _metric_outcome(metrics: object) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    for key in ("primary", "probe_score", "failure_score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_object(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeRetrievalError(code) from exc
    if not isinstance(value, dict):
        raise ProbeRetrievalError(code)
    return value
