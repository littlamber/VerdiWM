"""Small, provenance-first retrieval index for diagnostic experiences.

The index is deliberately a projection, not a second archive.  Every row is
bound to a settled auto-experiment receipt and a CAS result reference.  A
query returns only rows whose receipt and declared evidence still resolve through CAS;
textual similarity or an unreceipted JSON file is never enough to influence
candidate ordering.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.receipt_evidence import ReceiptEvidenceError, verify_receipt_evidence
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.diagnose.paired_observation import PairedObservationError, verify_paired_observations
from wmloop.storage import canonical_json_bytes


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
    paired_observations: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        result = {
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
        if self.paired_observations:
            result["paired_observations"] = deepcopy(list(self.paired_observations))
        return result


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
    _validate_receipt_body(stored_receipt)
    if stored_receipt.get("archive_trial_id") != archive_trial_id:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_ID_MISMATCH")
    stored_result = _json_object(cas.read_bytes(result_ref), "PROBE_RETRIEVAL_RESULT_INVALID")
    if stored_result.get("probe_id") != result.get("probe_id"):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_ID_MISMATCH")
    if stored_result != dict(result):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_CONTENT_MISMATCH")
    if any(stored_receipt.get(key) != receipt.get(key) for key in ("stage", "verdict", "artifact_refs")):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_CONTENT_MISMATCH")
    if _result_ref(stored_receipt, artifact_path=result_artifact_path) != result_ref:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_REF_MISMATCH")
    if not archive.has_settled_receipt(archive_trial_id, receipt_ref, str(receipt["receipt_hash"])):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_ARCHIVE_RECEIPT_MISMATCH")
    try:
        verify_receipt_evidence(receipt=stored_receipt, cas=cas)
    except ReceiptEvidenceError as exc:
        raise ProbeRetrievalError(f"PROBE_RETRIEVAL_EVIDENCE_{exc.reason}") from exc
    if asset_fingerprint != probe_asset_fingerprint(stored_receipt, cas=cas):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_ASSET_FINGERPRINT_MISMATCH")
    _paired_observations(stored_result, receipt=stored_receipt, cas=cas)

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
        "COALESCE(metric_outcome, -1e300) DESC, id ASC"
    )
    if archive_db is None or cas_root is None:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_DURABLE_STORES_REQUIRED")
    archive = ArchiveStore(Path(archive_db))
    cas = ContentAddressedStore(Path(cas_root))
    verified: list[ProbeExperience] = []
    verified_trials: dict[str, ProbeExperience | None] = {}
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        for record in connection.execute(query, parameters):
            row = ProbeExperience(**dict(record))
            cached = verified_trials.get(row.archive_trial_id)
            # A trial can occur under multiple failure signatures. Reuse its
            # replay only if this projection matches the verified projection.
            if row.archive_trial_id in verified_trials:
                current = None
                if cached is not None and _same_trial_projection(row, cached):
                    result = _json_object(cas.read_bytes(row.result_ref), "PROBE_RETRIEVAL_RESULT_INVALID")
                    if row.failure_signature in result["failure_signatures"]:
                        current = replace(row, paired_observations=cached.paired_observations)
            else:
                current = _load_verified_experience(row, archive=archive, cas=cas)
                if current is not None:
                    verified_trials[row.archive_trial_id] = current
            if current is not None:
                verified.append(current)
                if len(verified) == limit:
                    break
    finally:
        connection.close()
    return tuple(verified)


def _same_trial_projection(left: ProbeExperience, right: ProbeExperience) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in (
        "probe_id", "model_family", "runtime_capability", "asset_fingerprint", "primitive",
        "stage", "verdict", "metric_outcome", "archive_trial_id", "receipt_ref", "result_ref", "receipt_hash",
    ))


def recheck_probe_experiences(
    matches: Sequence[Mapping[str, Any]], *, archive_db: Path, cas_root: Path,
) -> tuple[ProbeExperience, ...]:
    """Recheck externally supplied projections before applying any prior.

    Reconstructs paired observations from authoritative stores; caller-supplied
    observation dictionaries are never numeric authority. A trial's paired
    proof is replayed once even when several signatures reference it.
    """
    archive, cas = ArchiveStore(Path(archive_db)), ContentAddressedStore(Path(cas_root))
    verified = []
    cached: dict[str, ProbeExperience] = {}
    fields = tuple(field for field in ProbeExperience.__dataclass_fields__ if field != "paired_observations")
    for match in matches:
        try:
            row = ProbeExperience(**{field: match[field] for field in fields})
            previous = cached.get(row.archive_trial_id)
            if previous is not None:
                result = _json_object(cas.read_bytes(row.result_ref), "PROBE_RETRIEVAL_RESULT_INVALID")
                current = (replace(row, paired_observations=previous.paired_observations)
                           if _same_trial_projection(row, previous) and row.failure_signature in result["failure_signatures"] else None)
            else:
                current = _load_verified_experience(row, archive=archive, cas=cas)
        except (KeyError, TypeError, ValueError, OSError, ContractValidationError):
            continue
        if current is not None:
            cached[row.archive_trial_id] = current
            verified.append(current)
    return tuple(verified)


def _load_verified_experience(
    row: ProbeExperience, *, archive: ArchiveStore, cas: ContentAddressedStore
) -> ProbeExperience | None:
    """Validate an index projection against its durable evidence."""
    if not archive.has_settled_receipt(row.archive_trial_id, row.receipt_ref, row.receipt_hash):
        return None
    try:
        receipt_bytes = cas.read_bytes(row.receipt_ref)
        result_bytes = cas.read_bytes(row.result_ref)
    except Exception:
        return None
    if hashlib.sha256(receipt_bytes).hexdigest() != row.receipt_hash:
        return None
    try:
        receipt = _json_object(receipt_bytes, "PROBE_RETRIEVAL_RECEIPT_INVALID")
        result = _json_object(result_bytes, "PROBE_RETRIEVAL_RESULT_INVALID")
        _validate_receipt_body(receipt)
        _validate_result(result)
        verify_receipt_evidence(receipt=receipt, cas=cas)
        fingerprint = probe_asset_fingerprint(receipt, cas=cas)
        observations = _paired_observations(result, receipt=receipt, cas=cas)
    except (ProbeRetrievalError, ContractValidationError, OSError, ValueError, TypeError):
        return None
    refs = receipt.get("artifact_refs")
    if (
        receipt.get("archive_trial_id") != row.archive_trial_id
        or receipt.get("stage") != row.stage
        or receipt["verdict"]["verdict"] != row.verdict
        or not isinstance(refs, Mapping)
        or row.result_ref not in refs.values()
        or any(result.get(key) != getattr(row, key) for key in ("probe_id", "model_family", "runtime_capability"))
        or row.failure_signature not in result["failure_signatures"]
        or _optional_string(result.get("primitive")) != row.primitive
        or _metric_outcome(result.get("metrics")) != row.metric_outcome
        or fingerprint != row.asset_fingerprint
    ):
        return None
    return replace(row, paired_observations=observations)


def _paired_observations(
    result: Mapping[str, Any], *, receipt: Mapping[str, Any], cas: ContentAddressedStore,
) -> tuple[dict[str, object], ...]:
    observations = result.get("paired_observations", [])
    refs = receipt.get("artifact_refs", {})

    def read(path: str) -> bytes:
        ref = refs.get(path)
        if not isinstance(ref, str):
            raise ProbeRetrievalError("PROBE_RETRIEVAL_PAIRED_ARTIFACT_MISSING")
        return cas.read_bytes(ref)

    try:
        for observation in observations:
            if observation["query"]["primitive"] != result.get("primitive"):
                raise ProbeRetrievalError("PROBE_RETRIEVAL_PAIRED_PRIMITIVE_MISMATCH")
        verify_paired_observations(observations, read_artifact=read)
    except (PairedObservationError, ContractValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_PAIRED_EVIDENCE_INVALID") from exc
    return tuple(deepcopy(observations))


def probe_asset_fingerprint(receipt: Mapping[str, Any], *, cas: ContentAddressedStore) -> str | None:
    """Derive asset identity from the conformance evidence in a settled receipt.

    Older receipts lack this evidence and therefore have no usable asset
    identity; their diagnostic observations remain eligible for general reuse.
    """
    refs = receipt.get("support_refs")
    if refs is None:
        return None
    if not isinstance(refs, Mapping):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_CONFORMANCE_INVALID")
    ref = refs.get("onboarding_conformance")
    if ref is None:
        return None
    if not isinstance(ref, str):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_CONFORMANCE_INVALID")
    try:
        conformance = _json_object(cas.read_bytes(ref), "PROBE_RETRIEVAL_CONFORMANCE_INVALID")
        validate_document("model_conformance_receipt", conformance)
    except Exception as exc:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_CONFORMANCE_INVALID") from exc
    if conformance.get("verdict") != "PASS" or conformance.get("optimization_launch_allowed") is not True:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_CONFORMANCE_INVALID")
    bindings = conformance["asset_bindings"]
    if any(row["state"] != "bound" or row["observed_fingerprint"] != row["onboarding_fingerprint"] for row in bindings):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_CONFORMANCE_INVALID")
    fingerprints = sorted(row["observed_fingerprint"] for row in bindings)
    return hashlib.sha256(canonical_json_bytes(fingerprints)).hexdigest() if fingerprints else None


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
    try:
        validate_document("diagnostic_probe_result", result)
    except ContractValidationError as exc:
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_CONTRACT_INVALID") from exc
    if result["state"] != "ready":
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RESULT_NOT_READY")


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    _validate_receipt_body(receipt)
    for field in ("archive_trial_id", "receipt_ref", "receipt_hash", "stage"):
        if not isinstance(receipt.get(field), str) or not str(receipt[field]):
            raise ProbeRetrievalError(f"PROBE_RETRIEVAL_RECEIPT_FIELD_INVALID:{field}")


def _validate_receipt_body(receipt: Mapping[str, Any]) -> None:
    verdict = receipt.get("verdict")
    if (
        receipt.get("settlement_state") != "settled"
        or not isinstance(verdict, Mapping)
        or verdict.get("verdict") not in {"PASS", "VOID"}
    ):
        raise ProbeRetrievalError("PROBE_RETRIEVAL_RECEIPT_NOT_SETTLED")


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
