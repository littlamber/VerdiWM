"""SQLite baseline archive and filesystem content-addressed store.

This is deliberately small and conservative: the database stores immutable
metadata, while every byte payload is addressed by its SHA-256 content.  A
baseline record is generation zero and append-only; it cannot be edited into a
new reference frame after candidate experiments start.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.propose.scheduler import CellStats, InterventionCell


class ArchiveInvariantError(ValueError):
    """A CAS or archive invariant failed closed."""


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size: int
    media_type: str
    uri: str
    path: Path


@dataclass(frozen=True)
class BaselineRecord:
    environment: str
    model_ref: str
    evaluator_freeze_sha256: str
    heldout_split_sha256: str
    receipt_ref: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class SettledTrialRecord:
    """The archive-level record admitted only after terminal receipt settlement."""

    trial_id: str
    proposal_id: str
    goal_id: str
    library_version: str
    failure_context_ref: str
    verdict_ref: str
    receipt_ref: str
    gpu_hours: float
    hypothesis_hash: str
    impl_diff_hash: str
    evaluator_hash: str
    settlement_state: str
    receipt_hash: str
    cell: InterventionCell | None = None
    verified_gain: float | None = None
    exploratory: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "proposal_id": self.proposal_id,
            "goal_id": self.goal_id,
            "library_version": self.library_version,
            "failure_context_ref": self.failure_context_ref,
            "verdict_ref": self.verdict_ref,
            "receipt_ref": self.receipt_ref,
            "cost": {"gpu_hours": self.gpu_hours},
            "fingerprint": {
                "hypothesis_hash": self.hypothesis_hash,
                "impl_diff_hash": self.impl_diff_hash,
                "evaluator_hash": self.evaluator_hash,
            },
            "settlement": {
                "state": self.settlement_state,
                "receipt_hash": self.receipt_hash,
                "evidence_scope": "exploratory" if self.exploratory else "verified",
            },
        }


@dataclass(frozen=True)
class CellProjectionRecord:
    cell: InterventionCell
    stats: CellStats

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": {
                "environment": self.cell.environment,
                "layer": self.cell.layer,
                "primitive_family": self.cell.primitive_family,
                "parameter_bucket": self.cell.parameter_bucket,
            },
            "stats": {
                "visits": self.stats.visits,
                "mean_verified_improvement": self.stats.mean_verified_improvement,
            },
        }


class ContentAddressedStore:
    """Regular-file-only SHA-256 content storage rooted at ``<root>/cas``."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._cas = self._root / "cas"
        self._cas.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_directory(self._cas, "CAS_ROOT_INVALID")

    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactRef:
        if not isinstance(payload, bytes) or not media_type:
            raise ArchiveInvariantError("CAS_PAYLOAD_OR_MEDIA_TYPE_INVALID")
        digest = hashlib.sha256(payload).hexdigest()
        target = self._path_for(digest)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_directory(target.parent, "CAS_PREFIX_DIRECTORY_INVALID")
        if target.exists() or target.is_symlink():
            self._verify_existing(target, digest)
            return ArtifactRef(digest, len(payload), media_type, _uri(digest), target)
        temporary = target.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._verify_existing(target, digest)
            else:
                self._verify_existing(target, digest)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        return ArtifactRef(digest, len(payload), media_type, _uri(digest), target)

    def read_bytes(self, uri: str) -> bytes:
        digest = _digest_from_uri(uri)
        target = self._path_for(digest)
        _require_regular_file(target, "CAS_MEMBER_INVALID")
        payload = _read_regular_stable(target, "CAS_MEMBER_CHANGED")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ArchiveInvariantError("CAS_DIGEST_MISMATCH")
        return payload

    def _path_for(self, digest: str) -> Path:
        _require_digest(digest, "CAS_DIGEST_INVALID")
        return self._cas / digest[:2] / digest

    def _verify_existing(self, target: Path, digest: str) -> None:
        _require_regular_file(target, "CAS_MEMBER_INVALID")
        payload = _read_regular_stable(target, "CAS_MEMBER_CHANGED")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ArchiveInvariantError("CAS_DIGEST_MISMATCH")


class ArchiveStore:
    """Authoritative M0 baseline records with SQLite DELETE/FULL durability."""

    def __init__(self, database_path: Path) -> None:
        self._path = Path(database_path).resolve()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def record_baseline(self, record: BaselineRecord) -> None:
        _validate_baseline(record)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO baselines (
                        environment, generation, model_ref, evaluator_freeze_sha256,
                        heldout_split_sha256, receipt_ref, metrics_json
                    ) VALUES (?, 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.environment,
                        record.model_ref,
                        record.evaluator_freeze_sha256,
                        record.heldout_split_sha256,
                        record.receipt_ref,
                        _canonical_metrics(record.metrics),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ArchiveInvariantError("BASELINE_ALREADY_RECORDED") from exc
            _record_artifact_reference(connection, record.model_ref)
            _record_artifact_reference(connection, record.receipt_ref)
            connection.commit()
        finally:
            connection.close()

    def list_baselines(self) -> list[BaselineRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT environment, model_ref, evaluator_freeze_sha256,
                       heldout_split_sha256, receipt_ref, metrics_json
                FROM baselines ORDER BY environment
                """
            ).fetchall()
        finally:
            connection.close()
        return [
            BaselineRecord(
                environment=row["environment"],
                model_ref=row["model_ref"],
                evaluator_freeze_sha256=row["evaluator_freeze_sha256"],
                heldout_split_sha256=row["heldout_split_sha256"],
                receipt_ref=row["receipt_ref"],
                metrics=json.loads(row["metrics_json"]),
            )
            for row in rows
        ]

    def record_settled_trial(self, record: SettledTrialRecord) -> None:
        """Atomically publish a trial to all archive projections after settlement.

        There is deliberately no method for recording an admitted/running
        record.  Schedulers query only this projection, so a zombie worker or
        an unreceipted run cannot influence UCB, priors or a formal verdict.
        """

        _validate_settled_trial(record)
        payload = record.to_dict()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO trials(
                        trial_id, proposal_id, goal_id, library_version, failure_context_ref,
                        verdict_ref, receipt_ref, cost_json, fingerprint_json, settlement_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.trial_id,
                        record.proposal_id,
                        record.goal_id,
                        record.library_version,
                        record.failure_context_ref,
                        record.verdict_ref,
                        record.receipt_ref,
                        _canonical_metrics({"gpu_hours": record.gpu_hours}),
                        json.dumps(payload["fingerprint"], sort_keys=True, separators=(",", ":")),
                        json.dumps(payload["settlement"], sort_keys=True, separators=(",", ":")),
                    ),
                )
                connection.execute(
                    "INSERT INTO proposals(proposal_id, trial_id, goal_id, library_version) VALUES (?, ?, ?, ?)",
                    (record.proposal_id, record.trial_id, record.goal_id, record.library_version),
                )
                connection.execute(
                    "INSERT INTO verdicts(trial_id, verdict_ref) VALUES (?, ?)", (record.trial_id, record.verdict_ref)
                )
                connection.execute(
                    "INSERT INTO receipts(trial_id, receipt_ref, receipt_hash) VALUES (?, ?, ?)",
                    (record.trial_id, record.receipt_ref, record.receipt_hash),
                )
                for uri in (record.failure_context_ref, record.verdict_ref, record.receipt_ref):
                    _record_artifact_reference(connection, uri)
                if record.cell is not None and not record.exploratory:
                    _record_cell_observation(connection, record.cell, _require_verified_gain(record.verified_gain))
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ArchiveInvariantError("SETTLED_TRIAL_ALREADY_RECORDED") from exc
            connection.commit()
        finally:
            connection.close()

    def list_cells(self) -> list[CellProjectionRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT environment, layer, primitive_family, parameter_bucket,
                       settled_trial_count, mean_verified_gain
                FROM cells ORDER BY environment, layer, primitive_family, parameter_bucket
                """
            ).fetchall()
        finally:
            connection.close()
        return [
            CellProjectionRecord(
                cell=InterventionCell(
                    environment=str(row["environment"]),
                    layer=str(row["layer"]),
                    primitive_family=str(row["primitive_family"]),
                    parameter_bucket=str(row["parameter_bucket"]),
                ),
                stats=CellStats(
                    visits=int(row["settled_trial_count"]),
                    mean_verified_improvement=float(row["mean_verified_gain"]),
                ),
            )
            for row in rows
        ]

    def cell_statistics(self) -> dict[InterventionCell, CellStats]:
        return {record.cell: record.stats for record in self.list_cells()}

    def record_artifact_reference(self, uri: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _record_artifact_reference(connection, uri)
            connection.commit()
        finally:
            connection.close()

    def reconcile_artifacts(self) -> dict[str, int]:
        """Index CAS references from existing immutable records.

        Older records may predate artifact indexing.  Reconciliation is
        idempotent and only derives entries from already-published baseline and
        settled-trial rows, so it cannot make an unreceipted run scheduler
        visible.
        """

        connection = self._connect()
        try:
            before = int(connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT model_ref, receipt_ref FROM baselines").fetchall()
            for row in rows:
                _record_artifact_reference(connection, str(row["model_ref"]))
                _record_artifact_reference(connection, str(row["receipt_ref"]))
            rows = connection.execute("SELECT failure_context_ref, verdict_ref, receipt_ref FROM trials").fetchall()
            for row in rows:
                for key in ("failure_context_ref", "verdict_ref", "receipt_ref"):
                    _record_artifact_reference(connection, str(row[key]))
            connection.commit()
            after = int(connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
            return {"before": before, "after": after, "inserted": after - before}
        finally:
            connection.close()

    def visible_settled_trials(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT trial_id FROM trials ORDER BY trial_id").fetchall()
        finally:
            connection.close()
        return tuple(str(row["trial_id"]) for row in rows)

    def archive_statistics(self) -> dict[str, int]:
        connection = self._connect()
        try:
            return {
                "settled_trials": int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]),
                "baselines": int(connection.execute("SELECT COUNT(*) FROM baselines").fetchone()[0]),
                "artifacts": int(connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]),
            }
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS baselines (
                    environment TEXT PRIMARY KEY NOT NULL,
                    generation INTEGER NOT NULL CHECK (generation = 0),
                    model_ref TEXT NOT NULL,
                    evaluator_freeze_sha256 TEXT NOT NULL,
                    heldout_split_sha256 TEXT NOT NULL,
                    receipt_ref TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY NOT NULL,
                    proposal_id TEXT UNIQUE NOT NULL,
                    goal_id TEXT NOT NULL,
                    library_version TEXT NOT NULL,
                    failure_context_ref TEXT NOT NULL,
                    verdict_ref TEXT NOT NULL,
                    receipt_ref TEXT NOT NULL,
                    cost_json TEXT NOT NULL,
                    fingerprint_json TEXT NOT NULL,
                    settlement_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id TEXT PRIMARY KEY NOT NULL,
                    trial_id TEXT UNIQUE NOT NULL REFERENCES trials(trial_id),
                    goal_id TEXT NOT NULL,
                    library_version TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verdicts (
                    trial_id TEXT PRIMARY KEY NOT NULL REFERENCES trials(trial_id),
                    verdict_ref TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipts (
                    trial_id TEXT PRIMARY KEY NOT NULL REFERENCES trials(trial_id),
                    receipt_ref TEXT NOT NULL,
                    receipt_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    uri TEXT PRIMARY KEY NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cells (
                    cell_key TEXT PRIMARY KEY NOT NULL,
                    environment TEXT NOT NULL DEFAULT '',
                    layer TEXT NOT NULL DEFAULT '',
                    primitive_family TEXT NOT NULL DEFAULT '',
                    parameter_bucket TEXT NOT NULL DEFAULT '',
                    settled_trial_count INTEGER NOT NULL DEFAULT 0,
                    mean_verified_gain REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            _ensure_column(connection, table="cells", column="environment", declaration="TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, table="cells", column="layer", declaration="TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, table="cells", column="primitive_family", declaration="TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, table="cells", column="parameter_bucket", declaration="TEXT NOT NULL DEFAULT ''")
            connection.commit()
        finally:
            connection.close()
        os.chmod(self._path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def _validate_baseline(record: BaselineRecord) -> None:
    if not record.environment:
        raise ArchiveInvariantError("BASELINE_ENVIRONMENT_INVALID")
    if not _is_cas_uri(record.model_ref):
        raise ArchiveInvariantError("BASELINE_MODEL_REF_INVALID")
    if not _is_cas_uri(record.receipt_ref):
        raise ArchiveInvariantError("BASELINE_RECEIPT_REF_INVALID")
    _require_digest(record.evaluator_freeze_sha256, "BASELINE_EVALUATOR_FREEZE_INVALID")
    _require_digest(record.heldout_split_sha256, "BASELINE_SPLIT_INVALID")
    if not record.metrics:
        raise ArchiveInvariantError("BASELINE_METRICS_EMPTY")
    for name, value in record.metrics.items():
        if not isinstance(name, str) or not name or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ArchiveInvariantError("BASELINE_METRICS_INVALID")


def _validate_settled_trial(record: SettledTrialRecord) -> None:
    try:
        validate_document("trial_record", record.to_dict())
    except ContractValidationError as exc:
        raise ArchiveInvariantError(f"TRIAL_RECORD_CONTRACT_INVALID:{exc}") from exc
    if record.settlement_state != "settled":
        raise ArchiveInvariantError("TRIAL_RECORD_UNSETTLED")
    if _digest_from_uri(record.receipt_ref) != record.receipt_hash:
        raise ArchiveInvariantError("TRIAL_RECEIPT_HASH_MISMATCH")
    if not math.isfinite(record.gpu_hours) or record.gpu_hours < 0:
        raise ArchiveInvariantError("TRIAL_COST_INVALID")
    for digest in (record.hypothesis_hash, record.impl_diff_hash, record.evaluator_hash):
        _require_digest(digest, "TRIAL_FINGERPRINT_INVALID")
    if record.cell is None:
        if record.verified_gain is not None:
            raise ArchiveInvariantError("TRIAL_CELL_GAIN_WITHOUT_CELL")
        return
    _validate_cell(record.cell)
    if not record.exploratory:
        _require_verified_gain(record.verified_gain)


def _record_cell_observation(connection: sqlite3.Connection, cell: InterventionCell, verified_gain: float) -> None:
    key = _cell_key(cell)
    row = connection.execute(
        "SELECT settled_trial_count, mean_verified_gain FROM cells WHERE cell_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        connection.execute(
            """
            INSERT INTO cells(
                cell_key, environment, layer, primitive_family, parameter_bucket,
                settled_trial_count, mean_verified_gain
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (
                key,
                cell.environment,
                cell.layer,
                cell.primitive_family,
                cell.parameter_bucket,
                verified_gain,
            ),
        )
        return
    count = int(row["settled_trial_count"])
    mean = float(row["mean_verified_gain"])
    updated_count = count + 1
    updated_mean = mean + (verified_gain - mean) / updated_count
    connection.execute(
        """
        UPDATE cells
        SET settled_trial_count = ?, mean_verified_gain = ?,
            environment = ?, layer = ?, primitive_family = ?, parameter_bucket = ?
        WHERE cell_key = ?
        """,
        (
            updated_count,
            updated_mean,
            cell.environment,
            cell.layer,
            cell.primitive_family,
            cell.parameter_bucket,
            key,
        ),
    )


def _validate_cell(cell: InterventionCell) -> None:
    fields = (cell.environment, cell.layer, cell.primitive_family, cell.parameter_bucket)
    if any(not isinstance(value, str) or not value for value in fields):
        raise ArchiveInvariantError("TRIAL_CELL_INVALID")
    if any("\0" in value for value in fields):
        raise ArchiveInvariantError("TRIAL_CELL_INVALID")


def _require_verified_gain(value: float | None) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ArchiveInvariantError("TRIAL_VERIFIED_GAIN_INVALID")
    return float(value)


def _cell_key(cell: InterventionCell) -> str:
    return json.dumps(
        {
            "environment": cell.environment,
            "layer": cell.layer,
            "primitive_family": cell.primitive_family,
            "parameter_bucket": cell.parameter_bucket,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _ensure_column(connection: sqlite3.Connection, *, table: str, column: str, declaration: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if column in {str(row["name"]) for row in rows}:
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _canonical_metrics(metrics: Mapping[str, float]) -> str:
    return json.dumps(dict(sorted(metrics.items())), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _uri(digest: str) -> str:
    return f"cas://sha256/{digest}"


def _is_cas_uri(value: str) -> bool:
    try:
        _digest_from_uri(value)
    except ArchiveInvariantError:
        return False
    return True


def _record_artifact_reference(connection: sqlite3.Connection, uri: str) -> None:
    connection.execute("INSERT OR IGNORE INTO artifacts(uri, sha256) VALUES (?, ?)", (uri, _digest_from_uri(uri)))


def _digest_from_uri(uri: str) -> str:
    prefix = "cas://sha256/"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ArchiveInvariantError("CAS_URI_INVALID")
    digest = uri[len(prefix) :]
    _require_digest(digest, "CAS_URI_INVALID")
    return digest


def _require_digest(value: str, code: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ArchiveInvariantError(code)


def _require_directory(path: Path, code: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ArchiveInvariantError(code) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveInvariantError(code)


def _require_regular_file(path: Path, code: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ArchiveInvariantError(code) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArchiveInvariantError(code)


def _read_regular_stable(path: Path, changed_code: str) -> bytes:
    before = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveInvariantError(changed_code) from exc
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ArchiveInvariantError(changed_code)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise ArchiveInvariantError(changed_code)
    return b"".join(chunks)
