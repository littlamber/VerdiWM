"""Durable SQLite state for the Ctrl-World autonomous transfer loop."""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class AutonomousTransferStateError(RuntimeError):
    """The controller state could not preserve its durable invariants."""


TERMINAL_STATES = {"terminal", "imported_terminal"}


class DurableLoopStore:
    """Small transactional ledger for discovery and experiment progression."""

    def __init__(self, path: Path, *, loop_id: str, config_digest: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize(loop_id=loop_id, config_digest=config_digest)

    def recover_interrupted(self) -> int:
        """Return running work to its pending state after controller loss."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT work_id, state FROM work_items WHERE state LIKE 'running_%'"
            ).fetchall()
            for row in rows:
                pending = "pending_" + str(row["state"])[len("running_") :]
                connection.execute(
                    "UPDATE work_items SET state = ?, updated_at = ? WHERE work_id = ?",
                    (pending, _utc_now(), row["work_id"]),
                )
            connection.execute(
                "UPDATE stage_receipts SET state = 'interrupted', finished_at = ? "
                "WHERE state = 'running'",
                (_utc_now(),),
            )
            connection.execute(
                "UPDATE discovery_cycles SET state = 'interrupted', finished_at = ?, "
                "error = 'CONTROLLER_RESTART' WHERE state = 'running'",
                (_utc_now(),),
            )
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    def begin_discovery(self, output_root: Path) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "INSERT INTO discovery_cycles(state, started_at, output_root) "
                "VALUES ('running', ?, ?) RETURNING cycle_id",
                (_utc_now(), str(output_root)),
            ).fetchone()
            connection.commit()
            assert row is not None
            return int(row["cycle_id"])
        finally:
            connection.close()

    def finish_discovery(
        self,
        cycle_id: int,
        *,
        state: str,
        manifest: Mapping[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        if state not in {"completed", "failed", "interrupted"}:
            raise AutonomousTransferStateError("AUTONOMOUS_DISCOVERY_STATE_INVALID")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE discovery_cycles SET state = ?, finished_at = ?, manifest_json = ?, "
                "error = ? WHERE cycle_id = ? AND state = 'running'",
                (
                    state,
                    _utc_now(),
                    _canonical_json(manifest) if manifest is not None else None,
                    error,
                    cycle_id,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise AutonomousTransferStateError("AUTONOMOUS_DISCOVERY_FENCE_STALE")
            connection.commit()
        finally:
            connection.close()

    def discovery_due(self, *, interval_seconds: float, now: dt.datetime | None = None) -> bool:
        instant = now or dt.datetime.now(dt.timezone.utc)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COALESCE(finished_at, started_at) AS timestamp "
                "FROM discovery_cycles ORDER BY cycle_id DESC LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        if row is None or row["timestamp"] is None:
            return True
        previous = dt.datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        return (instant - previous).total_seconds() >= interval_seconds

    def ingest_intake(
        self,
        cycle_id: int,
        manifest: Mapping[str, object],
        *,
        initial_state: str = "pending_materialization",
    ) -> dict[str, int]:
        if initial_state not in {"pending_portrait", "pending_materialization"}:
            raise AutonomousTransferStateError("AUTONOMOUS_INITIAL_WORK_STATE_INVALID")
        ideas = _documents_by_source(manifest.get("idea_paths"), source_nested=True)
        work_orders = _documents_by_idea(manifest.get("work_order_paths"))
        assessments = _documents(manifest.get("assessment_paths"))
        inserted_sources = 0
        inserted_work = 0
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for path, assessment in assessments:
                source_digest = str(assessment["source_digest"])
                assessment_digest = str(assessment["assessment_digest"])
                inserted_sources += int(
                    connection.execute(
                        "INSERT OR IGNORE INTO sources(source_digest, assessment_digest, source_id, "
                        "profile_id, execution_state, assessment_path, first_cycle_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            source_digest,
                            assessment_digest,
                            assessment["source_id"],
                            assessment["profile_id"],
                            assessment["execution_state"],
                            str(path),
                            cycle_id,
                            _utc_now(),
                        ),
                    ).rowcount
                )
                if assessment.get("execution_state") != "materialization_required":
                    continue
                idea_entry = ideas.get((source_digest, assessment_digest))
                if idea_entry is None:
                    connection.rollback()
                    raise AutonomousTransferStateError("AUTONOMOUS_IDEA_BINDING_MISSING")
                idea_path, idea = idea_entry
                work_order_path = work_orders.get(str(idea["idea_id"]))
                if work_order_path is None:
                    connection.rollback()
                    raise AutonomousTransferStateError("AUTONOMOUS_WORK_ORDER_BINDING_MISSING")
                work_id = _work_id(source_digest, assessment_digest)
                inserted_work += int(
                    connection.execute(
                        "INSERT OR IGNORE INTO work_items(work_id, source_digest, assessment_digest, "
                        "source_id, profile_id, idea_id, idea_path, work_order_path, assessment_path, "
                        "state, failure_count, context_json, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '{}', ?, ?)",
                        (
                            work_id,
                            source_digest,
                            assessment_digest,
                            assessment["source_id"],
                            assessment["profile_id"],
                            idea["idea_id"],
                            str(idea_path),
                            str(work_order_path),
                            str(path),
                            initial_state,
                            _utc_now(),
                            _utc_now(),
                        ),
                    ).rowcount
                )
            connection.commit()
        finally:
            connection.close()
        return {"inserted_sources": inserted_sources, "inserted_work_items": inserted_work}

    def import_verified_evidence(
        self,
        *,
        source_path: Path,
        local_path: Path,
        record: Mapping[str, object],
    ) -> bool:
        source_digest = str(record["source_digest"])
        assessment_digest = str(record["assessment_digest"])
        work_id = _work_id(source_digest, assessment_digest)
        context = {
            "imported_evidence_path": str(local_path),
            "imported_from": str(source_path),
            "decision": record.get("outcome"),
            "verdict_ref": record.get("verdict_ref"),
        }
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO sources(source_digest, assessment_digest, source_id, "
                "profile_id, execution_state, assessment_path, first_cycle_id, created_at) "
                "VALUES (?, ?, ?, 'imported_verified_evidence', 'verified', ?, NULL, ?)",
                (
                    source_digest,
                    assessment_digest,
                    record["source_id"],
                    str(source_path),
                    _utc_now(),
                ),
            )
            inserted = connection.execute(
                "INSERT OR IGNORE INTO work_items(work_id, source_digest, assessment_digest, "
                "source_id, profile_id, idea_id, idea_path, work_order_path, assessment_path, "
                "state, failure_count, context_json, terminal_outcome, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'imported_verified_evidence', ?, ?, ?, ?, "
                "'imported_terminal', 0, ?, ?, ?, ?)",
                (
                    work_id,
                    source_digest,
                    assessment_digest,
                    record["source_id"],
                    str(record.get("candidate_id") or "imported"),
                    str(source_path),
                    str(source_path),
                    str(source_path),
                    _canonical_json(context),
                    str(record.get("outcome") or "imported_verified"),
                    _utc_now(),
                    _utc_now(),
                ),
            ).rowcount
            connection.commit()
            return bool(inserted)
        finally:
            connection.close()

    def list_work(self, states: Sequence[str], *, limit: int = 100) -> list[dict[str, Any]]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM work_items WHERE state IN ({placeholders}) "
                "ORDER BY created_at, work_id LIMIT ?",
                [*states, limit],
            ).fetchall()
        finally:
            connection.close()
        return [_work_row(row) for row in rows]

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic, JSON-shaped view for closed-loop audits."""

        connection = self._connect()
        try:
            work_rows = connection.execute(
                "SELECT * FROM work_items ORDER BY created_at, work_id"
            ).fetchall()
            stage_rows = connection.execute(
                "SELECT * FROM stage_receipts ORDER BY work_id, stage, attempt"
            ).fetchall()
        finally:
            connection.close()
        works = [_work_row(row) for row in work_rows]
        stages: list[dict[str, object]] = []
        for row in stage_rows:
            value = dict(row)
            raw_payload = value.pop("payload_json")
            value["payload"] = (
                json.loads(str(raw_payload)) if raw_payload is not None else {}
            )
            stages.append(value)
        return {"work_items": works, "stage_receipts": stages}

    def queue_imported_replans(self) -> int:
        """Move newly imported terminal evidence through the archive/audit stage once."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE work_items SET state = 'pending_replan', updated_at = ? "
                "WHERE state = 'imported_terminal'",
                (_utc_now(),),
            ).rowcount
            connection.commit()
            return int(changed)
        finally:
            connection.close()

    def claim(self, work_id: str, *, pending_state: str) -> dict[str, Any] | None:
        running = pending_state.replace("pending_", "running_", 1)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE work_items SET state = ?, updated_at = ? "
                "WHERE work_id = ? AND state = ?",
                (running, _utc_now(), work_id, pending_state),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id = ?", (work_id,)
            ).fetchone()
            connection.commit()
            assert row is not None
            return _work_row(row)
        finally:
            connection.close()

    def begin_stage(self, work_id: str, stage: str, root_parent: Path) -> tuple[int, Path]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            attempt = int(
                connection.execute(
                    "SELECT COALESCE(MAX(attempt), 0) + 1 FROM stage_receipts "
                    "WHERE work_id = ? AND stage = ?",
                    (work_id, stage),
                ).fetchone()[0]
            )
            root = Path(root_parent).expanduser().resolve() / f"attempt-{attempt:03d}"
            connection.execute(
                "INSERT INTO stage_receipts(work_id, stage, attempt, state, root, started_at) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (work_id, stage, attempt, str(root), _utc_now()),
            )
            connection.commit()
            return attempt, root
        finally:
            connection.close()

    def finish_stage(
        self,
        work_id: str,
        stage: str,
        attempt: int,
        *,
        state: str,
        payload: Mapping[str, object],
        receipt_path: Path | None,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE stage_receipts SET state = ?, payload_json = ?, receipt_path = ?, "
                "finished_at = ? WHERE work_id = ? AND stage = ? AND attempt = ? "
                "AND state = 'running'",
                (
                    state,
                    _canonical_json(payload),
                    str(receipt_path) if receipt_path is not None else None,
                    _utc_now(),
                    work_id,
                    stage,
                    attempt,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise AutonomousTransferStateError("AUTONOMOUS_STAGE_FENCE_STALE")
            connection.commit()
        finally:
            connection.close()

    def transition(
        self,
        work_id: str,
        *,
        expected_state: str,
        next_state: str,
        context_update: Mapping[str, object] | None = None,
        terminal_outcome: str | None = None,
        error: str | None = None,
        increment_failure: bool = False,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT context_json, failure_count FROM work_items WHERE work_id = ? AND state = ?",
                (work_id, expected_state),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise AutonomousTransferStateError("AUTONOMOUS_WORK_FENCE_STALE")
            context = json.loads(str(row["context_json"]))
            if context_update:
                context.update(context_update)
            failures = int(row["failure_count"]) + int(increment_failure)
            connection.execute(
                "UPDATE work_items SET state = ?, failure_count = ?, context_json = ?, "
                "terminal_outcome = ?, last_error = ?, updated_at = ? WHERE work_id = ?",
                (
                    next_state,
                    failures,
                    _canonical_json(context),
                    terminal_outcome,
                    error,
                    _utc_now(),
                    work_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def status(self) -> dict[str, object]:
        connection = self._connect()
        try:
            state_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM work_items GROUP BY state ORDER BY state"
            ).fetchall()
            outcome_rows = connection.execute(
                "SELECT terminal_outcome, COUNT(*) AS count FROM work_items "
                "WHERE terminal_outcome IS NOT NULL GROUP BY terminal_outcome ORDER BY terminal_outcome"
            ).fetchall()
            cycles = connection.execute("SELECT COUNT(*) FROM discovery_cycles").fetchone()[0]
            sources = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            stages = connection.execute("SELECT COUNT(*) FROM stage_receipts").fetchone()[0]
        finally:
            connection.close()
        return {
            "discovery_cycles": int(cycles),
            "source_assessments": int(sources),
            "stage_receipts": int(stages),
            "work_state_counts": {str(row["state"]): int(row["count"]) for row in state_rows},
            "terminal_outcome_counts": {
                str(row["terminal_outcome"]): int(row["count"]) for row in outcome_rows
            },
        }

    def _initialize(self, *, loop_id: str, config_digest: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY NOT NULL,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discovery_cycles (
                    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    output_root TEXT NOT NULL,
                    manifest_json TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_digest TEXT NOT NULL,
                    assessment_digest TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    execution_state TEXT NOT NULL,
                    assessment_path TEXT NOT NULL,
                    first_cycle_id INTEGER,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_digest, assessment_digest)
                );
                CREATE TABLE IF NOT EXISTS work_items (
                    work_id TEXT PRIMARY KEY NOT NULL,
                    source_digest TEXT NOT NULL,
                    assessment_digest TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    idea_id TEXT NOT NULL,
                    idea_path TEXT NOT NULL,
                    work_order_path TEXT NOT NULL,
                    assessment_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL,
                    context_json TEXT NOT NULL,
                    terminal_outcome TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_digest)
                );
                CREATE TABLE IF NOT EXISTS stage_receipts (
                    work_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    root TEXT NOT NULL,
                    payload_json TEXT,
                    receipt_path TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    PRIMARY KEY(work_id, stage, attempt)
                );
                CREATE INDEX IF NOT EXISTS work_state_index ON work_items(state, created_at);
                CREATE INDEX IF NOT EXISTS stage_work_index ON stage_receipts(work_id, stage);
                """
            )
            existing = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            expected = {"loop_id": loop_id, "config_digest": config_digest}
            if existing and any(existing.get(key) != value for key, value in expected.items()):
                connection.rollback()
                raise AutonomousTransferStateError("AUTONOMOUS_STATE_CONFIG_MISMATCH")
            for key, value in expected.items():
                connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value)
                )
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


def _documents(raw_paths: object) -> list[tuple[Path, dict[str, object]]]:
    if not isinstance(raw_paths, list):
        raise AutonomousTransferStateError("AUTONOMOUS_INTAKE_PATHS_INVALID")
    rows = []
    for value in raw_paths:
        path = Path(str(value)).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AutonomousTransferStateError("AUTONOMOUS_INTAKE_DOCUMENT_INVALID")
        rows.append((path, payload))
    return rows


def _documents_by_source(
    raw_paths: object, *, source_nested: bool
) -> dict[tuple[str, str], tuple[Path, dict[str, object]]]:
    result = {}
    for path, payload in _documents(raw_paths):
        source = payload.get("source") if source_nested else payload
        if not isinstance(source, Mapping):
            raise AutonomousTransferStateError("AUTONOMOUS_SOURCE_BINDING_INVALID")
        key = (str(source["source_digest"]), str(source["assessment_digest"]))
        result[key] = (path, payload)
    return result


def _documents_by_idea(raw_paths: object) -> dict[str, Path]:
    return {str(payload["idea_id"]): path for path, payload in _documents(raw_paths)}


def _work_id(source_digest: str, assessment_digest: str) -> str:
    import hashlib

    digest = hashlib.sha256(f"{source_digest}:{assessment_digest}".encode()).hexdigest()
    return f"transfer-{digest[:20]}"


def _work_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["context"] = json.loads(str(payload.pop("context_json")))
    return payload


def _canonical_json(payload: Mapping[str, object] | None) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
