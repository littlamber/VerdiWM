"""Durable SQLite state for runs, sources, experiments, and transferable evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, objective TEXT NOT NULL,
  state TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
  status TEXT NOT NULL, local_path TEXT, content_digest TEXT, text TEXT,
  metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ideas (
  idea_id TEXT PRIMARY KEY, title TEXT NOT NULL, mechanism TEXT NOT NULL,
  status TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS probes (
  probe_id TEXT PRIMARY KEY, status TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, run_id TEXT, hypothesis_id TEXT,
  state TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY, experiment_id TEXT, outcome TEXT NOT NULL,
  model_id TEXT NOT NULL, delta REAL NOT NULL, protected_ok INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_edges (
  subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
  evidence_id TEXT, PRIMARY KEY(subject, predicate, object, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_evidence_outcome ON evidence(outcome);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
"""


class SQLiteState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _put(self, table: str, key: str, values: dict[str, Any]) -> None:
        if key not in values:
            raise ValueError(f"missing primary key {key!r} for {table}")
        # Callers pass complete rows. Keep the key once and derive stable
        # parameter ordering from the mapping itself.
        columns = list(values)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column != key)
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT({key}) DO UPDATE SET {updates}"
        self._conn.execute(sql, [values[column] for column in columns])
        self._conn.commit()

    def put_document(self, document_id: str, payload: dict[str, Any]) -> None:
        self._put("documents", "document_id", {"document_id": document_id, "url": payload.get("url"), "title": payload.get("title", ""), "source": payload.get("source", ""), "status": payload.get("status", "discovered"), "local_path": payload.get("local_path"), "content_digest": payload.get("content_digest"), "text": payload.get("text"), "metadata_json": json.dumps(payload, sort_keys=True)})

    def put_idea(self, idea_id: str, payload: dict[str, Any], status: str = "proposed") -> None:
        self._put("ideas", "idea_id", {"idea_id": idea_id, "title": str(payload.get("title", "")), "mechanism": str(payload.get("mechanism", "")), "status": status, "payload_json": json.dumps(payload, sort_keys=True)})

    def put_probe(self, probe_id: str, payload: dict[str, Any], status: str = "proposed") -> None:
        self._put("probes", "probe_id", {"probe_id": probe_id, "status": status, "payload_json": json.dumps(payload, sort_keys=True)})

    def put_experiment(self, experiment_id: str, payload: dict[str, Any], state: str = "queued") -> None:
        self._put("experiments", "experiment_id", {"experiment_id": experiment_id, "run_id": payload.get("run_id"), "hypothesis_id": payload.get("hypothesis_id"), "state": state, "payload_json": json.dumps(payload, sort_keys=True)})

    def put_evidence(self, evidence_id: str, payload: dict[str, Any]) -> None:
        self._put("evidence", "evidence_id", {"evidence_id": evidence_id, "experiment_id": payload.get("experiment_id"), "outcome": payload.get("outcome", "abstain"), "model_id": payload.get("model_id", ""), "delta": float(payload.get("delta", 0.0)), "protected_ok": int(bool(payload.get("protected_ok", False))), "payload_json": json.dumps(payload, sort_keys=True)})

    def add_edge(self, subject: str, predicate: str, object_: str, evidence_id: str | None = None) -> None:
        self._conn.execute("INSERT OR IGNORE INTO knowledge_edges(subject,predicate,object,evidence_id) VALUES (?,?,?,?)", (subject, predicate, object_, evidence_id))
        self._conn.commit()

    def search_evidence(self, *, outcome: str | None = None, model_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        clauses, args = [], []
        if outcome:
            clauses.append("outcome=?"); args.append(outcome)
        if model_id:
            clauses.append("model_id=?"); args.append(model_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(f"SELECT payload_json FROM evidence{where} ORDER BY rowid DESC LIMIT ?", [*args, limit]).fetchall()
        return [json.loads(row[0]) for row in rows]

    def count(self, table: str) -> int:
        if table not in {"runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges"}:
            raise ValueError("unknown table")
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def list_rows(self, table: str, *, limit: int = 100) -> list[dict[str, Any]]:
        if table not in {"runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges"}:
            raise ValueError("unknown table")
        return [dict(row) for row in self._conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()]

    def export_json(self) -> dict[str, list[dict[str, Any]]]:
        return {table: self.list_rows(table, limit=100000) for table in ("runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges")}
