"""Durable SQLite state for runs, sources, experiments, and transferable evidence."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
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
CREATE TABLE IF NOT EXISTS graph_nodes (
  node_id TEXT PRIMARY KEY, kind TEXT NOT NULL, node_key TEXT NOT NULL,
  layer TEXT NOT NULL, status TEXT, payload_json TEXT NOT NULL,
  content_digest TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_edges (
  edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, relation TEXT NOT NULL,
  target_id TEXT NOT NULL, evidence_id TEXT, payload_json TEXT NOT NULL,
  content_digest TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(source_id, relation, target_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS knowledge_records (
  record_id TEXT PRIMARY KEY, record_type TEXT NOT NULL, layer TEXT NOT NULL,
  status TEXT, payload_json TEXT NOT NULL, record_digest TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transfer_assessments (
  assessment_id TEXT PRIMARY KEY, source_method_id TEXT NOT NULL,
  target_model_id TEXT NOT NULL, state TEXT NOT NULL, score REAL NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_artifacts (
  artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, relative_ref TEXT,
  content_digest TEXT, size_bytes INTEGER, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind ON graph_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_layer ON graph_nodes(layer);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_transfer_target ON transfer_assessments(target_model_id);
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
        if table not in {"runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges", "graph_nodes", "graph_edges", "knowledge_records", "transfer_assessments", "graph_artifacts"}:
            raise ValueError("unknown table")
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def list_rows(self, table: str, *, limit: int = 100) -> list[dict[str, Any]]:
        if table not in {"runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges", "graph_nodes", "graph_edges", "knowledge_records", "transfer_assessments", "graph_artifacts"}:
            raise ValueError("unknown table")
        return [dict(row) for row in self._conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()]

    def export_json(self) -> dict[str, list[dict[str, Any]]]:
        return {table: self.list_rows(table, limit=100000) for table in ("runs", "documents", "ideas", "probes", "experiments", "evidence", "knowledge_edges")}

    # The graph projection deliberately lives beside the original state tables.
    # This keeps existing callers compatible while giving the public release a
    # typed, queryable and append-only knowledge layer.
    @staticmethod
    def _digest(value: Any) -> str:
        body = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def put_graph_node(
        self,
        node_id: str,
        *,
        kind: str,
        key: str,
        layer: str,
        payload: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        body = dict(payload or {})
        self._put(
            "graph_nodes",
            "node_id",
            {
                "node_id": node_id,
                "kind": kind,
                "node_key": key,
                "layer": layer,
                "status": status,
                "payload_json": json.dumps(body, sort_keys=True),
                "content_digest": self._digest(body),
                "created_at": self._now(),
            },
        )

    def add_graph_edge(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        *,
        evidence_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        identity = {"source": source_id, "relation": relation, "target": target_id, "evidence": evidence_id}
        edge_id = "edge-" + self._digest(identity)[7:23]
        body = dict(payload or {})
        self._put(
            "graph_edges",
            "edge_id",
            {
                "edge_id": edge_id,
                "source_id": source_id,
                "relation": relation,
                "target_id": target_id,
                "evidence_id": evidence_id,
                "payload_json": json.dumps(body, sort_keys=True),
                "content_digest": self._digest({**identity, "payload": body}),
                "created_at": self._now(),
            },
        )
        return edge_id

    def append_knowledge_record(
        self,
        record: dict[str, Any],
        *,
        record_type: str = "evidence",
        layer: str = "L3",
        status: str | None = None,
    ) -> str:
        body = json.loads(json.dumps(record, sort_keys=True))
        digest = self._digest(body)
        record_id = str(body.get("record_id") or body.get("evidence_id") or "record-" + digest[7:23])
        existing = self._conn.execute("SELECT record_digest FROM knowledge_records WHERE record_id=?", (record_id,)).fetchone()
        if existing is not None:
            if str(existing[0]) == digest:
                return record_id
            # Never rewrite a historical record: a changed payload receives a
            # new deterministic identity and remains separately auditable.
            record_id = record_id + "-" + digest[7:23]
        self._put(
            "knowledge_records",
            "record_id",
            {
                "record_id": record_id,
                "record_type": record_type,
                "layer": layer,
                "status": status or str(body.get("status") or body.get("outcome") or "unknown"),
                "payload_json": json.dumps(body, sort_keys=True),
                "record_digest": digest,
                "created_at": self._now(),
            },
        )
        return record_id

    def put_transfer_assessment(
        self,
        assessment_id: str,
        *,
        source_method_id: str,
        target_model_id: str,
        state: str,
        score: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._put(
            "transfer_assessments",
            "assessment_id",
            {
                "assessment_id": assessment_id,
                "source_method_id": source_method_id,
                "target_model_id": target_model_id,
                "state": state,
                "score": float(score),
                "payload_json": json.dumps(payload or {}, sort_keys=True),
                "created_at": self._now(),
            },
        )

    def graph_nodes(self, *, kind: str | None = None, layer: str | None = None, limit: int = 100000) -> list[dict[str, Any]]:
        clauses, args = [], []
        if kind:
            clauses.append("kind=?"); args.append(kind)
        if layer:
            clauses.append("layer=?"); args.append(layer)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return [dict(row) for row in self._conn.execute(f"SELECT * FROM graph_nodes{where} ORDER BY node_id LIMIT ?", [*args, limit]).fetchall()]

    def graph_edges(self, *, node_id: str | None = None, relation: str | None = None, limit: int = 100000) -> list[dict[str, Any]]:
        clauses, args = [], []
        if node_id:
            clauses.append("(source_id=? OR target_id=?)"); args.extend([node_id, node_id])
        if relation:
            clauses.append("relation=?"); args.append(relation)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return [dict(row) for row in self._conn.execute(f"SELECT * FROM graph_edges{where} ORDER BY edge_id LIMIT ?", [*args, limit]).fetchall()]

    def graph_document(self, *, portable: bool = False) -> dict[str, Any]:
        from .knowledge_graph import build_graph_document

        return build_graph_document(self, portable=portable)

    def export_graph(self, path: Path, *, portable: bool = False) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.graph_document(portable=portable), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination
