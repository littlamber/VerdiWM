"""Append-only, path-free knowledge projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Evidence, canonical_digest, to_document
from .storage import SQLiteState


class KnowledgeGraph:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "knowledge.jsonl"
        self.state = SQLiteState(self.root / "knowledge.sqlite3")
        self._hydrate_state()

    def _hydrate_state(self) -> None:
        """Rebuild the query projection after a restart or legacy JSONL import."""
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.state.put_evidence(record["evidence_id"], record)
            self.state.add_edge(record["experiment_id"], "produced", record["evidence_id"], record["evidence_id"])

    def append(self, evidence: Evidence) -> dict[str, Any]:
        record = {"artifact_type": "verdiwm-knowledge-record", **to_document(evidence)}
        record["record_digest"] = canonical_digest(record)
        if self.path.exists():
            existing = {
                json.loads(line)["record_digest"]
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if record["record_digest"] in existing:
                self.state.put_evidence(record["evidence_id"], record)
                return record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.state.put_evidence(record["evidence_id"], record)
        self.state.add_edge(record["experiment_id"], "produced", record["evidence_id"], record["evidence_id"])
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def search(self, *, outcome: str | None = None, model_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.state.search_evidence(outcome=outcome, model_id=model_id, limit=limit)
