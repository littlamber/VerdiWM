"""Append-only, path-free knowledge projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Evidence, canonical_digest, to_document


class KnowledgeGraph:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "knowledge.jsonl"

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
                return record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

