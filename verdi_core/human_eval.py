"""Generic ingestion of human-labelled video evaluation batches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .contracts import canonical_digest


@dataclass(frozen=True)
class HumanVideoBatch:
    batch_id: str
    episode_ids: tuple[str, ...]
    video_paths: tuple[str, ...]
    split: str

    def manifest(self) -> dict[str, Any]:
        value = {"batch_id": self.batch_id, "episode_ids": self.episode_ids, "video_paths": self.video_paths, "split": self.split}
        return {**value, "digest": canonical_digest(value)}


def evaluate_labels(batch: HumanVideoBatch, labels: dict[str, Any]) -> dict[str, Any]:
    expected = set(batch.episode_ids)
    received = set(labels)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        return {"outcome": "abstain", "reason": "label_set_mismatch", "missing": missing, "extra": extra, "batch": batch.manifest()}
    successes = []
    for episode_id in batch.episode_ids:
        value = labels[episode_id]
        success = value if isinstance(value, bool) else value.get("success") if isinstance(value, dict) else None
        if not isinstance(success, bool):
            return {"outcome": "abstain", "reason": "invalid_human_label", "episode_id": episode_id, "batch": batch.manifest()}
        successes.append(success)
    rate = sum(successes) / len(successes) if successes else 0.0
    return {"outcome": "measured", "success_rate": rate, "successes": sum(successes), "episodes": len(successes), "split": batch.split, "batch": batch.manifest(), "labels_digest": canonical_digest(labels)}


def load_labels(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("human label file must contain a JSON object")
    return value
