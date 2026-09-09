"""Batch identity admission and status projection shared by CLI and publication."""

from __future__ import annotations

import json
from pathlib import Path
import re

from wmloop.storage import canonical_bytes, atomic_write, checked_path


class ModelBatchError(ValueError):
    """A batch input, identity or state projection failed validation."""


MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"


def validate_model_ids(values: object) -> list[str]:
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or re.fullmatch(MODEL_ID_PATTERN, value) is None for value in values):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_MODEL_IDS_INVALID")
    if len(set(values)) != len(values):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_MODEL_IDS_INVALID")
    return sorted(values)


def admit_batch(root: Path, plan: dict) -> None:
    """Bind the output before compilation, enqueueing, or model side effects."""
    identity = {"batch_id": plan["batch_id"], "plan_sha256": plan["plan_sha256"]}
    for path in (root / ".batch-identity.json", root / "execution.json", root / "plan.json"):
        checked_path(path, code="MODEL_BATCH_EXECUTION_OUTPUT_INVALID", error=ModelBatchError)
        if not path.exists():
            continue
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelBatchError("MODEL_BATCH_EXECUTION_OUTPUT_INVALID") from exc
        if not isinstance(previous, dict) or any(previous.get(k) != v for k, v in identity.items()):
            raise ModelBatchError("MODEL_BATCH_EXECUTION_PLAN_CONFLICT")
    atomic_write(root / ".batch-identity.json", canonical_bytes(identity) + b"\n")


def aggregate_states(states) -> str:
    observed = set(states)
    if observed == {"completed"}:
        return "completed"
    if "unavailable" in observed:
        return "unavailable"
    if "running" in observed:
        return "running"
    if "queued" in observed:
        return "partial" if observed & {"blocked", "failed", "cancelled"} else "queued"
    if observed == {"blocked"}:
        return "blocked"
    if observed and observed <= {"failed", "cancelled"}:
        return "failed"
    return "partial"
