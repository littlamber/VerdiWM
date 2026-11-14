"""Shared standard-library contracts for Agibot training traces."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TRACE_ARTIFACT_TYPE = "verdiwm-agibot-training-trace"
TRACE_CONFIG_TYPE = "verdiwm-agibot-training-trace-config"
_TRACE_KEY = "_verdi_agibot_trace_samples"

class AgibotTraceError(RuntimeError):
    """The native training run did not produce trustworthy consumption trace."""


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sample(sample: Mapping[str, object], *, split: str) -> dict[str, object]:
    if sample.get("split_id") != split:
        raise AgibotTraceError("AGIBOT_TRACE_SAMPLE_SPLIT_MISMATCH")
    required = (
        "dataset_index",
        "dataset_key",
        "episode_entry_index",
        "source_sample_index",
        "episode_uid",
        "first_future_frame",
        "sample_window_id",
    )
    if any(key not in sample for key in required):
        raise AgibotTraceError("AGIBOT_TRACE_SAMPLE_IDENTITY_MISSING")
    if (
        type(sample["dataset_index"]) is not int
        or type(sample["episode_entry_index"]) is not int
        or type(sample["source_sample_index"]) is not int
        or type(sample["first_future_frame"]) is not int
        or any(not isinstance(sample[key], str) or not sample[key] for key in ("dataset_key", "episode_uid", "sample_window_id"))
        or sample["sample_window_id"] != f"{sample['episode_uid']}:{sample['first_future_frame']:06d}"
    ):
        raise AgibotTraceError("AGIBOT_TRACE_SAMPLE_IDENTITY_INVALID")
    return {key: sample[key] for key in ("split_id", *required)}


def _rank_from_environment() -> int:
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise AgibotTraceError("AGIBOT_TRACE_RANK_INVALID") from exc
    if rank < 0:
        raise AgibotTraceError("AGIBOT_TRACE_RANK_INVALID")
    return rank


def _finite_float(value: object) -> float:
    try:
        result = float(value.detach().item())
    except Exception:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise AgibotTraceError("AGIBOT_TRACE_LOSS_INVALID") from exc
    if not math.isfinite(result):
        raise AgibotTraceError("AGIBOT_TRACE_LOSS_INVALID")
    return result


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotTraceError("AGIBOT_TRACE_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise AgibotTraceError("AGIBOT_TRACE_JSON_INVALID")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
