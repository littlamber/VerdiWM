"""Training and frozen evaluation smoke components: io."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.storage import atomic_write, canonical_json_bytes


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_JSON_INVALID"
        ) from exc
    if not isinstance(payload, dict):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_JSON_INVALID")
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(payload, ensure_ascii=False)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    atomic_write(path, payload)
