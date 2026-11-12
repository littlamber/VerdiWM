"""Resumable pipeline state and evidence components: io."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.storage import atomic_write, canonical_json_bytes, checked_path, sha256_bytes


def _resolved(value: Path | None, default: Path) -> Path:
    return Path(value if value is not None else default).expanduser().resolve()


def _optional_file(value: Path | None, code: str) -> Path | None:
    if value is None:
        return None
    path = checked_path(value, code=code, error=AutonomousPipelineError)
    if not path.is_file() or path.is_symlink():
        raise AutonomousPipelineError(code)
    return path


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousPipelineError(code) from exc
    if not isinstance(value, dict):
        raise AutonomousPipelineError(code)
    return value


def _canonical_json(value: object) -> bytes:
    return canonical_json_bytes(value)


def _sha256(payload: bytes) -> str:
    return sha256_bytes(payload)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    atomic_write(path, _canonical_json(value))
