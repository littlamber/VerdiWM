"""Read-only onboarding scan components: io."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from wmloop.control.onboarding_scan.types import _MAX_ERROR
from wmloop.storage import atomic_write, canonical_json_bytes, sha256_file


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _sha256_file(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def _bounded(value: str, *, limit: int = _MAX_ERROR) -> str:
    value = value.replace("\x00", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(payload, ensure_ascii=False)


def _write_bytes(path: Path, payload: bytes) -> None:
    atomic_write(path, payload)
