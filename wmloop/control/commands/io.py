"""Output helpers for CLI protocol responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def repository_root() -> Path:
    """Return the checkout root without depending on the current directory."""

    # ``io.py`` lives at wmloop/control/commands/io.py.
    return Path(__file__).resolve().parents[3]


def emit_json(payload: object) -> None:
    """Emit one stable machine-readable CLI response on stdout."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def emit_text(value: str) -> None:
    """Emit human-readable CLI text without bypassing the output boundary."""

    sys.stdout.write(value)
