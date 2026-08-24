"""Composition root for a model-independent VerdiWM runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .adapter import ModelAdapter


class AIProvider(Protocol):
    provider_id: str

    def complete(self, *, role: str, prompt: str) -> str: ...


@dataclass
class RuntimeBindings:
    """One place where a user-provided SDK is bound to the system."""

    model: ModelAdapter
    ai: AIProvider | None = None
    state_root: Path = Path("state/verdi")
    options: dict[str, Any] = field(default_factory=dict)


_ACTIVE: RuntimeBindings | None = None


def configure(bindings: RuntimeBindings) -> RuntimeBindings:
    global _ACTIVE
    _ACTIVE = bindings
    return bindings


def active() -> RuntimeBindings:
    if _ACTIVE is None:
        raise RuntimeError("VerdiWM is not configured; call configure(RuntimeBindings(...))")
    return _ACTIVE
