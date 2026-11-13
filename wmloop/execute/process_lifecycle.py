"""Compatibility exports for the shared local process lifecycle primitives."""

from __future__ import annotations

from wmloop.process_lifecycle import (
    LocalCommandSignals,
    LocalProcessCancelled,
    ProcessLifecycleError,
    retire_process_group,
)

__all__ = [
    "LocalCommandSignals",
    "LocalProcessCancelled",
    "ProcessLifecycleError",
    "retire_process_group",
]
