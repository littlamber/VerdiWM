"""Shared fail-closed apply target until a reviewed ACWM mechanism is ported."""

from __future__ import annotations

from typing import Mapping


class PrimitiveApplyUnavailable(RuntimeError):
    pass


def apply_requires_materialized_vendor(repo_worktree: str, params: Mapping[str, object]) -> str:
    del repo_worktree, params
    raise PrimitiveApplyUnavailable(
        "PRIMITIVE_TEMPLATE_NOT_PORTED: a registered mechanism needs a reviewed ACWM worktree template"
    )
