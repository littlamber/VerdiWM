"""Controlled primitive template rendering before a sandbox can be mutated.

The renderer is intentionally stricter than `git apply`: every chosen action
must be in the frozen registry, have a valid H1--H5 IO contract, render a
unified diff, and leave evaluator files untouched.  A registry entry without a
reviewed ACWM template fails before a worktree is changed.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.primitives.hooks import HookContractError, validate_hook_io
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveSelection, PrimitiveValidationError
from wmloop.primitives.unavailable import PrimitiveApplyUnavailable


class PrimitiveRenderError(RuntimeError):
    """A template was not safe to validate or apply to the trial worktree."""


_FROZEN_EVALUATOR_PATHS = frozenset({"eval.py", "scripts/eval_all.sh"})


@dataclass(frozen=True)
class RenderedPrimitive:
    name: str
    diff: str
    sha256: str


class PrimitiveRenderer:
    def __init__(self, registry: PrimitiveRegistry) -> None:
        self._registry = registry

    def render_checked(
        self,
        *,
        worktree: Path,
        interventions: Sequence[Mapping[str, Any]],
        hook_ios: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
    ) -> tuple[RenderedPrimitive, ...]:
        selections = self._selections(interventions)
        self._validate_hook_ios(selections, hook_ios)
        rendered: list[RenderedPrimitive] = []
        for selection in selections:
            manifest = self._registry.manifest(selection.name)
            try:
                module = importlib.import_module(manifest.apply_module)
                apply = getattr(module, "apply")
                diff = apply(str(Path(worktree).resolve()), dict(selection.params))
            except (ImportError, AttributeError, PrimitiveApplyUnavailable) as exc:
                raise PrimitiveRenderError(f"PRIMITIVE_RENDER_UNAVAILABLE:{selection.name}") from exc
            if not isinstance(diff, str) or not diff.strip():
                raise PrimitiveRenderError(f"PRIMITIVE_RENDER_DIFF_INVALID:{selection.name}")
            assert_diff_does_not_modify_frozen_evaluator(diff)
            self._git_apply_check(worktree, diff)
            rendered.append(RenderedPrimitive(selection.name, diff, hashlib.sha256(diff.encode("utf-8")).hexdigest()))
        return tuple(rendered)

    def _selections(self, interventions: Sequence[Mapping[str, Any]]) -> tuple[PrimitiveSelection, ...]:
        if not interventions:
            raise PrimitiveRenderError("PRIMITIVE_RENDER_INTERVENTIONS_EMPTY")
        try:
            pairs = []
            for item in interventions:
                primitive = item.get("primitive")
                params = item.get("params")
                if not isinstance(primitive, str) or not isinstance(params, Mapping):
                    raise PrimitiveValidationError("PRIMITIVE_RENDER_INTERVENTION_INVALID")
                pairs.append((primitive, params))
            return self._registry.validate_combination(pairs)
        except PrimitiveValidationError as exc:
            raise PrimitiveRenderError(str(exc)) from exc

    def _validate_hook_ios(
        self,
        selections: Sequence[PrimitiveSelection],
        hook_ios: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
    ) -> None:
        for selection in selections:
            for hook in selection.hooks:
                io = hook_ios.get(hook)
                if io is None or len(io) != 2:
                    raise PrimitiveRenderError(f"PRIMITIVE_HOOK_IO_MISSING:{hook}")
                try:
                    validate_hook_io(hook, inputs=io[0], outputs=io[1])
                except HookContractError as exc:
                    raise PrimitiveRenderError(str(exc)) from exc

    @staticmethod
    def _git_apply_check(worktree: Path, diff: str) -> None:
        completed = subprocess.run(
            ["git", "-C", str(Path(worktree).resolve()), "apply", "--check", "--whitespace=error-all", "-"],
            input=diff,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise PrimitiveRenderError("PRIMITIVE_RENDER_GIT_APPLY_CHECK_FAILED")


def assert_diff_does_not_modify_frozen_evaluator(diff: str) -> None:
    for line in diff.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        path = line[4:].split("\t", 1)[0].strip()
        if path in {"/dev/null", ""}:
            continue
        normalized = path.removeprefix("a/").removeprefix("b/")
        if normalized in _FROZEN_EVALUATOR_PATHS:
            raise PrimitiveRenderError("PRIMITIVE_RENDER_FROZEN_EVALUATOR_MODIFICATION")
