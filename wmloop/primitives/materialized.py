"""Small reviewed primitive diff helpers for M2 smoke execution.

These helpers intentionally materialize sidecar intervention manifests rather
than pretending that an ACWM training hook has already been ported.  They give
the executor a real git patch to render, apply, validate and receipt while the
heavier provider integration remains explicit future work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def render_sidecar_diff(
    *,
    repo_worktree: str,
    primitive: str,
    layer: str,
    hook: str,
    params: Mapping[str, object],
    notes: tuple[str, ...],
    intent_to_code_contract: Mapping[str, object] | None = None,
) -> str:
    worktree = Path(repo_worktree)
    if not primitive or "/" in primitive or "\\" in primitive:
        raise ValueError("PRIMITIVE_SIDECAR_NAME_INVALID")
    if not worktree.exists() or not worktree.is_dir():
        raise ValueError("PRIMITIVE_SIDECAR_WORKTREE_INVALID")
    path = f"wmloop_interventions/{primitive}.json"
    if (worktree / path).exists():
        raise ValueError("PRIMITIVE_SIDECAR_ALREADY_EXISTS")
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": primitive,
        "layer": layer,
        "hook": hook,
        "materialization_state": "smoke_sidecar_only",
        "params": dict(params),
        "intent_to_code_contract": dict(intent_to_code_contract)
        if intent_to_code_contract is not None
        else _sidecar_only_contract(primitive=primitive, hook=hook, notes=notes),
        "notes": list(notes),
    }
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    lines = text.splitlines()
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{path}",
            f"@@ -0,0 +1,{len(lines)} @@",
            *(f"+{line}" for line in lines),
            "",
        ]
    )


def _sidecar_only_contract(*, primitive: str, hook: str, notes: tuple[str, ...]) -> dict[str, object]:
    method_intent = notes[0] if notes else f"Stage primitive {primitive} for later executor materialization."
    return {
        "method_intent": method_intent,
        "runtime_behavior": (
            f"No ACWM runtime hook is activated for {primitive}; the rendered artifact is a {hook} work-order sidecar only."
        ),
        "declared_proxy": "Sidecar records parameters and routing intent for later agent implementation; it is not a behavioral proxy.",
        "not_claimed": [
            "does not change ACWM training",
            "does not change ACWM inference",
            "does not provide closed-loop evidence",
        ],
    }
