"""Offline campaign runner used to smoke-test autonomous orchestration."""

from __future__ import annotations

from typing import Any


_attempts: dict[tuple[str, str], int] = {}


def runner(idea: dict[str, Any], stage: str, context: dict[str, Any]) -> dict[str, Any]:
    key = (str(idea.get("idea_id", "idea")), stage)
    _attempts[key] = _attempts.get(key, 0) + 1
    if stage == "static_check" and _attempts[key] == 1:
        return {"state": "runtime_failed", "reason": "fixture generated code needs repair"}
    if stage == "heldout_evaluate":
        return {
            "state": "completed",
            "outcome": "confirmed_positive",
            "delta": 0.12,
            "protected_ok": True,
            "independent_replicates": 2,
            "split": "heldout",
            "verifier_digest": "sha256:" + "f" * 64,
            "claim_boundary": "fixture-only orchestration evidence",
        }
    return {"state": "completed", "stage": stage}
