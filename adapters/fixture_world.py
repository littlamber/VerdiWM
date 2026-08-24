"""A deterministic adapter used to test the control plane without a model."""

from __future__ import annotations

from typing import Any


class FixtureWorldAdapter:
    adapter_id = "fixture-world"
    version = "1"

    def inspect(self) -> dict[str, Any]:
        return {
            "model_id": "fixture-world-v1",
            "revision": "fixture-revision-1",
            "capabilities": ["inference", "evaluation", "rollout", "intervention"],
            "hooks": ["action-conditioning"],
            "evaluator_id": "fixture-heldout-verifier-v1",
        }

    def probe(self, probe_id: str) -> dict[str, Any]:
        if probe_id != "action-sensitivity":
            raise ValueError("unknown probe")
        return {"response_digest": "sha256:" + "1" * 64, "uncertainty": "deterministic"}

    def intervene(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        return dict(hypothesis)

    def evaluate(self, intervention: dict[str, Any], split: str) -> dict[str, Any]:
        outcome = {
            "baseline": ("null", 0.0, True),
            "bounded-repair": ("confirmed_positive", 0.12, True),
            "overdose-control": ("harmful", -0.18, False),
        }[str(intervention["hypothesis_id"])]
        return {"outcome": outcome[0], "delta": outcome[1], "protected_ok": outcome[2], "split": split}

