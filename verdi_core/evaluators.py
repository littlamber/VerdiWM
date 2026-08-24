"""Frozen, model-independent evidence evaluation."""

from __future__ import annotations

from typing import Any


class GenericEvaluator:
    """Classifies normalized worker output without knowing model internals."""

    evaluator_id = "verdi-generic-evaluator-v1"

    def evaluate(self, artifacts: dict[str, Any], *, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
        raw = artifacts.get("raw_result", artifacts)
        if "outcome" in raw:
            outcome = str(raw["outcome"])
        else:
            delta = float(raw.get("delta", 0.0))
            protected_ok = bool(raw.get("protected_ok", False))
            outcome = "confirmed_positive" if delta > 0 and protected_ok else ("harmful" if delta < 0 or not protected_ok else "null")
        return {
            "outcome": outcome,
            "delta": float(raw.get("delta", 0.0)),
            "protected_ok": bool(raw.get("protected_ok", False)),
            "split": split,
            "evaluator_id": self.evaluator_id,
            "metrics": metrics,
        }
