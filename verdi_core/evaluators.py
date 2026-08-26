"""Frozen, model-independent evidence evaluation."""

from __future__ import annotations

from typing import Any
import math

from .evidence import classify_paired_effect
from .benchmark_metrics import evaluate_metric_bundle


class GenericEvaluator:
    """Classifies normalized worker output without knowing model internals."""

    evaluator_id = "verdi-generic-evaluator-v1"

    def evaluate(self, artifacts: dict[str, Any], *, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
        raw = artifacts.get("raw_result", artifacts)
        # When an adapter returns a baseline/candidate bundle, the Kernel owns
        # the verdict gate.  An adapter may not hide a protected regression in
        # a weighted aggregate or a self-declared outcome.
        if isinstance(raw, dict) and "baseline_metrics" in raw and "candidate_metrics" in raw and metrics.get("metric_definitions"):
            plan = {
                "primary": metrics.get("primary"),
                "protected": metrics.get("protected", ()),
                "diagnostic": metrics.get("diagnostic", ()),
                "definitions": metrics.get("metric_definitions", ()),
                "practical_threshold": metrics.get("practical_threshold", 0.0),
                "catalog_digest": metrics.get("catalog_digest", ""),
            }
            result = evaluate_metric_bundle(raw, plan, split=split)
            return {**result, "evaluator_id": self.evaluator_id, "metrics": metrics}
        if "outcome" in raw:
            outcome = str(raw["outcome"])
        else:
            delta = float(raw.get("delta", 0.0))
            if "baseline" in raw and "candidate" in raw:
                direction = str(metrics.get("primary_direction", "maximize"))
                delta = float(raw["candidate"] - raw["baseline"] if direction == "maximize" else raw["baseline"] - raw["candidate"])
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


class StatisticalEvaluator(GenericEvaluator):
    """Adds repeat-aware effect and uncertainty checks to normalized metrics."""

    evaluator_id = "verdi-statistical-evaluator-v1"

    def evaluate(self, artifacts: dict[str, Any], *, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
        result = super().evaluate(artifacts, split=split, metrics=metrics)
        raw = artifacts.get("raw_result", artifacts)
        values = [float(value) for value in raw.get("replicate_deltas", [])]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            stderr = math.sqrt(variance / len(values))
            result.update({"replicates": len(values), "mean_delta": mean, "stderr": stderr, "uncertainty": "bounded" if stderr < max(0.01, abs(mean) * 0.5) else "high"})
            if result["outcome"] == "confirmed_positive" and (mean <= 0 or stderr >= max(0.01, abs(mean) * 0.5)):
                result["outcome"] = "abstain"
        return result


class PairedMetricEvaluator(GenericEvaluator):
    """Adapter-facing evaluator for direction-normalized paired held-out deltas."""

    evaluator_id = "verdi-paired-metric-evaluator-v1"

    def evaluate(self, artifacts: dict[str, Any], *, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
        raw = artifacts.get("raw_result", artifacts)
        values = [float(value) for value in raw.get("replicate_deltas", [])]
        result = classify_paired_effect(
            values,
            practical_threshold=float(metrics.get("practical_threshold", 0.0)),
            protected_ok=bool(raw.get("protected_ok", True)),
            min_replicates=int(metrics.get("min_replicates", 2)),
            bootstrap_samples=int(metrics.get("bootstrap_samples", 10000)),
            seed=int(metrics.get("seed", 20260825)),
        )
        return {
            **result,
            "ci95": [result["ci95_low"], result["ci95_high"]] if "ci95_low" in result else None,
            "metric_direction": str(metrics.get("primary_direction", "maximize")),
            "claim_boundary": str(raw.get("claim_boundary", "paired held-out metric evidence; task success requires domain evaluation.")),
            "artifact_digest": str(raw.get("artifact_digest", "")),
            "split": split,
            "evaluator_id": self.evaluator_id,
            "metrics": metrics,
        }
