"""Metric selection and adequacy checks independent of any benchmark suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .runtime import AIProvider
from .benchmark_metrics import MetricCatalog, WorldArenaMetricCatalog, validate_metric_selection


@dataclass(frozen=True)
class MetricPlan:
    primary: str
    protected: tuple[str, ...]
    diagnostic: tuple[str, ...]
    heldout_split: str
    rationale: str
    practical_threshold: float | None = None
    threshold_rationale: str = ""
    primary_direction: str = "maximize"
    protected_directions: tuple[str, ...] = ()
    benchmark: str = ""
    catalog_digest: str = ""
    metric_definitions: tuple[dict[str, Any], ...] = ()
    evaluation_order: tuple[str, ...] = ()
    pilot_metrics: tuple[str, ...] = ()
    promotion_metrics: tuple[str, ...] = ()
    selection_state: str = "legacy"


class MetricAdvisor:
    def __init__(self, ai: AIProvider | None = None):
        self.ai = ai

    def select(self, objective: str, available_signals: list[str], constraints: list[str]) -> MetricPlan:
        if self.ai is None:
            return MetricPlan(objective, tuple(constraints), tuple(available_signals), "heldout", "explicit fallback", primary_direction="maximize", protected_directions=("maximize",) * len(constraints))
        prompt = json.dumps({"objective": objective, "signals": available_signals, "constraints": constraints, "instruction": "Select primary, protected, diagnostic metrics and their directions (maximize or minimize). Also propose a practical improvement threshold for this specific task from baseline variance, measurement noise, and user impact. Return primary_direction and protected_directions. Do not use training loss as the primary metric."}, sort_keys=True)
        try:
            value = json.loads(self.ai.complete(role="metric_advisor", prompt=prompt))
            primary = str(value["primary"])
            protected = tuple(str(v) for v in value.get("protected", constraints))
            diagnostic = tuple(str(v) for v in value.get("diagnostic", available_signals))
            raw_threshold = value.get("practical_threshold")
            threshold = float(raw_threshold) if isinstance(raw_threshold, (int, float)) and float(raw_threshold) > 0 else None
            primary_direction = str(value.get("primary_direction", "maximize"))
            protected_directions = tuple(str(v) for v in value.get("protected_directions", ["maximize"] * len(protected)))
            if primary_direction not in {"maximize", "minimize"} or any(v not in {"maximize", "minimize"} for v in protected_directions):
                raise ValueError("invalid metric direction")
            return MetricPlan(primary, protected, diagnostic, str(value.get("heldout_split", "heldout")), str(value.get("rationale", "")), threshold, str(value.get("threshold_rationale", "")), primary_direction, protected_directions)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return MetricPlan(objective, tuple(constraints), tuple(available_signals), "heldout", "invalid advisor response; fallback", primary_direction="maximize", protected_directions=("maximize",) * len(constraints))

    def select_benchmark(
        self,
        objective: str,
        model_report: dict[str, Any],
        available_signals: list[str],
        *,
        catalog: MetricCatalog | None = None,
    ) -> dict[str, Any]:
        """Select suitable benchmark metrics, then validate deterministically."""
        catalog = catalog or WorldArenaMetricCatalog.default()
        payload = catalog.as_prompt_payload(model_report, available_signals)
        if self.ai is None:
            return {"state": "abstain", "reason": "AI provider not configured", **payload}
        prompt = json.dumps({
            "objective": objective,
            "model_report": model_report,
            "available_signals": available_signals,
            "benchmark_catalog": payload,
            "instruction": "Choose one formal primary, zero or more formal protected metrics, and diagnostic metrics. Use only eligible catalog metric_id values. Prefer low-cost metrics for pilot, but include a dynamics or long-horizon metric when eligible. Return JSON with primary, protected, diagnostic, evaluation_order, practical_threshold, rationale. Never promote a metric lacking machine ground truth.",
        }, sort_keys=True)
        try:
            selection = json.loads(self.ai.complete(role="benchmark_metric_advisor", prompt=prompt))
            if not isinstance(selection, dict):
                raise TypeError("benchmark selection must be an object")
        except (json.JSONDecodeError, TypeError, AttributeError):
            selection = {}
        result = validate_metric_selection(selection, catalog, model_report, available_signals)
        if result.get("state") != "validated":
            # A provider failure must not invent metrics or stop a safe pilot.
            # Pick only catalogued, machine-grounded entries using a stable
            # deterministic order; the reason remains visible in the audit.
            eligible, rejected = catalog.candidates(model_report, available_signals)
            formal = [item for item in eligible if not item.diagnostic_only and item.ground_truth and "primary" in item.role_candidates]
            protected = [item for item in eligible if not item.diagnostic_only and item.ground_truth and "protected" in item.role_candidates]
            diagnostics = [item for item in eligible if "diagnostic" in item.role_candidates]
            if formal and protected:
                primary = formal[0].metric_id
                fallback = {"primary": primary, "protected": [item.metric_id for item in protected if item.metric_id != primary][:3], "diagnostic": [item.metric_id for item in diagnostics if item.metric_id != primary][:3], "rationale": "deterministic catalog fallback after advisor failure"}
                result = validate_metric_selection(fallback, catalog, model_report, available_signals)
                result["advisor_fallback"] = True
            else:
                result = {"state": "abstain", "reason": "no eligible formal benchmark metrics", "rejected": rejected, **payload}
        result["practical_threshold"] = selection.get("practical_threshold")
        result["threshold_rationale"] = str(selection.get("threshold_rationale", ""))
        return result

    def adequate(self, plan: MetricPlan) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if not plan.primary:
            missing.append("primary metric")
        if not plan.heldout_split:
            missing.append("heldout split")
        if not plan.protected:
            missing.append("protected metric or explicit no-protection declaration")
        return not missing, missing

    def review_benchmarks(self, objective: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask the shared AI to identify benchmark gaps; never auto-promotes a metric."""
        if self.ai is None:
            return {"status": "abstain", "reason": "AI provider not configured", "gaps": []}
        prompt = json.dumps({"objective": objective, "sources": sources, "instruction": "Compare these benchmark descriptions. Return JSON with benchmark_names, coverage_gaps, confounds, and proposed_diagnostics."}, sort_keys=True)
        try:
            value = json.loads(self.ai.complete(role="benchmark_reviewer", prompt=prompt))
            if not isinstance(value, dict):
                raise TypeError("benchmark review must be an object")
            return {"status": "reviewed", **value}
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {"status": "abstain", "reason": "invalid benchmark review", "gaps": []}


class MetricRegistry:
    """Approved metric implementations are explicit and independently testable."""

    def __init__(self):
        self._metrics: dict[str, Any] = {}

    def register(self, metric_id: str, evaluator: Any) -> None:
        if metric_id in self._metrics:
            raise ValueError(f"metric already registered: {metric_id}")
        self._metrics[metric_id] = evaluator

    def evaluate(self, metric_id: str, prediction: Any, target: Any) -> float:
        if metric_id not in self._metrics:
            raise KeyError(metric_id)
        return float(self._metrics[metric_id](prediction, target))
