"""Metric selection and adequacy checks independent of any benchmark suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .runtime import AIProvider


@dataclass(frozen=True)
class MetricPlan:
    primary: str
    protected: tuple[str, ...]
    diagnostic: tuple[str, ...]
    heldout_split: str
    rationale: str


class MetricAdvisor:
    def __init__(self, ai: AIProvider | None = None):
        self.ai = ai

    def select(self, objective: str, available_signals: list[str], constraints: list[str]) -> MetricPlan:
        if self.ai is None:
            return MetricPlan(objective, tuple(constraints), tuple(available_signals), "heldout", "explicit fallback")
        prompt = json.dumps({"objective": objective, "signals": available_signals, "constraints": constraints, "instruction": "Select primary, protected, diagnostic metrics and justify sufficiency."}, sort_keys=True)
        try:
            value = json.loads(self.ai.complete(role="metric_advisor", prompt=prompt))
            primary = str(value["primary"])
            protected = tuple(str(v) for v in value.get("protected", constraints))
            diagnostic = tuple(str(v) for v in value.get("diagnostic", available_signals))
            return MetricPlan(primary, protected, diagnostic, str(value.get("heldout_split", "heldout")), str(value.get("rationale", "")))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return MetricPlan(objective, tuple(constraints), tuple(available_signals), "heldout", "invalid advisor response; fallback")

    def adequate(self, plan: MetricPlan) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if not plan.primary:
            missing.append("primary metric")
        if not plan.heldout_split:
            missing.append("heldout split")
        if not plan.protected:
            missing.append("protected metric or explicit no-protection declaration")
        return not missing, missing
