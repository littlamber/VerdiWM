"""Offline AI and source fixtures for the complete release demo."""

from __future__ import annotations


class FixtureResearchAI:
    provider_id = "fixture-ai"

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "research_planner":
            return '{"queries":["world model long horizon repair"],"rationale":"fixture","adjacent_fields":["sequential prediction"]}'
        if role == "metric_advisor":
            return '{"primary":"quality","protected":["safety"],"diagnostic":["horizon"],"heldout_split":"heldout","rationale":"fixture"}'
        if role in {"paper_extractor", "code_extractor"}:
            return '[{"title":"bounded repair","mechanism":"reduce long horizon drift","expected_effect":"quality","risks":["safety"],"evidence_basis":["fixture-source"]}]'
        if role == "benchmark_reviewer":
            return '{"benchmark_names":["fixture-benchmark"],"coverage_gaps":["short horizon only"],"confounds":[],"proposed_diagnostics":["long horizon"]}'
        return "[]"


class FixtureEngineeringAI:
    """Deterministic engineering backend for offline autonomy smoke tests."""

    provider_id = "fixture-engineering-ai"

    def complete(self, *, role: str, prompt: str) -> str:
        # The fixture repair is intentionally minimal: the important contract
        # under test is that a bounded AI tool loop records a receipt and lets
        # the campaign retry the failed stage without operator intervention.
        return '{"action":"done","args":{"state":"completed","reason":"fixture repair verified"}}'
