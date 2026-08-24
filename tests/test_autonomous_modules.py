from pathlib import Path

from adapters.fixture_world import FixtureWorldAdapter
from verdi_core.ideas import DualRouteIdeator
from verdi_core.research import ResearchSystem
from verdi_core.retrieval import OnlineRetriever, SearchHit
from verdi_core.runtime import RuntimeBindings


class FakeAI:
    provider_id = "fake-ai"

    def complete(self, *, role, prompt):
        if role == "research_planner":
            return '{"queries":["world model long horizon repair", "adjacent sequential prediction methods"], "rationale":"failure-driven", "adjacent_fields":["sequential prediction"]}'
        if role == "metric_advisor":
            return '{"primary":"quality","protected":["safety"],"diagnostic":["horizon"],"heldout_split":"heldout","rationale":"coverage"}'
        if role in {"paper_extractor", "code_extractor"}:
            return '[{"title":"bounded repair","mechanism":"reduce drift","expected_effect":"quality","risks":["safety"]}]'
        return "{}"


class FakeSearch:
    def search(self, query):
        return [SearchHit("https://invalid.local/missing", title=query)]


class Route:
    route_id = "route"

    def extract(self, documents, context):
        return [{"title": "bounded repair", "mechanism": "reduce drift", "expected_effect": "quality", "risks": ["safety"]}]


def test_autonomous_composition_keeps_model_boundary(tmp_path: Path) -> None:
    bindings = RuntimeBindings(FixtureWorldAdapter(), ai=FakeAI(), state_root=tmp_path, options={"budget": 1.0})
    system = ResearchSystem(bindings, retriever=OnlineRetriever(FakeSearch(), state_root=tmp_path, timeout=0.01), ideator=DualRouteIdeator((Route(), Route())))
    report = system.run_cycle(objective="quality", constraints=["safety"])
    assert report["metrics_adequate"] is True
    assert report["idea_count"] == 1
    assert report["evidence_count"] == 1
    assert (tmp_path / "retrieval" / "ledger.jsonl").exists()


def test_default_dual_ai_routes_use_shared_provider(tmp_path: Path) -> None:
    bindings = RuntimeBindings(FixtureWorldAdapter(), ai=FakeAI(), state_root=tmp_path, options={"budget": 1.0})
    system = ResearchSystem(bindings, retriever=OnlineRetriever(FakeSearch(), state_root=tmp_path, timeout=0.01))
    report = system.run_cycle(objective="quality", constraints=["safety"])
    assert report["idea_count"] == 1
