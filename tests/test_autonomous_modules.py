from pathlib import Path

from adapters.fixture_world import FixtureWorldAdapter
from verdi_core.ideas import AutonomousResearchPlanner, DualRouteIdeator, relevant_source_documents
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


def test_research_planner_extracts_query_from_structured_plan() -> None:
    class StructuredAI:
        def complete(self, *, role, prompt):
            return '{"queries":[{"query":"video diffusion temporal robustness","rationale":"targeted"}],"rationale":"gap driven","adjacent_fields":[]}'

    plan = AutonomousResearchPlanner(StructuredAI()).plan("improve", {})
    assert plan.queries == ("video diffusion temporal robustness",)


def test_source_relevance_rejects_unrelated_search_noise() -> None:
    documents = [
        {"title": "Video diffusion world models", "text": "Temporal action-conditioned diffusion model training", "source_url": "https://example.test/relevant"},
        {"title": "Groundwater governance", "text": "A randomized field experiment", "source_url": "https://example.test/noise"},
    ]
    selected = relevant_source_documents(
        documents,
        objective="improve world model sampling robustness",
        portrait={"architecture_facets": ["diffusion", "spatio_temporal_unet"], "capabilities": ["action_conditioned"]},
    )
    assert [item["source_url"] for item in selected] == ["https://example.test/relevant"]
