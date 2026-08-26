from pathlib import Path

from verdi_core.benchmark_metrics import (
    MetricCatalogDiscovery,
    MetricMaterializer,
    WorldArenaMetricCatalog,
    evaluate_metric_bundle,
    validate_metric_materialization,
    validate_metric_selection,
)
from verdi_core.metrics import MetricAdvisor
from verdi_core.research import ResearchSystem
from verdi_core.runtime import RuntimeBindings


class BenchmarkAI:
    provider_id = "benchmark-test-ai"

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "benchmark_metric_advisor":
            return '{"primary":"rollout_video_l1","protected":["segment_final_mae","horizon_drift_slope"],"diagnostic":["action_conditioning_sensitivity"],"evaluation_order":["rollout_video_l1","segment_final_mae","horizon_drift_slope"],"practical_threshold":0.01,"rationale":"ground truth rollout and long-horizon coverage"}'
        if role == "benchmark_catalog_extractor":
            return '{"metrics":[{"metric_id":"worldarena_extra","benchmark":"worldarena","description":"extra declared metric","direction":"maximize","role_candidates":["diagnostic"],"required_signals":["video_ground_truth"],"required_capabilities":["rollout"],"cost":"low","ground_truth":true,"evaluator_ref":"worldarena:extra:v1","source_refs":["official"],"diagnostic_only":true,"implementation_status":"catalogued"}]}'
        if role == "metric_advisor":
            return '{"primary":"quality","protected":["safety"],"diagnostic":[],"heldout_split":"heldout","rationale":"legacy"}'
        return "{}"


class BenchmarkAdapter:
    adapter_id = "benchmark-fixture"
    version = "1"

    def inspect(self):
        return {
            "model_id": "benchmark-world-v1", "revision": "r1", "evaluator_id": "benchmark-eval-v1",
            "capabilities": ["rollout", "evaluation", "intervention"], "hooks": ["action-conditioning"],
            "available_signals": ["video_ground_truth", "state_ground_truth", "interaction_index", "action_counterfactual"],
            "architecture_facets": ["dit"], "heldout_split": "worldarena-heldout-v1",
        }

    def probe(self, probe_id):
        return {"probe_id": probe_id, "response_digest": "sha256:" + "b" * 64}

    def intervene(self, hypothesis):
        return hypothesis

    def evaluate(self, intervention, split):
        return {"baseline_metrics": {"rollout_video_l1": 1.0, "segment_final_mae": 1.0, "horizon_drift_slope": 1.0}, "candidate_metrics": {"rollout_video_l1": 0.8, "segment_final_mae": 0.9, "horizon_drift_slope": 0.9}}


def _report():
    return BenchmarkAdapter().inspect()


def test_worldarena_selection_is_model_data_aware_and_staged() -> None:
    catalog = WorldArenaMetricCatalog.default()
    result = MetricAdvisor(BenchmarkAI()).select_benchmark("improve predictive rollout", _report(), _report()["available_signals"], catalog=catalog)
    assert result["state"] == "validated"
    assert result["primary"] == "rollout_video_l1"
    assert result["protected"] == ["segment_final_mae", "horizon_drift_slope"]
    assert "horizon_drift_slope" in result["evaluation_stages"]["promotion_metrics"]
    assert "horizon_drift_slope" not in result["evaluation_stages"]["pilot_metrics"]
    assert "segment_view_pair_mae" in result["rejected"]


def test_subjective_metric_cannot_be_promoted_and_protected_regression_blocks_positive() -> None:
    catalog = WorldArenaMetricCatalog.default()
    invalid = validate_metric_selection({"primary": "action_conditioning_sensitivity", "protected": [], "diagnostic": []}, catalog, _report(), _report()["available_signals"])
    assert invalid["state"] == "abstain"
    selected = validate_metric_selection({"primary": "rollout_video_l1", "protected": ["segment_final_mae"], "diagnostic": []}, catalog, _report(), _report()["available_signals"])
    result = evaluate_metric_bundle({"baseline_metrics": {"rollout_video_l1": 1.0, "segment_final_mae": 1.0}, "candidate_metrics": {"rollout_video_l1": 0.8, "segment_final_mae": 1.2}}, selected)
    assert result["outcome"] == "harmful"
    assert result["protected_ok"] is False


def test_materialized_metric_needs_references_frozen_split_and_repeatability() -> None:
    definition = WorldArenaMetricCatalog.default().get("rollout_video_l1")
    assert definition is not None
    rejected = validate_metric_materialization(definition, {"contract_tests_passed": True, "evaluator_digest": "sha256:x"})
    assert rejected["state"] == "abstain"
    accepted = validate_metric_materialization(definition, {"contract_tests_passed": True, "reference_alignment_passed": True, "deterministic_repeat_passed": True, "evaluator_digest": "sha256:" + "a" * 64, "frozen_split_digest": "sha256:" + "b" * 64})
    assert accepted["state"] == "validated"


def test_catalogued_long_horizon_metric_needs_a_materialization_receipt() -> None:
    catalog = WorldArenaMetricCatalog.default()
    selected = validate_metric_selection({"primary": "rollout_video_l1", "protected": ["horizon_drift_slope"], "diagnostic": []}, catalog, _report(), _report()["available_signals"])
    raw = {"baseline_metrics": {"rollout_video_l1": 1.0, "horizon_drift_slope": 1.0}, "candidate_metrics": {"rollout_video_l1": 0.8, "horizon_drift_slope": 0.8}}
    assert evaluate_metric_bundle(raw, selected)["reason"] == "metric_evaluator_not_validated"
    raw["metric_receipts"] = {"horizon_drift_slope": {"contract_tests_passed": True, "reference_alignment_passed": True, "deterministic_repeat_passed": True, "evaluator_digest": "sha256:" + "a" * 64, "frozen_split_digest": "sha256:" + "b" * 64}}
    assert evaluate_metric_bundle(raw, selected)["outcome"] == "confirmed_positive"


def test_materializer_rejects_an_unbacked_ai_claim() -> None:
    class UnbackedAgent:
        def run(self, **kwargs):
            return {"state": "completed", "result": {"contract_tests_passed": True, "reference_alignment_passed": True, "deterministic_repeat_passed": True, "evaluator_digest": "sha256:" + "a" * 64, "frozen_split_digest": "sha256:" + "b" * 64}, "events": []}

    definition = WorldArenaMetricCatalog.default().get("horizon_drift_slope")
    assert definition is not None
    result = MetricMaterializer(UnbackedAgent()).materialize(definition, model_report=_report())
    assert result["state"] == "abstain"
    assert "missing_engineering_actions:collect_artifacts,run_tests" in result["errors"]


def test_discovered_catalog_and_plan_are_retained_in_graph(tmp_path: Path) -> None:
    discovery = MetricCatalogDiscovery(BenchmarkAI()).discover([{"url": "https://official.example/worldarena", "title": "WorldArena", "content_digest": "sha256:" + "c" * 64, "text": "official benchmark metric"}])
    assert discovery["state"] == "discovered"
    system = ResearchSystem(RuntimeBindings(BenchmarkAdapter(), ai=BenchmarkAI(), state_root=tmp_path))
    report = system.run_cycle(objective="improve predictive rollout", constraints=["safety"])
    assert report["metrics"]["selection_state"] == "validated"
    assert report["metrics"]["primary"] == "rollout_video_l1"
    nodes = system.state.graph_nodes(kind="metric_plan")
    assert len(nodes) == 1
