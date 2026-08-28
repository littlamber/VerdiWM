import json
from pathlib import Path

from verdi_core.ideas import diagnostic_gap_report
from verdi_core.knowledge_graph import export_bundle, import_settlement_entries, project_model_portrait
from verdi_core.storage import SQLiteState
from verdi_core.transfer import rank_transfer_candidates


def test_layered_graph_import_and_portable_export(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "knowledge.sqlite3")
    import_settlement_entries(
        state,
        [{
            "idea_id": "deep-ensemble-mean",
            "status": "confirmed_positive",
            "claim": "seed averaging improves replay",
            "replications": 2,
            "evidence_paths": ["/private/one/evaluation.json", "/private/two/evaluation.json"],
            "diagnostic_dimensions": ["sampling_robustness"],
            "architecture_facets": ["unet", "diffusion"],
        }],
    )
    project_model_portrait(state, model_id="dit-v1", revision="r1", architecture_facets=["dit", "diffusion"], probe_results=[])
    graph = state.graph_document(portable=True)
    assert graph["node_count"] == 7  # two models, method, two evidence nodes, architecture, portrait
    assert all("/private" not in json.dumps(node) for node in graph["nodes"])
    bundle = export_bundle(state, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "knowledge.sqlite3").exists()
    assert (tmp_path / "bundle" / "graph.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert len(bundle["graph_sha256"]) == 64
    exported = json.loads((tmp_path / "bundle" / "graph.json").read_text(encoding="utf-8"))
    method = next(node for node in exported["nodes"] if node["kind"] == "method")
    assert method["display_name"] == "Deep ensemble mean"


def test_transfer_ranking_prefers_diagnostic_overlap_and_persists(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "knowledge.sqlite3")
    import_settlement_entries(state, [{
        "idea_id": "method-a", "status": "confirmed_positive", "replications": 2,
        "diagnostic_dimensions": ["history_dependence"], "architecture_facets": ["unet"],
        "required_capabilities": ["rollout"],
    }, {"idea_id": "method-b", "status": "null", "diagnostic_dimensions": ["sampling_robustness"]}])
    result = rank_transfer_candidates(state, target_model_id="dit-v1", architecture_facets=["dit"], diagnostic_dimensions=["history_dependence"], capabilities=["rollout"])
    assert result[0]["source_method_key"] == "method-a"
    assert result[0]["state"] == "candidate"
    assert state.count("transfer_assessments") == 2


def test_diagnostic_gap_report_distinguishes_coverage_from_probe_measurement(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "knowledge.sqlite3")
    project_model_portrait(
        state,
        model_id="world-model",
        revision="r1",
        probe_results=[{"probe_id": "noise", "status": "admitted", "dimensions": ["sampling_robustness"]}],
    )
    import_settlement_entries(state, [{
        "idea_id": "noise-training", "title": "Noise training", "status": "null",
        "diagnostic_dimensions": ["sampling_robustness"],
    }], model_id="world-model")
    report = diagnostic_gap_report(state.graph_document())
    assert report["coverage"]["sampling_robustness"]["state"] == "attempted_without_positive"
    assert report["compatibility_only_probes"] == ["noise"]


def test_legacy_evidence_is_projected_to_readable_semantic_graph(tmp_path: Path) -> None:
    state = SQLiteState(tmp_path / "knowledge.sqlite3")
    state.put_evidence("e-1", {
        "evidence_id": "e-1",
        "experiment_id": "experiments-v17-campaign-final-replay",
        "hypothesis_id": "context-retention-training",
        "model_id": "ctrl-world",
        "outcome": "abstain",
        "delta": 0.0,
        "protected_ok": False,
        "variant_of": "context-retention-training-v2",
        "seed": 42,
        "claim_boundary": "requires domain evaluation",
    })
    state.add_edge("experiments-v17-campaign-final-replay", "produced", "e-1", "e-1")

    graph = state.graph_document(portable=True)
    assert graph["node_count"] >= 6
    assert graph["edge_count"] >= 6
    kinds = {node["kind"] for node in graph["nodes"]}
    assert {"diagnosis", "method", "conclusion", "evidence"}.issubset(kinds)
    method = next(node for node in graph["nodes"] if node["kind"] == "method")
    assert "v17" not in method["display_name"].lower()
    evidence = next(node for node in graph["nodes"] if node["kind"] == "evidence")
    assert "variant_of" not in evidence["payload"]
    assert evidence["payload"]["provenance"]["variant_of"] == "context-retention-training-v2"
