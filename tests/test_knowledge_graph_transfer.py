import json
from pathlib import Path

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
