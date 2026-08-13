import json
from pathlib import Path

from wmloop.experiments.evidence_graph import (
    build_evidence_graph,
    query_evidence_graph,
    write_evidence_graph,
)


def test_graph_projects_settled_transfer_and_provenance(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "artifact_type": "verdiwm-experiment-stage-receipt",
                "trial_id": "trial-1",
                "experiment_id": "exp-1",
                "target_backbone": "ctrl-world",
                "scenario": "cloth",
                "primitive": "next_forcing",
                "probe_id": "probe-1",
                "settlement_state": "settled",
                "outcome": "positive",
                "certificate_status": "licensed",
                "evidence_refs": ["cas://abc"],
            }
        ),
        encoding="utf-8",
    )
    first = build_evidence_graph(root)
    second = build_evidence_graph(root)
    assert first == second
    assert first["node_count"] >= 8
    assert first["edge_count"] >= 8
    assert any(node["kind"] == "verified_evidence" for node in first["nodes"])
    assert any(edge["relation"] == "licenses_transfer" for edge in first["edges"])
    report = write_evidence_graph(input_root=root, output_root=tmp_path / "graph")
    assert report["state"] == "ready"
    assert (tmp_path / "graph" / "graph.json").is_file()
    assert (tmp_path / "graph" / "manifest.json").is_file()
    queried = query_evidence_graph(tmp_path / "graph", entity="nodes", filters={"kind": "probe"})
    assert queried["total"] == 1
