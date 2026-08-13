import json
import sqlite3
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
                "stage": "confirm",
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
    assert (tmp_path / "graph" / "graph.db").is_file()
    queried = query_evidence_graph(tmp_path / "graph", entity="nodes", filters={"kind": "probe"})
    assert queried["total"] == 1
    assert queried["query_backend"] == "sqlite"


def test_ready_bundle_is_not_promoted_to_verified_evidence(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "bundle.json").write_text(
        json.dumps({"artifact_type": "example", "state": "ready"}), encoding="utf-8"
    )
    graph = build_evidence_graph(root)
    assert not any(node["kind"] == "verified_evidence" for node in graph["nodes"])


def test_graph_projects_archive_settled_trials(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    archive = tmp_path / "archive.db"
    connection = sqlite3.connect(archive)
    connection.execute(
        "CREATE TABLE trials (trial_id TEXT, proposal_id TEXT, goal_id TEXT, library_version TEXT, failure_context_ref TEXT, verdict_ref TEXT, receipt_ref TEXT, settlement_json TEXT)"
    )
    connection.execute(
        "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "trial-archive",
            "proposal-archive",
            "goal-archive",
            "v1",
            "cas://failure",
            "cas://verdict",
            "cas://receipt",
            json.dumps({"state": "settled", "receipt_hash": "a" * 64}),
        ),
    )
    connection.commit()
    connection.close()

    graph = build_evidence_graph(root, archive_db=archive)

    assert graph["source_count"] == 1
    assert any(node["kind"] == "trial" and node["key"] == "trial-archive" for node in graph["nodes"])
    assert any(node["kind"] == "verified_evidence" for node in graph["nodes"])
