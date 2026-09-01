from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.workbench import WorkbenchServer
from wmloop.experiments.evidence_graph import build_evidence_graph, load_evidence_graph


class ReadableGraphTests(unittest.TestCase):
    def test_model_family_is_primary_label_and_hash_is_technical(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {
                "artifact_type": "verdiwm-model-run",
                "record_id": "ctrl-world-screen-v1",
                "model_ref": "cas://sha256/" + "ab" * 32,
                "model_family": "ctrl-world",
                "state": "settled",
                "claim_boundary": "screen only",
            }
            (root / "run.json").write_text(json.dumps(payload), encoding="utf-8")
            graph = build_evidence_graph(root)

        model = next(node for node in graph["nodes"] if node["kind"] == "model")
        self.assertEqual(model["display_label"], "ctrl-world")
        self.assertEqual(model["display_kind"], "模型")
        self.assertEqual(model["ui_tier"], "primary")
        self.assertTrue(model["key"].startswith("cas://sha256/"))

    def test_artifact_label_does_not_include_hash_or_source_path(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {
                "artifact_type": "verdiwm-eval-horizon-profile",
                "record_id": "ctrl-world-horizon-v1",
                "model_family": "ctrl-world",
                "state": "settled",
                "claim_boundary": "horizon profile only",
            }
            (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            graph = build_evidence_graph(root)

        artifact = next(node for node in graph["nodes"] if node["kind"] == "artifact")
        self.assertIn("Eval horizon profile", artifact["display_label"])
        self.assertNotIn(str(root), artifact["display_label"])
        self.assertEqual(artifact["ui_tier"], "technical")

    def test_projection_artifact_is_not_reingested(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = {
                "artifact_type": "verdiwm-model-run",
                "record_id": "ctrl-world-screen-v1",
                "model_family": "ctrl-world",
                "state": "settled",
                "claim_boundary": "screen only",
            }
            (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
            projection = {
                "artifact_type": "verdiwm-evidence-graph",
                "nodes": [{"kind": "model", "key": "stale"}],
            }
            (root / "graph.json").write_text(json.dumps(projection), encoding="utf-8")
            graph = build_evidence_graph(root)

        self.assertEqual(graph["source_count"], 1)
        self.assertEqual(sum(node["kind"] == "backbone" for node in graph["nodes"]), 1)
        self.assertNotIn("stale", {node["key"] for node in graph["nodes"]})

    def test_existing_materialized_graph_receives_readable_labels(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            graph = {
                "artifact_type": "verdiwm-evidence-graph",
                "source_count": 4,
                "nodes": [
                    {
                        "id": "model:hash",
                        "kind": "model",
                        "key": "cas://sha256/" + "ab" * 32,
                        "family": "ctrl-world",
                    },
                    {"id": "assessment:hash", "kind": "source_assessment", "key": "ab" * 32},
                ],
                "edges": [],
            }
            (root / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
            loaded = load_evidence_graph(root)

        assert loaded is not None
        self.assertEqual(loaded["presentation_source"], "materialized_graph")
        self.assertEqual(loaded["nodes"][0]["display_label"], "ctrl-world")
        self.assertEqual(loaded["nodes"][1]["display_label"], "来源评估 #1")


class WorkbenchEvidenceRootTests(unittest.TestCase):
    def test_workbench_defaults_graph_input_to_state_root(self) -> None:
        with TemporaryDirectory() as temp:
            state_root = Path(temp)
            server = WorkbenchServer(("127.0.0.1", 0), state_root=state_root)
            try:
                self.assertEqual(server.evidence_root, state_root.resolve())
                self.assertEqual(server.store.root, (state_root / "campaigns").resolve())
            finally:
                server.server_close()

    def test_workbench_accepts_separate_evidence_root(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            state_root = root / "state"
            evidence_root = root / "evidence"
            evidence_root.mkdir()
            server = WorkbenchServer(
                ("127.0.0.1", 0),
                state_root=state_root,
                evidence_root=evidence_root,
            )
            try:
                self.assertEqual(server.evidence_root, evidence_root.resolve())
            finally:
                server.server_close()


if __name__ == "__main__":
    unittest.main()
