from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from wmloop.control.intermediate_ir import build_model_capability_ir
from wmloop.contracts import validate_document
from wmloop.experiments.community_export import (
    CommunityExportError,
    export_community_knowledge,
)
from wmloop.experiments.portable_knowledge_graph import (
    audit_portable_knowledge_graph,
    build_portable_knowledge_graph,
)


class CommunityExportTests(unittest.TestCase):
    def _document(self) -> dict[str, object]:
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-model-onboarding-report",
            "repo_name": "fixture",
            "source_revision": {"kind": "source_tree_sha256", "revision": "a" * 64},
            "capabilities": [
                {"capability": "inference", "state": "discovered", "evidence": ["fixture"]}
            ],
            "connector": {
                "entrypoints_by_kind": {"inference": ["infer"]},
                "asset_bindings": [],
            },
            "evaluator_contract": {
                "state": "ready",
                "evaluator_id": "fixture-evaluator",
                "contract_sha256": "b" * 64,
                "verifier": "fixture-verifier",
            },
        }
        return build_model_capability_ir(report, model_family="fixture")

    def _execution(self, root: Path) -> Path:
        body = {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-batch-execution",
            "state": "completed",
            "batch_id": "fixture-batch",
            "plan_sha256": "c" * 64,
            "shared_budget": {
                "path": str(root / "private-budget.db"),
                "total_gpu_hours": 2.0,
            },
            "campaigns_root": str(root / "private-campaigns"),
            "rows": [
                {
                    "model_id": "fixture-model",
                    "state": "completed",
                    "campaign_id": "fixture-campaign",
                }
            ],
            "summary": {"model_count": 1, "campaign_count": 1, "blocked_count": 0},
            "dispatcher": None,
            "claim_boundary": "This manifest records campaign orchestration and execution status only.",
        }
        canonical = json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        body["execution_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = root / "execution.json"
        path.write_text(
            json.dumps(body, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def test_discovers_runtime_json_and_stages_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            (source / "runtime.json").write_text(
                json.dumps({"artifact_type": "verdiwm-model-batch-execution", "state": "queued"}),
                encoding="utf-8",
            )
            document = self._document()
            (source / "capability.json").write_text(json.dumps(document), encoding="utf-8")

            report = export_community_knowledge(
                source_roots=[source], output_root=root / "export"
            )

            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["ignored_reason_counts"], {"unsupported_artifact_type": 1})
            self.assertTrue((root / "export" / "export.json").is_file())
            records = list((root / "export" / "records").glob("*.json"))
            self.assertEqual(len(records), 1)
            loaded_record = json.loads(records[0].read_text(encoding="utf-8"))
            graph = json.loads((root / "export" / "graph.json").read_text(encoding="utf-8"))
            audit = json.loads((root / "export" / "quality-audit.json").read_text(encoding="utf-8"))
            self.assertEqual(graph, build_portable_knowledge_graph([loaded_record]))
            self.assertEqual(
                audit,
                audit_portable_knowledge_graph(documents=[loaded_record], graph=graph),
            )
            validate_document(
                "community_export",
                json.loads((root / "export" / "export.json").read_text(encoding="utf-8")),
            )
            digest = hashlib.sha256(
                json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            self.assertEqual(records[0].name, f"{digest}.json")

    def test_json_array_is_split_and_duplicates_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            document = self._document()
            (source / "array.json").write_text(
                json.dumps([document, document]), encoding="utf-8"
            )
            report = export_community_knowledge(
                source_roots=[source], output_root=root / "export"
            )
            self.assertEqual(report["candidate_document_count"], 2)
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["artifact_type_counts"], {"verdiwm-model-capability-ir": 2})

    def test_invalid_recognized_document_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            invalid = self._document()
            invalid["evaluator"]["verifier"] = "/private/verifier.py"  # type: ignore[index]
            (source / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(CommunityExportError, "COMMUNITY_EXPORT_DOCUMENT_INVALID"):
                export_community_knowledge(source_roots=[source], output_root=root / "export")

    def test_export_is_idempotent_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            (source / "capability.json").write_text(json.dumps(self._document()), encoding="utf-8")
            output = root / "export"
            first = export_community_knowledge(source_roots=[source], output_root=output)
            second = export_community_knowledge(source_roots=[source], output_root=output)
            self.assertEqual(first["export_id"], second["export_id"])
            graph = output / "graph.json"
            graph.write_text(graph.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CommunityExportError, "COMMUNITY_EXPORT_OUTPUT_CONFLICT"):
                export_community_knowledge(source_roots=[source], output_root=output)

    def test_export_rejects_symlink_or_extra_directory_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            (source / "capability.json").write_text(json.dumps(self._document()), encoding="utf-8")
            output = root / "export"
            export_community_knowledge(source_roots=[source], output_root=output)
            (output / "extra").mkdir()
            with self.assertRaisesRegex(CommunityExportError, "COMMUNITY_EXPORT_OUTPUT_CONFLICT"):
                export_community_knowledge(source_roots=[source], output_root=output)

    def test_execution_binding_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            (source / "capability.json").write_text(json.dumps(self._document()), encoding="utf-8")
            execution = self._execution(root)
            output = root / "export"
            report = export_community_knowledge(
                source_roots=[source], output_root=output, execution_manifest=execution
            )
            self.assertEqual(report["source_execution"]["batch_id"], "fixture-batch")  # type: ignore[index]
            export_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertNotIn(str(execution), export_text)
            self.assertNotIn(str(root / "private-budget.db"), export_text)

    def test_file_limit_fails_before_unbounded_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts"
            source.mkdir()
            (source / "a.json").write_text(json.dumps({"runtime": True}), encoding="utf-8")
            (source / "b.json").write_text(json.dumps({"runtime": True}), encoding="utf-8")
            with self.assertRaisesRegex(CommunityExportError, "COMMUNITY_EXPORT_FILE_LIMIT_EXCEEDED"):
                export_community_knowledge(
                    source_roots=[source], output_root=root / "export", max_files=1
                )


if __name__ == "__main__":
    unittest.main()
