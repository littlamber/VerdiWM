from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wmloop.control.adapter_profiles import ResolvedAdapter
from wmloop.control.model_batch import (
    ModelBatchError,
    compile_model_batch,
    load_model_batch_plan,
    run_model_batch,
    summarize_model_batch,
)


class ModelBatchTests(unittest.TestCase):
    def _request(self, root: Path, *, duplicate: bool = False) -> Path:
        model_a = root / "model-a"
        model_b = root / "model-b"
        model_a.mkdir()
        model_b.mkdir()
        data = root / "data"
        data.mkdir()
        evaluator = root / "evaluator.json"
        evaluator.write_text(json.dumps({"metrics": ["score"]}), encoding="utf-8")
        rows = [
            {"model_id": "model-a", "model": str(model_a), "data": str(data), "evaluator_contract": str(evaluator), "budget": "0.5gpu-hour"},
            {"model_id": "model-a" if duplicate else "model-b", "model": str(model_b), "data": str(data), "evaluator_contract": str(evaluator), "budget": 1},
        ]
        path = root / "request.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "artifact_type": "verdiwm-model-batch-request",
            "batch_id": "batch-test-v1",
            "goal": "improve long horizon prediction",
            "models": rows,
        }), encoding="utf-8")
        return path

    def test_compile_model_batch_is_durable_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            output = root / "batch"
            plan = compile_model_batch(request_path=request, output_root=output)
            self.assertEqual(plan["state"], "ready_for_dispatch")
            self.assertEqual(plan["budget"]["declared_gpu_hours"], 1.5)
            loaded = load_model_batch_plan(output / "plan.json")
            self.assertEqual(loaded["plan_sha256"], plan["plan_sha256"])
            again = compile_model_batch(request_path=request, output_root=output)
            self.assertEqual(again["plan_sha256"], plan["plan_sha256"])

    def test_empty_destination_directory_is_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            output = root / "batch"
            output.mkdir()
            plan = compile_model_batch(request_path=request, output_root=output)
            self.assertEqual(plan["state"], "ready_for_dispatch")
            self.assertTrue((output / "plan.json").is_file())

    def test_duplicate_model_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ModelBatchError, "MODEL_BATCH_MODEL_ID_DUPLICATE"):
                compile_model_batch(
                    request_path=self._request(Path(temporary), duplicate=True),
                    output_root=Path(temporary) / "batch",
                )

    def test_missing_optional_binding_blocks_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            payload = json.loads(request.read_text(encoding="utf-8"))
            payload["models"][0]["evaluator_contract"] = str(root / "missing.json")
            request.write_text(json.dumps(payload), encoding="utf-8")
            plan = compile_model_batch(request_path=request, output_root=root / "batch")
            self.assertEqual(plan["state"], "blocked")
            self.assertEqual(plan["models"][0]["state"], "blocked")
            self.assertIn("evaluator_contract_not_found", plan["models"][0]["blockers"])

    def test_bound_file_drift_is_rejected_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            output = root / "batch"
            compile_model_batch(request_path=request, output_root=output)
            (root / "model-a" / "new.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ModelBatchError, "MODEL_BATCH_BINDING_DRIFT:model-a:model"):
                load_model_batch_plan(output / "plan.json")

    def test_run_queue_only_materializes_campaigns_with_shared_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a = root / "model-a"
            model_b = root / "model-b"
            model_a.mkdir()
            model_b.mkdir()
            data = root / "data"
            data.mkdir()
            evaluator = root / "evaluator.json"
            evaluator.write_text(json.dumps({"metrics": ["score"]}), encoding="utf-8")
            request = root / "request.json"
            request.write_text(json.dumps({
                "schema_version": 1,
                "artifact_type": "verdiwm-model-batch-request",
                "batch_id": "batch-run-v1",
                "goal": "improve score",
                "models": [
                    {"model_id": "model-a", "model": str(model_a), "data": str(data), "evaluator_contract": str(evaluator), "budget": 0.5},
                    {"model_id": "model-b", "model": str(model_b), "data": str(data), "evaluator_contract": str(evaluator), "budget": 1},
                ],
            }), encoding="utf-8")
            batch = root / "batch"
            compile_model_batch(request_path=request, output_root=batch)

            def fake_compile(**kwargs):
                campaign_id = kwargs["campaign_id"]
                return ResolvedAdapter(
                    profile_id="fake-profile",
                    model_family="fake",
                    capability_level="smoke",
                    constitution_freeze=str(evaluator),
                    execution={
                        "kind": "pipeline",
                        "repo_root": str(kwargs["model"]),
                        "output_root": str(batch / "runs" / campaign_id),
                        "evaluator_contract": str(evaluator),
                        "runtime_python": str(Path("/usr/bin/python3").resolve()),
                        "asset_bindings": {"--checkpoint": str(kwargs["model"])},
                        "budget_total_gpu_hours": float(kwargs["budget"]),
                    },
                )

            with patch("wmloop.control.adapter_profiles.compile_adapter_execution", side_effect=fake_compile):
                result = run_model_batch(plan_path=batch / "plan.json", queue_only=True, output_root=batch)
            self.assertEqual(result["state"], "queued")
            self.assertEqual(result["summary"]["campaign_count"], 2)
            self.assertEqual(result["shared_budget"]["total_gpu_hours"], 1.5)
            self.assertTrue((batch / "budget.db").exists() is False)
            self.assertTrue((batch / "execution.json").is_file())
            self.assertTrue(all(row["state"] == "queued" for row in result["rows"]))

    def test_run_blocks_row_without_evaluator_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            payload = json.loads(request.read_text(encoding="utf-8"))
            payload["models"][0].pop("evaluator_contract", None)
            request.write_text(json.dumps(payload), encoding="utf-8")
            batch = root / "batch"
            compile_model_batch(request_path=request, output_root=batch)
            result = run_model_batch(plan_path=batch / "plan.json", queue_only=True, output_root=batch)
            self.assertEqual(result["state"], "blocked")
            self.assertEqual(result["summary"]["campaign_count"], 0)
            self.assertIn("evaluator_contract_required", result["rows"][0]["blockers"])

    def test_status_is_read_only_and_refreshes_campaign_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            batch = root / "batch"
            compile_model_batch(request_path=request, output_root=batch)

            def fake_compile(**kwargs):
                campaign_id = kwargs["campaign_id"]
                return ResolvedAdapter(
                    profile_id="fake-profile",
                    model_family="fake",
                    capability_level="smoke",
                    constitution_freeze=str(root / "evaluator.json"),
                    execution={
                        "kind": "pipeline",
                        "repo_root": str(kwargs["model"]),
                        "output_root": str(batch / "runs" / campaign_id),
                        "evaluator_contract": str(root / "evaluator.json"),
                        "runtime_python": str(Path("/usr/bin/python3").resolve()),
                        "asset_bindings": {"--checkpoint": str(kwargs["model"])},
                        "budget_total_gpu_hours": float(kwargs["budget"]),
                    },
                )

            with patch("wmloop.control.adapter_profiles.compile_adapter_execution", side_effect=fake_compile):
                queued = run_model_batch(
                    plan_path=batch / "plan.json", queue_only=True, output_root=batch
                )
            report = summarize_model_batch(batch / "execution.json")
            self.assertEqual(report["batch_id"], queued["batch_id"])
            self.assertEqual(report["state"], "queued")
            self.assertEqual(report["execution_state"], "queued")
            self.assertEqual(report["state_counts"], {"queued": 2})
            self.assertEqual(report["rows"][0]["campaign"]["status"], "queued")


if __name__ == "__main__":
    unittest.main()
