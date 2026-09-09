"""Regression cases for races, crash recovery and complete batch consumption."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wmloop.control.campaign_api import CampaignStore, CampaignAPIError
from wmloop.control.model_batch import compile_model_batch, run_model_batch, summarize_model_batch, ModelBatchError, _digest
from wmloop.control.adapter_profiles import ResolvedAdapter


def _running_update(root, read, release):
    store = CampaignStore(Path(root))
    with store.transaction():
        store.get("race")
        read.set()
        if not release.wait(10):
            raise RuntimeError("test barrier timed out")
        store.record_dispatch_result("race", status="running")


def _cancel(root, entered):
    entered.set()
    CampaignStore(Path(root)).cancel("race")


class CampaignRecoveryTests(unittest.TestCase):
    def test_cross_process_cancel_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = CampaignStore(root)
            store._write(root / "race.json", {"campaign_id": "race", "status": "queued"})
            ctx = multiprocessing.get_context("spawn")
            read, release, entered = ctx.Event(), ctx.Event(), ctx.Event()
            writer = ctx.Process(target=_running_update, args=(str(root), read, release))
            cancel = ctx.Process(target=_cancel, args=(str(root), entered))
            writer.start()
            try:
                self.assertTrue(read.wait(10))
                cancel.start()
                self.assertTrue(entered.wait(10))
                release.set()
                writer.join(10)
                cancel.join(10)
                self.assertEqual(writer.exitcode, 0)
                self.assertEqual(cancel.exitcode, 0)
                self.assertEqual(store.get("race")["status"], "cancelled")
                self.assertEqual(store.record_dispatch_result("race", status="completed")["status"], "cancelled")
            finally:
                release.set()
                for process in (writer, cancel):
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_stale_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = CampaignStore(root), CampaignStore(root)
            a._write(root / "race.json", {"campaign_id": "race", "status": "queued"})
            stale = a.get("race")
            b.cancel("race")
            stale["status"] = "running"
            with self.assertRaisesRegex(CampaignAPIError, "VERSION_CONFLICT"):
                a._write(root / "race.json", stale)

    def test_committed_outbox_recovers_after_projection_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = CampaignStore(root)
            with patch("wmloop.control.campaign_repository.atomic_write", side_effect=OSError("disk temporarily unavailable")):
                with self.assertRaises(OSError):
                    store._write(root / "race.json", {"campaign_id": "race", "status": "queued"})
            self.assertFalse((root / "race.json").exists())
            recovered = CampaignStore(root)
            self.assertEqual(recovered.get("race")["status"], "queued")
            self.assertEqual(json.loads((root / "race.json").read_text())["status"], "queued")

    def test_rollback_publishes_neither_state_nor_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = CampaignStore(root)
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with store.transaction():
                    store._write(root / "race.json", {"campaign_id": "race", "status": "queued"})
                    store._write(root / "dispatch/pending/race.json", {"campaign_id": "race"})
                    raise RuntimeError("abort")
            with self.assertRaisesRegex(CampaignAPIError, "NOT_FOUND"):
                store.get("race")
            self.assertFalse((root / "dispatch/pending/race.json").exists())

    def test_legacy_migrates_on_mutation_and_read_only_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "legacy.json"
            path.write_text(json.dumps({"campaign_id": "legacy", "status": "queued"}))
            reader = CampaignStore(root, read_only=True)
            self.assertEqual(reader.get("legacy")["status"], "queued")
            self.assertFalse((root / "campaigns.sqlite3").exists())
            store = CampaignStore(root)
            store.cancel("legacy")
            path.write_text('{"status":"running"}')
            self.assertEqual(reader.get("legacy")["status"], "cancelled")


class BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batch = self.root / "batch"
        self.evaluator = self.root / "evaluator.json"
        self.evaluator.write_text('{"metrics":["score"]}')
        self.request = self.root / "request.json"
        (self.root / "data").mkdir()
        models = []
        for name in ("model-one", "model-two", "model-three"):
            model = self.root / name
            model.mkdir()
            models.append({"model_id": name, "model": str(model), "data": str(self.root / "data"), "budget": 1, "evaluator_contract": str(self.evaluator)})
        self.request.write_text(json.dumps({"schema_version": 1, "artifact_type": "verdiwm-model-batch-request", "batch_id": "test-batch", "goal": "improve score", "models": models}))
        compile_model_batch(request_path=self.request, output_root=self.batch)
        adapter = patch("wmloop.control.adapter_profiles.compile_adapter_execution", side_effect=self.compile)
        adapter.start()
        self.addCleanup(adapter.stop)
        runner = patch("wmloop.control.campaign_dispatcher._run_subprocess", return_value={"outcome": "completed"})
        self.runner = runner.start()
        self.addCleanup(runner.stop)

    def compile(self, **kw):
        return ResolvedAdapter(profile_id="test", model_family="test", capability_level="smoke", constitution_freeze=str(self.evaluator), execution={"kind": "pipeline", "repo_root": str(kw["model"]), "output_root": str(self.batch / "runs" / kw["campaign_id"]), "evaluator_contract": str(self.evaluator), "runtime_python": "/usr/bin/python3", "asset_bindings": {"--checkpoint": str(kw["model"])}, "budget_total_gpu_hours": float(kw["budget"])})

    def run_batch(self, **kw):
        return run_model_batch(plan_path=self.batch / "plan.json", output_root=self.batch, **kw)

    def test_drains_all_models_and_repeated_run_reuses_terminal_results(self):
        result = self.run_batch(max_parallel=2)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.runner.call_count, 3)
        self.assertEqual(result["dispatcher"]["pending_count"], 0)
        repeated = self.run_batch()
        self.assertEqual(repeated["state"], "completed")
        self.assertEqual(self.runner.call_count, 3)
        self.assertTrue(all(not r["blockers"] for r in repeated["rows"]))

    def test_destination_conflict_is_rejected_before_any_campaign(self):
        self.run_batch(queue_only=True)
        before = list((self.batch / "campaigns").glob("*.json"))
        request = json.loads(self.request.read_text())
        request["batch_id"] = "other-batch"
        self.request.write_text(json.dumps(request))
        other = self.root / "other"
        compile_model_batch(request_path=self.request, output_root=other)
        with self.assertRaisesRegex(ModelBatchError, "PLAN_CONFLICT"):
            run_model_batch(plan_path=other / "plan.json", output_root=self.batch)
        self.assertEqual(list((self.batch / "campaigns").glob("*.json")), before)
        self.runner.assert_not_called()

    def test_running_and_cancelled_states_survive_reconciliation(self):
        queued = self.run_batch(queue_only=True)
        store = CampaignStore(Path(queued["campaigns_root"]))
        store.record_dispatch_result(queued["rows"][0]["campaign_id"], status="running")
        store.cancel(queued["rows"][1]["campaign_id"])
        result = self.run_batch(queue_only=True)
        self.assertEqual(result["state"], "running")
        self.assertEqual([r["state"] for r in result["rows"]], ["running", "cancelled", "queued"])

    def test_corrupt_authoritative_database_reports_unavailable(self):
        result = self.run_batch()
        database = Path(result["campaigns_root"]) / "campaigns.sqlite3"
        database.write_bytes(b"corrupt fixture")
        report = summarize_model_batch(self.batch / "execution.json")
        self.assertEqual(report["state"], "unavailable")
        self.assertTrue(all(r["last_known_state"] == "completed" for r in report["rows"]))

    def test_missing_legacy_state_reports_unavailable_without_creating_database(self):
        result = self.run_batch()
        missing = self.root / "absent"
        result["campaigns_root"] = str(missing)
        result.pop("execution_sha256")
        result["execution_sha256"] = _digest(result)
        (self.batch / "execution.json").write_text(json.dumps(result))
        report = summarize_model_batch(self.batch / "execution.json")
        self.assertEqual(report["state"], "unavailable")
        self.assertFalse(missing.exists())

    def test_deleted_database_does_not_fall_back_to_json_exports(self):
        result = self.run_batch()
        root = Path(result["campaigns_root"])
        (root / "campaigns.sqlite3").unlink()
        self.assertEqual(summarize_model_batch(self.batch / "execution.json")["state"], "unavailable")
        with self.assertRaisesRegex(CampaignAPIError, "STORE_UNAVAILABLE"):
            CampaignStore(root)

    def test_live_status_exists_before_first_worker_and_cancel_is_terminal(self):
        def worker(execution, **kwargs):
            report = summarize_model_batch(self.batch / "execution.json")
            self.assertEqual(report["state"], "running")
            active = next(row for row in report["rows"] if row["state"] == "running")
            CampaignStore(Path(report["campaigns_root"])).cancel(active["campaign_id"])
            return {"outcome": "completed"}
        self.runner.side_effect = worker
        result = self.run_batch()
        self.assertTrue(all(row["state"] == "cancelled" for row in result["rows"]))
        for row in result["rows"]:
            name = row["campaign_id"] + ".json"
            root = Path(result["campaigns_root"]) / "dispatch"
            self.assertTrue((root / "cancelled" / name).is_file())
            self.assertFalse((root / "completed" / name).exists())
            self.assertFalse((root / "running" / name).exists())
