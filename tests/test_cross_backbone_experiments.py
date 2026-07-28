from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.ledger import ExperimentLedgerError, load_settled_receipts
from wmloop.experiments.lobo import build_lobo_plan, run_experiment_plan
from wmloop.experiments.report import build_experiment_report, run_experiment_report
from wmloop.experiments.spec import ExperimentSpecError, load_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "configs" / "experiments" / "three_backbone_lobo_pilot_v1.json"


class CrossBackboneExperimentTests(unittest.TestCase):
    def test_lobo_plan_has_no_target_leak_and_distinct_random_search(self) -> None:
        spec = load_experiment_spec(BASE_SPEC)
        plan = build_lobo_plan(spec)

        self.assertEqual(plan["planned_trial_count"], 108)
        self.assertEqual(plan["planned_stage_task_count"], 324)
        for trial in plan["trials"]:
            self.assertNotIn(trial["target_backbone"], trial["source_backbones"])
            if trial["arm"] == "warm_start":
                self.assertIn(trial["selector"], {"environment_label", "static_probe", "raw_response", "irg"})
                self.assertTrue(trial["source_experience_allowed"])
            elif trial["arm"] == "random_search":
                self.assertEqual(trial["selector"], "none")
                self.assertEqual(trial["candidate_policy"], "uniform_target_compatible_registry_sampling")
                self.assertFalse(trial["source_experience_allowed"])

    def test_lobo_spec_rejects_target_leak(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
            payload["folds"][0]["source_backbones"] = ["acwm_phys", "ctrl_world"]
            path = root / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentSpecError, "LOBO_SOURCE_LEAK_OR_OMISSION"):
                load_experiment_spec(path)

    def test_planner_writes_deterministic_contract_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_experiment_plan(spec_path=BASE_SPEC, output_root=root / "first")
            second = run_experiment_plan(spec_path=BASE_SPEC, output_root=root / "second")

            self.assertFalse(first["launch_ready"])
            self.assertEqual(first["blocker_count"], 1)
            self.assertEqual(
                (root / "first" / "planned-trials.csv").read_bytes(),
                (root / "second" / "planned-trials.csv").read_bytes(),
            )

    def test_unsettled_receipt_is_rejected_and_never_costed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = _ready_spec(root)
            spec = load_experiment_spec(spec_path)
            trial = build_lobo_plan(spec)["trials"][0]
            path = _write_receipt(root, trial, stage="screen", outcome="positive", delta=1.0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["settlement_state"] = "running"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentLedgerError, "RECEIPT_NOT_SETTLED"):
                load_settled_receipts(spec=spec, receipt_paths=[path])

    def test_screen_positive_does_not_count_as_formal_positive(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = load_experiment_spec(_ready_spec(root))
            trial = build_lobo_plan(spec)["trials"][0]
            screen = _write_receipt(root, trial, stage="screen", outcome="positive", delta=1.0)
            receipts = load_settled_receipts(spec=spec, receipt_paths=[screen])
            report = build_experiment_report(spec=spec, receipts=receipts)

            self.assertEqual(report["formal_positive_count"], 0)
            self.assertFalse(report["claim_ready"])
            self.assertTrue(any(item["code"] == "confirm_receipts_missing" for item in report["blockers"]))

    def test_confirm_requires_positive_screen_and_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = load_experiment_spec(_ready_spec(root))
            trial = build_lobo_plan(spec)["trials"][0]
            confirm = _write_receipt(root, trial, stage="confirm", outcome="positive", delta=1.0)

            with self.assertRaisesRegex(ExperimentLedgerError, "CONFIRM_WITHOUT_POSITIVE_GATE"):
                load_settled_receipts(spec=spec, receipt_paths=[confirm])

    def test_nonfinite_gpu_cost_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = load_experiment_spec(_ready_spec(root))
            trial = build_lobo_plan(spec)["trials"][0]
            screen = _write_receipt(root, trial, stage="screen", outcome="positive", delta=1.0)
            payload = json.loads(screen.read_text(encoding="utf-8"))
            payload["gpu_hours"] = float("nan")
            screen.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentLedgerError, "NONFINITE_VALUE"):
                load_settled_receipts(spec=spec, receipt_paths=[screen])

    def test_transfer_metrics_and_costs_use_settled_confirm_receipts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = _ready_spec(root)
            spec = load_experiment_spec(spec_path)
            candidates = [
                trial
                for trial in build_lobo_plan(spec)["trials"]
                if trial["fold_id"] == "holdout_acwm_phys"
                and trial["arm"] == "warm_start"
                and trial["selector"] == "environment_label"
            ][:3]
            receipt_paths: list[Path] = []
            outcomes = [("negative", 0.0), ("positive", 1.0), ("abstain", 0.0)]
            for trial, (outcome, delta) in zip(candidates, outcomes, strict=True):
                receipt_paths.extend(
                    [
                        _write_receipt(root, trial, stage="screen", outcome="positive", delta=1.0),
                        _write_receipt(root, trial, stage="gate", outcome="positive", delta=1.0),
                        _write_receipt(root, trial, stage="confirm", outcome=outcome, delta=delta),
                    ]
                )
            receipts = load_settled_receipts(spec=spec, receipt_paths=receipt_paths)
            report = build_experiment_report(spec=spec, receipts=receipts)
            row = next(
                item
                for item in report["summary_rows"]
                if item["fold_id"] == "holdout_acwm_phys"
                and item["arm"] == "warm_start"
                and item["selector"] == "environment_label"
            )

            self.assertEqual(row["formal_positive_count"], 1)
            self.assertEqual(row["formal_negative_count"], 1)
            self.assertEqual(row["abstain_count"], 1)
            self.assertEqual(row["trials_to_first_positive"], 2)
            self.assertEqual(row["negative_transfer_rate"], 0.5)
            self.assertEqual(row["risk"], 0.5)
            self.assertAlmostEqual(row["abstention_rate"], 1 / 3)
            self.assertEqual(row["total_gpu_hours"], 9.0)

            manifest = run_experiment_report(
                spec_path=spec_path,
                receipt_paths=receipt_paths,
                output_root=root / "report",
            )
            self.assertEqual(manifest["formal_positive_count"], 1)
            self.assertTrue((root / "report" / "tables" / "lobo-summary.tex").is_file())


def _ready_spec(root: Path) -> Path:
    payload = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
    for backbone in payload["backbones"]:
        backbone["status"] = "ready"
    path = root / "spec.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_receipt(
    root: Path,
    trial: dict[str, object],
    *,
    stage: str,
    outcome: str,
    delta: float,
) -> Path:
    certificate = "not_applicable"
    if trial["arm"] == "warm_start":
        certificate = "abstain" if outcome == "abstain" else "licensed"
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-experiment-stage-receipt",
        "experiment_id": "three_backbone_lobo_pilot_v1",
        "fold_id": trial["fold_id"],
        "trial_id": trial["trial_id"],
        "target_backbone": trial["target_backbone"],
        "scenario": trial["scenario"],
        "arm": trial["arm"],
        "selector": trial["selector"],
        "seed": trial["seed"],
        "stage": stage,
        "settlement_state": "settled",
        "outcome": outcome,
        "certificate_status": certificate,
        "metric_name": "normalized_goal_progress",
        "baseline_value": 0.0,
        "candidate_value": delta,
        "delta": delta,
        "threshold": 0.0,
        "gpu_hours": {"screen": 1.0, "gate": 2.0, "confirm": 0.0}[stage],
        "sequence_index": trial["sequence_index"],
        "evidence_refs": [f"cas://sha256/{'0' * 64}"],
    }
    path = root / "receipts" / str(trial["trial_id"]) / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
