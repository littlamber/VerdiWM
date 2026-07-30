from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.probe_information import export_probe_information_study


class ProbeInformationTests(unittest.TestCase):
    def test_keeps_pending_random_and_missing_collision_truth_explicit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = root / "replay"
            (replay / "tables").mkdir(parents=True)
            (replay / "manifest.json").write_text(json.dumps({"state": "partial", "evaluated_cell_count": 2, "abstained_cell_count": 1}))
            (replay / "tables/selector-metrics.csv").write_text(
                "selector,top1_positive_hit,benefit_sign_accuracy,ranking_kendall_tau,selection_regret,negative_selection\n"
                "irg,1,0.5,0.2,0.1,0\n"
            )
            collision = root / "collision"
            collision.mkdir()
            (collision / "probe-smoke-redundancy.json").write_text(
                json.dumps(
                    {
                        "reference_probe_id": "a",
                        "candidate_probe_id": "b",
                        "comparisons": [
                            {
                                "environment": "push_rope",
                                "cosine_similarity": 0.1,
                                "relative_l2": 1.0,
                                "candidate_locality_residual": 0.2,
                                "candidate_locality_pass": True,
                                "redundant": False,
                            }
                        ],
                    }
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "study_id": "S4_probe_information_and_collision",
                        "conditions": {
                            "fixed": {"selector_replay_report": str(replay)},
                            "random": {"status": "pending"},
                        },
                        "collision_redundancy_reports": [str(collision)],
                        "collision_ground_truth_available": False,
                    }
                )
            )
            manifest = export_probe_information_study(config_path=config, output_root=root / "output")

            self.assertEqual(manifest["state"], "partial")
            report = json.loads((root / "output/probe-information-and-collision.json").read_text())
            self.assertEqual(report["pending_condition_count"], 1)
            self.assertIsNone(report["collision"]["collision_detection_f1"])
            self.assertTrue((root / "output/tables/paper-summary.tex").is_file())

    def test_uses_independent_collision_labels_but_keeps_zero_coverage_blocker(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = root / "replay"
            (replay / "tables").mkdir(parents=True)
            (replay / "tables/selector-metrics.csv").write_text(
                "selector,benefit_sign_accuracy,selection_regret,probe_gpu_hours,gram_condition_number\n"
                "irg_random_subset,0.6,0.1,0.0,12.0\n"
            )
            (replay / "manifest.json").write_text(
                json.dumps({"state": "ready", "evaluated_cell_count": 24, "abstained_cell_count": 0})
            )
            labels = root / "labels"
            labels.mkdir()
            (labels / "collision-label-evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-collision-label-evaluation",
                        "state": "ready",
                        "positive_collision_count": 1,
                        "negative_collision_count": 1,
                        "collision_detection_f1": 0.8,
                        "post_evolution_collision_rate": None,
                        "accepted_case_count": 0,
                        "accepted_coverage": 0.0,
                        "pre_certificate_collision_rate": 0.5,
                        "cases": [
                            {"target_environment": "a", "ground_truth_collision": True},
                            {"target_environment": "b", "ground_truth_collision": False},
                        ],
                    }
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "study_id": "S4_probe_information_and_collision",
                        "conditions": {
                            "fixed": {"selector_replay_report": str(replay)},
                            "random": {"selector_replay_report": str(replay)},
                            "evolved": {"selector_replay_report": str(replay)},
                        },
                        "collision_redundancy_reports": [],
                        "collision_label_report": str(labels),
                    }
                )
            )
            export_probe_information_study(config_path=config, output_root=root / "output")
            report = json.loads((root / "output/probe-information-and-collision.json").read_text())
            self.assertEqual(report["observed_condition_count"], 3)
            self.assertEqual(report["collision"]["collision_detection_f1"], 0.8)
            self.assertEqual(report["collision"]["labeled_case_count"], 2)
            self.assertIsNone(report["collision"]["post_evolution_collision_rate"])
            self.assertEqual(report["state"], "partial")


if __name__ == "__main__":
    unittest.main()
