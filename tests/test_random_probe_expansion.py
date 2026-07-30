from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.random_probe_expansion import (
    deterministic_path_order,
    filter_projection_rows,
    run_random_probe_expansion,
)


class RandomProbeExpansionTests(unittest.TestCase):
    def test_hash_order_is_deterministic_and_nested(self) -> None:
        paths = ["a", "b", "c", "d"]
        first = deterministic_path_order(paths, 17)
        self.assertEqual(first, deterministic_path_order(paths, 17))
        self.assertEqual(set(first), set(paths))
        self.assertEqual(first[:2], first[:3][:2])

    def test_filter_changes_only_irg_features(self) -> None:
        rows = [
            {"environment": "a", "selector": "static_probe", "feature_names": ["static"], "features": [9.0]},
            {
                "environment": "a",
                "selector": "irg",
                "feature_names": ["response_coordinate:0:path_a", "locality:path_a", "response_coordinate:0:path_b"],
                "features": [1.0, 2.0, 3.0],
            },
        ]
        filtered = filter_projection_rows(rows, ["path_b"])
        self.assertEqual(filtered[0], rows[0])
        self.assertEqual(filtered[1]["feature_names"], ["response_coordinate:0:path_b"])
        self.assertEqual(filtered[1]["features"], [3.0])

    def test_runs_preregistered_random_condition_without_static_substitution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            projection = root / "projections.jsonl"
            rows = []
            for environment, offset in (("a", 0.0), ("b", 1.0)):
                for selector in ("environment_label", "static_probe", "raw_response"):
                    rows.append(
                        {
                            "environment": environment,
                            "selector": selector,
                            "feature_names": [selector],
                            "features": [offset],
                        }
                    )
                rows.append(
                    {
                        "environment": environment,
                        "selector": "irg",
                        "feature_names": ["response_coordinate:0:path_a", "response_coordinate:0:path_b"],
                        "features": [offset, 1.0 - offset],
                    }
                )
            projection.write_text("".join(json.dumps(row) + "\n" for row in rows))
            trials = []
            sequence = 0
            for target, source in (("a", "b"), ("b", "a")):
                for selector in ("environment_label", "static_probe", "raw_response", "irg"):
                    sequence += 1
                    trials.append(
                        {
                            "trial_id": f"{target}-{selector}",
                            "fold_id": f"holdout-{target}",
                            "target_environment": target,
                            "source_environments": [source],
                            "selector": selector,
                            "seed": 1,
                            "sequence_index": sequence,
                        }
                    )
            plan = root / "plan.json"
            plan.write_text(json.dumps({"trials": trials}))
            labels = []
            for environment in ("a", "b"):
                for primitive, positive in (("p1", environment == "a"), ("p2", environment == "b")):
                    labels.append(
                        {
                            "environment": environment,
                            "primitive": primitive,
                            "positive": positive,
                            "settled": True,
                            "delta_candidate_minus_baseline": {"psnr": 1.0 if positive else -1.0},
                        }
                    )
            label_path = root / "labels.json"
            label_path.write_text(json.dumps({"labels": labels}))
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-random-probe-expansion-preregistration",
                        "experiment_id": "test",
                        "study_id": "S4_probe_information_and_collision",
                        "selector": "irg",
                        "source_projection_path": str(projection),
                        "selector_plan_path": str(plan),
                        "effect_label_index_path": str(label_path),
                        "randomization_algorithm": "sha256_seed_path_ascending_nested_prefix",
                        "path_pool": ["path_a", "path_b"],
                        "subset_sizes": [1, 2],
                        "randomization_seeds": [11, 22],
                    }
                )
            )
            run_random_probe_expansion(config_path=config, output_root=root / "output")
            report = json.loads((root / "output/random-probe-expansion.json").read_text())
            self.assertEqual(report["execution_count"], 4)
            self.assertTrue(report["cpu_only"])
            metrics = (root / "output/tables/selector-metrics.csv").read_text()
            self.assertIn("irg_random_subset", metrics)
            self.assertNotIn("static_probe", metrics)


if __name__ == "__main__":
    unittest.main()
