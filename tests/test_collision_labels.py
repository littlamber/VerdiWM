from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.collision_labels import evaluate_collision_labels


class CollisionLabelTests(unittest.TestCase):
    def test_labels_are_frozen_from_fixed_errors_and_evolved_abstention_alerts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self._replay(
                root / "fixed",
                candidates={"a": ("p", "negative"), "b": ("p", "negative")},
                states={"a": ("evaluated", ""), "b": ("evaluated", "")},
            )
            evolved = self._replay(
                root / "evolved",
                candidates={"a": ("p", "positive"), "b": ("p", "negative")},
                states={"a": ("abstained", "certificate_failed"), "b": ("evaluated", "")},
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            {"environment": "a", "primitive": "p", "positive": True, "settled": True},
                            {"environment": "b", "primitive": "p", "positive": False, "settled": True},
                        ]
                    }
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-collision-label-preregistration",
                        "study_id": "S4_probe_information_and_collision",
                        "selector": "irg",
                        "fixed_replay_root": str(fixed),
                        "evolved_replay_root": str(evolved),
                        "effect_label_index_path": str(labels),
                        "evolved_replay_selection_rule": "latest_completed_iteration_not_best_metric",
                    }
                )
            )
            evaluate_collision_labels(config_path=config, output_root=root / "output")
            report = json.loads((root / "output/collision-label-evaluation.json").read_text())
            self.assertEqual(report["positive_collision_count"], 1)
            self.assertEqual(report["negative_collision_count"], 1)
            self.assertEqual(report["collision_detection_f1"], 1.0)
            self.assertEqual(report["accepted_coverage"], 0.5)
            self.assertEqual(report["post_evolution_collision_rate"], 0.0)

    def _replay(
        self,
        root: Path,
        *,
        candidates: dict[str, tuple[str, str]],
        states: dict[str, tuple[str, str]],
    ) -> Path:
        (root / "tables").mkdir(parents=True)
        with (root / "tables/candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["target_environment", "selector", "seed", "rank", "primitive", "predicted_sign"],
            )
            writer.writeheader()
            for target, (primitive, predicted) in candidates.items():
                for seed in (101, 202, 303):
                    writer.writerow(
                        {
                            "target_environment": target,
                            "selector": "irg",
                            "seed": seed,
                            "rank": 1,
                            "primitive": primitive,
                            "predicted_sign": predicted,
                        }
                    )
        with (root / "tables/cells.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["target_environment", "selector", "seed", "state", "abstention_reason"],
            )
            writer.writeheader()
            for target, (state, reason) in states.items():
                for seed in (101, 202, 303):
                    writer.writerow(
                        {
                            "target_environment": target,
                            "selector": "irg",
                            "seed": seed,
                            "state": state,
                            "abstention_reason": reason,
                        }
                    )
        return root


if __name__ == "__main__":
    unittest.main()
