from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.selector_replay import run_selector_replay


class ACWMSelectorCPUReplayTests(unittest.TestCase):
    def test_replays_supported_folds_and_abstains_without_source_support(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("a", "b", "c")
            trials = []
            sequence = 0
            for target in environments:
                for selector in selectors:
                    sequence += 1
                    trials.append(
                        {
                            "trial_id": f"{target}-{selector}",
                            "fold_id": f"holdout-{target}",
                            "target_environment": target,
                            "source_environments": [value for value in environments if value != target],
                            "selector": selector,
                            "seed": 101,
                        }
                    )
            plan = root / "plan.json"
            plan.write_text(json.dumps({"trials": trials}), encoding="utf-8")
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps(
                        {
                            "environment": environment,
                            "selector": selector,
                            "features": [float(index), float(index + 1)],
                        }
                    )
                    + "\n"
                    for index, environment in enumerate(environments)
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("a", "shared", True, 1.0),
                            self._label("b", "shared", True, 0.5),
                            self._label("c", "target_only", False, -1.0),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )
            self.assertEqual(manifest["state"], "partial")
            self.assertEqual(manifest["evaluated_cell_count"], 8)
            self.assertEqual(manifest["abstained_cell_count"], 4)
            report = json.loads((root / "output" / "selector-replay.json").read_text())
            self.assertEqual(report["evaluated_environment_count"], 2)
            self.assertFalse(report["formal_comparison_ready"])
            self.assertFalse(report["selection_discrimination_ready"])
            self.assertTrue(all(row["top1_positive_hit"] == 1.0 for row in report["selectors"]))

    def test_complete_identical_choices_are_not_selection_discriminative(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("a", "b")
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "trial_id": f"{target}-{selector}",
                                "fold_id": f"holdout-{target}",
                                "target_environment": target,
                                "source_environments": [source for source in environments if source != target],
                                "selector": selector,
                                "seed": 101,
                            }
                            for target in environments
                            for selector in selectors
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps({"environment": environment, "selector": selector, "features": [0.0]}) + "\n"
                    for environment in environments
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label(environment, primitive, positive, 1.0 if positive else -1.0)
                            for environment in environments
                            for primitive, positive in (("positive", True), ("negative", False))
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )

            self.assertTrue(manifest["formal_comparison_ready"])
            self.assertFalse(manifest["selection_discrimination_ready"])

    def test_ambiguous_source_effect_does_not_support_transfer(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("source", "target")
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "trial_id": f"target-{selector}",
                                "fold_id": "holdout-target",
                                "target_environment": "target",
                                "source_environments": ["source"],
                                "selector": selector,
                                "seed": 101,
                            }
                            for selector in selectors
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps(
                        {
                            "environment": environment,
                            "selector": selector,
                            "features": [0.0],
                        }
                    )
                    + "\n"
                    for environment in environments
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("source", "mixed", True, 0.2),
                            self._label("source", "mixed", False, -0.1),
                            self._label("target", "mixed", True, 0.3),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )

            self.assertEqual(manifest["state"], "partial")
            self.assertEqual(manifest["evaluated_cell_count"], 0)
            self.assertEqual(manifest["abstained_cell_count"], 4)

    @staticmethod
    def _label(environment: str, primitive: str, positive: bool, psnr: float) -> dict[str, object]:
        return {
            "environment": environment,
            "primitive": primitive,
            "settled": True,
            "positive": positive,
            "delta_candidate_minus_baseline": {"psnr": psnr},
        }


if __name__ == "__main__":
    unittest.main()
