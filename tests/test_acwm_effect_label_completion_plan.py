from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.effect_labels import build_effect_label_completion_plan


class AcwmEffectLabelCompletionPlanTests(unittest.TestCase):
    def test_reuses_retained_checkpoint_before_scheduling_new_screens(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            label_index = root / "labels.json"
            screen_summary = root / "screens.json"
            frontier = root / "frontier.json"
            label_index.write_text(
                json.dumps(
                    {
                        "missing_environments": ["reacher"],
                        "settled_labels_by_environment": {"reacher": 0},
                    }
                ),
                encoding="utf-8",
            )
            screen_summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "environment": "reacher",
                                "candidate_checkpoint_retained": True,
                                "latest_checkpoint_path": "latest.pt",
                                "candidate_checkpoint_sha256": "abc",
                                "delta_primary_metric": 3.0,
                                "primitive": "next_forcing",
                                "seed": 101,
                                "train_steps": 512,
                                "campaign_id": "screen-r1",
                                "manifest_path": "screen-r1/manifest.json",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            frontier.write_text(
                json.dumps(
                    {
                        "environments": [
                            {
                                "environment": "reacher",
                                "open_candidates": [
                                    {
                                        "primitive": "action_dimension_balancing",
                                        "parameters": {"action_balance_blend": 0.5},
                                        "failure_evidence_penalty": 0,
                                    },
                                    {
                                        "primitive": "self_forcing_finetune",
                                        "parameters": {"self_forcing_steps": 1},
                                        "failure_evidence_penalty": 1,
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            build_effect_label_completion_plan(
                effect_label_index_path=label_index,
                screen_summary_path=screen_summary,
                candidate_frontier_path=frontier,
                output_root=output,
                candidates_per_environment=3,
            )
            report = json.loads((output / "effect-label-completion-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(report["reusable_checkpoint_action_count"], 1)
            self.assertEqual(report["new_screen_action_count"], 2)
            self.assertEqual(report["actions"][0]["action"], "official_gate_existing_checkpoint")


if __name__ == "__main__":
    unittest.main()
