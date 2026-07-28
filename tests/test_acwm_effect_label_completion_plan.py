from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.effect_labels import build_effect_label_completion_plan, build_effect_label_index


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
            self.assertEqual(report["new_screen_action_count"], 1)
            self.assertEqual(report["actions"][0]["action"], "official_gate_existing_checkpoint")
            self.assertEqual(report["environment_plans"][0]["current_settled_primitive_count"], 0)
            self.assertEqual(report["environment_plans"][0]["candidate_count"], 2)

    def test_completion_gate_manifest_becomes_a_settled_label(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            gate = reports / "acwm-effect-label-gate-reacher-next_forcing-s1-r1"
            gate.mkdir(parents=True)
            (gate / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "environment": "reacher",
                        "primitive": "next_forcing",
                        "seed": 1,
                        "steps": 50,
                        "official_quality_gate": {
                            "state": "fail",
                            "pass": False,
                            "checks": {
                                "psnr_strictly_improves": False,
                                "ssim_does_not_regress": False,
                            },
                            "delta_candidate_minus_baseline": {"psnr": -1.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "index"
            build_effect_label_index(
                reports_root=reports,
                output_root=output,
                expected_environments=("reacher",),
            )
            report = json.loads((output / "effect-label-index.json").read_text(encoding="utf-8"))
            self.assertEqual(report["settled_label_count"], 1)
            self.assertEqual(report["settled_negative_count"], 1)
            self.assertEqual(report["labels"][0]["label_source"], "retained_checkpoint_completion_gate")

    def test_completion_plan_refills_ambiguous_checkpoint_effects(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            label_index = root / "labels.json"
            label_index.write_text(
                json.dumps(
                    {
                        "expected_environments": ["reacher"],
                        "settled_labels_by_environment": {"reacher": 3},
                        "settled_primitives_by_environment": {
                            "reacher": ["self_forcing_finetune", "next_forcing"]
                        },
                        "labels": [
                            {
                                "environment": "reacher",
                                "primitive": "self_forcing_finetune",
                                "settled": True,
                                "positive": False,
                            },
                            {
                                "environment": "reacher",
                                "primitive": "next_forcing",
                                "settled": True,
                                "positive": True,
                            },
                            {
                                "environment": "reacher",
                                "primitive": "next_forcing",
                                "settled": True,
                                "positive": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            screen_summary = root / "screens.json"
            screen_summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "environment": "reacher",
                                "candidate_checkpoint_retained": True,
                                "latest_checkpoint_path": "candidate.pt",
                                "candidate_checkpoint_sha256": "abc",
                                "delta_primary_metric": 0.2,
                                "primitive": "motion_region_reweight",
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
            frontier = root / "frontier.json"
            frontier.write_text(
                json.dumps({"environments": [{"environment": "reacher", "open_candidates": []}]}),
                encoding="utf-8",
            )

            build_effect_label_completion_plan(
                effect_label_index_path=label_index,
                screen_summary_path=screen_summary,
                candidate_frontier_path=frontier,
                output_root=root / "output",
            )

            report = json.loads((root / "output" / "effect-label-completion-plan.json").read_text())
            self.assertEqual(report["environment_count"], 1)
            self.assertEqual(report["environment_plans"][0]["current_settled_primitives"], ["self_forcing_finetune"])
            self.assertEqual(report["actions"][0]["primitive"], "motion_region_reweight")

    def test_selector_index_excludes_invalidated_and_unmaterialized_runtime_only_primitives(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            cases = (
                ("push_cube", "latent_motion_prior", 11),
                ("stack_cube", "latent_spatial_memory", 12),
            )
            for environment, primitive, seed in cases:
                gate = reports / f"acwm-effect-label-gate-{environment}-{primitive}-s{seed}-r1"
                gate.mkdir(parents=True)
                (gate / "manifest.json").write_text(
                    json.dumps(
                        {
                            "state": "ready",
                            "environment": environment,
                            "primitive": primitive,
                            "seed": seed,
                            "steps": 50,
                            "baseline_runtime_sha256": "same",
                            "candidate_runtime_sha256": "same",
                            "candidate_runtime_contract": {},
                            "official_quality_gate": {
                                "state": "fail",
                                "pass": False,
                                "checks": {"psnr_strictly_improves": False},
                                "delta_candidate_minus_baseline": {"psnr": -1.0},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            output = root / "index"
            build_effect_label_index(
                reports_root=reports,
                output_root=output,
                expected_environments=("push_cube", "stack_cube"),
            )
            report = json.loads((output / "effect-label-index.json").read_text(encoding="utf-8"))
            self.assertEqual(report["settled_label_count"], 0)
            self.assertEqual(report["excluded_settled_evidence_count"], 2)
            self.assertEqual(report["missing_environments"], ["push_cube", "stack_cube"])
            reasons = {row["selector_exclusion_reason"] for row in report["labels"]}
            self.assertEqual(
                reasons,
                {
                    "primitive_invalidated_for_quality_claims",
                    "runtime_only_primitive_missing_materialized_candidate_runtime",
                },
            )


if __name__ == "__main__":
    unittest.main()
