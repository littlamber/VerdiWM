from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_horizon_effect_profile import (
    AcwmHorizonEffectProfileError,
    build_horizon_effect_profile,
)


class AcwmHorizonEffectProfileTests(unittest.TestCase):
    def test_hard_case_gain_is_not_promoted_to_aggregate_uplift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True)

            report = build_horizon_effect_profile(
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                primitive="cfg_guidance_schedule",
                output_root=root / "out",
            )

            effect = report["effect_classification"]
            self.assertEqual(effect["effect_scope"], "hard_case_only_positive")
            self.assertEqual(effect["positive_trajectory_indices_at_max_horizon"], [72])
            self.assertFalse(effect["aggregate_max_horizon_pass"])
            self.assertFalse(report["transfer_prior"]["causal_credit_eligible"])
            self.assertFalse(report["selection_policy"]["candidate_gain_used_for_sample_selection"])
            self.assertTrue((root / "out" / "horizon-effects.csv").is_file())

    def test_mismatched_trajectory_pool_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True, indices=(35, 99))

            with self.assertRaisesRegex(AcwmHorizonEffectProfileError, "TRAJECTORY_SET_MISMATCH"):
                build_horizon_effect_profile(
                    baseline_manifest=baseline,
                    candidate_manifest=candidate,
                    primitive="cfg_guidance_schedule",
                    output_root=root / "out",
                )

    def test_composite_intervention_uses_source_primitive_mechanism_card(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True)
            cards = root / "mechanism-cards.csv"
            cards.write_text(
                "primitive,mechanism_family,layer,transfer_preconditions,known_anti_conditions\n"
                "self_forcing_finetune,On-policy rollout objective,L3,rollout hook,error reinforcement\n",
                encoding="utf-8",
            )

            report = build_horizon_effect_profile(
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                primitive="checkpoint_delta_scaling",
                mechanism_primitive="self_forcing_finetune",
                mechanism_cards=cards,
                output_root=root / "out",
            )

            self.assertEqual(report["mechanism_primitive"], "self_forcing_finetune")
            self.assertEqual(
                report["transfer_prior"]["intervention_chain"],
                ["self_forcing_finetune", "checkpoint_delta_scaling"],
            )
            self.assertEqual(
                report["transfer_prior"]["mechanism_family"],
                "On-policy rollout objective",
            )
            self.assertIn(
                "Mechanism primitive: `self_forcing_finetune`",
                (root / "out/horizon-effect-profile.md").read_text(encoding="utf-8"),
            )

    def test_metric_gain_with_failed_event_becomes_anti_condition(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True)
            event_gate = root / "event-gate.json"
            event_gate.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-acwm-pour-water-event-gate",
                        "state": "ready",
                        "environment": "robot_arm",
                        "classification": "event_failure",
                        "candidate_event_pass": False,
                        "baseline_event_pass": False,
                        "baseline_mean_completion_ratio": 0.01,
                        "candidate_mean_completion_ratio": 0.01,
                        "mean_completion_uplift": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            report = build_horizon_effect_profile(
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                primitive="cfg_guidance_schedule",
                event_gate=event_gate,
                output_root=root / "out",
            )

            classification = report["effect_classification"]
            self.assertEqual(classification["metric_effect_scope"], "hard_case_only_positive")
            self.assertEqual(classification["effect_scope"], "metric_positive_event_failure")
            self.assertFalse(classification["candidate_event_pass"])
            self.assertIn("Event classification: `event_failure`", (root / "out/horizon-effect-profile.md").read_text())

    def test_checkpoint_ladder_records_later_training_regression(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True)
            report_path = root / "checkpoint-ladder.json"
            report_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-acwm-checkpoint-ladder-finalization",
                        "state": "ready",
                        "environment": "robot_arm",
                        "primitive": "cfg_guidance_schedule",
                        "best_checkpoint_step": 800,
                        "evaluated_steps": [512, 800, 1000],
                        "selection": {
                            "state": "ready",
                            "extension_allowed": False,
                            "stop_requested": False,
                            "records": [
                                {
                                    "checkpoint_step": 800,
                                    "official_gate_passed": True,
                                    "regressed_from_running_best": False,
                                    "official_quality_gate": {
                                        "delta_candidate_minus_baseline": {"psnr": 0.4}
                                    },
                                },
                                {
                                    "checkpoint_step": 1000,
                                    "official_gate_passed": False,
                                    "regressed_from_running_best": True,
                                    "official_quality_gate": {
                                        "delta_candidate_minus_baseline": {"psnr": -1.3}
                                    },
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            ladder_manifest = root / "checkpoint-ladder-manifest.json"
            ladder_manifest.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-acwm-checkpoint-ladder-finalization-manifest",
                        "state": "ready",
                        "environment": "robot_arm",
                        "primitive": "cfg_guidance_schedule",
                        "confirmation_passed": True,
                        "report_path": str(report_path),
                    }
                ),
                encoding="utf-8",
            )

            report = build_horizon_effect_profile(
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                primitive="cfg_guidance_schedule",
                checkpoint_ladder_manifest=ladder_manifest,
                output_root=root / "out",
            )

            window = report["transfer_prior"]["effective_training_window"]
            self.assertEqual(window["selected_step"], 800)
            self.assertTrue(window["later_regression_observed"])
            self.assertEqual(window["first_later_regression_step"], 1000)

    def test_training_seed_stability_manifest_binds_selected_window(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _manifest(root / "baseline.json", candidate=False)
            candidate = _manifest(root / "candidate.json", candidate=True)
            stability = root / "stability.json"
            stability.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary",
                        "state": "ready",
                        "environment": "robot_arm",
                        "primitive": "cfg_guidance_schedule",
                        "checkpoint_steps": [512, 800, 1000],
                        "global_selection_reason": "all_eval_seeds_pass_then_max_mean_psnr_tie_earlier_step",
                        "per_training_seed": [
                            {
                                "training_seed": 41,
                                "checkpoint_candidates": [
                                    {"checkpoint_step": 512, "all_eval_seeds_pass": True, "mean_delta_psnr": 0.4},
                                    {"checkpoint_step": 800, "all_eval_seeds_pass": True, "mean_delta_psnr": 0.7},
                                    {"checkpoint_step": 1000, "all_eval_seeds_pass": False, "mean_delta_psnr": 0.2},
                                ],
                            }
                        ],
                        "selected_checkpoints": [
                            {"training_seed": 41, "checkpoint_step": 800, "mean_delta_psnr": 0.7}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "stability-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary-manifest",
                        "state": "ready",
                        "summary_path": str(stability),
                    }
                ),
                encoding="utf-8",
            )

            report = build_horizon_effect_profile(
                baseline_manifest=baseline,
                candidate_manifest=candidate,
                primitive="cfg_guidance_schedule",
                checkpoint_ladder_manifest=manifest,
                training_seed=41,
                output_root=root / "out",
            )

            self.assertEqual(report["training_seed"], 41)
            window = report["transfer_prior"]["effective_training_window"]
            self.assertEqual(window["selected_step"], 800)
            self.assertTrue(window["later_regression_observed"])
            self.assertEqual(window["first_later_regression_step"], 1000)


def _manifest(path: Path, *, candidate: bool, indices: tuple[int, ...] = (35, 72)) -> Path:
    base_psnr = {16: 28.0, 32: 27.5, 48: 27.0}
    candidate_psnr = {16: 27.9, 32: 27.2, 48: 26.8}
    horizon_metrics = {}
    trajectories = []
    for horizon in (16, 32, 48):
        psnr = candidate_psnr[horizon] if candidate else base_psnr[horizon]
        horizon_metrics[str(horizon)] = {
            "sample_count": 2,
            "psnr": psnr,
            "ssim": 0.96 + (0.001 if candidate else 0.0),
            "mse": 0.002 + (0.0001 if candidate else 0.0),
            "masked_mse": 0.02 + (0.001 if candidate else 0.0),
        }
    for index in indices:
        metrics = {}
        for horizon in (16, 32, 48):
            if candidate and index == 72:
                values = {"psnr": 29.0, "ssim": 0.97, "mse": 0.0015, "masked_mse": 0.015}
            elif candidate:
                values = {"psnr": 25.0, "ssim": 0.95, "mse": 0.003, "masked_mse": 0.03}
            elif index == 72:
                values = {"psnr": 25.5, "ssim": 0.95, "mse": 0.003, "masked_mse": 0.03}
            else:
                values = {"psnr": 30.0, "ssim": 0.97, "mse": 0.001, "masked_mse": 0.01}
            metrics[str(horizon)] = values
        trajectories.append(
            {
                "trajectory_index": index,
                "metrics_by_horizon": metrics,
                "per_frame_psnr": {str(frame): 28.0 for frame in range(48)},
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-probe-run",
        "state": "ready",
        "environment": "robot_arm",
        "split": "ind_test",
        "metadata_sha256": "a" * 64,
        "mode": "autoregressive",
        "num_inference_steps": 50,
        "horizons": [16, 32, 48],
        "checkpoint_path": "candidate.pt" if candidate else "baseline.pt",
        "checkpoint_step": 512 if candidate else 100000,
        "aggregate": {
            "horizon_metrics": horizon_metrics,
            "per_frame_psnr": {
                str(frame): 27.5 if candidate else 28.0 for frame in range(48)
            },
        },
        "trajectory_results": trajectories,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
