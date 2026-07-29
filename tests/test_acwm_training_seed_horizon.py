from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_training_seed_horizon_queue import build_training_seed_horizon_queue
from scripts.export.acwm_training_seed_horizon_summary import build_training_seed_horizon_summary


class AcwmTrainingSeedHorizonTests(unittest.TestCase):
    def test_queue_binds_selected_checkpoints_and_shared_baseline(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            reports = repo / "results/reports"
            reports.mkdir(parents=True)
            data = root / "data"
            checkpoints = root / "checkpoints"
            data.mkdir()
            checkpoints.mkdir()
            runtime_python = root / "python"
            runtime_python.write_bytes(b"python")
            selected = []
            for seed in (41, 42):
                training = reports / f"screen-{seed}/envs/cloth_move/retained_training"
                training.mkdir(parents=True)
                checkpoint = training / "latest.pt"
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                (training.parent / "retained_runtime").mkdir()
                selected.append(
                    {
                        "training_seed": seed,
                        "checkpoint_step": 512,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "mean_delta_psnr": 0.5,
                    }
                )
                for eval_seed in (101, 202):
                    gate = reports / (
                        "acwm-formal-trainseed-gate-cloth_move-self_forcing_finetune-"
                        f"ts{seed}-es{eval_seed}-r1/manifest.json"
                    )
                    gate.parent.mkdir(parents=True)
                    gate.write_text(
                        json.dumps({"state": "ready", "official_quality_gate": {"pass": True}}),
                        encoding="utf-8",
                    )
            stability = reports / "stability/summary.json"
            stability.parent.mkdir()
            stability.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary",
                        "state": "ready",
                        "environment": "cloth_move",
                        "primitive": "self_forcing_finetune",
                        "eval_seeds": [101, 202],
                        "selected_checkpoints": selected,
                    }
                ),
                encoding="utf-8",
            )
            stability_manifest = reports / "stability/manifest.json"
            stability_manifest.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary-manifest",
                        "state": "ready",
                        "summary_path": str(stability),
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_training_seed_horizon_queue(
                stability_manifest=stability_manifest,
                output_root=reports / "queue",
                report_root=reports,
                repo_root=repo,
                runtime_python=runtime_python,
                data_root=data,
                checkpoint_root=checkpoints,
                gpus=(0, 1, 2),
            )

            self.assertEqual(manifest["row_count"], 9)
            queue = json.loads((reports / "queue/autoloop-queue.json").read_text(encoding="utf-8"))
            phases = [row["phase"] for row in queue["rows"]]
            self.assertEqual(phases.count("long_horizon_baseline"), 1)
            self.assertEqual(phases.count("long_horizon_candidate"), 2)
            self.assertEqual(phases.count("horizon_effect_profile"), 2)
            self.assertEqual(phases.count("horizon_triptych"), 2)
            baseline_manifest = Path(queue["rows"][0]["output_root"]) / "manifest.json"
            for row in queue["rows"][1:3]:
                self.assertIn(str(baseline_manifest), row["requires_ready_manifests"])
                self.assertIn("--vendor-root", row["launch_argv_template"])

    def test_summary_reports_training_seed_sensitive_effect(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stability = root / "stability.json"
            stability.write_text(
                json.dumps(
                    {
                        "environment": "cloth_move",
                        "primitive": "self_forcing_finetune",
                        "global_selection_reason": "frozen",
                        "selected_checkpoints": [
                            {"training_seed": 41, "checkpoint_step": 512, "checkpoint_sha256": "a" * 64},
                            {"training_seed": 42, "checkpoint_step": 800, "checkpoint_sha256": "b" * 64},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
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
            profiles = []
            for seed, passing in ((41, True), (42, False)):
                path = root / f"profile-{seed}.json"
                effects = []
                for horizon in (16, 32):
                    effects.append(
                        {
                            "horizon": horizon,
                            "strict_quality_pass": passing,
                            "delta_candidate_minus_baseline": {
                                "psnr": 0.5 if passing else -0.2,
                                "ssim": 0.01 if passing else -0.01,
                                "mse": -0.01 if passing else 0.01,
                                "masked_mse": -0.01 if passing else 0.01,
                            },
                        }
                    )
                path.write_text(
                    json.dumps(
                        {
                            "artifact_type": "wmloop-acwm-horizon-effect-profile",
                            "state": "ready",
                            "environment": "cloth_move",
                            "primitive": "self_forcing_finetune",
                            "training_seed": seed,
                            "horizons": [16, 32],
                            "effect_classification": {
                                "effect_scope": "aggregate_long_horizon_positive" if passing else "no_positive_effect",
                                "aggregate_max_horizon_pass": passing,
                                "positive_trajectory_rate_at_max_horizon": 1.0 if passing else 0.0,
                            },
                            "horizon_effects": effects,
                        }
                    ),
                    encoding="utf-8",
                )
                profiles.append(path)

            report = build_training_seed_horizon_summary(
                profile_paths=profiles,
                stability_manifest=manifest,
                output_root=root / "out",
            )

            self.assertEqual(report["stability_verdict"], "training_seed_sensitive_long_horizon_effect")
            summary = json.loads((root / "out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["max_horizon_pass_count"], 1)


if __name__ == "__main__":
    unittest.main()
