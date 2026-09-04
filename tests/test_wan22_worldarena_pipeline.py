import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_wan22_worldarena_input import (
    _bind_evaluator_paths,
    materialize,
    materialize_many,
)
from scripts.verify_wan22_droid_worldarena import verify


class Wan22WorldArenaPipelineTests(unittest.TestCase):
    def test_runtime_config_is_bound_structurally(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            worldarena = root / "WorldArena"
            (worldarena / "video_quality").mkdir(parents=True)
            (worldarena / "video_quality" / "evaluate.py").write_text("# evaluator\n")
            assets = root / "assets"
            required = [
                "clip/ViT-B-32.pt",
                "raft/raft-things.pth",
                "vfimamba/model.pkl",
                "sea-raft/Tartan-C-T-TSKH-spring540x960-M.pth",
                "dino/dino_vitbase16_pretrain.pth",
            ]
            for name in required:
                path = assets / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            (assets / "dino" / "facebookresearch_dino").mkdir()
            sea_config = root / "sea.json"
            sea_config.write_text("{}")
            config = {}
            _bind_evaluator_paths(
                config,
                worldarena_root=worldarena,
                asset_root=assets,
                sea_raft_config=sea_config,
            )
            self.assertEqual(
                config["ckpt"]["motion_smoothness"]["model"],
                str(assets / "vfimamba" / "model.pkl"),
            )
            self.assertEqual(
                config["ckpt"]["photometric_smoothness"]["cfg"], str(sea_config)
            )

    def test_materializer_rejects_bound_or_nonempty_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "run"
            run.mkdir()
            for name in ("generated_150f.mp4", "ground_truth_150f.mp4"):
                (run / name).write_bytes(b"not-a-video")
            (run / "first_frame.png").write_bytes(b"x")
            (run / "worldarena_input.json").write_text(json.dumps({"episode_id": "e"}))
            (root / "out").mkdir()
            (root / "out" / "leftover").write_text("x")
            with self.assertRaisesRegex(ValueError, "OUTPUT_UNBOUND"):
                materialize(
                    run_root=run,
                    output_root=root / "out",
                    config_template=root / "missing.yaml",
                )

    def test_verifier_fails_closed_on_single_seed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "run"
            run.mkdir()
            (run / "training_receipt.json").write_text(
                json.dumps({"seed": 4101, "sample_id": "episode-a:0:150"})
            )
            metrics = {
                metric: {"raw": 0.1}
                for metric in (
                    "subject_consistency",
                    "background_consistency",
                    "photometric_smoothness",
                )
            }
            receipt = run / "worldarena_metrics_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
                        "state": "evaluated_partial",
                        "returncode": 0,
                        "metrics": metrics,
                        "video": {"generated_frames": 150, "fps": 5.0},
                    }
                )
            )
            result = verify([receipt])
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(
                any("SEED_SET_INVALID" in code for code in result["blockers"])
            )

    def test_verifier_fails_closed_on_malformed_training_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "run"
            run.mkdir()
            (run / "training_receipt.json").write_text("not-json", encoding="utf-8")
            receipt = run / "worldarena_metrics_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
                        "state": "evaluated_partial",
                        "returncode": 0,
                        "video": {"generated_frames": 150, "fps": 5.0},
                        "metrics": {
                            metric: {"raw": 0.1}
                            for metric in (
                                "subject_consistency",
                                "background_consistency",
                                "photometric_smoothness",
                            )
                        },
                    }
                )
            )
            result = verify([receipt])
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(
                any("TRAINING_RECEIPT_INVALID" in code for code in result["blockers"])
            )

    def test_verifier_requires_same_frozen_panel_for_each_seed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipts = []
            metrics = {
                metric: {"raw": 0.1}
                for metric in (
                    "subject_consistency",
                    "background_consistency",
                    "photometric_smoothness",
                )
            }
            for seed in (1, 2):
                seed_root = root / f"seed-{seed}"
                seed_root.mkdir()
                (seed_root / "training_receipt.json").write_text(
                    json.dumps({"seed": seed}), encoding="utf-8"
                )
                for sample in ("episode-a:0:150", "episode-b:0:150"):
                    receipt = seed_root / f"metrics-{sample.split(':', 1)[0]}.json"
                    receipt.write_text(
                        json.dumps(
                            {
                                "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
                                "state": "evaluated_partial",
                                "returncode": 0,
                                "metrics": metrics,
                                "video": {
                                    "generated_frames": 150,
                                    "fps": 5.0,
                                    "sample_id": sample,
                                    "episode_id": sample.split(":", 1)[0],
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    receipts.append(receipt)
            result = verify(receipts, expected_seeds=(1, 2), expected_panel_size=2)
            self.assertEqual(result["state"], "verified")
            self.assertEqual(result["observed_panel_sizes"], {"1": 2, "2": 2})

    def test_verifier_explicit_formal_metrics_cannot_be_downgraded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipts = []
            visual_metrics = {
                metric: {"raw": 0.1}
                for metric in (
                    "subject_consistency",
                    "background_consistency",
                    "motion_smoothness",
                    "photometric_smoothness",
                )
            }
            for seed in (1, 2, 3):
                seed_root = root / f"seed-{seed}"
                seed_root.mkdir()
                (seed_root / "training_receipt.json").write_text(
                    json.dumps({"seed": seed}), encoding="utf-8"
                )
                receipt = seed_root / "metrics.json"
                receipt.write_text(
                    json.dumps(
                        {
                            "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
                            "state": "evaluated_partial",
                            "returncode": 0,
                            "metrics": visual_metrics,
                            "video": {
                                "generated_frames": 150,
                                "fps": 5.0,
                                "sample_id": "episode-a:0:150",
                                "episode_id": "episode-a",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                receipts.append(receipt)
            result = verify(
                receipts,
                required_metrics=(
                    "subject_consistency",
                    "background_consistency",
                    "motion_smoothness",
                    "photometric_smoothness",
                    "trajectory_accuracy",
                    "action_following",
                ),
                expected_seeds=(1, 2, 3),
            )
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(
                any("REQUIRED_METRICS_MISSING" in code for code in result["blockers"])
            )

    def test_verifier_rejects_repeated_episode_when_formal_panel_requires_three(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipts = []
            metrics = {
                metric: {"raw": 0.1}
                for metric in (
                    "subject_consistency",
                    "background_consistency",
                    "motion_smoothness",
                    "photometric_smoothness",
                    "trajectory_accuracy",
                    "action_following",
                )
            }
            for seed in (1, 2, 3):
                seed_root = root / f"seed-{seed}"
                seed_root.mkdir()
                (seed_root / "training_receipt.json").write_text(
                    json.dumps({"seed": seed}), encoding="utf-8"
                )
                for ordinal in range(3):
                    receipt = seed_root / f"metrics-{ordinal}.json"
                    receipt.write_text(
                        json.dumps(
                            {
                                "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
                                "state": "evaluated_partial",
                                "returncode": 0,
                                "metrics": metrics,
                                "video": {
                                    "generated_frames": 150,
                                    "fps": 5.0,
                                    "sample_id": f"episode-a:{ordinal}:150",
                                    "episode_id": "episode-a",
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    receipts.append(receipt)
            result = verify(
                receipts,
                required_metrics=tuple(metrics),
                expected_seeds=(1, 2, 3),
                expected_panel_size=3,
                expected_panel_episode_count=3,
            )
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(
                any(
                    "VALIDATION_PANEL_EPISODE_COUNT_INVALID" in code
                    for code in result["blockers"]
                )
            )


if __name__ == "__main__":
    unittest.main()
