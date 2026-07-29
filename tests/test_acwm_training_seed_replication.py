from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_training_seed_replication_queue import build_queue
from scripts.export.acwm_training_seed_replication_summary import (
    AcwmTrainingSeedSummaryError,
    build_summary,
)


class AcwmTrainingSeedReplicationTests(unittest.TestCase):
    def test_queue_builds_complete_factorial_without_positive_dependency(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-acwm-targeted-gap-plan",
                        "state": "ready",
                        "experiment_id": "test",
                        "training_seed_contract": {
                            "seeds": [11, 22, 33],
                            "evaluation_seeds": [101, 202, 303],
                        },
                        "environment_records": [
                            {
                                "environment": "cloth_move",
                                "recommended_existing_primitives": ["self_forcing_finetune"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            build_queue(
                experiment_config=config,
                output_root=root / "queue",
                repo_root=root / "repo",
                report_root=root / "reports",
                candidate_gpus=[0, 1, 2],
            )
            payload = json.loads((root / "queue/queue.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["row_count"], 9)
            self.assertEqual(
                {(row["training_seed"], row["eval_seed"]) for row in payload["rows"]},
                {(training, evaluation) for training in (11, 22, 33) for evaluation in (101, 202, 303)},
            )
            self.assertTrue(all(row["requires_positive_manifest"] == "" for row in payload["rows"]))
            self.assertTrue(all(row["requires_ready_manifest"].endswith("/manifest.json") for row in payload["rows"]))
            self.assertTrue(all("--training-seed" in row["launch_argv_template"] for row in payload["rows"]))

    def test_summary_requires_distinct_training_checkpoints_and_full_matrix(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for training_seed in (11, 22, 33):
                for eval_seed in (101, 202, 303):
                    path = root / f"ts{training_seed}-es{eval_seed}.json"
                    _write_receipt(path, training_seed, eval_seed, chr(96 + training_seed // 11) * 64)
                    manifests.append(path)

            build_summary(manifests=manifests, output_root=root / "summary")
            payload = json.loads((root / "summary/summary.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["factorial_shape"], [3, 3])
            self.assertEqual(payload["distinct_candidate_checkpoint_count"], 3)
            self.assertEqual(payload["official_gate_pass_count"], 9)
            self.assertAlmostEqual(payload["metric_summary"]["psnr"]["mean"], 0.2)
            public = json.loads((root / "summary/public-summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("manifest_path", public["records"][0])
            self.assertNotIn("checkpoint_path", public["records"][0])
            public_csv = (root / "summary/public-factorial-cells.csv").read_text(encoding="utf-8")
            self.assertNotIn("manifest_path", public_csv.splitlines()[0])

    def test_summary_rejects_missing_cell(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for training_seed in (11, 22):
                for eval_seed in (101, 202):
                    if (training_seed, eval_seed) == (22, 202):
                        continue
                    path = root / f"ts{training_seed}-es{eval_seed}.json"
                    _write_receipt(path, training_seed, eval_seed, str(training_seed // 11) * 64)
                    manifests.append(path)

            with self.assertRaisesRegex(AcwmTrainingSeedSummaryError, "FACTORIAL_INCOMPLETE"):
                build_summary(manifests=manifests, output_root=root / "summary")


def _write_receipt(path: Path, training_seed: int, eval_seed: int, checkpoint_sha: str) -> None:
    offset = (training_seed // 11 + eval_seed // 101) / 10.0
    path.write_text(
        json.dumps(
            {
                "artifact_type": "wmloop-acwm-formal-visualization-export",
                "state": "ready",
                "environment": "cloth_move",
                "primitive": "self_forcing_finetune",
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "candidate_checkpoint_sha256": checkpoint_sha,
                "candidate_checkpoint": f"/checkpoints/{training_seed}.pt",
                "official_quality_gate": {
                    "pass": True,
                    "delta_candidate_minus_baseline": {
                        "psnr": offset - 0.2,
                        "ssim": offset / 100.0,
                        "mse": -offset / 1000.0,
                        "masked_mse": -offset / 100.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
