from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.effect_label_queue import build_effect_label_gate_queue


class AcwmEffectLabelGateQueueTests(unittest.TestCase):
    def test_builds_ready_dependency_gate_without_positive_screen_requirement(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            reports = repo / "results/reports"
            screen = reports / "screen-r1/envs/reacher"
            retained = screen / "retained_training"
            retained.mkdir(parents=True)
            (repo / "vendor/ACWM-Phys").mkdir(parents=True)
            (repo / ".venv/bin").mkdir(parents=True)
            (repo / ".venv/bin/python3").write_text("", encoding="utf-8")
            data = root / "data"
            checkpoints = root / "checkpoints"
            data.mkdir()
            checkpoints.mkdir()
            checkpoint = retained / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            (retained / "manifest.json").write_text(
                json.dumps({"retained_path": str(checkpoint), "sha256": digest}),
                encoding="utf-8",
            )
            source_manifest = screen / "manifest.json"
            source_manifest.write_text(
                json.dumps({"state": "ready", "environment": "reacher"}),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-acwm-effect-label-completion-plan",
                        "actions": [
                            {
                                "ordinal": 1,
                                "action": "official_gate_existing_checkpoint",
                                "environment": "reacher",
                                "primitive": "next_forcing",
                                "seed": 101,
                                "source_manifest": "screen-r1/envs/reacher/manifest.json",
                                "checkpoint_ref": "screen-r1/envs/reacher/retained_training/latest.pt",
                                "checkpoint_sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "queue"
            build_effect_label_gate_queue(
                completion_plan_path=plan,
                reports_root=reports,
                repo_root=repo,
                output_root=output,
                runtime_python=repo / ".venv/bin/python3",
                data_root=data,
                checkpoint_root=checkpoints,
                gpus=(0, 1),
            )
            queue = json.loads((output / "autoloop-queue.json").read_text(encoding="utf-8"))
            self.assertEqual(queue["row_count"], 1)
            row = queue["rows"][0]
            self.assertEqual(row["requires_positive_manifest"], "")
            self.assertEqual(row["requires_ready_manifest"], str(source_manifest))
            self.assertEqual(row["candidate_gpus"], [0, 1])
            self.assertIn("--steps", row["launch_argv_template"])
            self.assertEqual(row["launch_argv_template"][row["launch_argv_template"].index("--steps") + 1], "50")


if __name__ == "__main__":
    unittest.main()
