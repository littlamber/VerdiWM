import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_wan22_confirmations import METRICS, METRICS_RECEIPT, summarize


class Wan22ConfirmationSummaryTests(unittest.TestCase):
    def _run(self, root: Path, seed: int, subject: float, background: float, photo: float) -> Path:
        run = root / str(seed)
        run.mkdir()
        training = {
            "seed": seed, "gpu_hours": 0.1, "conditioning_mode": "action_proprio_ema",
            "history_decay": 0.8, "anchor_policy": "initial_reference_blend",
            "anchor_refresh_strength": 0.25, "horizon_frames": 150, "chunk_frames": 45, "steps": 8,
        }
        (run / "training_receipt.json").write_text(json.dumps(training), encoding="utf-8")
        values = {"subject_consistency": subject, "background_consistency": background, "motion_smoothness": 0.5, "photometric_smoothness": photo}
        metrics = {key: {"per_video": [{"video_results_normalized": values[key]}]} for key in METRICS}
        (run / METRICS_RECEIPT).write_text(json.dumps({"state": "evaluated_partial", "returncode": 0, "metrics": metrics}), encoding="utf-8")
        return run

    def test_aggregates_and_emits_configurable_failure_signatures(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = [self._run(root, 1, 0.8, 0.9, 0.0), self._run(root, 2, 0.1, 0.2, 0.0)]
            receipt = summarize(runs, candidate_id="candidate", visual_bifurcation_range=0.5)
            self.assertEqual(receipt["seed_count"], 2)
            self.assertAlmostEqual(receipt["aggregate"]["subject_consistency"]["mean"], 0.45)
            self.assertIn("visual_seed_bifurcation", receipt["failure_signatures"])
            self.assertIn("photometric_smoothness_floor", receipt["failure_signatures"])

    def test_rejects_mixed_candidate_configurations(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = [self._run(root, 1, 0.5, 0.5, 0.1), self._run(root, 2, 0.5, 0.5, 0.1)]
            payload = json.loads((runs[1] / "training_receipt.json").read_text(encoding="utf-8"))
            payload["anchor_refresh_strength"] = 0.5
            (runs[1] / "training_receipt.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "WAN22_CONFIRMATION_CONFIGURATION_MISMATCH"):
                summarize(runs, candidate_id="mixed")


if __name__ == "__main__":
    unittest.main()
