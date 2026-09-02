import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_wan22_worldarena_input import materialize
from scripts.verify_wan22_droid_worldarena import verify


class Wan22WorldArenaPipelineTests(unittest.TestCase):
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
                materialize(run_root=run, output_root=root / "out", config_template=root / "missing.yaml")

    def test_verifier_fails_closed_on_single_seed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "run"
            run.mkdir()
            (run / "training_receipt.json").write_text(json.dumps({"seed": 4101, "sample_id": "episode-a:0:150"}))
            metrics = {metric: {"raw": 0.1} for metric in ("subject_consistency", "background_consistency", "photometric_smoothness")}
            receipt = run / "worldarena_metrics_receipt.json"
            receipt.write_text(json.dumps({"artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt", "state": "evaluated_partial", "returncode": 0, "metrics": metrics, "video": {"generated_frames": 150, "fps": 5.0}}))
            result = verify([receipt])
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(any("SEED_SET_INVALID" in code for code in result["blockers"]))


if __name__ == "__main__":
    unittest.main()
