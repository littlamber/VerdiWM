from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export.acwm_multiseed_eval_summary import (
    AcwmMultiseedSummaryError,
    export_acwm_multiseed_eval_summary,
)


class AcwmMultiseedEvalSummaryTest(unittest.TestCase):
    def test_exports_complete_eval_seed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = [self._receipt(root, seed, passed=seed != 3) for seed in (1, 2, 3)]
            output = root / "bundle"
            summary = export_acwm_multiseed_eval_summary(
                receipt_paths=receipts,
                output_root=output,
                expected_seeds_per_cell=3,
            )
            self.assertEqual(summary["receipt_count"], 3)
            self.assertEqual(summary["cells"][0]["pass_count"], 2)
            self.assertEqual(summary["cells"][0]["stability"], "eval_seed_sensitive_positive")
            self.assertEqual(len(summary["videos"]), 3)
            self.assertTrue((output / "tables/eval-seed-results.csv").is_file())
            self.assertTrue((output / "figures/eval-seed-replication.svg").is_file())
            self.assertTrue((output / "MANIFEST.sha256").is_file())

    def test_rejects_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = [self._receipt(root, seed, passed=True) for seed in (1, 2, 3)]
            payload = json.loads(receipts[-1].read_text(encoding="utf-8"))
            payload["candidate_checkpoint_sha256"] = "f" * 64
            receipts[-1].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AcwmMultiseedSummaryError, "CHECKPOINT_MISMATCH"):
                export_acwm_multiseed_eval_summary(
                    receipt_paths=receipts,
                    output_root=root / "bundle",
                    expected_seeds_per_cell=3,
                )

    def _receipt(self, root: Path, seed: int, *, passed: bool) -> Path:
        receipt_root = root / f"receipt-{seed}"
        (receipt_root / "hard_cases").mkdir(parents=True)
        video = receipt_root / "hard_cases/selected.mp4"
        video.write_bytes(f"video-{seed}".encode())
        sign = 1.0 if passed else -1.0
        payload = {
            "artifact_type": "wmloop-acwm-formal-visualization-export",
            "state": "ready",
            "environment": "cloth_move",
            "primitive": "self_forcing_finetune",
            "eval_seed": seed,
            "steps": 50,
            "candidate_checkpoint_sha256": "a" * 64,
            "baseline_checkpoint_sha256": "b" * 64,
            "official_quality_gate": {
                "state": "pass" if passed else "fail",
                "pass": passed,
                "protocol": "official_acwm_eval_py",
                "delta_candidate_minus_baseline": {
                    "psnr": sign * 0.5,
                    "ssim": sign * 0.01,
                    "mse": -sign * 0.001,
                    "masked_mse": -sign * 0.002,
                },
            },
            "hard_case_visualization": {
                "selected": [{"selected_video_path": str(video.resolve())}]
            },
        }
        path = receipt_root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(len(hashlib.sha256(video.read_bytes()).hexdigest()), 64)
        return path


if __name__ == "__main__":
    unittest.main()
