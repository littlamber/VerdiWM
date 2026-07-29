from __future__ import annotations

import unittest

import numpy as np

from wmloop.evaluate.cosmos3_paired_gt import (
    Cosmos3PairedGroundTruthError,
    compute_cosmos3_paired_metrics,
)


class Cosmos3PairedGroundTruthTests(unittest.TestCase):
    def test_excludes_conditioning_frame_and_uses_top_left_content(self) -> None:
        ground_truth = np.zeros((17, 8, 10, 3), dtype=np.uint8)
        ground_truth[1:] = 100
        rollout = ground_truth[:, :6, :8].copy()
        rollout[1:] = 110

        metrics, alignment = compute_cosmos3_paired_metrics(
            ground_truth=ground_truth,
            rollout=rollout,
        )

        expected = 10.0 / 255.0
        self.assertAlmostEqual(metrics["rollout_video_l1"], expected)
        self.assertAlmostEqual(metrics["final_frame_mae"], expected)
        self.assertEqual(alignment["future_start_index"], 1)
        self.assertEqual(alignment["spatial_policy"], "top_left_content_crop_to_rollout")

    def test_rejects_wrong_conditioning_frame(self) -> None:
        ground_truth = np.zeros((17, 8, 10, 3), dtype=np.uint8)
        rollout = ground_truth.copy()
        rollout[0] = 255
        with self.assertRaisesRegex(Cosmos3PairedGroundTruthError, "CONDITIONING_FRAME_MISALIGNED"):
            compute_cosmos3_paired_metrics(ground_truth=ground_truth, rollout=rollout)

    def test_rejects_incorrect_frame_count(self) -> None:
        ground_truth = np.zeros((17, 8, 10, 3), dtype=np.uint8)
        rollout = np.zeros((16, 8, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(Cosmos3PairedGroundTruthError, "FRAME_COUNT_MISMATCH"):
            compute_cosmos3_paired_metrics(ground_truth=ground_truth, rollout=rollout)


if __name__ == "__main__":
    unittest.main()
