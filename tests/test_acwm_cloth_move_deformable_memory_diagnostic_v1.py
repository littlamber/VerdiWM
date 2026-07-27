from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_cloth_move_deformable_memory_diagnostic_v1 import (
    ClothMoveDeformableMemoryProbeError,
    measure_deformable_memory,
)


class ClothMoveDeformableMemoryProbeTests(unittest.TestCase):
    def test_measures_deformable_memory_without_verdict_exposure(self) -> None:
        output = measure_deformable_memory(frames=_remembered_frames())

        self.assertEqual(output["probe_id"], "acwm_cloth_move_deformable_memory_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["deformable_memory_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_landmark_shape_and_recovery_failures(self) -> None:
        output = measure_deformable_memory(
            frames=[
                {"frame": 0, "landmark_error": 0.10, "shape_memory_loss": 0.20, "recovery_fraction": 0.40},
                {"frame": 1, "landmark_error": 0.20, "shape_memory_loss": 0.60, "recovery_fraction": 0.20},
            ],
        )

        self.assertEqual(
            output["flags"],
            [
                "high_mean_deformable_landmark_error",
                "final_deformable_landmark_miss",
                "shape_memory_loss",
                "weak_deformable_recovery",
            ],
        )
        self.assertLess(output["metrics"]["deformable_memory_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ClothMoveDeformableMemoryProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_deformable_memory(frames=_remembered_frames()[:1])
        with self.assertRaisesRegex(ClothMoveDeformableMemoryProbeError, "FRAME_INVALID"):
            measure_deformable_memory(
                frames=[
                    _remembered_frames()[0],
                    {"frame": 1, "landmark_error": -0.1, "shape_memory_loss": 0.1, "recovery_fraction": 0.8},
                ]
            )
        with self.assertRaisesRegex(ClothMoveDeformableMemoryProbeError, "EVIDENCE_REFS_INVALID"):
            measure_deformable_memory(frames=_remembered_frames(), evidence_refs=["bad-ref"])


def _remembered_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "landmark_error": 0.03, "shape_memory_loss": 0.05, "recovery_fraction": 0.60},
        {"frame": 1, "landmark_error": 0.04, "shape_memory_loss": 0.08, "recovery_fraction": 0.75},
    ]


if __name__ == "__main__":
    unittest.main()
