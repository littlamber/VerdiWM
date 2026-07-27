from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_cloth_move_cloth_identity_drift_diagnostic_v1 import (
    ClothMoveIdentityDriftProbeError,
    measure_cloth_identity_drift,
)


class ClothMoveIdentityDriftProbeTests(unittest.TestCase):
    def test_measures_stable_identity_without_verdict_exposure(self) -> None:
        output = measure_cloth_identity_drift(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_cloth_move_cloth_identity_drift_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["cloth_identity_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_identity_texture_and_mask_drift(self) -> None:
        output = measure_cloth_identity_drift(
            frames=[
                {"frame": 0, "identity_confidence": 0.90, "texture_drift": 0.05, "mask_iou_to_initial": 0.95},
                {"frame": 1, "identity_confidence": 0.40, "texture_drift": 0.60, "mask_iou_to_initial": 0.40},
            ],
        )

        self.assertEqual(
            output["flags"],
            ["low_cloth_identity_confidence", "cloth_texture_drift", "cloth_mask_identity_drift"],
        )
        self.assertLess(output["metrics"]["cloth_identity_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ClothMoveIdentityDriftProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_cloth_identity_drift(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(ClothMoveIdentityDriftProbeError, "FRAME_INVALID"):
            measure_cloth_identity_drift(
                frames=[
                    _stable_frames()[0],
                    {"frame": 1, "identity_confidence": 0.8, "texture_drift": -0.1, "mask_iou_to_initial": 0.9},
                ]
            )
        with self.assertRaisesRegex(ClothMoveIdentityDriftProbeError, "EVIDENCE_REFS_INVALID"):
            measure_cloth_identity_drift(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "identity_confidence": 0.92, "texture_drift": 0.04, "mask_iou_to_initial": 0.98},
        {"frame": 1, "identity_confidence": 0.88, "texture_drift": 0.06, "mask_iou_to_initial": 0.94},
    ]


if __name__ == "__main__":
    unittest.main()
