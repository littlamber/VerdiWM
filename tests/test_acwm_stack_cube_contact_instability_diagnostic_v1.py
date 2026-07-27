from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_stack_cube_contact_instability_diagnostic_v1 import (
    ContactInstabilityThresholds,
    StackCubeContactInstabilityProbeError,
    measure_contact_instability,
)


class StackCubeContactInstabilityProbeTests(unittest.TestCase):
    def test_measures_stable_contact_without_verdict_exposure(self) -> None:
        output = measure_contact_instability(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_stack_cube_contact_instability_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertLess(output["metrics"]["instability_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_contact_flicker_and_jitter(self) -> None:
        output = measure_contact_instability(
            frames=[
                {"frame": 0, "top_cube_centroid_x": 0.00, "top_cube_centroid_y": 0.00, "contact_score": 0.80},
                {"frame": 1, "top_cube_centroid_x": 0.00, "top_cube_centroid_y": 0.00, "contact_score": 0.10},
                {"frame": 2, "top_cube_centroid_x": 0.00, "top_cube_centroid_y": 0.00, "contact_score": 0.90},
                {"frame": 3, "top_cube_centroid_x": 0.08, "top_cube_centroid_y": 0.00, "contact_score": 0.90},
                {"frame": 4, "top_cube_centroid_x": 0.08, "top_cube_centroid_y": 0.00, "contact_score": 0.10},
            ],
            thresholds=ContactInstabilityThresholds(min_contact_active_fraction=0.75, max_contact_transition_count=2),
        )

        self.assertEqual(output["flags"], ["weak_stack_contact", "contact_flicker", "high_contact_pose_jitter"])
        self.assertGreater(output["metrics"]["instability_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(StackCubeContactInstabilityProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_contact_instability(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(StackCubeContactInstabilityProbeError, "VALUE_INVALID"):
            measure_contact_instability(
                frames=[
                    _stable_frames()[0],
                    {"frame": 1, "top_cube_centroid_x": float("nan"), "top_cube_centroid_y": 0.0, "contact_score": 0.8},
                ]
            )
        with self.assertRaisesRegex(StackCubeContactInstabilityProbeError, "EVIDENCE_REFS_INVALID"):
            measure_contact_instability(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "top_cube_centroid_x": 0.00, "top_cube_centroid_y": 0.00, "contact_score": 0.80},
        {"frame": 1, "top_cube_centroid_x": 0.01, "top_cube_centroid_y": 0.00, "contact_score": 0.85},
        {"frame": 2, "top_cube_centroid_x": 0.02, "top_cube_centroid_y": 0.00, "contact_score": 0.82},
    ]


if __name__ == "__main__":
    unittest.main()
