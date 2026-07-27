from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_reacher_target_conditioning_diagnostic_v1 import (
    ReacherTargetConditioningProbeError,
    measure_target_conditioning,
)


class ReacherTargetConditioningProbeTests(unittest.TestCase):
    def test_measures_target_progress_without_verdict_exposure(self) -> None:
        output = measure_target_conditioning(frames=_targeted_frames())

        self.assertEqual(output["probe_id"], "acwm_reacher_target_conditioning_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["target_conditioning_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_target_miss_and_regression(self) -> None:
        output = measure_target_conditioning(
            frames=[
                {"frame": 0, "target_x": 1.0, "target_y": 0.0, "endpoint_x": 0.90, "endpoint_y": 0.00},
                {"frame": 1, "target_x": 1.0, "target_y": 0.0, "endpoint_x": 0.82, "endpoint_y": 0.00},
            ],
        )

        self.assertEqual(output["flags"], ["weak_target_progress", "final_target_miss", "target_distance_regression"])
        self.assertLess(output["metrics"]["target_conditioning_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ReacherTargetConditioningProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_target_conditioning(frames=_targeted_frames()[:1])
        with self.assertRaisesRegex(ReacherTargetConditioningProbeError, "VALUE_INVALID"):
            measure_target_conditioning(
                frames=[
                    _targeted_frames()[0],
                    {"frame": 1, "target_x": float("nan"), "target_y": 0.0, "endpoint_x": 0.1, "endpoint_y": 0.0},
                ]
            )
        with self.assertRaisesRegex(ReacherTargetConditioningProbeError, "EVIDENCE_REFS_INVALID"):
            measure_target_conditioning(frames=_targeted_frames(), evidence_refs=["bad-ref"])


def _targeted_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "target_x": 1.0, "target_y": 0.0, "endpoint_x": 0.80, "endpoint_y": 0.00},
        {"frame": 1, "target_x": 1.0, "target_y": 0.0, "endpoint_x": 0.95, "endpoint_y": 0.01},
    ]


if __name__ == "__main__":
    unittest.main()
