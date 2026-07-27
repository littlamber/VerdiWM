from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_cube_action_binding_diagnostic_v1 import (
    ActionBindingThresholds,
    PushCubeActionBindingProbeError,
    measure_action_binding,
)


class PushCubeActionBindingProbeTests(unittest.TestCase):
    def test_measures_action_binding_without_verdict_exposure(self) -> None:
        output = measure_action_binding(steps=_aligned_steps())

        self.assertEqual(output["probe_id"], "acwm_push_cube_action_binding_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["mean_response_ratio"], 0.1)
        self.assertGreater(output["metrics"]["mean_alignment_cosine"], 0.9)
        validate_document("diagnostic_probe_output", output)

    def test_flags_ignored_and_misaligned_action_response(self) -> None:
        output = measure_action_binding(
            steps=[
                {"frame": 0, "action_dx": 1.0, "action_dy": 0.0, "cube_delta_x": -0.01, "cube_delta_y": 0.0},
                {"frame": 1, "action_dx": 1.0, "action_dy": 0.0, "cube_delta_x": -0.02, "cube_delta_y": 0.0},
            ],
            thresholds=ActionBindingThresholds(min_action_step_count=3),
        )

        self.assertEqual(
            output["flags"],
            ["insufficient_action_coverage", "action_ignored", "action_response_misaligned"],
        )
        self.assertLess(output["metrics"]["action_binding_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushCubeActionBindingProbeError, "NO_ACTION_STEPS"):
            measure_action_binding(steps=[{"frame": 0, "action_dx": 0.0, "action_dy": 0.0, "cube_delta_x": 0.0, "cube_delta_y": 0.0}])
        with self.assertRaisesRegex(PushCubeActionBindingProbeError, "VALUE_INVALID"):
            measure_action_binding(
                steps=[
                    {"frame": 0, "action_dx": 1.0, "action_dy": 0.0, "cube_delta_x": float("nan"), "cube_delta_y": 0.0}
                ]
            )
        with self.assertRaisesRegex(PushCubeActionBindingProbeError, "EVIDENCE_REFS_INVALID"):
            measure_action_binding(steps=_aligned_steps(), evidence_refs=["bad-ref"])


def _aligned_steps() -> list[dict[str, float]]:
    return [
        {"frame": 0, "action_dx": 1.0, "action_dy": 0.0, "cube_delta_x": 0.20, "cube_delta_y": 0.00},
        {"frame": 1, "action_dx": 0.0, "action_dy": 1.0, "cube_delta_x": 0.00, "cube_delta_y": 0.15},
    ]


if __name__ == "__main__":
    unittest.main()
