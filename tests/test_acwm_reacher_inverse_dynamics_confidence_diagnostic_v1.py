from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_reacher_inverse_dynamics_confidence_diagnostic_v1 import (
    InverseDynamicsConfidenceThresholds,
    ReacherInverseDynamicsConfidenceProbeError,
    measure_inverse_dynamics_confidence,
)


class ReacherInverseDynamicsConfidenceProbeTests(unittest.TestCase):
    def test_measures_inverse_dynamics_confidence_without_verdict_exposure(self) -> None:
        output = measure_inverse_dynamics_confidence(steps=_confident_steps())

        self.assertEqual(output["probe_id"], "acwm_reacher_inverse_dynamics_confidence_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["inverse_dynamics_confidence_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_noisy_low_confidence_inverse_dynamics(self) -> None:
        output = measure_inverse_dynamics_confidence(
            steps=[
                {
                    "frame": 0,
                    "target_action_x": 1.0,
                    "target_action_y": 0.0,
                    "predicted_action_x": -0.10,
                    "predicted_action_y": 0.0,
                    "confidence": 0.20,
                },
                {
                    "frame": 1,
                    "target_action_x": 1.0,
                    "target_action_y": 0.0,
                    "predicted_action_x": -0.20,
                    "predicted_action_y": 0.0,
                    "confidence": 0.10,
                },
            ],
            thresholds=InverseDynamicsConfidenceThresholds(min_active_step_count=3),
        )

        self.assertEqual(
            output["flags"],
            [
                "insufficient_active_actions",
                "high_inverse_dynamics_error",
                "low_inverse_dynamics_confidence",
                "inverse_dynamics_misaligned",
            ],
        )
        self.assertLess(output["metrics"]["inverse_dynamics_confidence_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ReacherInverseDynamicsConfidenceProbeError, "NO_ACTIVE_STEPS"):
            measure_inverse_dynamics_confidence(
                steps=[
                    {
                        "frame": 0,
                        "target_action_x": 0.0,
                        "target_action_y": 0.0,
                        "predicted_action_x": 0.0,
                        "predicted_action_y": 0.0,
                        "confidence": 0.5,
                    }
                ]
            )
        with self.assertRaisesRegex(ReacherInverseDynamicsConfidenceProbeError, "STEP_INVALID"):
            measure_inverse_dynamics_confidence(
                steps=[
                    {
                        "frame": 0,
                        "target_action_x": 1.0,
                        "target_action_y": 0.0,
                        "predicted_action_x": 1.0,
                        "predicted_action_y": 0.0,
                        "confidence": 2.0,
                    }
                ]
            )
        with self.assertRaisesRegex(ReacherInverseDynamicsConfidenceProbeError, "EVIDENCE_REFS_INVALID"):
            measure_inverse_dynamics_confidence(steps=_confident_steps(), evidence_refs=["bad-ref"])


def _confident_steps() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "target_action_x": 1.0,
            "target_action_y": 0.0,
            "predicted_action_x": 0.98,
            "predicted_action_y": 0.01,
            "confidence": 0.90,
        },
        {
            "frame": 1,
            "target_action_x": 0.0,
            "target_action_y": 1.0,
            "predicted_action_x": 0.01,
            "predicted_action_y": 0.96,
            "confidence": 0.86,
        },
    ]


if __name__ == "__main__":
    unittest.main()
