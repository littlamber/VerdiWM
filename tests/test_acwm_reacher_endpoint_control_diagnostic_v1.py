from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_reacher_endpoint_control_diagnostic_v1 import (
    EndpointControlThresholds,
    ReacherEndpointControlProbeError,
    measure_endpoint_control,
)


class ReacherEndpointControlProbeTests(unittest.TestCase):
    def test_measures_endpoint_response_without_verdict_exposure(self) -> None:
        output = measure_endpoint_control(steps=_controlled_steps())

        self.assertEqual(output["probe_id"], "acwm_reacher_endpoint_control_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["endpoint_control_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_underresponsive_misaligned_endpoint(self) -> None:
        output = measure_endpoint_control(
            steps=[
                {"frame": 0, "command_dx": 1.0, "command_dy": 0.0, "endpoint_delta_x": -0.01, "endpoint_delta_y": 0.0},
                {"frame": 1, "command_dx": 1.0, "command_dy": 0.0, "endpoint_delta_x": -0.02, "endpoint_delta_y": 0.0},
            ],
            thresholds=EndpointControlThresholds(min_command_step_count=3),
        )

        self.assertEqual(
            output["flags"],
            ["insufficient_endpoint_commands", "endpoint_underresponsive", "endpoint_response_misaligned"],
        )
        self.assertLess(output["metrics"]["endpoint_control_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ReacherEndpointControlProbeError, "NO_COMMAND_STEPS"):
            measure_endpoint_control(
                steps=[{"frame": 0, "command_dx": 0.0, "command_dy": 0.0, "endpoint_delta_x": 0.0, "endpoint_delta_y": 0.0}]
            )
        with self.assertRaisesRegex(ReacherEndpointControlProbeError, "VALUE_INVALID"):
            measure_endpoint_control(
                steps=[
                    {
                        "frame": 0,
                        "command_dx": 1.0,
                        "command_dy": 0.0,
                        "endpoint_delta_x": float("nan"),
                        "endpoint_delta_y": 0.0,
                    }
                ]
            )
        with self.assertRaisesRegex(ReacherEndpointControlProbeError, "EVIDENCE_REFS_INVALID"):
            measure_endpoint_control(steps=_controlled_steps(), evidence_refs=["bad-ref"])


def _controlled_steps() -> list[dict[str, float]]:
    return [
        {"frame": 0, "command_dx": 1.0, "command_dy": 0.0, "endpoint_delta_x": 0.20, "endpoint_delta_y": 0.00},
        {"frame": 1, "command_dx": 0.0, "command_dy": 1.0, "endpoint_delta_x": 0.00, "endpoint_delta_y": 0.16},
    ]


if __name__ == "__main__":
    unittest.main()
