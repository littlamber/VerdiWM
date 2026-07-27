from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_rope_endpoint_path_diagnostic_v1 import (
    PushRopeEndpointPathProbeError,
    measure_endpoint_path,
)


class PushRopeEndpointPathProbeTests(unittest.TestCase):
    def test_measures_endpoint_path_without_verdict_exposure(self) -> None:
        output = measure_endpoint_path(frames=_tracked_frames())

        self.assertEqual(output["probe_id"], "acwm_push_rope_endpoint_path_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["endpoint_path_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_endpoint_path_error_and_regression(self) -> None:
        output = measure_endpoint_path(
            frames=[
                {"frame": 0, "endpoint_x": 0.00, "endpoint_y": 0.00, "target_path_x": 0.02, "target_path_y": 0.00},
                {"frame": 1, "endpoint_x": 0.00, "endpoint_y": 0.00, "target_path_x": 0.16, "target_path_y": 0.00},
            ],
        )

        self.assertEqual(
            output["flags"],
            ["endpoint_path_error_high", "final_endpoint_path_miss", "endpoint_path_regression"],
        )
        self.assertLess(output["metrics"]["endpoint_path_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushRopeEndpointPathProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_endpoint_path(frames=_tracked_frames()[:1])
        with self.assertRaisesRegex(PushRopeEndpointPathProbeError, "VALUE_INVALID"):
            measure_endpoint_path(
                frames=[
                    _tracked_frames()[0],
                    {"frame": 1, "endpoint_x": float("nan"), "endpoint_y": 0.0, "target_path_x": 0.1, "target_path_y": 0.0},
                ]
            )
        with self.assertRaisesRegex(PushRopeEndpointPathProbeError, "EVIDENCE_REFS_INVALID"):
            measure_endpoint_path(frames=_tracked_frames(), evidence_refs=["bad-ref"])


def _tracked_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "endpoint_x": 0.00, "endpoint_y": 0.00, "target_path_x": 0.02, "target_path_y": 0.00},
        {"frame": 1, "endpoint_x": 0.10, "endpoint_y": 0.01, "target_path_x": 0.12, "target_path_y": 0.00},
    ]


if __name__ == "__main__":
    unittest.main()
