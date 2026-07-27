from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_sand_granular_frontier_diagnostic_v1 import (
    PushSandGranularFrontierProbeError,
    measure_granular_frontier,
)


class PushSandGranularFrontierProbeTests(unittest.TestCase):
    def test_measures_frontier_motion_without_verdict_exposure(self) -> None:
        output = measure_granular_frontier(frames=_moving_frames())

        self.assertEqual(output["probe_id"], "acwm_push_sand_granular_frontier_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["granular_frontier_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_fragmented_stalled_frontier(self) -> None:
        output = measure_granular_frontier(
            frames=[
                {
                    "frame": 0,
                    "frontier_position": 0.10,
                    "active_sand_area": 100.0,
                    "frontier_roughness": 0.20,
                    "frontier_speed": 0.001,
                },
                {
                    "frame": 1,
                    "frontier_position": 0.11,
                    "active_sand_area": 40.0,
                    "frontier_roughness": 0.60,
                    "frontier_speed": 0.001,
                },
            ],
        )

        self.assertEqual(
            output["flags"],
            [
                "weak_granular_frontier_motion",
                "active_area_collapse",
                "frontier_fragmentation",
                "stalled_frontier_velocity",
            ],
        )
        self.assertLess(output["metrics"]["granular_frontier_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushSandGranularFrontierProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_granular_frontier(frames=_moving_frames()[:1])
        with self.assertRaisesRegex(PushSandGranularFrontierProbeError, "INITIAL_AREA_INVALID"):
            measure_granular_frontier(
                frames=[
                    {
                        "frame": 0,
                        "frontier_position": 0.0,
                        "active_sand_area": 0.0,
                        "frontier_roughness": 0.1,
                        "frontier_speed": 0.0,
                    },
                    {
                        "frame": 1,
                        "frontier_position": 0.1,
                        "active_sand_area": 1.0,
                        "frontier_roughness": 0.1,
                        "frontier_speed": 0.1,
                    },
                ]
            )
        with self.assertRaisesRegex(PushSandGranularFrontierProbeError, "EVIDENCE_REFS_INVALID"):
            measure_granular_frontier(frames=_moving_frames(), evidence_refs=["bad-ref"])


def _moving_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "frontier_position": 0.10,
            "active_sand_area": 100.0,
            "frontier_roughness": 0.08,
            "frontier_speed": 0.006,
        },
        {
            "frame": 1,
            "frontier_position": 0.16,
            "active_sand_area": 92.0,
            "frontier_roughness": 0.10,
            "frontier_speed": 0.007,
        },
    ]


if __name__ == "__main__":
    unittest.main()
