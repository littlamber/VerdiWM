from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_sand_mass_redistribution_diagnostic_v1 import (
    PushSandMassRedistributionProbeError,
    measure_mass_redistribution,
)


class PushSandMassRedistributionProbeTests(unittest.TestCase):
    def test_measures_mass_transport_without_verdict_exposure(self) -> None:
        output = measure_mass_redistribution(frames=_redistributed_frames())

        self.assertEqual(output["probe_id"], "acwm_push_sand_mass_redistribution_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["redistribution_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_nonconserved_off_axis_weak_transport(self) -> None:
        output = measure_mass_redistribution(
            frames=[
                {
                    "frame": 0,
                    "total_mass": 100.0,
                    "mass_centroid_x": 0.0,
                    "mass_centroid_y": 0.0,
                    "redistributed_mass_fraction": 0.01,
                },
                {
                    "frame": 1,
                    "total_mass": 86.0,
                    "mass_centroid_x": 0.01,
                    "mass_centroid_y": 0.10,
                    "redistributed_mass_fraction": 0.02,
                },
            ],
        )

        self.assertEqual(
            output["flags"],
            [
                "mass_not_conserved",
                "weak_centroid_transport",
                "low_redistributed_mass",
                "off_axis_mass_drift",
            ],
        )
        self.assertLess(output["metrics"]["redistribution_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushSandMassRedistributionProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_mass_redistribution(frames=_redistributed_frames()[:1])
        with self.assertRaisesRegex(PushSandMassRedistributionProbeError, "INITIAL_MASS_INVALID"):
            measure_mass_redistribution(
                frames=[
                    {
                        "frame": 0,
                        "total_mass": 0.0,
                        "mass_centroid_x": 0.0,
                        "mass_centroid_y": 0.0,
                        "redistributed_mass_fraction": 0.0,
                    },
                    {
                        "frame": 1,
                        "total_mass": 1.0,
                        "mass_centroid_x": 0.1,
                        "mass_centroid_y": 0.0,
                        "redistributed_mass_fraction": 0.1,
                    },
                ]
            )
        with self.assertRaisesRegex(PushSandMassRedistributionProbeError, "EXPECTED_MOTION_INVALID"):
            measure_mass_redistribution(frames=_redistributed_frames(), expected_motion=(0.0, 0.0))
        with self.assertRaisesRegex(PushSandMassRedistributionProbeError, "EVIDENCE_REFS_INVALID"):
            measure_mass_redistribution(frames=_redistributed_frames(), evidence_refs=["bad-ref"])


def _redistributed_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "total_mass": 100.0,
            "mass_centroid_x": 0.00,
            "mass_centroid_y": 0.00,
            "redistributed_mass_fraction": 0.02,
        },
        {
            "frame": 1,
            "total_mass": 99.0,
            "mass_centroid_x": 0.06,
            "mass_centroid_y": 0.01,
            "redistributed_mass_fraction": 0.14,
        },
    ]


if __name__ == "__main__":
    unittest.main()
