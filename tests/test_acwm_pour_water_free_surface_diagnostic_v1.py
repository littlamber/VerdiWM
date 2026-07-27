from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_pour_water_free_surface_diagnostic_v1 import (
    PourWaterFreeSurfaceProbeError,
    measure_free_surface,
)


class PourWaterFreeSurfaceProbeTests(unittest.TestCase):
    def test_measures_stable_surface_without_verdict_exposure(self) -> None:
        output = measure_free_surface(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_pour_water_free_surface_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["surface_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_unstable_surface(self) -> None:
        output = measure_free_surface(
            frames=[
                {"frame": 0, "surface_area": 100.0, "surface_centroid_y": 0.10, "surface_roughness": 0.10},
                {"frame": 1, "surface_area": 60.0, "surface_centroid_y": 0.45, "surface_roughness": 0.40},
            ],
        )

        self.assertEqual(output["flags"], ["surface_unstable", "surface_area_flicker", "surface_centroid_oscillation"])
        self.assertLess(output["metrics"]["surface_score"], 0.1)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PourWaterFreeSurfaceProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_free_surface(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(PourWaterFreeSurfaceProbeError, "INITIAL_AREA_INVALID"):
            measure_free_surface(
                frames=[
                    {"frame": 0, "surface_area": 0.0, "surface_centroid_y": 0.0, "surface_roughness": 0.0},
                    {"frame": 1, "surface_area": 1.0, "surface_centroid_y": 0.0, "surface_roughness": 0.0},
                ]
            )
        with self.assertRaisesRegex(PourWaterFreeSurfaceProbeError, "EVIDENCE_REFS_INVALID"):
            measure_free_surface(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "surface_area": 100.0, "surface_centroid_y": 0.10, "surface_roughness": 0.04},
        {"frame": 1, "surface_area": 98.0, "surface_centroid_y": 0.12, "surface_roughness": 0.05},
    ]


if __name__ == "__main__":
    unittest.main()
