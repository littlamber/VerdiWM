from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_cloth_move_surface_fold_diagnostic_v1 import (
    ClothMoveSurfaceFoldProbeError,
    measure_surface_fold,
)


class ClothMoveSurfaceFoldProbeTests(unittest.TestCase):
    def test_measures_low_fold_without_verdict_exposure(self) -> None:
        output = measure_surface_fold(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_cloth_move_surface_fold_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["fold_stability_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_large_sharp_progressive_fold(self) -> None:
        output = measure_surface_fold(
            frames=[
                {"frame": 0, "fold_area_fraction": 0.05, "fold_sharpness": 0.20},
                {"frame": 1, "fold_area_fraction": 0.55, "fold_sharpness": 0.70},
            ],
        )

        self.assertEqual(output["flags"], ["large_surface_fold", "sharp_surface_fold", "progressive_fold_growth"])
        self.assertLess(output["metrics"]["fold_stability_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ClothMoveSurfaceFoldProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_surface_fold(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(ClothMoveSurfaceFoldProbeError, "FRAME_INVALID"):
            measure_surface_fold(
                frames=[
                    _stable_frames()[0],
                    {"frame": 1, "fold_area_fraction": 2.0, "fold_sharpness": 0.1},
                ]
            )
        with self.assertRaisesRegex(ClothMoveSurfaceFoldProbeError, "EVIDENCE_REFS_INVALID"):
            measure_surface_fold(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "fold_area_fraction": 0.08, "fold_sharpness": 0.10},
        {"frame": 1, "fold_area_fraction": 0.12, "fold_sharpness": 0.12},
    ]


if __name__ == "__main__":
    unittest.main()
