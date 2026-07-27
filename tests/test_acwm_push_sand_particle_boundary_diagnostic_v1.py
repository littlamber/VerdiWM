from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_sand_particle_boundary_diagnostic_v1 import (
    ParticleBoundaryThresholds,
    PushSandParticleBoundaryProbeError,
    measure_particle_boundary,
)


class PushSandParticleBoundaryProbeTests(unittest.TestCase):
    def test_measures_boundary_integrity_without_verdict_exposure(self) -> None:
        output = measure_particle_boundary(frames=_contained_frames())

        self.assertEqual(output["probe_id"], "acwm_push_sand_particle_boundary_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["boundary_integrity_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_particle_escape_and_weak_boundary_contact(self) -> None:
        output = measure_particle_boundary(
            frames=[
                {
                    "frame": 0,
                    "inside_boundary_mass": 96.0,
                    "outside_boundary_mass": 4.0,
                    "boundary_contact_score": 0.10,
                },
                {
                    "frame": 1,
                    "inside_boundary_mass": 72.0,
                    "outside_boundary_mass": 28.0,
                    "boundary_contact_score": 0.15,
                },
            ],
            thresholds=ParticleBoundaryThresholds(min_mean_boundary_contact_score=0.25),
        )

        self.assertEqual(
            output["flags"],
            [
                "particle_boundary_escape",
                "progressive_boundary_escape",
                "low_boundary_retention",
                "weak_boundary_contact",
            ],
        )
        self.assertLess(output["metrics"]["boundary_integrity_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushSandParticleBoundaryProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_particle_boundary(frames=_contained_frames()[:1])
        with self.assertRaisesRegex(PushSandParticleBoundaryProbeError, "TOTAL_MASS_INVALID"):
            measure_particle_boundary(
                frames=[
                    {
                        "frame": 0,
                        "inside_boundary_mass": 0.0,
                        "outside_boundary_mass": 0.0,
                        "boundary_contact_score": 0.5,
                    },
                    {
                        "frame": 1,
                        "inside_boundary_mass": 1.0,
                        "outside_boundary_mass": 0.0,
                        "boundary_contact_score": 0.5,
                    },
                ]
            )
        with self.assertRaisesRegex(PushSandParticleBoundaryProbeError, "EVIDENCE_REFS_INVALID"):
            measure_particle_boundary(frames=_contained_frames(), evidence_refs=["bad-ref"])


def _contained_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "inside_boundary_mass": 98.0,
            "outside_boundary_mass": 2.0,
            "boundary_contact_score": 0.40,
        },
        {
            "frame": 1,
            "inside_boundary_mass": 96.0,
            "outside_boundary_mass": 3.0,
            "boundary_contact_score": 0.45,
        },
    ]


if __name__ == "__main__":
    unittest.main()
