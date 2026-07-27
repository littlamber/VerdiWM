from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_stack_cube_object_identity_diagnostic_v1 import (
    ObjectIdentityThresholds,
    StackCubeObjectIdentityProbeError,
    measure_object_identity,
)


class StackCubeObjectIdentityProbeTests(unittest.TestCase):
    def test_measures_stable_identity_without_verdict_exposure(self) -> None:
        output = measure_object_identity(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_stack_cube_object_identity_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["identity_score"], 0.8)
        validate_document("diagnostic_probe_output", output)

    def test_flags_identity_drop_and_swap(self) -> None:
        output = measure_object_identity(
            frames=[
                {"frame": 0, "top_identity_confidence": 0.90, "bottom_identity_confidence": 0.92, "id_swap_score": 0.02},
                {"frame": 1, "top_identity_confidence": 0.50, "bottom_identity_confidence": 0.55, "id_swap_score": 0.60},
            ],
            thresholds=ObjectIdentityThresholds(max_identity_drop_count=0),
        )

        self.assertEqual(output["flags"], ["low_identity_confidence", "object_identity_swap", "identity_unstable"])
        self.assertEqual(output["metrics"]["identity_drop_count"], 1)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(StackCubeObjectIdentityProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_object_identity(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(StackCubeObjectIdentityProbeError, "VALUE_INVALID"):
            measure_object_identity(
                frames=[
                    _stable_frames()[0],
                    {
                        "frame": 1,
                        "top_identity_confidence": float("nan"),
                        "bottom_identity_confidence": 0.9,
                        "id_swap_score": 0.1,
                    },
                ]
            )
        with self.assertRaisesRegex(StackCubeObjectIdentityProbeError, "EVIDENCE_REFS_INVALID"):
            measure_object_identity(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "top_identity_confidence": 0.92, "bottom_identity_confidence": 0.94, "id_swap_score": 0.01},
        {"frame": 1, "top_identity_confidence": 0.90, "bottom_identity_confidence": 0.93, "id_swap_score": 0.02},
    ]


if __name__ == "__main__":
    unittest.main()
