from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_rope_deformable_contact_diagnostic_v1 import (
    PushRopeDeformableContactProbeError,
    measure_deformable_contact,
)


class PushRopeDeformableContactProbeTests(unittest.TestCase):
    def test_measures_contact_response_without_verdict_exposure(self) -> None:
        output = measure_deformable_contact(frames=_contact_frames())

        self.assertEqual(output["probe_id"], "acwm_push_rope_deformable_contact_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["deformable_contact_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_missing_contact_and_weak_response(self) -> None:
        output = measure_deformable_contact(
            frames=[
                {
                    "frame": 0,
                    "rope_centroid_x": 0.00,
                    "rope_centroid_y": 0.00,
                    "pusher_centroid_x": 0.00,
                    "pusher_centroid_y": 0.00,
                    "contact_score": 0.10,
                },
                {
                    "frame": 1,
                    "rope_centroid_x": 0.03,
                    "rope_centroid_y": 0.00,
                    "pusher_centroid_x": 0.20,
                    "pusher_centroid_y": 0.00,
                    "contact_score": 0.20,
                },
            ],
        )

        self.assertEqual(
            output["flags"],
            ["missing_deformable_contact", "weak_rope_contact_response", "pre_contact_rope_drift"],
        )
        self.assertLess(output["metrics"]["deformable_contact_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushRopeDeformableContactProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_deformable_contact(frames=_contact_frames()[:1])
        with self.assertRaisesRegex(PushRopeDeformableContactProbeError, "VALUE_INVALID"):
            measure_deformable_contact(
                frames=[
                    _contact_frames()[0],
                    {
                        "frame": 1,
                        "rope_centroid_x": float("nan"),
                        "rope_centroid_y": 0.0,
                        "pusher_centroid_x": 0.2,
                        "pusher_centroid_y": 0.0,
                        "contact_score": 0.8,
                    },
                ]
            )
        with self.assertRaisesRegex(PushRopeDeformableContactProbeError, "EVIDENCE_REFS_INVALID"):
            measure_deformable_contact(frames=_contact_frames(), evidence_refs=["bad-ref"])


def _contact_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "rope_centroid_x": 0.00,
            "rope_centroid_y": 0.00,
            "pusher_centroid_x": 0.00,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.20,
        },
        {
            "frame": 1,
            "rope_centroid_x": 0.00,
            "rope_centroid_y": 0.00,
            "pusher_centroid_x": 0.04,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.80,
        },
        {
            "frame": 2,
            "rope_centroid_x": 0.06,
            "rope_centroid_y": 0.00,
            "pusher_centroid_x": 0.12,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.85,
        },
    ]


if __name__ == "__main__":
    unittest.main()
