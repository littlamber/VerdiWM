from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_cube_rigid_pose_slip_diagnostic_v1 import (
    PushCubeRigidPoseSlipProbeError,
    measure_rigid_pose_slip,
)


class PushCubeRigidPoseSlipProbeTests(unittest.TestCase):
    def test_measures_stable_pose_without_verdict_exposure(self) -> None:
        output = measure_rigid_pose_slip(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_push_cube_rigid_pose_slip_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["pose_stability_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_no_contact_slip(self) -> None:
        output = measure_rigid_pose_slip(
            frames=[
                {"frame": 0, "cube_centroid_x": 0.00, "cube_centroid_y": 0.00, "cube_angle": 0.00, "contact_score": 0.0},
                {"frame": 1, "cube_centroid_x": 0.10, "cube_centroid_y": 0.00, "cube_angle": 0.20, "contact_score": 0.0},
            ],
        )

        self.assertEqual(output["flags"], ["no_contact_translation_slip", "no_contact_rotation_slip"])
        self.assertLess(output["metrics"]["pose_stability_score"], 0.1)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushCubeRigidPoseSlipProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_rigid_pose_slip(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(PushCubeRigidPoseSlipProbeError, "VALUE_INVALID"):
            measure_rigid_pose_slip(
                frames=[
                    _stable_frames()[0],
                    {
                        "frame": 1,
                        "cube_centroid_x": 0.0,
                        "cube_centroid_y": float("nan"),
                        "cube_angle": 0.0,
                        "contact_score": 0.0,
                    },
                ]
            )
        with self.assertRaisesRegex(PushCubeRigidPoseSlipProbeError, "EVIDENCE_REFS_INVALID"):
            measure_rigid_pose_slip(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "cube_centroid_x": 0.00, "cube_centroid_y": 0.00, "cube_angle": 0.00, "contact_score": 0.0},
        {"frame": 1, "cube_centroid_x": 0.01, "cube_centroid_y": 0.00, "cube_angle": 0.01, "contact_score": 0.0},
        {"frame": 2, "cube_centroid_x": 0.06, "cube_centroid_y": 0.00, "cube_angle": 0.02, "contact_score": 0.8},
    ]


if __name__ == "__main__":
    unittest.main()
