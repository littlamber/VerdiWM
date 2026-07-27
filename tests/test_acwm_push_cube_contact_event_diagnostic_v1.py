from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_cube_contact_event_diagnostic_v1 import (
    ContactEventThresholds,
    PushCubeContactEventProbeError,
    main,
    measure_contact_event,
)


class PushCubeContactEventProbeTests(unittest.TestCase):
    def test_measures_contact_response_without_verdict_exposure(self) -> None:
        output = measure_contact_event(frames=_responsive_frames())

        self.assertEqual(output["probe_id"], "acwm_push_cube_contact_event_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertEqual(output["metrics"]["first_contact_frame"], 1)
        self.assertGreater(output["metrics"]["post_contact_cube_displacement"], 0.05)
        validate_document("diagnostic_probe_output", output)

    def test_flags_missing_or_weak_contact_event(self) -> None:
        output = measure_contact_event(
            frames=[
                {
                    "frame": 0,
                    "cube_centroid_x": 0.00,
                    "cube_centroid_y": 0.00,
                    "pusher_centroid_x": -1.00,
                    "pusher_centroid_y": 0.00,
                    "contact_score": 0.10,
                },
                {
                    "frame": 1,
                    "cube_centroid_x": 0.10,
                    "cube_centroid_y": 0.00,
                    "pusher_centroid_x": -0.50,
                    "pusher_centroid_y": 0.00,
                    "contact_score": 0.20,
                },
                {
                    "frame": 2,
                    "cube_centroid_x": 0.11,
                    "cube_centroid_y": 0.00,
                    "pusher_centroid_x": 0.00,
                    "pusher_centroid_y": 0.00,
                    "contact_score": 0.30,
                },
            ],
            thresholds=ContactEventThresholds(max_pre_contact_drift=0.02),
        )

        self.assertEqual(
            output["flags"],
            ["missing_contact_event", "weak_contact_response", "pre_contact_cube_drift"],
        )
        self.assertLess(output["metrics"]["event_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushCubeContactEventProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_contact_event(frames=_responsive_frames()[:1])
        with self.assertRaisesRegex(PushCubeContactEventProbeError, "VALUE_INVALID"):
            measure_contact_event(
                frames=[
                    _responsive_frames()[0],
                    {
                        "frame": 1,
                        "cube_centroid_x": float("nan"),
                        "cube_centroid_y": 0.0,
                        "pusher_centroid_x": 0.0,
                        "pusher_centroid_y": 0.0,
                        "contact_score": 0.8,
                    },
                ]
            )
        with self.assertRaisesRegex(PushCubeContactEventProbeError, "EVIDENCE_REFS_INVALID"):
            measure_contact_event(frames=_responsive_frames(), evidence_refs=["not-a-cas-ref"])

    def test_cli_prints_and_writes_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            measurements = root / "measurements.json"
            measurements.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "wmloop-push-cube-contact-event-measurements",
                        "environment": "push_cube",
                        "frames": _responsive_frames(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--measurements", str(measurements)]), 0)
            self.assertEqual(json.loads(output.getvalue())["state"], "measured")

            out_path = root / "probe-output.json"
            self.assertEqual(main(["--measurements", str(measurements), "--output", str(out_path)]), 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["signature"], "contact_event")


def _responsive_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "cube_centroid_x": 0.00,
            "cube_centroid_y": 0.00,
            "pusher_centroid_x": -0.50,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.10,
        },
        {
            "frame": 1,
            "cube_centroid_x": 0.00,
            "cube_centroid_y": 0.00,
            "pusher_centroid_x": -0.03,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.80,
        },
        {
            "frame": 2,
            "cube_centroid_x": 0.08,
            "cube_centroid_y": 0.00,
            "pusher_centroid_x": 0.05,
            "pusher_centroid_y": 0.00,
            "contact_score": 0.90,
        },
    ]


if __name__ == "__main__":
    unittest.main()
