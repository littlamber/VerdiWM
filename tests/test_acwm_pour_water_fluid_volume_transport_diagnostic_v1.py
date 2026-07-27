from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_pour_water_fluid_volume_transport_diagnostic_v1 import (
    FluidTransportThresholds,
    FluidVolumeTransportProbeError,
    main,
    measure_fluid_volume_transport,
)


class FluidVolumeTransportProbeTests(unittest.TestCase):
    def test_measures_good_transport_without_verdict_exposure(self) -> None:
        output = measure_fluid_volume_transport(frames=_good_frames())

        self.assertEqual(output["probe_id"], "acwm_pour_water_fluid_volume_transport_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["retained_area_ratio"], 0.8)
        self.assertGreater(output["metrics"]["target_progress"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_volume_loss_poor_progress_and_leak(self) -> None:
        output = measure_fluid_volume_transport(
            frames=[
                {
                    "frame": 0,
                    "water_area": 100.0,
                    "water_centroid_x": 0.0,
                    "water_centroid_y": 0.0,
                    "target_centroid_x": 10.0,
                    "target_centroid_y": 0.0,
                    "spill_area": 0.0,
                },
                {
                    "frame": 1,
                    "water_area": 60.0,
                    "water_centroid_x": 1.0,
                    "water_centroid_y": 0.0,
                    "target_centroid_x": 10.0,
                    "target_centroid_y": 0.0,
                    "spill_area": 30.0,
                },
            ],
            thresholds=FluidTransportThresholds(
                min_retained_area_ratio=0.75,
                min_target_progress=0.25,
                max_spill_fraction=0.2,
            ),
        )

        self.assertEqual(
            output["flags"],
            ["low_volume_retention", "poor_transport_progress", "container_boundary_leak"],
        )
        self.assertLess(output["metrics"]["transport_score"], 0.7)

    def test_invalid_frame_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(FluidVolumeTransportProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_fluid_volume_transport(frames=_good_frames()[:1])
        with self.assertRaisesRegex(FluidVolumeTransportProbeError, "VALUE_INVALID"):
            measure_fluid_volume_transport(
                frames=[
                    _good_frames()[0],
                    {
                        "frame": 1,
                        "water_area": float("nan"),
                        "water_centroid_x": 1.0,
                        "water_centroid_y": 0.0,
                        "target_centroid_x": 10.0,
                        "target_centroid_y": 0.0,
                    },
                ]
            )

    def test_cli_prints_and_writes_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            measurements = root / "measurements.json"
            measurements.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "wmloop-fluid-transport-measurements",
                        "environment": "pour_water",
                        "frames": _good_frames(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--measurements", str(measurements)])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["state"], "measured")

            out_path = root / "probe-output.json"
            self.assertEqual(main(["--measurements", str(measurements), "--output", str(out_path)]), 0)
            self.assertTrue(out_path.is_file())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["signature"], "fluid_volume_transport")


def _good_frames() -> list[dict[str, float]]:
    return [
        {
            "frame": 0,
            "water_area": 100.0,
            "water_centroid_x": 0.0,
            "water_centroid_y": 0.0,
            "target_centroid_x": 10.0,
            "target_centroid_y": 0.0,
            "spill_area": 0.0,
        },
        {
            "frame": 1,
            "water_area": 95.0,
            "water_centroid_x": 4.0,
            "water_centroid_y": 0.0,
            "target_centroid_x": 10.0,
            "target_centroid_y": 0.0,
            "spill_area": 2.0,
        },
        {
            "frame": 2,
            "water_area": 92.0,
            "water_centroid_x": 7.0,
            "water_centroid_y": 0.0,
            "target_centroid_x": 10.0,
            "target_centroid_y": 0.0,
            "spill_area": 4.0,
        },
    ]


if __name__ == "__main__":
    unittest.main()
