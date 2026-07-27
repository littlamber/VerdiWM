from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_pour_water_container_boundary_leak_diagnostic_v1 import (
    ContainerBoundaryLeakProbeError,
    ContainerLeakThresholds,
    main,
    measure_container_boundary_leak,
)


class ContainerBoundaryLeakProbeTests(unittest.TestCase):
    def test_measures_containment_without_verdict_exposure(self) -> None:
        output = measure_container_boundary_leak(frames=_contained_frames())

        self.assertEqual(output["probe_id"], "acwm_pour_water_container_boundary_leak_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["final_containment_ratio"], 0.9)
        validate_document("diagnostic_probe_output", output)

    def test_flags_boundary_leak_and_low_containment(self) -> None:
        output = measure_container_boundary_leak(
            frames=[
                {"frame": 0, "in_container_area": 100.0, "outside_container_area": 2.0},
                {"frame": 1, "in_container_area": 82.0, "outside_container_area": 18.0},
                {"frame": 2, "in_container_area": 70.0, "outside_container_area": 30.0},
            ],
            thresholds=ContainerLeakThresholds(
                max_leak_fraction=0.12,
                max_leak_growth=0.08,
                min_final_containment_ratio=0.80,
            ),
        )

        self.assertEqual(
            output["flags"],
            ["boundary_leak_detected", "progressive_leak", "low_final_containment"],
        )
        self.assertGreater(output["metrics"]["max_leak_fraction"], 0.25)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ContainerBoundaryLeakProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_container_boundary_leak(frames=_contained_frames()[:1])
        with self.assertRaisesRegex(ContainerBoundaryLeakProbeError, "TOTAL_AREA_INVALID"):
            measure_container_boundary_leak(
                frames=[
                    _contained_frames()[0],
                    {"frame": 1, "in_container_area": 0.0, "outside_container_area": 0.0},
                ]
            )
        with self.assertRaisesRegex(ContainerBoundaryLeakProbeError, "EVIDENCE_REFS_INVALID"):
            measure_container_boundary_leak(frames=_contained_frames(), evidence_refs=["bad-ref"])

    def test_cli_prints_and_writes_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            measurements = root / "measurements.json"
            measurements.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "wmloop-container-boundary-leak-measurements",
                        "environment": "pour_water",
                        "frames": _contained_frames(),
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
            self.assertEqual(written["signature"], "container_boundary_leak")


def _contained_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "in_container_area": 100.0, "outside_container_area": 1.0},
        {"frame": 1, "in_container_area": 98.0, "outside_container_area": 2.0},
        {"frame": 2, "in_container_area": 95.0, "outside_container_area": 3.0},
    ]


if __name__ == "__main__":
    unittest.main()
