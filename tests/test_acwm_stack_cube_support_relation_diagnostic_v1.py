from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_stack_cube_support_relation_diagnostic_v1 import (
    StackCubeSupportRelationProbeError,
    SupportRelationThresholds,
    main,
    measure_support_relation,
)


class StackCubeSupportRelationProbeTests(unittest.TestCase):
    def test_measures_stable_support_without_verdict_exposure(self) -> None:
        output = measure_support_relation(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_stack_cube_support_relation_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["stable_frame_fraction"], 0.9)
        self.assertGreater(output["metrics"]["support_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_lost_support_relation(self) -> None:
        output = measure_support_relation(
            frames=[
                {"frame": 0, "support_overlap_ratio": 0.80, "vertical_gap": 0.02},
                {"frame": 1, "support_overlap_ratio": 0.40, "vertical_gap": 0.12},
                {"frame": 2, "support_overlap_ratio": 0.30, "vertical_gap": 0.18},
            ],
            thresholds=SupportRelationThresholds(
                min_support_overlap_ratio=0.55,
                max_vertical_gap=0.08,
                min_stable_frame_fraction=0.75,
            ),
        )

        self.assertEqual(
            output["flags"],
            [
                "support_relation_unstable",
                "low_final_support_overlap",
                "vertical_gap_unstable",
                "support_relation_lost",
            ],
        )
        self.assertEqual(output["metrics"]["support_loss_count"], 1)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(StackCubeSupportRelationProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_support_relation(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(StackCubeSupportRelationProbeError, "VALUE_INVALID"):
            measure_support_relation(
                frames=[
                    _stable_frames()[0],
                    {"frame": 1, "support_overlap_ratio": float("inf"), "vertical_gap": 0.0},
                ]
            )
        with self.assertRaisesRegex(StackCubeSupportRelationProbeError, "EVIDENCE_REFS_INVALID"):
            measure_support_relation(frames=_stable_frames(), evidence_refs=["bad-ref"])

    def test_cli_prints_and_writes_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            measurements = root / "measurements.json"
            measurements.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "wmloop-stack-cube-support-relation-measurements",
                        "environment": "stack_cube",
                        "frames": _stable_frames(),
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
            self.assertEqual(written["signature"], "support_relation")


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "support_overlap_ratio": 0.72, "vertical_gap": 0.02},
        {"frame": 1, "support_overlap_ratio": 0.78, "vertical_gap": 0.01},
        {"frame": 2, "support_overlap_ratio": 0.75, "vertical_gap": 0.02},
    ]


if __name__ == "__main__":
    unittest.main()
