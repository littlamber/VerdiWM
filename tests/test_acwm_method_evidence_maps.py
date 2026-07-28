from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_method_evidence_maps import (
    ENVIRONMENT_ORDER,
    REPO_ROOT,
    _affinity_rows,
    _irg_rows,
    _portableize,
    _probe_response_rows,
    _read_csv,
    _write_csv,
)


class AcwmMethodEvidenceMapsTests(unittest.TestCase):
    def test_probe_rows_preserve_jacobian_and_locality(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = root / "atlas"
            charts = atlas / "charts"
            charts.mkdir(parents=True)
            projection = atlas / "selector-input-projections.jsonl"
            projection.write_text("", encoding="utf-8")
            for environment in ENVIRONMENT_ORDER:
                (charts / f"{environment}.json").write_text(
                    json.dumps(
                        {
                            "chart_id": f"campaign:{environment}",
                            "outcome_names": ["psnr", "ssim", "negative_mse", "negative_masked_mse"],
                            "intervention_names": ["probe_a"],
                            "jacobian": [[1.0], [0.1], [0.01], [0.02]],
                            "locality_residuals": {"probe_a": 0.25},
                            "repeat_count": 3,
                        }
                    ),
                    encoding="utf-8",
                )
            rows = _probe_response_rows(
                {"probe_sources": [{"projection_path": str(projection), "path_order": ["probe_a"]}]}
            )
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["d_psnr_d_dose"], 1.0)
            self.assertTrue(rows[0]["locality_pass"])

    def test_affinity_marks_active_missing_successor_and_unmapped(self) -> None:
        rows, matrix = _affinity_rows(
            affinity={
                "primitives": {
                    "repair_a": {
                        "coverage_state": "covered",
                        "required_probe_paths": ["probe_a", "probe_missing"],
                        "successor_probe_axis": "probe_b",
                        "mechanism_rationale": "fixture",
                    }
                }
            },
            primitives=("repair_a", "repair_unmapped"),
            active_probes=("probe_a",),
        )
        relations = {(row["primitive"], row["probe"]): row["relation"] for row in rows}
        self.assertEqual(relations[("repair_a", "probe_a")], "required_active")
        self.assertEqual(relations[("repair_a", "probe_missing")], "required_missing")
        self.assertEqual(relations[("repair_a", "probe_b")], "candidate_successor")
        self.assertEqual(relations[("repair_unmapped", "unmapped")], "unmapped")
        self.assertEqual(matrix[0]["probe_a"], "required_active")

    def test_irg_projection_emits_two_finite_coordinates(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed in the unit-test environment")
        rows = []
        for index, environment in enumerate(ENVIRONMENT_ORDER):
            rows.append(
                {
                    "environment": environment,
                    "probe": "probe_a",
                    "d_psnr_d_dose": float(index),
                    "d_ssim_d_dose": float(index % 3),
                    "d_negative_mse_d_dose": float(index % 2),
                    "d_negative_masked_mse_d_dose": float(index * index),
                    "locality_pass": True,
                }
            )
        projection = _irg_rows(rows)
        self.assertEqual(len(projection), len(ENVIRONMENT_ORDER))
        self.assertTrue(all(math.isfinite(float(row["pc1"])) for row in projection))
        self.assertTrue(all(math.isfinite(float(row["pc2"])) for row in projection))

    def test_portableize_removes_repo_local_prefix(self) -> None:
        portable = _portableize(
            {"path": str(REPO_ROOT / "results/reports/example/manifest.json")}
        )
        self.assertEqual(portable["path"], "results/reports/example/manifest.json")

    def test_csv_round_trip_normalizes_crlf_to_lf(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.csv"
            destination = root / "destination.csv"
            source.write_bytes(b"environment,verdict\r\npush_cube,pass\r\n")

            _write_csv(destination, _read_csv(source))

            self.assertEqual(
                destination.read_bytes(),
                b"environment,verdict\npush_cube,pass\n",
            )


if __name__ == "__main__":
    unittest.main()
