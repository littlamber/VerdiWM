from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_paper_primitive_matrix import export_paper_primitive_matrix


class AcwmPaperPrimitiveMatrixTests(unittest.TestCase):
    def test_exports_pass_fail_excluded_and_checkpoint_sensitive_cells(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            labels = [
                self._label(evidence, "push_cube", "repair", 800, True, 0.4),
                self._label(evidence, "push_cube", "repair", 1000, False, -0.2),
                self._label(evidence, "stack_cube", "repair", 512, False, -1.0),
                self._label(evidence, "reacher", "hook_only", 512, None, -0.1),
            ]
            index = root / "labels.json"
            index.write_text(json.dumps({"labels": labels}), encoding="utf-8")

            report = export_paper_primitive_matrix(effect_label_index=index, output_root=root / "out")

            self.assertEqual(report["counts"]["positive_cell_count"], 1)
            self.assertEqual(report["counts"]["failed_cell_count"], 1)
            self.assertEqual(report["counts"]["excluded_cell_count"], 1)
            with (root / "out" / "tables" / "all_environment_primitive_cells.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            push = next(row for row in rows if row["environment"] == "push_cube")
            self.assertEqual(push["verdict"], "pass")
            self.assertEqual(push["stability"], "checkpoint_sensitive")
            self.assertEqual(push["selected_checkpoint_step"], "800")
            svg = (root / "out" / "figures" / "environment_primitive_gate_heatmap.svg").read_text()
            self.assertIn("push_cube", svg)
            self.assertIn("+0.40", svg)

    def _label(
        self,
        evidence_root: Path,
        environment: str,
        primitive: str,
        step: int,
        positive: bool | None,
        delta_psnr: float,
    ) -> dict[str, object]:
        path = evidence_root / f"{environment}-{primitive}-step{step}.json"
        path.write_text(
            json.dumps(
                {
                    "candidate_checkpoint": f"/tmp/relative_step_{step:06d}.pt",
                    "candidate_checkpoint_sha256": str(step) * 8,
                }
            ),
            encoding="utf-8",
        )
        admissible = isinstance(positive, bool)
        return {
            "environment": environment,
            "primitive": primitive,
            "positive": positive,
            "settled": admissible,
            "selector_admissible": admissible,
            "selector_exclusion_reason": None if admissible else "runtime_only",
            "seed": 7,
            "label_id": f"{environment}-{primitive}-step{step}",
            "evidence_ref": str(path),
            "delta_candidate_minus_baseline": {
                "psnr": delta_psnr,
                "ssim": 0.01 if positive else -0.01,
                "mse": -0.001 if positive else 0.001,
                "masked_mse": -0.002 if positive else 0.002,
            },
            "official_gate_checks": {
                "psnr_strictly_improves": bool(positive),
                "ssim_does_not_regress": bool(positive),
                "mse_does_not_regress": bool(positive),
                "masked_mse_does_not_regress": bool(positive),
            },
        }


if __name__ == "__main__":
    unittest.main()
