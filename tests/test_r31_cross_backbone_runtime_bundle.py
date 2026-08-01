from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.export.r31_cross_backbone_runtime_bundle import export_r31_runtime_bundle


class R31CrossBackboneRuntimeBundleTests(unittest.TestCase):
    def test_exports_fixture_canaries_without_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture"
            paths = self._write_fixture(fixture)
            manifest = export_r31_runtime_bundle(
                ctrl_roots=paths["ctrl_roots"],
                cosmos_roots=paths["cosmos_roots"],
                ctrl_compile_report=paths["ctrl_compile_report"],
                cosmos_compile_report=paths["cosmos_compile_report"],
                reuse_audit=paths["reuse_audit"],
                output_root=Path(temporary) / "bundle",
            )
            self.assertEqual(manifest["state"], "ready")
            payload = (Path(temporary) / "bundle/bundle.json").read_text(encoding="utf-8")
            self.assertNotIn("/" + "mnt" + "/", payload)
            report = json.loads(payload)
            self.assertTrue(report["reuse_audit"]["r31_exact_portability_ready"])
            self.assertFalse(report["reuse_audit"]["cold_vs_warm_identifiable"])
            self.assertEqual(
                {row["dose"] for row in report["runtime"]["ctrl_world"]["metrics"]},
                {-0.05, 0.0, 0.05},
            )

    @staticmethod
    def _write_fixture(root: Path) -> dict[str, object]:
        def write(path: Path, payload: dict[str, object]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

        metric_names = {
            "rollout_video_psnr": 23.0,
            "rollout_video_l1": 0.03,
            "final_frame_mae": 0.05,
        }
        ctrl_roots = []
        for index, doses in enumerate(((-0.05,), (0.0, 0.05))):
            campaign = root / f"ctrl-{index}"
            rows = []
            audits = []
            for dose in doses:
                receipt = campaign / "receipts" / f"dose-{dose:+.2f}.json"
                write(receipt, {"metrics": metric_names})
                rows.append({"dose": dose, "receipt_ref": str(receipt)})
                audits.append(
                    {
                        "dose": dose,
                        "invocation_count": 4,
                        "maximum_temporal_mean_abs_error": 0.001,
                        "maximum_temporal_mean_tolerance": 0.01,
                    }
                )
            write(campaign / "receipt-index.json", {"rows": rows})
            write(campaign / "runtime-probe-audit.json", {"state": "passed", "rows": audits})
            ctrl_roots.append(campaign)

        cosmos_roots = []
        for index, doses in enumerate(((-0.05,), (0.0, 0.05))):
            campaign = root / f"cosmos-{index}"
            records = []
            for dose in doses:
                receipt_dir = campaign / "receipts" / f"dose-{dose:+.2f}"
                receipt = receipt_dir / "prediction-receipt.json"
                write(
                    receipt_dir / "intervention.json",
                    {
                        "audit": {
                            "embedding_hook_invocation_count": 5,
                            "maximum_temporal_mean_abs_error": 0.001,
                            "maximum_temporal_mean_tolerance": 0.01,
                        }
                    },
                )
                write(receipt, {"intervention_ref": "intervention.json"})
                records.append(
                    {"dose": dose, "receipt_ref": str(receipt), "metrics": metric_names}
                )
            write(campaign / "manifest.json", {"state": "ready", "records": records})
            cosmos_roots.append(campaign)

        compile_reports = {}
        for backbone in ("ctrl_world", "cosmos3"):
            path = root / "compile" / f"{backbone}.json"
            write(
                path,
                {
                    "typed_compile_receipt": {"compiled": True},
                    "missing_required_semantics": [],
                    "semantic_substitution_used": False,
                },
            )
            compile_reports[backbone] = path
        reuse_audit = root / "reuse-audit.json"
        write(
            reuse_audit,
            {
                "state": "blocked",
                "r31_exact_portability_ready": True,
                "formal_chart_missing_backbones": ["acwm_phys", "ctrl_world"],
                "gpu_launch_decision": "bounded_runtime_canaries_and_repair_screens_only",
                "lobo": {"cold_vs_warm_identifiable": False, "observed_confirm_count": 0},
            },
        )
        return {
            "ctrl_roots": ctrl_roots,
            "cosmos_roots": cosmos_roots,
            "ctrl_compile_report": compile_reports["ctrl_world"],
            "cosmos_compile_report": compile_reports["cosmos3"],
            "reuse_audit": reuse_audit,
        }


if __name__ == "__main__":
    unittest.main()
