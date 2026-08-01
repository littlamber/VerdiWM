from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.cross_backbone_reuse_audit import run_cross_backbone_reuse_audit


class CrossBackboneReuseAuditTests(unittest.TestCase):
    def test_fixture_evidence_remains_blocked_without_confirm_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            config_path = self._write_fixture(root)
            manifest = run_cross_backbone_reuse_audit(
                config_path=config_path,
                repo_root=root,
                output_root=Path(temporary) / "output",
            )
            self.assertEqual(manifest["state"], "blocked")
            self.assertFalse(manifest["cold_vs_warm_identifiable"])
            self.assertEqual(manifest["observed_confirm_count"], 0)
            report = json.loads((Path(temporary) / "output/cross-backbone-reuse-audit.json").read_text())
            self.assertTrue(report["r31_exact_portability_ready"])
            self.assertEqual(
                report["gpu_launch_decision"],
                "bounded_runtime_canaries_and_repair_screens_only",
            )
            self.assertNotIn(
                "materialize an embedding-level H2 adapter and exact r31 operations for Cosmos3",
                report["minimum_next_work"],
            )
            self.assertEqual(report["certificate_ablation"]["certificate_prevented_negative_count"], 9)

    @staticmethod
    def _write_fixture(root: Path) -> Path:
        def write(relative: str, payload: dict[str, object]) -> None:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

        for backbone in ("ctrl_world", "cosmos3"):
            write(
                f"compile/{backbone}.json",
                {
                    "artifact_type": "verdiwm-probe-semantic-compile-report",
                    "state": "compiled",
                    "typed_compile_receipt": {"compiled": True},
                    "missing_required_semantics": [],
                },
            )
        fingerprints = (
            ("ctrl_dev", "passed", True),
            ("ctrl_formal", "failed", False),
            ("cosmos_formal", "passed", True),
        )
        for name, state, eligible in fingerprints:
            write(
                f"fingerprints/{name}.json",
                {
                    "campaign_id": name,
                    "protocol": "paper" if name.endswith("formal") else "pilot",
                    "measurement_count": 3,
                    "locality_admission": {
                        "state": state,
                        "cross_backbone_transfer_eligible": eligible,
                        "path_residuals": {"action_conditioning_scale": 0.25},
                    },
                },
            )
        write(
            "lobo/report.json",
            {
                "artifact_type": "verdiwm-cross-backbone-experiment-report",
                "state": "blocked",
                "settled_receipt_count": 0,
                "expected_confirm_count": 108,
                "observed_confirm_count": 0,
                "formal_positive_count": 0,
            },
        )
        write(
            "certificate/report.json",
            {
                "artifact_type": "verdiwm-transfer-certificate-ablation",
                "state": "ready",
                "source_scope": "fixture_leave_one_environment_out_cpu_replay",
                "certificate_changed_cell_count": 12,
                "certificate_prevented_negative_count": 9,
                "certificate_blocked_positive_count": 3,
            },
        )
        config = {
            "artifact_type": "verdiwm-cross-backbone-reuse-audit-config",
            "expected_backbones": ["acwm_phys", "ctrl_world", "cosmos3"],
            "probe_compile_reports": [
                {"backbone": name, "path": f"compile/{name}.json"}
                for name in ("ctrl_world", "cosmos3")
            ],
            "fingerprints": [
                {
                    "backbone": "ctrl_world",
                    "split_role": "development",
                    "path": "fingerprints/ctrl_dev.json",
                },
                {
                    "backbone": "ctrl_world",
                    "split_role": "formal",
                    "path": "fingerprints/ctrl_formal.json",
                },
                {
                    "backbone": "cosmos3",
                    "split_role": "formal",
                    "path": "fingerprints/cosmos_formal.json",
                },
            ],
            "lobo_report": {"path": "lobo/report.json"},
            "certificate_ablation": {"path": "certificate/report.json"},
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path
