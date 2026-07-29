from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.ctrl_world_fingerprint_settlement import settle_ctrl_world_fingerprints


class CtrlWorldFingerprintSettlementTests(unittest.TestCase):
    def test_selects_widest_admitted_radius(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            failed = self._write_candidate(root / "wide", campaign_id="wide", radius=0.1, residual=1.2)
            passed = self._write_candidate(root / "narrow", campaign_id="narrow", radius=0.025, residual=0.3)
            manifest = settle_ctrl_world_fingerprints(
                fingerprint_roots=[failed, passed],
                protocol="pilot",
                output_root=root / "settled",
            )
            self.assertEqual(manifest["state"], "settled_admitted")
            self.assertEqual(manifest["selected_campaign_id"], "narrow")
            report = json.loads((root / "settled/settlement.json").read_text())
            self.assertTrue(report["cross_backbone_transfer_eligible"])

    def test_settles_abstention_when_every_radius_fails(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._write_candidate(root / "first", campaign_id="first", radius=0.1, residual=1.2)
            second = self._write_candidate(root / "second", campaign_id="second", radius=0.025, residual=0.8)
            manifest = settle_ctrl_world_fingerprints(
                fingerprint_roots=[first, second],
                protocol="pilot",
                output_root=root / "settled",
            )
            self.assertEqual(manifest["state"], "settled_abstained")
            self.assertIsNone(manifest["selected_campaign_id"])

    def _write_candidate(self, root: Path, *, campaign_id: str, radius: float, residual: float) -> Path:
        root.mkdir()
        state = "passed" if residual <= 0.5 else "failed"
        campaign = {
            "artifact_type": "verdiwm-ctrl-world-fingerprint-campaign",
            "campaign_id": campaign_id,
            "probe": {"probe_id": "action_conditioning_scale", "doses": [-radius, 0.0, radius]},
            "protocols": {"pilot": {"required_receipts_per_dose": 3}},
        }
        report = {
            "artifact_type": "verdiwm-ctrl-world-target-local-fingerprint",
            "campaign_id": campaign_id,
            "protocol": "pilot",
            "measurement_count": 9,
            "locality_admission": {
                "state": state,
                "maximum_residual": 0.5,
                "path_residuals": {"action_conditioning_scale": residual},
                "supported_local_paths": ["action_conditioning_scale"] if state == "passed" else [],
            },
        }
        manifest = {
            "artifact_type": "verdiwm-ctrl-world-target-local-fingerprint-manifest",
            "campaign_id": campaign_id,
            "protocol": "pilot",
            "locality_admission_state": state,
        }
        (root / "input-campaign.json").write_text(json.dumps(campaign), encoding="utf-8")
        (root / "target-local-fingerprint.json").write_text(json.dumps(report), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root


if __name__ == "__main__":
    unittest.main()
