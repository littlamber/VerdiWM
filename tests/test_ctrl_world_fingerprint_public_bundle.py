from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.ctrl_world_fingerprint_public_bundle import export_ctrl_world_fingerprint_public_bundle


class CtrlWorldFingerprintPublicBundleTests(unittest.TestCase):
    def test_exports_sanitized_chart_and_response_tables(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fingerprint = root / "private/fingerprint"
            fingerprint.mkdir(parents=True)
            campaign_id = "candidate"
            campaign = {
                "artifact_type": "verdiwm-ctrl-world-fingerprint-campaign",
                "campaign_id": campaign_id,
                "probe": {"doses": [-0.1, 0.0, 0.1]},
            }
            chart = {
                "outcome_names": ["quality", "negative_error"],
                "jacobian": [[1.0], [2.0]],
                "response_coordinate": [1.0, 2.0],
                "covariance": [[1.0, 0.0], [0.0, 2.0]],
            }
            report = {
                "artifact_type": "verdiwm-ctrl-world-target-local-fingerprint",
                "campaign_id": campaign_id,
                "chart": chart,
            }
            rows = []
            for dose in (-0.1, 0.0, 0.1):
                for seed in (101, 202, 303):
                    rows.append(
                        {
                            "campaign_id": campaign_id,
                            "dose": dose,
                            "identity": {"task_id": "pickplace", "episode_id": str(seed), "seed": seed},
                            "outcomes": [dose + seed / 1000.0, 2.0 * dose],
                            "receipt_ref": f"/{'mnt'}/private/{seed}.json",
                        }
                    )
            (fingerprint / "input-campaign.json").write_text(json.dumps(campaign), encoding="utf-8")
            (fingerprint / "target-local-fingerprint.json").write_text(json.dumps(report), encoding="utf-8")
            (fingerprint / "measurements.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            settlement = root / "settlement"
            settlement.mkdir()
            (settlement / "settlement.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-ctrl-world-fingerprint-settlement",
                        "state": "settled_admitted",
                        "protocol": "pilot",
                        "selected_campaign_id": campaign_id,
                        "cross_backbone_transfer_eligible": True,
                        "claim_boundary": "target local only",
                        "candidates": [
                            {
                                "campaign_id": campaign_id,
                                "fingerprint_root": str(fingerprint),
                                "measurement_count": 9,
                                "locality_state": "passed",
                                "dose_radius": 0.1,
                                "maximum_residual": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bundle = export_ctrl_world_fingerprint_public_bundle(
                settlement_root=settlement,
                output_root=root / "public",
            )
            self.assertEqual(bundle["measurement_count"], 9)
            self.assertEqual(bundle["baseline_reproducibility_state"], "not_applicable")
            self.assertTrue((root / "public/tables/dose-response.csv").is_file())
            self.assertTrue((root / "public/tables/chart-summary.csv").is_file())
            self.assertNotIn("/" + "mnt" + "/", (root / "public/settlement.json").read_text())
            self.assertNotIn("receipt_ref", (root / "public/candidates/candidate/measurements.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
