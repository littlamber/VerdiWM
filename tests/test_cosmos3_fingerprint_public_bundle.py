from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export.cosmos3_fingerprint_public_bundle import (
    _selected_video_doses,
    export_cosmos3_fingerprint_public_bundle,
)


class Cosmos3FingerprintPublicBundleTests(unittest.TestCase):
    def test_selects_three_distinct_symmetric_or_one_sided_video_doses(self) -> None:
        self.assertEqual(_selected_video_doses([-0.1, -0.05, 0.0, 0.05, 0.1]), (-0.1, 0.0, 0.1))
        self.assertEqual(_selected_video_doses([0.0, 0.0125, 0.025]), (0.0, 0.0125, 0.025))
        self.assertEqual(_selected_video_doses([-0.025, -0.0125, 0.0]), (-0.025, -0.0125, 0.0))

    def test_exports_sanitized_tables_and_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = root / "fingerprint"
            fingerprint.mkdir()
            outcomes = [
                "rollout_video_psnr",
                "negative_rollout_video_l1",
                "negative_final_frame_mae",
                "negative_temporal_difference_mae",
            ]
            report = {
                "artifact_type": "verdiwm-cosmos3-target-local-fingerprint",
                "state": "ready",
                "campaign_id": "test",
                "protocol": "pilot",
                "split": "dev",
                "measurement_count": 3,
                "repeat_count": 1,
                "chart": {"outcome_names": outcomes},
                "locality_admission": {
                    "state": "passed",
                    "cross_backbone_transfer_eligible": True,
                },
                "claim_boundary": "target local only",
            }
            campaign = {
                "campaign_id": "test",
                "probe": {
                    "probe_id": "action_embedding_temporal_mix",
                    "dose_unit": "temporal_action_input_mix",
                    "doses": [-0.1, 0.0, 0.1],
                },
            }
            measurements = []
            records = []
            for dose in (-0.1, 0.0, 0.1):
                receipt_root = root / f"receipt-{dose}"
                receipt_root.mkdir()
                receipt = receipt_root / "prediction-receipt.json"
                receipt.write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "rollout_video_psnr": 20.0 + dose,
                                "rollout_video_l1": 0.1 - dose / 10,
                                "final_frame_mae": 0.2,
                                "temporal_difference_mae": 0.3,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
                measurements.append(
                    {
                        "dose": dose,
                        "sample_index": 0,
                        "seed": 101,
                        "outcomes": [20.0 + dose, -0.1, -0.2, -0.3],
                        "receipt_sha256": digest,
                        "gpu_uuid": "GPU-test",
                        "gpu_audit_sample_count": 2,
                    }
                )
                records.append(
                    {
                        "dose": dose,
                        "sample_index": 0,
                        "seed": 101,
                        "receipt_ref": str(receipt),
                        "receipt_sha256": digest,
                        "gpu_exclusivity_audit": {
                            "state": "ready",
                            "foreign_pid_events": [],
                            "gpu_uuid": "GPU-test",
                        },
                    }
                )
            (fingerprint / "target-local-fingerprint.json").write_text(json.dumps(report))
            (fingerprint / "input-campaign.json").write_text(json.dumps(campaign))
            (fingerprint / "measurements.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in measurements)
            )
            shard = root / "shard.json"
            shard.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
                        "state": "ready",
                        "campaign_id": "test",
                        "protocol": "pilot",
                        "records": records,
                    }
                )
            )
            bundle = export_cosmos3_fingerprint_public_bundle(
                fingerprint_root=fingerprint,
                shard_manifests=[shard],
                output_root=root / "public",
                include_videos=False,
            )
            self.assertEqual(bundle["measurement_count"], 3)
            self.assertTrue((root / "public/tables/dose-metrics.csv").is_file())
            self.assertTrue((root / "public/tables/dose-response.csv").is_file())
            self.assertTrue((root / "public/figures/dose-response.svg").is_file())
            self.assertIn("temporal action mix", (root / "public/figures/dose-response.svg").read_text())
            self.assertIn("action_embedding_temporal_mix", (root / "public/README.md").read_text())
            machine_path_prefix = "/" + "mnt" + "/"
            self.assertNotIn(machine_path_prefix, (root / "public/measurements.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
