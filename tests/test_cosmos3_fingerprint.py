from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.cosmos3_fingerprint import (
    Cosmos3FingerprintError,
    fit_cosmos3_fingerprint,
)


class Cosmos3FingerprintTests(unittest.TestCase):
    def test_fits_complete_paired_dose_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self._campaign(root)
            split = self._split(root)
            records = []
            for dose in (-0.1, 0.0, 0.1):
                for sample, seed in ((0, 101), (16, 202), (32, 303)):
                    receipt = self._receipt(root, dose, sample, seed)
                    records.append(
                        {
                            "dose": dose,
                            "sample_index": sample,
                            "seed": seed,
                            "receipt_ref": str(receipt),
                            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                            "gpu_exclusivity_audit": _gpu_audit(),
                        }
                    )
            shard = root / "shard.json"
            shard.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
                        "state": "ready",
                        "campaign_id": "test-cosmos3",
                        "protocol": "pilot",
                        "records": records,
                    }
                )
            )
            result = fit_cosmos3_fingerprint(
                campaign_path=campaign,
                shard_manifests=[shard],
                split_path=split,
                protocol="pilot",
                output_root=root / "output",
            )
            self.assertEqual(result["measurement_count"], 9)
            report = json.loads((root / "output/target-local-fingerprint.json").read_text())
            self.assertEqual(report["repeat_count"], 3)
            self.assertIn("action_conditioning_scale", report["chart"]["intervention_names"])

    def test_rejects_incomplete_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self._campaign(root)
            split = self._split(root)
            shard = root / "shard.json"
            shard.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
                        "state": "ready",
                        "campaign_id": "test-cosmos3",
                        "protocol": "pilot",
                        "records": [],
                    }
                )
            )
            with self.assertRaisesRegex(Cosmos3FingerprintError, "INCOMPLETE"):
                fit_cosmos3_fingerprint(
                    campaign_path=campaign,
                    shard_manifests=[shard],
                    split_path=split,
                    protocol="pilot",
                    output_root=root / "output",
                )

    def test_rejects_receipt_from_different_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self._campaign(root)
            payload = json.loads(campaign.read_text())
            payload["probe"]["probe_id"] = "action_embedding_temporal_mix"
            campaign.write_text(json.dumps(payload))
            split = self._split(root)
            records = []
            for dose in (-0.1, 0.0, 0.1):
                for sample, seed in ((0, 101), (16, 202), (32, 303)):
                    receipt = self._receipt(root, dose, sample, seed)
                    records.append(
                        {
                            "dose": dose,
                            "sample_index": sample,
                            "seed": seed,
                            "receipt_ref": str(receipt),
                            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                            "gpu_exclusivity_audit": _gpu_audit(),
                        }
                    )
            shard = root / "shard.json"
            shard.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
                        "state": "ready",
                        "campaign_id": "test-cosmos3",
                        "protocol": "pilot",
                        "records": records,
                    }
                )
            )
            with self.assertRaisesRegex(Cosmos3FingerprintError, "INTERVENTION_MISMATCH"):
                fit_cosmos3_fingerprint(
                    campaign_path=campaign,
                    shard_manifests=[shard],
                    split_path=split,
                    protocol="pilot",
                    output_root=root / "output",
                )

    def _campaign(self, root: Path) -> Path:
        path = root / "campaign.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
                    "campaign_id": "test-cosmos3",
                    "claim_scope": "test",
                    "probe": {
                        "probe_id": "action_conditioning_scale",
                        "doses": [-0.1, 0.0, 0.1],
                    },
                    "outcomes": [
                        {
                            "name": "rollout_video_psnr",
                            "source_metric": "rollout_video_psnr",
                            "sign": 1.0,
                            "weight": 1.0,
                        },
                        {
                            "name": "negative_rollout_video_l1",
                            "source_metric": "rollout_video_l1",
                            "sign": -1.0,
                            "weight": 1.0,
                        },
                    ],
                    "protocols": {"pilot": {"split": "dev", "required_receipts_per_dose": 3}},
                    "locality_admission": {
                        "maximum_residual": 1.0,
                        "failure_policy": "abstain",
                    },
                }
            )
        )
        return path

    def _split(self, root: Path) -> Path:
        path = root / "split.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-cosmos3-forward-dynamics-split",
                    "split_id": "cosmos3_droid_lerobot_sample_v1",
                    "dataset_ref": "fixture",
                    "dev": [
                        {"sample_index": 0, "seed": 101},
                        {"sample_index": 16, "seed": 202},
                        {"sample_index": 32, "seed": 303},
                    ],
                    "accept": [{"sample_index": 48, "seed": 101}],
                }
            )
        )
        return path

    def _receipt(self, root: Path, dose: float, sample: int, seed: int) -> Path:
        receipt_root = root / f"receipt-{dose}-{sample}-{seed}"
        receipt_root.mkdir()
        hook = receipt_root / "action-hook-receipt.json"
        hook.write_text(json.dumps({"dose": dose}))
        hook_sha = hashlib.sha256(hook.read_bytes()).hexdigest()
        receipt = receipt_root / "prediction-receipt.json"
        receipt.write_text(json.dumps(_receipt(dose, sample, seed, hook_sha)))
        return receipt


def _receipt(dose: float, sample: int, seed: int, hook_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-prediction-receipt",
        "evidence_source": "paired_ground_truth_rollout",
        "model_mode": "forward_dynamics",
        "split_id": "cosmos3_droid_lerobot_sample_v1",
        "split_name": "dev",
        "dataset_freeze_id": "cosmos3_droid_lerobot_cookbook_sample_v1",
        "sample_index": sample,
        "seed": seed,
        "viewpoint": "concat_view",
        "action_shape": [16, 10],
        "horizon_frames": 16,
        "metrics": {
            "rollout_video_psnr": 20.0 + dose + seed / 10000.0,
            "rollout_video_l1": 0.1 - dose / 100.0,
            "final_frame_mae": 0.2,
            "temporal_difference_mae": 0.15,
        },
        "frame_alignment": {
            "ground_truth_shape": [17, 16, 16, 3],
            "rollout_shape": [17, 16, 16, 3],
            "condition_frame_index": 0,
            "future_start_index": 1,
            "future_frame_count": 16,
            "spatial_policy": "top_left_content_crop_to_rollout",
            "conditioning_frame_mae": 0.0,
            "max_conditioning_frame_mae": 0.031,
        },
        "action_conditioned": True,
        "intervention_ref": "action-hook-receipt.json",
        "intervention": {
            "probe_id": "action_conditioning_scale",
            "dose": dose,
            "dose_unit": "relative_action_input_scale",
            "hook_receipt_sha256": hook_sha,
        },
        "conditioning_ref": "condition.png",
        "action_ref": "action.json",
        "ground_truth_ref": "ground-truth.npy",
        "rollout_ref": "rollout.mp4",
        "sha256": {
            "action_input": "a" * 64,
            "conditioning": "b" * 64,
            "ground_truth": "c" * 64,
            "rollout": "d" * 64,
        },
        "evaluator_version": "test",
    }


def _gpu_audit() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-gpu-exclusivity-audit",
        "state": "ready",
        "gpu_index": 0,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "monitor_interval_seconds": 0.5,
        "sample_count": 3,
        "observed_campaign_pids": [123],
        "foreign_pid_events": [],
    }


if __name__ == "__main__":
    unittest.main()
