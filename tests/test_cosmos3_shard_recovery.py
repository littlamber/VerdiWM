from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.cosmos3_shard_recovery import (
    Cosmos3ShardRecoveryError,
    recover_cosmos3_shard,
)


class Cosmos3ShardRecoveryTests(unittest.TestCase):
    def test_recovers_only_revalidated_completed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, split, source = _fixture(root)
            result = recover_cosmos3_shard(
                campaign_path=campaign,
                source_manifest_path=source,
                split_path=split,
                protocol="pilot",
                doses=[-0.1],
                output_path=root / "recovered.json",
            )
            self.assertEqual(result["state"], "ready")
            self.assertEqual(result["receipt_count"], 1)
            self.assertEqual(result["doses"], [-0.1])
            self.assertEqual(result["recovery"]["source_state"], "running")

    def test_rejects_tampered_receipt_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, split, source = _fixture(root)
            payload = json.loads(source.read_text())
            payload["records"][0]["receipt_sha256"] = "0" * 64
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(Cosmos3ShardRecoveryError, "RECEIPT_SHA_MISMATCH"):
                recover_cosmos3_shard(
                    campaign_path=campaign,
                    source_manifest_path=source,
                    split_path=split,
                    protocol="pilot",
                    doses=[-0.1],
                    output_path=root / "recovered.json",
                )


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    campaign = root / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
                "campaign_id": "recovery-test",
                "probe": {
                    "probe_id": "action_conditioning_scale",
                    "doses": [-0.1, 0.0, 0.1],
                },
                "protocols": {"pilot": {"split": "dev"}},
            }
        )
    )
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-cosmos3-forward-dynamics-split",
                "split_id": "recovery-split",
                "dataset_ref": "fixture",
                "dev": [{"sample_index": 4, "seed": 7}],
                "accept": [{"sample_index": 48, "seed": 9}],
            }
        )
    )
    receipt_root = root / "receipt"
    receipt_root.mkdir()
    hook = receipt_root / "hook.json"
    hook.write_text("{}\n")
    hook_sha = hashlib.sha256(hook.read_bytes()).hexdigest()
    metrics = {
        "rollout_video_psnr": 20.0,
        "rollout_video_l1": 0.1,
        "final_frame_mae": 0.2,
        "temporal_difference_mae": 0.15,
    }
    receipt = receipt_root / "prediction-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-cosmos3-prediction-receipt",
                "evidence_source": "paired_ground_truth_rollout",
                "model_mode": "forward_dynamics",
                "split_id": "recovery-split",
                "split_name": "dev",
                "dataset_freeze_id": "cosmos3_droid_lerobot_cookbook_sample_v1",
                "sample_index": 4,
                "seed": 7,
                "viewpoint": "concat_view",
                "action_shape": [16, 10],
                "horizon_frames": 16,
                "metrics": metrics,
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
                "intervention_ref": hook.name,
                "intervention": {
                    "probe_id": "action_conditioning_scale",
                    "dose": -0.1,
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
        )
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    source = root / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
                "state": "running",
                "campaign_id": "recovery-test",
                "protocol": "pilot",
                "probe_id": "action_conditioning_scale",
                "identities": [{"sample_index": 4, "seed": 7}],
                "records": [
                    {
                        "dose": -0.1,
                        "sample_index": 4,
                        "seed": 7,
                        "receipt_ref": str(receipt),
                        "receipt_sha256": receipt_sha,
                        "metrics": metrics,
                        "gpu_exclusivity_audit": {
                            "artifact_type": "verdiwm-cosmos3-gpu-exclusivity-audit",
                            "state": "ready",
                            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                            "sample_count": 2,
                            "foreign_pid_events": [],
                        },
                    }
                ],
            }
        )
    )
    return campaign, split, source


if __name__ == "__main__":
    unittest.main()
