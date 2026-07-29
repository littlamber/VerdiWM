from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.ctrl_world_receipt_merge import (
    CtrlWorldReceiptMergeError,
    merge_ctrl_world_receipt_indexes,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/experiments/ctrl_world_irg_calibration_pilot_v1.json"
SPLIT = ROOT / "configs/goal/ctrl_world_heldout_split.json"


class CtrlWorldReceiptMergeTests(unittest.TestCase):
    def test_merges_complete_shards_into_audited_index(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = self._write_shards(root)
            manifest = merge_ctrl_world_receipt_indexes(
                campaign_path=CAMPAIGN,
                heldout_split_path=SPLIT,
                protocol="pilot",
                receipt_index_paths=indexes,
                output_root=root / "merged",
            )
            self.assertEqual(manifest["receipt_count"], 15)
            self.assertEqual(manifest["repeat_count"], 3)
            merged = json.loads((root / "merged/receipt-index.json").read_text())
            self.assertEqual([row["dose"] for row in merged["rows"]][::3], [-0.1, -0.05, 0.0, 0.05, 0.1])
            self.assertTrue(all(Path(row["receipt_ref"]).is_file() for row in merged["rows"]))

    def test_rejects_duplicate_receipt_identity(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = self._write_shards(root)
            with self.assertRaisesRegex(CtrlWorldReceiptMergeError, "DUPLICATE"):
                merge_ctrl_world_receipt_indexes(
                    campaign_path=CAMPAIGN,
                    heldout_split_path=SPLIT,
                    protocol="pilot",
                    receipt_index_paths=[*indexes, indexes[0]],
                    output_root=root / "merged",
                )

    def test_rejects_incomplete_paired_frame(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = self._write_shards(root)
            payload = json.loads(indexes[-1].read_text())
            payload["rows"].pop()
            indexes[-1].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CtrlWorldReceiptMergeError, "PAIRED_FRAME_INVALID"):
                merge_ctrl_world_receipt_indexes(
                    campaign_path=CAMPAIGN,
                    heldout_split_path=SPLIT,
                    protocol="pilot",
                    receipt_index_paths=indexes,
                    output_root=root / "merged",
                )

    def _write_shards(self, root: Path) -> list[Path]:
        campaign = json.loads(CAMPAIGN.read_text())
        split = json.loads(SPLIT.read_text())["dev"]
        indexes = []
        for shard_id, doses in enumerate(((-0.1, -0.05), (0.0, 0.05), (0.1,))):
            shard = root / f"shard-{shard_id}"
            shard.mkdir()
            rows = []
            for dose in doses:
                for item in split:
                    receipt = shard / f"receipt-{dose}-{item['seed']}.json"
                    receipt.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "artifact_type": "verdiwm-ctrl-world-prediction-receipt",
                                "evidence_source": "paired_ground_truth_rollout",
                                "task_id": item["task_id"],
                                "episode_id": item["episode_id"],
                                "seed": item["seed"],
                                "horizon_frames": 32,
                                "metrics": {
                                    "rollout_video_psnr": 20.0 + dose,
                                    "rollout_video_l1": 0.1 - dose / 100.0,
                                    "segment_final_mae": 0.2 - dose / 100.0,
                                    "segment_view_pair_mae": 0.3 - dose / 100.0,
                                    "segment_view_fused_mae": 0.25 - dose / 100.0,
                                },
                                "action_conditioned": True,
                                "rollout_ref": f"rollout-{dose}-{item['seed']}",
                                "evaluator_version": "test-v1",
                            }
                        ),
                        encoding="utf-8",
                    )
                    rows.append({"dose": dose, "receipt_ref": str(receipt)})
            index = shard / "receipt-index.json"
            index.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-ctrl-world-fingerprint-receipt-index",
                        "campaign_id": campaign["campaign_id"],
                        "protocol": "pilot",
                        "rows": rows,
                    }
                ),
                encoding="utf-8",
            )
            indexes.append(index)
        return indexes


if __name__ == "__main__":
    unittest.main()
