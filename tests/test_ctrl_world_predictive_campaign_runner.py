from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_ctrl_world_predictive_campaign import (
    CtrlWorldPredictiveCampaignError,
    _asset_identity,
    _load_upstream_module_with_local_model_base,
    prediction_receipts_from_summary,
    protocol_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class CtrlWorldPredictiveCampaignRunnerTests(unittest.TestCase):
    def test_protocol_rows_preserve_frozen_episode_seed_pairs(self) -> None:
        campaign = json.loads(
            (ROOT / "configs/experiments/ctrl_world_irg_calibration_pilot_v1.json").read_text(encoding="utf-8")
        )
        split = json.loads((ROOT / "configs/goal/ctrl_world_heldout_split.json").read_text(encoding="utf-8"))
        rows = protocol_rows(split_payload=split, campaign=campaign, protocol="pilot")
        self.assertEqual(
            rows,
            [
                {"task_id": "pickplace", "episode_id": "0000", "seed": 101},
                {"task_id": "pickplace", "episode_id": "0001", "seed": 202},
                {"task_id": "pickplace", "episode_id": "0002", "seed": 303},
            ],
        )

    def test_receipt_projection_uses_predictive_metrics_not_task_success(self) -> None:
        rows = [{"task_id": "pickplace", "episode_id": "0000", "seed": 101}]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            video_dir = root / "run" / "videos"
            video_dir.mkdir(parents=True)
            (video_dir / "00_0000_demo.mp4").write_bytes(b"video")
            summary = {
                "run_dir": str(root / "run"),
                "episodes": [
                    {
                        "traj_id": "0000",
                        "rollout_video_psnr": 20.0,
                        "rollout_video_l1": 0.1,
                        "segment_final_mae_mean": 0.2,
                        "segment_view_pair_mae_mean": 0.3,
                        "segment_view_fused_mae_mean": 0.25,
                    }
                ],
            }
            index = prediction_receipts_from_summary(
                summary=summary,
                rows=rows,
                dose=0.0,
                horizon_frames=32,
                evaluator_version="test-v1",
                receipt_root=root / "receipts",
            )
            receipt = json.loads(Path(index[0]["receipt_ref"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["evidence_source"], "paired_ground_truth_rollout")
            self.assertNotIn("task_success", receipt)
            self.assertEqual(receipt["horizon_frames"], 32)

    def test_receipt_projection_rejects_identity_drift(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run" / "videos").mkdir(parents=True)
            (root / "run" / "videos" / "00_9999_demo.mp4").write_bytes(b"video")
            with self.assertRaisesRegex(CtrlWorldPredictiveCampaignError, "IDENTITY_MISMATCH"):
                prediction_receipts_from_summary(
                    summary={"run_dir": str(root / "run"), "episodes": [{"traj_id": "9999"}]},
                    rows=[{"task_id": "pickplace", "episode_id": "0000", "seed": 101}],
                    dose=0.0,
                    horizon_frames=32,
                    evaluator_version="test-v1",
                    receipt_root=root / "receipts",
                )

    def test_large_asset_identity_can_defer_and_reuse_a_cached_hash(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            path.write_bytes(b"checkpoint")
            cache = {}
            deferred = _asset_identity(path, cache=cache, hash_large_assets=False)
            self.assertEqual(deferred["sha256_state"], "deferred")
            computed = _asset_identity(path, cache=cache, hash_large_assets=True)
            self.assertEqual(computed["sha256_state"], "computed")
            reused = _asset_identity(path, cache=cache, hash_large_assets=False)
            self.assertEqual(reused["sha256_state"], "cached")
            self.assertEqual(reused["sha256"], computed["sha256"])

    def test_upstream_import_prefers_and_then_restores_local_model_base(self) -> None:
        original = os.environ.get("CTRL_WORLD_MODEL_ROOT")
        os.environ["CTRL_WORLD_MODEL_ROOT"] = "/previous/model/base"
        observed: list[str | None] = []

        def fake_loader(*, eval_root: Path, model_root: Path) -> object:
            observed.append(os.environ.get("CTRL_WORLD_MODEL_ROOT"))
            return object()

        try:
            with patch("scripts.run_ctrl_world_predictive_campaign._load_upstream_module", fake_loader):
                _load_upstream_module_with_local_model_base(
                    eval_root=ROOT,
                    model_root=ROOT,
                    local_model_base=ROOT / "models",
                )
            self.assertEqual(observed, [str((ROOT / "models").resolve())])
            self.assertEqual(os.environ.get("CTRL_WORLD_MODEL_ROOT"), "/previous/model/base")
        finally:
            if original is None:
                os.environ.pop("CTRL_WORLD_MODEL_ROOT", None)
            else:
                os.environ["CTRL_WORLD_MODEL_ROOT"] = original


if __name__ == "__main__":
    unittest.main()
