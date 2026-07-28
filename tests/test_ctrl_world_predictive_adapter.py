from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.evaluate.adapters.ctrl_world_predictive import (
    CtrlWorldPredictiveEvaluationError,
    evaluate_ctrl_world_prediction_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "configs" / "goal" / "ctrl_world_heldout_split.json"


class CtrlWorldPredictiveAdapterTests(unittest.TestCase):
    def test_accepts_predictive_receipt_and_rejects_downstream_success_source(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            payload = {
                "schema_version": 1,
                "artifact_type": "verdiwm-ctrl-world-prediction-receipt",
                "evidence_source": "paired_ground_truth_rollout",
                "task_id": "pickplace",
                "episode_id": "0000",
                "seed": 101,
                "horizon_frames": 32,
                "metrics": {
                    "rollout_video_psnr": 20.0,
                    "rollout_video_l1": 0.1,
                    "segment_final_mae": 0.2,
                    "segment_view_pair_mae": 0.3,
                    "segment_view_fused_mae": 0.25,
                },
                "action_conditioned": True,
                "rollout_ref": "rollout.mp4",
                "evaluator_version": "ctrl-world-long-v1",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            evidence = evaluate_ctrl_world_prediction_receipt(
                receipt_path=path,
                heldout_split_path=SPLIT,
                split_name="dev",
            )
            self.assertFalse(evidence["downstream_task_success_used_for_verdict"])
            payload["evidence_source"] = "downstream_task_success"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CtrlWorldPredictiveEvaluationError, "DOWNSTREAM_SUCCESS_FORBIDDEN"):
                evaluate_ctrl_world_prediction_receipt(
                    receipt_path=path,
                    heldout_split_path=SPLIT,
                    split_name="dev",
                )


if __name__ == "__main__":
    unittest.main()
