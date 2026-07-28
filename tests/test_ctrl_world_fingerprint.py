from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.ctrl_world_fingerprint import (
    CtrlWorldActionEmbeddingDose,
    evaluate_ctrl_world_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs" / "experiments" / "ctrl_world_irg_calibration_pilot_v1.json"
SPLIT = ROOT / "configs" / "goal" / "ctrl_world_heldout_split.json"


class _Output:
    def __init__(self, value: float) -> None:
        self.value = value

    def __mul__(self, scale: float) -> "_Output":
        return _Output(self.value * scale)


class _Encoder:
    def forward(self, value: float) -> _Output:
        return _Output(value)


class _Model:
    def __init__(self) -> None:
        self.action_encoder = _Encoder()


class CtrlWorldFingerprintTests(unittest.TestCase):
    def test_campaign_uses_predictive_acwm_instance(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        self.assertEqual(
            campaign["backbone_instance_ref"],
            "configs/backbones/ctrl_world_predictive_quality_pilot_v1.json",
        )
        self.assertNotIn("action_success", campaign["backbone_instance_ref"])

    def test_action_encoder_hook_is_reversible(self) -> None:
        model = _Model()
        self.assertEqual(model.action_encoder.forward(2.0).value, 2.0)
        with CtrlWorldActionEmbeddingDose(model, 0.1):
            self.assertAlmostEqual(model.action_encoder.forward(2.0).value, 2.2)
        self.assertEqual(model.action_encoder.forward(2.0).value, 2.0)

    def test_evaluates_complete_paired_environment_receipts(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = json.loads(CAMPAIGN.read_text())
            split = json.loads(SPLIT.read_text())["dev"]
            rows = []
            for dose in campaign["probe"]["doses"]:
                for item in split:
                    receipt = root / f"receipt_{dose}_{item['seed']}.json"
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
                                    "rollout_video_psnr": 20.0 + 2.0 * dose,
                                    "rollout_video_l1": 0.1 - 0.01 * dose,
                                    "segment_final_mae": 0.2 - 0.02 * dose,
                                    "segment_view_pair_mae": 0.3 - 0.01 * dose,
                                    "segment_view_fused_mae": 0.25 - 0.01 * dose,
                                },
                                "action_conditioned": True,
                                "rollout_ref": f"rollout-{dose}-{item['seed']}",
                                "evaluator_version": "test-v1",
                            }
                        ),
                        encoding="utf-8",
                    )
                    rows.append({"dose": dose, "receipt_ref": str(receipt)})
            index = root / "index.json"
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
            manifest = evaluate_ctrl_world_fingerprint(
                campaign_path=CAMPAIGN,
                receipt_index_path=index,
                heldout_split_path=SPLIT,
                protocol="pilot",
                output_root=root / "output",
            )
            self.assertEqual(manifest["measurement_count"], 15)
            report = json.loads((root / "output" / "target-local-fingerprint.json").read_text())
            self.assertFalse(report["downstream_task_success_used_for_verdict"])
            self.assertEqual(report["chart"]["repeat_count"], 3)


if __name__ == "__main__":
    unittest.main()
