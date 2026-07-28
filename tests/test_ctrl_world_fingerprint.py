from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.ctrl_world_fingerprint import (
    CtrlWorldActionEmbeddingDose,
    CtrlWorldActionEmbeddingTemporalMixDose,
    evaluate_ctrl_world_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs" / "experiments" / "ctrl_world_irg_calibration_pilot_v1.json"
NARROW_CAMPAIGN = ROOT / "configs" / "experiments" / "ctrl_world_irg_calibration_narrow_pilot_v1.json"
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


class _TemporalOutput:
    ndim = 3

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def mean(self, *, dim: int, keepdim: bool) -> "_TemporalOutput":
        assert dim == 1 and keepdim is True
        return _TemporalOutput((sum(self.values) / len(self.values),) * len(self.values))

    def __sub__(self, other: "_TemporalOutput") -> "_TemporalOutput":
        return _TemporalOutput(tuple(a - b for a, b in zip(self.values, other.values, strict=True)))

    def __mul__(self, value: float) -> "_TemporalOutput":
        return _TemporalOutput(tuple(value * item for item in self.values))

    __rmul__ = __mul__

    def __add__(self, other: "_TemporalOutput") -> "_TemporalOutput":
        return _TemporalOutput(tuple(a + b for a, b in zip(self.values, other.values, strict=True)))


class _TemporalEncoder:
    def forward(self, value: float) -> _TemporalOutput:
        return _TemporalOutput((value, 3.0 * value))


class _TemporalModel:
    def __init__(self) -> None:
        self.action_encoder = _TemporalEncoder()


class CtrlWorldFingerprintTests(unittest.TestCase):
    def test_campaign_uses_predictive_acwm_instance(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        self.assertEqual(
            campaign["backbone_instance_ref"],
            "configs/backbones/ctrl_world_predictive_quality_pilot_v1.json",
        )
        self.assertNotIn("action_success", campaign["backbone_instance_ref"])

    def test_narrow_campaign_is_exploratory_and_reduces_the_dose_radius(self) -> None:
        wide = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        narrow = json.loads(NARROW_CAMPAIGN.read_text(encoding="utf-8"))
        self.assertEqual(narrow["backbone_instance_ref"], wide["backbone_instance_ref"])
        self.assertLess(
            max(abs(value) for value in narrow["probe"]["doses"]),
            max(abs(value) for value in wide["probe"]["doses"]),
        )
        self.assertIn("exploratory fingerprint repair", narrow["claim_scope"])

    def test_action_encoder_hook_is_reversible(self) -> None:
        model = _Model()
        self.assertEqual(model.action_encoder.forward(2.0).value, 2.0)
        with CtrlWorldActionEmbeddingDose(model, 0.1):
            self.assertAlmostEqual(model.action_encoder.forward(2.0).value, 2.2)
        self.assertEqual(model.action_encoder.forward(2.0).value, 2.0)

    def test_temporal_mix_hook_is_reversible_and_preserves_mean(self) -> None:
        model = _TemporalModel()
        self.assertEqual(model.action_encoder.forward(1.0).values, (1.0, 3.0))
        with CtrlWorldActionEmbeddingTemporalMixDose(model, 0.5):
            self.assertEqual(model.action_encoder.forward(1.0).values, (1.5, 2.5))
        self.assertEqual(model.action_encoder.forward(1.0).values, (1.0, 3.0))

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
            self.assertEqual(report["locality_admission"]["state"], "passed")
            self.assertTrue(report["locality_admission"]["cross_backbone_transfer_eligible"])
            self.assertEqual(manifest["locality_admission_state"], "passed")


if __name__ == "__main__":
    unittest.main()
