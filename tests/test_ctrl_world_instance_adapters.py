from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.evaluate.adapters.ctrl_world import CtrlWorldEvaluationError, evaluate_ctrl_world_receipt
from wmloop.primitives.adapters.ctrl_world_hooks import audit_ctrl_world_hooks


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "configs" / "goal" / "ctrl_world_heldout_split.json"


class CtrlWorldInstanceAdapterTests(unittest.TestCase):
    def test_environment_receipt_projects_verdict_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            receipt = _write_receipt(Path(temporary) / "receipt.json")
            evidence = evaluate_ctrl_world_receipt(receipt_path=receipt, heldout_split_path=SPLIT, split_name="dev")
            self.assertTrue(evidence["validity_gate_pass"])
            self.assertTrue(evidence["accept_eligible"])
            self.assertFalse(evidence["generated_video_scores_used_for_verdict"])

    def test_generated_video_score_cannot_become_verdict(self) -> None:
        with TemporaryDirectory() as temporary:
            receipt = _write_receipt(Path(temporary) / "receipt.json", source="generated_video_score")
            with self.assertRaisesRegex(CtrlWorldEvaluationError, "VERDICT_SOURCE_FORBIDDEN"):
                evaluate_ctrl_world_receipt(receipt_path=receipt, heldout_split_path=SPLIT, split_name="dev")

    def test_receipt_outside_frozen_split_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            receipt = _write_receipt(Path(temporary) / "receipt.json", episode="9999")
            with self.assertRaisesRegex(CtrlWorldEvaluationError, "OUTSIDE_FROZEN_SPLIT"):
                evaluate_ctrl_world_receipt(receipt_path=receipt, heldout_split_path=SPLIT, split_name="dev")

    def test_observed_checkout_exposes_all_five_hook_anchors(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {
                "dataset/dataset_droid_exp33.py": "class Dataset_mix:\n    pass\n",
                "models/ctrl_world.py": "class CrtlWorld:\n    metrics = {\"loss_total\": 0.0}\n",
                "models/pipeline_ctrl_world.py": "class CtrlWorldDiffusionPipeline:\n    pass\n",
                "scripts/train_wm.py": "optimizer = torch.optim.AdamW(model.parameters())\n",
            }
            for relative, content in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            report = audit_ctrl_world_hooks(root)
            self.assertEqual(report["state"], "ready")
            self.assertEqual(report["available_hooks"], ["H1", "H2", "H3", "H4", "H5"])


def _write_receipt(path: Path, *, source: str = "environment_task_receipt", episode: str = "0000") -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-rollout-receipt",
        "evidence_source": source,
        "task_id": "pickplace",
        "episode_id": episode,
        "seed": 101,
        "task_success": True,
        "task_progress": 1.0,
        "safety_events": [],
        "action_valid": True,
        "rollout_ref": "cas://sha256/fixture",
        "evaluator_version": "ctrl-world-g2-v1",
    }, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
