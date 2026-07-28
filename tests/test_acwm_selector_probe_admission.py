from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.selector_probe_admission import evaluate_selector_probe_admission


class AcwmSelectorProbeAdmissionTests(unittest.TestCase):
    def test_admits_heldout_correction_without_risk_regression(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = self._replay(root / "baseline", predicted="positive", work_orders=6)
            candidate = self._replay(root / "candidate", predicted="negative", work_orders=6)
            atlas = root / "atlas.json"
            atlas.write_text(json.dumps({"artifact_type": "verdiwm-acwm-fingerprint-atlas-manifest", "measurement_complete_count": 8, "locality_calibrated_count": 6}))
            affinity = root / "affinity.json"
            affinity.write_text(json.dumps({"artifact_type": "verdiwm-primitive-probe-affinity", "contract_id": "candidate"}))

            manifest = evaluate_selector_probe_admission(
                baseline_replay_root=baseline,
                candidate_replay_root=candidate,
                candidate_atlas_manifest=atlas,
                candidate_affinity=affinity,
                output_root=root / "output",
            )

            self.assertTrue(manifest["promote_affinity_allowed"])
            report = json.loads((root / "output" / "selector-probe-admission.json").read_text())
            self.assertEqual(report["corrected_targets"], ["cloth_move"])
            self.assertEqual(report["regressed_targets"], [])

    def test_rejects_candidate_when_work_orders_increase(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = self._replay(root / "baseline", predicted="positive", work_orders=6)
            candidate = self._replay(root / "candidate", predicted="negative", work_orders=7)
            atlas = root / "atlas.json"
            atlas.write_text(json.dumps({"artifact_type": "verdiwm-acwm-fingerprint-atlas-manifest", "measurement_complete_count": 8, "locality_calibrated_count": 6}))
            affinity = root / "affinity.json"
            affinity.write_text(json.dumps({"artifact_type": "verdiwm-primitive-probe-affinity", "contract_id": "candidate"}))

            manifest = evaluate_selector_probe_admission(
                baseline_replay_root=baseline,
                candidate_replay_root=candidate,
                candidate_atlas_manifest=atlas,
                candidate_affinity=affinity,
                output_root=root / "output",
            )

            self.assertFalse(manifest["promote_affinity_allowed"])

    def _replay(self, root: Path, *, predicted: str, work_orders: int) -> Path:
        (root / "tables").mkdir(parents=True)
        manifest = {
            "artifact_type": "verdiwm-acwm-selector-cpu-replay-manifest",
            "transfer_certificate_abstention_count": 9,
            "transfer_work_order_count": work_orders,
            "probe_evolution_work_order_count": 8,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        with (root / "tables" / "candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["target_environment", "selector", "seed", "rank", "primitive", "predicted_sign", "target_positive"])
            writer.writeheader()
            for seed in (101, 202, 303):
                writer.writerow({"target_environment": "cloth_move", "selector": "irg", "seed": seed, "rank": 1, "primitive": "inv_dyn_reward_finetune", "predicted_sign": predicted, "target_positive": "False"})
        return root


if __name__ == "__main__":
    unittest.main()
