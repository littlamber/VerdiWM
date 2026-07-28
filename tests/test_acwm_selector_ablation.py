from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.selector_ablation import build_selector_ablation_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "acwm_phys_selector_ablation_v1.json"


class ACWMSelectorAblationTests(unittest.TestCase):
    def test_plans_matched_trials_and_blocks_missing_labels(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = self._write_atlas(root)
            manifest = build_selector_ablation_plan(
                config_path=CONFIG,
                fingerprint_atlas_root=atlas,
                output_root=root / "plan",
            )
            self.assertEqual(manifest["planned_trial_count"], 96)
            self.assertTrue(manifest["cpu_replay_ready"])
            self.assertFalse(manifest["gpu_confirmation_ready"])
            report = json.loads((root / "plan" / "selector-ablation-plan.json").read_text())
            self.assertEqual(report["settled_label_environment_count"], 0)
            self.assertEqual(report["candidate_supported_fold_count"], 0)
            first = report["trials"][0]
            self.assertNotIn(first["target_environment"], first["source_environments"])
            self.assertEqual(first["formal_evidence_requires"], "settled_official_gate_receipt")

    def test_ready_requires_settled_labels_for_all_targets(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = self._write_atlas(root)
            environments = json.loads(CONFIG.read_text())["environments"]
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "environment": environment,
                                "primitive": primitive,
                                "settled": True,
                                "positive": True,
                            }
                            for environment in environments
                            for primitive in ("shared_a", "shared_b")
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_selector_ablation_plan(
                config_path=CONFIG,
                fingerprint_atlas_root=atlas,
                effect_label_index=labels,
                output_root=root / "plan",
            )
            self.assertEqual(manifest["state"], "ready")
            self.assertTrue(manifest["gpu_confirmation_ready"])
            self.assertEqual(manifest["selector_identifiable_fold_count"], 8)

    def test_blocks_gpu_confirmation_when_a_fold_has_no_source_supported_candidate(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = self._write_atlas(root)
            environments = json.loads(CONFIG.read_text())["environments"]
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "environment": environment,
                                "primitive": "target_only" if environment == environments[0] else "shared",
                                "settled": True,
                                "positive": False,
                            }
                            for environment in environments
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_selector_ablation_plan(
                config_path=CONFIG,
                fingerprint_atlas_root=atlas,
                effect_label_index=labels,
                output_root=root / "plan",
            )
            self.assertEqual(manifest["state"], "blocked")
            self.assertFalse(manifest["gpu_confirmation_ready"])
            self.assertEqual(manifest["candidate_supported_fold_count"], 7)
            self.assertEqual(manifest["selector_identifiable_fold_count"], 0)

    def test_ambiguous_checkpoint_signs_do_not_count_as_identifiable_candidates(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = self._write_atlas(root)
            environments = json.loads(CONFIG.read_text())["environments"]
            rows = [
                {
                    "environment": environment,
                    "primitive": primitive,
                    "settled": True,
                    "positive": True,
                }
                for environment in environments
                for primitive in ("shared_a", "shared_b")
            ]
            rows.append(
                {
                    "environment": environments[0],
                    "primitive": "shared_b",
                    "settled": True,
                    "positive": False,
                }
            )
            labels = root / "labels.json"
            labels.write_text(json.dumps({"labels": rows}), encoding="utf-8")

            manifest = build_selector_ablation_plan(
                config_path=CONFIG,
                fingerprint_atlas_root=atlas,
                effect_label_index=labels,
                output_root=root / "plan",
            )

            self.assertEqual(manifest["state"], "blocked")
            self.assertEqual(manifest["selector_identifiable_fold_count"], 7)
            report = json.loads((root / "plan" / "selector-ablation-plan.json").read_text())
            self.assertEqual(report["ambiguous_candidates_by_environment"][environments[0]], ["shared_b"])
            self.assertEqual(report["fold_candidate_support"][environments[0]], ["shared_a"])

    def _write_atlas(self, root: Path) -> Path:
        atlas = root / "atlas"
        atlas.mkdir()
        environments = json.loads(CONFIG.read_text())["environments"]
        selectors = ["environment_label", "static_probe", "raw_response", "irg"]
        (atlas / "manifest.json").write_text(
            json.dumps(
                {
                    "state": "ready",
                    "environment_count": 8,
                    "measurement_complete_count": 8,
                    "locality_calibrated_count": 8,
                }
            ),
            encoding="utf-8",
        )
        (atlas / "fingerprint-atlas.json").write_text(json.dumps({"state": "ready"}), encoding="utf-8")
        rows = [
            {"environment": environment, "selector": selector, "features": [1.0]}
            for environment in environments
            for selector in selectors
        ]
        (atlas / "selector-input-projections.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return atlas


if __name__ == "__main__":
    unittest.main()
