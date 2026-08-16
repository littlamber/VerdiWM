from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluate_ctrl_world_directional_selector_canary import evaluate


class DirectionalSelectorCanaryTests(unittest.TestCase):
    def test_narrow_slope_selects_positive_wider_candidate(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = {
                "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-atlas",
                "state": "ready",
                "campaign_id": "narrow",
                "outcome_names": ["quality"],
                "outcome_weights": [1.0],
                "contexts": [
                    {
                        "context": {"context_id": "ctx", "episode_id": "1", "start_idx": 0, "seeds": [1, 2, 3]},
                        "chart": {
                            "outcome_names": ["quality"],
                            "intervention_names": ["probe"],
                            "jacobian": [[2.0]],
                            "covariance": [[0.01]],
                            "repeat_count": 3,
                        },
                        "locality_admission": {"supported_local_paths": ["probe"]},
                    }
                ],
            }
            atlas_path = root / "atlas.json"
            atlas_path.write_text(json.dumps(atlas), encoding="utf-8")
            measurements = []
            for dose in (-0.025, 0.0, 0.025):
                for seed in (1, 2, 3):
                    measurements.append(
                        {
                            "dose": dose,
                            "identity": {"context_id": "ctx", "episode_id": "1", "start_idx": 0, "seed": seed},
                            "outcomes": {"quality": 1.0 + 2.0 * dose},
                        }
                    )
            candidate = {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
                "state": "ready",
                "probe_id": "probe",
                "outcome_names": ["quality"],
                "hook_activation": {"state": "passed"},
                "measurements": measurements,
            }
            candidate_path = root / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            manifest = evaluate(
                fingerprint_atlas_path=atlas_path,
                candidate_result_paths=[candidate_path],
                output_root=root / "out",
            )
            self.assertEqual(manifest["decision"], "promising_dev_only")
            report = json.loads((root / "out" / "directional-selector-canary.json").read_text())
            self.assertEqual(report["contexts"][0]["selected_candidate"]["dose"], 0.025)
            self.assertGreater(report["metrics"]["regret_reduction_vs_uniform"], 0.0)
            self.assertEqual(report["metrics"]["positive_selected_effect_lcb_rate"], 1.0)
            self.assertEqual(report["routing_readiness"]["state"], "not_licensed")


if __name__ == "__main__":
    unittest.main()
