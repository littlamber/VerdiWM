from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.settle_ctrl_world_directional_data_admission import settle


class DirectionalDataAdmissionTests(unittest.TestCase):
    def _write(self, path: Path, payload: dict[str, object]) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _campaign(self, campaign_id: str) -> dict[str, object]:
        return {
            "artifact_type": "verdiwm-ctrl-world-local-fingerprint-campaign",
            "campaign_id": campaign_id,
            "protocol": {
                "candidate_radius": 0.025,
                "selector_confidence_critical_value": 4.303,
                "minimum_predicted_gain_lcb": 0.0,
            },
            "data_admission_gate": {"classes": [-1, 0, 1], "minimum_contexts_per_class": 2},
        }

    def _selector(self, campaign_id: str, targets: list[int]) -> dict[str, object]:
        contexts = []
        for index, target in enumerate(targets):
            selected = None
            action = "abstain"
            if target:
                action = "execute"
                selected = {
                    "dose": 0.025 * target,
                    "predicted_weighted_gain_lcb": 0.1,
                }
            contexts.append(
                {
                    "context": {
                        "context_id": f"{campaign_id}-ctx-{index}",
                        "episode_id": str(index),
                        "start_idx": index,
                        "seeds": [101, 202, 303],
                    },
                    "selector_action": action,
                    "selected_candidate": selected,
                }
            )
        return {
            "artifact_type": "verdiwm-ctrl-world-frozen-directional-selector",
            "state": "frozen",
            "fingerprint_campaign_id": campaign_id,
            "candidate_radius": 0.025,
            "confidence_z": 4.303,
            "minimum_predicted_gain_lcb": 0.0,
            "effect_labels_observed": False,
            "contexts": contexts,
        }

    def test_blocks_when_one_class_is_underrepresented(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = self._write(root / "campaign.json", self._campaign("v1"))
            selector = self._write(root / "selector.json", self._selector("v1", [-1, -1, 0, 0, 1]))
            manifest = settle(
                campaign_paths=[campaign],
                selector_paths=[selector],
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "blocked")
            self.assertFalse(manifest["candidate_training_licensed"])
            self.assertEqual(manifest["class_counts"], {"-1": 2, "0": 2, "1": 1})

    def test_admits_compatible_nonoverlapping_batches(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            campaigns = [
                self._write(root / f"campaign-{name}.json", self._campaign(name))
                for name in ("v1", "v2")
            ]
            selectors = [
                self._write(root / f"selector-{name}.json", self._selector(name, [-1, 0, 1]))
                for name in ("v1", "v2")
            ]
            manifest = settle(
                campaign_paths=campaigns,
                selector_paths=selectors,
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "admitted")
            self.assertTrue(manifest["candidate_training_licensed"])
            self.assertEqual(manifest["class_counts"], {"-1": 2, "0": 2, "1": 2})


if __name__ == "__main__":
    unittest.main()
