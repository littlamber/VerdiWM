from __future__ import annotations

import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
BASE_REGISTRY = ROOT / "configs/probes/irg_base_v1.json"
OUTCOME_REGISTRY = ROOT / "configs/probes/acwm_v1.json"
JOINT_CAMPAIGN = (
    ROOT / "configs/experiments/acwm_phys_joint_irg_autoregressive_pilot_v1.json"
)


class ProbeTaxonomyTests(unittest.TestCase):
    def test_base_registry_matches_schema_and_canonical_ids(self) -> None:
        registry = json.loads(BASE_REGISTRY.read_text(encoding="utf-8"))
        schema = json.loads(
            (ROOT / "configs/schemas/irg_base_probe_registry.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(registry)
        self.assertEqual(
            [probe["id"] for probe in registry["base_probes"]],
            [
                "action_scaling",
                "controlled_context_retention",
                "first_frame_anchoring_strength",
                "sampler_noise_stress",
            ],
        )

    def test_outcome_diagnostics_are_not_base_interventions(self) -> None:
        base = json.loads(BASE_REGISTRY.read_text(encoding="utf-8"))
        outcome = json.loads(OUTCOME_REGISTRY.read_text(encoding="utf-8"))
        base_ids = {probe["id"] for probe in base["base_probes"]}
        outcome_ids = {probe["id"] for probe in outcome["probes"]}
        self.assertTrue(base_ids.isdisjoint(outcome_ids))
        self.assertEqual(
            outcome_ids,
            {"horizon_curve", "action_following", "appearance_drift", "ood_profile"},
        )

    def test_declared_acwm_paths_exist_in_joint_campaign(self) -> None:
        base = json.loads(BASE_REGISTRY.read_text(encoding="utf-8"))
        campaign = json.loads(JOINT_CAMPAIGN.read_text(encoding="utf-8"))
        admitted = {path["path_name"] for path in campaign["semantic_paths"]}
        declared = {
            path
            for probe in base["base_probes"]
            for path in probe["acwm_v1_admitted_paths"]
        }
        self.assertTrue(declared)
        self.assertTrue(declared.issubset(admitted))
        self.assertEqual(len(admitted), 7)


if __name__ == "__main__":
    unittest.main()
