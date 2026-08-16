from __future__ import annotations

import json
from pathlib import Path
import unittest

from wmloop.constitution import ConstitutionalFreezeError, verify_constitutional_freeze
from wmloop.contracts import load_yaml_document, validate_document


ROOT = Path(__file__).resolve().parents[1]


class CtrlWorldPredictiveInstanceTests(unittest.TestCase):
    def test_current_instance_is_acwm_predictive_and_constitution_is_frozen(self) -> None:
        instance = json.loads(
            (ROOT / "configs/backbones/ctrl_world_predictive_quality_pilot_v2.json").read_text(encoding="utf-8")
        )
        validate_document("backbone_instance", instance, root=ROOT)
        self.assertEqual(instance["backbone_family"], "ctrl_world")
        surfaces = {row["surface_id"]: row for row in instance["surfaces"]}
        self.assertIn("model_repo", surfaces)
        self.assertIn("experiment_repo", surfaces)
        self.assertTrue(any("Downstream task success" in value for value in instance["invariants"]))
        goal = load_yaml_document(ROOT / "configs/goal/ctrl_world_predictive_quality_pilot_v2.yaml")
        validate_document("goal_spec", goal, root=ROOT)
        self.assertNotIn("action_success", goal["metric_family"])
        freeze = json.loads(
            (ROOT / "configs/constitution/ctrl_world_predictive_quality_pilot_v2.freeze.json").read_text(
                encoding="utf-8"
            )
        )
        verify_constitutional_freeze(freeze, root=ROOT)

    def test_historical_v1_freeze_detects_verifier_drift(self) -> None:
        freeze = json.loads(
            (ROOT / "configs/constitution/ctrl_world_predictive_quality_pilot_v1.freeze.json").read_text(
                encoding="utf-8"
            )
        )
        with self.assertRaisesRegex(
            ConstitutionalFreezeError,
            "CONSTITUTION_ENTRY_MISMATCH:wmloop/experiments/ctrl_world_fingerprint.py",
        ):
            verify_constitutional_freeze(freeze, root=ROOT)

    def test_ctrl_world_adapter_declares_candidate_compilation_inputs(self) -> None:
        profile_path = ROOT / "configs/adapters/ctrl_world_predictive_v2.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        validate_document("adapter_profile", profile, root=ROOT)
        self.assertEqual(
            profile["candidate_catalog"],
            "configs/methods/ctrl_world_predictive_v1.json",
        )
        self.assertIn("settlement_manifest_candidates", profile)
        parameters = {row["parameter"] for row in profile["asset_bindings"]}
        self.assertIn("--droid_subset_path", parameters)
        self.assertIn("--data_stat_path", parameters)
        local_share_prefix = "/" + "share" + "/project/"
        self.assertNotIn(local_share_prefix, profile_path.read_text(encoding="utf-8"))

        catalog_path = ROOT / profile["candidate_catalog"]
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        validate_document("method_candidate_catalog", catalog, root=ROOT)
        self.assertEqual(catalog["candidates"][0]["primitive_reference"], "first_frame_anchor")
        environment = catalog["candidates"][0]["candidate_template"]["stages"][0][
            "environment"
        ]
        self.assertEqual(environment["PYTHONPATH"], "{control_root}")
        self.assertEqual(
            catalog["capability_gaps"][0]["reason"],
            "ADAPTER_RECIPE_NOT_IMPLEMENTED",
        )


if __name__ == "__main__":
    unittest.main()
