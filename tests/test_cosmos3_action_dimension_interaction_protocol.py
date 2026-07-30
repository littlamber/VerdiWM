from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.audit_cosmos3_window_freeze import _audit_passed
from wmloop.contracts import validate_document


ROOT = Path(__file__).resolve().parents[1]


class Cosmos3ActionDimensionInteractionProtocolTests(unittest.TestCase):
    def test_input_freeze_audit_treats_uninspected_outcomes_as_required(self) -> None:
        checks = {
            "selected_windows_disjoint": True,
            "input_shapes_valid": True,
            "outcomes_inspected_before_freeze": False,
        }
        self.assertTrue(_audit_passed(checks))
        self.assertFalse(_audit_passed({**checks, "outcomes_inspected_before_freeze": True}))
        self.assertFalse(_audit_passed({**checks, "input_shapes_valid": False}))

    def test_split_v4_is_schema_valid_and_disjoint_from_all_retired_windows(self) -> None:
        retired = []
        for name in (
            "cosmos3_forward_dynamics_split_v1.json",
            "cosmos3_forward_dynamics_split_v2.json",
            "cosmos3_forward_dynamics_split_v3.json",
        ):
            payload = json.loads((ROOT / "configs/goal" / name).read_text())
            retired.extend(row for split in ("dev", "accept") for row in payload[split])
        current = json.loads(
            (ROOT / "configs/goal/cosmos3_forward_dynamics_split_v4.json").read_text()
        )
        validate_document("cosmos3_forward_dynamics_split", current, root=ROOT)

        retired_windows = [
            set(range(int(row["sample_index"]), int(row["sample_index"]) + 17))
            for row in retired
        ]
        current_rows = [row for split in ("dev", "accept") for row in current[split]]
        current_windows = [
            set(range(int(row["sample_index"]), int(row["sample_index"]) + 17))
            for row in current_rows
        ]
        for position, window in enumerate(current_windows):
            self.assertTrue(all(window.isdisjoint(old) for old in retired_windows))
            self.assertTrue(
                all(window.isdisjoint(other) for other in current_windows[position + 1 :])
            )

    def test_campaign_freezes_off_diagonal_axis_before_accept4(self) -> None:
        campaign = json.loads(
            (
                ROOT
                / "configs/experiments/cosmos3_irg_calibration_action_dimension_interaction_v4.json"
            ).read_text()
        )
        self.assertEqual(campaign["probe"]["probe_id"], "action_dimension_interaction")
        self.assertEqual(campaign["probe"]["doses"], [-0.05, 0.0, 0.05])
        self.assertIn("centered_action_energy_preserved", campaign["probe"]["invariants"])
        self.assertEqual(campaign["locality_admission"]["maximum_residual"], 0.5)
        self.assertEqual(campaign["acceptance"]["maximum_dev_accept_alignment_error"], 0.5)
        self.assertFalse(campaign["selection_policy"]["accept_data_used_for_selection"])
        self.assertEqual(campaign["predecessor_campaign"]["settlement_state"], "settled_abstained")

    def test_registry_and_receipt_contract_admit_interaction_as_diagnostic(self) -> None:
        registry = json.loads(
            (ROOT / "configs/probes/cosmos3_acwm_forward_dynamics_v3.json").read_text()
        )
        rows = {row["id"]: row for row in registry["probes"]}
        self.assertEqual(rows["action_dimension_interaction"]["role"], "diagnostic")
        schema = json.loads(
            (ROOT / "configs/schemas/cosmos3_prediction_receipt.schema.json").read_text()
        )
        intervention = schema["properties"]["intervention"]["properties"]
        self.assertIn("action_dimension_interaction", intervention["probe_id"]["enum"])
        self.assertIn("radians_action_dimension_coupling", intervention["dose_unit"]["enum"])


if __name__ == "__main__":
    unittest.main()
