from __future__ import annotations

import json
from pathlib import Path
import unittest

from wmloop.contracts import validate_document


ROOT = Path(__file__).resolve().parents[1]


class Cosmos3ActionDimensionAnisotropyProtocolTests(unittest.TestCase):
    def test_split_v3_is_schema_valid_and_disjoint_from_retired_windows(self) -> None:
        retired = []
        for name in (
            "cosmos3_forward_dynamics_split_v1.json",
            "cosmos3_forward_dynamics_split_v2.json",
        ):
            payload = json.loads((ROOT / "configs/goal" / name).read_text())
            retired.extend(row for split in ("dev", "accept") for row in payload[split])
        current = json.loads(
            (ROOT / "configs/goal/cosmos3_forward_dynamics_split_v3.json").read_text()
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

    def test_campaign_freezes_new_axis_accept_and_original_thresholds(self) -> None:
        campaign = json.loads(
            (
                ROOT
                / "configs/experiments/cosmos3_irg_calibration_action_dimension_anisotropy_v3.json"
            ).read_text()
        )
        self.assertEqual(campaign["probe"]["probe_id"], "action_dimension_anisotropy")
        self.assertEqual(campaign["probe"]["doses"], [-0.1, 0.0, 0.1])
        self.assertEqual(campaign["locality_admission"]["maximum_residual"], 0.5)
        self.assertEqual(campaign["acceptance"]["maximum_dev_accept_alignment_error"], 0.5)
        self.assertFalse(campaign["selection_policy"]["accept_data_used_for_selection"])
        self.assertEqual(campaign["predecessor_campaign"]["settlement_state"], "settled_abstained")

    def test_registry_v2_keeps_new_probe_diagnostic_only(self) -> None:
        registry = json.loads(
            (ROOT / "configs/probes/cosmos3_acwm_forward_dynamics_v2.json").read_text()
        )
        rows = {row["id"]: row for row in registry["probes"]}
        self.assertEqual(rows["action_dimension_anisotropy"]["role"], "diagnostic")
        self.assertEqual(rows["rollout_video_psnr"]["role"], "verdict")

    def test_prediction_receipt_contract_admits_new_probe_and_dose_unit(self) -> None:
        schema = json.loads(
            (ROOT / "configs/schemas/cosmos3_prediction_receipt.schema.json").read_text()
        )
        intervention = schema["properties"]["intervention"]["properties"]
        self.assertIn("action_dimension_anisotropy", intervention["probe_id"]["enum"])
        self.assertIn(
            "relative_action_dimension_contrast_balance",
            intervention["dose_unit"]["enum"],
        )


if __name__ == "__main__":
    unittest.main()
