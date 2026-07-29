from __future__ import annotations

import json
from pathlib import Path
import unittest

from wmloop.contracts import validate_document


ROOT = Path(__file__).resolve().parents[1]


class Cosmos3SplitV2ProtocolTests(unittest.TestCase):
    def test_fresh_windows_are_disjoint_and_schema_valid(self) -> None:
        old = json.loads((ROOT / "configs/goal/cosmos3_forward_dynamics_split_v1.json").read_text())
        new = json.loads((ROOT / "configs/goal/cosmos3_forward_dynamics_split_v2.json").read_text())
        validate_document("cosmos3_forward_dynamics_split", new, root=ROOT)

        old_indices = {row["sample_index"] for split in ("dev", "accept") for row in old[split]}
        new_rows = [row for split in ("dev", "accept") for row in new[split]]
        new_indices = {row["sample_index"] for row in new_rows}
        self.assertTrue(old_indices.isdisjoint(new_indices))
        self.assertEqual(len(new_rows), len(new_indices))

        windows = [set(range(index, index + 17)) for index in sorted(new_indices)]
        for position, window in enumerate(windows):
            for other in windows[position + 1 :]:
                self.assertTrue(window.isdisjoint(other))

    def test_narrow_campaign_freezes_new_accept_and_original_thresholds(self) -> None:
        campaign = json.loads(
            (ROOT / "configs/experiments/cosmos3_irg_calibration_translation_positive_narrow_v2.json").read_text()
        )
        self.assertEqual(campaign["heldout_split_ref"], "configs/goal/cosmos3_forward_dynamics_split_v2.json")
        self.assertEqual(campaign["probe"]["doses"], [0.0, 0.00625, 0.0125])
        self.assertEqual(campaign["locality_admission"]["maximum_residual"], 0.5)
        self.assertEqual(campaign["acceptance"]["maximum_dev_accept_alignment_error"], 0.5)
        self.assertFalse(campaign["selection_policy"]["accept_data_used_for_selection"])
        self.assertEqual(campaign["predecessor_campaign"]["settlement_state"], "settled_abstained")


if __name__ == "__main__":
    unittest.main()
