import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_wan22_droid_budget import audit


class Wan22DroidBudgetTests(unittest.TestCase):
    def test_audit_sums_runner_receipts_and_enforces_cap(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name, hours in (("a", 1.5), ("b", 2.0)):
                path = root / name
                path.mkdir()
                (path / "training_receipt.json").write_text(json.dumps({"artifact_type": "verdiwm-wan22-droid-training-receipt", "gpu_hours": hours, "budget_gpu_hours": 4.0}))
            result = audit(root, cap_gpu_hours=4.0)
            self.assertEqual(result["state"], "within_budget")
            self.assertEqual(result["consumed_gpu_hours"], 3.5)

    def test_audit_blocks_over_cap(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "training_receipt.json").write_text(json.dumps({"artifact_type": "verdiwm-wan22-droid-training-receipt", "gpu_hours": 4.1, "budget_gpu_hours": 4.1}))
            self.assertEqual(audit(root, cap_gpu_hours=4.0)["state"], "blocked")


if __name__ == "__main__":
    unittest.main()
