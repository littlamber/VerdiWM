from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.evaluate.local_maturity_audit import run_local_maturity_audit


class LocalMaturityAuditTests(unittest.TestCase):
    def test_cpu_local_smoke_is_ready_and_claim_bounded(self) -> None:
        with TemporaryDirectory() as temporary:
            report = run_local_maturity_audit(repo_root=Path(__file__).resolve().parents[1], output_root=Path(temporary) / "audit")
            self.assertEqual(report["state"], "ready")
            self.assertEqual(report["operational_state"], "ready")
            self.assertEqual(report["evaluator_state"], "ready")
            self.assertEqual(report["receipt_state"], "ready")
            self.assertEqual(report["knowledge_state"], "ready")
            self.assertEqual(report["blocked_gate_count"], 0)
            self.assertFalse(report["side_effects"]["gpu_execution_started"])
            self.assertIn("does not establish", report["claim_boundary"])
            self.assertTrue((Path(temporary) / "audit" / "local-maturity-audit.json").is_file())


if __name__ == "__main__":
    unittest.main()
