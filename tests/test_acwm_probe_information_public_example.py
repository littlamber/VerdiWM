from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "examples" / "acwm_probe_information_collision_s4_v1"


class AcwmProbeInformationPublicExampleTests(unittest.TestCase):
    def test_public_snapshot_is_path_free_and_integrity_checked(self) -> None:
        summary = json.loads((ROOT / "summary.json").read_text())
        self.assertEqual(summary["state"], "partial")
        self.assertEqual(summary["observed_condition_count"], 3)
        self.assertEqual(summary["random_probe_expansion"]["execution_count"], 80)
        self.assertEqual(summary["collision"]["labeled_case_count"], 8)
        self.assertEqual(summary["collision"]["accepted_coverage"], 0.0)
        self.assertIsNone(summary["collision"]["post_evolution_collision_rate"])
        self.assertEqual(len(summary["prior_probe_evolution_attempts"]), 3)
        self.assertEqual(
            summary["prior_probe_evolution_attempts"][-1]["probe_id"],
            "self_rollout_horizon_recovery_curvature",
        )
        self.assertTrue(
            summary["prior_probe_evolution_attempts"][-1]["protocol_matched_to_reference"]
        )
        for path in ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn("/" + "mnt" + "/", path.read_text(errors="ignore"))

        manifest = ROOT / "MANIFEST.sha256"
        self.assertTrue(manifest.is_file())
        for line in manifest.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            payload = (ROOT / relative).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
