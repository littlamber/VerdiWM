from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.effect_labels import build_effect_label_index


class ACWMEffectLabelIndexTests(unittest.TestCase):
    def test_indexes_only_complete_official_gate_as_settled(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            valid = root / "reports" / "acwm-autoloop-confirm-official-gate-push_cube-next_forcing-s1-r1"
            valid.mkdir(parents=True)
            (valid / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "environment": "push_cube",
                        "primitive": "next_forcing",
                        "seed": 1,
                        "steps": 50,
                        "candidate_checkpoint_step": 1000,
                        "official_quality_gate": {
                            "state": "pass",
                            "pass": True,
                            "checks": {"psnr": True, "ssim": True},
                            "delta_candidate_minus_baseline": {"psnr": 0.2},
                        },
                    }
                ),
                encoding="utf-8",
            )
            partial = root / "reports" / "acwm-autoloop-confirm-official-gate-stack_cube-next_forcing-s1-r1"
            partial.mkdir(parents=True)
            (partial / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "environment": "stack_cube",
                        "primitive": "next_forcing",
                        "seed": 1,
                        "official_quality_gate": {"state": "pass", "pass": True},
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_effect_label_index(
                reports_root=root / "reports",
                output_root=root / "index",
                expected_environments=("push_cube", "stack_cube"),
            )
            self.assertEqual(manifest["label_count"], 2)
            self.assertEqual(manifest["settled_label_count"], 1)
            report = json.loads((root / "index" / "effect-label-index.json").read_text())
            self.assertEqual(report["missing_environments"], ["stack_cube"])


if __name__ == "__main__":
    unittest.main()
