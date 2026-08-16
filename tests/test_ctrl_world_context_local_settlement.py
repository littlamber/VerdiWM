from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.settle_ctrl_world_context_local_fingerprints import settle


class ContextLocalSettlementTests(unittest.TestCase):
    def _atlas(self, root: Path, *, campaign: str, radius: float, residuals: dict[str, float]) -> Path:
        root.mkdir()
        report = {
            "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-atlas",
            "state": "ready",
            "campaign_id": campaign,
            "outcome_names": ["quality"],
            "outcome_weights": [1.0],
            "input_receipts": [{"doses": [-radius, 0.0, radius]}],
            "contexts": [
                {
                    "context": {"context_id": "ctx", "episode_id": "1", "start_idx": 0, "seeds": [1, 2, 3]},
                    "chart": {
                        "outcome_names": ["quality"],
                        "intervention_names": ["a", "b"],
                        "jacobian": [[1.0, 2.0]],
                        "covariance": [[0.0, 0.0], [0.0, 0.0]],
                        "repeat_count": 3,
                    },
                    "locality_admission": {"maximum_residual": 0.5, "path_residuals": residuals},
                }
            ],
        }
        (root / "context-local-fingerprint-atlas.json").write_text(json.dumps(report), encoding="utf-8")
        return root

    def test_selects_widest_passing_radius_per_path(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            wide = self._atlas(root / "wide", campaign="wide", radius=0.025, residuals={"a": 0.4, "b": 0.8})
            narrow = self._atlas(root / "narrow", campaign="narrow", radius=0.0125, residuals={"a": 0.1, "b": 0.2})
            manifest = settle(atlas_roots=[wide, narrow], output_root=root / "settled")
            self.assertEqual(manifest["admitted_context_probe_count"], 2)
            report = json.loads((root / "settled" / "settlement.json").read_text())
            selections = {row["probe_id"]: row for row in report["selections"]}
            self.assertEqual(selections["a"]["selected_radius"], 0.025)
            self.assertEqual(selections["b"]["selected_radius"], 0.0125)
            self.assertEqual(report["routing_readiness"]["state"], "not_licensed")


if __name__ == "__main__":
    unittest.main()
