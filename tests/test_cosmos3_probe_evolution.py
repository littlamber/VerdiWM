from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.probe_evolution import build_probe_evolution_proposal


class Cosmos3ProbeEvolutionTests(unittest.TestCase):
    def test_admits_temporal_mix_without_changing_frozen_outcomes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            failures = []
            for index, residual in enumerate((0.9510001507610469, 45.647629841426685), start=1):
                path = root / f"failed-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "campaign_id": f"cosmos3-scale-{index}",
                            "chart": {"intervention_names": ["action_conditioning_scale"]},
                            "locality_admission": {
                                "state": "failed",
                                "cross_backbone_transfer_eligible": False,
                                "maximum_residual": 0.5,
                                "path_residuals": {"action_conditioning_scale": residual},
                                "supported_local_paths": [],
                            },
                        }
                    )
                )
                failures.append(path)
            successor = root / "successor.json"
            successor.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
                        "probe": {
                            "probe_id": "action_embedding_temporal_mix",
                            "scope": "inference_only",
                            "reversible": True,
                        },
                        "outcomes": [
                            {"name": "rollout_video_psnr"},
                            {"name": "negative_rollout_video_l1"},
                            {"name": "negative_final_frame_mae"},
                            {"name": "negative_temporal_difference_mae"},
                        ],
                    }
                )
            )
            build_probe_evolution_proposal(
                failed_fingerprints=failures,
                successor_campaign=successor,
                output_root=root / "proposal",
            )
            report = json.loads((root / "proposal/probe-evolution-proposal.json").read_text())
            self.assertEqual(report["backbone_family"], "Cosmos3 ACWM forward dynamics")
            self.assertEqual(report["retired_probe_ids"], ["action_conditioning_scale"])
            self.assertEqual(report["successor_probe"]["probe_id"], "action_embedding_temporal_mix")


if __name__ == "__main__":
    unittest.main()
