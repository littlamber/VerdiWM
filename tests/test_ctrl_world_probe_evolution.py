from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.probe_evolution import build_probe_evolution_proposal


class CtrlWorldProbeEvolutionTests(unittest.TestCase):
    def test_records_failed_scale_probe_and_admits_novel_reversible_successor(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            failures = []
            for index, residual in enumerate((4.8, 1.8), start=1):
                path = root / f"failure-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "campaign_id": f"failed-{index}",
                            "chart": {"intervention_names": ["action_conditioning_scale"]},
                            "locality_admission": {
                                "state": "failed",
                                "cross_backbone_transfer_eligible": False,
                                "maximum_residual": 0.5,
                                "path_residuals": {"action_conditioning_scale": residual},
                                "supported_local_paths": [],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                failures.append(path)
            successor = root / "successor.json"
            successor.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-ctrl-world-fingerprint-campaign",
                        "probe": {"probe_id": "action_embedding_temporal_mix", "scope": "inference_only", "reversible": True},
                        "outcomes": [
                            {"name": "rollout_video_psnr"},
                            {"name": "negative_rollout_video_l1"},
                            {"name": "negative_segment_final_mae"},
                            {"name": "negative_segment_view_pair_mae"},
                            {"name": "negative_segment_view_fused_mae"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            build_probe_evolution_proposal(
                failed_fingerprints=failures,
                successor_campaign=successor,
                output_root=root / "proposal",
            )
            report = json.loads((root / "proposal/probe-evolution-proposal.json").read_text(encoding="utf-8"))
            self.assertEqual(report["retired_probe_ids"], ["action_conditioning_scale"])
            self.assertEqual(report["successor_probe"]["probe_id"], "action_embedding_temporal_mix")
            self.assertEqual(report["counterexample_count"], 2)


if __name__ == "__main__":
    unittest.main()
