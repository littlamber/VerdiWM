from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.cosmos3_directional_probe import (
    Cosmos3DirectionalProbeError,
    build_cosmos3_directional_probe_evolution,
)


class Cosmos3DirectionalProbeTests(unittest.TestCase):
    def test_selects_one_sided_dev_path_without_using_accept(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            successor = self._successor(root, source)
            manifest = build_cosmos3_directional_probe_evolution(
                source_fingerprint_root=source,
                successor_campaign_path=successor,
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "dev_selected")
            report = json.loads((root / "out/directional-probe-evolution.json").read_text())
            self.assertTrue(report["dev_locality_admitted"])
            self.assertLessEqual(report["selected_directional_residual"], 0.5)
            selected = json.loads((root / "out/selected-dev-fingerprint.json").read_text())
            self.assertFalse(selected["locality_admission"]["cross_backbone_transfer_eligible"])

    def test_rejects_threshold_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            successor = self._successor(root, source)
            payload = json.loads(successor.read_text())
            payload["locality_admission"]["maximum_residual"] = 10.0
            successor.write_text(json.dumps(payload))
            with self.assertRaisesRegex(Cosmos3DirectionalProbeError, "THRESHOLD_DRIFT"):
                build_cosmos3_directional_probe_evolution(
                    source_fingerprint_root=source,
                    successor_campaign_path=successor,
                    output_root=root / "out",
                )

    def _source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        campaign = {
            "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
            "campaign_id": "symmetric",
            "probe": {"probe_id": "action_conditioning_scale", "doses": [-0.025, -0.0125, 0.0, 0.0125, 0.025]},
            "outcomes": [
                {"name": "psnr", "source_metric": "psnr", "sign": 1.0, "weight": 1.0},
                {"name": "negative_l1", "source_metric": "l1", "sign": -1.0, "weight": 2.0},
            ],
            "locality_admission": {"maximum_residual": 0.5},
        }
        (source / "input-campaign.json").write_text(json.dumps(campaign))
        rows = []
        for sample, seed, offset in ((0, 101, 0.0), (16, 202, 0.2), (32, 303, -0.1)):
            for dose in campaign["probe"]["doses"]:
                positive_response = [20.0 + offset - 2.0 * dose, -0.1 - 0.01 * dose]
                if dose < 0.0:
                    positive_response[0] += 30.0 * dose * dose * (1 + sample / 16)
                rows.append({"dose": dose, "sample_index": sample, "seed": seed, "outcomes": positive_response})
        (source / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        fingerprint = {
            "artifact_type": "verdiwm-cosmos3-target-local-fingerprint",
            "protocol": "pilot",
            "split": "dev",
            "locality_admission": {"path_residuals": {"action_conditioning_scale": 3.0}},
        }
        (source / "target-local-fingerprint.json").write_text(json.dumps(fingerprint))
        return source

    def _successor(self, root: Path, source: Path) -> Path:
        source_campaign = json.loads((source / "input-campaign.json").read_text())
        path = root / "successor.json"
        payload = {
            "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
            "campaign_id": "positive",
            "probe": {"probe_id": "action_conditioning_scale", "dose_domain": "positive_one_sided_local", "doses": [0.0, 0.0125, 0.025]},
            "outcomes": source_campaign["outcomes"],
            "protocols": {"pilot": {"split": "dev"}, "paper": {"split": "accept"}},
            "locality_admission": {"maximum_residual": 0.5},
            "selection_policy": {"selection_split": "dev", "validation_split": "accept", "accept_data_used_for_selection": False},
            "predecessor_campaign": {
                "campaign_id": "symmetric",
                "fingerprint_sha256": hashlib.sha256((source / "target-local-fingerprint.json").read_bytes()).hexdigest(),
            },
        }
        path.write_text(json.dumps(payload))
        return path


if __name__ == "__main__":
    unittest.main()
