from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.cosmos3_directional_settlement import (
    Cosmos3DirectionalSettlementError,
    settle_cosmos3_fingerprint_pair,
    settle_cosmos3_directional_probe,
)


class Cosmos3DirectionalSettlementTests(unittest.TestCase):
    def test_licenses_aligned_dev_and_accept_jacobians(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, accept = self._inputs(
                root, dev_jacobian=[[-1.0], [-0.1]], accept_jacobian=[[-2.0], [-0.2]]
            )
            manifest = settle_cosmos3_directional_probe(
                dev_selection_root=dev,
                accept_fingerprint_root=accept,
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "settled_licensed")
            self.assertAlmostEqual(manifest["alignment_error"], 0.0)

    def test_abstains_when_accept_jacobian_reverses(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, accept = self._inputs(
                root, dev_jacobian=[[-1.0], [-0.1]], accept_jacobian=[[2.0], [0.2]]
            )
            manifest = settle_cosmos3_directional_probe(
                dev_selection_root=dev,
                accept_fingerprint_root=accept,
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "settled_abstained")
            self.assertAlmostEqual(manifest["alignment_error"], 2.0)
            report = json.loads((root / "out/directional-probe-settlement.json").read_text())
            self.assertEqual(report["abstention_reasons"], ["dev_accept_jacobian_aligned"])
            self.assertFalse(report["cross_backbone_transfer_eligible"])

    def test_rejects_campaign_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, accept = self._inputs(root, dev_jacobian=[[-1.0]], accept_jacobian=[[-1.0]])
            campaign_path = accept / "input-campaign.json"
            campaign = json.loads(campaign_path.read_text())
            campaign["acceptance"]["maximum_dev_accept_alignment_error"] = 2.0
            campaign_path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(
                Cosmos3DirectionalSettlementError, "CAMPAIGN_SHA_MISMATCH"
            ):
                settle_cosmos3_directional_probe(
                    dev_selection_root=dev,
                    accept_fingerprint_root=accept,
                    output_root=root / "out",
                )

    def test_settles_direct_fingerprint_pair_without_dev_subset_adapter(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev, accept = self._direct_inputs(
                root,
                dev_jacobian=[[-1.0], [-0.1]],
                accept_jacobian=[[2.0], [0.2]],
            )
            manifest = settle_cosmos3_fingerprint_pair(
                dev_fingerprint_root=dev,
                accept_fingerprint_root=accept,
                output_root=root / "out",
            )
            self.assertEqual(manifest["state"], "settled_abstained")
            self.assertAlmostEqual(manifest["alignment_error"], 2.0)
            report = json.loads((root / "out/directional-probe-settlement.json").read_text())
            self.assertEqual(report["abstention_reasons"], ["dev_accept_jacobian_aligned"])
            self.assertIn("dev_manifest_sha256", report["inputs"])

    def _direct_inputs(
        self,
        root: Path,
        *,
        dev_jacobian: list[list[float]],
        accept_jacobian: list[list[float]],
    ) -> tuple[Path, Path]:
        dev = root / "direct-dev"
        accept = root / "direct-accept"
        dev.mkdir()
        accept.mkdir()
        campaign = {
            "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
            "campaign_id": "direct-positive",
            "probe": {"probe_id": "scale"},
            "locality_admission": {"maximum_residual": 0.5},
            "acceptance": {
                "maximum_dev_accept_alignment_error": 0.5,
                "require_accept_locality_admission": True,
            },
        }
        campaign_bytes = json.dumps(campaign, sort_keys=True)
        for destination in (dev, accept):
            (destination / "input-campaign.json").write_text(campaign_bytes)
            (destination / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-target-local-fingerprint-manifest",
                        "state": "ready",
                    }
                )
            )
        (dev / "target-local-fingerprint.json").write_text(
            json.dumps(self._fingerprint("pilot", "dev", dev_jacobian, "direct-positive"))
        )
        (accept / "target-local-fingerprint.json").write_text(
            json.dumps(self._fingerprint("paper", "accept", accept_jacobian, "direct-positive"))
        )
        return dev, accept

    def _inputs(
        self,
        root: Path,
        *,
        dev_jacobian: list[list[float]],
        accept_jacobian: list[list[float]],
    ) -> tuple[Path, Path]:
        dev = root / "dev"
        accept = root / "accept"
        dev.mkdir()
        accept.mkdir()
        campaign = {
            "artifact_type": "verdiwm-cosmos3-fingerprint-campaign",
            "campaign_id": "positive",
            "probe": {"probe_id": "scale"},
            "locality_admission": {"maximum_residual": 0.5},
            "acceptance": {
                "maximum_dev_accept_alignment_error": 0.5,
                "require_accept_locality_admission": True,
            },
        }
        campaign_bytes = json.dumps(campaign, sort_keys=True)
        (dev / "input-successor-campaign.json").write_text(campaign_bytes)
        (accept / "input-campaign.json").write_text(campaign_bytes)
        dev_fingerprint = self._fingerprint("pilot", "dev", dev_jacobian)
        accept_fingerprint = self._fingerprint("paper", "accept", accept_jacobian)
        (dev / "selected-dev-fingerprint.json").write_text(json.dumps(dev_fingerprint))
        (accept / "target-local-fingerprint.json").write_text(json.dumps(accept_fingerprint))
        (dev / "directional-probe-evolution.json").write_text(json.dumps({"state": "dev_selected"}))
        (dev / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "verdiwm-cosmos3-directional-probe-evolution-manifest",
                    "state": "dev_selected",
                }
            )
        )
        (accept / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "verdiwm-cosmos3-target-local-fingerprint-manifest",
                    "state": "ready",
                }
            )
        )
        return dev, accept

    @staticmethod
    def _fingerprint(
        protocol: str,
        split: str,
        jacobian: list[list[float]],
        campaign_id: str = "positive",
    ) -> dict[str, object]:
        return {
            "artifact_type": "verdiwm-cosmos3-target-local-fingerprint",
            "campaign_id": campaign_id,
            "protocol": protocol,
            "split": split,
            "chart": {
                "goal_schema": "goal",
                "outcome_names": ["quality"] * len(jacobian),
                "intervention_names": ["scale"],
                "jacobian": jacobian,
            },
            "locality_admission": {
                "state": "passed",
                "path_residuals": {"scale": 0.2},
            },
        }


if __name__ == "__main__":
    unittest.main()
