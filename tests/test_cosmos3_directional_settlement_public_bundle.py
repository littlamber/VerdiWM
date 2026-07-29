from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.cosmos3_directional_settlement_public_bundle import (
    export_cosmos3_directional_settlement_public_bundle,
)


class Cosmos3DirectionalSettlementPublicBundleTests(unittest.TestCase):
    def test_exports_final_abstention_without_machine_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev = root / "dev"
            accept = root / "accept"
            settled = root / "settled"
            dev.mkdir()
            accept.mkdir()
            settled.mkdir()
            (dev / "directional-probe-evolution.json").write_text(
                json.dumps({"state": "dev_selected"})
            )
            (dev / "selected-dev-fingerprint.json").write_text(
                json.dumps({"campaign_id": "campaign", "split": "dev"})
            )
            (accept / "bundle.json").write_text(
                json.dumps(
                    {
                        "campaign_id": "campaign",
                        "split": "accept",
                        "locality_admission_state": "passed",
                        "videos": [],
                    }
                )
            )
            (accept / "target-local-fingerprint.json").write_text(
                json.dumps({"campaign_id": "campaign"})
            )
            (settled / "directional-probe-settlement.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-cosmos3-directional-probe-settlement",
                        "state": "settled_abstained",
                        "campaign_id": "campaign",
                        "probe_id": "scale",
                        "cross_backbone_transfer_eligible": False,
                        "abstention_reasons": ["dev_accept_jacobian_aligned"],
                        "locality": {
                            "dev_residual": 0.3,
                            "accept_residual": 0.32,
                            "maximum_residual": 0.5,
                        },
                        "alignment": {
                            "error": 2.0,
                            "maximum_error": 0.5,
                            "dev_jacobian": [-1.0, -0.1],
                            "accept_jacobian": [1.0, 0.1],
                        },
                        "claim_boundary": "diagnostic only",
                    }
                )
            )
            bundle = export_cosmos3_directional_settlement_public_bundle(
                dev_selection_root=dev,
                accept_public_root=accept,
                settlement_root=settled,
                output_root=root / "public",
            )
            self.assertEqual(bundle["settlement_state"], "settled_abstained")
            self.assertFalse(bundle["cross_backbone_transfer_eligible"])
            self.assertTrue((root / "public/figures/dev-accept-alignment.svg").is_file())
            self.assertTrue((root / "public/MANIFEST.sha256").is_file())
            self.assertNotIn("/" + "mnt" + "/", (root / "public/bundle.json").read_text())


if __name__ == "__main__":
    unittest.main()
