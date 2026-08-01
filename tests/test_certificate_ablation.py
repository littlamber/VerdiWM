from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.certificate_ablation import run_certificate_ablation


class CertificateAblationTests(unittest.TestCase):
    def test_replays_certificate_without_changing_selector_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "tables").mkdir(parents=True)
            cells = [
                {"trial_id": "a", "selector": "irg", "state": "abstained", "abstention_reason": "transfer_certificate_failed", "target_environment": "a", "seed": 1},
                {"trial_id": "b", "selector": "irg", "state": "abstained", "abstention_reason": "transfer_certificate_failed", "target_environment": "b", "seed": 1},
                {"trial_id": "c", "selector": "irg", "state": "abstained", "abstention_reason": "candidate_probe_coverage_incomplete", "target_environment": "c", "seed": 1},
            ]
            (source / "selector-replay.json").write_text(json.dumps({
                "artifact_type": "verdiwm-acwm-selector-cpu-replay",
                "transfer_certificate_enabled": True,
                "cells": cells,
            }))
            with (source / "tables/candidates.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trial_id", "selector", "rank", "primitive", "target_positive"])
                writer.writeheader()
                writer.writerow({"trial_id": "a", "selector": "irg", "rank": 1, "primitive": "p", "target_positive": "False"})
                writer.writerow({"trial_id": "b", "selector": "irg", "rank": 1, "primitive": "q", "target_positive": "True"})
            manifest = run_certificate_ablation(
                selector_bundle_root=source,
                output_root=root / "output",
            )
            self.assertEqual(manifest["certificate_prevented_negative_count"], 1)
            report = json.loads((root / "output/certificate-ablation.json").read_text())
            on, off = report["arms"]
            self.assertEqual(on["coverage"], 0.0)
            self.assertEqual(off["coverage"], 2 / 3)
            self.assertEqual(off["negative_transfer_rate"], 0.5)
