from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.acwm_selector_public_bundle import build_public_selector_bundle


class AcwmSelectorPublicBundleTests(unittest.TestCase):
    def test_exports_path_free_complete_selector_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            environments = [f"env_{index}" for index in range(8)]
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "expected_environments": environments,
                        "settled_label_count": 16,
                        "settled_positive_count": 8,
                        "settled_negative_count": 8,
                        "labels": [
                            {
                                "environment": environment,
                                "primitive": primitive,
                                "settled": True,
                                "positive": positive,
                                "evidence_ref": f"/{'mnt'}/private/{environment}.json",
                            }
                            for environment in environments
                            for primitive, positive in (("positive", True), ("negative", False))
                        ],
                    }
                ),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "experiment_id": "selector-v1",
                        "fold_count": 8,
                        "selector_count": 4,
                        "seed_count": 3,
                        "planned_trial_count": 96,
                        "candidate_supported_fold_count": 8,
                        "selector_identifiable_fold_count": 8,
                        "fold_candidate_support": {environment: ["negative", "positive"] for environment in environments},
                        "ambiguous_candidates_by_environment": {environment: [] for environment in environments},
                    }
                ),
                encoding="utf-8",
            )
            selector_rows = [
                {
                    "selector": selector,
                    "evaluated_fold_count": 8,
                    "top1_positive_hit": 0.5,
                    "negative_selection": 0.5,
                    "benefit_sign_accuracy": 0.5,
                    "ranking_kendall_tau": 0.0,
                    "selection_regret": 0.25,
                }
                for selector in ("environment_label", "static_probe", "raw_response", "irg")
            ]
            replay = root / "replay.json"
            replay.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "verdiwm-acwm-selector-cpu-replay",
                        "state": "ready",
                        "planned_cell_count": 96,
                        "evaluated_cell_count": 96,
                        "abstained_cell_count": 0,
                        "environment_count": 8,
                        "evaluated_environment_count": 8,
                        "multi_candidate_environment_count": 8,
                        "selector_choice_divergence_environment_count": 0,
                        "selector_choice_divergence_environments": [],
                        "formal_comparison_ready": True,
                        "selection_discrimination_ready": False,
                        "selectors": selector_rows,
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "output"
            build_public_selector_bundle(
                effect_label_index=labels,
                selector_plan=plan,
                selector_replay=replay,
                output_root=output,
            )

            combined = "".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file() and path.suffix in {".json", ".md", ".csv"}
            )
            self.assertNotIn("/" + "mnt" + "/", combined)
            manifest_rows = (output / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
            for row in manifest_rows:
                digest, relative = row.split("  ", 1)
                self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
