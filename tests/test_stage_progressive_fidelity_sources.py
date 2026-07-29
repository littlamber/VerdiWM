from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export.stage_progressive_fidelity_sources import (
    stage_progressive_fidelity_sources,
)


class ProgressiveFidelitySourceStageTests(unittest.TestCase):
    def test_stages_screen_gate_and_confirmation_json_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "screen-campaign"
            screen_env = campaign / "envs/push_cube"
            screen_env.mkdir(parents=True)
            _write(campaign / "status.json", {"state": "ready"})
            screen_report = _write(screen_env / "report.json", {"actual_gpu_hours": 1.0})
            screen = _write(
                screen_env / "manifest.json",
                {"report_path": str(screen_report)},
            )
            gate = _write(root / "gate.json", {"state": "ready"})

            confirm_campaign = root / "confirm-campaign"
            confirm_env = confirm_campaign / "envs/push_cube"
            retained = confirm_env / "retained_training"
            retained.mkdir(parents=True)
            _write(confirm_campaign / "status.json", {"state": "ready"})
            confirm_report = _write(confirm_env / "report.json", {"actual_gpu_hours": 2.0})
            _write(confirm_env / "manifest.json", {"report_path": str(confirm_report)})
            checkpoint_manifest = _write(retained / "manifest.json", {"state": "ready"})
            confirm_gate = _write(root / "confirm-gate.json", {"state": "ready"})
            finalizer_root = root / "finalizer"
            finalizer_root.mkdir()
            _write(finalizer_root / "manifest.json", {"state": "ready"})
            finalizer = _write(
                finalizer_root / "checkpoint-ladder-finalization.json",
                {
                    "checkpoint_manifest": str(checkpoint_manifest),
                    "selection": {
                        "records": [
                            {
                                "checkpoint_step": 800,
                                "official_manifest_path": str(confirm_gate),
                            }
                        ]
                    },
                },
            )
            report = _write(
                root / "s6.json",
                {
                    "artifact_type": "verdiwm-progressive-fidelity-efficiency",
                    "study_id": "S6_progressive_fidelity_efficiency",
                    "candidate_count": 1,
                    "candidate_rows": [
                        {
                            "environment": "push_cube",
                            "primitive": "next_forcing",
                            "seed": 1,
                            "screen_manifest": str(screen),
                            "gate_manifest": str(gate),
                            "confirmation_manifest": str(finalizer),
                        }
                    ],
                },
            )

            result = stage_progressive_fidelity_sources(
                report_path=report,
                output_root=root / "output",
            )

            self.assertEqual(result["state"], "ready")
            self.assertEqual(result["reference_count"], 11)
            source_map = json.loads((root / "output/source-map.json").read_text())
            self.assertEqual(source_map["candidate_count"], 1)
            for row in source_map["references"]:
                self.assertTrue((root / "output" / row["object_ref"]).is_file())


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
