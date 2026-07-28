from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export" / "verdiwm_paper_experiment_matrix.py"
CONFIG = ROOT / "configs" / "experiments" / "verdiwm_iclr_evidence_matrix_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("verdiwm_paper_experiment_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerdiWMPaperExperimentMatrixTests(unittest.TestCase):
    def test_matrix_encodes_reference_and_cross_backbone_claim_boundaries(self) -> None:
        module = _load_module()
        payload = module.load_matrix(CONFIG)
        studies = {item["study_id"]: item for item in payload["studies"]}

        self.assertEqual(len(payload["reference_instance"]["environments"]), 8)
        self.assertFalse(studies["S1_reference_selector_ablation"]["cross_backbone_required"])
        self.assertTrue(studies["S2_cross_backbone_lobo"]["cross_backbone_required"])
        self.assertEqual(payload["external_backbone_policy"]["minimum_external_targets"], 2)
        self.assertEqual(
            payload["external_backbone_policy"]["required_targets"],
            ["ctrl_world", "cosmos3"],
        )

    def test_export_writes_reviewable_csv_latex_and_manifest(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = module.export_matrix(config_path=CONFIG, output_root=root)

            self.assertEqual(manifest["study_count"], 8)
            self.assertEqual(manifest["must_have_count"], 7)
            self.assertTrue((root / "evidence-matrix.md").is_file())
            self.assertTrue((root / "tables" / "evidence-matrix.csv").is_file())
            self.assertTrue((root / "tables" / "evidence-matrix.tex").is_file())
            persisted = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["files"], manifest["files"])


if __name__ == "__main__":
    unittest.main()
