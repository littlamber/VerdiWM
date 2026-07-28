from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export" / "acwm_fingerprint_calibration_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("acwm_fingerprint_calibration_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ACWMFingerprintCalibrationAuditTests(unittest.TestCase):
    def test_existing_trials_are_not_mislabeled_as_irg_calibration(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen = root / "screen.csv"
            horizon = root / "horizon.csv"
            screen.write_text(
                "campaign_id,environment,primitive,seed,train_steps\n"
                "c1,push_cube,p1,1,512\n"
                "c2,push_cube,p1,2,512\n"
                "c3,push_cube,p1,3,512\n",
                encoding="utf-8",
            )
            horizon.write_text(
                "campaign_id,environment,primitive,seed,train_steps,horizon\n"
                "c1,push_cube,p1,1,512,16\n"
                "c2,push_cube,p1,2,512,16\n"
                "c3,push_cube,p1,3,512,16\n",
                encoding="utf-8",
            )
            output_root = root / "out"
            module.audit(screen_trials=screen, horizon_metrics=horizon, output_root=output_root)
            report = json.loads((output_root / "fingerprint-calibration-audit.json").read_text(encoding="utf-8"))

            row = report["environments"]["push_cube"]
            self.assertEqual(row["repeated_primitive_seed_counts_ge3"], {"p1": 3})
            self.assertEqual(row["irg_calibration_status"], "candidate_response_inventory_only")
            self.assertFalse(row["has_three_point_dose_sweep"])


if __name__ == "__main__":
    unittest.main()
