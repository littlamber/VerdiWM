from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from wmloop.experiments.joint_fingerprint import (
    JointFingerprintError,
    compose_joint_irg_asset,
    condition_schedule,
    fit_joint_fingerprint,
    load_joint_campaign,
    load_joint_sources,
)
from wmloop.geometry.assets import validate_irg_asset


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs" / "experiments" / "acwm_phys_joint_irg_autoregressive_pilot_v1.json"


class AcwmJointFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.joint = load_joint_campaign(CAMPAIGN)
        self.sources = load_joint_sources(self.joint, repo_root=ROOT)

    def test_campaign_matches_static_schema(self) -> None:
        schema = json.loads(
            (ROOT / "configs/schemas/joint_fingerprint_campaign.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(self.joint)

    def test_real_campaign_has_one_baseline_per_seed_and_seventy_five_conditions(self) -> None:
        schedule = condition_schedule(self.joint, self.sources)
        baselines = [row for row in schedule if row["condition_kind"] == "baseline"]

        self.assertEqual(len(self.sources), 6)
        self.assertEqual(len(self.joint["semantic_paths"]), 7)
        self.assertEqual(
            {source["probe"]["generation_mode"] for source in self.sources},
            {"autoregressive"},
        )
        self.assertEqual(len(schedule), 75)
        self.assertEqual(len(baselines), 3)
        self.assertEqual({row["seed"] for row in baselines}, {101, 202, 303})

    def test_mixed_runtime_mode_without_explicit_override_fails_closed(self) -> None:
        broken = copy.deepcopy(self.joint)
        broken["source_campaigns"][0]["generation_mode_override"] = "parallel"

        with self.assertRaisesRegex(JointFingerprintError, "MODE_OVERRIDE_MISSING"):
            condition_schedule(broken, self.sources)

    def test_joint_fit_preserves_path_order_and_emits_full_covariance_asset(self) -> None:
        measurements = self._measurements()

        fit = fit_joint_fingerprint(
            self.joint,
            self.sources,
            environment="push_cube",
            measurements=measurements,
        )
        asset = compose_joint_irg_asset(
            self.joint,
            self.sources,
            fit,
            environment="push_cube",
            checkpoint_step=100000,
            provenance={"measurement_sha256": "0" * 64},
        )

        self.assertEqual(
            fit.chart.intervention_names,
            tuple(row["path_name"] for row in self.joint["semantic_paths"]),
        )
        self.assertAlmostEqual(fit.chart.jacobian[0][0], 1.1)
        self.assertAlmostEqual(fit.chart.jacobian[0][1], 2.2)
        self.assertEqual(asset["dimensions"]["response_coordinate_count"], 28)
        self.assertEqual(asset["covariance_contract"]["joint_baseline_group_count"], 1)
        self.assertEqual(asset["transfer_state"], "ready")
        self.assertNotEqual(asset["response_covariance"][0][1], 0.0)
        validate_irg_asset(asset)

    def test_missing_condition_fails_closed(self) -> None:
        with self.assertRaisesRegex(JointFingerprintError, "MEASUREMENT_FRAME_INCOMPLETE"):
            fit_joint_fingerprint(
                self.joint,
                self.sources,
                environment="push_cube",
                measurements=self._measurements()[:-1],
            )

    def _measurements(self) -> list[dict[str, object]]:
        slopes = {
            "self_rollout_temporal_mix": 3.0,
            "motion_region_scale": 4.0,
            "action_embedding_temporal_mix": 5.0,
            "motion_region_event_alignment": 6.0,
            "action_temporal_alignment": 7.0,
        }
        rows: list[dict[str, object]] = []
        for condition in condition_schedule(self.joint, self.sources):
            seed = int(condition["seed"])
            seed_scale = {101: 1.0, 202: 1.1, 303: 1.2}[seed]
            baseline = {
                "psnr": 20.0 + seed / 1000.0,
                "ssim": 0.8 + seed / 100000.0,
                "mse": 0.1 + seed / 1000000.0,
                "masked_mse": 0.2 + seed / 1000000.0,
            }
            response = 0.0
            probe_id = condition["source_probe_id"]
            dose = float(condition["dose"])
            if probe_id == "action_conditioning_scale":
                response = seed_scale * (dose if dose > 0.0 else 2.0 * abs(dose))
            elif probe_id is not None:
                response = seed_scale * slopes[str(probe_id)] * dose
            metrics = dict(baseline)
            metrics["psnr"] += response
            metrics["ssim"] += 0.01 * response
            metrics["mse"] -= 0.001 * response
            metrics["masked_mse"] -= 0.002 * response
            rows.append(
                {
                    **condition,
                    "environment": "push_cube",
                    "metrics": metrics,
                }
            )
        return rows


if __name__ == "__main__":
    unittest.main()
