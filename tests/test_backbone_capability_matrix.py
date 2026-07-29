from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.backbone_capability_matrix import (
    BackboneCapabilityMatrixError,
    main,
    run_backbone_capability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


class BackboneCapabilityMatrixTests(unittest.TestCase):
    def test_acwm_instance_exports_all_primitive_hooks_as_eligible(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest = run_backbone_capability_matrix(
                instance_config=ROOT / "configs/backbones/acwm_phys_g1_long_horizon_ladder_v1.json",
                output_root=Path(temporary) / "capability",
                repo_root=ROOT,
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(manifest["available_hooks"], ["H1", "H2", "H3", "H4", "H5"])
            self.assertEqual(manifest["eligible_primitive_count"], 17)
            self.assertEqual(manifest["blocked_primitive_count"], 0)

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(report["capability_summary"]["closed_loop_surface_ready"])
            self.assertTrue(all(row["status"] == "eligible_for_instance_canary" for row in report["primitive_matrix"]))
            self.assertFalse(report["side_effects"]["gpu_execution_started"])

    def test_missing_hook_adapter_blocks_non_acwm_instance_primitives(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "instance.json"
            config.write_text(json.dumps(_ctrl_world_missing_hook_adapter(), sort_keys=True), encoding="utf-8")

            manifest = run_backbone_capability_matrix(
                instance_config=config,
                output_root=root / "capability",
                repo_root=ROOT,
            )

            self.assertEqual(manifest["state"], "pilot_draft")
            self.assertEqual(manifest["available_hooks"], [])
            self.assertEqual(manifest["eligible_primitive_count"], 0)
            self.assertEqual(manifest["blocked_primitive_count"], 17)

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(any(blocker["code"] == "hook_adapter_missing_or_not_ready" for blocker in report["blockers"]))
            self.assertTrue(
                all("hook_adapter_missing_or_not_ready" in row["blockers"] for row in report["primitive_matrix"])
            )

    def test_existing_output_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "capability"
            output_root.mkdir()

            with self.assertRaisesRegex(BackboneCapabilityMatrixError, "BACKBONE_CAPABILITY_MATRIX_OUTPUT_EXISTS"):
                run_backbone_capability_matrix(
                    instance_config=ROOT / "configs/backbones/acwm_phys_g1_long_horizon_ladder_v1.json",
                    output_root=output_root,
                    repo_root=ROOT,
                )

    def test_cosmos3_matrix_only_exposes_explicit_backbone_bindings(self) -> None:
        smoke = ROOT / "results/reports/cosmos3-forward-dynamics-instance-smoke-r1/manifest.json"
        if not smoke.is_file():
            self.skipTest("live Cosmos3 CPU smoke has not been materialized")
        with TemporaryDirectory() as temporary:
            manifest = run_backbone_capability_matrix(
                instance_config=ROOT / "configs/backbones/cosmos3_forward_dynamics_predictive_pilot_v1.json",
                output_root=Path(temporary) / "capability",
                repo_root=ROOT,
            )
            self.assertEqual(manifest["state"], "pilot_draft")
            self.assertEqual(manifest["eligible_primitive_count"], 4)
            self.assertEqual(manifest["blocked_primitive_count"], 13)
            report = json.loads(Path(manifest["report_path"]).read_text())
            by_name = {row["primitive"]: row for row in report["primitive_matrix"]}
            self.assertEqual(by_name["action_dimension_balancing"]["status"], "eligible_for_instance_canary")
            self.assertIn("primitive_not_bound_for_backbone", by_name["next_forcing"]["blockers"])

    def test_cli_prints_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--instance-config",
                        str(ROOT / "configs/backbones/acwm_phys_g1_long_horizon_ladder_v1.json"),
                        "--output-root",
                        str(Path(temporary) / "capability"),
                        "--repo-root",
                        str(ROOT),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-backbone-capability-matrix-manifest")
            self.assertEqual(payload["eligible_primitive_count"], 17)


def _ctrl_world_missing_hook_adapter() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-backbone-instance",
        "instance_id": "ctrl_world_missing_hook_adapter",
        "backbone_family": "ctrl_world",
        "goal_id": "ctrl_world_g2_action_success_pilot_v1",
        "campaign_state": "pilot_draft",
        "claim_scope": "pilot only",
        "surfaces": [
            {
                "surface_id": "goal_spec",
                "role": "constitutional",
                "status": "draft",
                "artifact_ref": "configs/goal/ctrl_world_g2_action_success_pilot_v1.yaml",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": True,
                "notes": "Draft goal validates but is not frozen.",
            },
            {
                "surface_id": "primitive_registry",
                "role": "adapter",
                "status": "ready",
                "artifact_ref": "configs/registry_frozen.sha256",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": False,
                "notes": "Registry exists, but the non-ACWM hook adapter is intentionally absent.",
            },
            {
                "surface_id": "hook_adapter",
                "role": "adapter",
                "status": "missing",
                "artifact_ref": "wmloop/primitives/adapters/ctrl_world_hooks.py",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": False,
                "notes": "Missing WAM hook adapter.",
            },
        ],
        "invariants": ["Do not use ACWM probes as WAM verdict probes."],
        "next_actions": ["Implement hook adapter."],
    }


if __name__ == "__main__":
    unittest.main()
