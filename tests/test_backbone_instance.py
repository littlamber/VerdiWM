from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.backbone_instance import BackboneInstantiationError, main, run_backbone_instantiation_audit


ROOT = Path(__file__).resolve().parents[1]


class BackboneInstanceTests(unittest.TestCase):
    def test_minimal_ready_instance_validates_contract_surfaces(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "instance.json"
            config.write_text(json.dumps(_ready_instance(), sort_keys=True), encoding="utf-8")

            manifest = run_backbone_instantiation_audit(
                instance_config=config,
                output_root=root / "reports" / "instance",
                repo_root=ROOT,
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertTrue(manifest["closed_loop_instance_ready"])
            self.assertTrue(manifest["formal_verdict_instance_ready"])
            self.assertTrue(manifest["instance_formal_launch_allowed"])

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["blockers"], [])
            self.assertEqual({check["contract"] for check in report["contract_checks"]}, {"goal_spec", "probe_registry"})

    def test_pilot_draft_surfaces_are_explicit_blockers_without_launch_permission(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "instance.json"
            config.write_text(json.dumps(_pilot_instance(), sort_keys=True), encoding="utf-8")

            manifest = run_backbone_instantiation_audit(
                instance_config=config,
                output_root=root / "reports" / "instance",
                repo_root=ROOT,
            )

            self.assertEqual(manifest["state"], "pilot_draft")
            self.assertFalse(manifest["closed_loop_instance_ready"])
            self.assertFalse(manifest["formal_verdict_instance_ready"])
            self.assertFalse(manifest["instance_formal_launch_allowed"])

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            blocker_codes = {blocker["code"] for blocker in report["blockers"]}
            self.assertIn("surface_draft", blocker_codes)
            self.assertIn("surface_missing", blocker_codes)
            self.assertFalse(report["side_effects"]["gpu_execution_started"])

    def test_declared_ready_missing_path_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _ready_instance()
            payload["surfaces"].append(
                {
                    "surface_id": "missing_ready_adapter",
                    "role": "adapter",
                    "status": "ready",
                    "artifact_ref": "wmloop/does_not_exist.py",
                    "required_for_closed_loop": True,
                    "required_for_formal_verdict": False,
                    "notes": "Ready surfaces must physically exist.",
                }
            )
            config = root / "instance.json"
            config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            manifest = run_backbone_instantiation_audit(
                instance_config=config,
                output_root=root / "reports" / "instance",
                repo_root=ROOT,
            )

            self.assertEqual(manifest["state"], "blocked")
            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(any(blocker["code"] == "declared_ready_path_missing" for blocker in report["blockers"]))

    def test_existing_output_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "instance.json"
            config.write_text(json.dumps(_ready_instance(), sort_keys=True), encoding="utf-8")
            output_root = root / "reports" / "instance"
            output_root.mkdir(parents=True)

            with self.assertRaisesRegex(BackboneInstantiationError, "BACKBONE_INSTANCE_OUTPUT_EXISTS"):
                run_backbone_instantiation_audit(
                    instance_config=config,
                    output_root=output_root,
                    repo_root=ROOT,
                )

    def test_cli_prints_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "instance.json"
            config.write_text(json.dumps(_ready_instance(), sort_keys=True), encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(
                    [
                        "--instance-config",
                        str(config),
                        "--output-root",
                        str(root / "reports" / "instance"),
                        "--repo-root",
                        str(ROOT),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-backbone-instantiation-manifest")
            self.assertEqual(payload["state"], "ready")


def _ready_instance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-backbone-instance",
        "instance_id": "minimal_acwm_ready",
        "backbone_family": "acwm_phys",
        "goal_id": "g1_long_horizon_ladder_v1",
        "campaign_state": "active_ready",
        "claim_scope": "minimal contract-valid instance for tests",
        "surfaces": [
            {
                "surface_id": "goal_spec",
                "role": "constitutional",
                "status": "ready",
                "artifact_ref": "configs/goal/g1_long_horizon_ladder_v1.yaml",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": True,
                "notes": "Goal contract surface.",
            },
            {
                "surface_id": "probe_registry",
                "role": "constitutional",
                "status": "ready",
                "artifact_ref": "configs/probes/acwm_v1.json",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": True,
                "notes": "Probe registry contract surface.",
            },
        ],
        "invariants": ["Do not mutate verdict surfaces during a run."],
        "next_actions": ["Run the next audit."],
    }


def _pilot_instance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-backbone-instance",
        "instance_id": "ctrl_world_pilot",
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
                "surface_id": "probe_registry",
                "role": "constitutional",
                "status": "draft",
                "artifact_ref": "configs/probes/ctrl_world_wam_v1.json",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": True,
                "notes": "Draft probe registry validates but is not frozen.",
            },
            {
                "surface_id": "evaluator_adapter",
                "role": "constitutional",
                "status": "missing",
                "artifact_ref": "wmloop/evaluate/adapters/ctrl_world.py",
                "required_for_closed_loop": True,
                "required_for_formal_verdict": True,
                "notes": "Missing verifier adapter.",
            },
        ],
        "invariants": ["Do not use ACWM probes as WAM verdict probes."],
        "next_actions": ["Implement evaluator adapter."],
    }


if __name__ == "__main__":
    unittest.main()
