from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.archive.store import ArchiveStore
from wmloop.propose.primitive_materialization_prompt import (
    PrimitiveMaterializationPromptError,
    main,
    run_primitive_materialization_prompt,
    run_primitive_materialization_prompt_batch,
)


ROOT = Path(__file__).resolve().parents[1]


class PrimitiveMaterializationPromptTests(unittest.TestCase):
    def test_prompt_packet_contains_alignment_and_forbidden_path_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_order = root / "work-order.json"
            work_order.write_text(json.dumps(_work_order("drift_token_trim"), sort_keys=True), encoding="utf-8")
            intent = _write_intent_manifest(root / "intent")

            manifest = run_primitive_materialization_prompt(
                repo_root=ROOT,
                work_order=work_order,
                intent_compilation_manifest=intent,
                output_root=root / "reports" / "prompt",
                archive_db=root / "results" / "archive.db",
                cas_root=root / "results",
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(manifest["primitive"], "drift_token_trim")
            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            prompt = report["prompt_text"]
            self.assertIn("configuration intent and engineering behavior aligned", prompt)
            self.assertIn("No silent implementation compromise", prompt)
            self.assertIn("fail closed with a blocker report", prompt)
            self.assertIn("method intent to exact code touchpoints", prompt)
            self.assertIn("Do not modify goal specs", prompt)
            self.assertIn("WM-Dx routing boundary", prompt)
            self.assertIn("Diagnostic probe candidates", prompt)
            self.assertIn("Google-style software engineering discipline", prompt)
            self.assertIn("Required CI/harness receipts", prompt)
            contract = report["alignment_contract"]
            self.assertIn("faithfully realize", contract["faithful_materialization_rule"])
            self.assertIn("silently substitute", contract["faithful_materialization_rule"])
            self.assertEqual(
                contract["engineering_practice_policy"]["policy_id"],
                "verdiwm_agent_engineering_policy_v1",
            )
            self.assertIn("fail closed with a blocker report", contract["compromise_blocker_policy"])
            self.assertIn(
                "declared compromises or blockers, with no silent substitutions",
                contract["required_intent_to_code_receipts"],
            )
            self.assertIn("eval.py", report["alignment_contract"]["forbidden_paths"])
            self.assertFalse(report["side_effects"]["source_code_mutated"])
            self.assertGreaterEqual(ArchiveStore(root / "results" / "archive.db").archive_statistics()["artifacts"], 4)

    def test_existing_output_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_order = root / "work-order.json"
            work_order.write_text(json.dumps(_work_order("cfg_guidance_schedule"), sort_keys=True), encoding="utf-8")
            output_root = root / "reports" / "prompt"
            output_root.mkdir(parents=True)
            marker = output_root / "marker"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(PrimitiveMaterializationPromptError, "OUTPUT_EXISTS"):
                run_primitive_materialization_prompt(
                    repo_root=ROOT,
                    work_order=work_order,
                    output_root=output_root,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_batch_prompt_packet_from_materialization_gate_work_orders(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "work-orders" / "cfg_guidance_schedule.json"
            second = root / "work-orders" / "dino_rep_injection.json"
            _write_work_order(first, "cfg_guidance_schedule")
            _write_work_order(second, "dino_rep_injection")
            gate = _write_gate_manifest(root / "gate", {"cfg_guidance_schedule": first, "dino_rep_injection": second})
            intent = _write_intent_manifest(root / "intent")

            manifest = run_primitive_materialization_prompt_batch(
                repo_root=ROOT,
                materialization_gate_manifest=gate,
                intent_compilation_manifest=intent,
                output_root=root / "reports" / "prompt-batch",
                archive_db=root / "results" / "archive.db",
                cas_root=root / "results",
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(manifest["prompt_count"], 2)
            self.assertEqual(manifest["primitives"], ["cfg_guidance_schedule", "dino_rep_injection"])
            self.assertFalse(manifest["side_effects"]["source_code_mutated"])
            self.assertFalse(manifest["side_effects"]["gpu_execution_started"])
            self.assertFalse(manifest["side_effects"]["primitive_promoted"])
            records = {record["primitive"]: record for record in manifest["records"]}
            self.assertTrue(Path(records["cfg_guidance_schedule"]["prompt_manifest_path"]).is_file())
            self.assertTrue(Path(records["dino_rep_injection"]["prompt_path"]).is_file())
            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["source_materialization_gate"]["work_order_count"], 2)
            self.assertGreaterEqual(ArchiveStore(root / "results" / "archive.db").archive_statistics()["artifacts"], 11)

    def test_batch_filter_unknown_primitive_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "work-orders" / "cfg_guidance_schedule.json"
            _write_work_order(first, "cfg_guidance_schedule")
            gate = _write_gate_manifest(root / "gate", {"cfg_guidance_schedule": first})

            with self.assertRaisesRegex(PrimitiveMaterializationPromptError, "UNKNOWN_PRIMITIVE"):
                run_primitive_materialization_prompt_batch(
                    repo_root=ROOT,
                    materialization_gate_manifest=gate,
                    primitives=("missing_primitive",),
                    output_root=root / "reports" / "prompt-batch",
                )

            self.assertFalse((root / "reports" / "prompt-batch").exists())

    def test_cli_prints_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_order = root / "work-order.json"
            work_order.write_text(json.dumps(_work_order("cfg_guidance_schedule"), sort_keys=True), encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(
                    [
                        "run",
                        "--repo-root",
                        str(ROOT),
                        "--work-order",
                        str(work_order),
                        "--output-root",
                        str(root / "reports" / "prompt"),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-primitive-materialization-prompt-manifest")
            self.assertEqual(payload["primitive"], "cfg_guidance_schedule")

    def test_cli_prints_batch_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_order = root / "work-orders" / "cfg_guidance_schedule.json"
            _write_work_order(work_order, "cfg_guidance_schedule")
            gate = _write_gate_manifest(root / "gate", {"cfg_guidance_schedule": work_order})
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(
                    [
                        "batch",
                        "--repo-root",
                        str(ROOT),
                        "--materialization-gate-manifest",
                        str(gate),
                        "--primitive",
                        "cfg_guidance_schedule",
                        "--output-root",
                        str(root / "reports" / "prompt-batch"),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-primitive-materialization-prompt-batch-manifest")
            self.assertEqual(payload["prompt_count"], 1)


def _work_order(primitive: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-materialization-work-order",
        "primitive": primitive,
        "current_admission_state": "sidecar_only",
        "target_admission_state": "closed_loop_runtime_ready",
        "layer": "L4",
        "hooks": ["H4"],
        "targets_failures": ["appearance_drift", "train_infer_mismatch"],
        "params_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["keep_tokens"],
            "properties": {"keep_tokens": {"type": "integer", "minimum": 1}},
        },
        "allowed_mutation_paths": [
            f"wmloop/primitives/definitions/{primitive}/apply.py",
            f"wmloop/primitives/definitions/{primitive}/templates/",
            "tests/test_primitive_render.py",
            "tests/test_primitive_apply_audit.py",
            "tests/test_primitive_runtime_smoke.py",
            "vendor/ACWM-Phys/acwm/wmloop_hooks/",
            "vendor/ACWM-Phys/acwm/trainer/train_dynamics.py",
        ],
        "forbidden_paths": ["eval.py", "scripts/eval_all.sh", "results/", "configs/goal/", "runs/m0/protocol/"],
        "required_gates": [
            "schema_valid",
            "clean_diff_no_frozen_evaluator",
            "primitive_apply_audit_passed",
            "runtime_hook_unit_passed",
            "gpu_training_smoke_passed",
        ],
        "source_revision": "a" * 40,
        "registry_digest": "b" * 64,
        "promotion_rule": "gate only",
        "apply_module": f"wmloop.primitives.definitions.{primitive}.apply",
        "suggested_checks": [],
    }


def _write_intent_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-user-intent-compilation-manifest",
        "state": "staged_blocked",
        "intent_binding_ready": False,
        "goal_id": "g1_long_horizon_ladder_v1",
        "goal_config": "/tmp/goal.yaml",
        "compiled_goal_family": "long_horizon_consistency",
        "compiled_backbone_family": "acwm_phys",
        "compiled_environment_scope": "seven_env_without_cloth_move",
        "limited_execution_allowed": True,
        "formal_m4_launch_permission_granted": False,
        "blockers": [{"code": "environment_scope_mismatch"}],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_work_order(path: Path, primitive: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_work_order(primitive), sort_keys=True), encoding="utf-8")


def _write_gate_manifest(root: Path, work_orders: dict[str, Path]) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-materialization-gate-manifest",
        "state": "blocked",
        "primitive_count": 13,
        "closed_loop_ready_count": 3,
        "sidecar_only_count": len(work_orders),
        "closed_loop_ready_primitives": ["drift_token_trim", "history_noise_schedule", "latent_motion_prior"],
        "blockers": [{"code": "sidecar_only_primitives_present"}],
        "report_path": str(root / "primitive-materialization-gate.json"),
        "markdown_path": str(root / "primitive-materialization-gate.md"),
        "work_order_paths": {primitive: str(path) for primitive, path in work_orders.items()},
        "cas_refs": {},
        "limitations": [],
    }
    (root / "primitive-materialization-gate.json").write_text(json.dumps({"state": "blocked"}), encoding="utf-8")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return path
