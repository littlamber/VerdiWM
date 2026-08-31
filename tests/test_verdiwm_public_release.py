from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.export.build_github_staging import GithubStagingError, audit_release_tree, build_github_staging
from scripts.export.acwm_public_experience_bundle import validate_public_experience_bundle
from scripts.export.acwm_training_seed_horizon_public_bundle import (
    validate_training_seed_horizon_public_bundle,
)
from scripts.export.validate_public_example import PublicExampleValidationError, validate_public_example
from scripts.export.verdiwm_minimal_loop_bundle import MinimalLoopBundleError, export_minimal_loop_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2"
EXPERIENCE_ROOT = REPO_ROOT / "examples" / "acwm_experience_atlas_v1"
SELECTOR_ROOT = REPO_ROOT / "examples" / "acwm_selector_ablation_v1"
TRAINING_SEED_HORIZON_ROOT = REPO_ROOT / "examples" / "acwm_training_seed_horizon_stability_v1"


class PublicExampleValidatorTests(unittest.TestCase):
    def test_checked_in_example_is_integral(self) -> None:
        report = validate_public_example(EXAMPLE_ROOT)
        self.assertEqual(report["state"], "ready")
        self.assertEqual(report["artifact_count"], 9)
        self.assertFalse(report["paper_confirmed_effect"])

    def test_mutated_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "example"
            shutil.copytree(EXAMPLE_ROOT, copy)
            target = copy / "evidence" / "failure_report.json"
            original = target.read_text(encoding="utf-8")
            self.assertIn("mixed", original)
            target.write_text(original.replace("mixed", "other", 1), encoding="utf-8")
            with self.assertRaisesRegex(PublicExampleValidationError, "SHA256_MISMATCH"):
                validate_public_example(copy)

    def test_checked_in_experience_snapshot_is_integral(self) -> None:
        report = validate_public_experience_bundle(EXPERIENCE_ROOT)
        self.assertEqual(report["state"], "ready")
        self.assertEqual(report["screen_trial_count"], 284)
        self.assertEqual(report["showcase_case_count"], 4)

    def test_checked_in_training_seed_horizon_bundle_is_integral(self) -> None:
        report = validate_training_seed_horizon_public_bundle(TRAINING_SEED_HORIZON_ROOT)
        self.assertEqual(report["state"], "ready")
        self.assertEqual(report["training_seed_count"], 3)
        self.assertEqual(report["video_count"], 9)
        self.assertEqual(
            report["stability_verdict"],
            "training_seed_sensitive_long_horizon_effect",
        )


class MinimalLoopExporterTests(unittest.TestCase):
    def _sources(self) -> dict[str, Path]:
        evidence = EXAMPLE_ROOT / "evidence"
        return {
            "failure_report": evidence / "failure_report.json",
            "intervention_receipt": evidence / "intervention_receipt.json",
            "screen_manifest": evidence / "screen_manifest.json",
            "official_gate_manifest": evidence / "official_gate_manifest.json",
            "confirmation_manifest": evidence / "confirmation_manifest.json",
            "checkpoint_ladder": evidence / "checkpoint_ladder.json",
            "effect_profile": evidence / "effect_profile.json",
            "experience_map": evidence / "experience_map.json",
        }

    def test_reexports_a_coherent_public_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            report = export_minimal_loop_bundle(
                **self._sources(),
                output_root=output,
                showcase_video=EXAMPLE_ROOT / "media" / "showcase_video.mp4",
            )
            self.assertTrue(report["operational_minimal_loop_pass"])
            self.assertEqual(validate_public_example(output)["artifact_count"], 9)

    def test_rejects_a_negative_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mutated = Path(temp) / "screen.json"
            payload = json.loads((EXAMPLE_ROOT / "evidence" / "screen_manifest.json").read_text(encoding="utf-8"))
            payload["delta_m_ver"]["ladder_auc_psnr_envmax"] = -1.0
            mutated.write_text(json.dumps(payload), encoding="utf-8")
            sources = self._sources()
            sources["screen_manifest"] = mutated
            with self.assertRaisesRegex(MinimalLoopBundleError, "screen_ready"):
                export_minimal_loop_bundle(**sources, output_root=Path(temp) / "bundle")


class GithubStagingTests(unittest.TestCase):
    def test_selector_showcase_preserves_official_gate_and_media_integrity(self) -> None:
        showcase = SELECTOR_ROOT / "showcase" / "pour_water_inv_dyn_reward_finetune"
        case = json.loads((showcase / "case.json").read_text(encoding="utf-8"))

        self.assertEqual(case["state"], "ready")
        self.assertEqual(case["environment"], "pour_water")
        self.assertEqual(case["primitive"], "inv_dyn_reward_finetune")
        self.assertTrue(case["official_quality_gate"]["pass"])
        self.assertTrue(all(case["official_quality_gate"]["checks"].values()))
        self.assertEqual(case["visual_evidence"]["selection_rule"], "descending baseline-only RGB video MSE against GT")
        self.assertTrue((showcase / "paired_gt_baseline_ours.mp4").is_file())
        self.assertTrue((showcase / "poster.png").is_file())

        for row in (showcase / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest, relative = row.split("  ", 1)
            self.assertEqual(hashlib.sha256((showcase / relative).read_bytes()).hexdigest(), digest)

    def test_audit_rejects_local_paths_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bad.txt").write_text("/" + "mnt" + "/private/run", encoding="utf-8")
            (root / "shared.txt").write_text("/" + "share" + "/project/run", encoding="utf-8")
            (root / "model.pt").write_bytes(b"weight")
            findings = audit_release_tree(root)
            self.assertIn("local_path:bad.txt", findings)
            self.assertIn("local_path:shared.txt", findings)
            self.assertIn("blocked_extension:model.pt", findings)

    def test_builder_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(GithubStagingError, "OUTPUT_EXISTS"):
                build_github_staging(source_root=REPO_ROOT, output_root=Path(temp))

    @unittest.skipIf(
        (REPO_ROOT / ".verdiwm-public-release").is_file(),
        "the staging-builder integration test runs only in the development repository",
    )
    def test_builder_includes_cross_backbone_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "release"
            audit = build_github_staging(source_root=REPO_ROOT, output_root=output)
            self.assertFalse(
                (output / "configs/experiments/ctrl_world_autonomous_transfer_loop_v1.json").exists()
            )

            self.assertEqual(audit["state"], "ready")
            self.assertTrue((output / ".verdiwm-public-release").is_file())
            self.assertTrue((output / "README_zh.md").is_file())
            self.assertTrue((output / "wmloop" / "experiments" / "lobo.py").is_file())
            self.assertTrue(
                (output / "configs" / "experiments" / "three_backbone_lobo_pilot_v1.json").is_file()
            )
            self.assertTrue((output / "tests" / "test_cross_backbone_experiments.py").is_file())
            self.assertTrue((output / "tests" / "test_evidence_capsule.py").is_file())
            self.assertTrue((output / "tests" / "test_mechanism_discovery.py").is_file())
            self.assertTrue((output / "tests" / "test_transferable_experience.py").is_file())
            self.assertTrue((output / "wmloop" / "retrieve" / "evidence_capsule.py").is_file())
            self.assertTrue((output / "wmloop" / "retrieve" / "mechanism_discovery.py").is_file())
            self.assertTrue(
                (output / "configs" / "retrieval" / "mechanism_tag_ontology_v1.json").is_file()
            )
            self.assertTrue((output / "docs" / "TRANSFERABLE_EXPERIENCE.md").is_file())
            self.assertTrue((output / "docs" / "RELEASE_CHECKLIST.md").is_file())
            self.assertTrue(
                (output / "docs" / "PORTRAIT_FIRST_AUTONOMOUS_RESEARCH_EXECUTION_PLAN.md").is_file()
            )
            self.assertTrue(
                (output / "experiments" / "ctrl_world_autonomous_transfer_v1" / "controller.py").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "experiments"
                    / "ctrl_world_autonomous_transfer_v1"
                    / "local_method_intake.py"
                ).is_file()
            )
            self.assertTrue(
                (output / "experiments" / "ctrl_world_hybrid_memory_transfer_v1" / "run.py").is_file()
            )
            self.assertTrue(
                (output / "experiments" / "ctrl_world_research_loop_v2" / "research_intake.py").is_file()
            )
            self.assertTrue(
                (output / "experiments" / "ctrl_world_acwm_guidance_v1" / "research_intake.py").is_file()
            )
            self.assertTrue(
                (output / "experiments" / "ctrl_world_autonomous_transfer_v1" / "engineering_manifest.json").is_file()
            )
            self.assertTrue((output / "configs" / "plugins" / "automatic_module_abis_v1.json").is_file())
            self.assertTrue(
                (output / "examples" / "portrait_first_minimal_loop_v1" / "run.py").is_file()
            )
            self.assertEqual(
                audit["portrait_first_validation"]["readiness_state"],
                "ready_for_gap_planning",
            )
            self.assertEqual(
                audit["portrait_first_validation"]["gap_plan_state"],
                "ready_for_portfolio",
            )
            self.assertTrue((output / "tests" / "test_acwm_multiseed_eval_summary.py").is_file())
            self.assertTrue((output / "tests" / "test_progressive_fidelity.py").is_file())
            self.assertTrue((output / "tests" / "test_stage_progressive_fidelity_sources.py").is_file())
            self.assertTrue(
                (output / "examples" / "acwm_eval_seed_replication_v1" / "MANIFEST.sha256").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "acwm_training_seed_horizon_stability_v1"
                    / "MANIFEST.sha256"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "acwm_training_seed_replication_cloth_self_forcing_v1"
                    / "summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "acwm_progressive_fidelity_efficiency_v1"
                    / "MANIFEST.sha256"
                ).is_file()
            )
            self.assertTrue(
                (output / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v1.json").is_file()
            )
            self.assertTrue((output / "configs" / "eval_ctrl_world_predictive_v1.sha256").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_predictive_adapter.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_predictive_instance.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_fingerprint.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_receipt_merge.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_fingerprint_settlement.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_fingerprint_public_bundle.py").is_file())
            self.assertTrue((output / "tests" / "test_cosmos3_probe_evolution.py").is_file())
            self.assertTrue((output / "tests" / "test_cosmos3_directional_probe.py").is_file())
            self.assertTrue(
                (output / "tests" / "test_cosmos3_directional_settlement.py").is_file()
            )
            self.assertTrue(
                (output / "tests" / "test_cosmos3_split_v2_protocol.py").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "tests"
                    / "test_cosmos3_action_dimension_anisotropy_protocol.py"
                ).is_file()
            )
            self.assertTrue(
                (output / "tests" / "test_cosmos3_shard_recovery.py").is_file()
            )
            self.assertTrue((output / "examples" / "ctrl_world_target_local_irg_v1" / "bundle.json").is_file())
            self.assertTrue(
                (output / "examples" / "cosmos3_target_local_irg_narrow_v1" / "bundle.json").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_target_local_irg_temporal_mix_v1"
                    / "bundle.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_action_dimension_anisotropy_counterexample_v3"
                    / "bundle.json"
                ).is_file()
            )
            translation_narrow = audit[
                "cosmos3_translation_narrow_settlement_validation"
            ]
            self.assertEqual(
                translation_narrow["settlement_state"], "settled_abstained"
            )
            self.assertFalse(
                translation_narrow["cross_backbone_transfer_eligible"]
            )
            self.assertEqual(
                translation_narrow["accept_validation"]["locality_admission_state"],
                "failed",
            )
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_translation_narrow_split_reversal_v2"
                    / "bundle.json"
                ).is_file()
            )
            self.assertEqual(
                set(audit["cosmos3_fingerprint_validations"]),
                {
                    "cosmos3_target_local_irg_wide_v1",
                    "cosmos3_target_local_irg_narrow_v1",
                    "cosmos3_target_local_irg_temporal_mix_v1",
                    "cosmos3_action_dimension_anisotropy_counterexample_v3",
                },
            )
            directional = audit["cosmos3_directional_settlement_validation"]
            self.assertEqual(directional["settlement_state"], "settled_abstained")
            self.assertFalse(directional["cross_backbone_transfer_eligible"])
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_directional_probe_split_reversal_v1"
                    / "bundle.json"
                ).is_file()
            )
            translation = audit["cosmos3_translation_settlement_validation"]
            self.assertEqual(translation["settlement_state"], "settled_abstained")
            self.assertFalse(translation["cross_backbone_transfer_eligible"])
            self.assertEqual(translation["accept_validation"]["locality_admission_state"], "failed")
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_translation_locality_counterexample_v1"
                    / "bundle.json"
                ).is_file()
            )
            interaction = audit["cosmos3_interaction_settlement_validation"]
            self.assertEqual(interaction["settlement_state"], "settled_abstained")
            self.assertFalse(interaction["cross_backbone_transfer_eligible"])
            self.assertEqual(interaction["accept_validation"]["locality_admission_state"], "failed")
            self.assertTrue(
                (
                    output
                    / "examples"
                    / "cosmos3_action_dimension_interaction_split_reversal_v4"
                    / "bundle.json"
                ).is_file()
            )
            self.assertTrue(
                (output / "configs" / "constitution" / "ctrl_world_predictive_quality_pilot_v1.freeze.json").is_file()
            )
            self.assertTrue(
                (output / "configs" / "backbones" / "ctrl_world_g2_action_success_pilot_v1.json").is_file()
            )
            self.assertTrue((output / "configs" / "registry_ctrl_world_g2.sha256").is_file())
            self.assertTrue(
                (output / "configs" / "constitution" / "ctrl_world_g2_action_success_pilot_v1.freeze.json").is_file()
            )

            predictive = json.loads(
                (output / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(predictive["backbone_family"], "ctrl_world")
            self.assertEqual(predictive["campaign_state"], "pilot_draft")
            local_mount_prefix = "/" + "mnt" + "/"
            self.assertFalse(
                any(local_mount_prefix in surface["artifact_ref"] for surface in predictive["surfaces"])
            )
            self.assertTrue(any("Downstream task success" in value for value in predictive["invariants"]))

            lobo = json.loads(
                (output / "configs" / "experiments" / "three_backbone_lobo_pilot_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            ctrl_world = next(item for item in lobo["backbones"] if item["backbone_id"] == "ctrl_world")
            self.assertEqual(
                ctrl_world["instance_ref"],
                "configs/backbones/ctrl_world_predictive_quality_pilot_v1.json",
            )
            self.assertNotIn("action_success", ctrl_world["goal_contract"])

    def test_release_contract_selects_a_license_and_build_backend(self) -> None:
        license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
        project_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        readme_path = REPO_ROOT / "README_PUBLIC.md"
        if not readme_path.is_file():
            readme_path = REPO_ROOT / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")

        self.assertIn("Apache License", license_text)
        self.assertIn('license = "Apache-2.0"', project_text)
        self.assertIn('build-backend = "hatchling.build"', project_text)
        self.assertIn("[tool.hatch.build.targets.wheel.force-include]", project_text)
        self.assertIn("Apache License 2.0", readme_text)

    def test_python_310_release_installs_runtime_dependencies(self) -> None:
        public_project_text = (
            REPO_ROOT / "scripts" / "export" / "public_pyproject.toml"
        ).read_text(encoding="utf-8")
        preflight_text = (
            REPO_ROOT / "scripts" / "ci" / "release_preflight.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"tomli>=2,<3; python_version < \'3.11\'"',
            public_project_text,
        )
        self.assertNotIn("uv pip install --python \"$venv_dir/bin/python\" --no-deps", preflight_text)
        self.assertNotIn(
            "uv pip install --python \"$staged_venv_dir/bin/python\" --no-deps",
            preflight_text,
        )
        self.assertNotIn(
            "uv pip install --python \"$venv_dir/bin/python\" --offline",
            preflight_text,
        )
        self.assertNotIn(
            "uv pip install --python \"$staged_venv_dir/bin/python\" --offline",
            preflight_text,
        )
        self.assertIn('staging_validation_env="$work_dir/public-tree-validation-venv"', preflight_text)
        self.assertIn('test ! -e "$staging_dir/.venv"', preflight_text)


if __name__ == "__main__":
    unittest.main()
