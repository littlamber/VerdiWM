from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.export.build_github_staging import GithubStagingError, audit_release_tree, build_github_staging
from scripts.export.acwm_public_experience_bundle import validate_public_experience_bundle
from scripts.export.validate_public_example import PublicExampleValidationError, validate_public_example
from scripts.export.verdiwm_minimal_loop_bundle import MinimalLoopBundleError, export_minimal_loop_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2"
EXPERIENCE_ROOT = REPO_ROOT / "examples" / "acwm_experience_atlas_v1"


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
    def test_audit_rejects_local_paths_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bad.txt").write_text("/" + "mnt" + "/private/run", encoding="utf-8")
            (root / "model.pt").write_bytes(b"weight")
            findings = audit_release_tree(root)
            self.assertIn("local_path:bad.txt", findings)
            self.assertIn("blocked_extension:model.pt", findings)

    def test_builder_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(GithubStagingError, "OUTPUT_EXISTS"):
                build_github_staging(source_root=REPO_ROOT, output_root=Path(temp))

    def test_builder_includes_cross_backbone_control_plane(self) -> None:
        if not (REPO_ROOT / "README_PUBLIC.md").is_file():
            self.skipTest("an exported release tree is not itself a release-builder source")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "release"
            audit = build_github_staging(source_root=REPO_ROOT, output_root=output)

            self.assertEqual(audit["state"], "ready")
            self.assertTrue((output / "wmloop" / "experiments" / "lobo.py").is_file())
            self.assertTrue(
                (output / "configs" / "experiments" / "three_backbone_lobo_pilot_v1.json").is_file()
            )
            self.assertTrue((output / "tests" / "test_cross_backbone_experiments.py").is_file())
            self.assertTrue(
                (output / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v1.json").is_file()
            )
            self.assertTrue((output / "configs" / "eval_ctrl_world_predictive_v1.sha256").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_predictive_adapter.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_predictive_instance.py").is_file())
            self.assertTrue((output / "tests" / "test_ctrl_world_fingerprint.py").is_file())
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


if __name__ == "__main__":
    unittest.main()
