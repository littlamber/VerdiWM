#!/usr/bin/env python3
"""Build an allowlisted, audited VerdiWM tree suitable for GitHub staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export.validate_public_example import validate_public_example
from scripts.export.acwm_public_experience_bundle import validate_public_experience_bundle
from scripts.export.acwm_training_seed_horizon_public_bundle import (
    validate_training_seed_horizon_public_bundle,
)
from scripts.export.validate_portrait_first_public_example import (
    validate_portrait_first_public_example,
)


class GithubStagingError(RuntimeError):
    """The release tree could not be built without violating publication rules."""


ROOT_FILES = (
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "README_zh.md",
    "SECURITY.md",
    "uv.lock",
)
PUBLIC_TREES = ("wmloop", "scripts", ".github", "ops")
PUBLIC_DOC_FILES = (
    "ARCHITECTURE.md",
    "ARTIFACT_CONVENTION.md",
    "AUTONOMOUS_TRANSFER_SYSTEM_PLAN.md",
    "AUTO_EXPERIMENTS.md",
    "BACKBONE_INSTANTIATION.md",
    "COSMOS3_FORWARD_DYNAMICS.md",
    "CONSTITUTION_EVOLUTION.md",
    "CPBE.md",
    "EVIDENCE_CAPSULE.md",
    "EXPERIMENT_ENGINEERING.md",
    "INTERMEDIATE_REPRESENTATIONS.md",
    "LLM_SKILL_CONTROL_PLANE.md",
    "MECHANISM_DISCOVERY.md",
    "METHOD_CANDIDATE_COMPILATION.md",
    "METHOD_TO_CODE.md",
    "ONBOARDING.md",
    "PORTRAIT_FIRST_AUTONOMOUS_RESEARCH_EXECUTION_PLAN.md",
    "RELEASE_CHECKLIST.md",
    "REPRODUCIBILITY.md",
    "TRAINING_RECIPE_RESEARCH.md",
    "TRANSFERABLE_EXPERIENCE.md",
    "acwm_probe_evolution_r1.md",
)
PUBLIC_TEST_FILES = (
    "test_verdiwm_geometry.py",
    "test_acwm_unified_irg_assets.py",
    "test_acwm_joint_fingerprint.py",
    "test_acwm_joint_fingerprint_campaign_runner.py",
    "test_verdiwm_public_release.py",
    "test_acwm_public_experience_bundle.py",
    "test_acwm_multiseed_eval_summary.py",
    "test_acwm_formal_visualization.py",
    "test_acwm_training_seed_replication.py",
    "test_acwm_horizon_effect_profile.py",
    "test_acwm_training_seed_horizon.py",
    "test_agent_engineering_policy.py",
    "test_backbone_capability_matrix.py",
    "test_backbone_instance.py",
    "test_cross_backbone_experiments.py",
    "test_verdiwm_paper_experiment_matrix.py",
    "test_acwm_fingerprint_calibration_audit.py",
    "test_acwm_autoloop_worker.py",
    "test_acwm_effect_label_completion_plan.py",
    "test_acwm_effect_label_gate_queue.py",
    "test_acwm_effect_labels.py",
    "test_acwm_source_effect_audit.py",
    "test_acwm_source_effect_repair.py",
    "test_acwm_selector_ablation.py",
    "test_acwm_selector_cpu_replay.py",
    "test_acwm_selector_projection_compose.py",
    "test_acwm_selector_probe_admission.py",
    "test_acwm_paper_primitive_matrix.py",
    "test_acwm_method_evidence_maps.py",
    "test_acwm_selector_public_bundle.py",
    "test_acwm_self_rollout_history_probe.py",
    "test_acwm_fingerprint_campaign_runner.py",
    "test_progressive_fidelity.py",
    "test_stage_progressive_fidelity_sources.py",
    "test_probe_information.py",
    "test_acwm_probe_smoke_redundancy.py",
    "test_random_probe_expansion.py",
    "test_collision_labels.py",
    "test_acwm_probe_information_public_example.py",
    "test_ctrl_world_instance_adapters.py",
    "test_ctrl_world_predictive_adapter.py",
    "test_ctrl_world_predictive_instance.py",
    "test_ctrl_world_fingerprint.py",
    "test_ctrl_world_receipt_merge.py",
    "test_ctrl_world_fingerprint_settlement.py",
    "test_ctrl_world_fingerprint_public_bundle.py",
    "test_ctrl_world_probe_evolution.py",
    "test_ctrl_world_predictive_campaign_runner.py",
    "test_cosmos3_forward_dynamics_public_bundle.py",
    "test_cosmos3_gpu_runtime_receipt.py",
    "test_cosmos3_paired_gt.py",
    "test_cosmos3_fingerprint.py",
    "test_cosmos3_fingerprint_campaign_runner.py",
    "test_cosmos3_fingerprint_public_bundle.py",
    "test_cosmos3_probe_evolution.py",
    "test_cosmos3_directional_probe.py",
    "test_cosmos3_directional_settlement.py",
    "test_cosmos3_directional_settlement_public_bundle.py",
    "test_cosmos3_split_v2_protocol.py",
    "test_cosmos3_action_dimension_anisotropy_protocol.py",
    "test_cosmos3_action_dimension_interaction_protocol.py",
    "test_cosmos3_shard_recovery.py",
    "test_primitive_materialization_prompt.py",
    "test_diagnostic_probe_materialization_prompt.py",
    "test_diagnostic_probe_routing_admission.py",
    "test_cpbe.py",
    "test_cpbe_public_example.py",
    "test_acwm_push_cube_contact_event_diagnostic_v1.py",
    "test_acwm_push_cube_action_binding_diagnostic_v1.py",
    "test_acwm_push_cube_rigid_pose_slip_diagnostic_v1.py",
    "test_acwm_stack_cube_support_relation_diagnostic_v1.py",
    "test_acwm_stack_cube_contact_instability_diagnostic_v1.py",
    "test_acwm_stack_cube_object_identity_diagnostic_v1.py",
    "test_acwm_pour_water_container_boundary_leak_diagnostic_v1.py",
    "test_acwm_pour_water_free_surface_diagnostic_v1.py",
    "test_acwm_pour_water_fluid_volume_transport_diagnostic_v1.py",
    "test_acwm_push_sand_granular_frontier_diagnostic_v1.py",
    "test_acwm_push_sand_mass_redistribution_diagnostic_v1.py",
    "test_acwm_push_sand_particle_boundary_diagnostic_v1.py",
    "test_acwm_reacher_target_conditioning_diagnostic_v1.py",
    "test_acwm_reacher_inverse_dynamics_confidence_diagnostic_v1.py",
    "test_acwm_reacher_endpoint_control_diagnostic_v1.py",
    "test_acwm_push_rope_topology_change_diagnostic_v1.py",
    "test_acwm_push_rope_endpoint_path_diagnostic_v1.py",
    "test_acwm_push_rope_deformable_contact_diagnostic_v1.py",
    "test_acwm_cloth_move_surface_fold_diagnostic_v1.py",
    "test_acwm_cloth_move_cloth_identity_drift_diagnostic_v1.py",
    "test_acwm_cloth_move_deformable_memory_diagnostic_v1.py",
    "test_evidence_capsule.py",
    "test_mechanism_discovery.py",
    "test_transferable_experience.py",
    "test_system_utility.py",
    "test_artifact_lint.py",
    "test_portrait_first_public_example.py",
)
COSMOS3_PUBLIC_EXAMPLES = (
    "cosmos3_target_local_irg_wide_v1",
    "cosmos3_target_local_irg_narrow_v1",
    "cosmos3_target_local_irg_temporal_mix_v1",
    "cosmos3_action_dimension_anisotropy_counterexample_v3",
)
COSMOS3_DIRECTIONAL_PUBLIC_EXAMPLE = "cosmos3_directional_probe_split_reversal_v1"
COSMOS3_TRANSLATION_PUBLIC_EXAMPLE = "cosmos3_translation_locality_counterexample_v1"
COSMOS3_TRANSLATION_NARROW_PUBLIC_EXAMPLE = "cosmos3_translation_narrow_split_reversal_v2"
COSMOS3_INTERACTION_PUBLIC_EXAMPLE = "cosmos3_action_dimension_interaction_split_reversal_v4"
CONFIG_TREES = (
    "adapters",
    "constitution",
    "diagnose",
    "envs",
    "experiments",
    "goal",
    "loop",
    "methods",
    "plugins",
    "primitives",
    "probes",
    "references",
    "retrieval",
    "schemas",
    "smoke",
)

PUBLIC_AUTONOMOUS_TRANSFER_FILES = (
    "controller.py",
    "history_selection_abi_test.py",
    "local_method_intake.py",
    "replanning.py",
    "scale_plan.json",
    "state.py",
    "workflow.py",
)
PUBLIC_TRANSFER_DEPENDENCIES = {
    "ctrl_world_hybrid_memory_transfer_v1": (
        "README.md",
        "evaluate.py",
        "materialize.py",
        "run.py",
    ),
    "ctrl_world_research_loop_v2": (
        "README.md",
        "research_intake.py",
    ),
    "ctrl_world_acwm_guidance_v1": (
        "README.md",
        "research_intake.py",
    ),
}

LOCAL_DEPLOYMENT_CONFIGS = (
    "ctrl_world_autonomous_transfer_loop_v1.json",
    "ctrl_world_autonomous_transfer_loop_v2.json",
    "ctrl_world_autonomous_transfer_loop_v3.json",
    "ctrl_world_dawm_bounded_history_materialization_v1.json",
    "ctrl_world_hybrid_relevance_memory_materialization_v1.json",
    "ctrl_world_masked_intermediate_action_adapter_materialization_v1.json",
)
TEXT_SUFFIXES = {
    ".css", ".csv", ".html", ".json", ".jsonl", ".js", ".md", ".py",
    ".service", ".sh", ".socket", ".svg", ".tex", ".toml", ".txt", ".yaml", ".yml"
}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".atom", ".lock", ".mp4", ".pdf", ".png", ".sha256", ""}
BLOCKED_SUFFIXES = {".bin", ".ckpt", ".db", ".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth", ".safetensors"}
BLOCKED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__", "results", "runs", "vendor"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def build_github_staging(*, source_root: Path, output_root: Path) -> dict[str, object]:
    source = Path(source_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise GithubStagingError("GITHUB_STAGING_OUTPUT_EXISTS")
    if not (source / "wmloop" / "geometry" / "irg.py").is_file():
        raise GithubStagingError("GITHUB_STAGING_SOURCE_INVALID")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        release_readme = source / "README_PUBLIC.md"
        if not release_readme.is_file():
            release_readme = source / "README.md"
        _copy_file(release_readme, temporary / "README.md")
        for relative in ROOT_FILES:
            _copy_file(source / relative, temporary / relative)
        _copy_file(
            source / "scripts/export/public_pyproject.toml",
            temporary / "pyproject.toml",
        )
        (temporary / ".verdiwm-public-release").write_text(
            "This tree is generated by build_github_staging.py.\n",
            encoding="utf-8",
        )
        for tree in PUBLIC_TREES:
            _copy_tree(source / tree, temporary / tree)
        for name in PUBLIC_TEST_FILES:
            _copy_file(source / "tests" / name, temporary / "tests" / name)
        for tree in CONFIG_TREES:
            config_tree = source / "configs" / tree
            if not config_tree.is_dir() and (source / ".verdiwm-public-release").is_file():
                # Public staging intentionally omits development-only empty trees.
                continue
            _copy_tree(config_tree, temporary / "configs" / tree)
        public_transfer_root = temporary / "experiments/ctrl_world_autonomous_transfer_v1"
        for relative in PUBLIC_AUTONOMOUS_TRANSFER_FILES:
            _copy_file(
                source / "experiments/ctrl_world_autonomous_transfer_v1" / relative,
                public_transfer_root / relative,
            )
        _copy_file(
            source / "scripts/export/ctrl_world_autonomous_transfer_public_README.md",
            public_transfer_root / "README.md",
        )
        _copy_file(
            source / "scripts/export/ctrl_world_autonomous_transfer_public_engineering_manifest.json",
            public_transfer_root / "engineering_manifest.json",
        )
        for package, files in PUBLIC_TRANSFER_DEPENDENCIES.items():
            dependency_root = temporary / "experiments" / package
            for relative in files:
                _copy_file(source / "experiments" / package / relative, dependency_root / relative)
        _copy_tree(
            source / "examples/portrait_first_minimal_loop_v1",
            temporary / "examples/portrait_first_minimal_loop_v1",
        )
        for name in LOCAL_DEPLOYMENT_CONFIGS:
            path = temporary / "configs" / "experiments" / name
            if path.exists():
                path.unlink()
        # These fresh autonomous-run manifests are local deployment recipes;
        # they contain host checkout paths and are never part of the public tree.
        for relative in (
            "configs/experiments/ctrl_world_fresh_action_fingerprint_screen_v1.json",
            "configs/experiments/ctrl_world_fresh_observation_smoke_v1.json",
            "scripts/compose_ctrl_world_fresh_autonomous_research.py",
        ):
            path = temporary / relative
            if path.exists():
                path.unlink()
        for name in (
            "acwm_self_forcing_source_effect_repair_r1.json",
            "acwm_self_forcing_source_effect_repair_r1.sha256",
            "acwm_self_forcing_robot3311_eval_seed_repair_r1.json",
            "acwm_self_forcing_robot3311_eval_seed_repair_r1.sha256",
            "acwm_self_forcing_robot3322_eval_seed_repair_r1.json",
            "acwm_self_forcing_robot3322_eval_seed_repair_r1.sha256",
        ):
            path = temporary / "configs" / "experiments" / name
            if path.exists():
                path.unlink()
        for name in (
            "cosmos3_forward_dynamics_predictive_pilot_v1.json",
            "cosmos3_forward_dynamics_predictive_pilot_v1.freeze.json",
        ):
            path = temporary / "configs" / "constitution" / name
            if path.exists():
                path.unlink()
        for name in (
            "eval_frozen.sha256",
            "eval_ctrl_world_g2_frozen.sha256",
            "eval_ctrl_world_predictive_v1.sha256",
            "registry_frozen.sha256",
            "registry_ctrl_world_g2.sha256",
            "registry_cosmos3_forward_dynamics_v1.json",
        ):
            _copy_file(source / "configs" / name, temporary / "configs" / name)
        _copy_file(
            _public_source(
                source,
                "configs/backbones/acwm_phys_g1_long_horizon_ladder_public_v1.json",
                "configs/backbones/acwm_phys_g1_long_horizon_ladder_v1.json",
            ),
            temporary / "configs" / "backbones" / "acwm_phys_g1_long_horizon_ladder_v1.json",
        )
        _copy_file(
            _public_source(
                source,
                "configs/backbones/ctrl_world_predictive_quality_public_v1.json",
                "configs/backbones/ctrl_world_predictive_quality_pilot_v1.json",
            ),
            temporary / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v2.json",
            temporary / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v2.json",
        )
        _copy_file(
            _public_source(
                source,
                "configs/backbones/ctrl_world_g2_action_success_public_v1.json",
                "configs/backbones/ctrl_world_g2_action_success_pilot_v1.json",
            ),
            temporary / "configs" / "backbones" / "ctrl_world_g2_action_success_pilot_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "ctrl_world_g2_dataset_freeze.json",
            temporary / "configs" / "backbones" / "ctrl_world_g2_dataset_freeze.json",
        )
        _copy_file(
            _public_source(
                source,
                "configs/backbones/cosmos3_forward_dynamics_predictive_public_v1.json",
                "configs/backbones/cosmos3_forward_dynamics_predictive_pilot_v1.json",
            ),
            temporary / "configs" / "backbones" / "cosmos3_forward_dynamics_predictive_pilot_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "cosmos3_droid_lerobot_dataset_freeze_v1.json",
            temporary / "configs" / "backbones" / "cosmos3_droid_lerobot_dataset_freeze_v1.json",
        )
        public_docs = source / "docs" / "public"
        if public_docs.is_dir():
            _copy_tree(public_docs, temporary / "docs")
        for name in PUBLIC_DOC_FILES:
            _copy_file(source / "docs" / name, temporary / "docs" / name)
        _copy_tree(source / "figures", temporary / "figures")
        _copy_tree(
            source / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2",
            temporary / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2",
        )
        _copy_tree(
            source / "examples" / "acwm_experience_atlas_v1",
            temporary / "examples" / "acwm_experience_atlas_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_eval_seed_replication_v1",
            temporary / "examples" / "acwm_eval_seed_replication_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_training_seed_horizon_stability_v1",
            temporary / "examples" / "acwm_training_seed_horizon_stability_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_training_seed_replication_cloth_self_forcing_v1",
            temporary / "examples" / "acwm_training_seed_replication_cloth_self_forcing_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_source_effect_repair_v1",
            temporary / "examples" / "acwm_source_effect_repair_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_progressive_fidelity_efficiency_v1",
            temporary / "examples" / "acwm_progressive_fidelity_efficiency_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_selector_ablation_v1",
            temporary / "examples" / "acwm_selector_ablation_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_probe_information_collision_s4_v1",
            temporary / "examples" / "acwm_probe_information_collision_s4_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_paper_primitive_matrix_v1",
            temporary / "examples" / "acwm_paper_primitive_matrix_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_method_evidence_maps_v1",
            temporary / "examples" / "acwm_method_evidence_maps_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_unified_irg_assets_v1",
            temporary / "examples" / "acwm_unified_irg_assets_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_joint_irg_assets_v2",
            temporary / "examples" / "acwm_joint_irg_assets_v2",
        )
        _copy_tree(
            source / "examples" / "cpbe_algorithm_smoke_v1",
            temporary / "examples" / "cpbe_algorithm_smoke_v1",
        )
        _copy_tree(
            source / "examples" / "cpbe_algorithm_smoke_settlement_v1",
            temporary / "examples" / "cpbe_algorithm_smoke_settlement_v1",
        )
        _copy_tree(
            source / "examples" / "ctrl_world_target_local_irg_v1",
            temporary / "examples" / "ctrl_world_target_local_irg_v1",
        )
        _copy_tree(
            source / "examples" / "ctrl_world_paper_split_abstention_v1",
            temporary / "examples" / "ctrl_world_paper_split_abstention_v1",
        )
        _copy_tree(
            source / "examples" / "r31_cross_backbone_runtime_v1",
            temporary / "examples" / "r31_cross_backbone_runtime_v1",
        )
        _copy_tree(
            source / "examples" / "cosmos3_forward_dynamics_instance_v1",
            temporary / "examples" / "cosmos3_forward_dynamics_instance_v1",
        )
        _copy_tree(
            source / "examples" / "cosmos3_paired_gt_dev_v1",
            temporary / "examples" / "cosmos3_paired_gt_dev_v1",
        )
        for name in COSMOS3_PUBLIC_EXAMPLES:
            _copy_tree(source / "examples" / name, temporary / "examples" / name)
        _copy_tree(
            source / "examples" / COSMOS3_DIRECTIONAL_PUBLIC_EXAMPLE,
            temporary / "examples" / COSMOS3_DIRECTIONAL_PUBLIC_EXAMPLE,
        )
        _copy_tree(
            source / "examples" / COSMOS3_TRANSLATION_PUBLIC_EXAMPLE,
            temporary / "examples" / COSMOS3_TRANSLATION_PUBLIC_EXAMPLE,
        )
        _copy_tree(
            source / "examples" / COSMOS3_TRANSLATION_NARROW_PUBLIC_EXAMPLE,
            temporary / "examples" / COSMOS3_TRANSLATION_NARROW_PUBLIC_EXAMPLE,
        )
        _copy_tree(
            source / "examples" / COSMOS3_INTERACTION_PUBLIC_EXAMPLE,
            temporary / "examples" / COSMOS3_INTERACTION_PUBLIC_EXAMPLE,
        )

        validation = validate_public_example(temporary / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2")
        experience_validation = validate_public_experience_bundle(
            temporary / "examples" / "acwm_experience_atlas_v1"
        )
        training_seed_horizon_validation = validate_training_seed_horizon_public_bundle(
            temporary / "examples" / "acwm_training_seed_horizon_stability_v1"
        )
        portrait_first_validation = validate_portrait_first_public_example(
            temporary / "examples" / "portrait_first_minimal_loop_v1"
        )
        cosmos3_fingerprint_validations = {
            name: _validate_cosmos3_fingerprint_bundle(temporary / "examples" / name)
            for name in COSMOS3_PUBLIC_EXAMPLES
        }
        cosmos3_directional_settlement_validation = (
            _validate_cosmos3_directional_settlement_bundle(
                temporary / "examples" / COSMOS3_DIRECTIONAL_PUBLIC_EXAMPLE
            )
        )
        cosmos3_translation_settlement_validation = (
            _validate_cosmos3_directional_settlement_bundle(
                temporary / "examples" / COSMOS3_TRANSLATION_PUBLIC_EXAMPLE
            )
        )
        cosmos3_translation_narrow_settlement_validation = (
            _validate_cosmos3_directional_settlement_bundle(
                temporary / "examples" / COSMOS3_TRANSLATION_NARROW_PUBLIC_EXAMPLE
            )
        )
        cosmos3_interaction_settlement_validation = (
            _validate_cosmos3_directional_settlement_bundle(
                temporary / "examples" / COSMOS3_INTERACTION_PUBLIC_EXAMPLE
            )
        )
        findings = audit_release_tree(temporary)
        if findings:
            raise GithubStagingError("GITHUB_STAGING_AUDIT_FAILED:" + ";".join(findings[:20]))
        files = [path for path in sorted(temporary.rglob("*")) if path.is_file() and not path.is_symlink()]
        audit = {
            "schema_version": 1,
            "artifact_type": "verdiwm-github-release-audit",
            "state": "ready",
            "file_count_before_manifest": len(files),
            "total_bytes_before_manifest": sum(path.stat().st_size for path in files),
            "public_example_validation": validation,
            "public_experience_validation": experience_validation,
            "training_seed_horizon_validation": training_seed_horizon_validation,
            "portrait_first_validation": portrait_first_validation,
            "cosmos3_fingerprint_validations": cosmos3_fingerprint_validations,
            "cosmos3_directional_settlement_validation": (
                cosmos3_directional_settlement_validation
            ),
            "cosmos3_translation_settlement_validation": (
                cosmos3_translation_settlement_validation
            ),
            "cosmos3_translation_narrow_settlement_validation": (
                cosmos3_translation_narrow_settlement_validation
            ),
            "cosmos3_interaction_settlement_validation": cosmos3_interaction_settlement_validation,
            "checks": {
                "allowlist_copy": True,
                "no_symlinks": True,
                "no_blocked_paths": True,
                "no_blocked_extensions": True,
                "no_oversized_files": True,
                "no_machine_local_paths": True,
                "no_high_confidence_secrets": True,
            },
            "release_notes": [
                "Training data, model checkpoints, run outputs, archive databases, and vendor checkouts are excluded.",
                "VerdiWM source code is distributed under Apache-2.0; external assets retain their upstream terms.",
            ],
            "source_license": "Apache-2.0",
        }
        audit_path = temporary / "RELEASE_AUDIT.json"
        audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _write_manifest(temporary)
        post_findings = audit_release_tree(temporary)
        if post_findings:
            raise GithubStagingError("GITHUB_STAGING_FINAL_AUDIT_FAILED:" + ";".join(post_findings[:20]))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return audit


def audit_release_tree(root: Path) -> list[str]:
    findings: list[str] = []
    host_prefixes = tuple(
        "/" + part + "/" for part in ("home", "mnt", "root", "Users")
    ) + ("/" + "share" + "/project/",)
    secret_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    )
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"symlink:{relative}")
            continue
        if any(part in BLOCKED_PARTS for part in relative.parts):
            findings.append(f"blocked_path:{relative}")
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in BLOCKED_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
            findings.append(f"blocked_extension:{relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"oversized:{relative}")
        if suffix in TEXT_SUFFIXES or path.name in {"uv.lock", "MANIFEST.sha256"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append(f"invalid_text_encoding:{relative}")
                continue
            if any(prefix in text for prefix in host_prefixes):
                findings.append(f"local_path:{relative}")
            if any(pattern.search(text) for pattern in secret_patterns):
                findings.append(f"secret:{relative}")
    return findings


def _validate_cosmos3_fingerprint_bundle(root: Path) -> dict[str, object]:
    bundle_root = Path(root).resolve(strict=True)
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    if (
        not isinstance(bundle, dict)
        or bundle.get("artifact_type")
        != "verdiwm-cosmos3-target-local-fingerprint-public-bundle"
        or bundle.get("state") != "ready"
        or int(bundle.get("measurement_count", 0)) < 1
        or int(bundle.get("repeat_count", 0)) < 1
    ):
        raise GithubStagingError(f"GITHUB_STAGING_COSMOS3_BUNDLE_INVALID:{bundle_root.name}")

    manifest_path = bundle_root / "MANIFEST.sha256"
    declared: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise GithubStagingError(
                f"GITHUB_STAGING_COSMOS3_MANIFEST_INVALID:{bundle_root.name}"
            ) from exc
        target = (bundle_root / relative).resolve(strict=True)
        if bundle_root not in target.parents or target.is_symlink() or relative in declared:
            raise GithubStagingError(
                f"GITHUB_STAGING_COSMOS3_MANIFEST_PATH_INVALID:{bundle_root.name}"
            )
        if _sha256(target) != digest:
            raise GithubStagingError(
                f"GITHUB_STAGING_COSMOS3_MANIFEST_SHA_MISMATCH:{bundle_root.name}:{relative}"
            )
        declared.add(relative)
    actual = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if declared != actual:
        raise GithubStagingError(
            f"GITHUB_STAGING_COSMOS3_MANIFEST_COVERAGE_INVALID:{bundle_root.name}"
        )
    videos = bundle.get("videos")
    if not isinstance(videos, list) or any(
        not isinstance(row, dict) or row.get("path") not in actual for row in videos
    ):
        raise GithubStagingError(f"GITHUB_STAGING_COSMOS3_VIDEO_INDEX_INVALID:{bundle_root.name}")
    return {
        "artifact_type": bundle["artifact_type"],
        "state": bundle["state"],
        "campaign_id": bundle["campaign_id"],
        "probe_id": bundle.get("probe_id"),
        "measurement_count": bundle["measurement_count"],
        "repeat_count": bundle["repeat_count"],
        "locality_admission_state": bundle["locality_admission_state"],
        "cross_backbone_transfer_eligible": bundle["cross_backbone_transfer_eligible"],
        "video_count": len(videos),
        "manifest_file_count": len(declared),
    }


def _validate_cosmos3_directional_settlement_bundle(root: Path) -> dict[str, object]:
    bundle_root = Path(root).resolve(strict=True)
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    if (
        not isinstance(bundle, dict)
        or bundle.get("artifact_type")
        != "verdiwm-cosmos3-directional-settlement-public-bundle"
        or bundle.get("state") != "ready"
        or bundle.get("settlement_state") not in {"settled_licensed", "settled_abstained"}
        or bool(bundle.get("cross_backbone_transfer_eligible"))
        != (bundle.get("settlement_state") == "settled_licensed")
    ):
        raise GithubStagingError("GITHUB_STAGING_COSMOS3_DIRECTIONAL_BUNDLE_INVALID")
    if float(bundle["dev_accept_alignment_error"]) > float(bundle["maximum_alignment_error"]):
        if bundle["settlement_state"] != "settled_abstained":
            raise GithubStagingError(
                "GITHUB_STAGING_COSMOS3_DIRECTIONAL_ALIGNMENT_VERDICT_INVALID"
            )
    accept_validation = _validate_cosmos3_fingerprint_bundle(bundle_root / "accept")
    manifest_path = bundle_root / "MANIFEST.sha256"
    declared: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = (bundle_root / relative).resolve(strict=True)
        if bundle_root not in target.parents or target.is_symlink() or relative in declared:
            raise GithubStagingError("GITHUB_STAGING_COSMOS3_DIRECTIONAL_MANIFEST_INVALID")
        if _sha256(target) != digest:
            raise GithubStagingError(
                f"GITHUB_STAGING_COSMOS3_DIRECTIONAL_MANIFEST_SHA_MISMATCH:{relative}"
            )
        declared.add(relative)
    actual = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if declared != actual:
        raise GithubStagingError("GITHUB_STAGING_COSMOS3_DIRECTIONAL_MANIFEST_COVERAGE_INVALID")
    return {
        "artifact_type": bundle["artifact_type"],
        "state": bundle["state"],
        "campaign_id": bundle["campaign_id"],
        "settlement_state": bundle["settlement_state"],
        "dev_accept_alignment_error": bundle["dev_accept_alignment_error"],
        "maximum_alignment_error": bundle["maximum_alignment_error"],
        "cross_backbone_transfer_eligible": bundle["cross_backbone_transfer_eligible"],
        "accept_validation": accept_validation,
        "manifest_file_count": len(declared),
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise GithubStagingError(f"GITHUB_STAGING_SOURCE_TREE_MISSING:{source.name}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in BLOCKED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise GithubStagingError(f"GITHUB_STAGING_SOURCE_SYMLINK:{source.name}/{relative}")
        if path.is_file():
            _copy_file(path, destination / relative)


def _public_source(source_root: Path, preferred: str, released: str) -> Path:
    preferred_path = source_root / preferred
    if preferred_path.is_file():
        return preferred_path
    return source_root / released


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise GithubStagingError(f"GITHUB_STAGING_SOURCE_FILE_INVALID:{source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "MANIFEST.sha256":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_github_staging(source_root=args.source_root, output_root=args.output_root)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
