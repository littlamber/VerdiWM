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


class GithubStagingError(RuntimeError):
    """The release tree could not be built without violating publication rules."""


ROOT_FILES = (
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
)
PUBLIC_TREES = ("wmloop", "scripts", ".github", "ops")
PUBLIC_TEST_FILES = (
    "test_verdiwm_geometry.py",
    "test_verdiwm_public_release.py",
    "test_acwm_public_experience_bundle.py",
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
    "test_acwm_selector_ablation.py",
    "test_acwm_selector_cpu_replay.py",
    "test_acwm_selector_projection_compose.py",
    "test_acwm_selector_probe_admission.py",
    "test_acwm_paper_primitive_matrix.py",
    "test_acwm_method_evidence_maps.py",
    "test_acwm_selector_public_bundle.py",
    "test_acwm_self_rollout_history_probe.py",
    "test_acwm_fingerprint_campaign_runner.py",
    "test_ctrl_world_instance_adapters.py",
    "test_ctrl_world_predictive_adapter.py",
    "test_ctrl_world_predictive_instance.py",
    "test_ctrl_world_fingerprint.py",
    "test_ctrl_world_probe_evolution.py",
    "test_ctrl_world_predictive_campaign_runner.py",
    "test_primitive_materialization_prompt.py",
    "test_diagnostic_probe_materialization_prompt.py",
    "test_diagnostic_probe_routing_admission.py",
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
)
CONFIG_TREES = ("constitution", "diagnose", "envs", "experiments", "goal", "loop", "probes", "references", "schemas", "smoke")
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".service", ".sh", ".socket", ".svg", ".tex", ".toml", ".txt", ".yaml", ".yml"
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
        _copy_file(source / "README_PUBLIC.md", temporary / "README.md")
        for relative in ROOT_FILES:
            _copy_file(source / relative, temporary / relative)
        for tree in PUBLIC_TREES:
            _copy_tree(source / tree, temporary / tree)
        for name in PUBLIC_TEST_FILES:
            _copy_file(source / "tests" / name, temporary / "tests" / name)
        for tree in CONFIG_TREES:
            _copy_tree(source / "configs" / tree, temporary / "configs" / tree)
        for name in (
            "eval_frozen.sha256",
            "eval_ctrl_world_g2_frozen.sha256",
            "eval_ctrl_world_predictive_v1.sha256",
            "registry_frozen.sha256",
            "registry_ctrl_world_g2.sha256",
        ):
            _copy_file(source / "configs" / name, temporary / "configs" / name)
        _copy_file(
            source / "configs" / "backbones" / "acwm_phys_g1_long_horizon_ladder_public_v1.json",
            temporary / "configs" / "backbones" / "acwm_phys_g1_long_horizon_ladder_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "ctrl_world_predictive_quality_public_v1.json",
            temporary / "configs" / "backbones" / "ctrl_world_predictive_quality_pilot_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "ctrl_world_g2_action_success_public_v1.json",
            temporary / "configs" / "backbones" / "ctrl_world_g2_action_success_pilot_v1.json",
        )
        _copy_file(
            source / "configs" / "backbones" / "ctrl_world_g2_dataset_freeze.json",
            temporary / "configs" / "backbones" / "ctrl_world_g2_dataset_freeze.json",
        )
        _copy_tree(source / "docs" / "public", temporary / "docs")
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
            source / "examples" / "acwm_selector_ablation_v1",
            temporary / "examples" / "acwm_selector_ablation_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_paper_primitive_matrix_v1",
            temporary / "examples" / "acwm_paper_primitive_matrix_v1",
        )
        _copy_tree(
            source / "examples" / "acwm_method_evidence_maps_v1",
            temporary / "examples" / "acwm_method_evidence_maps_v1",
        )

        validation = validate_public_example(temporary / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2")
        experience_validation = validate_public_experience_bundle(
            temporary / "examples" / "acwm_experience_atlas_v1"
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
    host_prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
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
