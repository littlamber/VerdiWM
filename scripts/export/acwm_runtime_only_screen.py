#!/usr/bin/env python3
"""Run a runtime-only primitive through the official ACWM quality gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export.acwm_formal_visualization import run_export
from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.execute.acwm_primitive_routes import RUNTIME_ONLY_PRIMITIVES
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.primitive_runtime_smoke import hook_unit_script
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import WorktreeSandbox
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.runtime_env import runtime_subprocess_env
from wmloop.vendor import verify_vendor_checkout


class RuntimeOnlyScreenError(RuntimeError):
    """A runtime-only quality screen failed closed."""


def run_runtime_only_screen(
    *,
    repo_root: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    environment: str,
    primitive: str,
    parameters: Mapping[str, object],
    gpu_index: int,
    gpu_audit_manifest: Path,
    seed: int,
    max_trajs: int = 8,
    max_saved_vids: int = 8,
    hard_case_top_k: int = 4,
) -> dict[str, object]:
    if primitive not in RUNTIME_ONLY_PRIMITIVES:
        raise RuntimeOnlyScreenError(f"RUNTIME_ONLY_PRIMITIVE_UNSUPPORTED:{primitive}")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_OUTPUT_EXISTS")
    authorization = verify_gpu_exclusivity_ready(
        gpu_audit_manifest,
        gpu_index=gpu_index,
        max_age_seconds=3600.0,
    )
    root = Path(repo_root).resolve(strict=True)
    runtime = Path(runtime_python).resolve(strict=True)
    data = Path(data_root).resolve(strict=True)
    checkpoints = Path(checkpoint_root).resolve(strict=True)
    freeze = Path(dataset_freeze).resolve(strict=True)
    heldout = Path(heldout_protocol).resolve(strict=True)
    spec = next((item for item in CANONICAL_ACWM_ENVIRONMENTS if item.environment == environment), None)
    if spec is None:
        raise RuntimeOnlyScreenError(f"RUNTIME_ONLY_ENVIRONMENT_UNKNOWN:{environment}")
    baseline_checkpoint = checkpoints / spec.checkpoint_relative_path
    if not baseline_checkpoint.is_file() or baseline_checkpoint.is_symlink():
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_BASELINE_CHECKPOINT_MISSING")
    source_revision = verify_vendor_checkout(root)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging_root = destination.parent / f".{destination.name}.runtime-staging-{uuid.uuid4().hex}"
    candidate_runtime = destination.parent / f"{destination.name}.retained_runtime"
    sandbox = WorktreeSandbox(vendor_root=root / "vendor" / "ACWM-Phys", runs_root=staging_root / "sandbox-runs")
    trial_id = f"runtime-only-{environment}-{primitive}-{seed}-{uuid.uuid4().hex[:8]}"
    lease = None
    try:
        lease = sandbox.create(trial_id=trial_id, expected_revision=source_revision)
        registry = PrimitiveRegistry.from_root(root)
        renderer = PrimitiveRenderer(registry)
        rendered = renderer.render_checked(
            worktree=lease.worktree,
            interventions=[{"primitive": primitive, "params": dict(parameters)}],
            hook_ios=_default_hook_ios(),
        )
        for item in rendered:
            _apply_diff(lease.worktree, item.diff)
        _run_hook_smoke(runtime, lease.worktree, primitive, data, checkpoints, gpu_index)
        if candidate_runtime.exists() or candidate_runtime.is_symlink():
            runtime_sha = _validate_existing_candidate_runtime(
                candidate_runtime,
                source_revision=source_revision,
                rendered=rendered,
            )
        else:
            shutil.copytree(
                lease.worktree,
                candidate_runtime,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
            )
            runtime_sha = runtime_tree_sha256(candidate_runtime)
            _write_json(
                candidate_runtime / "wmloop-runtime-manifest.json",
                {
                    "schema_version": 1,
                    "artifact_type": "wmloop-materialized-candidate-runtime",
                    "state": "ready",
                    "source_revision": source_revision,
                    "runtime_path": str(candidate_runtime),
                    "tree_sha256": runtime_sha,
                    "rendered_primitives": [
                        {"name": item.name, "diff_sha256": item.sha256} for item in rendered
                    ],
                },
            )
    finally:
        if lease is not None:
            sandbox.remove(lease)
        if staging_root.exists() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)
    result = run_export(
        output_root=destination,
        environment=environment,
        primitive=primitive,
        seed=seed,
        runtime_python=runtime,
        data_root=data,
        checkpoint_root=checkpoints,
        dataset_freeze=freeze,
        heldout_protocol=heldout,
        candidate_checkpoint=baseline_checkpoint,
        candidate_runtime_root=candidate_runtime,
        gpu_index=gpu_index,
        steps=50,
        split="ind_test",
        max_trajs=max_trajs,
        max_saved_vids=max_saved_vids,
        batch_size=1,
        num_workers=2,
        test_cuts=1,
        hard_case_top_k=hard_case_top_k,
        require_candidate_runtime_hook=True,
        dry_run=False,
    )
    result["execution_mode"] = "runtime_only"
    result["gpu_exclusivity_audit"] = authorization
    result["runtime_parameters"] = dict(parameters)
    hard_cases = result.get("hard_case_visualization")
    measurements = hard_cases.get("all_measurements", []) if isinstance(hard_cases, Mapping) else []
    identical = bool(measurements) and all(
        isinstance(record, Mapping) and record.get("candidate_prediction_identical_to_baseline") is True
        for record in measurements
    )
    result["runtime_effect_gate"] = {
        "pass": not identical,
        "reason": "candidate_predictions_identical_to_baseline" if identical else "candidate_outputs_changed",
        "sample_count": len(measurements) if isinstance(measurements, list) else 0,
    }
    result["runtime_result_classification"] = _classify_runtime_only_result(result)
    _write_json(destination / "manifest.json", result)
    return result


def _classify_runtime_only_result(result: Mapping[str, object]) -> dict[str, object]:
    receipt = result.get("candidate_runtime_hook_receipt")
    effect_gate = result.get("runtime_effect_gate")
    quality_gate = result.get("official_quality_gate")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("state") != "ready"
        or not isinstance(receipt.get("call_count"), int)
        or int(receipt["call_count"]) < 1
    ):
        return {
            "class": "invalid_materialization",
            "reason": "runtime_binding_failure",
            "conclusive_quality_attempt": False,
        }
    if not isinstance(effect_gate, Mapping) or effect_gate.get("pass") is not True:
        return {
            "class": "runtime_hook_no_effect",
            "reason": str(effect_gate.get("reason") if isinstance(effect_gate, Mapping) else "runtime_effect_gate_missing"),
            "conclusive_quality_attempt": True,
        }
    if isinstance(quality_gate, Mapping) and quality_gate.get("pass") is True:
        return {
            "class": "official_gate_pass",
            "reason": "candidate_outputs_changed_and_official_gate_passed",
            "conclusive_quality_attempt": True,
        }
    return {
        "class": "official_gate_failure",
        "reason": "candidate_outputs_changed_but_official_gate_failed",
        "conclusive_quality_attempt": True,
    }


def _run_hook_smoke(
    runtime: Path,
    worktree: Path,
    primitive: str,
    data_root: Path,
    checkpoint_root: Path,
    gpu_index: int,
) -> None:
    env = runtime_subprocess_env(
        runtime,
        extra={
            "PYTHONPATH": os.pathsep.join((str(worktree), str(ROOT))),
            "ACWM_DATA_ROOT": str(data_root),
            "WAN_VAE_PATH": str(checkpoint_root / "Wan2.1_VAE.pth"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        },
    )
    completed = subprocess.run(
        [str(runtime), "-c", hook_unit_script(primitive)],
        cwd=worktree,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeOnlyScreenError(f"RUNTIME_ONLY_HOOK_SMOKE_FAILED:{primitive}:{completed.returncode}")


def _validate_existing_candidate_runtime(
    candidate_runtime: Path,
    *,
    source_revision: str,
    rendered: list[object],
) -> str:
    """Reuse only a complete runtime matching the current materialization contract."""

    if not candidate_runtime.is_dir() or candidate_runtime.is_symlink():
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_INVALID")
    manifest_path = candidate_runtime / "wmloop-runtime-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_MANIFEST_INVALID") from None
    if not isinstance(manifest, Mapping) or manifest.get("state") != "ready":
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_NOT_READY")
    if manifest.get("source_revision") != source_revision:
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_SOURCE_MISMATCH")
    expected = [{"name": item.name, "diff_sha256": item.sha256} for item in rendered]
    if manifest.get("rendered_primitives") != expected:
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_MATERIALIZATION_MISMATCH")
    actual_sha = runtime_tree_sha256(candidate_runtime)
    if manifest.get("tree_sha256") != actual_sha:
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_CANDIDATE_RUNTIME_CHECKSUM_MISMATCH")
    return actual_sha


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-freeze", type=Path, required=True)
    parser.add_argument("--heldout-protocol", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-audit-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-trajs", type=int, default=8)
    parser.add_argument("--max-saved-vids", type=int, default=8)
    parser.add_argument("--hard-case-top-k", type=int, default=4)
    args = parser.parse_args(argv)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, Mapping):
        raise RuntimeOnlyScreenError("RUNTIME_ONLY_PARAMETERS_INVALID")
    result = run_runtime_only_screen(
        repo_root=args.repo_root,
        output_root=args.output_root,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        dataset_freeze=args.dataset_freeze,
        heldout_protocol=args.heldout_protocol,
        environment=args.environment,
        primitive=args.primitive,
        parameters=parameters,
        gpu_index=args.gpu_index,
        gpu_audit_manifest=args.gpu_audit_manifest,
        seed=args.seed,
        max_trajs=args.max_trajs,
        max_saved_vids=args.max_saved_vids,
        hard_case_top_k=args.hard_case_top_k,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
