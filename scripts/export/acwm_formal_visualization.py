#!/usr/bin/env python3
"""Export retained ACWM baseline-vs-candidate videos for a formal positive cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.runtime_env import runtime_subprocess_env


VENDOR_ROOT = ROOT / "vendor" / "ACWM-Phys"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))
DEFAULT_DATASET_FREEZE = ROOT / "runs/m0/protocol/dataset-freeze.json"
DEFAULT_HELDOUT_PROTOCOL = ROOT / "runs/m0/protocol/heldout-protocol.json"
DEFAULT_OUT = ROOT / "results/reports/acwm-formal-visualization-robot_arm-latent_motion_prior-1w-r1"


class AcwmFormalVisualizationError(RuntimeError):
    """Formal visualization export failed closed."""


def run_export(
    *,
    output_root: Path,
    environment: str = "robot_arm",
    primitive: str = "latent_motion_prior",
    seed: int = 621,
    training_seed: int | None = None,
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    dataset_freeze: Path = DEFAULT_DATASET_FREEZE,
    heldout_protocol: Path = DEFAULT_HELDOUT_PROTOCOL,
    candidate_checkpoint: Path | None = None,
    candidate_runtime_root: Path | None = None,
    gpu_index: int = 0,
    steps: int = 50,
    split: str = "ind_test",
    max_trajs: int = 3,
    max_saved_vids: int = 3,
    batch_size: int = 1,
    num_workers: int = 2,
    test_cuts: int = 1,
    hard_case_top_k: int = 0,
    require_candidate_runtime_hook: bool = False,
    checkpoint_transform_manifest: Path | None = None,
    checkpoint_delta_alpha: float | None = None,
    source_primitive: str | None = None,
    source_official_gate_manifest: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    if (
        seed < 1
        or (training_seed is not None and training_seed < 1)
        or gpu_index < 0
        or steps < 1
        or max_trajs < 1
        or max_saved_vids < 1
        or batch_size < 1
        or num_workers < 0
        or hard_case_top_k < 0
        or hard_case_top_k > max_saved_vids
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_ARGUMENT_INVALID")
    if split not in {"ind_test", "ood_test", "both"}:
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_SPLIT_INVALID")
    spec = _environment_spec(environment)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_OUTPUT_EXISTS")
    runtime = Path(runtime_python).resolve()
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    dataset_freeze = Path(dataset_freeze).resolve()
    heldout_protocol = Path(heldout_protocol).resolve()
    candidate = Path(candidate_checkpoint).resolve() if candidate_checkpoint is not None else _default_candidate_checkpoint(
        environment=environment,
        primitive=primitive,
        seed=seed,
    )
    candidate_runtime = (
        Path(candidate_runtime_root).resolve() if candidate_runtime_root is not None else VENDOR_ROOT
    )
    checkpoint_transform_provenance = _checkpoint_transform_provenance(
        primitive=primitive,
        environment=environment,
        seed=seed,
        candidate_checkpoint=candidate,
        transform_manifest=checkpoint_transform_manifest,
        delta_alpha=checkpoint_delta_alpha,
        source_primitive=source_primitive,
        source_official_gate_manifest=source_official_gate_manifest,
        dry_run=dry_run,
    )
    baseline = checkpoints / spec.checkpoint_relative_path
    required_paths = [
        runtime,
        baseline,
        data,
        checkpoints / "Wan2.1_VAE.pth",
        VENDOR_ROOT / "eval.py",
        dataset_freeze,
        heldout_protocol,
        candidate_runtime,
    ]
    if not dry_run:
        required_paths.append(candidate)
    for required in required_paths:
        if required.is_symlink() or not required.exists():
            raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_REQUIRED_PATH_MISSING:{required}")
    candidate_runtime_contract = (
        _candidate_runtime_contract(candidate_runtime) if not dry_run and candidate_runtime != VENDOR_ROOT else {}
    )
    if require_candidate_runtime_hook and candidate_runtime == VENDOR_ROOT:
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_RUNTIME_HOOK_CANDIDATE_REQUIRED")
    temporary = destination.parent / f".{destination.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        config = _write_eval_config(
            environment=environment,
            checkpoint_root=checkpoints,
            output_path=temporary / f"{environment}-eval-config.yaml",
        )
        baseline_root = temporary / "baseline_eval"
        candidate_root = temporary / "candidate_eval"
        candidate_hook_receipt = (
            temporary / "logs" / "candidate-runtime-hook-receipts.jsonl"
            if require_candidate_runtime_hook
            else None
        )
        env_arg = _vendor_environment_name(environment)
        baseline_command = _eval_command(
            runtime=runtime,
            env_arg=env_arg,
            config_path=config,
            checkpoint_path=baseline,
            output_root=baseline_root,
            runtime_root=VENDOR_ROOT,
            steps=steps,
            split=split,
            max_trajs=max_trajs,
            max_saved_vids=max_saved_vids,
            batch_size=batch_size,
            num_workers=num_workers,
            test_cuts=test_cuts,
            eval_seed=seed,
        )
        candidate_command = _eval_command(
            runtime=runtime,
            env_arg=env_arg,
            config_path=config,
            checkpoint_path=candidate,
            output_root=candidate_root,
            runtime_root=candidate_runtime,
            steps=steps,
            split=split,
            max_trajs=max_trajs,
            max_saved_vids=max_saved_vids,
            batch_size=batch_size,
            num_workers=num_workers,
            test_cuts=test_cuts,
            eval_seed=seed,
            hook_receipt_path=candidate_hook_receipt,
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-formal-visualization-export",
            "state": "planned" if dry_run else "running",
            "environment": environment,
            "primitive": primitive,
            "seed": seed,
            "eval_seed": seed,
            "training_seed": training_seed,
            "paired_randomness_policy": "common Python, NumPy, Torch, and CUDA seed for baseline and candidate",
            "steps": steps,
            "split": split,
            "max_trajs": max_trajs,
            "max_saved_vids": max_saved_vids,
            "hard_case_top_k": hard_case_top_k,
            "runtime_python": str(runtime),
            "data_root": str(data),
            "checkpoint_root": str(checkpoints),
            "baseline_checkpoint": str(baseline),
            "baseline_checkpoint_sha256": _sha256(baseline),
            "candidate_checkpoint": str(candidate),
            "candidate_checkpoint_sha256": _sha256(candidate) if candidate.is_file() else "",
            "baseline_runtime_sha256": runtime_tree_sha256(VENDOR_ROOT),
            "candidate_runtime_root": str(candidate_runtime),
            "candidate_runtime_sha256": runtime_tree_sha256(candidate_runtime),
            "candidate_runtime_contract": candidate_runtime_contract,
            "candidate_runtime_hook_required": require_candidate_runtime_hook,
            "checkpoint_transform_provenance": checkpoint_transform_provenance,
            "eval_config_path": str(destination / config.name),
            "baseline_command": baseline_command,
            "candidate_command": candidate_command,
            "baseline_command_text": _command_text(baseline_command),
            "candidate_command_text": _command_text(candidate_command),
            "protocol_provenance": {
                "eval_script_path": str(VENDOR_ROOT / "eval.py"),
                "eval_script_sha256": _sha256(VENDOR_ROOT / "eval.py"),
                "eval_config_path": str(destination / config.name),
                "eval_config_sha256": _sha256(config),
                "dataset_freeze_path": str(dataset_freeze),
                "dataset_freeze_sha256": _sha256(dataset_freeze),
                "heldout_protocol_path": str(heldout_protocol),
                "heldout_protocol_sha256": _sha256(heldout_protocol),
            },
            "paired_videos": [],
            "created_at": _utc_now(),
            "claim_boundary": (
                "Visualization export only. Numeric claims remain governed by the frozen M4 evaluator receipts; "
                "this script uses official ACWM eval.py --save_videos for retained visual evidence."
            ),
        }
        final_manifest = _public_manifest(manifest, temporary=temporary, destination=destination)
        _write_json(temporary / "manifest.json", final_manifest)
        if not dry_run:
            baseline_env = runtime_subprocess_env(
                runtime,
                extra={
                    "ACWM_DATA_ROOT": str(data),
                    "CUDA_VISIBLE_DEVICES": str(gpu_index),
                    "WANDB_MODE": "offline",
                },
            )
            candidate_env = dict(baseline_env)
            candidate_env["PYTHONPATH"] = _runtime_pythonpath(candidate_runtime)
            _run_command(
                baseline_command,
                cwd=VENDOR_ROOT,
                env=baseline_env,
                stdout_path=temporary / "logs" / "baseline.stdout",
                stderr_path=temporary / "logs" / "baseline.stderr",
            )
            _run_command(
                candidate_command,
                cwd=candidate_runtime,
                env=candidate_env,
                stdout_path=temporary / "logs" / "candidate.stdout",
                stderr_path=temporary / "logs" / "candidate.stderr",
            )
            if require_candidate_runtime_hook:
                rendered = candidate_runtime_contract.get("rendered_primitives", [])
                expected_primitives = {
                    str(record.get("name"))
                    for record in rendered
                    if isinstance(record, Mapping) and isinstance(record.get("name"), str)
                }
                manifest["candidate_runtime_hook_receipt"] = _load_candidate_runtime_hook_receipt(
                    candidate_hook_receipt,
                    expected_primitives=expected_primitives,
                )
            baseline_metrics = _parse_eval_results(baseline_root / "results.md", environment=environment, split=split)
            candidate_metrics = _parse_eval_results(candidate_root / "results.md", environment=environment, split=split)
            manifest["official_quality_gate"] = _official_quality_gate(
                baseline=baseline_metrics,
                candidate=candidate_metrics,
            )
            paired = _pair_saved_videos(
                runtime=runtime,
                env=baseline_env,
                baseline_root=baseline_root,
                candidate_root=candidate_root,
                output_root=temporary / "paired_videos",
                env_arg=env_arg,
                steps=steps,
                split=split,
                max_saved_vids=max_saved_vids,
            )
            manifest["state"] = "ready"
            manifest["paired_videos"] = paired
            manifest["paired_video_count"] = len(paired)
            if hard_case_top_k:
                manifest["hard_case_visualization"] = _select_baseline_hard_cases(
                    runtime=runtime,
                    env=baseline_env,
                    paired_videos=paired,
                    output_root=temporary / "hard_cases",
                    top_k=hard_case_top_k,
                )
            manifest["completed_at"] = _utc_now()
            final_manifest = _public_manifest(manifest, temporary=temporary, destination=destination)
            _write_visual_csv(temporary / "visual_asset_manifest.csv", final_manifest["paired_videos"])  # type: ignore[arg-type]
            _write_json(temporary / "manifest.json", final_manifest)
        os.replace(temporary, destination)
        return final_manifest
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink():
                temporary.unlink()
            else:
                import shutil

                shutil.rmtree(temporary)
        raise


def _environment_spec(environment: str) -> Any:
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_ENVIRONMENT_UNKNOWN:{environment}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_runtime_contract(runtime_root: Path) -> dict[str, object]:
    manifest_path = runtime_root / "wmloop-runtime-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CANDIDATE_RUNTIME_MANIFEST_MISSING")
    payload = _load_json(manifest_path)
    if payload.get("artifact_type") != "wmloop-materialized-candidate-runtime" or payload.get("state") != "ready":
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CANDIDATE_RUNTIME_MANIFEST_INVALID")
    expected = payload.get("tree_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or runtime_tree_sha256(runtime_root) != expected:
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CANDIDATE_RUNTIME_HASH_MISMATCH")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "tree_sha256": expected,
        "source_revision": payload.get("source_revision"),
        "rendered_primitives": payload.get("rendered_primitives", []),
    }


def _vendor_environment_name(environment: str) -> str:
    return "clothmove" if environment == "cloth_move" else environment


def _default_candidate_checkpoint(*, environment: str, primitive: str, seed: int) -> Path:
    return (
        ROOT
        / "results"
        / "reports"
        / f"acwm-gap-formal_confirmation-{environment}-{primitive}-s{seed}-r1"
        / "retained_training"
        / "latest.pt"
    )


def _parse_eval_results(path: Path, *, environment: str, split: str) -> dict[str, float]:
    """Parse the aggregate metric row emitted by official ACWM eval.py."""
    if not path.is_file():
        raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_RESULTS_MISSING:{path}")
    requested_splits = {"ind_test", "ood_test"} if split == "both" else {split}
    accepted_environments = {environment, _vendor_environment_name(environment)}
    matches: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) != 7 or cells[0] not in accepted_environments or cells[1] not in requested_splits:
            continue
        try:
            matches.append(
                {
                    "steps": float(cells[2]),
                    "mse": float(cells[3]),
                    "masked_mse": float(cells[4]),
                    "psnr": float(cells[5]),
                    "ssim": float(cells[6]),
                }
            )
        except ValueError as error:
            raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_RESULTS_INVALID:{path}") from error
    if len(matches) != 1:
        raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_RESULTS_ROW_AMBIGUOUS:{path}:{len(matches)}")
    return matches[0]


def _official_quality_gate(*, baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    delta = {
        "psnr": candidate["psnr"] - baseline["psnr"],
        "ssim": candidate["ssim"] - baseline["ssim"],
        "mse": candidate["mse"] - baseline["mse"],
        "masked_mse": candidate["masked_mse"] - baseline["masked_mse"],
    }
    checks = {
        "psnr_strictly_improves": delta["psnr"] > 0.0,
        "ssim_does_not_regress": delta["ssim"] >= 0.0,
        "mse_does_not_regress": delta["mse"] <= 0.0,
        "masked_mse_does_not_regress": delta["masked_mse"] <= 0.0,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "state": "pass" if passed else "fail",
        "pass": passed,
        "protocol": "official_acwm_eval_py",
        "rule": "PSNR must improve; SSIM, MSE, and masked-MSE must not regress.",
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": delta,
        "checks": checks,
    }


def _write_eval_config(*, environment: str, checkpoint_root: Path, output_path: Path) -> Path:
    env_name = _vendor_environment_name(environment)
    config_path = VENDOR_ROOT / "configs" / "envs" / f"{env_name}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CONFIG_INVALID")
    model_type = config.pop("model_type", None)
    if isinstance(model_type, str) and model_type:
        model_path = VENDOR_ROOT / "configs" / "model" / f"{model_type}.yaml"
        with model_path.open("r", encoding="utf-8") as handle:
            model_spec = yaml.safe_load(handle)
        if not isinstance(model_spec, dict):
            raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CONFIG_INVALID")
        model_config = dict(config.get("model_config") or {})
        model_config.update(dict(model_spec.get("model_config") or {}))
        config["model_config"] = model_config
    model_config = dict(config.get("model_config") or {})
    model_config["vae_config"] = [str((checkpoint_root / "Wan2.1_VAE.pth").resolve())]
    model_config["use_flash_attn"] = False
    config["model_config"] = model_config
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)
    return output_path


def _eval_command(
    *,
    runtime: Path,
    env_arg: str,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    runtime_root: Path,
    steps: int,
    split: str,
    max_trajs: int,
    max_saved_vids: int,
    batch_size: int,
    num_workers: int,
    test_cuts: int,
    eval_seed: int,
    hook_receipt_path: Path | None = None,
) -> list[str]:
    seed_wrapper = "\n".join(
        (
            "import importlib, json, os, random, runpy, sys",
            "from pathlib import Path",
            "seed = int(sys.argv[1])",
            "runtime_root = Path(sys.argv[2])",
            "receipt_arg = sys.argv[3]",
            "script = sys.argv[4]",
            "sys.path.insert(0, str(runtime_root))",
            "import numpy as np",
            "import torch",
            "receipt_path = Path(receipt_arg) if receipt_arg else None",
            "manifest_path = runtime_root / 'wmloop-runtime-manifest.json'",
            "payload = json.loads(manifest_path.read_text(encoding='utf-8')) if receipt_path and manifest_path.is_file() else {}",
            "rendered = payload.get('rendered_primitives', []) if isinstance(payload, dict) else []",
            "if receipt_path:",
            "    for item in rendered:",
            "        if not isinstance(item, dict) or not isinstance(item.get('name'), str):",
            "            continue",
            "        primitive = str(item['name'])",
            "        sidecar_path = runtime_root / 'wmloop_interventions' / f'{primitive}.json'",
            "        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))",
            "        hook = sidecar.get('runtime_hook') if isinstance(sidecar, dict) else None",
            "        if not isinstance(hook, dict):",
            "            continue",
            "        module_name = hook.get('module')",
            "        function_name = hook.get('function')",
            "        if not isinstance(module_name, str) or not isinstance(function_name, str):",
            "            continue",
            "        module = importlib.import_module(module_name)",
            "        original = getattr(module, function_name)",
            "        def tracked(*args, __original=original, __primitive=primitive, __module=module_name, __function=function_name, **kwargs):",
            "            receipt_path.parent.mkdir(parents=True, exist_ok=True)",
            "            record = {'primitive': __primitive, 'module': __module, 'function': __function, 'pid': os.getpid()}",
            "            with receipt_path.open('a', encoding='utf-8') as handle:",
            "                handle.write(json.dumps(record, sort_keys=True) + '\\n')",
            "            return __original(*args, **kwargs)",
            "        setattr(module, function_name, tracked)",
            "random.seed(seed)",
            "np.random.seed(seed)",
            "torch.manual_seed(seed)",
            "torch.cuda.manual_seed_all(seed)",
            "sys.argv = [script, *sys.argv[5:]]",
            "runpy.run_path(script, run_name='__main__')",
        )
    )
    return [
        str(runtime),
        "-c",
        seed_wrapper,
        str(eval_seed),
        str(runtime_root),
        str(hook_receipt_path) if hook_receipt_path is not None else "",
        str(VENDOR_ROOT / "eval.py"),
        "--env",
        env_arg,
        "--cfg",
        str(config_path),
        "--ckpt",
        str(checkpoint_path),
        "--steps",
        str(steps),
        "--split",
        split,
        "--max_trajs",
        str(max_trajs),
        "--test_cuts",
        str(test_cuts),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--output_root",
        str(output_root),
        "--save_videos",
        "--max_saved_vids",
        str(max_saved_vids),
    ]


def _load_candidate_runtime_hook_receipt(
    path: Path | None,
    *,
    expected_primitives: set[str],
) -> dict[str, object]:
    if path is None or not path.is_file() or path.is_symlink():
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_RUNTIME_HOOK_NOT_INVOKED")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_RUNTIME_HOOK_RECEIPT_INVALID") from exc
        if not isinstance(record, dict) or not isinstance(record.get("primitive"), str):
            raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_RUNTIME_HOOK_RECEIPT_INVALID")
        records.append(record)
    observed = {str(record["primitive"]) for record in records}
    if not records or not expected_primitives or not expected_primitives.issubset(observed):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_RUNTIME_HOOK_NOT_INVOKED")
    return {
        "schema_version": 1,
        "state": "ready",
        "path": str(path),
        "call_count": len(records),
        "expected_primitives": sorted(expected_primitives),
        "observed_primitives": sorted(observed),
    }


def _runtime_pythonpath(runtime_root: Path) -> str:
    """Put a trial runtime ahead of the frozen vendor tree for candidate eval."""

    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(runtime_root), str(VENDOR_ROOT)]
    if existing:
        parts.extend(part for part in existing.split(os.pathsep) if part)
    return os.pathsep.join(dict.fromkeys(parts))


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_EVAL_FAILED:{completed.returncode}:{stdout_path}:{stderr_path}")


def _pair_saved_videos(
    *,
    runtime: Path,
    env: dict[str, str],
    baseline_root: Path,
    candidate_root: Path,
    output_root: Path,
    env_arg: str,
    steps: int,
    split: str,
    max_saved_vids: int,
) -> list[dict[str, object]]:
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec_path = output_root.parent / "pairing-spec.json"
    result_path = output_root.parent / "pairing-result.json"
    _write_json(
        spec_path,
        {
            "baseline_root": str(baseline_root),
            "candidate_root": str(candidate_root),
            "output_root": str(output_root),
            "env_arg": env_arg,
            "steps": steps,
            "split": split,
            "max_saved_vids": max_saved_vids,
        },
    )
    helper = r"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result_path = Path(sys.argv[2])
baseline_root = Path(spec["baseline_root"])
candidate_root = Path(spec["candidate_root"])
output_root = Path(spec["output_root"])
env_arg = spec["env_arg"]
steps = int(spec["steps"])
splits = ["ind_test", "ood_test"] if spec["split"] == "both" else [spec["split"]]
max_saved_vids = int(spec["max_saved_vids"])
rows = []
for split_name in splits:
    for sample_index in range(max_saved_vids):
        baseline_video = baseline_root / env_arg / f"steps_{steps}" / split_name / f"sample_{sample_index}" / "video.mp4"
        candidate_video = candidate_root / env_arg / f"steps_{steps}" / split_name / f"sample_{sample_index}" / "video.mp4"
        if not baseline_video.is_file() or not candidate_video.is_file():
            continue
        out_path = output_root / f"{split_name}_sample_{sample_index:02d}_gt_baseline_candidate.mp4"
        baseline_frames = imageio.mimread(baseline_video)
        candidate_frames = imageio.mimread(candidate_video)
        frame_count = min(len(baseline_frames), len(candidate_frames))
        if frame_count < 1:
            continue
        paired_frames = []
        for baseline_frame, candidate_frame in zip(baseline_frames[:frame_count], candidate_frames[:frame_count]):
            baseline_np = np.asarray(baseline_frame)
            candidate_np = np.asarray(candidate_frame)
            width = min(baseline_np.shape[1], candidate_np.shape[1])
            half = width // 2
            height = min(baseline_np.shape[0], candidate_np.shape[0])
            gt = baseline_np[:height, :half]
            baseline_pred = baseline_np[:height, half : half * 2]
            candidate_pred = candidate_np[:height, half : half * 2]
            triptych = np.concatenate([gt, baseline_pred, candidate_pred], axis=1)
            header_height = 30
            canvas = np.zeros((height + header_height, triptych.shape[1], 3), dtype=np.uint8)
            canvas[:header_height] = 20
            canvas[header_height:] = triptych
            image = Image.fromarray(canvas)
            draw = ImageDraw.Draw(image)
            font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
            font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
            for panel_index, label in enumerate(('GT', 'Baseline', 'Ours')):
                box = draw.textbbox((0, 0), label, font=font)
                text_width = box[2] - box[0]
                x = panel_index * half + max(0, (half - text_width) // 2)
                draw.text((x, 5), label, fill=(255, 255, 255), font=font)
            paired_frames.append(np.asarray(image))
        imageio.mimsave(out_path, paired_frames, fps=10)
        rows.append(
            {
                "split": split_name,
                "sample_index": sample_index,
                "paired_video_path": str(out_path),
                "baseline_video_path": str(baseline_video),
                "candidate_video_path": str(candidate_video),
                "layout": "labeled_GT|baseline_prediction|ours_prediction",
                "frame_count": frame_count,
            }
        )
result_path.write_text(json.dumps({"rows": rows}, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
"""
    completed = subprocess.run(
        [str(runtime), "-c", helper, str(spec_path), str(result_path)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AcwmFormalVisualizationError(
            f"ACWM_FORMAL_VIS_PAIRING_FAILED:{completed.returncode}:{completed.stderr[-1000:]}"
        )
    result = _load_json(result_path)
    raw_rows = result.get("rows")
    rows = [dict(row) for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    if not rows:
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_NO_VIDEOS_PAIRED")
    return rows


def _write_visual_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fields = ["split", "sample_index", "layout", "frame_count", "paired_video_path", "baseline_video_path", "candidate_video_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rank_baseline_hard_cases(
    records: list[Mapping[str, object]],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Rank qualitative cases using baseline error only, never candidate gain."""

    if top_k < 1:
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_HARD_CASE_TOP_K_INVALID")
    parsed: list[dict[str, object]] = []
    for record in records:
        baseline_mse = record.get("baseline_video_mse")
        sample_index = record.get("sample_index")
        if (
            isinstance(baseline_mse, bool)
            or not isinstance(baseline_mse, (int, float))
            or not isinstance(sample_index, int)
        ):
            raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_HARD_CASE_RECORD_INVALID")
        parsed.append(dict(record))
    return sorted(
        parsed,
        key=lambda item: (-float(item["baseline_video_mse"]), str(item.get("split") or ""), int(item["sample_index"])),
    )[:top_k]


def _select_baseline_hard_cases(
    *,
    runtime: Path,
    env: dict[str, str],
    paired_videos: list[dict[str, object]],
    output_root: Path,
    top_k: int,
) -> dict[str, object]:
    spec_path = output_root.parent / "hard-case-measurement-spec.json"
    result_path = output_root.parent / "hard-case-measurements.json"
    _write_json(spec_path, {"paired_videos": paired_videos})
    helper = r"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = []
for record in spec["paired_videos"]:
    baseline_path = Path(record["baseline_video_path"])
    candidate_path = Path(record["candidate_video_path"])
    baseline_frames = imageio.mimread(baseline_path)
    candidate_frames = imageio.mimread(candidate_path)
    baseline_errors = []
    candidate_errors = []
    prediction_identical = True
    for baseline_frame, candidate_frame in zip(baseline_frames, candidate_frames):
        baseline_array = np.asarray(baseline_frame, dtype=np.float32) / 255.0
        candidate_array = np.asarray(candidate_frame, dtype=np.float32) / 255.0
        width = min(baseline_array.shape[1], candidate_array.shape[1])
        half = width // 2
        height = min(baseline_array.shape[0], candidate_array.shape[0])
        if half < 1 or height < 1:
            continue
        gt = baseline_array[:height, :half]
        baseline = baseline_array[:height, half:half * 2]
        candidate = candidate_array[:height, half:half * 2]
        baseline_errors.append(float(np.mean((baseline - gt) ** 2)))
        candidate_errors.append(float(np.mean((candidate - gt) ** 2)))
        prediction_identical = prediction_identical and np.array_equal(baseline, candidate)
    if not baseline_errors:
        continue
    rows.append({
        **record,
        "baseline_video_mse": float(np.mean(baseline_errors)),
        "candidate_video_mse": float(np.mean(candidate_errors)),
        "candidate_minus_baseline_video_mse": float(np.mean(candidate_errors) - np.mean(baseline_errors)),
        "candidate_prediction_identical_to_baseline": prediction_identical,
    })
Path(sys.argv[2]).write_text(json.dumps({"rows": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
"""
    completed = subprocess.run(
        [str(runtime), "-c", helper, str(spec_path), str(result_path)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AcwmFormalVisualizationError(
            f"ACWM_FORMAL_VIS_HARD_CASE_MEASUREMENT_FAILED:{completed.returncode}:{completed.stderr[-1000:]}"
        )
    measured = _load_json(result_path).get("rows")
    if not isinstance(measured, list):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_HARD_CASE_RECORD_INVALID")
    selected = rank_baseline_hard_cases(
        [dict(record) for record in measured if isinstance(record, Mapping)],
        top_k=top_k,
    )
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for rank, record in enumerate(selected, start=1):
        source = Path(str(record["paired_video_path"]))
        target = output_root / f"rank_{rank:02d}_{record.get('split', 'ind_test')}_sample_{int(record['sample_index']):02d}.mp4"
        import shutil

        shutil.copy2(source, target)
        record["hard_case_rank"] = rank
        record["selected_video_path"] = str(target)
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-baseline-blind-hard-case-visualization",
        "state": "ready",
        "selection_rule": "descending baseline-only RGB video MSE against GT; candidate metrics are never used for ranking",
        "selection_pool_size": len(measured),
        "top_k": top_k,
        "selected": selected,
        "all_measurements": measured,
        "claim_boundary": "Qualitative hard-case view only. Aggregate official metrics over the frozen evaluation pool remain claim-governing.",
    }
    _write_json(output_root / "manifest.json", payload)
    return payload


def _public_manifest(manifest: dict[str, object], *, temporary: Path, destination: Path) -> dict[str, object]:
    return _rewrite_temp_prefix(json.loads(json.dumps(manifest)), temporary=temporary, destination=destination)


def _rewrite_temp_prefix(value: Any, *, temporary: Path, destination: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_temp_prefix(item, temporary=temporary, destination=destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_temp_prefix(item, temporary=temporary, destination=destination) for item in value]
    if isinstance(value, str):
        temp = str(temporary)
        if value == temp:
            return str(destination)
        if temp in value:
            return value.replace(temp, str(destination))
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_transform_provenance(
    *,
    primitive: str,
    environment: str,
    seed: int,
    candidate_checkpoint: Path,
    transform_manifest: Path | None,
    delta_alpha: float | None,
    source_primitive: str | None,
    source_official_gate_manifest: Path | None,
    dry_run: bool,
) -> dict[str, object] | None:
    values_present = any(
        value is not None
        for value in (
            transform_manifest,
            delta_alpha,
            source_primitive,
            source_official_gate_manifest,
        )
    )
    if primitive != "checkpoint_delta_scaling" and not values_present:
        return None
    if (
        primitive != "checkpoint_delta_scaling"
        or transform_manifest is None
        or delta_alpha is None
        or not source_primitive
        or source_official_gate_manifest is None
        or float(delta_alpha) == 0.0
        or not (-1.0 <= float(delta_alpha) <= 1.0)
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_ARGUMENT_INVALID")

    transform_path = Path(transform_manifest).resolve()
    source_gate_path = Path(source_official_gate_manifest).resolve()
    if dry_run and (not transform_path.is_file() or not source_gate_path.is_file()):
        return {
            "state": "planned",
            "transform_manifest_path": str(transform_path),
            "source_official_gate_manifest_path": str(source_gate_path),
            "source_primitive": source_primitive,
            "alpha": float(delta_alpha),
        }
    for path in (transform_path, source_gate_path):
        if path.is_symlink() or not path.is_file():
            raise AcwmFormalVisualizationError(
                f"ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_EVIDENCE_MISSING:{path}"
            )

    transform = _load_json(transform_path)
    if (
        transform.get("artifact_type") != "wmloop-checkpoint-delta-scaling"
        or transform.get("state") != "ready"
        or transform.get("environment") != environment
        or transform.get("source_primitive") != source_primitive
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_MANIFEST_INVALID")
    outputs = transform.get("outputs")
    if not isinstance(outputs, list):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_OUTPUTS_INVALID")
    selected = next(
        (
            record
            for record in outputs
            if isinstance(record, Mapping)
            and isinstance(record.get("alpha"), (int, float))
            and not isinstance(record.get("alpha"), bool)
            and abs(float(record["alpha"]) - float(delta_alpha)) < 1e-12
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_ALPHA_MISSING")
    selected_path = Path(str(selected.get("path") or "")).resolve()
    selected_sha = str(selected.get("sha256") or "")
    if (
        selected_path != candidate_checkpoint
        or not candidate_checkpoint.is_file()
        or selected_sha != _sha256(candidate_checkpoint)
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_CHECKPOINT_TRANSFORM_CANDIDATE_MISMATCH")

    source_gate = _load_json(source_gate_path)
    source_quality_gate = source_gate.get("official_quality_gate")
    if (
        source_gate.get("artifact_type") != "wmloop-acwm-formal-visualization-export"
        or source_gate.get("state") != "ready"
        or source_gate.get("environment") != environment
        or source_gate.get("primitive") != source_primitive
        or source_gate.get("seed") != transform.get("seed")
        or not isinstance(source_quality_gate, Mapping)
        or source_quality_gate.get("pass") is not False
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_SOURCE_OFFICIAL_GATE_INVALID")
    if str(source_gate.get("candidate_checkpoint_sha256") or "") != str(
        transform.get("candidate_sha256") or ""
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_SOURCE_CHECKPOINT_SHA_MISMATCH")
    if str(source_gate.get("baseline_checkpoint_sha256") or "") != str(
        transform.get("baseline_sha256") or ""
    ):
        raise AcwmFormalVisualizationError("ACWM_FORMAL_VIS_BASELINE_CHECKPOINT_SHA_MISMATCH")

    return {
        "state": "verified",
        "method_intent": "Apply a signed scale to the exact learned update after an internal-positive but official-negative result; negative alpha reflects the harmful direction.",
        "source_primitive": source_primitive,
        "source_seed": transform.get("seed"),
        "eval_seed": seed,
        "alpha": float(delta_alpha),
        "rule": str(transform.get("rule") or ""),
        "transform_manifest_path": str(transform_path),
        "transform_manifest_sha256": _sha256(transform_path),
        "source_official_gate_manifest_path": str(source_gate_path),
        "source_official_gate_manifest_sha256": _sha256(source_gate_path),
        "source_candidate_checkpoint_sha256": str(transform.get("candidate_sha256") or ""),
        "scaled_checkpoint_sha256": selected_sha,
        "claim_boundary": "The transform is provenance-verified; quality is established only by this official gate.",
    }


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmFormalVisualizationError(f"ACWM_FORMAL_VIS_JSON_NOT_OBJECT:{path}")
    return payload


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--environment", default="robot_arm")
    parser.add_argument("--primitive", default="latent_motion_prior")
    parser.add_argument("--seed", type=int, default=621)
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Independent repair-training seed associated with the candidate checkpoint.",
    )
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-freeze", type=Path, default=DEFAULT_DATASET_FREEZE)
    parser.add_argument("--heldout-protocol", type=Path, default=DEFAULT_HELDOUT_PROTOCOL)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-runtime-root", type=Path)
    parser.add_argument(
        "--require-candidate-runtime-hook",
        action="store_true",
        help="Fail unless every primitive declared by the candidate runtime is observed during evaluation.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--split", default="ind_test", choices=["ind_test", "ood_test", "both"])
    parser.add_argument("--max-trajs", type=int, default=3)
    parser.add_argument("--max-saved-vids", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-cuts", type=int, default=1)
    parser.add_argument("--hard-case-top-k", type=int, default=0)
    parser.add_argument("--checkpoint-transform-manifest", type=Path)
    parser.add_argument("--checkpoint-delta-alpha", type=float)
    parser.add_argument("--source-primitive")
    parser.add_argument("--source-official-gate-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = run_export(
        output_root=args.output_root,
        environment=args.environment,
        primitive=args.primitive,
        seed=args.seed,
        training_seed=args.training_seed,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        dataset_freeze=args.dataset_freeze,
        heldout_protocol=args.heldout_protocol,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_runtime_root=args.candidate_runtime_root,
        require_candidate_runtime_hook=args.require_candidate_runtime_hook,
        gpu_index=args.gpu_index,
        steps=args.steps,
        split=args.split,
        max_trajs=args.max_trajs,
        max_saved_vids=args.max_saved_vids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_cuts=args.test_cuts,
        hard_case_top_k=args.hard_case_top_k,
        checkpoint_transform_manifest=args.checkpoint_transform_manifest,
        checkpoint_delta_alpha=args.checkpoint_delta_alpha,
        source_primitive=args.source_primitive,
        source_official_gate_manifest=args.source_official_gate_manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
