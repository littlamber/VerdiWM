#!/usr/bin/env python3
"""Launch a constitution-checked, one-step Ctrl-World runtime smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmloop.constitution import verify_constitutional_freeze
from wmloop.primitives.adapters.backbone_registry import load_backbone_primitive_registry
from wmloop.primitives.adapters.ctrl_world_hooks import audit_ctrl_world_hooks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-meta-root", type=Path, required=True)
    parser.add_argument("--task-data-root", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer-python", type=Path, required=True)
    parser.add_argument("--instance-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve(strict=True)
    output = args.output_root.resolve()
    if args.reconcile_existing:
        return reconcile_existing(output=output, args=args)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    preflight = build_preflight(args, repo=repo)
    _write_json(output / "preflight.json", preflight)
    if preflight["state"] != "ready" or args.preflight_only:
        return 0 if preflight["state"] == "ready" else 2

    trainer_output = output / "trainer-output"
    trainer_output.mkdir(mode=0o700)
    log_path = output / "train.log"
    command = [
        str(args.trainer_python.resolve(strict=True)),
        "-u",
        "scripts/launch_train_wm_local.py",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "CTRLWORLD_SVD_MODEL_PATH": str(args.svd_model_path.resolve()),
            "CTRLWORLD_CLIP_MODEL_PATH": str(args.clip_model_path.resolve()),
            "CTRLWORLD_CKPT_PATH": str(args.checkpoint.resolve()),
            "CTRLWORLD_DATASET_ROOT": str(args.dataset_root.resolve()),
            "CTRLWORLD_DATASET_META": str(args.dataset_meta_root.resolve()),
            "CTRLWORLD_DATASET_NAMES": "droid_1.0.1",
            "CTRLWORLD_DATASET_CFGS": "droid_1.0.1",
            "CTRLWORLD_DATA_STAT_PATH": str((args.dataset_meta_root / "droid_1.0.1" / "stat.json").resolve()),
            "CTRLWORLD_TAG": output.name,
            "CTRLWORLD_OUTPUT_DIR": str(trainer_output),
            "CTRLWORLD_LOG_WITH": "none",
            "CTRLWORLD_BATCH_SIZE": "1",
            "CTRLWORLD_GRAD_ACCUM": "1",
            "CTRLWORLD_MAX_TRAIN_STEPS": str(args.max_train_steps),
            "CTRLWORLD_CHECKPOINT_STEPS": "1000",
            "CTRLWORLD_VALIDATION_STEPS": "1000",
            "CTRLWORLD_VIDEO_NUM": "1",
            "CTRLWORLD_NUM_WORKERS": "0",
            "CTRLWORLD_PERSISTENT_WORKERS": "false",
            "CTRLWORLD_PIN_MEMORY": "true",
            "CTRLWORLD_LOG_EVERY_STEPS": "1",
            "CTRLWORLD_MIXED_PRECISION": "bf16",
            "CTRLWORLD_USE_LAMO_TRAIN": "true",
            "CTRLWORLD_USE_LAMO_INFER": "true",
            "WANDB_MODE": "offline",
            "SWANLAB_MODE": "disabled",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DIR": str(output / "wandb"),
            "WANDB_CACHE_DIR": str(output / "wandb-cache"),
            "WANDB_DATA_DIR": str(output / "wandb-data"),
        }
    )
    launch = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-bounded-smoke-launch",
        "state": "launching",
        "physical_gpu": args.physical_gpu,
        "visible_gpu": 0,
        "max_train_steps": args.max_train_steps,
        "timeout_seconds": args.timeout_seconds,
        "command": command,
        "cwd": str(args.ctrl_world_root.resolve()),
        "primitive": "latent_motion_prior",
        "quality_claimed": False,
        "started_at_unix": time.time(),
    }
    _write_json(output / "run-state.json", launch)
    timed_out = False
    return_code: int | None = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=args.ctrl_world_root.resolve(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        launch["state"] = "running"
        launch["worker_pid"] = process.pid
        _write_json(output / "run-state.json", launch)
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.kill(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.kill(process.pid, signal.SIGKILL)
                return_code = process.wait()

    receipt = build_receipt(
        output=output,
        args=args,
        launch=launch,
        return_code=return_code,
        timed_out=timed_out,
    )
    passed = bool(receipt["runtime_smoke_passed"])
    _write_json(output / "receipt.json", receipt)
    launch.update({"state": receipt["state"], "completed_at_unix": receipt["completed_at_unix"]})
    _write_json(output / "run-state.json", launch)
    return 0 if passed else 3


def reconcile_existing(*, output: Path, args: argparse.Namespace) -> int:
    if not output.is_dir():
        raise ValueError("CTRL_WORLD_SMOKE_OUTPUT_MISSING")
    launch = dict(_load_json(output / "run-state.json"))
    previous = dict(_load_json(output / "receipt.json"))
    receipt = build_receipt(
        output=output,
        args=args,
        launch=launch,
        return_code=int(previous["return_code"]),
        timed_out=bool(previous["timed_out"]),
    )
    receipt["reconciled_from_existing_log"] = True
    _write_json(output / "receipt.json", receipt)
    launch.update({"state": receipt["state"], "completed_at_unix": receipt["completed_at_unix"]})
    _write_json(output / "run-state.json", launch)
    return 0 if receipt["runtime_smoke_passed"] else 3


def build_receipt(
    *,
    output: Path,
    args: argparse.Namespace,
    launch: Mapping[str, Any],
    return_code: int | None,
    timed_out: bool,
) -> dict[str, object]:
    log_path = output / "train.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    loss_lines = []
    for line in text.splitlines():
        marker = line.find("loss_window ")
        if marker >= 0:
            loss_lines.append(line[marker:].strip())
    passed = (
        return_code == 0
        and not timed_out
        and f"step={args.max_train_steps}" in "\n".join(loss_lines)
        and "use_lamo_train=True" in text
        and "use_lamo_infer=True" in text
        and "Traceback (most recent call last)" not in text
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-bounded-smoke-receipt",
        "state": "completed" if passed else "failed",
        "runtime_smoke_passed": passed,
        "quality_claimed": False,
        "transfer_claimed": False,
        "primitive": "latent_motion_prior",
        "physical_gpu": args.physical_gpu,
        "worker_pid": launch.get("worker_pid"),
        "return_code": return_code,
        "timed_out": timed_out,
        "max_train_steps": args.max_train_steps,
        "loss_window_lines": loss_lines,
        "train_log": str(log_path),
        "train_log_sha256": _sha256(log_path),
        "trainer_output": str(output / "trainer-output"),
        "completed_at_unix": time.time(),
    }


def build_preflight(args: argparse.Namespace, *, repo: Path) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    instance = _load_json(args.instance_manifest)
    if instance.get("state") != "ready" or instance.get("blocker_count") != 0:
        blockers.append({"code": "instance_not_ready", "detail": str(args.instance_manifest)})

    constitution_path = repo / "configs/constitution/ctrl_world_g2_action_success_pilot_v1.freeze.json"
    try:
        verify_constitutional_freeze(_load_json(constitution_path), root=repo)
        constitution_ready = True
    except Exception as exc:  # fail closed and preserve the concrete verifier reason
        constitution_ready = False
        blockers.append({"code": "constitution_invalid", "detail": str(exc)})

    try:
        primitive_registry = load_backbone_primitive_registry(
            repo / "configs/registry_ctrl_world_g2.sha256",
            root=repo,
        )
        primitive_ready = "latent_motion_prior" in primitive_registry.runtime_ready_primitives
        if not primitive_ready:
            blockers.append({"code": "primitive_not_runtime_ready", "detail": "latent_motion_prior"})
    except Exception as exc:
        primitive_ready = False
        blockers.append({"code": "primitive_registry_invalid", "detail": str(exc)})

    hook_audit = audit_ctrl_world_hooks(args.ctrl_world_root)
    if hook_audit["state"] != "ready":
        blockers.append({"code": "hook_audit_failed", "detail": json.dumps(hook_audit, sort_keys=True)})

    asset_checks = [
        _asset_check("ctrl_world_root", args.ctrl_world_root, kind="dir"),
        _asset_check("dataset_root", args.dataset_root / "droid_1.0.1", kind="dir"),
        _asset_check("dataset_train_meta", args.dataset_meta_root / "droid_1.0.1" / "train_sample.json", kind="file"),
        _asset_check("dataset_stat", args.dataset_meta_root / "droid_1.0.1" / "stat.json", kind="file"),
        _asset_check("svd_model", args.svd_model_path / "model_index.json", kind="file"),
        _asset_check("clip_model", args.clip_model_path / "config.json", kind="file"),
        _asset_check("checkpoint", args.checkpoint, kind="file", minimum_bytes=1_000_000_000),
    ]
    for check in asset_checks:
        if not check["passed"]:
            blockers.append({"code": "asset_missing_or_invalid", "detail": str(check["path"])})

    data_freeze = _load_json(repo / "configs/backbones/ctrl_world_g2_dataset_freeze.json")
    data_checks = []
    for entry in data_freeze["selected_files"]:
        path = (args.task_data_root / str(entry["path"])).resolve()
        passed = path.is_file() and path.stat().st_size == int(entry["size"]) and _sha256(path) == entry["sha256"]
        data_checks.append({"path": str(entry["path"]), "passed": passed})
        if not passed:
            blockers.append({"code": "heldout_data_drift", "detail": str(entry["path"])})

    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-bounded-smoke-preflight",
        "state": "ready" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "instance_ready": instance.get("state") == "ready" and instance.get("blocker_count") == 0,
        "constitution_ready": constitution_ready,
        "primitive_runtime_ready": primitive_ready,
        "hook_audit": hook_audit,
        "asset_checks": asset_checks,
        "heldout_data_check_count": len(data_checks),
        "heldout_data_checks_passed": sum(bool(item["passed"]) for item in data_checks),
        "gpu_requested": args.physical_gpu,
        "gpu_execution_started": False,
    }


def _asset_check(name: str, path: Path, *, kind: str, minimum_bytes: int = 1) -> dict[str, object]:
    resolved = path.resolve()
    passed = resolved.is_dir() if kind == "dir" else resolved.is_file() and resolved.stat().st_size >= minimum_bytes
    return {"name": name, "path": str(resolved), "kind": kind, "passed": passed}


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
