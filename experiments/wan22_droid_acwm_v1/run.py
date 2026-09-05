#!/usr/bin/env python3
"""Prepare, validate, and execute the WAN2.2-DROID closed loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_wan22_droid_worldarena import (
    verify as verify_worldarena,
)  # noqa: E402
from wmloop.wan22_droid import (  # noqa: E402
    Wan22DroidError,
    validate_contract,
    write_sample_manifest,
)
from wmloop.experiments.training_scale import (  # noqa: E402
    TrainingScaleError,
    build_training_scale_plan,
)
from wmloop.experiments.artifact_lint import enforce_lint  # noqa: E402
from wmloop.execute.job_supervisor import submit_job  # noqa: E402
from wmloop.experiments.job_spec import JobSpec  # noqa: E402


DEFAULT_SEEDS = (4101, 4202, 4303)
VISUAL_DIAGNOSTIC_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "photometric_smoothness",
)
FORMAL_DIMENSIONS = VISUAL_DIAGNOSTIC_DIMENSIONS + (
    "trajectory_accuracy",
    "action_following",
)
FORMAL_DIMENSIONS_WITHOUT_TRAJECTORY = VISUAL_DIAGNOSTIC_DIMENSIONS + (
    "action_following",
)


def _dump(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _estimated_gpu_hours(steps: int, seed_count: int) -> float:
    """Conservative estimate calibrated from the retained 1/8-step probes."""

    if steps < 1 or seed_count < 1:
        raise ValueError("TRAINING_ESTIMATE_INPUT_INVALID")
    return float(seed_count) * (0.04 + float(steps) * 0.002)


def _budget_snapshot(
    path: Path | None, total_gpu_hours: float
) -> dict[str, float | str | None]:
    if (
        not math.isfinite(total_gpu_hours)
        or total_gpu_hours <= 0
        or total_gpu_hours > 40
    ):
        raise ValueError("TOTAL_GPU_BUDGET_INVALID")
    consumed = 0.0
    source = None
    if path is not None:
        resolved = path.expanduser().resolve(strict=True)
        payload = _read_json(resolved)
        consumed = float(payload.get("consumed_gpu_hours", float("nan")))
        declared_cap = float(payload.get("cap_gpu_hours", float("nan")))
        if (
            not math.isfinite(consumed)
            or consumed < 0
            or declared_cap != total_gpu_hours
        ):
            raise ValueError("PRIOR_BUDGET_RECEIPT_INVALID")
        source = str(resolved)
    remaining = total_gpu_hours - consumed
    return {
        "cap_gpu_hours": total_gpu_hours,
        "consumed_before_gpu_hours": consumed,
        "remaining_before_gpu_hours": remaining,
        "prior_receipt": source,
    }


def _runner_command(
    args: argparse.Namespace, run_root: Path, seed: int, max_gpu_hours: float
) -> list[str]:
    command = [
        str(args.runtime_python.expanduser().absolute()),
        str(args.runner.expanduser().resolve()),
        "--source",
        str(args.source.expanduser().resolve()),
        "--model",
        str(args.model.expanduser().resolve()),
        "--adapter",
        str(args.adapter.expanduser().resolve()),
        "--train-manifest",
        str(args.train_manifest.expanduser().resolve()),
        "--validation-manifest",
        str(args.validation_manifest.expanduser().resolve()),
        "--evaluator-contract",
        str(args.evaluator_contract.expanduser().resolve()),
        "--output-root",
        str(run_root),
        "--gpu-index",
        "0" if args.cuda_visible_devices else str(args.gpu_index),
        "--seed",
        str(seed),
        "--conditioning-mode",
        args.conditioning_mode,
        "--history-decay",
        str(args.history_decay),
        "--anchor-policy",
        args.anchor_policy,
        "--anchor-refresh-strength",
        str(args.anchor_refresh_strength),
        "--branch-count",
        str(args.branch_count),
        "--branch-selection",
        args.branch_selection,
        "--branch-reference-weight",
        str(args.branch_reference_weight),
        "--sample-index",
        str(args.sample_index),
        "--horizon-frames",
        str(args.horizon_frames),
        "--chunk-frames",
        str(args.chunk_frames),
        "--steps",
        str(args.steps),
        "--training-stage",
        "pilot",
        "--training-mode",
        args.training_mode,
        "--training-sampler",
        args.training_sampler,
        "--train-record-limit",
        str(args.train_record_limit),
        "--rollout-steps",
        str(args.rollout_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--max-gpu-hours",
        str(max_gpu_hours),
    ]
    checkpoint_steps = getattr(args, "checkpoint_eval_steps", None)
    if checkpoint_steps:
        command.extend(
            ["--checkpoint-eval-steps", *(str(step) for step in checkpoint_steps)]
        )
    if (
        "action_following" in args.worldarena_dimensions
        and not args.visual_only_diagnostic
    ):
        command.append("--emit-branches")
    panel = getattr(args, "validation_sample_indices", None)
    if panel:
        command.extend(
            ["--validation-sample-indices", *(str(index) for index in panel)]
        )
    return command


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_seconds),
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def _metrics_receipt_name(dimensions: Sequence[str]) -> str:
    if list(dimensions) == [
        "subject_consistency",
        "background_consistency",
        "photometric_smoothness",
    ]:
        return "worldarena_metrics_receipt.json"
    return "worldarena_metrics_receipt_" + "_".join(dimensions) + ".json"


def _validate_training_receipt(
    path: Path, args: argparse.Namespace, seed: int
) -> dict[str, Any]:
    payload = _read_json(path)
    manifest_count = len(
        _read_json(args.train_manifest.expanduser().resolve())["records"]
    )
    eligible_windows = (
        manifest_count
        if args.train_record_limit == 0
        else min(manifest_count, args.train_record_limit)
    )
    expected_windows = (
        1 if args.training_mode == "probe" else min(args.steps, eligible_windows)
    )
    checks = {
        "seed": seed,
        "training_stage": "pilot",
        "training_mode": args.training_mode,
        "training_sampler": args.training_sampler,
        "steps": args.steps,
        "optimization_updates": args.steps,
        "training_windows": expected_windows,
        "conditioning_mode": args.conditioning_mode,
        "anchor_policy": args.anchor_policy,
        "adapter": str(args.adapter.expanduser().resolve()),
        "validation_panel_size": len(args.validation_sample_indices),
        "validation_sample_indices": args.validation_sample_indices,
    }
    expected_checkpoint_steps = [
        int(value) for value in (getattr(args, "checkpoint_eval_steps", None) or [])
    ]
    ladder = payload.get("checkpoint_ladder")
    if not isinstance(ladder, str) or not Path(ladder).is_file():
        mismatches = ["checkpoint_ladder=missing_or_invalid"]
    else:
        ladder_payload = _read_json(Path(ladder))
        mismatches = []
        if ladder_payload.get("state") != "completed":
            mismatches.append("checkpoint_ladder.state!=completed")
        observed_steps = [
            int(value) for value in ladder_payload.get("checkpoint_eval_steps", [])
        ]
        if expected_checkpoint_steps and observed_steps != expected_checkpoint_steps:
            mismatches.append(
                f"checkpoint_eval_steps={observed_steps!r}:expected={expected_checkpoint_steps!r}"
            )
    mismatches.extend(
        f"{key}={payload.get(key)!r}:expected={value!r}"
        for key, value in checks.items()
        if payload.get(key) != value
    )
    visualization = payload.get("paired_visualization")
    if not isinstance(visualization, str) or not Path(visualization).is_file():
        mismatches.append("paired_visualization=missing_or_invalid")
    panel_path = payload.get("validation_panel")
    if not isinstance(panel_path, str) or not Path(panel_path).is_file():
        mismatches.append("validation_panel=missing_or_invalid")
    if mismatches:
        raise ValueError("TRAINING_RECEIPT_CONTRACT_MISMATCH:" + ",".join(mismatches))
    return payload


def _next_attempt_path(base: Path) -> Path:
    if not base.exists():
        return base
    attempt = 2
    while base.with_name(f"{base.name}-attempt-{attempt}").exists():
        attempt += 1
    return base.with_name(f"{base.name}-attempt-{attempt}")


def _gpu_free_memory_mib(cuda_visible_devices: str) -> float:
    first_device = cuda_visible_devices.split(",", 1)[0].strip()
    if not first_device.isdigit():
        raise ValueError("CUDA_VISIBLE_DEVICES_INVALID")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("NVIDIA_SMI_FAILED")
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3 and fields[0] == first_device:
            return float(fields[1]) - float(fields[2])
    raise ValueError(f"CUDA_DEVICE_NOT_FOUND:{first_device}")


def _worldarena_commands(
    args: argparse.Namespace,
    run_roots: Sequence[Path],
    prepared_root: Path,
    *,
    evaluation_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    if not run_roots:
        raise ValueError("WORLDARENA_RUN_ROOTS_EMPTY")
    evaluation_root = evaluation_root or run_roots[0]
    runtime = str(args.worldarena_runtime_python.expanduser().absolute())
    prepare = [
        runtime,
        str(ROOT / "scripts" / "prepare_wan22_worldarena_input.py"),
        "--run-root",
        *(str(root) for root in run_roots),
        "--output-root",
        str(prepared_root),
        "--config-template",
        str(args.worldarena_config_template.expanduser().resolve()),
        "--worldarena-root",
        str(args.worldarena_root.expanduser().resolve()),
        "--asset-root",
        str(args.worldarena_asset_root.expanduser().resolve()),
        "--sea-raft-config",
        str(args.worldarena_sea_raft_config.expanduser().resolve()),
    ]
    if getattr(args, "worldarena_trajectory_detector_root", None) is not None:
        prepare.extend(
            [
                "--trajectory-detector-root",
                str(args.worldarena_trajectory_detector_root.expanduser().resolve()),
            ]
        )
    evaluate = [
        runtime,
        str(ROOT / "scripts" / "evaluate_wan22_worldarena.py"),
        # The evaluator writes one authoritative receipt for the panel member
        # root. ``run_roots`` may contain multiple branch roots used only for
        # GID materialization, so keep the receipt on the panel root.
        "--run-root",
        str(evaluation_root),
        "--worldarena-root",
        str(args.worldarena_root.expanduser().resolve()),
        "--config",
        str(prepared_root / "config.yaml"),
        "--prepared-root",
        str(prepared_root),
        "--runtime-python",
        runtime,
        "--dimensions",
        *args.worldarena_dimensions,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--asset-manifest",
        str(args.worldarena_asset_manifest.expanduser().resolve()),
        "--asset-root",
        str(args.worldarena_asset_root.expanduser().resolve()),
    ]
    return prepare, evaluate


def _closed_loop(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validation_payload = _read_json(args.validation_manifest.expanduser().resolve())
    validation_records = validation_payload.get("records")
    if not isinstance(validation_records, list):
        validation_records = []
    if args.validation_sample_indices is None:
        args.validation_sample_indices = (
            [args.sample_index]
            if args.training_mode == "probe"
            else [args.sample_index + offset for offset in range(3)]
        )
    validation_panel_blocker: str | None = None
    if not validation_records:
        validation_panel_blocker = "VALIDATION_PANEL_EMPTY"
    elif len(set(args.validation_sample_indices)) != len(
        args.validation_sample_indices
    ):
        validation_panel_blocker = "VALIDATION_PANEL_INDICES_DUPLICATE"
    elif any(
        index < 0 or index >= len(validation_records)
        for index in args.validation_sample_indices
    ):
        validation_panel_blocker = "VALIDATION_PANEL_INDEX_OUT_OF_RANGE"
    else:
        panel_episodes = [
            str(validation_records[index].get("episode_id", "")).strip()
            for index in args.validation_sample_indices
        ]
        if any(not episode for episode in panel_episodes):
            validation_panel_blocker = "VALIDATION_PANEL_EPISODE_ID_MISSING"
        elif len(set(panel_episodes)) != len(panel_episodes):
            validation_panel_blocker = "VALIDATION_PANEL_EPISODES_NOT_DISTINCT"
    if args.training_mode == "long" and len(args.validation_sample_indices) < 3:
        validation_panel_blocker = "FORMAL_VALIDATION_PANEL_REQUIRES_THREE_EPISODES"
    report = validate_contract(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        model=args.model,
        source=args.source,
        evaluator_contract=args.evaluator_contract,
        adapter=args.adapter,
        horizon_frames=args.horizon_frames,
    )
    blockers = list(report["blockers"])
    if not args.visual_only_diagnostic and "trajectory_accuracy" in args.worldarena_dimensions:
        data_root = Path(str(validation_payload.get("data_root", ""))).expanduser().resolve()
        trajectory_candidates = []
        for index in args.validation_sample_indices:
            if 0 <= index < len(validation_records):
                record = validation_records[index]
                episode_id = str(record.get("episode_id", "")).strip()
                annotation = Path(str(record.get("annotation_path", "")))
                trajectory_candidates.extend(
                    (
                        data_root / episode_id / "traj" / "traj.npy",
                        data_root / annotation.parent / "traj" / "traj.npy",
                    )
                )
        if not trajectory_candidates or not all(path.is_file() for path in trajectory_candidates):
            blockers.append(
                "TRAJECTORY_ACCURACY_INPUT_UNAVAILABLE:official traj/traj.npy is required before GPU training"
            )
    runtime = args.runtime_python.expanduser().absolute()
    worldarena_runtime = (
        args.worldarena_runtime_python.expanduser().absolute()
        if args.worldarena_runtime_python
        else runtime
    )
    args.worldarena_runtime_python = worldarena_runtime
    path_bindings = {
        "RUNTIME_PYTHON_INVALID": runtime,
        "WAN22_DROID_RUNNER_INVALID": args.runner,
        "WORLDARENA_RUNTIME_PYTHON_INVALID": worldarena_runtime,
        "WORLDARENA_ROOT_INVALID": args.worldarena_root,
        "WORLDARENA_CONFIG_TEMPLATE_INVALID": args.worldarena_config_template,
        "WORLDARENA_ASSET_MANIFEST_INVALID": args.worldarena_asset_manifest,
        "WORLDARENA_ASSET_ROOT_INVALID": args.worldarena_asset_root,
        "WORLDARENA_SEA_RAFT_CONFIG_INVALID": args.worldarena_sea_raft_config,
    }
    if "trajectory_accuracy" in args.worldarena_dimensions and not args.visual_only_diagnostic:
        path_bindings["WORLDARENA_TRAJECTORY_DETECTOR_ROOT_INVALID"] = getattr(
            args, "worldarena_trajectory_detector_root", None
        )
    for code, value in path_bindings.items():
        if value is None or not value.expanduser().resolve().exists():
            blockers.append(code)
    detector_root = getattr(args, "worldarena_trajectory_detector_root", None)
    if "trajectory_accuracy" in args.worldarena_dimensions and not args.visual_only_diagnostic:
        if detector_root is not None:
            detector_root = detector_root.expanduser().resolve()
            for filename in ("sam3.pt", "bpe_simple_vocab_16e6.txt.gz"):
                if not (detector_root / filename).is_file():
                    blockers.append(f"SAM3_CHECKPOINT_FILE_MISSING:{filename}")
    if not args.cuda_visible_devices:
        blockers.append("CUDA_VISIBLE_DEVICES_REQUIRED")
    else:
        try:
            free_mib = _gpu_free_memory_mib(args.cuda_visible_devices)
            if free_mib < args.min_free_gpu_memory_mib:
                blockers.append("GPU_FREE_MEMORY_BELOW_TRAINING_FLOOR")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            blockers.append("GPU_MEMORY_PROBE_FAILED")
    if not args.execute:
        blockers.append("GPU_EXECUTION_NOT_ARMED")
    if args.training_mode == "long" and args.steps < 64:
        blockers.append("LONG_TRAINING_STEPS_TOO_LOW")
    if args.training_mode == "long" and args.train_record_limit < 0:
        blockers.append("LONG_TRAINING_RECORD_LIMIT_INVALID")
    if args.training_mode == "probe" and len(args.seeds) != 1:
        blockers.append("PROBE_REQUIRES_ONE_SEED")
    if args.training_mode == "probe":
        blockers.append("CLOSED_LOOP_FORMAL_STAGE_REQUIRES_LONG_TRAINING")
    requested_dimensions = tuple(dict.fromkeys(args.worldarena_dimensions))
    if not requested_dimensions:
        blockers.append("WORLDARENA_DIMENSIONS_EMPTY")
    elif args.visual_only_diagnostic:
        if any(
            metric not in VISUAL_DIAGNOSTIC_DIMENSIONS
            for metric in requested_dimensions
        ):
            blockers.append("VISUAL_DIAGNOSTIC_METRICS_INVALID")
    elif args.omit_trajectory_accuracy:
        if set(requested_dimensions) != set(FORMAL_DIMENSIONS_WITHOUT_TRAJECTORY):
            blockers.append("FORMAL_METRIC_OMISSION_SET_INVALID")
    elif set(requested_dimensions) != set(FORMAL_DIMENSIONS):
        blockers.append("FORMAL_WORLDARENA_METRICS_REQUIRED")
    if (
        not args.visual_only_diagnostic
        and "action_following" in requested_dimensions
        and args.branch_count < 2
    ):
        blockers.append("ACTION_FOLLOWING_REQUIRES_MULTIPLE_BRANCHES")
    try:
        evaluator_payload = _read_json(args.evaluator_contract.expanduser().resolve())
        declared_metrics = tuple(
            str(value) for value in evaluator_payload.get("metrics", ())
        )
        if declared_metrics and not set(requested_dimensions).issubset(set(declared_metrics)):
            blockers.append("WORLDARENA_EVALUATOR_METRIC_CONTRACT_INVALID")
    except (OSError, ValueError, json.JSONDecodeError):
        blockers.append("WORLDARENA_EVALUATOR_METRIC_CONTRACT_INVALID")
    if args.training_mode == "long" and args.steps < 256:
        blockers.append("FORMAL_TRAINING_STEPS_BELOW_PILOT_FLOOR")
    if args.training_sampler != "episode_balanced":
        blockers.append("FORMAL_TRAINING_EPISODE_BALANCED_REQUIRED")
    if validation_panel_blocker:
        blockers.append(validation_panel_blocker)
    if len(args.seeds) < 3:
        blockers.append("FORMAL_TRAINING_REQUIRES_THREE_SEEDS")
    if len(set(args.seeds)) != len(args.seeds):
        blockers.append("SEEDS_MUST_BE_UNIQUE")

    training_scale: dict[str, Any] | None = None
    try:
        training_scale = build_training_scale_plan(
            train_manifest=args.train_manifest,
            val_manifest=args.validation_manifest,
            stage="pilot",
            requested_seed_count=len(args.seeds),
        )
        if training_scale["state"] != "ready":
            blockers.extend(str(value) for value in training_scale["blockers"])
        # The policy plan is expressed at the full pilot scale (1024..4096),
        # while a bounded execution may intentionally request fewer updates.
        # Keep only checkpoints reachable within this run and always evaluate
        # the terminal update so the bounded ladder remains valid.
        args.checkpoint_eval_steps = sorted(
            {
                int(value)
                for value in training_scale["updates"]["checkpoint_eval_steps"]
                if 1 <= int(value) <= int(args.steps)
            }
            | {int(args.steps)}
        )
    except (OSError, KeyError, TrainingScaleError, json.JSONDecodeError) as exc:
        blockers.append(f"TRAINING_SCALE_PLAN_INVALID:{type(exc).__name__}:{exc}")

    output = args.output_root.expanduser().resolve()
    receipt_path = output / "closed_loop_receipt.json"
    previous: dict[str, Any] | None = None
    resumed_training: dict[int, tuple[Path, dict[str, Any]]] = {}
    resume_spent = 0.0
    if receipt_path.exists():
        if not args.resume:
            blockers.append("OUTPUT_ROOT_ALREADY_BOUND")
        else:
            previous = _read_json(receipt_path)
            expected_resume = {
                "steps": args.steps,
                "training_mode": args.training_mode,
                "training_sampler": args.training_sampler,
                "train_record_limit": args.train_record_limit,
                "seeds": args.seeds,
                "validation_sample_indices": args.validation_sample_indices,
            }
            if any(
                previous.get(key) != value for key, value in expected_resume.items()
            ):
                blockers.append("RESUME_CONTRACT_MISMATCH")
            for seed in args.seeds:
                candidates = sorted(output.glob(f"seed-{seed}*/training_receipt.json"))
                for candidate in candidates:
                    try:
                        training = _validate_training_receipt(candidate, args, seed)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        continue
                    resumed_training[seed] = (candidate.parent, training)
                    resume_spent += float(training.get("gpu_hours", 0.0))
                    break
    remaining_training = len(args.seeds) - len(resumed_training)
    resume_eval_allowance = 0.04 * len(resumed_training)
    estimate = args.estimated_gpu_hours or (
        (
            _estimated_gpu_hours(args.steps, remaining_training)
            if remaining_training
            else 0.0
        )
        + resume_eval_allowance
    )
    budget = _budget_snapshot(args.prior_budget_receipt, args.total_gpu_hours)
    budget["resumed_run_gpu_hours"] = resume_spent
    remaining = float(budget["remaining_before_gpu_hours"]) - resume_spent
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours <= 0:
        blockers.append("RUN_GPU_BUDGET_INVALID")
    elif estimate > args.max_gpu_hours or args.max_gpu_hours > remaining:
        blockers.append("RUN_GPU_BUDGET_NOT_ADMITTED")
    if estimate > remaining:
        blockers.append("GLOBAL_GPU_BUDGET_EXHAUSTED")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or ""
    # The pinned runtime may be an external virtualenv that does not install
    # VerdiWM as a wheel. Bind the repository root explicitly so the external
    # runner can import the model-agnostic control plane modules.
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if training_scale is not None:
        scale_digest = hashlib.sha256(
            json.dumps(
                training_scale, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        env.update(
            {
                "VERDIWM_TRAINING_CONTRACT": "VERDIWM_TRAINING_CONTRACT_V1",
                "VERDIWM_TRAINING_STAGE": "pilot",
                "VERDIWM_TRAINING_MODE": "long",
                "VERDIWM_TRAINING_STEPS": str(args.steps),
                "VERDIWM_TRAINING_RECORD_LIMIT": str(args.train_record_limit),
                "VERDIWM_TRAINING_SAMPLER": args.training_sampler,
                "VERDIWM_TRAINING_SEED_COUNT": str(len(args.seeds)),
                "VERDIWM_TRAINING_SCALE_PLAN_SHA256": scale_digest,
                "VERDIWM_VALIDATION_PANEL_SIZE": str(
                    len(args.validation_sample_indices)
                ),
            }
        )
    if runtime.is_file() and args.execute:
        try:
            probe = subprocess.run(
                [str(runtime), "-c", "import torch; assert torch.cuda.is_available()"],
                capture_output=True,
                env=env,
                text=True,
                # Torch's first import can exceed 30s on the network filesystem.
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        if probe is None or probe.returncode != 0:
            blockers.append("RUNTIME_TORCH_CUDA_UNAVAILABLE")

    candidate_binding = {
        "conditioning_mode": args.conditioning_mode,
        "history_decay": args.history_decay,
        "anchor_policy": args.anchor_policy,
        "anchor_refresh_strength": args.anchor_refresh_strength,
        "branch_count": args.branch_count,
        "branch_selection": args.branch_selection,
    }
    if previous is not None and previous.get("candidate") != candidate_binding:
        blockers.append("RESUME_CANDIDATE_MISMATCH")
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "verdiwm-wan22-droid-closed-loop-receipt",
        "state": "blocked" if blockers else "admitted",
        "conformance": report,
        "budget": {
            **budget,
            "estimated_gpu_hours": estimate,
            "run_cap_gpu_hours": args.max_gpu_hours,
        },
        "runtime_python": str(runtime),
        "worldarena_runtime_python": str(worldarena_runtime),
        "cuda_visible_devices": args.cuda_visible_devices,
        "worldarena_dimensions": list(args.worldarena_dimensions),
        "worldarena_metrics_omitted": [
            metric for metric in FORMAL_DIMENSIONS if metric not in args.worldarena_dimensions
        ],
        "visual_only_diagnostic": bool(args.visual_only_diagnostic),
        "runner": str(args.runner.expanduser().resolve()) if args.runner else None,
        "adapter": str(args.adapter.expanduser().resolve()),
        "output_root": str(output),
        "stages": [
            "train",
            "rollout_150f",
            "worldarena_frozen_eval",
            "multi_seed_verification",
            "promotion_decision",
        ],
        "candidate": candidate_binding,
        "training_mode": args.training_mode,
        "training_sampler": args.training_sampler,
        "train_record_limit": args.train_record_limit,
        "steps": args.steps,
        "seeds": args.seeds,
        "validation_sample_indices": args.validation_sample_indices,
        "training_scale": training_scale,
        "training_scale_execution": {
            "requested_steps": args.steps,
            "requested_record_limit": args.train_record_limit,
            "requested_seed_count": len(args.seeds),
            "execution_is_bounded_pilot_cap": True,
            "claim_boundary": (
                "The requested execution cap is reported separately from the scale "
                "policy target; it must not be described as full-dataset training."
            ),
        },
        "runs": [],
        "previous_attempts": previous.get("runs", []) if previous else [],
        "blockers": sorted(set(blockers)),
        "claim_boundary": "Completion proves real optimization, 150-frame rollout, and the declared frozen metrics. Promotion still requires a separately frozen quality threshold and metrics that are valid for this data layout.",
    }
    _write_json(receipt_path, receipt)
    if blockers:
        return receipt, 2

    receipt["state"] = "running"
    receipt["started_unix_seconds"] = time.time()
    _write_json(receipt_path, receipt)
    started = time.monotonic()
    metrics_receipts: list[Path] = []
    failure: dict[str, Any] | None = None
    for seed in args.seeds:
        resumed = resumed_training.get(seed)
        run_root = (
            resumed[0] if resumed else _next_attempt_path(output / f"seed-{seed}")
        )
        if resumed is None:
            run_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        elapsed_hours = (time.monotonic() - started) / 3600.0
        available_hours = min(
            args.max_gpu_hours - resume_spent - elapsed_hours, remaining - elapsed_hours
        )
        row: dict[str, Any] = {
            "seed": seed,
            "state": "running",
            "run_root": str(run_root),
            "resumed_training": resumed is not None,
        }
        receipt["runs"].append(row)
        _write_json(receipt_path, receipt)
        try:
            if available_hours <= 0:
                raise RuntimeError("RUN_GPU_BUDGET_EXHAUSTED")
            if resumed is None:
                free_mib = _gpu_free_memory_mib(args.cuda_visible_devices)
                row["free_gpu_memory_before_training_mib"] = free_mib
                if free_mib < args.min_free_gpu_memory_mib:
                    raise RuntimeError(
                        f"GPU_FREE_MEMORY_BELOW_TRAINING_FLOOR:{free_mib}"
                    )
                runner_command = _runner_command(args, run_root, seed, available_hours)
                row["runner_command"] = runner_command
                completed = _run_logged(
                    runner_command,
                    cwd=args.runner.expanduser().resolve().parent,
                    env=env,
                    stdout_path=run_root / "runner.stdout.log",
                    stderr_path=run_root / "runner.stderr.log",
                    timeout_seconds=available_hours * 3600.0,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"RUNNER_FAILED:{completed.returncode}")
                training = _validate_training_receipt(
                    run_root / "training_receipt.json", args, seed
                )
            else:
                training = resumed[1]
            row["training_receipt"] = str(run_root / "training_receipt.json")
            row["optimization_updates"] = training["optimization_updates"]
            row["training_windows"] = training["training_windows"]
            row["training_episodes"] = training["training_episodes"]
            row["training_chunk_offsets"] = training["training_chunk_offsets"]
            row["paired_visualization"] = training["paired_visualization"]

            panel_manifest = _read_json(run_root / "validation_panel.json")
            panel_rows = panel_manifest.get("rows")
            if panel_manifest.get("state") != "frozen" or not isinstance(
                panel_rows, list
            ):
                raise ValueError("VALIDATION_PANEL_RECEIPT_INVALID")
            if [
                int(item.get("sample_index", -1))
                for item in panel_rows
                if isinstance(item, dict)
            ] != list(args.validation_sample_indices):
                raise ValueError("VALIDATION_PANEL_RECEIPT_MISMATCH")
            panel_metrics: list[dict[str, Any]] = []
            for panel_item in panel_rows:
                if not isinstance(panel_item, dict):
                    raise ValueError("VALIDATION_PANEL_ROW_INVALID")
                sample_index = int(panel_item["sample_index"])
                panel_run_root = (
                    Path(str(panel_item["run_root"])).expanduser().resolve(strict=True)
                )
                panel_roots = [panel_run_root]
                if (
                    not args.visual_only_diagnostic
                    and "action_following" in args.worldarena_dimensions
                ):
                    branch_rows = panel_item.get("branch_roots", [])
                    if not isinstance(branch_rows, list) or len(branch_rows) < 2:
                        raise ValueError(
                            f"ACTION_FOLLOWING_BRANCH_OUTPUTS_MISSING:panel={sample_index}"
                        )
                    panel_roots = [
                        Path(str(row["root"])).expanduser().resolve(strict=True)
                        for row in branch_rows
                        if isinstance(row, dict) and row.get("root")
                    ]
                    if len(panel_roots) < 2:
                        raise ValueError(
                            f"ACTION_FOLLOWING_BRANCH_OUTPUTS_MISSING:panel={sample_index}"
                        )
                prepared_root = _next_attempt_path(
                    output / "worldarena" / f"seed-{seed}" / f"panel-{sample_index}"
                )
                free_mib = _gpu_free_memory_mib(args.cuda_visible_devices)
                row.setdefault("free_gpu_memory_before_evaluation_mib", free_mib)
                if free_mib < args.min_free_evaluator_gpu_memory_mib:
                    raise RuntimeError(
                        f"GPU_FREE_MEMORY_BELOW_EVALUATOR_FLOOR:{free_mib}"
                    )
                prepare_command, evaluate_command = _worldarena_commands(
                    args,
                    panel_roots,
                    prepared_root,
                    evaluation_root=panel_run_root,
                )
                prepared_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                prepared = _run_logged(
                    prepare_command,
                    cwd=ROOT,
                    env=env,
                    stdout_path=panel_run_root / "worldarena_prepare.stdout.log",
                    stderr_path=panel_run_root / "worldarena_prepare.stderr.log",
                    timeout_seconds=max(
                        1.0,
                        (args.max_gpu_hours - (time.monotonic() - started) / 3600.0)
                        * 3600.0,
                    ),
                )
                if prepared.returncode != 0:
                    raise RuntimeError(
                        f"WORLDARENA_PREPARE_FAILED:{prepared.returncode}:panel={sample_index}"
                    )
                evaluated = _run_logged(
                    evaluate_command,
                    cwd=ROOT,
                    env=env,
                    stdout_path=panel_run_root / "worldarena_adapter.stdout.log",
                    stderr_path=panel_run_root / "worldarena_adapter.stderr.log",
                    timeout_seconds=max(
                        1.0,
                        (args.max_gpu_hours - (time.monotonic() - started) / 3600.0)
                        * 3600.0,
                    ),
                )
                if evaluated.returncode != 0:
                    raise RuntimeError(
                        f"WORLDARENA_EVALUATION_FAILED:{evaluated.returncode}:panel={sample_index}"
                    )
                metrics_path = panel_run_root / _metrics_receipt_name(
                    args.worldarena_dimensions
                )
                metrics = _read_json(metrics_path)
                if (
                    metrics.get("state") != "evaluated_partial"
                    or metrics.get("returncode") != 0
                ):
                    raise ValueError(f"WORLDARENA_RECEIPT_INVALID:panel={sample_index}")
                metrics_receipts.append(metrics_path)
                panel_metrics.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": panel_item.get("sample_id"),
                        "episode_id": panel_item.get("episode_id"),
                        "metrics_receipt": str(metrics_path),
                        "metrics": metrics.get("metrics", {}),
                    }
                )
            row.update(
                {
                    "state": "evaluated",
                    "metrics_receipt": panel_metrics[0]["metrics_receipt"],
                    "metrics": panel_metrics[0]["metrics"],
                    "validation_panel_metrics": panel_metrics,
                }
            )
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            row["state"] = "failed"
            row["failure_signature"] = f"{type(exc).__name__}:{exc}"
            failure = {"seed": seed, "signature": row["failure_signature"]}
            break
        finally:
            row["elapsed_gpu_hours_upper_bound"] = (time.monotonic() - started) / 3600.0
            _write_json(receipt_path, receipt)

    actual_hours = resume_spent + (time.monotonic() - started) / 3600.0
    receipt["budget"]["actual_gpu_hours_upper_bound"] = actual_hours
    receipt["budget"]["remaining_after_upper_bound_gpu_hours"] = (
        float(budget["remaining_before_gpu_hours"]) - actual_hours
    )
    if failure is not None:
        receipt.update(
            {
                "state": "failed",
                "failure": failure,
                "next_state": "capability_gap_triage",
            }
        )
        _write_json(receipt_path, receipt)
        return receipt, 2

    verification = verify_worldarena(
        metrics_receipts,
        required_metrics=tuple(args.worldarena_dimensions),
        expected_seeds=tuple(args.seeds),
        expected_panel_size=len(args.validation_sample_indices),
        expected_panel_episode_count=(
            len(args.validation_sample_indices)
            if not args.visual_only_diagnostic
            else 1
        ),
    )
    verification_path = output / "worldarena_verification_receipt.json"
    _write_json(verification_path, verification)
    # Keep stable top-level evidence names for downstream archive/index tools;
    # per-panel metric receipts remain the authoritative detailed records.
    if metrics_receipts:
        first_raw_result = metrics_receipts[0].with_name(
            metrics_receipts[0].name.replace(
                "worldarena_metrics_receipt", "worldarena_result", 1
            )
        )
        if first_raw_result.is_file():
            (output / "worldarena_result.json").write_bytes(
                first_raw_result.read_bytes()
            )
    frozen_verifier_path = output / "frozen_verifier_receipt.json"
    _write_json(frozen_verifier_path, verification)
    receipt["verification_receipt"] = str(verification_path)
    receipt["frozen_verifier_receipt"] = str(frozen_verifier_path)
    artifact_lint = enforce_lint(output)
    artifact_lint_path = output / "artifact_lint.json"
    _write_json(artifact_lint_path, artifact_lint)
    receipt["artifact_lint"] = str(artifact_lint_path)
    if verification["state"] != "verified":
        receipt.update(
            {
                "state": "failed",
                "blockers": verification["blockers"],
                "next_state": "capability_gap_triage",
            }
        )
        code = 2
    elif not args.visual_only_diagnostic and artifact_lint["state"] != "pass":
        receipt.update(
            {
                "state": "failed",
                "blockers": ["ARTIFACT_LINT_BLOCKED"],
                "artifact_lint_errors": artifact_lint["error_count"],
                "next_state": "artifact_repair",
            }
        )
        code = 2
    elif args.visual_only_diagnostic:
        receipt.update(
            {
                "state": "diagnostic_only",
                "promotion_state": "not_quality_eligible",
                "next_state": "formal_metric_completion",
                "claim_boundary": (
                    "This visual-only diagnostic proves execution and visual metrics only; "
                    "it cannot establish trajectory accuracy, action following, or promotion."
                ),
            }
        )
        code = 0
    else:
        receipt.update(
            {
                "state": "completed",
                "promotion_state": "quality_threshold_unbound",
                "next_state": "mechanism_discovery",
            }
        )
        code = 0
    receipt["completed_unix_seconds"] = time.time()
    _write_json(receipt_path, receipt)
    return receipt, code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--horizon-frames", type=int, default=150)
    prepare.add_argument("--stride", type=int, default=30)
    conformance = commands.add_parser("conformance")
    conformance.add_argument("--train-manifest", type=Path, required=True)
    conformance.add_argument("--validation-manifest", type=Path, required=True)
    conformance.add_argument("--model", type=Path, required=True)
    conformance.add_argument("--source", type=Path, required=True)
    conformance.add_argument("--evaluator-contract", type=Path, required=True)
    conformance.add_argument("--adapter", type=Path, required=True)
    conformance.add_argument("--horizon-frames", type=int, default=150)

    closed = commands.add_parser(
        "closed-loop", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    closed.add_argument("--train-manifest", type=Path, required=True)
    closed.add_argument("--validation-manifest", type=Path, required=True)
    closed.add_argument("--model", type=Path, required=True)
    closed.add_argument("--source", type=Path, required=True)
    closed.add_argument("--adapter", type=Path, required=True)
    closed.add_argument("--evaluator-contract", type=Path, required=True)
    closed.add_argument("--runtime-python", type=Path, required=True)
    closed.add_argument("--runner", type=Path, required=True)
    closed.add_argument("--output-root", type=Path, required=True)
    closed.add_argument("--cuda-visible-devices", required=True)
    closed.add_argument("--gpu-index", type=int, default=0)
    closed.add_argument("--min-free-gpu-memory-mib", type=float, default=22000.0)
    closed.add_argument(
        "--min-free-evaluator-gpu-memory-mib", type=float, default=8000.0
    )
    closed.add_argument(
        "--execute", action="store_true", help="allow the bound runner to use GPU"
    )
    closed.add_argument(
        "--background",
        action="store_true",
        help="submit the closed loop as a detached job and return immediately",
    )
    closed.add_argument(
        "--background-job-root",
        type=Path,
        help="job state directory; defaults to a sibling of output-root",
    )
    closed.add_argument(
        "--visual-only-diagnostic",
        action="store_true",
        help="run only the four visual metrics; result is never quality-eligible",
    )
    closed.add_argument(
        "--resume",
        action="store_true",
        help="reuse matching completed seed training receipts and preserve failed attempts",
    )
    closed.add_argument("--horizon-frames", type=int, default=150)
    closed.add_argument("--chunk-frames", type=int, default=45)
    closed.add_argument("--training-mode", choices=("probe", "long"), default="long")
    closed.add_argument(
        "--training-sampler",
        choices=("sequential", "episode_balanced"),
        default="episode_balanced",
    )
    closed.add_argument(
        "--train-record-limit",
        type=int,
        default=256,
        help="bounded long-training windows; 0 makes the full manifest eligible within the global step count",
    )
    closed.add_argument(
        "--steps",
        type=int,
        default=512,
        help="global optimization updates, not updates per window",
    )
    closed.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    closed.add_argument("--sample-index", type=int, default=0)
    closed.add_argument(
        "--validation-sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="frozen validation panel indices; formal runs require three distinct episodes",
    )
    closed.add_argument(
        "--conditioning-mode",
        choices=(
            "visual_anchor_only",
            "action",
            "action_proprio",
            "action_proprio_history",
            "action_proprio_ema",
        ),
        default="action_proprio_ema",
    )
    closed.add_argument("--history-decay", type=float, default=0.8)
    closed.add_argument(
        "--anchor-policy",
        choices=("previous_generated", "initial_reference_blend"),
        default="initial_reference_blend",
    )
    closed.add_argument("--anchor-refresh-strength", type=float, default=0.25)
    closed.add_argument("--branch-count", type=int, default=2)
    closed.add_argument(
        "--branch-selection",
        choices=("first", "terminal_reference_consistency"),
        default="terminal_reference_consistency",
    )
    closed.add_argument("--branch-reference-weight", type=float, default=0.7)
    closed.add_argument("--rollout-steps", type=int, default=2)
    closed.add_argument("--learning-rate", type=float, default=1e-4)
    closed.add_argument("--total-gpu-hours", type=float, default=40.0)
    closed.add_argument(
        "--max-gpu-hours",
        type=float,
        default=15.0,
        help="hard wall-clock GPU-hour upper bound for this serial batch",
    )
    closed.add_argument("--estimated-gpu-hours", type=float, default=None)
    closed.add_argument("--prior-budget-receipt", type=Path)
    closed.add_argument("--worldarena-root", type=Path, required=True)
    closed.add_argument("--worldarena-config-template", type=Path, required=True)
    closed.add_argument("--worldarena-runtime-python", type=Path)
    closed.add_argument("--worldarena-asset-manifest", type=Path, required=True)
    closed.add_argument("--worldarena-asset-root", type=Path, required=True)
    closed.add_argument(
        "--worldarena-trajectory-detector-root",
        type=Path,
        help="SAM3 checkpoint directory used by the official trajectory extractor",
    )
    closed.add_argument(
        "--worldarena-sea-raft-config",
        type=Path,
        default=ROOT / "configs" / "evaluators" / "sea_raft_192x320.json",
    )
    closed.add_argument(
        "--worldarena-dimensions", nargs="+", default=list(FORMAL_DIMENSIONS)
    )
    closed.add_argument(
        "--omit-trajectory-accuracy",
        action="store_true",
        help="formal five-metric run; omit the SAM3-dependent trajectory metric",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if getattr(args, "omit_trajectory_accuracy", False):
        args.worldarena_dimensions = [
            metric
            for metric in args.worldarena_dimensions
            if metric != "trajectory_accuracy"
        ]
    try:
        if args.command == "prepare":
            output = args.output_root.expanduser().resolve()
            train = write_sample_manifest(
                args.data_root,
                "train",
                output / "train.json",
                horizon_frames=args.horizon_frames,
                stride=args.stride,
            )
            val = write_sample_manifest(
                args.data_root,
                "val",
                output / "val.json",
                horizon_frames=args.horizon_frames,
                stride=args.stride,
            )
            _dump(
                {
                    "state": "ready",
                    "train": train,
                    "validation": val,
                    "output_root": str(output),
                }
            )
            return 0
        if args.command == "closed-loop":
            if args.background:
                if not args.execute:
                    raise ValueError("BACKGROUND_CLOSED_LOOP_REQUIRES_EXECUTE")
                job_root = (
                    args.background_job_root.expanduser().resolve()
                    if args.background_job_root is not None
                    else args.output_root.expanduser().resolve().with_name(
                        args.output_root.expanduser().resolve().name + ".job"
                    )
                )
                child_argv = [item for item in raw_argv if item != "--background"]
                if "--background-job-root" in child_argv:
                    index = child_argv.index("--background-job-root")
                    del child_argv[index : index + 2]
                result = submit_job(
                    JobSpec(
                        command=(sys.executable, str(Path(__file__).resolve()), *child_argv),
                        cwd=ROOT,
                        job_root=job_root,
                        output_root=args.output_root,
                        metadata={
                            "kind": "wan22_droid_closed_loop",
                            "output_root": str(args.output_root.expanduser().resolve()),
                            "training_mode": args.training_mode,
                            "steps": args.steps,
                            "seeds": args.seeds,
                        },
                    )
                )
                _dump(result)
                return 0
            receipt, code = _closed_loop(args)
            _dump(receipt)
            return code
        report = validate_contract(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            model=args.model,
            source=args.source,
            evaluator_contract=args.evaluator_contract,
            adapter=args.adapter,
            horizon_frames=args.horizon_frames,
        )
        _dump(report)
        return 0 if report["state"] == "ready_for_execution" else 2
    except (OSError, Wan22DroidError, ValueError, json.JSONDecodeError) as exc:
        _dump({"state": "blocked", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
