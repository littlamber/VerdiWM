#!/usr/bin/env python3
"""Build a dependency-aware ACWM exploration queue.

The queue is intentionally execution-facing: every candidate primitive gets a
512-step screen row, an official-protocol quality gate, and a dependent staged
confirmation through 1k. Every retained confirmation checkpoint is evaluated
independently before a finalizer selects the best passing checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wmloop.execute.acwm_primitive_routes import (
    RUNTIME_ONLY_PRIMITIVES,
    TRAINING_QUALITY_SCREEN_PRIMITIVES,
)
from wmloop.execute.training_monitor_policy import (
    DEFAULT_CONFIRMATION_STEPS,
    checkpoint_eval_ladder,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAP_PLAN = ROOT / "results/reports/acwm-gap-driven-staging-plan-r1/acwm-gap-driven-staging-plan.json"
DEFAULT_MATERIALIZATION_GATE = ROOT / "results/reports/primitive-materialization-gate-current-r1/manifest.json"
DEFAULT_OUT = ROOT / "results/reports/acwm-autoloop-queue-r1"
DEFAULT_LIMITED_GATE = ROOT / "results/reports/limited-campaign-gate-8env-official-current-warning-r1/manifest.json"
DEFAULT_FAILURE_MANIFEST = ROOT / "results/reports/m1-raw-failure-reports-ladder-r1/manifest.json"
DEFAULT_GOAL = ROOT / "configs/goal/g1_long_horizon_ladder_v1.yaml"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))
DEFAULT_DATASET_FREEZE = ROOT / "runs/m0/protocol/dataset-freeze.json"
DEFAULT_HELDOUT_PROTOCOL = ROOT / "runs/m0/protocol/heldout-protocol.json"
DEFAULT_ARCHIVE_DB = ROOT / "results/archive.db"
DEFAULT_CAS_ROOT = ROOT / "results"
DEFAULT_MECHANISM_CARDS = ROOT / "results/reports/primitive-mechanism-cards-r1/mechanism_cards.csv"

_HORIZONS_BY_ENVIRONMENT = {
    "push_cube": (16, 32, 48, 64),
    "stack_cube": (16, 32, 48),
    "push_rope": (16, 32, 48),
    "cloth_move": (16, 32, 48),
    "push_sand": (16, 32, 48),
    # The physical event starts around frame 100. Across the frozen 50-trajectory
    # test split, some pours do not reach 90% completion until frame 218, and
    # the remaining tail is needed to expose leakage and long-term drift.
    # Horizon 297 is the largest aligned rollout supported by 300-frame videos.
    "pour_water": (32, 64, 96, 128, 160, 200, 240, 297),
    "robot_arm": (16, 32, 48),
    "reacher": (16, 32),
}

# Search-only horizons may cover a late physical event without changing the
# frozen goal protocol. Formal verdicts remain owned by the official gate and
# the post-confirmation full-horizon evidence chain.
_DIAGNOSTIC_SCREEN_HORIZONS_BY_ENVIRONMENT = {
    "pour_water": (64, 96, 128, 160),
}

# The replicated pour-water semantic gate requires four independent full-event
# trajectories. Keep rollout retention aligned with that downstream contract so
# a valid candidate cannot become permanently unconfirmable after generation.
_LONG_HORIZON_TRAJECTORIES_BY_ENVIRONMENT = {
    "pour_water": 4,
}


class AcwmAutoloopQueueError(RuntimeError):
    """Autoloop queue construction failed closed."""


def build_autoloop_queue(
    *,
    gap_plan: Path,
    materialization_gate: Path,
    output_root: Path,
    repo_root: Path = ROOT,
    report_root: Path = ROOT / "results/reports",
    limited_gate: Path = DEFAULT_LIMITED_GATE,
    failure_manifest: Path = DEFAULT_FAILURE_MANIFEST,
    goal_config: Path = DEFAULT_GOAL,
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    dataset_freeze: Path = DEFAULT_DATASET_FREEZE,
    heldout_protocol: Path = DEFAULT_HELDOUT_PROTOCOL,
    archive_db: Path = DEFAULT_ARCHIVE_DB,
    cas_root: Path = DEFAULT_CAS_ROOT,
    gpus: Sequence[int] = (0, 1, 2),
    seeds: Sequence[int] = (801,),
    screen_steps: int = 512,
    confirmation_steps: int = DEFAULT_CONFIRMATION_STEPS,
    train_batch_size: int = 8,
    allow_repeat_cells: bool = False,
    include_discovered_positive_screens: bool = True,
) -> dict[str, object]:
    if screen_steps < 1 or confirmation_steps <= screen_steps:
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_STEP_POLICY_INVALID")
    if train_batch_size < 1:
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_BATCH_SIZE_INVALID")
    if not gpus:
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_GPUS_EMPTY")
    if not seeds:
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_SEEDS_EMPTY")

    gap = _load_json_object(gap_plan)
    records = gap.get("environment_records")
    if not isinstance(records, list):
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_GAP_RECORDS_INVALID")
    materialization = _load_json_object(materialization_gate)
    runtime_ready_primitives = _runtime_ready_primitives(materialization)
    ready_primitives = _ready_primitives(materialization)
    destination = Path(output_root).resolve()
    report_root = Path(report_root).resolve()
    existing_cells = {} if allow_repeat_cells else _discover_existing_cells(report_root)
    existing_confirm_cells = {} if allow_repeat_cells else _discover_existing_confirmation_cells(report_root, confirmation_steps)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    rank = 1

    for record in records:
        if not isinstance(record, Mapping):
            continue
        environment = str(record.get("environment") or "")
        if not environment or record.get("evidence_level") == "repeated_positive":
            continue
        for primitive in record.get("recommended_existing_primitives", []) or []:
            primitive = str(primitive)
            if primitive not in ready_primitives:
                skipped.append(
                    {
                        "environment": environment,
                        "primitive": primitive,
                        "reason": (
                            "primitive_runtime_only_requires_runtime_screen"
                            if primitive in runtime_ready_primitives and primitive in RUNTIME_ONLY_PRIMITIVES
                            else "primitive_not_quality_screen_routable"
                            if primitive in runtime_ready_primitives
                            else "primitive_not_closed_loop_ready"
                        ),
                    }
                )
                continue
            primitive_parameters = _record_primitive_parameters(record, primitive)
            existing = existing_cells.get((environment, primitive))
            if existing:
                skipped.append(
                    {
                        "environment": environment,
                        "primitive": primitive,
                        "reason": "cell_already_has_ready_or_running_screen",
                        "existing_campaigns": ";".join(existing),
                    }
                )
                continue
            for seed in seeds:
                screen = _queue_row(
                    rank=rank,
                    phase="screen_512",
                    environment=environment,
                    primitive=primitive,
                    seed=int(seed),
                    train_steps=screen_steps,
                    train_batch_size=train_batch_size,
                    campaign_id=f"acwm-autoloop-screen-{environment}-{primitive}-s{seed}-t{screen_steps}-r1",
                    report_root=report_root,
                    repo_root=repo_root,
                    limited_gate=limited_gate,
                    failure_manifest=failure_manifest,
                    goal_config=goal_config,
                    runtime_python=runtime_python,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    dataset_freeze=dataset_freeze,
                    heldout_protocol=heldout_protocol,
                    archive_db=archive_db,
                    cas_root=cas_root,
                    gpus=gpus,
                    primitive_parameters=primitive_parameters,
                    proposal_routing_plan=gap_plan,
                )
                rows.append(screen)
                rank += 1
                official_gate = _official_gate_row(
                    rank=rank,
                    environment=environment,
                    primitive=primitive,
                    seed=int(seed),
                    checkpoint_step=screen_steps,
                    screen_output_root=Path(str(screen["output_root"])),
                    report_root=report_root,
                    repo_root=repo_root,
                    runtime_python=runtime_python,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    gpus=gpus,
                    requires_positive_manifest=str(Path(str(screen["output_root"])) / "envs" / environment / "manifest.json"),
                )
                rows.append(official_gate)
                rank += 1
                confirm = _queue_row(
                    rank=rank,
                    phase="confirm_staged",
                    environment=environment,
                    primitive=primitive,
                    seed=int(seed),
                    train_steps=confirmation_steps,
                    train_batch_size=train_batch_size,
                    campaign_id=f"acwm-autoloop-confirm-{environment}-{primitive}-s{seed}-t{confirmation_steps}-r1",
                    report_root=report_root,
                    repo_root=repo_root,
                    limited_gate=limited_gate,
                    failure_manifest=failure_manifest,
                    goal_config=goal_config,
                    runtime_python=runtime_python,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    dataset_freeze=dataset_freeze,
                    heldout_protocol=heldout_protocol,
                    archive_db=archive_db,
                    cas_root=cas_root,
                    gpus=gpus,
                    requires_official_quality_manifest=str(Path(str(official_gate["output_root"])) / "manifest.json"),
                    requires_positive_output_root=str(official_gate["output_root"]),
                    primitive_parameters=primitive_parameters,
                    proposal_routing_plan=gap_plan,
                )
                rows.append(confirm)
                rank += 1
                confirmation_gates: list[dict[str, object]] = [official_gate]
                for checkpoint_step in checkpoint_eval_ladder(confirmation_steps):
                    if checkpoint_step == screen_steps:
                        continue
                    gate_row = _confirmation_official_gate_row(
                        rank=rank,
                        environment=environment,
                        primitive=primitive,
                        seed=int(seed),
                        checkpoint_step=checkpoint_step,
                        confirmation_output_root=Path(str(confirm["output_root"])),
                        report_root=report_root,
                        repo_root=repo_root,
                        runtime_python=runtime_python,
                        data_root=data_root,
                        checkpoint_root=checkpoint_root,
                        gpus=gpus,
                        requires_ready_manifest=str(
                            Path(str(confirm["output_root"])) / "envs" / environment / "manifest.json"
                        ),
                    )
                    rows.append(gate_row)
                    confirmation_gates.append(gate_row)
                    rank += 1
                finalizer = _checkpoint_finalizer_row(
                        rank=rank,
                        environment=environment,
                        primitive=primitive,
                        seed=int(seed),
                        confirmation_output_root=Path(str(confirm["output_root"])),
                        official_gate_rows=confirmation_gates,
                        supplemental_checkpoints={
                            screen_steps: Path(str(screen["output_root"]))
                            / "envs"
                            / environment
                            / "retained_training"
                            / "latest.pt"
                        },
                        report_root=report_root,
                        repo_root=repo_root,
                        gpus=gpus,
                    )
                rows.append(finalizer)
                rank += 1
                post_rows = _post_finalizer_rows(
                    start_rank=rank,
                    environment=environment,
                    primitive=primitive,
                    seed=int(seed),
                    finalizer_row=finalizer,
                    report_root=report_root,
                    repo_root=repo_root,
                    runtime_python=runtime_python,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    failure_manifest=failure_manifest,
                    gpus=gpus,
                )
                rows.extend(post_rows)
                rank += len(post_rows)

    queued_dependencies = {
        str(row.get("source_screen_manifest") or "")
        for row in rows
        if row.get("phase") == "official_eval_gate" and row.get("source_screen_manifest")
    }
    discovered_positive_screens = (
        _discover_positive_screens(
            report_root,
            min_steps=screen_steps,
            confirmation_steps=confirmation_steps,
        )
        if include_discovered_positive_screens
        else []
    )
    for screen in discovered_positive_screens:
        environment = str(screen["environment"])
        primitive = str(screen["primitive"])
        dependency = str(screen["manifest_path"])
        if (environment, primitive) in existing_confirm_cells:
            skipped.append(
                {
                    "environment": environment,
                    "primitive": primitive,
                    "reason": "positive_screen_already_has_confirmation",
                    "existing_campaigns": ";".join(existing_confirm_cells[(environment, primitive)]),
                }
            )
            continue
        if dependency in queued_dependencies:
            continue
        seed = int(screen["seed"])
        campaign_id = f"acwm-autoloop-confirm-{environment}-{primitive}-s{seed}-t{confirmation_steps}-r1"
        output_path = report_root / campaign_id
        if output_path.exists() or output_path.is_symlink():
            skipped.append(
                {
                    "environment": environment,
                    "primitive": primitive,
                    "reason": "positive_screen_confirmation_output_exists",
                    "existing_campaigns": campaign_id,
                }
            )
            continue
        official_gate = _official_gate_row(
            rank=rank,
            environment=environment,
            primitive=primitive,
            seed=seed,
            checkpoint_step=screen_steps,
            screen_output_root=Path(str(screen["output_root"])),
            report_root=report_root,
            repo_root=repo_root,
            runtime_python=runtime_python,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            gpus=gpus,
            requires_positive_manifest=dependency,
        )
        rows.append(official_gate)
        rank += 1
        confirm = _queue_row(
                rank=rank,
                phase="confirm_staged",
                environment=environment,
                primitive=primitive,
                seed=seed,
                train_steps=confirmation_steps,
                train_batch_size=train_batch_size,
                campaign_id=campaign_id,
                report_root=report_root,
                repo_root=repo_root,
                limited_gate=limited_gate,
                failure_manifest=failure_manifest,
                goal_config=goal_config,
                runtime_python=runtime_python,
                data_root=data_root,
                checkpoint_root=checkpoint_root,
                dataset_freeze=dataset_freeze,
                heldout_protocol=heldout_protocol,
                archive_db=archive_db,
                cas_root=cas_root,
                gpus=gpus,
                requires_official_quality_manifest=str(Path(str(official_gate["output_root"])) / "manifest.json"),
                requires_positive_output_root=str(official_gate["output_root"]),
                proposal_routing_plan=gap_plan,
            )
        rows.append(confirm)
        queued_dependencies.add(dependency)
        rank += 1
        confirmation_gates = [official_gate]
        for checkpoint_step in checkpoint_eval_ladder(confirmation_steps):
            if checkpoint_step == screen_steps:
                continue
            gate_row = _confirmation_official_gate_row(
                rank=rank,
                environment=environment,
                primitive=primitive,
                seed=seed,
                checkpoint_step=checkpoint_step,
                confirmation_output_root=Path(str(confirm["output_root"])),
                report_root=report_root,
                repo_root=repo_root,
                runtime_python=runtime_python,
                data_root=data_root,
                checkpoint_root=checkpoint_root,
                gpus=gpus,
                requires_ready_manifest=str(
                    Path(str(confirm["output_root"])) / "envs" / environment / "manifest.json"
                ),
            )
            rows.append(gate_row)
            confirmation_gates.append(gate_row)
            rank += 1
        finalizer = _checkpoint_finalizer_row(
                rank=rank,
                environment=environment,
                primitive=primitive,
                seed=seed,
                confirmation_output_root=Path(str(confirm["output_root"])),
                official_gate_rows=confirmation_gates,
                supplemental_checkpoints={
                    screen_steps: Path(str(screen["output_root"]))
                    / "envs"
                    / environment
                    / "retained_training"
                    / "latest.pt"
                },
                report_root=report_root,
                repo_root=repo_root,
                gpus=gpus,
            )
        rows.append(finalizer)
        rank += 1
        post_rows = _post_finalizer_rows(
            start_rank=rank,
            environment=environment,
            primitive=primitive,
            seed=seed,
            finalizer_row=finalizer,
            report_root=report_root,
            repo_root=repo_root,
            runtime_python=runtime_python,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            failure_manifest=failure_manifest,
            gpus=gpus,
        )
        rows.extend(post_rows)
        rank += len(post_rows)

    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "gap_plan": str(Path(gap_plan).resolve()),
        "materialization_gate": str(Path(materialization_gate).resolve()),
        "closed_loop_ready_primitives": sorted(runtime_ready_primitives),
        "quality_screen_ready_primitives": sorted(ready_primitives),
        "preferred_gpus": [int(gpu) for gpu in gpus],
        "screen_steps": screen_steps,
        "confirmation_steps": confirmation_steps,
        "train_batch_size": train_batch_size,
        "row_count": len(rows),
        "screen_row_count": sum(1 for row in rows if row["phase"] == "screen_512"),
        "official_gate_row_count": sum(1 for row in rows if row["phase"] == "official_eval_gate"),
        "confirmation_row_count": sum(1 for row in rows if row["phase"] == "confirm_staged"),
        "confirmation_official_gate_row_count": sum(
            1 for row in rows if row["phase"] == "confirm_official_eval_gate"
        ),
        "checkpoint_finalizer_row_count": sum(
            1 for row in rows if row["phase"] == "checkpoint_ladder_finalize"
        ),
        "long_horizon_probe_row_count": sum(
            1 for row in rows if row["phase"] in {"long_horizon_baseline", "long_horizon_candidate"}
        ),
        "horizon_effect_profile_row_count": sum(
            1 for row in rows if row["phase"] == "horizon_effect_profile"
        ),
        "event_semantic_gate_row_count": sum(
            1 for row in rows if row["phase"] == "event_semantic_gate"
        ),
        "horizon_triptych_row_count": sum(
            1 for row in rows if row["phase"] == "horizon_triptych"
        ),
        "horizon_experience_map_row_count": sum(
            1 for row in rows if row["phase"] == "horizon_experience_map"
        ),
        "skipped_count": len(skipped),
        "allow_repeat_cells": allow_repeat_cells,
        "include_discovered_positive_screens": include_discovered_positive_screens,
        "rows": rows,
        "skipped": skipped,
        "policy": {
            "screen_rule": "Run 512-step canaries for closed-loop-ready primitives matched to non-positive environments.",
            "diagnostic_horizon_rule": "Use an event-covering search horizon for late-event environments; this evidence may route a candidate but cannot alter the frozen official verdict protocol.",
            "promotion_rule": "A positive 512 probe is only a candidate. Launch staged confirmation only after official ACWM eval.py at 50 inference steps passes the PSNR/SSIM/MSE quality gate.",
            "confirmation_rule": "Retain and independently evaluate 512/800/1000 checkpoints, then select the best passing checkpoint; later regression cannot overwrite an earlier best.",
            "long_horizon_rule": "For the selected best checkpoint, run baseline then candidate autoregressive rollouts to the environment-specific maximum horizon and write an effect profile before updating transfer experience. Pour-water candidates retain h200 as an intermediate diagnostic and must additionally pass the full-event h297 semantic gate.",
            "materialization_rule": "Primitives missing closed-loop runtime evidence stay as work orders and are not launched.",
        },
    }
    return _write_bundle(destination, report)


def _queue_row(
    *,
    rank: int,
    phase: str,
    environment: str,
    primitive: str,
    seed: int,
    train_steps: int,
    train_batch_size: int,
    campaign_id: str,
    report_root: Path,
    repo_root: Path,
    limited_gate: Path,
    failure_manifest: Path,
    goal_config: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    archive_db: Path,
    cas_root: Path,
    gpus: Sequence[int],
    primitive_parameters: Mapping[str, object] | None = None,
    proposal_routing_plan: Path | None = None,
    requires_positive_manifest: str = "",
    requires_official_quality_manifest: str = "",
    requires_positive_output_root: str = "",
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(report_root, campaign_id)
    output_root = report_root / campaign_id
    parameters = _runtime_parameters(primitive_parameters)
    row = {
        "rank": rank,
        "phase": phase,
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": train_steps,
        "train_batch_size": train_batch_size,
        "primitive_parameters": parameters,
        "proposal_routing_plan": str(Path(proposal_routing_plan).resolve()) if proposal_routing_plan else "",
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": phase != "screen_512",
        "requires_positive_manifest": requires_positive_manifest,
        "requires_official_quality_manifest": requires_official_quality_manifest,
        "requires_positive_output_root": requires_positive_output_root,
        "archive_db": str(archive_db),
        "cas_root": str(cas_root),
        "gpu_audit_root_template": str(report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"),
        "launch_argv_template": [
            str(repo_root / ".venv/bin/python3"),
            "-m",
            "wmloop.execute.training_eval_limited_campaign",
            "run",
            "--repo-root",
            str(repo_root),
            "--limited-gate-manifest",
            str(limited_gate),
            "--failure-report-manifest",
            str(failure_manifest),
            "--goal-config",
            str(goal_config),
            "--output-root",
            str(output_root),
            "--runtime-python",
            str(runtime_python),
            "--data-root",
            str(data_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--dataset-freeze",
            str(dataset_freeze),
            "--heldout-protocol",
            str(heldout_protocol),
            "--gpu-exclusivity-audit-manifest",
            "{gpu_audit_manifest}",
            "--gpus",
            "{gpu}",
            "--environment",
            environment,
            "--parallel-slots",
            "1",
            "--campaign-id",
            campaign_id,
            "--train-steps",
            str(train_steps),
            *(["--allow-extended-confirmation"] if phase in {"confirm_4k", "confirm_staged"} else []),
            "--train-batch-size",
            str(train_batch_size),
            "--train-val-batch-size",
            "8",
            "--train-size",
            "32",
            "--train-num-workers",
            "2",
            "--proposal-primitive",
            primitive,
            *(
                ["--proposal-routing-plan", str(Path(proposal_routing_plan).resolve(strict=True))]
                if proposal_routing_plan is not None
                else []
            ),
            "--weight",
            str(parameters["weight"]),
            "--action-balance-blend",
            str(parameters["action_balance_blend"]),
            "--action-balance-max-gain",
            str(parameters["action_balance_max_gain"]),
            "--history-noise",
            str(parameters["history_noise"]),
            "--keep-tokens",
            str(parameters["keep_tokens"]),
            "--frontier-weight",
            str(parameters["frontier_weight"]),
            "--event-weight",
            str(parameters["event_weight"]),
            "--event-quantile",
            str(parameters["event_quantile"]),
            "--event-visual-blend",
            str(parameters["event_visual_blend"]),
            "--next-forcing-chunks",
            str(parameters["next_forcing_chunks"]),
            "--next-forcing-steps",
            str(parameters["next_forcing_steps"]),
            "--next-forcing-lr",
            str(parameters["next_forcing_lr"]),
            "--reward-weight",
            str(parameters["reward_weight"]),
            "--inv-dyn-steps",
            str(parameters["inv_dyn_steps"]),
            "--inv-dyn-lr",
            str(parameters["inv_dyn_lr"]),
            "--memory-slots",
            str(parameters["memory_slots"]),
            "--memory-weight",
            str(parameters["memory_weight"]),
            "--anchor-every",
            str(parameters["anchor_every"]),
            "--anchor-weight",
            str(parameters["anchor_weight"]),
            "--guidance-start",
            str(parameters["guidance_start"]),
            "--guidance-end",
            str(parameters["guidance_end"]),
            "--wmsd-teacher-ema",
            str(parameters["wmsd_teacher_ema"]),
            "--wmsd-steps",
            str(parameters["wmsd_steps"]),
            "--wmsd-lr",
            str(parameters["wmsd_lr"]),
            "--self-forcing-rollout-horizon",
            str(parameters["self_forcing_rollout_horizon"]),
            "--self-forcing-steps",
            str(parameters["self_forcing_steps"]),
            "--self-forcing-lr",
            str(parameters["self_forcing_lr"]),
            "--trial-seed",
            str(seed),
            "--max-accept-trajectories",
            "1",
            "--replication-count",
            "1",
            "--eval-inference-steps",
            "1",
            "--hook-timeout-seconds",
            "60",
            "--training-timeout-seconds",
            "43200",
            "--eval-timeout-seconds",
            "1800",
            "--gpu-exclusivity-max-age-seconds",
            "3600",
            "--archive-db",
            str(archive_db),
            "--cas-root",
            str(cas_root),
            "--poll-interval-seconds",
            "5",
        ],
    }
    diagnostic_horizons = (
        _DIAGNOSTIC_SCREEN_HORIZONS_BY_ENVIRONMENT.get(environment)
        if phase == "screen_512"
        else None
    )
    if diagnostic_horizons is not None:
        row["diagnostic_eval_horizons"] = list(diagnostic_horizons)
        row["diagnostic_horizon_claim_boundary"] = (
            "Search-only evidence; formal verdicts remain governed by the frozen official gate."
        )
        launch = row["launch_argv_template"]
        if not isinstance(launch, list):  # pragma: no cover - construction invariant
            raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_LAUNCH_TEMPLATE_INVALID")
        launch.extend(
            ["--diagnostic-eval-horizons", *(str(value) for value in diagnostic_horizons)]
        )
    return row


_RUNTIME_PARAMETER_DEFAULTS: dict[str, object] = {
    "weight": 0.2,
    "action_balance_blend": 0.5,
    "action_balance_max_gain": 4.0,
    "history_noise": 0.1,
    "keep_tokens": 2,
    "frontier_weight": 1.0,
    "event_weight": 4.0,
    "event_quantile": 0.75,
    "event_visual_blend": 0.7,
    "next_forcing_chunks": 2,
    "next_forcing_steps": 1,
    "next_forcing_lr": 1e-5,
    "reward_weight": 0.5,
    "inv_dyn_steps": 1,
    "inv_dyn_lr": 1e-5,
    "memory_slots": 16,
    "memory_weight": 0.2,
    "anchor_every": 8,
    "anchor_weight": 0.2,
    "guidance_start": 1.0,
    "guidance_end": 1.5,
    "wmsd_teacher_ema": 0.9,
    "wmsd_steps": 1,
    "wmsd_lr": 1e-5,
    "self_forcing_rollout_horizon": 4,
    "self_forcing_steps": 1,
    "self_forcing_lr": 1e-5,
}


def _record_primitive_parameters(record: Mapping[str, object], primitive: str) -> dict[str, object]:
    raw = record.get("primitive_parameters")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_PRIMITIVE_PARAMETERS_INVALID")
    selected = raw.get(primitive, {})
    if not isinstance(selected, Mapping):
        raise AcwmAutoloopQueueError(f"ACWM_AUTOLOOP_PRIMITIVE_PARAMETERS_INVALID:{primitive}")
    unknown = sorted(set(map(str, selected)) - set(_RUNTIME_PARAMETER_DEFAULTS))
    if unknown:
        raise AcwmAutoloopQueueError(
            f"ACWM_AUTOLOOP_PRIMITIVE_PARAMETER_UNKNOWN:{primitive}:{','.join(unknown)}"
        )
    return {str(key): value for key, value in selected.items()}


def _runtime_parameters(overrides: Mapping[str, object] | None) -> dict[str, object]:
    parameters = dict(_RUNTIME_PARAMETER_DEFAULTS)
    if overrides:
        parameters.update(overrides)
    return parameters


def _official_gate_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    checkpoint_step: int,
    screen_output_root: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int],
    requires_positive_manifest: str,
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-autoloop-official-gate-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    candidate_checkpoint = screen_output_root / "envs" / environment / "retained_training" / "latest.pt"
    candidate_runtime_root = screen_output_root / "envs" / environment / "retained_runtime"
    return {
        "rank": rank,
        "phase": "official_eval_gate",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": requires_positive_manifest,
        "source_screen_manifest": requires_positive_manifest,
        "source_screen_output_root": str(screen_output_root),
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"),
        "launch_argv_template": [
            str(repo_root / ".venv/bin/python3"),
            str(repo_root / "scripts/export/acwm_formal_visualization.py"),
            "--output-root", str(output_root),
            "--environment", environment,
            "--primitive", primitive,
            "--seed", str(seed),
            "--runtime-python", str(runtime_python),
            "--data-root", str(data_root),
            "--checkpoint-root", str(checkpoint_root),
            "--dataset-freeze", str(DEFAULT_DATASET_FREEZE),
            "--heldout-protocol", str(DEFAULT_HELDOUT_PROTOCOL),
            "--candidate-checkpoint", str(candidate_checkpoint),
            "--candidate-runtime-root", str(candidate_runtime_root),
            "--gpu-index", "{gpu}",
            "--steps", "50",
            "--split", "ind_test",
            "--max-trajs", "3",
            "--max-saved-vids", "3",
            "--batch-size", "1",
            "--num-workers", "2",
            "--test-cuts", "1",
            "--hard-case-top-k", "1",
        ],
    }


def _confirmation_official_gate_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    checkpoint_step: int | None = None,
    confirmation_output_root: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int],
    requires_ready_manifest: str,
) -> dict[str, object]:
    step_suffix = f"-step{checkpoint_step}" if checkpoint_step is not None else ""
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-autoloop-confirm-official-gate-{environment}-{primitive}-s{seed}{step_suffix}-r1",
    )
    output_root = report_root / campaign_id
    retained_root = confirmation_output_root / "envs" / environment / "retained_training"
    candidate_checkpoint = (
        retained_root / "checkpoints" / f"relative_step_{checkpoint_step:06d}.pt"
        if checkpoint_step is not None
        else retained_root / "latest.pt"
    )
    candidate_runtime_root = confirmation_output_root / "envs" / environment / "retained_runtime"
    return {
        "rank": rank,
        "phase": "confirm_official_eval_gate",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "checkpoint_step": checkpoint_step or 0,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifest": requires_ready_manifest,
        "requires_official_quality_manifest": "",
        "source_confirmation_manifest": requires_ready_manifest,
        "source_confirmation_output_root": str(confirmation_output_root),
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"),
        "launch_argv_template": [
            str(repo_root / ".venv/bin/python3"),
            str(repo_root / "scripts/export/acwm_formal_visualization.py"),
            "--output-root", str(output_root),
            "--environment", environment,
            "--primitive", primitive,
            "--seed", str(seed),
            "--runtime-python", str(runtime_python),
            "--data-root", str(data_root),
            "--checkpoint-root", str(checkpoint_root),
            "--dataset-freeze", str(DEFAULT_DATASET_FREEZE),
            "--heldout-protocol", str(DEFAULT_HELDOUT_PROTOCOL),
            "--candidate-checkpoint", str(candidate_checkpoint),
            "--candidate-runtime-root", str(candidate_runtime_root),
            "--gpu-index", "{gpu}",
            "--steps", "50",
            "--split", "ind_test",
            "--max-trajs", "3",
            "--max-saved-vids", "3",
            "--batch-size", "1",
            "--num-workers", "2",
            "--test-cuts", "1",
            "--hard-case-top-k", "1",
        ],
    }


def _checkpoint_finalizer_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    confirmation_output_root: Path,
    official_gate_rows: Sequence[Mapping[str, object]],
    supplemental_checkpoints: Mapping[int, Path],
    report_root: Path,
    repo_root: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-autoloop-checkpoint-finalize-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    gate_specs = [
        f"{int(row['checkpoint_step'])}={Path(str(row['output_root'])) / 'manifest.json'}"
        for row in official_gate_rows
    ]
    ready_manifests = [spec.split("=", 1)[1] for spec in gate_specs]
    argv = [
        str(repo_root / ".venv/bin/python3"),
        str(repo_root / "scripts/export/acwm_checkpoint_ladder_finalize.py"),
        "--checkpoint-manifest",
        str(confirmation_output_root / "envs" / environment / "retained_training" / "manifest.json"),
        "--output-root",
        str(output_root),
        "--environment",
        environment,
        "--primitive",
        primitive,
        "--seed",
        str(seed),
    ]
    for spec in gate_specs:
        argv.extend(("--official-gate", spec))
    for step, path in sorted(supplemental_checkpoints.items()):
        argv.extend(("--checkpoint", f"{step}={Path(path)}"))
    return {
        "rank": rank,
        "phase": "checkpoint_ladder_finalize",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": ready_manifests,
        "requires_official_quality_manifest": "",
        "source_confirmation_output_root": str(confirmation_output_root),
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": argv,
    }


def _post_finalizer_rows(
    *,
    start_rank: int,
    environment: str,
    primitive: str,
    seed: int,
    finalizer_row: Mapping[str, object],
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    failure_manifest: Path,
    gpus: Sequence[int],
) -> list[dict[str, object]]:
    horizons = _HORIZONS_BY_ENVIRONMENT.get(environment)
    if horizons is None:
        raise AcwmAutoloopQueueError(f"ACWM_AUTOLOOP_HORIZON_LADDER_MISSING:{environment}")
    finalizer_manifest = Path(str(finalizer_row["output_root"])) / "manifest.json"
    best_checkpoint = Path(str(finalizer_row["output_root"])) / "best_checkpoint.pt"
    baseline = _long_horizon_probe_row(
        rank=start_rank,
        side="baseline",
        environment=environment,
        primitive=primitive,
        seed=seed,
        checkpoint_path=None,
        finalizer_manifest=finalizer_manifest,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        horizons=horizons,
        gpus=gpus,
    )
    candidate = _long_horizon_probe_row(
        rank=start_rank + 1,
        side="candidate",
        environment=environment,
        primitive=primitive,
        seed=seed,
        checkpoint_path=best_checkpoint,
        vendor_root=Path(str(finalizer_row.get("source_confirmation_output_root") or "")) / "envs" / environment / "retained_runtime",
        finalizer_manifest=finalizer_manifest,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        horizons=horizons,
        gpus=gpus,
    )
    baseline_manifest = str(Path(str(baseline["output_root"])) / "manifest.json")
    candidate["requires_ready_manifests"] = [
        str(finalizer_manifest),
        baseline_manifest,
    ]
    candidate.pop("requires_ready_manifest", None)
    next_rank = start_rank + 2
    event_gate: dict[str, object] | None = None
    if environment == "pour_water":
        event_gate = _pour_water_event_gate_row(
            rank=next_rank,
            primitive=primitive,
            seed=seed,
            baseline_row=baseline,
            candidate_row=candidate,
            report_root=report_root,
            repo_root=repo_root,
            runtime_python=runtime_python,
            gpus=gpus,
        )
        next_rank += 1
    profile = _horizon_effect_profile_row(
        rank=next_rank,
        environment=environment,
        primitive=primitive,
        seed=seed,
        baseline_row=baseline,
        candidate_row=candidate,
        checkpoint_ladder_manifest=finalizer_manifest,
        event_gate_row=event_gate,
        failure_report=Path(failure_manifest).resolve().parent / "failure_reports" / f"{environment}.json",
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        gpus=gpus,
    )
    triptych = _horizon_triptych_row(
        rank=next_rank + 1,
        environment=environment,
        primitive=primitive,
        seed=seed,
        baseline_row=baseline,
        candidate_row=candidate,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        gpus=gpus,
    )
    experience = _horizon_experience_map_row(
        rank=next_rank + 2,
        environment=environment,
        primitive=primitive,
        seed=seed,
        profile_row=profile,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        gpus=gpus,
    )
    rows = [baseline, candidate]
    if event_gate is not None:
        rows.append(event_gate)
    rows.extend((profile, triptych, experience))
    return rows


def _pour_water_event_gate_row(
    *,
    rank: int,
    primitive: str,
    seed: int,
    baseline_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-pour-water-event-gate-{primitive}-s{seed}-h297-r1",
    )
    output_root = report_root / campaign_id
    dependencies = [
        str(Path(str(baseline_row["output_root"])) / "manifest.json"),
        str(Path(str(candidate_row["output_root"])) / "manifest.json"),
    ]
    return {
        "rank": rank,
        "phase": "event_semantic_gate",
        "campaign_id": campaign_id,
        "environment": "pour_water",
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": dependencies,
        "requires_official_quality_manifest": "",
        "resource_class": "cpu",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(runtime_python),
            str(repo_root / "scripts/export/acwm_pour_water_event_gate.py"),
            "--baseline-manifest",
            dependencies[0],
            "--candidate-manifest",
            dependencies[1],
            "--output-root",
            str(output_root),
            "--primitive",
            primitive,
            "--seed",
            str(seed),
        ],
    }


def _long_horizon_probe_row(
    *,
    rank: int,
    side: str,
    environment: str,
    primitive: str,
    seed: int,
    checkpoint_path: Path | None,
    vendor_root: Path | None = None,
    finalizer_manifest: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    horizons: Sequence[int],
    gpus: Sequence[int],
) -> dict[str, object]:
    trajectory_count = _LONG_HORIZON_TRAJECTORIES_BY_ENVIRONMENT.get(environment, 3)
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-autoloop-long-horizon-{side}-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    argv = [
        str(runtime_python),
        "-m",
        "wmloop.diagnose.horizon_runtime",
        "run",
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--environment",
        environment,
        "--split",
        "ind_test",
        "--output-root",
        str(output_root),
        "--horizons",
        *(str(value) for value in horizons),
        "--max-trajectories",
        str(trajectory_count),
        "--num-inference-steps",
        "50",
        "--device",
        "cuda",
        "--seed",
        "101",
        "--mode",
        "autoregressive",
        "--max-evidence",
        "1",
        "--max-video-evidence",
        str(trajectory_count),
        "--archive-db",
        str(DEFAULT_ARCHIVE_DB),
        "--cas-root",
        str(DEFAULT_CAS_ROOT),
        "--gpu-index",
        "{gpu}",
        "--gpu-exclusivity-audit-manifest",
        "{gpu_audit_manifest}",
        "--gpu-exclusivity-max-age-seconds",
        "3600",
    ]
    if vendor_root is not None:
        checkpoint_index = argv.index("--data-root")
        argv[checkpoint_index:checkpoint_index] = ["--vendor-root", str(vendor_root)]
    if checkpoint_path is not None:
        checkpoint_index = argv.index("--environment")
        argv[checkpoint_index:checkpoint_index] = ["--checkpoint-path", str(checkpoint_path)]
    return {
        "rank": rank,
        "phase": f"long_horizon_{side}",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifest": str(finalizer_manifest),
        "requires_official_quality_manifest": "",
        "source_finalizer_manifest": str(finalizer_manifest),
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": argv,
    }


def _horizon_effect_profile_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    baseline_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    checkpoint_ladder_manifest: Path,
    event_gate_row: Mapping[str, object] | None,
    failure_report: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-horizon-effect-profile-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    dependencies = [
        str(Path(str(baseline_row["output_root"])) / "manifest.json"),
        str(Path(str(candidate_row["output_root"])) / "manifest.json"),
        str(checkpoint_ladder_manifest),
    ]
    event_gate_path: Path | None = None
    if event_gate_row is not None:
        event_gate_manifest = Path(str(event_gate_row["output_root"])) / "manifest.json"
        dependencies.append(str(event_gate_manifest))
        event_gate_path = Path(str(event_gate_row["output_root"])) / "event-gate.json"
    argv = [
        str(runtime_python),
        str(repo_root / "scripts/export/acwm_horizon_effect_profile.py"),
        "--baseline-manifest",
        dependencies[0],
        "--candidate-manifest",
        dependencies[1],
        "--primitive",
        primitive,
        "--failure-report",
        str(failure_report),
        "--mechanism-cards",
        str(DEFAULT_MECHANISM_CARDS),
        "--checkpoint-ladder-manifest",
        str(checkpoint_ladder_manifest),
        "--output-root",
        str(output_root),
    ]
    if event_gate_path is not None:
        output_index = argv.index("--output-root")
        argv[output_index:output_index] = ["--event-gate", str(event_gate_path)]
    return {
        "rank": rank,
        "phase": "horizon_effect_profile",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": dependencies,
        "requires_official_quality_manifest": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": argv,
    }


def _horizon_experience_map_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    profile_row: Mapping[str, object],
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-horizon-experience-map-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    dependency = Path(str(profile_row["output_root"])) / "manifest.json"
    return {
        "rank": rank,
        "phase": "horizon_experience_map",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifest": str(dependency),
        "requires_official_quality_manifest": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(runtime_python),
            str(repo_root / "scripts/export/acwm_horizon_experience_map.py"),
            "--profile",
            str(Path(str(profile_row["output_root"])) / "horizon-effect-profile.json"),
            "--output-root",
            str(output_root),
        ],
    }


def _horizon_triptych_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    seed: int,
    baseline_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    campaign_id = _unique_campaign_id(
        report_root,
        f"acwm-horizon-triptych-{environment}-{primitive}-s{seed}-r1",
    )
    output_root = report_root / campaign_id
    dependencies = [
        str(Path(str(baseline_row["output_root"])) / "manifest.json"),
        str(Path(str(candidate_row["output_root"])) / "manifest.json"),
    ]
    return {
        "rank": rank,
        "phase": "horizon_triptych",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in gpus],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": dependencies,
        "requires_official_quality_manifest": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(runtime_python),
            str(repo_root / "scripts/export/acwm_horizon_triptych.py"),
            "--baseline-manifest",
            dependencies[0],
            "--candidate-manifest",
            dependencies[1],
            "--output-root",
            str(output_root),
        ],
    }


def _unique_campaign_id(report_root: Path, campaign_id: str) -> str:
    if not (report_root / campaign_id).exists() and not (report_root / campaign_id).is_symlink():
        return campaign_id
    for index in range(2, 1000):
        candidate = f"{campaign_id}-retry{index}"
        if not (report_root / candidate).exists() and not (report_root / candidate).is_symlink():
            return candidate
    raise AcwmAutoloopQueueError(f"ACWM_AUTOLOOP_CAMPAIGN_ID_EXHAUSTED:{campaign_id}")


def _ready_primitives(manifest: Mapping[str, object]) -> set[str]:
    values = manifest.get("quality_screen_ready_primitives")
    if not isinstance(values, list):
        values = manifest.get("closed_loop_ready_primitives")
    if not isinstance(values, list):
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_READY_PRIMITIVES_INVALID")
    return {str(value) for value in values if isinstance(value, str) and value} & TRAINING_QUALITY_SCREEN_PRIMITIVES


def _runtime_ready_primitives(manifest: Mapping[str, object]) -> set[str]:
    values = manifest.get("closed_loop_ready_primitives")
    if not isinstance(values, list):
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_READY_PRIMITIVES_INVALID")
    return {str(value) for value in values if isinstance(value, str) and value}


def _discover_existing_cells(report_root: Path) -> dict[tuple[str, str], list[str]]:
    cells: dict[tuple[str, str], list[str]] = {}
    for status_path in sorted(report_root.glob("*/status.json")):
        status = _load_optional_json(status_path)
        records = status.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping) or not _record_blocks_cell(record):
                continue
            environment = str(record.get("environment") or "")
            primitive = str(record.get("proposal_primitive") or "")
            train_steps = _int_or_none(_command_value(record, "--train-steps"))
            if environment and primitive and train_steps is not None and train_steps >= 512:
                cells.setdefault((environment, primitive), []).append(status_path.parent.name)
    for manifest_path in sorted(report_root.glob("*/envs/*/manifest.json")):
        manifest = _load_optional_json(manifest_path)
        if manifest.get("state") != "ready":
            continue
        environment = str(manifest.get("environment") or manifest_path.parent.name)
        primitive = _infer_primitive(str(manifest.get("proposal_id") or ""))
        if environment and primitive and primitive != "unknown":
            cells.setdefault((environment, primitive), []).append(manifest_path.parents[2].name)
    return {cell: sorted(set(campaigns)) for cell, campaigns in cells.items()}


def _discover_existing_confirmation_cells(report_root: Path, confirmation_steps: int) -> dict[tuple[str, str], list[str]]:
    cells: dict[tuple[str, str], list[str]] = {}
    for status_path in sorted(report_root.glob("*/status.json")):
        status = _load_optional_json(status_path)
        records = status.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping) or not _record_blocks_cell(record):
                continue
            environment = str(record.get("environment") or "")
            primitive = str(record.get("proposal_primitive") or "")
            train_steps = _int_or_none(_command_value(record, "--train-steps"))
            if environment and primitive and train_steps is not None and train_steps >= confirmation_steps:
                cells.setdefault((environment, primitive), []).append(status_path.parent.name)
    for manifest_path in sorted(report_root.glob("*/envs/*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        manifest = _load_optional_json(manifest_path)
        if manifest.get("state") != "ready":
            continue
        campaign = manifest_path.parents[2]
        status_record = _status_by_output(campaign / "status.json").get(str(manifest_path.parent.resolve()))
        train_steps = _first_int(
            manifest.get("train_steps"),
            _command_value(status_record, "--train-steps"),
            _parse_train_steps(campaign.name),
        )
        environment = str(manifest.get("environment") or manifest_path.parent.name)
        primitive = _infer_primitive(str(manifest.get("proposal_id") or ""))
        if environment and primitive and primitive != "unknown" and train_steps is not None and train_steps >= confirmation_steps:
            cells.setdefault((environment, primitive), []).append(campaign.name)
    return {cell: sorted(set(campaigns)) for cell, campaigns in cells.items()}


def _discover_positive_screens(report_root: Path, *, min_steps: int, confirmation_steps: int) -> list[dict[str, object]]:
    screens: list[dict[str, object]] = []
    for manifest_path in sorted(report_root.glob("*/envs/*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        manifest = _load_optional_json(manifest_path)
        if manifest.get("state") != "ready":
            continue
        campaign = manifest_path.parents[2]
        status_record = _status_by_output(campaign / "status.json").get(str(manifest_path.parent.resolve()))
        train_steps = _first_int(
            manifest.get("train_steps"),
            _command_value(status_record, "--train-steps"),
            _parse_train_steps(campaign.name),
        )
        if train_steps is None or train_steps < min_steps or train_steps >= confirmation_steps:
            continue
        primary_metric = str(manifest.get("primary_metric") or "ladder_auc_psnr_envmax")
        delta = _metric_delta(manifest, primary_metric)
        if delta is None or delta <= 0.0:
            continue
        action_gate = manifest.get("action_following_gate")
        if isinstance(action_gate, Mapping) and action_gate.get("enabled") is True and action_gate.get("pass") is not True:
            continue
        environment = str(manifest.get("environment") or manifest_path.parent.name)
        primitive = _infer_primitive(str(manifest.get("proposal_id") or ""))
        seed = _first_int(manifest.get("seed"), _command_value(status_record, "--trial-seed"), _parse_seed(campaign.name))
        if not environment or primitive == "unknown" or seed is None:
            continue
        screens.append(
            {
                "environment": environment,
                "primitive": primitive,
                "seed": seed,
                "train_steps": train_steps,
                "delta": delta,
                "manifest_path": str(manifest_path),
                "output_root": str(campaign),
            }
        )
    return screens


def _record_blocks_cell(record: Mapping[str, object]) -> bool:
    state = record.get("state")
    if state == "ready":
        return True
    if state != "running":
        return False
    pid = _int_or_none(record.get("pid"))
    return True if pid is None else _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _status_by_output(status_path: Path) -> dict[str, Mapping[str, object]]:
    status = _load_optional_json(status_path)
    records = status.get("records")
    if not isinstance(records, list):
        return {}
    output: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        output_root = record.get("output_root")
        if isinstance(output_root, str) and output_root:
            output[str(Path(output_root).resolve())] = record
    return output


def _command_value(record: Mapping[str, object], flag: str) -> object:
    if not isinstance(record, Mapping):
        return ""
    command = record.get("command")
    if not isinstance(command, list):
        return ""
    for index, value in enumerate(command):
        if value == flag and index + 1 < len(command):
            return command[index + 1]
    return ""


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _first_int(*values: object) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _parse_seed(value: str) -> int | None:
    match = re.search(r"(?:^|-)s(\d+)(?:-|$)", value)
    if match is None:
        return None
    return int(match.group(1))


def _parse_train_steps(value: str) -> int | None:
    for pattern in (r"(?:^|-)t(\d+)(?:-|$)", r"(?:^|-)(\d+)k(?:-|$)", r"step(\d+)"):
        match = re.search(pattern, value)
        if match is None:
            continue
        raw = int(match.group(1))
        if "k" in pattern:
            return raw * 1000
        return raw
    return None


def _metric_delta(manifest: Mapping[str, object], primary_metric: str) -> float | None:
    raw = manifest.get("delta_m_ver")
    if not isinstance(raw, Mapping):
        return None
    for key in (primary_metric, "ladder_auc_psnr_envmax"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _infer_primitive(proposal_id: str) -> str:
    match = re.search(r"training-eval-smoke-(.+?)-unlabeled", proposal_id)
    if match is not None:
        return match.group(1)
    return "unknown"


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmAutoloopQueueError("ACWM_AUTOLOOP_QUEUE_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "autoloop-queue.json", report)
        _write_csv(temporary / "tables" / "autoloop-queue.csv", report.get("rows", []))
        _write_csv(temporary / "tables" / "skipped.csv", report.get("skipped", []))
        _write_markdown(temporary / "autoloop-queue.md", report)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-autoloop-queue-manifest",
            "state": report["state"],
            "row_count": report["row_count"],
            "screen_row_count": report["screen_row_count"],
            "confirmation_row_count": report["confirmation_row_count"],
            "official_gate_row_count": report["official_gate_row_count"],
            "confirmation_official_gate_row_count": report["confirmation_official_gate_row_count"],
            "checkpoint_finalizer_row_count": report["checkpoint_finalizer_row_count"],
            "long_horizon_probe_row_count": report["long_horizon_probe_row_count"],
            "horizon_effect_profile_row_count": report["horizon_effect_profile_row_count"],
            "event_semantic_gate_row_count": report["event_semantic_gate_row_count"],
            "horizon_triptych_row_count": report["horizon_triptych_row_count"],
            "horizon_experience_map_row_count": report["horizon_experience_map_row_count"],
            "skipped_count": report["skipped_count"],
            "queue_path": str(destination / "autoloop-queue.json"),
            "markdown_path": str(destination / "autoloop-queue.md"),
            "queue_csv": str(destination / "tables" / "autoloop-queue.csv"),
            "skipped_csv": str(destination / "tables" / "skipped.csv"),
            "preferred_gpus": report["preferred_gpus"],
            "policy": report["policy"],
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# ACWM Autoloop Queue",
        "",
        f"State: `{report['state']}`",
        f"Rows: `{report['row_count']}`",
        f"512 screens: `{report['screen_row_count']}`",
        f"Official eval gates: `{report['official_gate_row_count']}`",
        f"Staged confirmations: `{report['confirmation_row_count']}`",
        f"Checkpoint official re-eval gates: `{report['confirmation_official_gate_row_count']}`",
        f"Checkpoint finalizers: `{report['checkpoint_finalizer_row_count']}`",
        f"Long-horizon probes: `{report['long_horizon_probe_row_count']}`",
        f"Horizon effect profiles: `{report['horizon_effect_profile_row_count']}`",
        f"Event semantic gates: `{report['event_semantic_gate_row_count']}`",
        f"Horizon triptychs: `{report['horizon_triptych_row_count']}`",
        f"Horizon experience entries: `{report['horizon_experience_map_row_count']}`",
        f"Skipped: `{report['skipped_count']}`",
        "",
        "| Rank | Phase | Env | Primitive | Seed | Steps | Depends on |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    rows = report.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                lines.append(
                    "| {rank} | `{phase}` | {environment} | {primitive} | {seed} | {train_steps} | {requires_positive_output_root} |".format(
                        rank=row.get("rank", ""),
                        phase=row.get("phase", ""),
                        environment=row.get("environment", ""),
                        primitive=row.get("primitive", ""),
                        seed=row.get("seed", ""),
                        train_steps=row.get("train_steps", ""),
                        requires_positive_output_root=(
                            row.get("requires_positive_output_root", "")
                            or row.get("source_screen_output_root", "")
                        ),
                    )
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        rows = []
    fieldnames = sorted({key for row in rows if isinstance(row, Mapping) for key in row})
    if not fieldnames:
        fieldnames = ["environment", "primitive", "reason"]
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if isinstance(row, Mapping):
                writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmAutoloopQueueError(f"ACWM_AUTOLOOP_JSON_NOT_OBJECT:{path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        if not path.is_file():
            return {}
        return _load_json_object(path)
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--gap-plan", type=Path, default=DEFAULT_GAP_PLAN)
    parser.add_argument("--materialization-gate", type=Path, default=DEFAULT_MATERIALIZATION_GATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--report-root", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--limited-gate", type=Path, default=DEFAULT_LIMITED_GATE)
    parser.add_argument("--failure-manifest", type=Path, default=DEFAULT_FAILURE_MANIFEST)
    parser.add_argument("--goal-config", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-freeze", type=Path, default=DEFAULT_DATASET_FREEZE)
    parser.add_argument("--heldout-protocol", type=Path, default=DEFAULT_HELDOUT_PROTOCOL)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--cas-root", type=Path, default=DEFAULT_CAS_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[801])
    parser.add_argument("--screen-steps", type=int, default=512)
    parser.add_argument("--confirmation-steps", type=int, default=DEFAULT_CONFIRMATION_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--allow-repeat-cells", action="store_true")
    parser.add_argument("--skip-discovered-positive-screens", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_autoloop_queue(
        gap_plan=args.gap_plan.resolve(strict=True),
        materialization_gate=args.materialization_gate.resolve(strict=True),
        output_root=args.output_root.resolve(),
        repo_root=args.repo_root.resolve(strict=True),
        report_root=args.report_root.resolve(),
        limited_gate=args.limited_gate.resolve(strict=True),
        failure_manifest=args.failure_manifest.resolve(strict=True),
        goal_config=args.goal_config,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        dataset_freeze=args.dataset_freeze,
        heldout_protocol=args.heldout_protocol,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
        gpus=args.gpus,
        seeds=args.seeds,
        screen_steps=args.screen_steps,
        confirmation_steps=args.confirmation_steps,
        train_batch_size=args.train_batch_size,
        allow_repeat_cells=args.allow_repeat_cells,
        include_discovered_positive_screens=not args.skip_discovered_positive_screens,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
