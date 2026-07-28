#!/usr/bin/env python3
"""Select and queue the next diagnosis-matched ACWM primitive experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from scripts.export.acwm_autoloop_queue import (
    DEFAULT_ARCHIVE_DB,
    DEFAULT_CAS_ROOT,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNTIME_PYTHON,
    ROOT,
    _HORIZONS_BY_ENVIRONMENT,
    _confirmation_official_gate_row,
    _discover_existing_cells,
    _runtime_parameters,
    _unique_campaign_id,
    build_autoloop_queue,
)
from scripts.export.acwm_confirmed_delta_horizon_queue import (
    build_confirmed_delta_horizon_queue,
)
from wmloop.execute.acwm_primitive_routes import (
    INVALIDATED_QUALITY_PRIMITIVES,
    QUALITY_SCREEN_PRIMITIVES,
    RUNTIME_ONLY_PRIMITIVES,
    TRAINING_QUALITY_SCREEN_PRIMITIVES,
)
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.verify.primitive_materialization_gate import run_primitive_materialization_gate


class AcwmAutoloopReplenisherError(RuntimeError):
    """Dynamic candidate selection or queue construction failed closed."""


_SCREEN_NAME = re.compile(r"^acwm-autoloop-screen-(.+)-([A-Za-z0-9_]+)-s(\d+)-t\d+-r\d+(?:-retry\d+)?$")
_DYNAMIC_GATE_NAME = re.compile(
    r"^primitive-materialization-gate-dynamic-r(\d+)(?:-[A-Za-z0-9][A-Za-z0-9_-]*)?$"
)
_CONFIRM_NAME = re.compile(
    r"^acwm-autoloop-confirm-(.+)-([A-Za-z0-9_]+)-s(\d+)-t(\d+)-r\d+(?:-retry\d+)?$"
)
_SCREEN_OFFICIAL_NAME = re.compile(
    r"^acwm-autoloop-official-gate-(.+)-([A-Za-z0-9_]+)-s(\d+)-r\d+(?:-retry\d+)?$"
)
_RUNTIME_CONFIRMATION_REQUIRED_PASSES = 2
_CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES = 2
_CHECKPOINT_DELTA_REPLICATION_TRAJECTORIES = 4
_UNLAUNCHED_QUEUE_RESERVATION_SECONDS = 3600.0
_CHECKPOINT_DELTA_RECOVERY_ALPHAS = (-0.02, -0.05, 0.02, 0.05, 0.10, 0.25, 0.50)
EVENT_SEMANTIC_REQUIRED_ENVIRONMENTS = frozenset({"pour_water"})
_POUR_WATER_EVENT_MIN_TRAJECTORIES = 4
_POUR_WATER_EVENT_MIN_FRAMES = 297

_PARAMETER_VARIANTS: dict[str, tuple[dict[str, object], ...]] = {
    "action_dimension_balancing": (
        {"action_balance_blend": 0.5, "action_balance_max_gain": 4.0},
        {"action_balance_blend": 0.25, "action_balance_max_gain": 2.0},
        {"action_balance_blend": 0.75, "action_balance_max_gain": 4.0},
        {"action_balance_blend": 0.5, "action_balance_max_gain": 8.0},
    ),
    "action_contrastive_finetune": (
        {"weight": 0.1},
        {"weight": 0.05},
        {"weight": 0.2},
        {"weight": 0.02},
        {"weight": 0.01},
        {"weight": 0.005},
    ),
    "dino_rep_injection": (
        {"weight": 0.1},
        {"weight": 0.05},
        {"weight": 0.2},
    ),
    "drift_token_trim": (
        {"keep_tokens": 2},
        {"keep_tokens": 4},
        {"keep_tokens": 1},
    ),
    "event_window_reweight": (
        {"event_weight": 4.0, "event_quantile": 0.75, "event_visual_blend": 0.7},
        {"event_weight": 8.0, "event_quantile": 0.75, "event_visual_blend": 0.7},
        {"event_weight": 4.0, "event_quantile": 0.85, "event_visual_blend": 0.8},
        {"event_weight": 2.0, "event_quantile": 0.70, "event_visual_blend": 0.5},
    ),
    "cfg_guidance_schedule": (
        {"guidance_start": 1.0, "guidance_end": 1.5},
        {"guidance_start": 0.8, "guidance_end": 1.2},
        {"guidance_start": 1.0, "guidance_end": 2.0},
        {"guidance_start": 1.0, "guidance_end": 1.1},
        {"guidance_start": 0.95, "guidance_end": 1.05},
    ),
    "first_frame_anchor": (
        {"anchor_every": 8, "anchor_weight": 0.2},
        {"anchor_every": 4, "anchor_weight": 0.1},
        {"anchor_every": 16, "anchor_weight": 0.1},
    ),
    "frontier_collection": (
        {"frontier_weight": 1.0},
        {"frontier_weight": 1.5},
        {"frontier_weight": 2.0},
    ),
    "history_noise_schedule": (
        {"history_noise": 0.1},
        {"history_noise": 0.05},
        {"history_noise": 0.03},
        {"history_noise": 0.15},
    ),
    "inv_dyn_reward_finetune": (
        {"reward_weight": 0.5, "inv_dyn_steps": 1, "inv_dyn_lr": 1e-5},
        {"reward_weight": 0.25, "inv_dyn_steps": 1, "inv_dyn_lr": 1e-5},
        {"reward_weight": 0.75, "inv_dyn_steps": 1, "inv_dyn_lr": 5e-6},
    ),
    "latent_motion_prior": (
        {"weight": 0.2},
        {"weight": 0.1},
        {"weight": 0.05},
    ),
    "latent_spatial_memory": (
        {"memory_slots": 16, "memory_weight": 0.2},
        {"memory_slots": 32, "memory_weight": 0.1},
        {"memory_slots": 16, "memory_weight": 0.1},
    ),
    "mixture_reweight": (
        {"frontier_weight": 1.0},
        {"frontier_weight": 1.5},
        {"frontier_weight": 2.0},
        {"frontier_weight": 0.5},
        {"frontier_weight": 0.25},
        {"frontier_weight": 0.1},
    ),
    "motion_region_reweight": (
        {"weight": 0.5},
        {"weight": 1.0},
        {"weight": 0.25},
        {"weight": 0.1},
        {"weight": 0.05},
    ),
    "next_forcing": (
        {"next_forcing_chunks": 2, "next_forcing_steps": 1, "next_forcing_lr": 1e-5},
        {"next_forcing_chunks": 3, "next_forcing_steps": 1, "next_forcing_lr": 1e-5},
        {"next_forcing_chunks": 4, "next_forcing_steps": 1, "next_forcing_lr": 5e-6},
    ),
    "self_forcing_finetune": (
        {"self_forcing_rollout_horizon": 4, "self_forcing_steps": 1, "self_forcing_lr": 1e-5},
        {"self_forcing_rollout_horizon": 2, "self_forcing_steps": 1, "self_forcing_lr": 5e-6},
        {"self_forcing_rollout_horizon": 8, "self_forcing_steps": 1, "self_forcing_lr": 1e-5},
    ),
    "wmsd_self_distill": (
        {"wmsd_teacher_ema": 0.9, "wmsd_steps": 1, "wmsd_lr": 1e-5},
        {"wmsd_teacher_ema": 0.99, "wmsd_steps": 1, "wmsd_lr": 5e-6},
        {"wmsd_teacher_ema": 0.95, "wmsd_steps": 1, "wmsd_lr": 1e-5},
    ),
}

_ADMISSION_PARAMETERS: dict[str, dict[str, object]] = {
    "action_dimension_balancing": {"action_balance_blend": 0.5, "action_balance_max_gain": 4.0},
    "dino_rep_injection": {"weight": 0.1},
    "event_window_reweight": {
        "event_weight": 4.0,
        "event_quantile": 0.75,
        "event_visual_blend": 0.7,
    },
    "cfg_guidance_schedule": {"guidance_start": 1.0, "guidance_end": 1.5},
    "first_frame_anchor": {"anchor_every": 8, "anchor_weight": 0.2},
    "motion_region_reweight": {"weight": 0.5},
    "wmsd_self_distill": {"wmsd_teacher_ema": 0.9, "wmsd_steps": 1, "wmsd_lr": 1e-5},
}


def replenish_autoloop_queue(
    *,
    staging_plan: Path,
    materialization_gate: Path,
    report_root: Path,
    output_root: Path,
    gpu: int,
    seed: int,
    repo_root: Path = ROOT,
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    quality_discovery_only: bool = False,
    promote_current_contract_queues: bool = False,
) -> dict[str, object]:
    if gpu < 0 or seed < 1:
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_ARGUMENT_INVALID")
    plan = _load_json_object(staging_plan)
    materialization_gate = _refresh_materialization_gate(
        configured_gate=Path(materialization_gate),
        report_root=Path(report_root),
        repo_root=Path(repo_root),
    )
    gate = _load_json_object(materialization_gate)
    gate_records = _materialization_records(gate)
    records = plan.get("environment_records")
    if not isinstance(records, list):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_PLAN_RECORDS_INVALID")
    failure_manifest = _current_failure_manifest(Path(report_root))
    ready_values = gate.get("quality_screen_ready_primitives")
    if not isinstance(ready_values, list):
        ready_values = gate.get("closed_loop_ready_primitives", [])
    quality_ready = {str(value) for value in ready_values if isinstance(value, str)} & QUALITY_SCREEN_PRIMITIVES
    ready = quality_ready & TRAINING_QUALITY_SCREEN_PRIMITIVES
    registry = _load_registry_if_present(Path(repo_root))
    attempted = _attempted_parameter_signatures(Path(report_root))
    attempted_runtime_only = _attempted_runtime_only_signatures(Path(report_root))
    official_gate_failures = _official_training_gate_failure_counts(Path(report_root))
    screen_negative_counts = _training_screen_negative_counts(Path(report_root))
    runtime_confirmation_candidates = _pending_runtime_only_confirmation_candidates(Path(report_root))
    runtime_confirmation_pending = _pending_runtime_only_confirmation_environments(Path(report_root))
    checkpoint_delta_confirmation_candidates = _pending_checkpoint_delta_confirmation_candidates(
        Path(report_root)
    )
    checkpoint_delta_confirmation_pending = _pending_checkpoint_delta_confirmation_environments(
        Path(report_root)
    )
    checkpoint_delta_horizon_candidates = _pending_checkpoint_delta_horizon_candidates(
        Path(report_root)
    )
    checkpoint_delta_replication_candidates = (
        _pending_checkpoint_delta_horizon_replication_candidates(Path(report_root))
    )
    event_semantic_unresolved = _event_semantic_unresolved_environments(Path(report_root))
    terminal_positive = _officially_positive_environments(Path(report_root))
    if quality_discovery_only:
        terminal_positive |= _quality_discovery_terminal_positive_environments(Path(report_root))
    terminal_positive -= event_semantic_unresolved
    transfer_primitives = _officially_positive_training_primitives(Path(report_root))
    confirmation_pending = _pending_positive_confirmation_environments(Path(report_root))
    running_screen_environments = _running_screen_environments(Path(report_root))
    paused_environments = (
        terminal_positive
        | runtime_confirmation_pending
        | checkpoint_delta_confirmation_pending
    )
    if not quality_discovery_only:
        paused_environments |= confirmation_pending
    if quality_discovery_only:
        # Independent parameter signatures may run concurrently on separate GPUs.
        # Signature admission below still prevents duplicate cells; only a newly
        # positive or internally-promoted environment remains paused.
        paused_environments |= _pending_internal_positive_environments(Path(report_root))
    experience_entries = _load_horizon_experience_entries(Path(report_root))
    candidates: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for evidence_rank, evidence in enumerate(checkpoint_delta_horizon_candidates):
        event_priority = str(evidence["environment"]) in event_semantic_unresolved
        candidates.append(
            (
                (-5 if event_priority else -2.5, evidence_rank, str(evidence["environment"])),
                evidence,
            )
        )
    for replication_rank, replication in enumerate(checkpoint_delta_replication_candidates):
        candidates.append(
            (
                (-2.25, replication_rank, str(replication["environment"])),
                replication,
            )
        )
    for confirmation_rank, confirmation in enumerate(checkpoint_delta_confirmation_candidates):
        candidates.append(
            (
                (-4, confirmation_rank, str(confirmation["environment"])),
                confirmation,
            )
        )
    for confirmation_rank, confirmation in enumerate(runtime_confirmation_candidates):
        candidates.append(
            (
                (-3, confirmation_rank, str(confirmation["environment"])),
                confirmation,
            )
        )
    unavailable: list[dict[str, object]] = []
    attempted_admissions = _attempted_admissions(Path(report_root))
    environment_attempts = {
        environment: sum(1 for attempted_environment, _, _ in attempted if attempted_environment == environment)
        for environment in {
            str(record.get("environment") or "")
            for record in records
            if isinstance(record, Mapping)
        }
    }
    if not quality_discovery_only or promote_current_contract_queues:
        for confirmation in _pending_confirmation_official_candidates(Path(report_root)):
            candidates.append(((-2, 0, 0, 0, str(confirmation["environment"])), confirmation))
    for recovery_rank, recovery in enumerate(
        _pending_checkpoint_delta_recovery_candidates(
            Path(report_root),
            excluded_environments=terminal_positive | checkpoint_delta_confirmation_pending,
        )
    ):
        candidates.append(
            (
                (-1, -2, recovery_rank, str(recovery["environment"])),
                recovery,
            )
        )
    for env_rank, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            continue
        environment = str(raw_record.get("environment") or "")
        if environment in paused_environments:
            continue
        recommended_raw = raw_record.get("recommended_existing_primitives")
        if not environment or not isinstance(recommended_raw, list):
            continue
        recommended = [str(value) for value in recommended_raw]
        target_failures = _diagnosed_failure_names(
            environment=environment,
            registry=registry,
            report_root=Path(report_root),
            failure_manifest=failure_manifest,
        )
        target_failures = tuple(
            dict.fromkeys(
                (
                    *target_failures,
                    *_staged_diagnostic_failure_names(plan=plan, record=raw_record),
                )
            )
        )
        for primitive in _diagnosis_matched_primitive_names(
            environment=environment,
            registry=registry,
            report_root=Path(report_root),
            failure_manifest=failure_manifest,
        ):
            if primitive not in recommended:
                recommended.append(primitive)
        for primitive in transfer_primitives:
            if primitive not in recommended:
                recommended.append(primitive)
        for primitive_rank, raw_primitive in enumerate(recommended):
            primitive = str(raw_primitive)
            if primitive not in transfer_primitives and not _primitive_matches_failures(
                primitive=primitive,
                failures=target_failures,
                registry=registry,
            ):
                unavailable.append(
                    {
                        "environment": environment,
                        "primitive": primitive,
                        "reason": "diagnostic_target_mismatch",
                        "target_failure_signatures": list(target_failures),
                        "primitive_target_failures": list(registry.manifest(primitive).targets_failures)
                        if registry is not None and primitive in registry.names()
                        else [],
                    }
                )
                continue
            if primitive in RUNTIME_ONLY_PRIMITIVES and primitive in quality_ready:
                variants = _PARAMETER_VARIANTS.get(primitive, ({},))
                for variant_rank, parameters in enumerate(variants):
                    signature = _signature(_runtime_parameters(parameters))
                    if (environment, primitive, signature) in attempted_runtime_only:
                        continue
                    candidates.append(
                        (
                            (
                                1,
                                1 if environment in running_screen_environments else 0,
                                environment_attempts.get(environment, 0),
                                primitive_rank,
                                variant_rank,
                                f"{env_rank:04d}:{environment}",
                            ),
                            {
                                "kind": "runtime_only_official_gate",
                                "environment": environment,
                                "primitive": primitive,
                                "parameters": parameters,
                                "variant_rank": variant_rank,
                                "cell_attempts": 0,
                            },
                        )
                    )
                continue
            if primitive not in ready or primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES:
                admission_state = str(gate_records.get(primitive, {}).get("admission_state") or "")
                unavailable.append(
                    {
                        "environment": environment,
                        "primitive": primitive,
                        "reason": (
                            "runtime_only_requires_runtime_screen"
                            if primitive in QUALITY_SCREEN_PRIMITIVES
                            and primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES
                            else "diagnostic_not_quality_screen_routable"
                            if primitive not in QUALITY_SCREEN_PRIMITIVES
                            else "not_materialized"
                        ),
                        "admission_state": admission_state,
                        "work_order_path": str((gate.get("work_order_paths") or {}).get(primitive, ""))
                        if isinstance(gate.get("work_order_paths"), Mapping)
                        else "",
                    }
                )
                if (
                    primitive in QUALITY_SCREEN_PRIMITIVES
                    and primitive in _ADMISSION_PARAMETERS
                    and admission_state in {"hook_only_runtime_ready", "runtime_hook_template_present"}
                    and primitive not in attempted_admissions
                ):
                    candidates.append(
                        (
                            (-1, 0, primitive_rank, 0, f"{env_rank:04d}:{environment}"),
                            {
                                "kind": "materialization_admission",
                                "environment": environment,
                                "primitive": primitive,
                                "parameters": _ADMISSION_PARAMETERS[primitive],
                                "variant_rank": 0,
                                "cell_attempts": 0,
                            },
                        )
                    )
                continue
            variants = _PARAMETER_VARIANTS.get(primitive, ({},))
            cell_attempts = sum(1 for env, name, _ in attempted if env == environment and name == primitive)
            official_gate_failure_count = official_gate_failures.get((environment, primitive), 0)
            screen_negative_count = screen_negative_counts.get((environment, primitive), 0)
            failure_evidence_penalty = 2 * official_gate_failure_count + screen_negative_count
            experience_routing = _candidate_experience_routing(
                environment=environment,
                primitive=primitive,
                target_failures=target_failures,
                entries=experience_entries,
            )
            for variant_rank, parameters in enumerate(variants):
                signature = _signature(_runtime_parameters(parameters))
                if (environment, primitive, signature) in attempted:
                    continue
                if quality_discovery_only:
                    score = (
                        0,
                        0 if environment in event_semantic_unresolved else 1,
                        1 if environment in running_screen_environments else 0,
                        environment_attempts.get(environment, 0),
                        int(experience_routing["rank_band"]),
                        failure_evidence_penalty,
                        2 if cell_attempts else 1,
                        primitive_rank,
                        variant_rank,
                        f"{env_rank:04d}:{environment}",
                    )
                else:
                    score = (
                        0,
                        0 if environment in event_semantic_unresolved else 1,
                        int(experience_routing["rank_band"]),
                        failure_evidence_penalty,
                        2 if cell_attempts else 1,
                        primitive_rank,
                        variant_rank,
                        f"{env_rank:04d}:{environment}",
                    )
                candidates.append(
                    (
                        score,
                        {
                            "kind": "quality_screen",
                            "environment": environment,
                            "primitive": primitive,
                            "parameters": parameters,
                            "variant_rank": variant_rank,
                            "cell_attempts": cell_attempts,
                            "official_gate_failure_count": official_gate_failure_count,
                            "screen_negative_count": screen_negative_count,
                            "failure_evidence_penalty": failure_evidence_penalty,
                            "target_failure_signatures": list(target_failures),
                            "experience_routing": experience_routing,
                        },
                    )
                )

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_OUTPUT_EXISTS")
    if not candidates:
        return _write_replenishment_bundle(
            destination,
            {
                "schema_version": 1,
                "artifact_type": "wmloop-acwm-autoloop-replenishment",
                "state": "blocked",
                "reason": "NO_UNTRIED_READY_DIAGNOSIS_MATCHED_CANDIDATE",
                "unavailable_materialization_candidates": unavailable,
                "attempted_signature_count": len(attempted),
            },
        )

    _, selected = min(candidates, key=lambda item: item[0])
    environment = str(selected["environment"])
    primitive = str(selected["primitive"])
    parameters = dict(selected["parameters"])  # type: ignore[arg-type]
    if selected["kind"] == "materialization_admission":
        return _build_materialization_admission_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            seed=seed,
            materialization_gate=Path(materialization_gate),
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    if selected["kind"] in {"runtime_only_official_gate", "runtime_only_confirmation"}:
        return _build_runtime_only_official_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            seed=seed,
            unavailable=unavailable,
            attempted_signature_count=len(attempted_runtime_only),
        )
    if selected["kind"] == "checkpoint_delta_confirmation":
        return _build_checkpoint_delta_confirmation_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            eval_seed=seed,
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    if selected["kind"] == "checkpoint_delta_horizon_evidence":
        return _build_checkpoint_delta_horizon_evidence_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    if selected["kind"] == "checkpoint_delta_horizon_replication":
        return _build_checkpoint_delta_horizon_replication_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            trajectory_seed=seed,
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    if selected["kind"] == "confirmation_official_gate":
        return _build_confirmation_official_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    if selected["kind"] == "checkpoint_delta_recovery":
        return _build_checkpoint_delta_recovery_queue(
            selected=selected,
            destination=destination,
            report_root=Path(report_root),
            repo_root=Path(repo_root),
            runtime_python=Path(runtime_python),
            data_root=Path(data_root),
            checkpoint_root=Path(checkpoint_root),
            gpu=gpu,
            unavailable=unavailable,
            attempted_signature_count=len(attempted),
        )
    destination.mkdir(mode=0o700, parents=True)
    gap_plan_path = destination / "gap-plan.json"
    source_record = next(
        (
            dict(record)
            for record in records
            if isinstance(record, Mapping) and record.get("environment") == environment
        ),
        {},
    )
    primitive_targets = (
        tuple(registry.manifest(primitive).targets_failures)
        if registry is not None and primitive in registry.names()
        else ()
    )
    routed_failures = [
        str(value)
        for value in selected.get("target_failure_signatures", [])
        if isinstance(value, str) and value and value != "mixed"
        and (not primitive_targets or value in primitive_targets)
    ]
    if not routed_failures and primitive_targets:
        routed_failures = [str(primitive_targets[0])]
    if not routed_failures:
        routed_failures = ["targeted_exploration"]
    probe_candidates = [
        str(value)
        for value in source_record.get("diagnostic_probe_candidates", [])
        if isinstance(value, str) and value
    ]
    if not probe_candidates:
        probe_candidates = ["existing_frozen_failure_report"]
    mechanism_hypothesis = str(source_record.get("mechanism_hypothesis") or "").strip()
    if not mechanism_hypothesis:
        mechanism_hypothesis = (
            f"{primitive} may reduce {', '.join(routed_failures)} in {environment}; "
            "the frozen verdict protocol determines whether the intervention is retained."
        )
    _write_json(
        gap_plan_path,
        {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-targeted-gap-plan",
            "state": "ready",
            "claim_boundary": (
                "Routing-only intervention hypothesis; frozen diagnosis and verdict protocol "
                "remain authoritative."
            ),
            "selection": selected,
            "environment_records": [
                {
                    "environment": environment,
                    "evidence_level": "candidate_unstable",
                    "recommended_existing_primitives": [primitive],
                    "primitive_parameters": {primitive: parameters},
                    "routed_failure_families": routed_failures,
                    "diagnostic_probe_candidates": probe_candidates,
                    "mechanism_hypothesis": mechanism_hypothesis,
                }
            ],
        },
    )
    queue_root = Path(report_root).resolve() / (
        f"acwm-autoloop-queue-dynamic-{environment}-{primitive}-s{seed}-v{int(selected['variant_rank']) + 1}"
    )
    queue_manifest = build_autoloop_queue(
        gap_plan=gap_plan_path,
        materialization_gate=Path(materialization_gate),
        output_root=queue_root,
        repo_root=Path(repo_root),
        report_root=Path(report_root),
        runtime_python=Path(runtime_python),
        data_root=Path(data_root),
        checkpoint_root=Path(checkpoint_root),
        gpus=[gpu],
        seeds=[seed],
        train_batch_size=8,
        allow_repeat_cells=bool(selected["cell_attempts"]),
        include_discovered_positive_screens=False,
        failure_manifest=failure_manifest,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": seed,
        "selection": selected,
        "selection_source": "staging recommendations plus live failure-report-to-registry routing and horizon-response experience, intersected with runtime admission and untried parameter signatures",
        "gap_plan_path": str(gap_plan_path),
        "queue_manifest": queue_manifest,
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": len(attempted),
    }
    _write_json(destination / "manifest.json", payload)
    return payload


def _build_runtime_only_official_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    seed: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = str(selected["environment"])
    primitive = str(selected["primitive"])
    parameters = dict(selected["parameters"])  # type: ignore[arg-type]
    is_confirmation = selected["kind"] == "runtime_only_confirmation"
    campaign_id = _unique_campaign_id(
        report_root.resolve(),
        (
            f"acwm-autoloop-official-gate-{environment}-{primitive}-s{seed}-runtime-confirm-r1"
            if is_confirmation
            else f"acwm-autoloop-official-gate-{environment}-{primitive}-s{seed}-runtime-r1"
        ),
    )
    output_root = report_root.resolve() / campaign_id
    row = {
        "rank": 1,
        "phase": "official_eval_gate",
        "execution_mode": "runtime_only",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 0,
        "primitive_parameters": _runtime_parameters(parameters),
        "output_root": str(output_root),
        "candidate_gpus": [gpu],
        "requires_positive_manifest": "",
        "requires_ready_manifest": str(selected.get("source_manifest") or ""),
        "requires_official_quality_manifest": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root.resolve() / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(repo_root.resolve() / ".venv/bin/python3"),
            str(repo_root.resolve() / "scripts/export/acwm_runtime_only_screen.py"),
            "run",
            "--repo-root",
            str(repo_root.resolve()),
            "--output-root",
            str(output_root),
            "--runtime-python",
            str(runtime_python),
            "--data-root",
            str(data_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--dataset-freeze",
            str(repo_root.resolve() / "runs/m0/protocol/dataset-freeze.json"),
            "--heldout-protocol",
            str(repo_root.resolve() / "runs/m0/protocol/heldout-protocol.json"),
            "--environment",
            environment,
            "--primitive",
            primitive,
            "--parameters-json",
            json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            .replace("{", "{{")
            .replace("}", "}}"),
            "--gpu-index",
            "{gpu}",
            "--gpu-audit-manifest",
            "{gpu_audit_manifest}",
            "--seed",
            str(seed),
            "--max-trajs",
            "8",
            "--max-saved-vids",
            "8",
            "--hard-case-top-k",
            "4",
        ],
    }
    if is_confirmation:
        row.update(
            {
                "runtime_confirmation_of": str(selected["source_manifest"]),
                "runtime_confirmation_signature": str(selected["signature"]),
                "runtime_confirmation_required_passes": _RUNTIME_CONFIRMATION_REQUIRED_PASSES,
            }
        )
    queue_root = report_root.resolve() / (
        f"acwm-autoloop-queue-runtime-confirm-{environment}-{primitive}-s{seed}-r1"
        if is_confirmation
        else f"acwm-autoloop-queue-runtime-{environment}-{primitive}-s{seed}-v{int(selected['variant_rank']) + 1}"
    )
    queue = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "row_count": 1,
        "screen_row_count": 0,
        "official_gate_row_count": 1,
        "confirmation_row_count": 0,
        "confirmation_official_gate_row_count": 0,
        "preferred_gpus": [gpu],
        "rows": [row],
        "policy": {
            "runtime_only_rule": (
                "A runtime-only canary pass is confirmed only after the same parameter signature passes the official 50-step gate on at least two distinct eval seeds."
                if is_confirmation
                else "Materialize the inference hook, keep the official checkpoint fixed, and run the official 50-step gate with baseline-only hard-case selection."
            ),
            "runtime_confirmation_required_passes": _RUNTIME_CONFIRMATION_REQUIRED_PASSES,
        },
    }
    if queue_root.exists() or queue_root.is_symlink():
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_RUNTIME_QUEUE_EXISTS")
    queue_root.mkdir(mode=0o700, parents=True)
    _write_json(queue_root / "autoloop-queue.json", queue)
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": seed,
        "selection": dict(selected),
        "selection_source": (
            "single-seed runtime-only official pass promoted to independent-seed confirmation"
            if is_confirmation
            else "diagnosis-matched runtime-only primitive evaluated without checkpoint training"
        ),
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    _write_json(destination / "manifest.json", payload)
    return payload


def _build_confirmation_official_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    environment = str(selected["environment"])
    primitive = str(selected["primitive"])
    seed = int(selected["seed"])
    confirmation_output_root = Path(str(selected["confirmation_output_root"]))
    ready_manifest = str(selected["requires_ready_manifest"])
    row = _confirmation_official_gate_row(
        rank=1,
        environment=environment,
        primitive=primitive,
        seed=seed,
        confirmation_output_root=confirmation_output_root,
        report_root=report_root.resolve(),
        repo_root=repo_root.resolve(),
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        gpus=[gpu],
        requires_ready_manifest=ready_manifest,
    )
    queue_root = report_root.resolve() / (
        f"acwm-autoloop-queue-confirm-official-{environment}-{primitive}-s{seed}-r1"
    )
    if queue_root.exists() or queue_root.is_symlink():
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_CONFIRM_OFFICIAL_QUEUE_EXISTS")
    queue = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "row_count": 1,
        "screen_row_count": 0,
        "official_gate_row_count": 0,
        "confirmation_row_count": 0,
        "confirmation_official_gate_row_count": 1,
        "preferred_gpus": [gpu],
        "rows": [row],
        "policy": {
            "confirmation_rule": "Re-evaluate completed legacy confirmations with official ACWM eval.py and retain paired video plus checkpoint SHA. New queues use staged checkpoint gates."
        },
    }
    queue_root.mkdir(mode=0o700, parents=True)
    _write_json(queue_root / "autoloop-queue.json", queue)
    _write_json(
        queue_root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-autoloop-queue-manifest",
            "state": "ready",
            "row_count": 1,
            "queue_path": str(queue_root / "autoloop-queue.json"),
            "preferred_gpus": [gpu],
        },
    )
    destination.mkdir(mode=0o700, parents=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": seed,
        "selection": dict(selected),
        "selection_source": "completed legacy confirmation missing official 50-step re-evaluation",
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    _write_json(destination / "manifest.json", payload)
    return payload


def _build_checkpoint_delta_recovery_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    environment = str(selected["environment"])
    source_primitive = str(selected["source_primitive"])
    seed = int(selected["seed"])
    baseline_checkpoint = Path(str(selected["baseline_checkpoint"])).resolve()
    candidate_checkpoint = Path(str(selected["candidate_checkpoint"])).resolve()
    candidate_runtime_root = Path(str(selected["candidate_runtime_root"])).resolve()
    source_gate_manifest = Path(str(selected["source_official_gate_manifest"])).resolve()
    pending_alphas = selected.get("pending_alphas")
    if not isinstance(pending_alphas, list) or not pending_alphas:
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_PENDING_ALPHAS_INVALID")
    normalized_pending_alphas = [float(alpha) for alpha in pending_alphas]
    scaling_prefix = f"acwm-checkpoint-delta-scaling-{environment}-{source_primitive}-s{seed}"
    existing_scaling_manifests = sorted(
        report_root.resolve().glob(f"{scaling_prefix}-r*/manifest.json")
    )
    scaling_manifest: Path | None = None
    scaling: dict[str, object] = {}
    for candidate_manifest in reversed(existing_scaling_manifests):
        candidate_scaling = _load_optional_json(candidate_manifest)
        candidate_outputs = candidate_scaling.get("outputs")
        available_alphas = {
            round(float(record["alpha"]), 12)
            for record in candidate_outputs
            if isinstance(candidate_outputs, list)
            and isinstance(record, Mapping)
            and isinstance(record.get("alpha"), (int, float))
            and not isinstance(record.get("alpha"), bool)
        } if isinstance(candidate_outputs, list) else set()
        if all(round(alpha, 12) in available_alphas for alpha in normalized_pending_alphas):
            scaling_manifest = candidate_manifest
            scaling = candidate_scaling
            break
    if scaling_manifest is None:
        revisions = []
        for candidate_manifest in existing_scaling_manifests:
            match = re.search(r"-r(\d+)$", candidate_manifest.parent.name)
            if match is not None:
                revisions.append(int(match.group(1)))
        scaling_root = report_root.resolve() / f"{scaling_prefix}-r{max(revisions, default=0) + 1}"
        scaling_manifest = scaling_root / "manifest.json"
        if scaling_root.exists() or scaling_root.is_symlink():
            raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_SCALING_OUTPUT_INVALID")
        command = [
            str(runtime_python),
            str(repo_root.resolve() / "scripts/export/acwm_checkpoint_delta_scaling.py"),
            "--output-root",
            str(scaling_root),
            "--baseline-checkpoint",
            str(baseline_checkpoint),
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--environment",
            environment,
            "--source-primitive",
            source_primitive,
            "--seed",
            str(seed),
        ]
        for alpha in normalized_pending_alphas:
            command.extend(("--alpha", str(alpha)))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=repo_root.resolve(),
        )
        if completed.returncode != 0:
            raise AcwmAutoloopReplenisherError(
                "ACWM_REPLENISH_DELTA_SCALING_FAILED:"
                f"{completed.returncode}:{completed.stderr[-1000:]}"
            )
        scaling = _load_json_object(scaling_manifest)
    if (
        scaling.get("artifact_type") != "wmloop-checkpoint-delta-scaling"
        or scaling.get("state") != "ready"
        or scaling.get("environment") != environment
        or scaling.get("source_primitive") != source_primitive
        or scaling.get("seed") != seed
        or Path(str(scaling.get("baseline_checkpoint") or "")).resolve() != baseline_checkpoint
        or Path(str(scaling.get("candidate_checkpoint") or "")).resolve() != candidate_checkpoint
    ):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_SCALING_MANIFEST_INVALID")
    outputs = scaling.get("outputs")
    if not isinstance(outputs, list):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_SCALING_OUTPUTS_INVALID")
    outputs_by_alpha = {
        round(float(record["alpha"]), 12): record
        for record in outputs
        if isinstance(record, Mapping)
        and isinstance(record.get("alpha"), (int, float))
        and not isinstance(record.get("alpha"), bool)
    }
    rows: list[dict[str, object]] = []
    for rank, alpha in enumerate(normalized_pending_alphas, start=1):
        output = outputs_by_alpha.get(round(alpha, 12))
        if not isinstance(output, Mapping):
            raise AcwmAutoloopReplenisherError(
                f"ACWM_REPLENISH_DELTA_SCALING_ALPHA_MISSING:{alpha}"
            )
        scaled_checkpoint = Path(str(output.get("path") or "")).resolve()
        if not scaled_checkpoint.is_file():
            raise AcwmAutoloopReplenisherError(
                f"ACWM_REPLENISH_DELTA_SCALING_CHECKPOINT_MISSING:{scaled_checkpoint}"
            )
        alpha_token = _checkpoint_delta_alpha_token(alpha)
        campaign_id = _unique_campaign_id(
            report_root.resolve(),
            (
                f"acwm-autoloop-official-gate-{environment}-checkpoint_delta_scaling-"
                f"s{seed}-alpha{alpha_token}-r1"
            ),
        )
        output_root = report_root.resolve() / campaign_id
        rows.append(
            {
                "rank": rank,
                "phase": "official_eval_gate",
                "campaign_id": campaign_id,
                "environment": environment,
                "primitive": "checkpoint_delta_scaling",
                "source_primitive": source_primitive,
                "checkpoint_delta_alpha": alpha,
                "seed": seed,
                "checkpoint_step": 512,
                "train_steps": 0,
                "output_root": str(output_root),
                "candidate_gpus": [gpu],
                "allow_any_idle_gpu": True,
                "requires_positive_manifest": "",
                "requires_ready_manifest": str(scaling_manifest),
                "requires_official_quality_manifest": "",
                "source_official_gate_manifest": str(source_gate_manifest),
                "source_screen_manifest": str(selected["source_screen_manifest"]),
                "archive_db": str(DEFAULT_ARCHIVE_DB),
                "cas_root": str(DEFAULT_CAS_ROOT),
                "gpu_audit_root_template": str(
                    report_root.resolve()
                    / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
                ),
                "launch_argv_template": [
                    str(repo_root.resolve() / ".venv/bin/python3"),
                    str(repo_root.resolve() / "scripts/export/acwm_formal_visualization.py"),
                    "--output-root",
                    str(output_root),
                    "--environment",
                    environment,
                    "--primitive",
                    "checkpoint_delta_scaling",
                    "--source-primitive",
                    source_primitive,
                    "--checkpoint-delta-alpha",
                    str(alpha),
                    "--checkpoint-transform-manifest",
                    str(scaling_manifest),
                    "--source-official-gate-manifest",
                    str(source_gate_manifest),
                    "--seed",
                    str(seed),
                    "--runtime-python",
                    str(runtime_python),
                    "--data-root",
                    str(data_root),
                    "--checkpoint-root",
                    str(checkpoint_root),
                    "--dataset-freeze",
                    str(repo_root.resolve() / "runs/m0/protocol/dataset-freeze.json"),
                    "--heldout-protocol",
                    str(repo_root.resolve() / "runs/m0/protocol/heldout-protocol.json"),
                    "--candidate-checkpoint",
                    str(scaled_checkpoint),
                    "--candidate-runtime-root",
                    str(candidate_runtime_root),
                    "--gpu-index",
                    "{gpu}",
                    "--steps",
                    "50",
                    "--split",
                    "ind_test",
                    "--max-trajs",
                    "3",
                    "--max-saved-vids",
                    "3",
                    "--batch-size",
                    "1",
                    "--num-workers",
                    "2",
                    "--test-cuts",
                    "1",
                    "--hard-case-top-k",
                    "1",
                ],
            }
        )

    queue_name = _unique_campaign_id(
        report_root.resolve(),
        f"acwm-autoloop-queue-delta-recovery-{environment}-{source_primitive}-s{seed}-r1",
    )
    queue_root = report_root.resolve() / queue_name
    queue = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "row_count": len(rows),
        "screen_row_count": 0,
        "official_gate_row_count": len(rows),
        "confirmation_row_count": 0,
        "confirmation_official_gate_row_count": 0,
        "preferred_gpus": [gpu],
        "rows": rows,
        "policy": {
            "trigger": "internal 512 proxy positive but official 50-step quality gate failed",
            "recovery_rule": "Evaluate signed checkpoint-update scaling at alpha -0.02/-0.05/0.02/0.05/0.10/0.25/0.50; negative alpha reflects a verifier-identified harmful update direction.",
            "claim_boundary": "Checkpoint transformation is not a quality claim; every alpha is independently evaluated by the frozen official gate.",
        },
    }
    queue_root.mkdir(mode=0o700, parents=True)
    _write_json(queue_root / "autoloop-queue.json", queue)
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": seed,
        "selection": dict(selected),
        "selection_source": "automatic recovery from internal-proxy positive and official-gate regression",
        "checkpoint_transform_manifest": str(scaling_manifest),
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "manifest.json", payload)
    return payload


def _build_checkpoint_delta_confirmation_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    eval_seed: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    environment = str(selected["environment"])
    source_primitive = str(selected["source_primitive"])
    alpha = float(selected["alpha"])
    signature = str(selected["signature"])
    attempted_seeds = {
        int(seed)
        for seed in selected.get("attempt_seeds", [])
        if isinstance(seed, int) and not isinstance(seed, bool)
    }
    while eval_seed in attempted_seeds:
        eval_seed += 1
    source_pass_manifest = Path(str(selected["source_pass_manifest"])).resolve()
    source_pass = _load_json_object(source_pass_manifest)
    provenance = source_pass.get("checkpoint_transform_provenance")
    gate = source_pass.get("official_quality_gate")
    if (
        source_pass.get("state") != "ready"
        or source_pass.get("primitive") != "checkpoint_delta_scaling"
        or not isinstance(provenance, Mapping)
        or provenance.get("state") != "verified"
        or not isinstance(gate, Mapping)
        or gate.get("pass") is not True
    ):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_CONFIRM_SOURCE_INVALID")
    candidate_checkpoint = Path(str(source_pass.get("candidate_checkpoint") or "")).resolve()
    candidate_runtime_root = Path(str(source_pass.get("candidate_runtime_root") or "")).resolve()
    transform_manifest = Path(str(provenance.get("transform_manifest_path") or "")).resolve()
    source_official_gate_manifest = Path(
        str(provenance.get("source_official_gate_manifest_path") or "")
    ).resolve()
    if (
        not candidate_checkpoint.is_file()
        or not candidate_runtime_root.is_dir()
        or not transform_manifest.is_file()
        or not source_official_gate_manifest.is_file()
    ):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_CONFIRM_EVIDENCE_MISSING")

    alpha_token = _checkpoint_delta_alpha_token(alpha)
    campaign_id = _unique_campaign_id(
        report_root.resolve(),
        (
            f"acwm-autoloop-official-gate-{environment}-checkpoint_delta_scaling-"
            f"confirm-s{eval_seed}-alpha{alpha_token}-r1"
        ),
    )
    output_root = report_root.resolve() / campaign_id
    row = {
        "rank": 1,
        "phase": "official_eval_gate",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": "checkpoint_delta_scaling",
        "source_primitive": source_primitive,
        "checkpoint_delta_alpha": alpha,
        "checkpoint_delta_confirmation_signature": signature,
        "checkpoint_delta_confirmation_of": str(source_pass_manifest),
        "checkpoint_delta_confirmation_required_passes": _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES,
        "seed": eval_seed,
        "checkpoint_step": 512,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": [gpu],
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifest": str(source_pass_manifest),
        "requires_official_quality_manifest": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(
            report_root.resolve()
            / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(repo_root.resolve() / ".venv/bin/python3"),
            str(repo_root.resolve() / "scripts/export/acwm_formal_visualization.py"),
            "--output-root",
            str(output_root),
            "--environment",
            environment,
            "--primitive",
            "checkpoint_delta_scaling",
            "--source-primitive",
            source_primitive,
            "--checkpoint-delta-alpha",
            str(alpha),
            "--checkpoint-transform-manifest",
            str(transform_manifest),
            "--source-official-gate-manifest",
            str(source_official_gate_manifest),
            "--seed",
            str(eval_seed),
            "--runtime-python",
            str(runtime_python),
            "--data-root",
            str(data_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--dataset-freeze",
            str(repo_root.resolve() / "runs/m0/protocol/dataset-freeze.json"),
            "--heldout-protocol",
            str(repo_root.resolve() / "runs/m0/protocol/heldout-protocol.json"),
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--candidate-runtime-root",
            str(candidate_runtime_root),
            "--gpu-index",
            "{gpu}",
            "--steps",
            "50",
            "--split",
            "ind_test",
            "--max-trajs",
            "3",
            "--max-saved-vids",
            "3",
            "--batch-size",
            "1",
            "--num-workers",
            "2",
            "--test-cuts",
            "1",
            "--hard-case-top-k",
            "1",
        ],
    }
    queue_name = _unique_campaign_id(
        report_root.resolve(),
        (
            f"acwm-autoloop-queue-delta-confirm-{environment}-{source_primitive}-"
            f"s{eval_seed}-alpha{alpha_token}-r1"
        ),
    )
    queue_root = report_root.resolve() / queue_name
    queue_root.mkdir(mode=0o700, parents=True)
    _write_json(
        queue_root / "autoloop-queue.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-autoloop-queue",
            "state": "ready",
            "row_count": 1,
            "screen_row_count": 0,
            "official_gate_row_count": 1,
            "confirmation_row_count": 0,
            "confirmation_official_gate_row_count": 0,
            "preferred_gpus": [gpu],
            "rows": [row],
            "policy": {
                "confirmation_rule": "A verifier-tuned signed checkpoint transform requires the same scaled checkpoint to pass the frozen official gate on at least two distinct eval seeds.",
                "required_passes": _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES,
            },
        },
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": eval_seed,
        "selection": dict(selected),
        "selection_source": "single-seed signed checkpoint-transform pass promoted to independent eval-seed confirmation",
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "manifest.json", payload)
    return payload


def _checkpoint_delta_alpha_token(alpha: float) -> str:
    return (
        f"m{int(round(abs(alpha) * 100)):03d}"
        if alpha < 0.0
        else f"{int(round(alpha * 100)):03d}"
    )


def _build_checkpoint_delta_horizon_evidence_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    environment = str(selected["environment"])
    source_primitive = str(selected["source_primitive"])
    alpha = float(selected["alpha"])
    pass_manifests = [
        Path(value).resolve()
        for value in selected.get("pass_manifests", [])
        if isinstance(value, str) and value
    ]
    horizons = _HORIZONS_BY_ENVIRONMENT.get(environment)
    if len(pass_manifests) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_HORIZON_PASSES_MISSING")
    if horizons is None:
        raise AcwmAutoloopReplenisherError(
            f"ACWM_REPLENISH_DELTA_HORIZON_LADDER_MISSING:{environment}"
        )
    revision = _next_checkpoint_delta_horizon_revision(
        report_root=report_root.resolve(),
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        max_horizon=max(horizons),
    )
    alpha_token = str(alpha).replace("-", "m").replace(".", "p")
    queue_root = report_root.resolve() / (
        f"acwm-confirmed-delta-horizon-queue-{environment}-{source_primitive}-"
        f"alpha{alpha_token}-h{max(horizons)}-r{revision}"
    )
    queue_manifest = build_confirmed_delta_horizon_queue(
        pass_manifests=pass_manifests,
        output_root=queue_root,
        report_root=report_root.resolve(),
        repo_root=repo_root.resolve(),
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        gpus=[gpu],
        horizons=horizons,
        mode="autoregressive",
        num_inference_steps=50,
        max_trajectories=1,
        revision=revision,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "selection": dict(selected),
        "selection_source": (
            "multi-seed checkpoint-delta confirmation promoted to environment-specific "
            "long-horizon and event evidence"
        ),
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "queue_manifest": queue_manifest,
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    return _write_replenishment_bundle(destination, payload)


def _build_checkpoint_delta_horizon_replication_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    trajectory_seed: int,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    environment = str(selected["environment"])
    source_primitive = str(selected["source_primitive"])
    alpha = float(selected["alpha"])
    pass_manifests = [
        Path(value).resolve()
        for value in selected.get("pass_manifests", [])
        if isinstance(value, str) and value
    ]
    horizons = tuple(
        int(value)
        for value in selected.get("horizons", [])
        if isinstance(value, int) and not isinstance(value, bool)
    )
    if len(pass_manifests) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_REPLICATION_PASSES_MISSING")
    if not horizons:
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_DELTA_REPLICATION_LADDER_MISSING")
    revision = _next_checkpoint_delta_horizon_revision(
        report_root=report_root.resolve(),
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        max_horizon=max(horizons),
    )
    alpha_token = str(alpha).replace("-", "m").replace(".", "p")
    queue_root = report_root.resolve() / (
        f"acwm-confirmed-delta-horizon-queue-{environment}-{source_primitive}-"
        f"alpha{alpha_token}-h{max(horizons)}-r{revision}"
    )
    queue_manifest = build_confirmed_delta_horizon_queue(
        pass_manifests=pass_manifests,
        output_root=queue_root,
        report_root=report_root.resolve(),
        repo_root=repo_root.resolve(),
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        gpus=[gpu],
        horizons=horizons,
        mode="autoregressive",
        num_inference_steps=50,
        max_trajectories=_CHECKPOINT_DELTA_REPLICATION_TRAJECTORIES,
        trajectory_seed=trajectory_seed,
        factorized_replication=True,
        revision=revision,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "selection": dict(selected),
        "selection_source": (
            "aggregate long-horizon positive promoted to independent-seed multi-trajectory "
            "held-out replication"
        ),
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "queue_manifest": queue_manifest,
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    return _write_replenishment_bundle(destination, payload)


def _next_checkpoint_delta_horizon_revision(
    *,
    report_root: Path,
    environment: str,
    source_primitive: str,
    alpha: float,
    max_horizon: int,
) -> int:
    alpha_token = str(alpha).replace("-", "m").replace(".", "p")
    prefixes = (
        (
            f"acwm-long-horizon-baseline-{environment}-checkpoint_delta_scaling-"
            f"{source_primitive}-alpha{alpha_token}-h{max_horizon}-r"
        ),
        (
            f"acwm-long-horizon-candidate-{environment}-checkpoint_delta_scaling-"
            f"{source_primitive}-alpha{alpha_token}-h{max_horizon}-r"
        ),
        (
            f"acwm-confirmed-delta-horizon-queue-{environment}-{source_primitive}-"
            f"alpha{alpha_token}-h{max_horizon}-r"
        ),
    )
    revisions: list[int] = []
    for prefix in prefixes:
        for path in report_root.glob(f"{prefix}*"):
            suffix = path.name[len(prefix) :]
            if suffix.isdigit():
                revisions.append(int(suffix))
    return max(revisions, default=0) + 1


def _build_materialization_admission_queue(
    *,
    selected: Mapping[str, object],
    destination: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu: int,
    seed: int,
    materialization_gate: Path,
    unavailable: list[dict[str, object]],
    attempted_signature_count: int,
) -> dict[str, object]:
    primitive = str(selected["primitive"])
    environment = str(selected["environment"])
    parameters = dict(selected["parameters"])  # type: ignore[arg-type]
    campaign_id = f"acwm-autoloop-materialize-{primitive}-s{seed}-r1"
    admission_root = report_root.resolve() / f"primitive-runtime-smoke-{primitive.replace('_', '-')}-dynamic-s{seed}-r1"
    queue_root = report_root.resolve() / f"acwm-autoloop-queue-materialize-{primitive}-s{seed}-r1"
    row = {
        "rank": 1,
        "phase": "materialize_runtime_admission",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "train_steps": 1,
        "primitive_parameters": parameters,
        "output_root": str(admission_root),
        "candidate_gpus": [gpu],
        "requires_positive_manifest": "",
        "requires_official_quality_manifest": "",
        "requires_positive_output_root": "",
        "archive_db": str(DEFAULT_ARCHIVE_DB),
        "cas_root": str(DEFAULT_CAS_ROOT),
        "gpu_audit_root_template": str(report_root.resolve() / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"),
        "launch_argv_template": [
            str(repo_root.resolve() / ".venv/bin/python3"),
            "-m",
            "wmloop.execute.primitive_runtime_smoke",
            "run",
            "--repo-root",
            str(repo_root.resolve()),
            "--output-root",
            str(admission_root),
            "--runtime-python",
            str(runtime_python),
            "--data-root",
            str(data_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--primitive",
            primitive,
            "--gpu-index",
            "{gpu}",
            *_admission_parameter_argv(primitive, parameters),
            "--run-training",
            "--gpu-exclusivity-audit-manifest",
            "{gpu_audit_manifest}",
            "--gpu-exclusivity-max-age-seconds",
            "3600",
            "--training-timeout-seconds",
            "1800",
            "--archive-db",
            str(DEFAULT_ARCHIVE_DB),
            "--cas-root",
            str(DEFAULT_CAS_ROOT),
        ],
    }
    queue = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "row_count": 1,
        "screen_row_count": 0,
        "official_gate_row_count": 0,
        "confirmation_row_count": 0,
        "preferred_gpus": [gpu],
        "rows": [row],
        "policy": {
            "materialization_rule": "Run one-step GPU admission before a hook-only primitive may enter quality screening."
        },
    }
    if queue_root.exists() or queue_root.is_symlink():
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_ADMISSION_QUEUE_EXISTS")
    queue_root.mkdir(mode=0o700, parents=True)
    _write_json(queue_root / "autoloop-queue.json", queue)
    _write_json(
        queue_root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-autoloop-queue-manifest",
            "state": "ready",
            "row_count": 1,
            "queue_path": str(queue_root / "autoloop-queue.json"),
            "preferred_gpus": [gpu],
        },
    )
    destination.mkdir(mode=0o700, parents=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-replenishment",
        "state": "ready",
        "gpu": gpu,
        "seed": seed,
        "selection": dict(selected),
        "selection_source": "diagnosed hook-only primitive requiring GPU runtime admission before quality screening",
        "materialization_gate": str(materialization_gate),
        "queue_path": str(queue_root / "autoloop-queue.json"),
        "unavailable_materialization_candidates": unavailable,
        "attempted_signature_count": attempted_signature_count,
    }
    _write_json(destination / "manifest.json", payload)
    return payload


def _admission_parameter_argv(primitive: str, parameters: Mapping[str, object]) -> list[str]:
    flags = {
        "action_dimension_balancing": (
            ("action_balance_blend", "--action-balance-blend"),
            ("action_balance_max_gain", "--action-balance-max-gain"),
        ),
        "dino_rep_injection": (("weight", "--weight"),),
        "event_window_reweight": (
            ("event_weight", "--event-weight"),
            ("event_quantile", "--event-quantile"),
            ("event_visual_blend", "--event-visual-blend"),
        ),
        "motion_region_reweight": (("weight", "--weight"),),
        "first_frame_anchor": (("anchor_every", "--anchor-every"), ("anchor_weight", "--anchor-weight")),
        "cfg_guidance_schedule": (("guidance_start", "--guidance-start"), ("guidance_end", "--guidance-end")),
        "wmsd_self_distill": (
            ("wmsd_teacher_ema", "--wmsd-teacher-ema"),
            ("wmsd_steps", "--wmsd-steps"),
            ("wmsd_lr", "--wmsd-lr"),
        ),
    }
    if primitive not in flags:
        raise AcwmAutoloopReplenisherError(f"ACWM_REPLENISH_ADMISSION_PARAMETERS_UNSUPPORTED:{primitive}")
    output: list[str] = []
    for key, flag in flags[primitive]:
        output.extend((flag, str(parameters[key])))
    return output


def _attempted_admissions(report_root: Path) -> set[str]:
    attempted: set[str] = set()
    for path in report_root.glob("acwm-autoloop-queue-*/autoloop-queue.json"):
        queue = _load_optional_json(path)
        for row in queue.get("rows", []):
            if isinstance(row, Mapping) and row.get("phase") == "materialize_runtime_admission":
                attempted.add(str(row.get("primitive") or ""))
    return attempted


def _pending_confirmation_official_candidates(report_root: Path) -> list[dict[str, object]]:
    attempted_roots: set[str] = set()
    for path in report_root.glob("acwm-autoloop-queue-*/autoloop-queue.json"):
        queue = _load_optional_json(path)
        for row in queue.get("rows", []):
            if isinstance(row, Mapping) and row.get("phase") == "confirm_official_eval_gate":
                value = row.get("source_confirmation_output_root")
                if isinstance(value, str) and value:
                    attempted_roots.add(str(Path(value).resolve()))
    candidates: list[dict[str, object]] = []
    for root in sorted(report_root.glob("acwm-autoloop-confirm-*-t*-r*")):
        match = _CONFIRM_NAME.fullmatch(root.name)
        if match is None or str(root.resolve()) in attempted_roots:
            continue
        environment, primitive, seed_raw, steps_raw = match.groups()
        if environment not in {
            "cloth_move", "pour_water", "push_cube", "push_rope",
            "push_sand", "reacher", "robot_arm", "stack_cube",
        }:
            continue
        ready_manifest = root / "envs" / environment / "manifest.json"
        manifest = _load_optional_json(ready_manifest)
        checkpoint = root / "envs" / environment / "retained_training" / "latest.pt"
        if manifest.get("state") != "ready" or not checkpoint.is_file():
            continue
        candidates.append(
            {
                "kind": "confirmation_official_gate",
                "environment": environment,
                "primitive": primitive,
                "seed": int(seed_raw),
                "confirmation_steps": int(steps_raw),
                "confirmation_output_root": str(root.resolve()),
                "requires_ready_manifest": str(ready_manifest.resolve()),
                "parameters": {},
                "variant_rank": 0,
                "cell_attempts": 0,
            }
        )
    return candidates


def _load_registry_if_present(repo_root: Path) -> PrimitiveRegistry | None:
    if not (repo_root / "wmloop/primitives/definitions").is_dir():
        return None
    return PrimitiveRegistry.from_root(repo_root)


def _primitive_matches_failures(
    *, primitive: str, failures: Sequence[str], registry: PrimitiveRegistry | None
) -> bool:
    if registry is None or not failures:
        return True
    if primitive not in registry.names():
        return False
    targets = set(registry.manifest(primitive).targets_failures)
    return bool(targets.intersection(failures))


def _diagnosis_matched_primitive_names(
    *,
    environment: str,
    registry: PrimitiveRegistry | None,
    report_root: Path,
    failure_manifest: Path | None = None,
) -> tuple[str, ...]:
    if registry is None:
        return ()
    failures = _diagnosed_failure_names(
        environment=environment,
        registry=registry,
        report_root=report_root,
        failure_manifest=failure_manifest,
    )
    names: list[str] = []
    for failure_name in failures:
        for manifest in registry.available_for(failure=failure_name):
            if manifest.name not in names:
                names.append(manifest.name)
    return tuple(names)


def _diagnosed_failure_names(
    *,
    environment: str,
    registry: PrimitiveRegistry | None,
    report_root: Path,
    failure_manifest: Path | None = None,
) -> tuple[str, ...]:
    manifest = failure_manifest or _current_failure_manifest(report_root)
    failure = _load_optional_json(manifest.parent / "failure_reports" / f"{environment}.json")
    dominant = str(failure.get("dominant_failure") or "")
    failures: list[str] = []
    if dominant and dominant != "mixed":
        failures.append(dominant)
    for key in ("dominant_failure_candidates", "routed_failure_families"):
        values = failure.get(key)
        if isinstance(values, list):
            failures.extend(str(value) for value in values if isinstance(value, str) and value != "mixed")
    if environment == "pour_water":
        failures.extend(_pour_water_event_failure_names(report_root))
    if not failures and dominant == "mixed" and registry is not None:
        failures.extend(
            sorted(
                {
                    failure_name
                    for name in registry.names()
                    for failure_name in registry.manifest(name).targets_failures
                    if failure_name != "mixed"
                }
            )
        )
    return tuple(dict.fromkeys(failures))


def _staged_diagnostic_failure_names(
    *,
    plan: Mapping[str, object],
    record: Mapping[str, object],
) -> tuple[str, ...]:
    """Admit diagnostic routing to proposals without changing verdict evidence."""

    if (
        plan.get("artifact_type") != "wmloop-acwm-targeted-gap-plan"
        or plan.get("state") != "ready"
        or not isinstance(plan.get("claim_boundary"), str)
        or not str(plan["claim_boundary"]).strip()
    ):
        return ()
    probes = record.get("diagnostic_probe_candidates")
    hypothesis = record.get("mechanism_hypothesis")
    routed = record.get("routed_failure_families")
    if (
        not isinstance(probes, list)
        or not any(isinstance(value, str) and value.strip() for value in probes)
        or not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or not isinstance(routed, list)
    ):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in routed
            if isinstance(value, str) and value.strip() and value != "mixed"
        )
    )


def _pour_water_event_failure_names(report_root: Path) -> tuple[str, ...]:
    candidates = sorted(
        report_root.glob("acwm-pour-water-event-gate-*/event-gate.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
        reverse=True,
    )
    for path in candidates:
        report = _load_optional_json(path)
        routing = report.get("diagnostic_routing")
        if (
            report.get("artifact_type") != "wmloop-acwm-pour-water-event-gate"
            or report.get("state") != "ready"
            or report.get("classification") not in {"event_failure", "event_regression"}
            or not isinstance(routing, Mapping)
        ):
            continue
        values = routing.get("routed_failure_families")
        if isinstance(values, list):
            return tuple(str(value) for value in values if isinstance(value, str) and value and value != "mixed")
    return ()


def _current_failure_manifest(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("m1-raw-failure-reports-ladder-r*/manifest.json"),
        key=lambda path: (path.stat().st_mtime, path.parent.name),
        reverse=True,
    )
    for path in candidates:
        manifest = _load_optional_json(path)
        if (
            manifest.get("artifact_type") == "wmloop-m1-raw-failure-report-batch"
            and manifest.get("state") in {"ready", "partial"}
            and int(manifest.get("report_count") or 0) > 0
        ):
            return path
    return report_root / "m1-raw-failure-reports-ladder-r1" / "manifest.json"


def _load_horizon_experience_entries(report_root: Path) -> tuple[dict[str, object], ...]:
    entries: list[dict[str, object]] = []
    for path in sorted(report_root.glob("acwm-horizon-effect-profile-*/horizon-effect-profile.json")):
        profile = _load_optional_json(path)
        classification = profile.get("effect_classification")
        transfer = profile.get("transfer_prior")
        if (
            profile.get("artifact_type") != "wmloop-acwm-horizon-effect-profile"
            or profile.get("state") != "ready"
            or not isinstance(classification, Mapping)
            or not isinstance(transfer, Mapping)
        ):
            continue
        environment = profile.get("environment")
        primitive = profile.get("primitive")
        scope = classification.get("effect_scope")
        signatures = transfer.get("failure_signatures")
        if (
            not isinstance(environment, str)
            or not environment
            or not isinstance(primitive, str)
            or not primitive
            or not isinstance(scope, str)
            or not scope
            or not isinstance(signatures, list)
        ):
            continue
        entries.append(
            {
                "environment": environment,
                "primitive": primitive,
                "effect_scope": scope,
                "failure_signatures": [value for value in signatures if isinstance(value, str) and value],
                "causal_credit_eligible": transfer.get("causal_credit_eligible") is True,
                "profile_path": str(path.resolve()),
            }
        )
    return tuple(entries)


def _candidate_experience_routing(
    *,
    environment: str,
    primitive: str,
    target_failures: Sequence[str],
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    target_set = set(target_failures)
    matched = []
    for entry in entries:
        if entry.get("primitive") != primitive:
            continue
        source_environment = str(entry.get("environment") or "")
        signatures = {
            value
            for value in entry.get("failure_signatures", [])
            if isinstance(value, str) and value
        }
        if source_environment != environment and (not target_set or not target_set.intersection(signatures)):
            continue
        matched.append(dict(entry))

    if not matched:
        return {
            "rank_band": 3,
            "decision": "explore_without_horizon_experience",
            "matched_profile_count": 0,
            "matched_profiles": [],
        }

    long_positive = [
        entry for entry in matched if entry.get("effect_scope") == "aggregate_long_horizon_positive"
    ]
    causal_positive = [entry for entry in long_positive if entry.get("causal_credit_eligible") is True]
    same_environment_positive = [entry for entry in long_positive if entry.get("environment") == environment]
    scopes = {str(entry.get("effect_scope") or "") for entry in matched}
    if causal_positive:
        rank_band, decision = 0, "prioritize_verified_causal_long_horizon_route"
    elif same_environment_positive:
        rank_band, decision = 1, "prioritize_same_environment_long_horizon_route"
    elif long_positive:
        rank_band, decision = 2, "prioritize_cross_environment_mechanism_transfer"
    elif "aggregate_negative_or_mixed" in scopes:
        rank_band, decision = 6, "demote_aggregate_negative_route"
    elif "short_horizon_only_positive" in scopes:
        rank_band, decision = 5, "demote_short_horizon_only_route"
    elif "hard_case_only_positive" in scopes:
        rank_band, decision = 4, "retain_as_hard_case_prior_without_general_uplift"
    else:
        rank_band, decision = 6, "demote_unknown_nonpositive_route"
    return {
        "rank_band": rank_band,
        "decision": decision,
        "matched_profile_count": len(matched),
        "matched_effect_scopes": sorted(scopes),
        "matched_profiles": [str(entry["profile_path"]) for entry in matched],
        "claim_boundary": "Horizon experience changes proposal ordering only; the target run still requires its own screen and official gates.",
    }


def _materialization_records(gate: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    report_path = gate.get("report_path")
    if not isinstance(report_path, str):
        return {}
    report = _load_optional_json(Path(report_path))
    records = report.get("records")
    if not isinstance(records, list):
        return {}
    return {
        str(record["primitive"]): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("primitive"), str)
    }


def _refresh_materialization_gate(*, configured_gate: Path, report_root: Path, repo_root: Path) -> Path:
    configured = configured_gate.resolve(strict=True)
    dynamic = list(report_root.glob("primitive-materialization-gate-dynamic-r*/manifest.json"))
    candidates = {configured, *(path.resolve() for path in dynamic)}
    current = max(
        candidates,
        key=lambda path: (
            _dynamic_gate_revision(path),
            path.stat().st_mtime_ns,
            path == configured,
        ),
    )
    gate = _load_json_object(current)
    report_path = gate.get("report_path")
    if not isinstance(report_path, str):
        return current
    report = _load_json_object(Path(report_path))
    evidence = [Path(value) for value in report.get("evidence_manifest_paths", []) if isinstance(value, str)]
    known = {path.resolve() for path in evidence if path.exists()}
    additions = []
    for path in report_root.glob("primitive-runtime-smoke-*/manifest.json"):
        manifest = _load_optional_json(path)
        if manifest.get("state") in {"ready", "hook_only"} and path.resolve() not in known:
            additions.append(path.resolve())
    if not additions:
        return current
    apply_manifest = report.get("primitive_apply_manifest")
    if not isinstance(apply_manifest, str):
        raise AcwmAutoloopReplenisherError("ACWM_REPLENISH_MATERIALIZATION_APPLY_MANIFEST_MISSING")
    next_index = max((_dynamic_gate_revision(path) for path in candidates), default=0) + 1
    output = report_root / f"primitive-materialization-gate-dynamic-r{next_index}"
    run_primitive_materialization_gate(
        repo_root=repo_root,
        primitive_apply_manifest=Path(apply_manifest),
        output_root=output,
        evidence_manifests=[*evidence, *sorted(additions)],
        archive_db=DEFAULT_ARCHIVE_DB,
        cas_root=DEFAULT_CAS_ROOT,
    )
    return output / "manifest.json"


def _dynamic_gate_revision(path: Path) -> int:
    match = _DYNAMIC_GATE_NAME.fullmatch(path.parent.name)
    if match is None:
        return -1
    return int(match.group(1))


def _officially_positive_environments(report_root: Path) -> set[str]:
    return {
        str(record["environment"])
        for record in _training_finalization_records(report_root)
        if record["confirmation_passed"] is True
    }


def _event_semantic_positive_methods(report_root: Path) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {}
    for path in sorted(report_root.glob("acwm-pour-water-event-gate-*/manifest.json")):
        manifest = _load_optional_json(path)
        if manifest.get("state") != "ready":
            continue
        classification = manifest.get("classification")
        improvement_pass = manifest.get("event_improvement_pass")
        if classification != "event_positive" and improvement_pass is not True:
            continue
        if not _event_gate_has_current_autoregressive_contract(path):
            continue
        if not _event_gate_has_replicated_full_event_contract(path):
            continue
        environment = str(manifest.get("environment") or "pour_water")
        primitive = str(manifest.get("primitive") or "")
        if not primitive:
            report = _load_optional_json(path.parent / "event-gate.json")
            primitive = str(report.get("primitive") or "")
        if not primitive:
            name = path.parent.name
            prefix = "acwm-pour-water-event-gate-"
            suffix = name[len(prefix):] if name.startswith(prefix) else ""
            primitive = re.split(r"-(?:s\d+-)?h\d+-r\d+$", suffix, maxsplit=1)[0]
        if environment and primitive:
            values.setdefault(environment, set()).add(primitive)
    return {environment: tuple(sorted(methods)) for environment, methods in values.items()}


def _event_gate_has_replicated_full_event_contract(manifest_path: Path) -> bool:
    report = _load_optional_json(manifest_path.parent / "event-gate.json")
    rows = report.get("trajectory_results")
    if not isinstance(rows, list):
        return False
    accepted: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("candidate_event_pass") is not True:
            continue
        trajectory_index = row.get("trajectory_index")
        frame_count = row.get("frame_count")
        if (
            isinstance(trajectory_index, bool)
            or not isinstance(trajectory_index, int)
            or trajectory_index < 0
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < _POUR_WATER_EVENT_MIN_FRAMES
        ):
            continue
        accepted.add(trajectory_index)
    return len(accepted) >= _POUR_WATER_EVENT_MIN_TRAJECTORIES


def _event_gate_has_current_autoregressive_contract(manifest_path: Path) -> bool:
    report = _load_optional_json(manifest_path.parent / "event-gate.json")
    if not report:
        report = _load_optional_json(manifest_path)
    source = report.get("source")
    if not isinstance(source, Mapping):
        return False
    required_horizon = max(_HORIZONS_BY_ENVIRONMENT["pour_water"])
    for field in ("baseline_manifest", "candidate_manifest"):
        value = source.get(field)
        if not isinstance(value, str) or not value:
            return False
        rollout = _load_optional_json(Path(value))
        horizons = rollout.get("horizons")
        if (
            rollout.get("state") != "ready"
            or rollout.get("mode") != "autoregressive"
            or not isinstance(horizons, list)
            or required_horizon not in horizons
            or not _rollout_has_bound_gpu_contract(rollout)
        ):
            return False
    return True


def _rollout_has_bound_gpu_contract(rollout: Mapping[str, object]) -> bool:
    gpu_index = rollout.get("gpu_index")
    binding = rollout.get("gpu_binding")
    audit = rollout.get("gpu_exclusivity_audit")
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        return False
    if not isinstance(binding, Mapping):
        return False
    physical = binding.get("physical_gpu_index")
    logical = binding.get("logical_device_index")
    visible = binding.get("cuda_visible_devices")
    if physical != gpu_index or isinstance(logical, bool) or not isinstance(logical, int) or logical < 0:
        return False
    if rollout.get("device") != f"cuda:{logical}":
        return False
    if visible is None:
        if logical != gpu_index:
            return False
    elif isinstance(visible, str):
        values = [value.strip() for value in visible.split(",") if value.strip()]
        if logical >= len(values) or values[logical] != str(gpu_index):
            return False
    else:
        return False
    if not isinstance(audit, Mapping):
        return False
    required = audit.get("required_gpus")
    if not isinstance(required, list):
        required = audit.get("audited_requested_gpus")
    return isinstance(required, list) and required == [gpu_index]


def _event_semantic_unresolved_environments(report_root: Path) -> set[str]:
    positive = _event_semantic_positive_methods(report_root)
    return {
        environment
        for environment in EVENT_SEMANTIC_REQUIRED_ENVIRONMENTS
        if not positive.get(environment)
    }


def _runtime_only_confirmation_groups(
    report_root: Path,
) -> dict[tuple[str, str, str], dict[str, object]]:
    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    paths = {
        *report_root.glob("acwm-autoloop-official-gate-*/manifest.json"),
        *report_root.glob("acwm-official-gate-*/manifest.json"),
    }
    for path in sorted(paths):
        manifest = _load_optional_json(path)
        primitive = str(manifest.get("primitive") or "")
        parameters = manifest.get("runtime_parameters")
        gate = manifest.get("official_quality_gate")
        seed = manifest.get("eval_seed", manifest.get("seed"))
        if (
            primitive not in RUNTIME_ONLY_PRIMITIVES
            or not isinstance(parameters, Mapping)
            or not isinstance(gate, Mapping)
            or not isinstance(gate.get("pass"), bool)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not _runtime_only_attempt_is_conclusive(path.parent)
        ):
            continue
        environment = str(manifest.get("environment") or _official_gate_environment_from_name(path.parent.name))
        if not environment:
            continue
        signature = _signature(parameters)
        key = (environment, primitive, signature)
        group = groups.setdefault(
            key,
            {
                "environment": environment,
                "primitive": primitive,
                "parameters": dict(parameters),
                "signature": signature,
                "attempt_seeds": set(),
                "pass_seeds": set(),
                "pass_manifests": [],
            },
        )
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        pass_manifests = group["pass_manifests"]
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        assert isinstance(pass_manifests, list)
        attempt_seeds.add(seed)
        if gate.get("pass") is True:
            pass_seeds.add(seed)
            pass_manifests.append(str(path.resolve()))
    return groups


def _checkpoint_delta_confirmation_key(
    manifest: Mapping[str, object],
) -> tuple[str, str] | None:
    provenance = manifest.get("checkpoint_transform_provenance")
    environment = str(manifest.get("environment") or "")
    if not isinstance(provenance, Mapping) or provenance.get("state") != "verified" or not environment:
        return None
    source_primitive = str(provenance.get("source_primitive") or "")
    scaled_sha = str(
        provenance.get("scaled_checkpoint_sha256")
        or manifest.get("candidate_checkpoint_sha256")
        or ""
    )
    alpha = provenance.get("alpha")
    if (
        not source_primitive
        or not scaled_sha
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
    ):
        return None
    return (
        environment,
        _signature(
            {
                "source_primitive": source_primitive,
                "scaled_checkpoint_sha256": scaled_sha,
                "alpha": float(alpha),
            }
        ),
    )


def _checkpoint_delta_confirmation_groups(
    report_root: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    paths = {
        *report_root.glob("acwm-autoloop-official-gate-*/manifest.json"),
        *report_root.glob("acwm-official-gate-*/manifest.json"),
    }
    for path in sorted(paths):
        manifest = _load_optional_json(path)
        provenance = manifest.get("checkpoint_transform_provenance")
        gate = manifest.get("official_quality_gate")
        eval_seed = manifest.get("eval_seed", manifest.get("seed"))
        if (
            manifest.get("state") != "ready"
            or manifest.get("primitive") != "checkpoint_delta_scaling"
            or not isinstance(provenance, Mapping)
            or provenance.get("state") != "verified"
            or not isinstance(gate, Mapping)
            or not isinstance(gate.get("pass"), bool)
            or isinstance(eval_seed, bool)
            or not isinstance(eval_seed, int)
        ):
            continue
        environment = str(manifest.get("environment") or "")
        source_primitive = str(provenance.get("source_primitive") or "")
        scaled_sha = str(
            provenance.get("scaled_checkpoint_sha256")
            or manifest.get("candidate_checkpoint_sha256")
            or ""
        )
        alpha = provenance.get("alpha")
        if (
            not environment
            or not source_primitive
            or not scaled_sha
            or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
        ):
            continue
        key = _checkpoint_delta_confirmation_key(manifest)
        if key is None:
            continue
        signature = key[1]
        group = groups.setdefault(
            key,
            {
                "environment": environment,
                "source_primitive": source_primitive,
                "scaled_checkpoint_sha256": scaled_sha,
                "alpha": float(alpha),
                "signature": signature,
                "attempt_seeds": set(),
                "pass_seeds": set(),
                "pass_manifests": [],
                "pass_psnr_deltas": [],
            },
        )
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        pass_manifests = group["pass_manifests"]
        pass_psnr_deltas = group["pass_psnr_deltas"]
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        assert isinstance(pass_manifests, list)
        assert isinstance(pass_psnr_deltas, list)
        attempt_seeds.add(eval_seed)
        if gate.get("pass") is True:
            pass_seeds.add(eval_seed)
            pass_manifests.append(str(path.resolve()))
            deltas = gate.get("delta_candidate_minus_baseline")
            psnr_delta = deltas.get("psnr") if isinstance(deltas, Mapping) else None
            if isinstance(psnr_delta, (int, float)) and not isinstance(psnr_delta, bool):
                pass_psnr_deltas.append(float(psnr_delta))
    return groups


def _checkpoint_delta_confirmation_in_flight_signatures(report_root: Path) -> set[str]:
    in_flight: set[str] = set()
    for queue_path in report_root.glob("acwm-autoloop-queue-delta-confirm-*/autoloop-queue.json"):
        queue = _load_optional_json(queue_path)
        rows = queue.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            signature = row.get("checkpoint_delta_confirmation_signature")
            output_value = row.get("output_root")
            if (
                not isinstance(signature, str)
                or not signature
                or not isinstance(output_value, str)
                or not output_value
            ):
                continue
            output_root = Path(output_value)
            if not (output_root / "manifest.json").is_file() and (
                queue_path.resolve() in _active_daemon_queue_paths(report_root)
                or _queue_has_fresh_unlaunched_reservation(queue_path)
            ):
                in_flight.add(signature)
    return in_flight


def _pending_checkpoint_delta_confirmation_candidates(
    report_root: Path,
) -> list[dict[str, object]]:
    in_flight = _checkpoint_delta_confirmation_in_flight_signatures(report_root)
    candidates: list[dict[str, object]] = []
    for group in _checkpoint_delta_confirmation_groups(report_root).values():
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        pass_manifests = group["pass_manifests"]
        pass_psnr_deltas = group["pass_psnr_deltas"]
        signature = str(group["signature"])
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        assert isinstance(pass_manifests, list)
        assert isinstance(pass_psnr_deltas, list)
        if len(pass_seeds) >= _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            continue
        if len(pass_seeds) != 1 or len(attempt_seeds) >= _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            continue
        if signature in in_flight:
            continue
        candidates.append(
            {
                "kind": "checkpoint_delta_confirmation",
                "environment": group["environment"],
                "primitive": "checkpoint_delta_scaling",
                "source_primitive": group["source_primitive"],
                "alpha": group["alpha"],
                "signature": signature,
                "attempt_seeds": sorted(attempt_seeds),
                "pass_seeds": sorted(pass_seeds),
                "source_pass_manifest": sorted(pass_manifests)[0],
                "mean_pass_psnr_delta": (
                    sum(pass_psnr_deltas) / len(pass_psnr_deltas)
                    if pass_psnr_deltas
                    else 0.0
                ),
                "parameters": {},
            }
        )
    return sorted(
        candidates,
        key=lambda record: (
            str(record["environment"]),
            -float(record["mean_pass_psnr_delta"]),
            abs(float(record["alpha"])),
            str(record["source_primitive"]),
        ),
    )


def _pending_checkpoint_delta_confirmation_environments(report_root: Path) -> set[str]:
    in_flight = _checkpoint_delta_confirmation_in_flight_signatures(report_root)
    environments: set[str] = set()
    for group in _checkpoint_delta_confirmation_groups(report_root).values():
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        signature = str(group["signature"])
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        if len(pass_seeds) >= _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            continue
        if len(pass_seeds) == 1 and (
            len(attempt_seeds) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES
            or signature in in_flight
        ):
            environments.add(str(group["environment"]))
    return environments


def _formally_confirmed_checkpoint_delta_keys(report_root: Path) -> set[tuple[str, str]]:
    confirmed: set[tuple[str, str]] = set()
    for key, group in _checkpoint_delta_confirmation_groups(report_root).items():
        pass_seeds = group["pass_seeds"]
        assert isinstance(pass_seeds, set)
        if len(pass_seeds) >= _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            confirmed.add(key)
    return confirmed


def _checkpoint_delta_horizon_evidence_keys(report_root: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in report_root.glob("acwm-confirmed-delta-horizon-queue-*/manifest.json"):
        manifest = _load_optional_json(path)
        contract = manifest.get("post_confirmation_evidence_contract")
        if manifest.get("state") != "ready" or not isinstance(contract, Mapping):
            continue
        if contract.get("mode") != "autoregressive":
            continue
        environment = str(manifest.get("environment") or "")
        candidate_sha = str(contract.get("candidate_checkpoint_sha256") or "")
        if environment and candidate_sha:
            keys.add((environment, candidate_sha))
    return keys


def _pending_checkpoint_delta_horizon_candidates(
    report_root: Path,
) -> list[dict[str, object]]:
    existing = _checkpoint_delta_horizon_evidence_keys(report_root)
    candidates: list[dict[str, object]] = []
    for group in _checkpoint_delta_confirmation_groups(report_root).values():
        pass_seeds = group["pass_seeds"]
        pass_manifest_values = group["pass_manifests"]
        assert isinstance(pass_seeds, set)
        assert isinstance(pass_manifest_values, list)
        if len(pass_seeds) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            continue
        environment = str(group["environment"])
        candidate_sha = str(group["scaled_checkpoint_sha256"])
        if (environment, candidate_sha) in existing:
            continue
        manifests_by_seed: dict[int, str] = {}
        psnr_delta_by_seed: dict[int, float] = {}
        for raw_path in sorted(pass_manifest_values):
            if not isinstance(raw_path, str) or not raw_path:
                continue
            manifest = _load_optional_json(Path(raw_path))
            eval_seed = manifest.get("eval_seed", manifest.get("seed"))
            if isinstance(eval_seed, int) and not isinstance(eval_seed, bool):
                manifests_by_seed.setdefault(eval_seed, str(Path(raw_path).resolve()))
                gate = manifest.get("official_quality_gate")
                deltas = gate.get("delta_candidate_minus_baseline") if isinstance(gate, Mapping) else None
                psnr_delta = deltas.get("psnr") if isinstance(deltas, Mapping) else None
                if isinstance(psnr_delta, (int, float)) and not isinstance(psnr_delta, bool):
                    psnr_delta_by_seed.setdefault(eval_seed, float(psnr_delta))
        if len(manifests_by_seed) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES:
            continue
        candidates.append(
            {
                "kind": "checkpoint_delta_horizon_evidence",
                "environment": environment,
                "primitive": "checkpoint_delta_scaling",
                "source_primitive": str(group["source_primitive"]),
                "alpha": float(group["alpha"]),
                "candidate_checkpoint_sha256": candidate_sha,
                "pass_manifests": [
                    manifests_by_seed[seed] for seed in sorted(manifests_by_seed)
                ],
                "mean_confirmation_psnr_delta": (
                    sum(psnr_delta_by_seed.values()) / len(psnr_delta_by_seed)
                    if psnr_delta_by_seed
                    else 0.0
                ),
                "parameters": {},
            }
        )
    return sorted(
        candidates,
        key=lambda record: (
            str(record["environment"]),
            -float(record["mean_confirmation_psnr_delta"]),
            abs(float(record["alpha"])),
            str(record["source_primitive"]),
        ),
    )


def _checkpoint_delta_horizon_sources(
    report_root: Path,
) -> dict[Path, dict[str, object]]:
    sources: dict[Path, dict[str, object]] = {}
    for manifest_path in report_root.glob("acwm-confirmed-delta-horizon-queue-*/manifest.json"):
        manifest = _load_optional_json(manifest_path)
        contract = manifest.get("post_confirmation_evidence_contract")
        queue_value = manifest.get("queue_path")
        if (
            manifest.get("state") != "ready"
            or not isinstance(contract, Mapping)
            or contract.get("mode") != "autoregressive"
            or not isinstance(queue_value, str)
            or not queue_value
        ):
            continue
        queue = _load_optional_json(Path(queue_value))
        rows = queue.get("rows")
        if not isinstance(rows, list):
            continue
        candidate_row = next(
            (
                row
                for row in rows
                if isinstance(row, Mapping) and row.get("phase") == "long_horizon_candidate"
            ),
            None,
        )
        output_value = candidate_row.get("output_root") if isinstance(candidate_row, Mapping) else None
        if not isinstance(output_value, str) or not output_value:
            continue
        sources[(Path(output_value) / "manifest.json").resolve()] = {
            "manifest_path": str(manifest_path.resolve()),
            "contract": dict(contract),
            "environment": str(manifest.get("environment") or ""),
            "source_primitive": str(manifest.get("source_primitive") or ""),
            "alpha": manifest.get("checkpoint_delta_alpha"),
        }
    return sources


def _checkpoint_delta_horizon_replication_keys(report_root: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for manifest_path in report_root.glob("acwm-confirmed-delta-horizon-queue-*/manifest.json"):
        manifest = _load_optional_json(manifest_path)
        contract = manifest.get("post_confirmation_evidence_contract")
        if (
            manifest.get("state") != "ready"
            or not isinstance(contract, Mapping)
            or contract.get("evidence_role") != "factorized_heldout_replication"
        ):
            continue
        environment = str(manifest.get("environment") or "")
        candidate_sha = str(contract.get("candidate_checkpoint_sha256") or "")
        if environment and candidate_sha:
            keys.add((environment, candidate_sha))
    return keys


def _pending_checkpoint_delta_horizon_replication_candidates(
    report_root: Path,
) -> list[dict[str, object]]:
    sources = _checkpoint_delta_horizon_sources(report_root)
    existing = _checkpoint_delta_horizon_replication_keys(report_root)
    candidates_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for profile_path in report_root.glob(
        "acwm-horizon-effect-profile-*/horizon-effect-profile.json"
    ):
        profile = _load_optional_json(profile_path)
        classification = profile.get("effect_classification")
        source = profile.get("source")
        transfer = profile.get("transfer_prior")
        if (
            profile.get("state") != "ready"
            or profile.get("primitive") != "checkpoint_delta_scaling"
            or not isinstance(classification, Mapping)
            or classification.get("effect_scope") != "aggregate_long_horizon_positive"
            or classification.get("aggregate_max_horizon_pass") is not True
            or not isinstance(source, Mapping)
            or not isinstance(transfer, Mapping)
            or transfer.get("causal_credit_eligible") is True
        ):
            continue
        environment = str(profile.get("environment") or "")
        if environment in EVENT_SEMANTIC_REQUIRED_ENVIRONMENTS and (
            classification.get("event_semantic_classification") != "event_positive"
            or classification.get("candidate_event_pass") is not True
        ):
            continue
        candidate_manifest_value = source.get("candidate_manifest")
        if not isinstance(candidate_manifest_value, str) or not candidate_manifest_value:
            continue
        horizon_source = sources.get(Path(candidate_manifest_value).resolve())
        if horizon_source is None:
            continue
        contract = horizon_source["contract"]
        assert isinstance(contract, Mapping)
        candidate_sha = str(contract.get("candidate_checkpoint_sha256") or "")
        source_primitive = str(horizon_source.get("source_primitive") or "")
        alpha = horizon_source.get("alpha")
        pass_manifests = contract.get("pass_manifests")
        horizons = contract.get("horizons")
        if (
            not environment
            or not candidate_sha
            or not source_primitive
            or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not isinstance(pass_manifests, list)
            or len(pass_manifests) < _CHECKPOINT_DELTA_CONFIRMATION_REQUIRED_PASSES
            or not isinstance(horizons, list)
            or not horizons
        ):
            continue
        key = (environment, candidate_sha)
        if key in existing:
            continue
        late_delta = classification.get("late_half_mean_delta_psnr")
        score = (
            float(late_delta)
            if isinstance(late_delta, (int, float)) and not isinstance(late_delta, bool)
            else 0.0
        )
        candidate = {
            "kind": "checkpoint_delta_horizon_replication",
            "environment": environment,
            "primitive": "checkpoint_delta_scaling",
            "source_primitive": source_primitive,
            "alpha": float(alpha),
            "candidate_checkpoint_sha256": candidate_sha,
            "pass_manifests": [str(value) for value in pass_manifests if isinstance(value, str)],
            "horizons": [
                int(value)
                for value in horizons
                if isinstance(value, int) and not isinstance(value, bool)
            ],
            "source_profile": str(profile_path.resolve()),
            "source_queue_manifest": horizon_source["manifest_path"],
            "source_trajectory_seed": contract.get("trajectory_seed", 101),
            "source_paired_trajectory_count": profile.get("paired_trajectory_count"),
            "late_half_mean_delta_psnr": score,
            "parameters": {},
        }
        previous = candidates_by_key.get(key)
        if previous is None or score > float(previous["late_half_mean_delta_psnr"]):
            candidates_by_key[key] = candidate
    return sorted(
        candidates_by_key.values(),
        key=lambda record: (
            -float(record["late_half_mean_delta_psnr"]),
            str(record["environment"]),
            str(record["source_primitive"]),
        ),
    )


def _runtime_only_confirmation_in_flight_signatures(report_root: Path) -> set[tuple[str, str, str]]:
    in_flight: set[tuple[str, str, str]] = set()
    for queue_path in report_root.glob("acwm-autoloop-queue-runtime-confirm-*/autoloop-queue.json"):
        queue = _load_optional_json(queue_path)
        rows = queue.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            signature = row.get("runtime_confirmation_signature")
            output_value = row.get("output_root")
            environment = str(row.get("environment") or "")
            primitive = str(row.get("primitive") or "")
            if not isinstance(signature, str) or not signature or not isinstance(output_value, str) or not output_value:
                continue
            output_root = Path(output_value)
            if _runtime_only_attempt_is_conclusive(output_root):
                continue
            if not output_root.exists() or _runtime_only_attempt_is_in_flight(output_root):
                in_flight.add((environment, primitive, signature))
    return in_flight


def _pending_runtime_only_confirmation_candidates(report_root: Path) -> list[dict[str, object]]:
    in_flight = _runtime_only_confirmation_in_flight_signatures(report_root)
    candidates: list[dict[str, object]] = []
    for key, group in sorted(_runtime_only_confirmation_groups(report_root).items()):
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        pass_manifests = group["pass_manifests"]
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        assert isinstance(pass_manifests, list)
        if (
            len(pass_seeds) < 1
            or len(pass_seeds) >= _RUNTIME_CONFIRMATION_REQUIRED_PASSES
            or len(attempt_seeds) >= _RUNTIME_CONFIRMATION_REQUIRED_PASSES
            or key in in_flight
        ):
            continue
        candidates.append(
            {
                "kind": "runtime_only_confirmation",
                "environment": group["environment"],
                "primitive": group["primitive"],
                "parameters": group["parameters"],
                "signature": group["signature"],
                "source_manifest": sorted(pass_manifests)[0],
                "source_pass_seeds": sorted(pass_seeds),
                "required_passes": _RUNTIME_CONFIRMATION_REQUIRED_PASSES,
            }
        )
    return candidates


def _pending_runtime_only_confirmation_environments(report_root: Path) -> set[str]:
    in_flight = _runtime_only_confirmation_in_flight_signatures(report_root)
    environments: set[str] = set()
    for key, group in _runtime_only_confirmation_groups(report_root).items():
        attempt_seeds = group["attempt_seeds"]
        pass_seeds = group["pass_seeds"]
        assert isinstance(attempt_seeds, set)
        assert isinstance(pass_seeds, set)
        if len(pass_seeds) >= _RUNTIME_CONFIRMATION_REQUIRED_PASSES:
            continue
        if len(pass_seeds) >= 1 and (
            len(attempt_seeds) < _RUNTIME_CONFIRMATION_REQUIRED_PASSES or key in in_flight
        ):
            environments.add(str(group["environment"]))
    return environments


def _formally_confirmed_runtime_only_keys(report_root: Path) -> set[tuple[str, str, str]]:
    confirmed: set[tuple[str, str, str]] = set()
    for key, group in _runtime_only_confirmation_groups(report_root).items():
        pass_seeds = group["pass_seeds"]
        assert isinstance(pass_seeds, set)
        if len(pass_seeds) >= _RUNTIME_CONFIRMATION_REQUIRED_PASSES:
            confirmed.add(key)
    return confirmed


def _pending_checkpoint_delta_recovery_candidates(
    report_root: Path,
    *,
    excluded_environments: set[str],
) -> list[dict[str, object]]:
    active_queues = _active_daemon_queue_paths(report_root)
    in_flight_source_gates: set[str] = set()
    for queue_path in report_root.glob("acwm-autoloop-queue-delta-recovery-*/autoloop-queue.json"):
        queue = _load_optional_json(queue_path)
        rows = queue.get("rows")
        if not isinstance(rows, list):
            continue
        queue_reserved = (
            queue_path.resolve() in active_queues
            or _queue_has_fresh_unlaunched_reservation(queue_path)
        )
        if not queue_reserved:
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            source = row.get("source_official_gate_manifest")
            output = row.get("output_root")
            if (
                isinstance(source, str)
                and source
                and isinstance(output, str)
                and output
                and not (Path(output) / "manifest.json").is_file()
            ):
                in_flight_source_gates.add(str(Path(source).resolve()))

    candidates: list[tuple[tuple[float, float, float], dict[str, object]]] = []
    for gate_path in report_root.glob("acwm-autoloop-official-gate-*/manifest.json"):
        gate_path = gate_path.resolve()
        manifest = _load_optional_json(gate_path)
        quality_gate = manifest.get("official_quality_gate")
        source_primitive = str(manifest.get("primitive") or "")
        environment = str(manifest.get("environment") or "")
        seed = manifest.get("seed")
        if (
            str(gate_path) in in_flight_source_gates
            or manifest.get("state") != "ready"
            or not isinstance(quality_gate, Mapping)
            or quality_gate.get("pass") is not False
            or source_primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES
            or source_primitive in INVALIDATED_QUALITY_PRIMITIVES
            or not environment
            or environment in excluded_environments
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            continue
        candidate_value = manifest.get("candidate_checkpoint")
        baseline_value = manifest.get("baseline_checkpoint")
        runtime_value = manifest.get("candidate_runtime_root")
        if not all(isinstance(value, str) and value for value in (candidate_value, baseline_value, runtime_value)):
            continue
        candidate_checkpoint = Path(str(candidate_value)).resolve()
        baseline_checkpoint = Path(str(baseline_value)).resolve()
        candidate_runtime_root = Path(str(runtime_value)).resolve()
        if (
            candidate_checkpoint.is_symlink()
            or not candidate_checkpoint.is_file()
            or baseline_checkpoint.is_symlink()
            or not baseline_checkpoint.is_file()
            or candidate_runtime_root.is_symlink()
            or not candidate_runtime_root.is_dir()
            or candidate_checkpoint.parent.name != "retained_training"
        ):
            continue
        source_screen_manifest = candidate_checkpoint.parent.parent / "manifest.json"
        screen = _load_optional_json(source_screen_manifest)
        primary_metric = str(screen.get("primary_metric") or "ladder_auc_psnr_envmax")
        deltas = screen.get("delta_m_ver")
        internal_delta = deltas.get(primary_metric) if isinstance(deltas, Mapping) else None
        action_gate = screen.get("action_following_gate")
        if (
            screen.get("state") != "ready"
            or isinstance(internal_delta, bool)
            or not isinstance(internal_delta, (int, float))
            or float(internal_delta) <= 0.0
            or (
                isinstance(action_gate, Mapping)
                and action_gate.get("enabled") is True
                and action_gate.get("pass") is not True
            )
        ):
            continue

        completed_alphas: set[float] = set()
        for recovery_path in report_root.glob(
            f"acwm-autoloop-official-gate-{environment}-checkpoint_delta_scaling-s{seed}-alpha*-r*/manifest.json"
        ):
            recovery = _load_optional_json(recovery_path)
            provenance = recovery.get("checkpoint_transform_provenance")
            if (
                recovery.get("state") != "ready"
                or not isinstance(provenance, Mapping)
                or Path(str(provenance.get("source_official_gate_manifest_path") or "")).resolve()
                != gate_path
            ):
                continue
            alpha = provenance.get("alpha")
            if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
                completed_alphas.add(round(float(alpha), 12))
        pending_alphas = [
            alpha
            for alpha in _CHECKPOINT_DELTA_RECOVERY_ALPHAS
            if round(alpha, 12) not in completed_alphas
        ]
        if not pending_alphas:
            continue
        official_deltas = quality_gate.get("delta_candidate_minus_baseline")
        official_delta_record = dict(official_deltas) if isinstance(official_deltas, Mapping) else {}
        psnr_delta = official_deltas.get("psnr") if isinstance(official_deltas, Mapping) else None
        psnr_regression = (
            abs(float(psnr_delta))
            if isinstance(psnr_delta, (int, float)) and not isinstance(psnr_delta, bool)
            else float("inf")
        )
        candidates.append(
            (
                (psnr_regression, -float(internal_delta), -gate_path.stat().st_mtime),
                {
                    "kind": "checkpoint_delta_recovery",
                    "environment": environment,
                    "source_primitive": source_primitive,
                    "primitive": "checkpoint_delta_scaling",
                    "seed": seed,
                    "parameters": {},
                    "pending_alphas": pending_alphas,
                    "internal_primary_metric": primary_metric,
                    "internal_primary_delta": float(internal_delta),
                    "official_gate_delta": official_delta_record,
                    "baseline_checkpoint": str(baseline_checkpoint),
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "candidate_runtime_root": str(candidate_runtime_root),
                    "source_screen_manifest": str(source_screen_manifest.resolve()),
                    "source_official_gate_manifest": str(gate_path),
                },
            )
        )
    return [record for _, record in sorted(candidates, key=lambda item: item[0])]


def _official_training_gate_failure_counts(report_root: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(report_root.glob("acwm-autoloop-official-gate-*/manifest.json")):
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        if (
            manifest.get("state") != "ready"
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not False
        ):
            continue
        environment = str(manifest.get("environment") or "")
        primitive = str(manifest.get("primitive") or "")
        if not environment or not primitive:
            match = _SCREEN_OFFICIAL_NAME.match(path.parent.name)
            if match is not None:
                environment, primitive, _ = match.groups()
        if not environment or primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES:
            continue
        key = (environment, primitive)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _training_screen_negative_counts(report_root: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(report_root.glob("acwm-autoloop-screen-*/envs/*/manifest.json")):
        screen_root = path.parents[2]
        match = _SCREEN_NAME.match(screen_root.name)
        if match is None:
            continue
        environment, primitive, _ = match.groups()
        if primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES:
            continue
        manifest = _load_optional_json(path)
        deltas = manifest.get("delta_m_ver")
        primary_metric = manifest.get("primary_metric")
        if manifest.get("state") != "ready" or not isinstance(deltas, Mapping):
            continue
        delta = deltas.get(primary_metric) if isinstance(primary_metric, str) else None
        if delta is None:
            delta = deltas.get("ladder_auc_psnr_envmax")
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or float(delta) > 0.0:
            continue
        key = (environment, primitive)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _screen_officially_positive_environments(report_root: Path) -> set[str]:
    environments: set[str] = set()
    confirmed_runtime_only = _formally_confirmed_runtime_only_keys(report_root)
    confirmed_checkpoint_delta = _formally_confirmed_checkpoint_delta_keys(report_root)
    paths = {
        *report_root.glob("acwm-autoloop-official-gate-*/manifest.json"),
        *report_root.glob("acwm-official-gate-*/manifest.json"),
    }
    for path in sorted(paths):
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        if manifest.get("state") != "ready" or not isinstance(gate, Mapping) or gate.get("pass") is not True:
            continue
        primitive = manifest.get("primitive")
        if primitive in INVALIDATED_QUALITY_PRIMITIVES:
            continue
        if primitive == "checkpoint_delta_scaling":
            key = _checkpoint_delta_confirmation_key(manifest)
            if key is None or key not in confirmed_checkpoint_delta:
                continue
        if primitive in RUNTIME_ONLY_PRIMITIVES:
            effect_gate = manifest.get("runtime_effect_gate")
            parameters = manifest.get("runtime_parameters")
            environment = str(
                manifest.get("environment") or _official_gate_environment_from_name(path.parent.name)
            )
            if (
                manifest.get("execution_mode") != "runtime_only"
                or not isinstance(effect_gate, Mapping)
                or effect_gate.get("pass") is not True
                or not isinstance(parameters, Mapping)
                or (environment, str(primitive), _signature(parameters)) not in confirmed_runtime_only
            ):
                continue
        environment = manifest.get("environment")
        if not isinstance(environment, str) or not environment:
            environment = _official_gate_environment_from_name(path.parent.name)
        if environment:
            environments.add(environment)
    return environments


def _quality_discovery_terminal_positive_environments(report_root: Path) -> set[str]:
    """Return only positives that satisfy their full method-specific confirmation contract."""

    environments: set[str] = set()
    confirmed_runtime_only = _formally_confirmed_runtime_only_keys(report_root)
    confirmed_checkpoint_delta = _formally_confirmed_checkpoint_delta_keys(report_root)
    paths = {
        *report_root.glob("acwm-autoloop-official-gate-*/manifest.json"),
        *report_root.glob("acwm-official-gate-*/manifest.json"),
    }
    for path in sorted(paths):
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        primitive = str(manifest.get("primitive") or "")
        if (
            manifest.get("state") != "ready"
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not True
            or primitive in INVALIDATED_QUALITY_PRIMITIVES
        ):
            continue
        environment = str(
            manifest.get("environment") or _official_gate_environment_from_name(path.parent.name)
        )
        if not environment:
            continue
        if primitive in TRAINING_QUALITY_SCREEN_PRIMITIVES:
            # Training interventions close only through the independent 1k ladder gate.
            continue
        if primitive == "checkpoint_delta_scaling":
            key = _checkpoint_delta_confirmation_key(manifest)
            if key is None or key not in confirmed_checkpoint_delta:
                continue
        if primitive in RUNTIME_ONLY_PRIMITIVES:
            parameters = manifest.get("runtime_parameters")
            effect_gate = manifest.get("runtime_effect_gate")
            if (
                manifest.get("execution_mode") != "runtime_only"
                or not isinstance(parameters, Mapping)
                or not isinstance(effect_gate, Mapping)
                or effect_gate.get("pass") is not True
                or (environment, primitive, _signature(parameters)) not in confirmed_runtime_only
            ):
                continue
        environments.add(environment)
    return environments


def _officially_positive_training_primitives(report_root: Path) -> tuple[str, ...]:
    primitives: list[str] = []
    for record in _training_finalization_records(report_root):
        primitive = record["primitive"]
        if (
            record["confirmation_passed"] is not True
            or not isinstance(primitive, str)
            or primitive not in TRAINING_QUALITY_SCREEN_PRIMITIVES
            or primitive in INVALIDATED_QUALITY_PRIMITIVES
        ):
            continue
        if primitive not in primitives:
            primitives.append(primitive)
    return tuple(primitives)


def _training_finalization_records(report_root: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for path in sorted(
        report_root.glob("acwm-autoloop-checkpoint-finalize-*/checkpoint-ladder-finalization.json")
    ):
        report = _load_optional_json(path)
        if (
            report.get("artifact_type") != "wmloop-acwm-checkpoint-ladder-finalization"
            or report.get("state") not in {"ready", "checks_failed"}
        ):
            continue
        environment = report.get("environment")
        primitive = report.get("primitive")
        seed = report.get("seed")
        selection = report.get("selection")
        ladder_records = selection.get("records") if isinstance(selection, Mapping) else None
        if (
            not isinstance(environment, str)
            or not environment
            or not isinstance(primitive, str)
            or not primitive
            or not isinstance(seed, int)
            or not isinstance(ladder_records, list)
        ):
            continue
        confirmation_passed = any(
            isinstance(row, Mapping)
            and isinstance(row.get("checkpoint_step"), int)
            and int(row["checkpoint_step"]) >= 800
            and isinstance(row.get("official_quality_gate"), Mapping)
            and row["official_quality_gate"].get("pass") is True
            for row in ladder_records
        )
        records.append(
            {
                "environment": environment,
                "primitive": primitive,
                "seed": seed,
                "confirmation_passed": confirmation_passed,
                "report_path": str(path.resolve()),
            }
        )
    return tuple(records)


def _running_screen_environments(report_root: Path) -> set[str]:
    environments: set[str] = set()
    for path in report_root.glob("acwm-autoloop-screen-*/status.json"):
        status = _load_optional_json(path)
        if status.get("state") != "running":
            continue
        records = status.get("records")
        if not isinstance(records, list):
            continue
        live = False
        for record in records:
            if not isinstance(record, Mapping) or record.get("state") != "running":
                continue
            pid = record.get("pid")
            if isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists():
                live = True
                break
        if not live:
            continue
        match = _SCREEN_NAME.fullmatch(path.parent.name)
        if match is not None:
            environments.add(match.group(1))
    return environments


def _pending_internal_positive_environments(report_root: Path) -> set[str]:
    environments: set[str] = set()
    for path in report_root.glob("acwm-autoloop-screen-*/envs/*/manifest.json"):
        manifest = _load_optional_json(path)
        primary_metric = str(manifest.get("primary_metric") or "ladder_auc_psnr_envmax")
        delta = manifest.get("delta_m_ver")
        value = delta.get(primary_metric) if isinstance(delta, Mapping) else None
        if manifest.get("state") != "ready" or isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        match = _SCREEN_NAME.fullmatch(path.parents[2].name)
        if match is None:
            continue
        environment, primitive, seed = match.groups()
        if primitive in INVALIDATED_QUALITY_PRIMITIVES:
            continue
        official_paths = list(
            report_root.glob(f"acwm-autoloop-official-gate-{environment}-{primitive}-s{seed}-r*/manifest.json")
        )
        official = [_load_optional_json(candidate) for candidate in official_paths]
        if not official or any(candidate.get("state") != "ready" for candidate in official):
            environments.add(environment)
    return environments


def _pending_positive_confirmation_environments(report_root: Path) -> set[str]:
    finalized = {
        (str(record["environment"]), str(record["primitive"]), int(record["seed"]))
        for record in _training_finalization_records(report_root)
    }

    pending: set[str] = set()
    for path in report_root.glob("acwm-autoloop-official-gate-*/manifest.json"):
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        if manifest.get("state") != "ready" or not isinstance(gate, Mapping) or gate.get("pass") is not True:
            continue
        match = _SCREEN_OFFICIAL_NAME.fullmatch(path.parent.name)
        if match is None:
            continue
        environment, primitive, seed_raw = match.groups()
        key = (environment, primitive, int(seed_raw))
        if key in finalized:
            continue
        confirmations = sorted(
            report_root.glob(
                f"acwm-autoloop-confirm-{environment}-{primitive}-s{seed_raw}-t*-r*"
            )
        )
        if any(_confirmation_failed(root) for root in confirmations):
            continue
        pending.add(environment)
    return pending


def _confirmation_failed(root: Path) -> bool:
    status = _load_optional_json(root / "status.json")
    if status.get("state") == "failed":
        return True
    records = status.get("records")
    return isinstance(records, list) and any(
        isinstance(record, Mapping) and record.get("state") == "failed"
        for record in records
    )


def _official_gate_environment_from_name(name: str) -> str:
    prefix = "acwm-autoloop-official-gate-"
    if not name.startswith(prefix):
        return ""
    suffix = name[len(prefix) :]
    for environment in (
        "cloth_move",
        "pour_water",
        "push_cube",
        "push_rope",
        "push_sand",
        "reacher",
        "robot_arm",
        "stack_cube",
    ):
        if suffix.startswith(f"{environment}-"):
            return environment
    return ""


def _attempted_parameter_signatures(report_root: Path) -> set[tuple[str, str, str]]:
    attempted: set[tuple[str, str, str]] = set()
    active_queues = _active_daemon_queue_paths(report_root)
    for path in report_root.glob("acwm-autoloop-queue-*/autoloop-queue.json"):
        queue = _load_optional_json(path)
        for row in queue.get("rows", []):
            if not isinstance(row, Mapping) or row.get("phase") != "screen_512":
                continue
            output_value = row.get("output_root")
            environment = str(row.get("environment") or "")
            output_root = Path(output_value) if isinstance(output_value, str) and output_value else None
            if output_root is None:
                if path.resolve() not in active_queues and not _queue_has_fresh_unlaunched_reservation(path):
                    continue
            elif (
                _screen_failed_without_evidence(output_root)
                and not _screen_failure_is_terminal(output_root)
                and not _screen_attempt_is_in_flight(output_root)
            ):
                continue
            elif not (
                _screen_attempt_has_evidence(output_root, environment)
                or _screen_failure_is_terminal(output_root)
                or _screen_attempt_is_in_flight(output_root)
                or path.resolve() in active_queues
                or _queue_has_fresh_unlaunched_reservation(path)
            ):
                continue
            attempted.add(
                (
                    environment,
                    str(row.get("primitive") or ""),
                    _signature(row.get("primitive_parameters") if isinstance(row.get("primitive_parameters"), Mapping) else {}),
                )
            )
    for path in report_root.glob("acwm-autoloop-screen-*"):
        match = _SCREEN_NAME.match(path.name)
        if match is None:
            continue
        environment, primitive, _ = match.groups()
        if not (
            _screen_attempt_has_evidence(path, environment)
            or _screen_failure_is_terminal(path)
            or _screen_attempt_is_in_flight(path)
        ):
            continue
        if not any(item[0] == environment and item[1] == primitive for item in attempted):
            attempted.add((environment, primitive, _signature({})))
    default_signature = _signature(_runtime_parameters({}))
    for environment, primitive in _discover_existing_cells(report_root):
        attempted.add((environment, primitive, default_signature))
    return attempted


def _active_daemon_queue_paths(report_root: Path) -> set[Path]:
    active: set[Path] = set()
    for status_path in report_root.glob("acwm-autoloop-daemon-*/status.json"):
        status = _load_optional_json(status_path)
        pid = status.get("pid")
        if status.get("state") != "running" or not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
            continue
        queues = status.get("queues")
        if not isinstance(queues, list):
            continue
        active.update(Path(value).resolve() for value in queues if isinstance(value, str) and value)
    return active


def _screen_attempt_has_evidence(output_root: Path, environment: str) -> bool:
    manifest = _load_optional_json(Path(output_root) / "envs" / environment / "manifest.json")
    return manifest.get("state") == "ready"


def _screen_attempt_is_in_flight(output_root: Path) -> bool:
    status = _load_optional_json(Path(output_root) / "status.json")
    records = status.get("records")
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, Mapping) or record.get("state") != "running":
            continue
        pid = record.get("pid")
        if isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists():
            return True
    return False


def _queue_has_fresh_unlaunched_reservation(queue_path: Path) -> bool:
    try:
        age_seconds = time.time() - queue_path.stat().st_mtime
    except OSError:
        return False
    return age_seconds <= _UNLAUNCHED_QUEUE_RESERVATION_SECONDS


def _attempted_runtime_only_signatures(report_root: Path) -> set[tuple[str, str, str]]:
    attempted: set[tuple[str, str, str]] = set()
    for path in report_root.glob("acwm-autoloop-queue-runtime-*/autoloop-queue.json"):
        queue = _load_optional_json(path)
        for row in queue.get("rows", []):
            if not isinstance(row, Mapping) or row.get("execution_mode") != "runtime_only":
                continue
            output_value = row.get("output_root")
            if isinstance(output_value, str) and output_value:
                output_root = Path(output_value)
                if not _runtime_only_attempt_is_conclusive(output_root):
                    if not _runtime_only_attempt_is_in_flight(output_root):
                        continue
            attempted.add(
                (
                    str(row.get("environment") or ""),
                    str(row.get("primitive") or ""),
                    _signature(
                        row.get("primitive_parameters")
                        if isinstance(row.get("primitive_parameters"), Mapping)
                        else {}
                    ),
                )
            )
    return attempted


def _runtime_only_attempt_is_conclusive(output_root: Path) -> bool:
    """Count only runtime-only trials that prove the candidate hook was invoked."""

    manifest = _load_optional_json(Path(output_root) / "manifest.json")
    receipt = manifest.get("candidate_runtime_hook_receipt")
    classification = manifest.get("runtime_result_classification")
    if (
        manifest.get("state") != "ready"
        or manifest.get("execution_mode") != "runtime_only"
        or not isinstance(receipt, Mapping)
        or receipt.get("state") != "ready"
        or not isinstance(receipt.get("call_count"), int)
        or int(receipt["call_count"]) < 1
    ):
        return False
    if isinstance(classification, Mapping):
        return classification.get("conclusive_quality_attempt") is True
    effect_gate = manifest.get("runtime_effect_gate")
    quality_gate = manifest.get("official_quality_gate")
    return isinstance(effect_gate, Mapping) and isinstance(effect_gate.get("pass"), bool) and isinstance(
        quality_gate, Mapping
    ) and isinstance(quality_gate.get("pass"), bool)


def _runtime_only_attempt_is_in_flight(output_root: Path) -> bool:
    """Reserve a launched runtime-only cell until its process exits or writes a manifest."""

    marker = output_root.with_name(f"{output_root.name}.launching")
    if not marker.is_file() or marker.is_symlink():
        return False
    try:
        pid = int(marker.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _screen_failed_without_evidence(output_root: Path) -> bool:
    status = _load_optional_json(Path(output_root) / "status.json")
    if status.get("state") not in {"failed", "checks_failed"}:
        return False
    envs_root = Path(output_root) / "envs"
    return not any(path.is_file() for path in envs_root.glob("*/manifest.json"))


_TERMINAL_SCREEN_FAILURE_MARKERS = (
    "M3_TRAINING_EVAL_SMOKE_PRIMITIVE_UNAVAILABLE:",
    "M3_TRAINING_EVAL_SMOKE_PRIMITIVE_NOT_MATERIALIZED:",
    "PROPOSAL_PRIMITIVE_NOT_ROUTED:",
    "PRIMITIVE_PARAMS_INVALID:",
    "PRIMITIVE_RUNTIME_SMOKE_PRIMITIVE_UNSUPPORTED:",
)


def _screen_failure_is_terminal(output_root: Path) -> bool:
    """Classify deterministic contract failures so the same trial is not requeued forever."""

    candidates = [Path(output_root) / "status.json"]
    candidates.extend(sorted((Path(output_root) / "logs").glob("*.log")))
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(marker in text for marker in _TERMINAL_SCREEN_FAILURE_MARKERS):
            return True
    return False


def _signature(parameters: Mapping[str, object]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_replenishment_bundle(destination: Path, payload: Mapping[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "manifest.json", payload)
        os.replace(temporary, destination)
        return dict(payload)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmAutoloopReplenisherError(f"ACWM_REPLENISH_JSON_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise AcwmAutoloopReplenisherError(f"ACWM_REPLENISH_JSON_NOT_OBJECT:{path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        return _load_json_object(path)
    except AcwmAutoloopReplenisherError:
        return {}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--staging-plan", type=Path, required=True)
    parser.add_argument("--materialization-gate", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quality-discovery-only", action="store_true")
    args = parser.parse_args(argv)
    result = replenish_autoloop_queue(
        staging_plan=args.staging_plan,
        materialization_gate=args.materialization_gate,
        report_root=args.report_root,
        output_root=args.output_root,
        gpu=args.gpu,
        seed=args.seed,
        repo_root=args.repo_root,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        quality_discovery_only=args.quality_discovery_only,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
