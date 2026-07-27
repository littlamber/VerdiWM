"""Run a scoped GPU-backed training+eval campaign over a limited env set."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmloop.contracts import load_yaml_document
from wmloop.execute.training_monitor_policy import training_monitor_policy_document
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.verify.m4_launch_guard import verify_m4_launch_allowed


class TrainingEvalLimitedCampaignError(RuntimeError):
    """The GPU-backed limited training+eval campaign failed closed."""


AUTO_BY_DIAGNOSIS = "auto_by_diagnosis"
RUNNABLE_PRIMITIVES = (
    "latent_motion_prior",
    "history_noise_schedule",
    "drift_token_trim",
    "mixture_reweight",
    "event_window_reweight",
    "motion_region_reweight",
    "next_forcing",
    "inv_dyn_reward_finetune",
    "latent_spatial_memory",
)
AUTO_ROUTABLE_PRIMITIVES = tuple(
    primitive for primitive in RUNNABLE_PRIMITIVES if primitive != "latent_spatial_memory"
)
AUTO_PRIMITIVE_PREFERENCES = {
    "action_binding": ("inv_dyn_reward_finetune", "latent_motion_prior"),
    "ood_physics": ("motion_region_reweight", "mixture_reweight", "latent_motion_prior"),
    "train_infer_mismatch": ("motion_region_reweight", "mixture_reweight", "next_forcing", "history_noise_schedule", "drift_token_trim"),
    "appearance_drift": ("motion_region_reweight", "next_forcing", "drift_token_trim", "history_noise_schedule"),
    "mixed": ("motion_region_reweight", "mixture_reweight", "next_forcing", "history_noise_schedule", "drift_token_trim", "latent_motion_prior"),
    "fluid_volume_transport": ("event_window_reweight", "self_forcing_finetune", "next_forcing"),
    "sparse_event_undercoverage": ("event_window_reweight", "mixture_reweight"),
}


@dataclass(frozen=True)
class TrainingEvalJob:
    environment: str
    gpu_index: int
    horizons: tuple[int, ...]
    horizon_source: str
    proposal_primitive: str
    output_root: Path
    log_path: Path
    command: tuple[str, ...]


def build_training_eval_jobs(
    *,
    repo_root: Path,
    limited_gate_manifest: Path,
    failure_report_manifest: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_exclusivity_audit_manifest: Path,
    gpus: Sequence[int],
    campaign_id: str,
    m4_phase_gate_manifest: Path | None = None,
    primitive_materialization_gate_manifest: Path | None = None,
    publish_settled_trials: bool = False,
    environments: Sequence[str] | None = None,
    diagnostic_eval_horizons: Sequence[int] | None = None,
    train_steps: int = 2,
    train_batch_size: int = 16,
    train_val_batch_size: int = 8,
    train_size: int = 32,
    train_num_workers: int = 2,
    allow_extended_confirmation: bool = False,
    proposal_primitive: str = "latent_motion_prior",
    proposal_routing_plan: Path | None = None,
    weight: float = 0.2,
    action_balance_blend: float = 0.5,
    action_balance_max_gain: float = 4.0,
    history_noise: float = 0.1,
    keep_tokens: int = 2,
    frontier_weight: float = 1.0,
    event_weight: float = 4.0,
    event_quantile: float = 0.75,
    event_visual_blend: float = 0.7,
    next_forcing_chunks: int = 2,
    next_forcing_steps: int = 1,
    next_forcing_lr: float = 0.00001,
    reward_weight: float = 0.5,
    inv_dyn_steps: int = 1,
    inv_dyn_lr: float = 0.00001,
    memory_slots: int = 16,
    memory_weight: float = 0.2,
    anchor_every: int = 8,
    anchor_weight: float = 0.2,
    guidance_start: float = 1.0,
    guidance_end: float = 1.5,
    wmsd_teacher_ema: float = 0.9,
    wmsd_steps: int = 1,
    wmsd_lr: float = 0.00001,
    self_forcing_rollout_horizon: int = 4,
    self_forcing_steps: int = 1,
    self_forcing_lr: float = 0.00001,
    trial_arm: str | None = None,
    trial_seed: int | None = None,
    max_accept_trajectories: int = 1,
    replication_count: int = 1,
    eval_inference_steps: int = 1,
    hook_timeout_seconds: float = 60.0,
    training_timeout_seconds: float = 1800.0,
    eval_timeout_seconds: float = 900.0,
    gpu_exclusivity_max_age_seconds: float = 3600.0,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> list[TrainingEvalJob]:
    """Build deterministic per-environment smoke commands for a limited campaign."""

    root = Path(repo_root).resolve()
    gpu_indices = _normalize_gpus(gpus)
    if not gpu_indices:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GPUS_EMPTY")
    if publish_settled_trials and m4_phase_gate_manifest is None:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_M4_GATE_REQUIRED")
    if publish_settled_trials and archive_db is None:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_ARCHIVE_REQUIRED")
    _training_monitor_policy_or_raise(
        train_steps=train_steps,
        batch_size=train_batch_size,
        train_size=train_size,
        allow_extended_confirmation=allow_extended_confirmation,
    )
    formal_ready_primitives = _formal_ready_primitives(
        primitive_materialization_gate_manifest,
        required=publish_settled_trials,
    )
    gate = _load_json_mapping(limited_gate_manifest, "TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_INVALID")
    included_envs, excluded_envs = _validate_limited_gate(gate)
    checkpoint_policy = _checkpoint_policy(gate)
    selected_envs = tuple(environments) if environments is not None else included_envs
    if not selected_envs:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_ENVS_EMPTY")
    invalid = sorted(set(selected_envs).difference(included_envs))
    if invalid:
        joined = ",".join(invalid)
        raise TrainingEvalLimitedCampaignError(f"TRAINING_EVAL_LIMITED_CAMPAIGN_ENV_NOT_IN_GATE:{joined}")
    if "cloth_move" in selected_envs and not checkpoint_policy["allow_official_current_checkpoint_warning"]:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_CLOTH_MOVE_REQUIRES_WARNING_GATE")
    diagnostic_horizons = _validated_diagnostic_eval_horizons(diagnostic_eval_horizons)
    if diagnostic_horizons is not None and publish_settled_trials:
        raise TrainingEvalLimitedCampaignError(
            "TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_DIAGNOSTIC_HORIZON_OVERRIDE_FORBIDDEN"
        )
    if diagnostic_horizons is not None and len(selected_envs) != 1:
        raise TrainingEvalLimitedCampaignError(
            "TRAINING_EVAL_LIMITED_CAMPAIGN_DIAGNOSTIC_HORIZON_OVERRIDE_REQUIRES_SINGLE_ENV"
        )
    failure_paths = _failure_report_paths(
        failure_manifest=_load_json_mapping(
            failure_report_manifest,
            "TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_MANIFEST_INVALID",
        ),
        goal_id=_goal_id(goal_config),
        included_envs=selected_envs,
    )
    horizons_by_env = _horizons_by_environment(repo_root=root, goal_config=goal_config)
    registry = PrimitiveRegistry.from_root(root) if proposal_primitive == AUTO_BY_DIAGNOSIS else None
    logs_dir = Path(output_root).resolve() / "logs"
    envs_dir = Path(output_root).resolve() / "envs"
    jobs: list[TrainingEvalJob] = []
    for index, env in enumerate(selected_envs):
        protocol_horizons = horizons_by_env.get(env)
        if not protocol_horizons:
            raise TrainingEvalLimitedCampaignError(f"TRAINING_EVAL_LIMITED_CAMPAIGN_HORIZONS_MISSING:{env}")
        horizons = diagnostic_horizons or protocol_horizons
        horizon_source = "diagnostic_override" if diagnostic_horizons is not None else "frozen_goal_protocol"
        gpu = gpu_indices[index % len(gpu_indices)]
        env_output = envs_dir / env
        log_path = logs_dir / f"{env}.log"
        env_proposal_primitive = _resolve_proposal_primitive(
            requested=proposal_primitive,
            failure_report=failure_paths[env],
            registry=registry,
            repo_root=root,
            formal_ready_primitives=formal_ready_primitives,
        )
        command = _training_eval_command(
            repo_root=root,
            environment=env,
            failure_report=failure_paths[env],
            goal_config=goal_config,
            output_root=env_output,
            runtime_python=runtime_python,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            dataset_freeze=dataset_freeze,
            heldout_protocol=heldout_protocol,
            gpu_index=gpu,
            horizons=horizons,
            proposal_primitive=env_proposal_primitive,
            proposal_routing_plan=proposal_routing_plan,
            weight=weight,
            action_balance_blend=action_balance_blend,
            action_balance_max_gain=action_balance_max_gain,
            history_noise=history_noise,
            keep_tokens=keep_tokens,
            frontier_weight=frontier_weight,
            event_weight=event_weight,
            event_quantile=event_quantile,
            event_visual_blend=event_visual_blend,
            next_forcing_chunks=next_forcing_chunks,
            next_forcing_steps=next_forcing_steps,
            next_forcing_lr=next_forcing_lr,
            reward_weight=reward_weight,
            inv_dyn_steps=inv_dyn_steps,
            inv_dyn_lr=inv_dyn_lr,
            memory_slots=memory_slots,
            memory_weight=memory_weight,
            anchor_every=anchor_every,
            anchor_weight=anchor_weight,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            wmsd_teacher_ema=wmsd_teacher_ema,
            wmsd_steps=wmsd_steps,
            wmsd_lr=wmsd_lr,
            self_forcing_rollout_horizon=self_forcing_rollout_horizon,
            self_forcing_steps=self_forcing_steps,
            self_forcing_lr=self_forcing_lr,
            trial_arm=trial_arm,
            trial_seed=trial_seed,
            train_steps=train_steps,
            train_batch_size=train_batch_size,
            train_val_batch_size=train_val_batch_size,
            train_size=train_size,
            train_num_workers=train_num_workers,
            max_accept_trajectories=max_accept_trajectories,
            replication_count=replication_count,
            eval_inference_steps=eval_inference_steps,
            hook_timeout_seconds=hook_timeout_seconds,
            training_timeout_seconds=training_timeout_seconds,
            eval_timeout_seconds=eval_timeout_seconds,
            gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
            m4_phase_gate_manifest=m4_phase_gate_manifest,
            publish_settled_trials=publish_settled_trials,
            archive_db=archive_db,
            cas_root=cas_root,
        )
        jobs.append(
            TrainingEvalJob(
                environment=env,
                gpu_index=gpu,
                horizons=tuple(horizons),
                horizon_source=horizon_source,
                proposal_primitive=env_proposal_primitive,
                output_root=env_output,
                log_path=log_path,
                command=command,
            )
        )
    return jobs


def run_training_eval_limited_campaign(
    *,
    repo_root: Path,
    limited_gate_manifest: Path,
    failure_report_manifest: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_exclusivity_audit_manifest: Path,
    gpus: Sequence[int],
    m4_phase_gate_manifest: Path | None = None,
    primitive_materialization_gate_manifest: Path | None = None,
    publish_settled_trials: bool = False,
    environments: Sequence[str] | None = None,
    diagnostic_eval_horizons: Sequence[int] | None = None,
    parallel_slots: int | None = None,
    campaign_id: str | None = None,
    train_steps: int = 2,
    train_batch_size: int = 16,
    train_val_batch_size: int = 8,
    train_size: int = 32,
    train_num_workers: int = 2,
    allow_extended_confirmation: bool = False,
    proposal_primitive: str = "latent_motion_prior",
    proposal_routing_plan: Path | None = None,
    weight: float = 0.2,
    action_balance_blend: float = 0.5,
    action_balance_max_gain: float = 4.0,
    history_noise: float = 0.1,
    keep_tokens: int = 2,
    frontier_weight: float = 1.0,
    event_weight: float = 4.0,
    event_quantile: float = 0.75,
    event_visual_blend: float = 0.7,
    next_forcing_chunks: int = 2,
    next_forcing_steps: int = 1,
    next_forcing_lr: float = 0.00001,
    reward_weight: float = 0.5,
    inv_dyn_steps: int = 1,
    inv_dyn_lr: float = 0.00001,
    memory_slots: int = 16,
    memory_weight: float = 0.2,
    anchor_every: int = 8,
    anchor_weight: float = 0.2,
    guidance_start: float = 1.0,
    guidance_end: float = 1.5,
    wmsd_teacher_ema: float = 0.9,
    wmsd_steps: int = 1,
    wmsd_lr: float = 0.00001,
    self_forcing_rollout_horizon: int = 4,
    self_forcing_steps: int = 1,
    self_forcing_lr: float = 0.00001,
    trial_arm: str | None = None,
    trial_seed: int | None = None,
    max_accept_trajectories: int = 1,
    replication_count: int = 1,
    eval_inference_steps: int = 1,
    hook_timeout_seconds: float = 60.0,
    training_timeout_seconds: float = 1800.0,
    eval_timeout_seconds: float = 900.0,
    gpu_exclusivity_max_age_seconds: float = 3600.0,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    poll_interval_seconds: float = 5.0,
    continue_on_failure: bool = True,
) -> dict[str, object]:
    """Launch per-env training+eval smokes with one active process per GPU slot."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_OUTPUT_EXISTS")
    campaign = campaign_id or destination.name
    slots = parallel_slots if parallel_slots is not None else len(_normalize_gpus(gpus))
    if slots < 1:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_PARALLEL_SLOTS_INVALID")
    if poll_interval_seconds <= 0:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_POLL_INTERVAL_INVALID")
    m4_authorization = None
    if m4_phase_gate_manifest is not None:
        m4_authorization = verify_m4_launch_allowed(m4_phase_gate_manifest).to_document()
    if publish_settled_trials and m4_authorization is None:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_M4_GATE_REQUIRED")
    if publish_settled_trials and archive_db is None:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_ARCHIVE_REQUIRED")
    training_monitor_policy = _training_monitor_policy_or_raise(
        train_steps=train_steps,
        batch_size=train_batch_size,
        train_size=train_size,
        allow_extended_confirmation=allow_extended_confirmation,
    )
    materialization_policy = _formal_materialization_policy(
        primitive_materialization_gate_manifest,
        required=publish_settled_trials,
    )
    jobs = build_training_eval_jobs(
        repo_root=repo_root,
        limited_gate_manifest=limited_gate_manifest,
        failure_report_manifest=failure_report_manifest,
        goal_config=goal_config,
        output_root=destination,
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        dataset_freeze=dataset_freeze,
        heldout_protocol=heldout_protocol,
        gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
        gpus=gpus,
        campaign_id=campaign,
        m4_phase_gate_manifest=m4_phase_gate_manifest,
        primitive_materialization_gate_manifest=primitive_materialization_gate_manifest,
        publish_settled_trials=publish_settled_trials,
        environments=environments,
        diagnostic_eval_horizons=diagnostic_eval_horizons,
        train_steps=train_steps,
        train_batch_size=train_batch_size,
        train_val_batch_size=train_val_batch_size,
        train_size=train_size,
        train_num_workers=train_num_workers,
        allow_extended_confirmation=allow_extended_confirmation,
        proposal_primitive=proposal_primitive,
        proposal_routing_plan=proposal_routing_plan,
        weight=weight,
        action_balance_blend=action_balance_blend,
        action_balance_max_gain=action_balance_max_gain,
        history_noise=history_noise,
        keep_tokens=keep_tokens,
        frontier_weight=frontier_weight,
        event_weight=event_weight,
        event_quantile=event_quantile,
        event_visual_blend=event_visual_blend,
        next_forcing_chunks=next_forcing_chunks,
        next_forcing_steps=next_forcing_steps,
        next_forcing_lr=next_forcing_lr,
        reward_weight=reward_weight,
        inv_dyn_steps=inv_dyn_steps,
        inv_dyn_lr=inv_dyn_lr,
        memory_slots=memory_slots,
        memory_weight=memory_weight,
        anchor_every=anchor_every,
        anchor_weight=anchor_weight,
        guidance_start=guidance_start,
        guidance_end=guidance_end,
        wmsd_teacher_ema=wmsd_teacher_ema,
        wmsd_steps=wmsd_steps,
        wmsd_lr=wmsd_lr,
        self_forcing_rollout_horizon=self_forcing_rollout_horizon,
        self_forcing_steps=self_forcing_steps,
        self_forcing_lr=self_forcing_lr,
        trial_arm=trial_arm,
        trial_seed=trial_seed,
        max_accept_trajectories=max_accept_trajectories,
        replication_count=replication_count,
        eval_inference_steps=eval_inference_steps,
        hook_timeout_seconds=hook_timeout_seconds,
        training_timeout_seconds=training_timeout_seconds,
        eval_timeout_seconds=eval_timeout_seconds,
        gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
        archive_db=archive_db,
        cas_root=cas_root,
    )
    gate = _load_json_mapping(limited_gate_manifest, "TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_INVALID")
    checkpoint_policy = _checkpoint_policy(gate)
    destination.mkdir(mode=0o700, parents=True)
    (destination / "logs").mkdir(mode=0o700)
    (destination / "envs").mkdir(mode=0o700)
    status_path = destination / "status.json"
    manifest_path = destination / "manifest.json"
    records: dict[str, dict[str, object]] = {
        job.environment: {
            "environment": job.environment,
            "gpu_index": job.gpu_index,
            "horizons": list(job.horizons),
            "horizon_source": job.horizon_source,
            "proposal_primitive": job.proposal_primitive,
            "state": "pending",
            "output_root": str(job.output_root),
            "log_path": str(job.log_path),
            "command": list(job.command),
        }
        for job in jobs
    }
    started_at = _now()
    _write_status(
        status_path,
        campaign_id=campaign,
        state="running",
        started_at=started_at,
        completed_at=None,
        records=records,
    )
    queue = list(jobs)
    running: dict[subprocess.Popen[bytes], tuple[TrainingEvalJob, Any, float]] = {}
    busy_gpus: set[int] = set()
    try:
        while queue or running:
            for job in list(queue):
                if len(running) >= slots:
                    break
                if job.gpu_index in busy_gpus:
                    continue
                queue.remove(job)
                job.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                handle = open(job.log_path, "ab")
                handle.write(f"[{_now()}] launch {' '.join(job.command)}\n".encode("utf-8"))
                handle.flush()
                process = subprocess.Popen(job.command, cwd=Path(repo_root).resolve(), stdout=handle, stderr=subprocess.STDOUT)
                running[process] = (job, handle, time.time())
                busy_gpus.add(job.gpu_index)
                records[job.environment].update({"state": "running", "pid": process.pid, "started_at": _now()})
                _write_status(
                    status_path,
                    campaign_id=campaign,
                    state="running",
                    started_at=started_at,
                    completed_at=None,
                    records=records,
                )
            time.sleep(poll_interval_seconds)
            for process, (job, handle, launch_time) in list(running.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                handle.write(f"[{_now()}] exit_code={exit_code}\n".encode("utf-8"))
                handle.close()
                running.pop(process)
                busy_gpus.discard(job.gpu_index)
                elapsed = time.time() - launch_time
                env_manifest = job.output_root / "manifest.json"
                records[job.environment].update(
                    {
                        "state": "ready" if exit_code == 0 and env_manifest.is_file() else "failed",
                        "exit_code": exit_code,
                        "duration_seconds": elapsed,
                        "completed_at": _now(),
                        "manifest_path": str(env_manifest) if env_manifest.is_file() else None,
                    }
                )
                _write_status(
                    status_path,
                    campaign_id=campaign,
                    state="running",
                    started_at=started_at,
                    completed_at=None,
                    records=records,
                )
                if exit_code != 0 and not continue_on_failure:
                    queue.clear()
                    for child, (child_job, child_handle, _) in list(running.items()):
                        child.terminate()
                        child_handle.write(f"[{_now()}] terminated_after_failure={job.environment}\n".encode("utf-8"))
                        child_handle.close()
                        records[child_job.environment].update({"state": "cancelled", "completed_at": _now()})
                        busy_gpus.discard(child_job.gpu_index)
                        running.pop(child)
                    break
    except BaseException:
        for process, (job, handle, _) in list(running.items()):
            process.terminate()
            handle.write(f"[{_now()}] terminated_by_campaign_exception\n".encode("utf-8"))
            handle.close()
            records[job.environment].update({"state": "cancelled", "completed_at": _now()})
        raise
    completed_at = _now()
    failed = [record for record in records.values() if record["state"] != "ready"]
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-training-eval-limited-campaign-manifest",
        "state": "ready" if not failed else "checks_failed",
        "campaign_id": campaign,
        "scope_type": "formal_m4_gpu_backed" if publish_settled_trials else "limited_pilot_gpu_backed",
        "m4_launch_allowed": bool(m4_authorization is not None),
        "formal_training_allowed": bool(publish_settled_trials),
        "m4_launch_gate": m4_authorization,
        "primitive_materialization_gate": materialization_policy,
        "checkpoint_policy": checkpoint_policy,
        "training_monitor_policy": training_monitor_policy,
        "cloth_move_excluded": "cloth_move" not in records,
        "cloth_move_included": "cloth_move" in records,
        "started_at": started_at,
        "completed_at": completed_at,
        "parallel_slots": slots,
        "gpus": list(_normalize_gpus(gpus)),
        "train_steps": train_steps,
        "train_batch_size": train_batch_size,
        "train_val_batch_size": train_val_batch_size,
        "train_size": train_size,
        "train_num_workers": train_num_workers,
        "proposal_primitive": proposal_primitive,
        "proposal_primitive_request": proposal_primitive,
        "proposal_routing_plan": str(Path(proposal_routing_plan).resolve()) if proposal_routing_plan else None,
        "proposal_primitives_by_environment": {
            str(record["environment"]): str(record["proposal_primitive"])
            for record in records.values()
        },
        "history_noise": history_noise,
        "keep_tokens": keep_tokens,
        "frontier_weight": frontier_weight,
        "event_weight": event_weight,
        "event_quantile": event_quantile,
        "event_visual_blend": event_visual_blend,
        "next_forcing_chunks": next_forcing_chunks,
        "next_forcing_steps": next_forcing_steps,
        "next_forcing_lr": next_forcing_lr,
        "reward_weight": reward_weight,
        "inv_dyn_steps": inv_dyn_steps,
        "inv_dyn_lr": inv_dyn_lr,
        "memory_slots": memory_slots,
        "memory_weight": memory_weight,
        "anchor_every": anchor_every,
        "anchor_weight": anchor_weight,
        "guidance_start": guidance_start,
        "guidance_end": guidance_end,
        "wmsd_teacher_ema": wmsd_teacher_ema,
        "wmsd_steps": wmsd_steps,
        "wmsd_lr": wmsd_lr,
        "self_forcing_rollout_horizon": self_forcing_rollout_horizon,
        "self_forcing_steps": self_forcing_steps,
        "self_forcing_lr": self_forcing_lr,
        **({"trial_arm": trial_arm} if trial_arm is not None else {}),
        **({"seed": trial_seed} if trial_seed is not None else {}),
        "replication_count": replication_count,
        "eval_inference_steps": eval_inference_steps,
        "diagnostic_eval_horizons_override": (
            list(_validated_diagnostic_eval_horizons(diagnostic_eval_horizons) or ())
        ),
        "diagnostic_horizon_claim_boundary": (
            "A diagnostic horizon override may route a screen to the frozen official gate, "
            "but it cannot publish a formal verdict or alter the goal protocol."
        ),
        "gpu_exclusivity_audit_manifest": str(Path(gpu_exclusivity_audit_manifest).resolve()),
        "limited_gate_manifest": str(Path(limited_gate_manifest).resolve()),
        "goal_config": str(Path(goal_config).resolve()),
        "environment_count": len(records),
        "ready_environment_count": len(records) - len(failed),
        "failed_environment_count": len(failed),
        "records": list(records.values()),
    }
    _write_json_atomic(manifest_path, manifest)
    _write_status(
        status_path,
        campaign_id=campaign,
        state=str(manifest["state"]),
        started_at=started_at,
        completed_at=completed_at,
        records=records,
    )
    return manifest


def _training_eval_command(
    *,
    repo_root: Path,
    environment: str,
    failure_report: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_index: int,
    horizons: Sequence[int],
    proposal_primitive: str,
    proposal_routing_plan: Path | None,
    weight: float,
    action_balance_blend: float,
    action_balance_max_gain: float,
    history_noise: float,
    keep_tokens: int,
    frontier_weight: float,
    event_weight: float,
    event_quantile: float,
    event_visual_blend: float,
    next_forcing_chunks: int,
    next_forcing_steps: int,
    next_forcing_lr: float,
    reward_weight: float,
    inv_dyn_steps: int,
    inv_dyn_lr: float,
    memory_slots: int,
    memory_weight: float,
    anchor_every: int,
    anchor_weight: float,
    guidance_start: float,
    guidance_end: float,
    wmsd_teacher_ema: float,
    wmsd_steps: int,
    wmsd_lr: float,
    self_forcing_rollout_horizon: int,
    self_forcing_steps: int,
    self_forcing_lr: float,
    trial_arm: str | None,
    trial_seed: int | None,
    train_steps: int,
    train_batch_size: int,
    train_val_batch_size: int,
    train_size: int,
    train_num_workers: int,
    max_accept_trajectories: int,
    replication_count: int,
    eval_inference_steps: int,
    hook_timeout_seconds: float,
    training_timeout_seconds: float,
    eval_timeout_seconds: float,
    gpu_exclusivity_audit_manifest: Path,
    gpu_exclusivity_max_age_seconds: float,
    m4_phase_gate_manifest: Path | None,
    publish_settled_trials: bool,
    archive_db: Path | None,
    cas_root: Path | None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "wmloop.orchestrator_training_eval_smoke",
        "run",
        "--repo-root",
        str(repo_root),
        "--environment",
        environment,
        "--failure-report",
        str(failure_report),
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
        "--gpu-index",
        str(gpu_index),
        "--proposal-primitive",
        proposal_primitive,
        "--weight",
        str(weight),
        "--action-balance-blend",
        str(action_balance_blend),
        "--action-balance-max-gain",
        str(action_balance_max_gain),
        "--history-noise",
        str(history_noise),
        "--keep-tokens",
        str(keep_tokens),
        "--frontier-weight",
        str(frontier_weight),
        "--event-weight",
        str(event_weight),
        "--event-quantile",
        str(event_quantile),
        "--event-visual-blend",
        str(event_visual_blend),
        "--next-forcing-chunks",
        str(next_forcing_chunks),
        "--next-forcing-steps",
        str(next_forcing_steps),
        "--next-forcing-lr",
        str(next_forcing_lr),
        "--reward-weight",
        str(reward_weight),
        "--inv-dyn-steps",
        str(inv_dyn_steps),
        "--inv-dyn-lr",
        str(inv_dyn_lr),
        "--memory-slots",
        str(memory_slots),
        "--memory-weight",
        str(memory_weight),
        "--anchor-every",
        str(anchor_every),
        "--anchor-weight",
        str(anchor_weight),
        "--guidance-start",
        str(guidance_start),
        "--guidance-end",
        str(guidance_end),
        "--wmsd-teacher-ema",
        str(wmsd_teacher_ema),
        "--wmsd-steps",
        str(wmsd_steps),
        "--wmsd-lr",
        str(wmsd_lr),
        "--self-forcing-rollout-horizon",
        str(self_forcing_rollout_horizon),
        "--self-forcing-steps",
        str(self_forcing_steps),
        "--self-forcing-lr",
        str(self_forcing_lr),
        "--train-steps",
        str(train_steps),
        "--train-batch-size",
        str(train_batch_size),
        "--train-val-batch-size",
        str(train_val_batch_size),
        "--train-size",
        str(train_size),
        "--train-num-workers",
        str(train_num_workers),
        "--eval-horizons",
        *(str(value) for value in horizons),
        "--max-accept-trajectories",
        str(max_accept_trajectories),
        "--replication-count",
        str(replication_count),
        "--eval-inference-steps",
        str(eval_inference_steps),
        "--hook-timeout-seconds",
        str(hook_timeout_seconds),
        "--training-timeout-seconds",
        str(training_timeout_seconds),
        "--eval-timeout-seconds",
        str(eval_timeout_seconds),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_exclusivity_audit_manifest),
        "--gpu-exclusivity-max-age-seconds",
        str(gpu_exclusivity_max_age_seconds),
    ]
    if trial_arm is not None:
        command.extend(["--trial-arm", trial_arm])
    if proposal_routing_plan is not None:
        command.extend(["--proposal-routing-plan", str(Path(proposal_routing_plan).resolve(strict=True))])
    if trial_seed is not None:
        command.extend(["--trial-seed", str(trial_seed)])
    if train_steps >= 512:
        command.append("--keep-temp-on-failure")
    if m4_phase_gate_manifest is not None:
        command.extend(["--m4-phase-gate-manifest", str(m4_phase_gate_manifest)])
    if publish_settled_trials:
        command.append("--publish-settled-trial")
    if archive_db is not None:
        command.extend(["--archive-db", str(archive_db)])
    if cas_root is not None:
        command.extend(["--cas-root", str(cas_root)])
    return tuple(command)


def _training_monitor_policy_or_raise(
    *,
    train_steps: int,
    batch_size: int,
    train_size: int,
    allow_extended_confirmation: bool,
) -> dict[str, object]:
    policy = training_monitor_policy_document(
        train_steps=train_steps,
        batch_size=batch_size,
        train_size=train_size,
        allow_extended_confirmation=allow_extended_confirmation,
    )
    if policy["state"] != "ready":
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_TRAIN_STEPS_EXCEED_DEFAULT_CAP")
    return policy


def _validate_limited_gate(gate: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if gate.get("artifact_type") != "wmloop-limited-campaign-gate-manifest" or gate.get("state") != "ready":
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_INVALID")
    if gate.get("limited_campaign_allowed") is not True:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_NOT_ALLOWED")
    if gate.get("m4_launch_allowed") is not False or gate.get("formal_training_allowed") is not False:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_SCOPE_INVALID")
    included = _string_tuple(gate.get("included_envs"), "TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_INCLUDED_INVALID")
    excluded = _string_tuple(gate.get("excluded_envs"), "TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_EXCLUDED_INVALID")
    if not included or set(included) & set(excluded):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GATE_SCOPE_INVALID")
    return included, excluded


def _checkpoint_policy(gate: Mapping[str, object]) -> dict[str, object]:
    value = gate.get("checkpoint_policy")
    if value is None:
        return {
            "allow_official_current_checkpoint_warning": False,
            "claim_boundary": "Every included environment must pass the strict checkpoint-step audit.",
        }
    if not isinstance(value, Mapping):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_CHECKPOINT_POLICY_INVALID")
    allow = value.get("allow_official_current_checkpoint_warning", False)
    if not isinstance(allow, bool):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_CHECKPOINT_POLICY_INVALID")
    policy = dict(value)
    policy["allow_official_current_checkpoint_warning"] = allow
    claim_boundary = policy.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        policy["claim_boundary"] = (
            "Paired-delta closed-loop trials may use official-current checkpoints with provenance warnings; "
            "official 100k reproduction claims remain disallowed for warned environments."
            if allow
            else "Every included environment must pass the strict checkpoint-step audit."
        )
    return policy


def _resolve_proposal_primitive(
    *,
    requested: str,
    failure_report: Path,
    registry: PrimitiveRegistry | None,
    repo_root: Path,
    formal_ready_primitives: frozenset[str] | None = None,
) -> str:
    if requested != AUTO_BY_DIAGNOSIS:
        _require_formal_ready_primitive(requested, formal_ready_primitives=formal_ready_primitives)
        return requested
    if registry is None:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_AUTO_REGISTRY_MISSING")
    failure = _load_json_mapping(failure_report, "TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_REPORT_INVALID")
    bank_route = _signature_bank_preferred_runnable_primitive(
        repo_root=repo_root,
        environment=str(failure.get("env") or ""),
        registry=registry,
        formal_ready_primitives=formal_ready_primitives,
    )
    if bank_route is not None:
        return bank_route
    dominant = str(failure.get("dominant_failure") or "")
    allowed = _allowed_runnable_primitives(failure_report=failure, registry=registry)
    preferences = AUTO_PRIMITIVE_PREFERENCES.get(dominant, AUTO_ROUTABLE_PRIMITIVES)
    for primitive in (*preferences, *AUTO_ROUTABLE_PRIMITIVES):
        if formal_ready_primitives is not None and primitive not in formal_ready_primitives:
            continue
        if primitive in allowed:
            return primitive
    env = failure.get("env")
    raise TrainingEvalLimitedCampaignError(
        f"TRAINING_EVAL_LIMITED_CAMPAIGN_AUTO_PRIMITIVE_UNAVAILABLE:{env}:{dominant}"
    )


def _signature_bank_preferred_runnable_primitive(
    *,
    repo_root: Path,
    environment: str,
    registry: PrimitiveRegistry,
    formal_ready_primitives: frozenset[str] | None = None,
) -> str | None:
    if not environment:
        return None
    bank_path = Path(repo_root) / "results" / "reports" / "failure-signature-bank-r1" / "failure-signature-bank.json"
    if not bank_path.is_file() or bank_path.is_symlink():
        return None
    try:
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-failure-signature-bank":
        return None
    rows = payload.get("primitive_routing")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("environment") != environment:
            continue
        if row.get("routing_decision") not in {"stage_canary_after_diagnostic_probe", "retain_as_source_exemplar"}:
            continue
        primitive = row.get("primitive")
        if not isinstance(primitive, str) or primitive not in AUTO_ROUTABLE_PRIMITIVES:
            continue
        if formal_ready_primitives is not None and primitive not in formal_ready_primitives:
            continue
        try:
            registry.manifest(primitive)
        except Exception:
            continue
        return primitive
    return None


def _allowed_runnable_primitives(
    *,
    failure_report: Mapping[str, object],
    registry: PrimitiveRegistry,
) -> set[str]:
    allowed: set[str] = set()
    for failure in _routing_failures(failure_report=failure_report, registry=registry):
        for manifest in registry.available_for(failure=failure):
            if manifest.name in RUNNABLE_PRIMITIVES:
                allowed.add(manifest.name)
    return allowed


def _formal_ready_primitives(
    manifest_path: Path | None,
    *,
    required: bool,
) -> frozenset[str] | None:
    policy = _formal_materialization_policy(manifest_path, required=required)
    if policy is None:
        return None
    ready = policy["closed_loop_ready_primitives"]
    if not isinstance(ready, list) or not all(isinstance(item, str) for item in ready):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_MATERIALIZATION_GATE_INVALID")
    return frozenset(str(item) for item in ready)


def _formal_materialization_policy(
    manifest_path: Path | None,
    *,
    required: bool,
) -> dict[str, object] | None:
    if manifest_path is None:
        if required:
            raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_MATERIALIZATION_GATE_REQUIRED")
        return None
    manifest = _load_json_mapping(manifest_path, "TRAINING_EVAL_LIMITED_CAMPAIGN_MATERIALIZATION_GATE_INVALID")
    if manifest.get("artifact_type") != "wmloop-primitive-materialization-gate-manifest":
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_MATERIALIZATION_GATE_INVALID")
    ready = manifest.get("closed_loop_ready_primitives")
    if not isinstance(ready, list) or not all(isinstance(item, str) and item for item in ready):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_MATERIALIZATION_GATE_INVALID")
    if required and not ready:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FORMAL_MATERIALIZATION_READY_EMPTY")
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-formal-primitive-materialization-policy",
        "state": "ready",
        "manifest_path": str(Path(manifest_path).resolve()),
        "source_gate_state": manifest.get("state"),
        "closed_loop_ready_primitives": list(dict.fromkeys(str(item) for item in ready)),
        "blockers": manifest.get("blockers", []),
    }


def _require_formal_ready_primitive(
    primitive: str,
    *,
    formal_ready_primitives: frozenset[str] | None,
) -> None:
    if formal_ready_primitives is not None and primitive not in formal_ready_primitives:
        raise TrainingEvalLimitedCampaignError(f"TRAINING_EVAL_LIMITED_CAMPAIGN_PRIMITIVE_NOT_MATERIALIZED:{primitive}")


def _routing_failures(
    *,
    failure_report: Mapping[str, object],
    registry: PrimitiveRegistry,
) -> tuple[str, ...]:
    dominant = str(failure_report.get("dominant_failure") or "")
    if dominant != "mixed":
        return (dominant,)
    candidates = failure_report.get("dominant_failure_candidates")
    ordered: list[str] = []
    if isinstance(candidates, list):
        ordered.extend(str(candidate) for candidate in candidates if isinstance(candidate, str) and candidate != "mixed")
    if not ordered:
        ordered.extend(_registered_failure_types(registry))
    return tuple(dict.fromkeys(ordered))


def _registered_failure_types(registry: PrimitiveRegistry) -> tuple[str, ...]:
    failures = {
        failure
        for name in registry.names()
        for failure in registry.manifest(name).targets_failures
        if failure != "mixed"
    }
    return tuple(sorted(failures))


def _failure_report_paths(
    *,
    failure_manifest: Mapping[str, object],
    goal_id: str,
    included_envs: Sequence[str],
) -> dict[str, Path]:
    if failure_manifest.get("artifact_type") != "wmloop-m1-raw-failure-report-batch":
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_MANIFEST_INVALID")
    if failure_manifest.get("goal_id") != goal_id:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_GOAL_MISMATCH")
    reports = failure_manifest.get("reports")
    if not isinstance(reports, list):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_REPORTS_INVALID")
    by_env: dict[str, Path] = {}
    for item in reports:
        if not isinstance(item, Mapping):
            raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_REPORTS_INVALID")
        env = item.get("environment")
        path = item.get("failure_report_path")
        if isinstance(env, str) and isinstance(path, str) and path:
            by_env[env] = Path(path)
    missing = [env for env in included_envs if env not in by_env]
    if missing:
        joined = ",".join(missing)
        raise TrainingEvalLimitedCampaignError(f"TRAINING_EVAL_LIMITED_CAMPAIGN_FAILURE_REPORT_MISSING:{joined}")
    return {env: by_env[env] for env in included_envs}


def _horizons_by_environment(*, repo_root: Path, goal_config: Path) -> dict[str, tuple[int, ...]]:
    goal = load_yaml_document(goal_config)
    goal_horizons = tuple(int(value) for value in goal.get("horizons", []))
    eval_protocol = goal.get("eval_protocol")
    if not isinstance(eval_protocol, Mapping):
        return {}
    ladder_path_value = eval_protocol.get("horizon_ladder_path")
    if not isinstance(ladder_path_value, str) or not ladder_path_value:
        envs = _string_tuple(goal.get("envs"), "TRAINING_EVAL_LIMITED_CAMPAIGN_GOAL_ENVS_INVALID")
        return {env: goal_horizons for env in envs}
    ladder_path = Path(ladder_path_value)
    if not ladder_path.is_absolute():
        ladder_path = repo_root / ladder_path
    ladder = load_yaml_document(ladder_path)
    raw = ladder.get("horizons_by_environment")
    if not isinstance(raw, Mapping):
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_LADDER_INVALID")
    horizons: dict[str, tuple[int, ...]] = {}
    for env, values in raw.items():
        if not isinstance(env, str) or not isinstance(values, list):
            raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_LADDER_INVALID")
        parsed = tuple(int(value) for value in values)
        if not parsed:
            raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_LADDER_INVALID")
        horizons[env] = parsed
    return horizons


def _validated_diagnostic_eval_horizons(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    parsed = tuple(int(value) for value in values)
    if (
        not parsed
        or len(set(parsed)) != len(parsed)
        or tuple(sorted(parsed)) != parsed
        or any(value < 2 for value in parsed)
    ):
        raise TrainingEvalLimitedCampaignError(
            "TRAINING_EVAL_LIMITED_CAMPAIGN_DIAGNOSTIC_HORIZONS_INVALID"
        )
    return parsed


def _goal_id(goal_config: Path) -> str:
    goal = load_yaml_document(goal_config)
    goal_id = goal.get("goal_id")
    if not isinstance(goal_id, str) or not goal_id:
        raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GOAL_INVALID")
    return goal_id


def _normalize_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for gpu in gpus:
        value = int(gpu)
        if value < 0:
            raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_GPU_INVALID")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _string_tuple(value: object, code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TrainingEvalLimitedCampaignError(code)
    return tuple(value)


def _load_json_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingEvalLimitedCampaignError(code) from exc
    if not isinstance(payload, Mapping):
        raise TrainingEvalLimitedCampaignError(code)
    return payload


def _write_status(
    path: Path,
    *,
    campaign_id: str,
    state: str,
    started_at: str,
    completed_at: str | None,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "artifact_type": "wmloop-training-eval-limited-campaign-status",
            "campaign_id": campaign_id,
            "state": state,
            "started_at": started_at,
            "completed_at": completed_at,
            "updated_at": _now(),
            "records": list(records.values()),
        },
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run GPU-backed training+eval smokes for the limited env set")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--limited-gate-manifest", type=Path, required=True)
    run.add_argument("--failure-report-manifest", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--dataset-freeze", type=Path, required=True)
    run.add_argument("--heldout-protocol", type=Path, required=True)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path, required=True)
    run.add_argument("--gpus", type=int, nargs="+", required=True)
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    run.add_argument("--primitive-materialization-gate-manifest", type=Path)
    run.add_argument("--publish-settled-trials", action="store_true")
    run.add_argument("--environment", dest="environments", action="append")
    run.add_argument("--diagnostic-eval-horizons", type=int, nargs="+")
    run.add_argument("--parallel-slots", type=int)
    run.add_argument("--campaign-id")
    run.add_argument("--train-steps", type=int, default=2)
    run.add_argument("--train-batch-size", type=int, default=16)
    run.add_argument("--train-val-batch-size", type=int, default=8)
    run.add_argument("--train-size", type=int, default=32)
    run.add_argument("--train-num-workers", type=int, default=2)
    run.add_argument("--allow-extended-confirmation", action="store_true")
    run.add_argument("--proposal-primitive", default="latent_motion_prior")
    run.add_argument("--proposal-routing-plan", type=Path)
    run.add_argument("--weight", type=float, default=0.2)
    run.add_argument("--action-balance-blend", type=float, default=0.5)
    run.add_argument("--action-balance-max-gain", type=float, default=4.0)
    run.add_argument("--history-noise", type=float, default=0.1)
    run.add_argument("--keep-tokens", type=int, default=2)
    run.add_argument("--frontier-weight", type=float, default=1.0)
    run.add_argument("--event-weight", type=float, default=4.0)
    run.add_argument("--event-quantile", type=float, default=0.75)
    run.add_argument("--event-visual-blend", type=float, default=0.7)
    run.add_argument("--next-forcing-chunks", type=int, default=2)
    run.add_argument("--next-forcing-steps", type=int, default=1)
    run.add_argument("--next-forcing-lr", type=float, default=0.00001)
    run.add_argument("--reward-weight", type=float, default=0.5)
    run.add_argument("--inv-dyn-steps", type=int, default=1)
    run.add_argument("--inv-dyn-lr", type=float, default=0.00001)
    run.add_argument("--memory-slots", type=int, default=16)
    run.add_argument("--memory-weight", type=float, default=0.2)
    run.add_argument("--anchor-every", type=int, default=8)
    run.add_argument("--anchor-weight", type=float, default=0.2)
    run.add_argument("--guidance-start", type=float, default=1.0)
    run.add_argument("--guidance-end", type=float, default=1.5)
    run.add_argument("--wmsd-teacher-ema", type=float, default=0.9)
    run.add_argument("--wmsd-steps", type=int, default=1)
    run.add_argument("--wmsd-lr", type=float, default=0.00001)
    run.add_argument("--self-forcing-rollout-horizon", type=int, default=4)
    run.add_argument("--self-forcing-steps", type=int, default=1)
    run.add_argument("--self-forcing-lr", type=float, default=0.00001)
    run.add_argument("--trial-arm", choices=("prior", "cold_start", "shuffled_prior"))
    run.add_argument("--trial-seed", type=int)
    run.add_argument("--max-accept-trajectories", type=int, default=1)
    run.add_argument("--replication-count", type=int, default=1)
    run.add_argument("--eval-inference-steps", type=int, default=1)
    run.add_argument("--hook-timeout-seconds", type=float, default=60.0)
    run.add_argument("--training-timeout-seconds", type=float, default=1800.0)
    run.add_argument("--eval-timeout-seconds", type=float, default=900.0)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=3600.0)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--poll-interval-seconds", type=float, default=5.0)
    run.add_argument("--stop-on-first-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_training_eval_limited_campaign(
            repo_root=args.repo_root,
            limited_gate_manifest=args.limited_gate_manifest,
            failure_report_manifest=args.failure_report_manifest,
            goal_config=args.goal_config,
            output_root=args.output_root,
            runtime_python=args.runtime_python,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            dataset_freeze=args.dataset_freeze,
            heldout_protocol=args.heldout_protocol,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpus=args.gpus,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
            primitive_materialization_gate_manifest=args.primitive_materialization_gate_manifest,
            publish_settled_trials=args.publish_settled_trials,
            environments=args.environments,
            diagnostic_eval_horizons=args.diagnostic_eval_horizons,
            parallel_slots=args.parallel_slots,
            campaign_id=args.campaign_id,
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
            train_val_batch_size=args.train_val_batch_size,
            train_size=args.train_size,
            train_num_workers=args.train_num_workers,
            allow_extended_confirmation=args.allow_extended_confirmation,
            proposal_primitive=args.proposal_primitive,
            proposal_routing_plan=args.proposal_routing_plan,
            weight=args.weight,
            action_balance_blend=args.action_balance_blend,
            action_balance_max_gain=args.action_balance_max_gain,
            history_noise=args.history_noise,
            keep_tokens=args.keep_tokens,
            frontier_weight=args.frontier_weight,
            event_weight=args.event_weight,
            event_quantile=args.event_quantile,
            event_visual_blend=args.event_visual_blend,
            next_forcing_chunks=args.next_forcing_chunks,
            next_forcing_steps=args.next_forcing_steps,
            next_forcing_lr=args.next_forcing_lr,
            reward_weight=args.reward_weight,
            inv_dyn_steps=args.inv_dyn_steps,
            inv_dyn_lr=args.inv_dyn_lr,
            memory_slots=args.memory_slots,
            memory_weight=args.memory_weight,
            anchor_every=args.anchor_every,
            anchor_weight=args.anchor_weight,
            guidance_start=args.guidance_start,
            guidance_end=args.guidance_end,
            wmsd_teacher_ema=args.wmsd_teacher_ema,
            wmsd_steps=args.wmsd_steps,
            wmsd_lr=args.wmsd_lr,
            self_forcing_rollout_horizon=args.self_forcing_rollout_horizon,
            self_forcing_steps=args.self_forcing_steps,
            self_forcing_lr=args.self_forcing_lr,
            trial_arm=args.trial_arm,
            trial_seed=args.trial_seed,
            max_accept_trajectories=args.max_accept_trajectories,
            replication_count=args.replication_count,
            eval_inference_steps=args.eval_inference_steps,
            hook_timeout_seconds=args.hook_timeout_seconds,
            training_timeout_seconds=args.training_timeout_seconds,
            eval_timeout_seconds=args.eval_timeout_seconds,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            poll_interval_seconds=args.poll_interval_seconds,
            continue_on_failure=not args.stop_on_first_failure,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise TrainingEvalLimitedCampaignError("TRAINING_EVAL_LIMITED_CAMPAIGN_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
