#!/usr/bin/env python3
"""Salvage a trained ACWM screen after a known post-training probe failure.

The salvage path is intentionally narrow: it accepts only campaigns whose
training and checkpoint checks passed and whose first failed check matches the
fixed raw-probe ``gpu_index`` plumbing bug. It never resumes or reruns training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import load_yaml_document
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.primitive_runtime_smoke import _runtime_environment
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import WorktreeSandbox
from wmloop.orchestrator_training_eval_smoke import (
    _action_following_records_with_refs,
    _copy_candidate_runtime,
    _environment_spec,
    _evaluation_summary,
    _goal_action_following_threshold,
    _goal_id,
    _goal_metric_name,
    _horizon_probe_command,
    _load_action_following_probe_pair,
    _load_probe_replications,
    _put_file,
    _raw_probe_measure_command,
    _replication_records_with_refs,
    _validate_probe_runtime_binding,
    _verification_evidence_from_probe_replications,
    _write_report_bundle,
)
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.vendor import verify_vendor_checkout
from wmloop.verify.judge import judge


FAILURE_FINGERPRINT = "TypeError: _runtime_device() missing 1 required keyword-only argument: 'gpu_index'"
FAILURE_STAGE_FINGERPRINT = "M3_TRAINING_EVAL_SMOKE_CHECK_FAILED:baseline_action_following_probe:1"
REQUIRED_PASSED_LABELS = {
    "accept_cohort_view_r01",
    "runtime_hook_unit",
    "acwm_train_smoke",
    "candidate_checkpoint_present",
}


class FailedScreenSalvageError(RuntimeError):
    """A failed screen did not satisfy the narrow salvage contract."""


@dataclass(frozen=True)
class SalvageSource:
    campaign_root: Path
    environment: str
    primitive: str
    seed: int
    proposal_id: str
    temp_root: Path
    checkpoint_path: Path
    training_config_path: Path
    training_sidecar_path: Path
    budget_db_path: Path
    session_path: Path
    source_revision: str
    original_attempts: tuple[dict[str, object], ...]
    record_command: tuple[str, ...]
    record_log_path: Path
    eval_horizons: tuple[int, ...]
    max_accept_trajectories: int
    replication_count: int
    eval_inference_steps: int
    train_steps: int
    train_batch_size: int
    train_size: int
    train_num_workers: int
    runtime_python: Path
    data_root: Path
    checkpoint_root: Path
    goal_config: Path
    archive_db: Path | None
    cas_root: Path


def discover_salvage_source(*, campaign_root: Path, environment: str | None = None) -> SalvageSource:
    campaign = Path(campaign_root).resolve(strict=True)
    manifest = _load_json(campaign / "manifest.json")
    if manifest.get("artifact_type") != "wmloop-training-eval-limited-campaign-manifest":
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_CAMPAIGN_TYPE_INVALID")
    if manifest.get("state") != "checks_failed" or manifest.get("failed_environment_count") != 1:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_CAMPAIGN_STATE_INVALID")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_RECORD_INVALID")
    record = records[0]
    env = str(environment or record.get("environment") or "")
    if not env or record.get("environment") != env or record.get("state") != "failed":
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_ENVIRONMENT_INVALID")
    output_root = Path(str(record.get("output_root") or "")).resolve()
    expected_output = campaign / "envs" / env
    if output_root != expected_output or output_root.exists() or output_root.is_symlink():
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_OUTPUT_STATE_INVALID")
    log_path = Path(str(record.get("log_path") or "")).resolve(strict=True)
    if FAILURE_STAGE_FINGERPRINT not in log_path.read_text(encoding="utf-8", errors="replace"):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_FAILURE_STAGE_FINGERPRINT_MISMATCH")

    candidates = []
    for temp_root in sorted((campaign / "envs").glob(f".{env}.*.tmp")):
        checkpoints = list(temp_root.glob("training-run/checkpoints/*/latest.pt"))
        sessions = list(temp_root.glob("sandbox-runs/*/staging/m3-training-eval-smoke/session.json"))
        configs = list(temp_root.glob("*-official-resume-train.yaml"))
        sidecars = list(temp_root.glob(f"training-run/wmloop_interventions/{manifest.get('proposal_primitive')}.json"))
        if (
            len(checkpoints) == 1
            and len(sessions) == 1
            and len(configs) == 1
            and len(sidecars) == 1
            and (temp_root / "budget.db").is_file()
        ):
            candidates.append((temp_root, checkpoints[0], sessions[0], configs[0], sidecars[0]))
    if len(candidates) != 1:
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_SOURCE_AMBIGUOUS:{len(candidates)}")
    temp_root, checkpoint, session_path, config_path, sidecar_path = candidates[0]
    if checkpoint.is_symlink() or checkpoint.stat().st_size < 1_000_000:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_CHECKPOINT_INVALID")

    session = _load_json(session_path)
    proposal_id = session_path.parents[2].name
    source_revision = str(session.get("source_revision") or "")
    if not proposal_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", proposal_id) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_SESSION_INVALID")
    attempts = tuple(_load_original_attempts(session_path.parent / "attempts"))
    passed = {str(item.get("label")) for item in attempts if item.get("passed") is True}
    if not REQUIRED_PASSED_LABELS.issubset(passed):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_TRAINING_EVIDENCE_INCOMPLETE")
    failures = [item for item in attempts if item.get("passed") is not True]
    if len(failures) != 1 or failures[0].get("label") != "baseline_action_following_probe":
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_FAILURE_STAGE_INVALID")
    failed_ordinal = failures[0].get("ordinal")
    if isinstance(failed_ordinal, bool) or not isinstance(failed_ordinal, int) or failed_ordinal < 1:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_FAILURE_ATTEMPT_INVALID")
    failed_stderr = session_path.parent / "attempts" / f"{failed_ordinal:03d}.stderr"
    if (
        not failed_stderr.is_file()
        or FAILURE_FINGERPRINT not in failed_stderr.read_text(encoding="utf-8", errors="replace")
    ):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_FAILURE_FINGERPRINT_MISMATCH")

    command_raw = record.get("command")
    if not isinstance(command_raw, list) or not all(isinstance(item, str) for item in command_raw):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_COMMAND_INVALID")
    command = tuple(command_raw)
    horizons_raw = record.get("horizons")
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_HORIZONS_INVALID")
    horizons = tuple(_positive_int(item, "FAILED_SCREEN_SALVAGE_HORIZONS_INVALID") for item in horizons_raw)
    primitive = str(manifest.get("proposal_primitive") or "")
    seed = _positive_int(manifest.get("seed"), "FAILED_SCREEN_SALVAGE_SEED_INVALID")
    return SalvageSource(
        campaign_root=campaign,
        environment=env,
        primitive=primitive,
        seed=seed,
        proposal_id=proposal_id,
        temp_root=temp_root,
        checkpoint_path=checkpoint.resolve(strict=True),
        training_config_path=config_path.resolve(strict=True),
        training_sidecar_path=sidecar_path.resolve(strict=True),
        budget_db_path=(temp_root / "budget.db").resolve(strict=True),
        session_path=session_path.resolve(strict=True),
        source_revision=source_revision,
        original_attempts=attempts,
        record_command=command,
        record_log_path=log_path,
        eval_horizons=horizons,
        max_accept_trajectories=_command_positive_int(command, "--max-accept-trajectories"),
        replication_count=_positive_int(manifest.get("replication_count"), "FAILED_SCREEN_SALVAGE_REPLICATION_INVALID"),
        eval_inference_steps=_positive_int(manifest.get("eval_inference_steps"), "FAILED_SCREEN_SALVAGE_INFERENCE_INVALID"),
        train_steps=_positive_int(manifest.get("train_steps"), "FAILED_SCREEN_SALVAGE_TRAIN_STEPS_INVALID"),
        train_batch_size=_positive_int(manifest.get("train_batch_size"), "FAILED_SCREEN_SALVAGE_BATCH_INVALID"),
        train_size=_positive_int(manifest.get("train_size"), "FAILED_SCREEN_SALVAGE_TRAIN_SIZE_INVALID"),
        train_num_workers=_nonnegative_int(manifest.get("train_num_workers"), "FAILED_SCREEN_SALVAGE_WORKERS_INVALID"),
        runtime_python=Path(_command_value(command, "--runtime-python")).resolve(strict=True),
        data_root=Path(_command_value(command, "--data-root")).resolve(strict=True),
        checkpoint_root=Path(_command_value(command, "--checkpoint-root")).resolve(strict=True),
        goal_config=Path(_command_value(command, "--goal-config")).resolve(strict=True),
        archive_db=_optional_command_path(command, "--archive-db"),
        cas_root=Path(_command_value(command, "--cas-root")).resolve(strict=True),
    )


def run_salvage(
    *,
    repo_root: Path,
    campaign_root: Path,
    gpu_index: int,
    gpu_audit_manifest: Path,
    environment: str | None = None,
    resume_staging: Path | None = None,
) -> dict[str, object]:
    root = Path(repo_root).resolve(strict=True)
    source = discover_salvage_source(campaign_root=campaign_root, environment=environment)
    destination = source.campaign_root / "envs" / source.environment
    if destination.exists() or destination.is_symlink():
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_OUTPUT_EXISTS")
    authorization = verify_gpu_exclusivity_ready(
        gpu_audit_manifest,
        gpu_index=gpu_index,
        max_age_seconds=3600.0,
    )
    current_revision = verify_vendor_checkout(root)
    if current_revision != source.source_revision:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_VENDOR_REVISION_MISMATCH")

    sidecar = _load_json(source.training_sidecar_path)
    if sidecar.get("primitive") != source.primitive or not isinstance(sidecar.get("params"), Mapping):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_SIDECAR_INVALID")
    params = dict(sidecar["params"])
    reused_staging = resume_staging is not None
    if reused_staging:
        staging = Path(resume_staging).resolve(strict=True)
        if staging.parent != destination.parent or not staging.name.startswith(f".{destination.name}.salvage-"):
            raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_RESUME_STAGING_INVALID")
    else:
        staging = destination.parent / f".{destination.name}.salvage-{uuid.uuid4().hex}.tmp"
        staging.mkdir(mode=0o700, parents=True)
    sandbox = (
        None
        if reused_staging
        else WorktreeSandbox(vendor_root=root / "vendor" / "ACWM-Phys", runs_root=staging / "sandbox-runs")
    )
    lease = None
    try:
        registry = PrimitiveRegistry.from_root(root)
        if reused_staging:
            candidate_runtime = _load_reused_candidate_runtime(
                staging=staging,
                source_revision=current_revision,
                primitive=source.primitive,
                expected_sidecar=sidecar,
            )
            rendered_records = list(candidate_runtime["rendered_primitives"])
        else:
            assert sandbox is not None
            lease = sandbox.create(trial_id=f"{source.proposal_id}-salvage", expected_revision=current_revision)
            renderer = PrimitiveRenderer(registry)
            rendered = renderer.render_checked(
                worktree=lease.worktree,
                interventions=[{"primitive": source.primitive, "params": params}],
                hook_ios=_default_hook_ios(),
            )
            for item in rendered:
                _apply_diff(lease.worktree, item.diff)
            rematerialized_sidecar = lease.worktree / "wmloop_interventions" / f"{source.primitive}.json"
            if _canonical_json(_load_json(rematerialized_sidecar)) != _canonical_json(sidecar):
                raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_MATERIALIZATION_DRIFT")
            rendered_records = [{"name": item.name, "diff_sha256": item.sha256} for item in rendered]
            candidate_runtime = _copy_candidate_runtime(
                source=lease.worktree,
                destination=staging / "candidate-runtime",
                source_revision=current_revision,
                rendered_primitives=rendered_records,
            )
            sandbox.remove(lease)
            lease = None

        replicates = _existing_replicates(source=source, staging=staging)
        official_checkpoint = source.checkpoint_root / _environment_spec(source.environment).checkpoint_relative_path
        if official_checkpoint.is_symlink() or not official_checkpoint.is_file():
            raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_OFFICIAL_CHECKPOINT_MISSING")
        env = _runtime_environment(
            runtime_python=source.runtime_python,
            worktree=root / "vendor" / "ACWM-Phys",
            repo_root=root,
            output_root=staging,
            data_root=source.data_root,
            checkpoint_root=source.checkpoint_root,
            gpu_index=gpu_index,
        )
        command_receipts: list[dict[str, object]] = []
        action_data_root = Path(str(replicates[0]["view_root"]))
        baseline_action_root = staging / "baseline-action-following-probe"
        candidate_action_root = staging / "candidate-action-following-probe"
        common = {
            "runtime": source.runtime_python,
            "repo_root": root,
            "data_root": action_data_root,
            "checkpoint_root": source.checkpoint_root,
            "inverse_summary_path": root / "results/m1/inverse-dynamics/summary-r1.json",
            "environment": source.environment,
            "horizons": source.eval_horizons,
            "primary_horizon": max(source.eval_horizons),
            "max_trajectories": source.max_accept_trajectories,
            "inference_steps": source.eval_inference_steps,
            "gpu_index": gpu_index,
            "gpu_exclusivity_audit_manifest": Path(gpu_audit_manifest).resolve(strict=True),
            "gpu_exclusivity_max_age_seconds": 86_400.0,
            "seed": 101,
            "action_only": True,
        }
        command_receipts.append(
            _run_command(
                label="salvage_baseline_action_following_probe",
                argv=_raw_probe_measure_command(
                    **common,
                    checkpoint_path=official_checkpoint,
                    output_root=baseline_action_root,
                ),
                cwd=root,
                env=env,
                log_root=staging / "logs",
                timeout_seconds=1800.0,
                reuse_existing=reused_staging,
            )
        )
        command_receipts.append(
            _run_command(
                label="salvage_candidate_action_following_probe",
                argv=_raw_probe_measure_command(
                    **common,
                    vendor_root=Path(candidate_runtime["runtime_path"]),
                    checkpoint_path=source.checkpoint_path,
                    output_root=candidate_action_root,
                ),
                cwd=root,
                env=env,
                log_root=staging / "logs",
                timeout_seconds=1800.0,
                reuse_existing=reused_staging,
            )
        )
        for replicate in replicates:
            label = str(replicate["label"])
            probe_common = {
                "runtime": source.runtime_python,
                "repo_root": root,
                "data_root": Path(str(replicate["view_root"])),
                "checkpoint_root": source.checkpoint_root,
                "environment": source.environment,
                "horizons": source.eval_horizons,
                "max_trajectories": source.max_accept_trajectories,
                "inference_steps": source.eval_inference_steps,
                "gpu_index": gpu_index,
                "gpu_exclusivity_audit_manifest": Path(gpu_audit_manifest).resolve(strict=True),
                "gpu_exclusivity_max_age_seconds": 86_400.0,
                "seed": int(replicate["seed"]),
            }
            command_receipts.append(
                _run_command(
                    label=f"salvage_baseline_horizon_probe_{label}",
                    argv=_horizon_probe_command(
                        **probe_common,
                        checkpoint_path=official_checkpoint,
                        output_root=Path(str(replicate["baseline_probe_root"])),
                    ),
                    cwd=root,
                    env=env,
                    log_root=staging / "logs",
                    timeout_seconds=1800.0,
                    reuse_existing=reused_staging,
                )
            )
            command_receipts.append(
                _run_command(
                    label=f"salvage_candidate_horizon_probe_{label}",
                    argv=_horizon_probe_command(
                        **probe_common,
                        vendor_root=Path(candidate_runtime["runtime_path"]),
                        checkpoint_path=source.checkpoint_path,
                        output_root=Path(str(replicate["candidate_probe_root"])),
                    ),
                    cwd=root,
                    env=env,
                    log_root=staging / "logs",
                    timeout_seconds=1800.0,
                    reuse_existing=reused_staging,
                )
            )

        runtime_binding = _validate_probe_runtime_binding(
            baseline_runtime_sha256=runtime_tree_sha256(root / "vendor" / "ACWM-Phys"),
            candidate_runtime_sha256=str(candidate_runtime["tree_sha256"]),
            baseline_manifest_paths=[
                baseline_action_root / "manifest.json",
                *(Path(str(item["baseline_probe_root"])) / "manifest.json" for item in replicates),
            ],
            candidate_manifest_paths=[
                candidate_action_root / "manifest.json",
                *(Path(str(item["candidate_probe_root"])) / "manifest.json" for item in replicates),
            ],
        )
        goal = load_yaml_document(source.goal_config)
        primary_metric = _goal_metric_name(goal=goal, horizons=source.eval_horizons)
        replication_records = _load_probe_replications(
            replicates=replicates,
            required_horizons=source.eval_horizons,
            metric_name=primary_metric,
        )
        archive = ArchiveStore(source.archive_db) if source.archive_db is not None else None
        cas = ContentAddressedStore(source.cas_root)
        action_following = _action_following_records_with_refs(
            action_following=_load_action_following_probe_pair(
                baseline_root=baseline_action_root,
                candidate_root=candidate_action_root,
            ),
            cas=cas,
            archive=archive,
        )
        evidence = _verification_evidence_from_probe_replications(
            proposal_id=source.proposal_id,
            replications=replication_records,
            diff_audit_passed=True,
            required_horizons=source.eval_horizons,
            metric_name=primary_metric,
            action_following=action_following,
            action_following_threshold=_goal_action_following_threshold(goal),
        )
        verdict = judge(evidence).to_dict()
        evaluation = _evaluation_summary(
            replications=replication_records,
            evidence=evidence,
            metric_name=primary_metric,
            action_following=action_following,
        )
        provenance_refs = {
            "failed_campaign_manifest": _put_file(cas, source.campaign_root / "manifest.json", archive=archive, media_type="application/json"),
            "failed_campaign_log": _put_file(cas, source.record_log_path, archive=archive, media_type="text/plain"),
            "training_config": _put_file(cas, source.training_config_path, archive=archive, media_type="application/json"),
            "training_sidecar": _put_file(cas, source.training_sidecar_path, archive=archive, media_type="application/json"),
            "staging_session": _put_file(cas, source.session_path, archive=archive, media_type="application/json"),
        }
        original_receipts = [dict(item) for item in source.original_attempts]
        all_receipts = [*original_receipts, *command_receipts]
        receipt = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-salvage-execution-receipt",
            "state": "ready",
            "proposal_id": source.proposal_id,
            "environment": source.environment,
            "primary_metric": primary_metric,
            "source_revision": current_revision,
            "registry_digest": registry.digest(),
            "gpu_index": gpu_index,
            "gpu_exclusivity_audit": authorization,
            "proposal_primitive": source.primitive,
            "proposal_params": params,
            "seed": source.seed,
            "training_scale": {
                "train_steps": source.train_steps,
                "batch_size": source.train_batch_size,
                "train_size": source.train_size,
                "num_workers": source.train_num_workers,
            },
            "rendered_primitives": rendered_records,
            "candidate_checkpoint_path": str(source.checkpoint_path),
            "candidate_runtime_path": str(candidate_runtime["runtime_path"]),
            "candidate_runtime_manifest_path": str(candidate_runtime["manifest_path"]),
            "candidate_runtime_sha256": str(candidate_runtime["tree_sha256"]),
            "runtime_binding": runtime_binding,
            "worktree_removed": True,
            "candidate_checkpoint_exists_before_cleanup": True,
            "candidate_checkpoint_retained": False,
            "candidate_runtime_retained": False,
            "candidate": {
                "candidate_id": "m3-training-eval-smoke-salvage",
                "ready_for_promotion": True,
                "receipts": all_receipts,
                "worktree_diff_sha256": _rendered_digest(rendered_records),
            },
            "replication_count": len(replication_records),
            "replications": _replication_records_with_refs(replications=replication_records, cas=cas, archive=archive),
            "evaluation": evaluation,
            "actual_gpu_hours": sum(float(item.get("duration_seconds", 0.0)) for item in all_receipts) / 3600.0,
            "salvage_provenance": {
                "failure_fingerprint": FAILURE_FINGERPRINT,
                "source_campaign_root": str(source.campaign_root),
                "source_temp_root": str(source.temp_root),
                "training_was_not_rerun": True,
                "original_failed_check": "baseline_action_following_probe",
                "provenance_refs": provenance_refs,
            },
        }
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-smoke-report",
            "state": "ready",
            "environment": source.environment,
            "goal_id": _goal_id(goal),
            "primary_metric": primary_metric,
            "proposal_primitive": source.primitive,
            "proposal_params": params,
            "proposal_id": source.proposal_id,
            "seed": source.seed,
            "verdict": verdict["verdict"],
            "violation": verdict["violation"],
            "gates": verdict["gates"],
            "action_following_gate": verdict["action_following_gate"],
            "delta_m_ver": verdict["delta_m_ver"],
            "settlement_state": "salvaged_after_infrastructure_failure",
            "actual_gpu_hours": receipt["actual_gpu_hours"],
            "runtime_python": str(source.runtime_python),
            "data_root": str(source.data_root),
            "checkpoint_root": str(source.checkpoint_root),
            "gpu_index": gpu_index,
            "gpu_exclusivity_audit": authorization,
            "train_steps": source.train_steps,
            "train_batch_size": source.train_batch_size,
            "train_size": source.train_size,
            "train_num_workers": source.train_num_workers,
            "eval_horizons": list(source.eval_horizons),
            "eval_inference_steps": source.eval_inference_steps,
            "max_accept_trajectories": source.max_accept_trajectories,
            "replication_count": source.replication_count,
            "receipt": receipt,
            "formal_settled_trial_published": False,
            "archive_db": str(source.archive_db) if source.archive_db is not None else None,
            "cas_root": str(source.cas_root),
            "salvage": receipt["salvage_provenance"],
            "limitations": [
                f"This is a recovered {source.train_steps}-step screen; training was completed before a fixed probe plumbing bug interrupted evaluation.",
                "Only the frozen post-training probes were rerun; the candidate checkpoint bytes were not modified.",
                "A positive screen remains exploratory until the independent official 50-step gate and staged confirmation pass.",
            ],
        }
        manifest = _write_report_bundle(
            report=report,
            output_root=destination,
            cas=cas,
            archive=archive,
            budget_db=source.budget_db_path,
        )
        shutil.rmtree(staging, ignore_errors=True)
        return manifest
    except Exception:
        raise
    finally:
        if lease is not None and sandbox is not None:
            try:
                sandbox.remove(lease)
            except Exception:
                pass


def build_salvage_queue(
    *,
    output_root: Path,
    repo_root: Path,
    campaign_roots: Sequence[Path],
    candidate_gpus: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_QUEUE_OUTPUT_EXISTS")
    gpus = [int(value) for value in candidate_gpus]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_QUEUE_GPUS_INVALID")
    rows: list[dict[str, object]] = []
    for rank, campaign_root in enumerate(campaign_roots, start=1):
        source = discover_salvage_source(campaign_root=campaign_root)
        output = source.campaign_root / "envs" / source.environment
        campaign_id = f"{source.campaign_root.name}-postprobe-salvage-r1"
        rows.append(
            {
                "rank": rank,
                "phase": "failed_screen_salvage",
                "campaign_id": campaign_id,
                "environment": source.environment,
                "primitive": source.primitive,
                "seed": source.seed,
                "train_steps": 0,
                "source_train_steps": source.train_steps,
                "resource_class": "gpu",
                "allow_any_idle_gpu": True,
                "candidate_gpus": gpus,
                "output_root": str(output),
                "gpu_audit_root_template": str(
                    Path(repo_root).resolve()
                    / "results/reports"
                    / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
                ),
                "requires_positive_manifest": "",
                "requires_official_quality_manifest": "",
                "launch_argv_template": [
                    str(Path(repo_root).resolve() / ".venv/bin/python3"),
                    str(Path(repo_root).resolve() / "scripts/export/acwm_failed_screen_salvage.py"),
                    "run",
                    "--repo-root",
                    str(Path(repo_root).resolve()),
                    "--campaign-root",
                    str(source.campaign_root),
                    "--environment",
                    source.environment,
                    "--gpu-index",
                    "{gpu}",
                    "--gpu-audit-manifest",
                    "{gpu_audit_manifest}",
                ],
            }
        )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "queue_role": "failed_screen_postprobe_salvage",
        "row_count": len(rows),
        "rows": rows,
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "autoloop-queue.json", report)
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "wmloop-acwm-autoloop-queue-manifest",
                "state": "ready",
                "row_count": len(rows),
                "queue_path": str(destination / "autoloop-queue.json"),
            },
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def _existing_replicates(*, source: SalvageSource, staging: Path) -> tuple[dict[str, object], ...]:
    roots = sorted((source.temp_root / "accept-cohorts").glob("r[0-9][0-9]"))
    if len(roots) != source.replication_count:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_ACCEPT_REPLICATION_MISMATCH")
    output = []
    for index, root in enumerate(roots, start=1):
        ids_path = root / "trajectory-ids.json"
        view_root = root / "view"
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if (
            not isinstance(ids, list)
            or len(ids) != source.max_accept_trajectories
            or not (view_root / "cohort-view-manifest.json").is_file()
        ):
            raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_ACCEPT_COHORT_INVALID")
        output.append(
            {
                "index": index,
                "label": root.name,
                "seed": 100 + index,
                "trajectory_ids": ids,
                "trajectory_count": len(ids),
                "view_root": view_root,
                "baseline_probe_root": staging / f"baseline-horizon-probe-{root.name}",
                "candidate_probe_root": staging / f"candidate-horizon-probe-{root.name}",
            }
        )
    return tuple(output)


def _load_reused_candidate_runtime(
    *,
    staging: Path,
    source_revision: str,
    primitive: str,
    expected_sidecar: Mapping[str, object],
) -> dict[str, object]:
    runtime_root = Path(staging) / "candidate-runtime"
    manifest_path = runtime_root / "wmloop-runtime-manifest.json"
    manifest = _load_json(manifest_path)
    rendered = manifest.get("rendered_primitives")
    if (
        manifest.get("artifact_type") != "wmloop-materialized-candidate-runtime"
        or manifest.get("state") != "ready"
        or manifest.get("source_revision") != source_revision
        or not isinstance(rendered, list)
        or not rendered
        or not all(isinstance(item, Mapping) for item in rendered)
    ):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_REUSED_RUNTIME_INVALID")
    expected_sha = manifest.get("tree_sha256")
    if not isinstance(expected_sha, str) or runtime_tree_sha256(runtime_root) != expected_sha:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_REUSED_RUNTIME_HASH_MISMATCH")
    runtime_sidecar = runtime_root / "wmloop_interventions" / f"{primitive}.json"
    if _canonical_json(_load_json(runtime_sidecar)) != _canonical_json(dict(expected_sidecar)):
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_REUSED_RUNTIME_SIDECAR_MISMATCH")
    return {
        "runtime_path": str(runtime_root),
        "manifest_path": str(manifest_path),
        "tree_sha256": expected_sha,
        "rendered_primitives": [dict(item) for item in rendered],
    }


def _run_command(
    *,
    label: str,
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_root: Path,
    timeout_seconds: float,
    reuse_existing: bool = False,
) -> dict[str, object]:
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = log_root / f"{label}.stdout"
    stderr_path = log_root / f"{label}.stderr"
    receipt_path = log_root / f"{label}.json"
    if reuse_existing:
        receipt = _load_json(receipt_path)
        if receipt.get("label") != label or receipt.get("passed") is not True:
            raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_REUSED_RECEIPT_INVALID:{label}")
        if (
            not stdout_path.is_file()
            or not stderr_path.is_file()
            or receipt.get("stdout_sha256") != _sha256(stdout_path)
            or receipt.get("stderr_sha256") != _sha256(stderr_path)
        ):
            raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_REUSED_LOG_HASH_MISMATCH:{label}")
        return receipt
    started = time.monotonic()
    timed_out = False
    exit_code = -1
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                list(argv),
                cwd=Path(cwd),
                env=dict(env),
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
            )
        exit_code = int(completed.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = time.monotonic() - started
    receipt = {
        "label": label,
        "argv": list(argv),
        "passed": exit_code == 0 and not timed_out,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    _write_json(receipt_path, receipt)
    if receipt["passed"] is not True:
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_PROBE_FAILED:{label}:{exit_code}")
    return receipt


def _load_original_attempts(attempt_root: Path) -> list[dict[str, object]]:
    records = [_load_json(path) for path in sorted(Path(attempt_root).glob("*.json"))]
    if not records:
        raise FailedScreenSalvageError("FAILED_SCREEN_SALVAGE_ATTEMPTS_MISSING")
    return records


def _command_value(command: Sequence[str], flag: str) -> str:
    matches = [index for index, value in enumerate(command) if value == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_COMMAND_OPTION_INVALID:{flag}")
    value = command[matches[0] + 1]
    if not value or value.startswith("--"):
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_COMMAND_OPTION_INVALID:{flag}")
    return value


def _command_positive_int(command: Sequence[str], flag: str) -> int:
    try:
        return _positive_int(int(_command_value(command, flag)), f"FAILED_SCREEN_SALVAGE_COMMAND_OPTION_INVALID:{flag}")
    except ValueError:
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_COMMAND_OPTION_INVALID:{flag}") from None


def _optional_command_path(command: Sequence[str], flag: str) -> Path | None:
    if flag not in command:
        return None
    return Path(_command_value(command, flag)).resolve()


def _positive_int(value: object, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FailedScreenSalvageError(error)
    return value


def _nonnegative_int(value: object, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FailedScreenSalvageError(error)
    return value


def _rendered_digest(records: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json([dict(item) for item in records])).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_JSON_INVALID:{path}") from error
    if not isinstance(payload, dict):
        raise FailedScreenSalvageError(f"FAILED_SCREEN_SALVAGE_JSON_INVALID:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=ROOT)
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--environment")
    run.add_argument("--gpu-index", type=int, required=True)
    run.add_argument("--gpu-audit-manifest", type=Path, required=True)
    run.add_argument("--resume-staging", type=Path)
    queue = subparsers.add_parser("queue")
    queue.add_argument("--repo-root", type=Path, default=ROOT)
    queue.add_argument("--output-root", type=Path, required=True)
    queue.add_argument("--campaign-root", type=Path, action="append", required=True)
    queue.add_argument("--candidate-gpus", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    if args.command == "queue":
        result = build_salvage_queue(
            output_root=args.output_root,
            repo_root=args.repo_root,
            campaign_roots=args.campaign_root,
            candidate_gpus=args.candidate_gpus,
        )
    else:
        result = run_salvage(
            repo_root=args.repo_root,
            campaign_root=args.campaign_root,
            environment=args.environment,
            gpu_index=args.gpu_index,
            gpu_audit_manifest=args.gpu_audit_manifest,
            resume_staging=args.resume_staging,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
