"""Bounded, receipt-first execution of one pre-registered GPU experiment.

This module is the generic transaction boundary missing from model-specific
campaign scripts: validate a rationale-bearing plan, reserve budget, lease and
audit one physical GPU, execute synchronously, verify a standard result,
archive every terminal outcome, and only then retire eligible scratch files.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding_admission import (
    OnboardingAdmissionError,
    verify_onboarding_admission,
)
from wmloop.execute.backends import LocalSubprocessBackend
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_exclusivity_audit import (
    run_gpu_exclusivity_audit,
    verify_gpu_exclusivity_ready,
)
from wmloop.execute.gpu_lease import GpuLeaseError, GpuLeaseManager
from wmloop.execute.gpu_sampling import GpuSamplingRecorder
from wmloop.verify import auto_experiment as auto_experiment_verifier
from wmloop.verify.auto_experiment import verify_auto_experiment_result


class AutoExperimentError(RuntimeError):
    """The auto-experiment transaction failed closed."""


_ENVIRONMENT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PLACEHOLDER = re.compile(
    r"\{(scratch_dir|workspace_root|output_root|gpu_index|gpu_uuid)\}"
)
_COST_CAPS = ((0.5, "very_low"), (8.0, "low"), (48.0, "medium"), (120.0, "high"))
_METRIC_OPERATORS = {"gte", "lte", "gt", "lt"}


def run_auto_experiment(
    *,
    plan_path: Path,
    output_root: Path,
    workspace_root: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path = Path("/tmp/verdiwm-gpu-leases"),
    budget_db: Path | None = None,
    budget_total_gpu_hours: float | None = None,
) -> dict[str, object]:
    """Run or resume one bounded plan and return its durable manifest."""

    plan_source = Path(plan_path).resolve(strict=True)
    workspace = Path(workspace_root).resolve(strict=True)
    if not workspace.is_dir() or workspace.is_symlink():
        raise AutoExperimentError("AUTO_EXPERIMENT_WORKSPACE_INVALID")
    plan = _load_plan(plan_source, workspace_root=workspace)
    destination = Path(output_root).resolve()
    plan_sha256 = _sha256_bytes(plan_source.read_bytes())
    _initialize_output(destination=destination, plan=plan, plan_sha256=plan_sha256)

    cas = ContentAddressedStore(Path(cas_root))
    archive = ArchiveStore(Path(archive_db))
    trial_signature = _trial_signature(plan=plan, workspace_root=workspace)
    archive_trial_id = f"auto-{trial_signature[:32]}"
    receipt_path = destination / "receipts" / f"{plan['trial_id']}.json"
    verdict_path = destination / "verdicts" / f"{plan['trial_id']}.json"
    manifest_path = destination / "manifest.json"
    budget_path = (
        Path(budget_db).resolve()
        if budget_db is not None
        else _campaign_budget_path(destination.parent, str(plan["campaign_id"]))
    )
    if budget_total_gpu_hours is not None:
        budget_total = _finite_float(
            budget_total_gpu_hours,
            "AUTO_EXPERIMENT_BUDGET_TOTAL_INVALID",
        )
        if budget_total <= 0:
            raise AutoExperimentError("AUTO_EXPERIMENT_BUDGET_TOTAL_INVALID")
    else:
        budget_total = float(plan["total_budget_gpu_hours"])
    budget = BudgetLedger(
        budget_path,
        BudgetPolicy(total_gpu_hours=budget_total),
    )

    if archive_trial_id in archive.visible_settled_trials():
        return _recover_archived_run(
            receipt_path=receipt_path,
            manifest_path=manifest_path,
            destination=destination,
            plan=plan,
            archive=archive,
        )
    existing = budget.get(archive_trial_id)
    if existing is not None and existing.state == "settled":
        if not receipt_path.is_file() or not verdict_path.is_file():
            raise AutoExperimentError("AUTO_EXPERIMENT_SETTLEMENT_RECOVERY_REQUIRED")
        receipt = _load_json_object(receipt_path, "AUTO_EXPERIMENT_RECEIPT_INVALID")
        _record_archive_from_receipt(archive=archive, receipt=receipt)
        manifest = _manifest_from_receipt(receipt=receipt, destination=destination)
        _write_json_atomic(manifest_path, manifest)
        return manifest

    human_approved = bool(plan.get("human_approved", False))
    cost_class = _cost_class(float(plan["estimated_gpu_hours"]))
    if existing is None:
        admission = budget.admit(
            archive_trial_id,
            cost_class=cost_class,
            estimated_gpu_hours=float(plan["estimated_gpu_hours"]),
            human_approved=human_approved,
        )
    else:
        running_marker = destination / "running" / f"{plan['trial_id']}.json"
        if _live_running_marker(running_marker):
            raise AutoExperimentError("AUTO_EXPERIMENT_ALREADY_RUNNING")
        admission = budget.takeover(
            archive_trial_id,
            expected_fencing_token=existing.fencing_token,
        )

    scratch = (
        destination
        / "scratch"
        / str(plan["trial_id"])
        / f"attempt-{admission.fencing_token:04d}"
    )
    if scratch.exists() or scratch.is_symlink():
        raise AutoExperimentError("AUTO_EXPERIMENT_SCRATCH_EXISTS")
    scratch.mkdir(mode=0o700, parents=True)
    scratch_marker = scratch / ".verdiwm-scratch.json"
    marker = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-scratch",
        "state": "running",
        "campaign_id": plan["campaign_id"],
        "trial_id": plan["trial_id"],
        "fencing_token": admission.fencing_token,
        "created_at": _utc_now(),
        "cleanup_eligible": plan["cleanup_policy"] == "archive_then_delete",
    }
    _write_json_atomic(scratch_marker, marker)
    running_marker = destination / "running" / f"{plan['trial_id']}.json"
    _write_json_atomic(
        running_marker,
        {
            "pid": os.getpid(),
            "trial_id": plan["trial_id"],
            "fencing_token": admission.fencing_token,
            "started_at": _utc_now(),
        },
    )

    try:
        try:
            execution = _execute_trial(
                plan=plan,
                destination=destination,
                scratch=scratch,
                workspace_root=workspace,
                archive_db=Path(archive_db),
                cas_root=Path(cas_root),
                lock_root=Path(lock_root),
                fencing_token=admission.fencing_token,
            )
        except GpuLeaseError as exc:
            if str(exc).startswith("GPU_LEASE_UNAVAILABLE"):
                # Capacity contention is a scheduling deferral, not scientific
                # evidence. No child process has started at this boundary.
                budget.release(
                    archive_trial_id,
                    fencing_token=admission.fencing_token,
                )
                shutil.rmtree(scratch)
                raise
            execution = _execution_failure(plan=plan, scratch=scratch, error=exc)
        except Exception as exc:
            # An admitted attempt must not remain invisible when preflight or
            # process launch fails before the normal execution receipt exists.
            execution = _execution_failure(plan=plan, scratch=scratch, error=exc)
        receipt = _settle_trial(
            plan=plan,
            plan_sha256=plan_sha256,
            trial_signature=trial_signature,
            archive_trial_id=archive_trial_id,
            admission_token=admission.fencing_token,
            execution=execution,
            scratch=scratch,
            workspace_root=workspace,
            cas=cas,
            archive=archive,
            budget_db=budget_path,
            budget_total_gpu_hours=budget_total,
        )
        _write_json_atomic(receipt_path, receipt)
        _write_json_atomic(verdict_path, receipt["verdict"])
        budget.settle(
            archive_trial_id,
            fencing_token=admission.fencing_token,
            actual_gpu_hours=float(receipt["cost"]["actual_gpu_hours"]),
            receipt_ref=str(receipt["receipt_ref"]),
        )
        _record_archive_from_receipt(archive=archive, receipt=receipt)
        marker.update(
            {
                "state": "settled",
                "settled_at": _utc_now(),
                "receipt_ref": receipt["receipt_ref"],
                "archive_recorded": True,
                "required_artifact_count": len(plan["artifacts"]),
                "archived_artifact_count": len(receipt["artifact_refs"]),
            }
        )
        _write_json_atomic(scratch_marker, marker)
        cleanup = _cleanup_settled_scratch(
            scratch=scratch,
            destination=destination,
            marker=marker,
            apply=plan["cleanup_policy"] == "archive_then_delete",
        )
        receipt["cleanup"] = cleanup
        _write_json_atomic(receipt_path, receipt)
        manifest = _manifest_from_receipt(receipt=receipt, destination=destination)
        _write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        if running_marker.exists() and not running_marker.is_symlink():
            running_marker.unlink()


def _execution_failure(
    *, plan: Mapping[str, Any], scratch: Path, error: Exception
) -> dict[str, object]:
    _write_json_atomic(
        scratch / "gpu-sampling.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-gpu-sampling-curve",
            "state": "empty",
            "gpu_index": None,
            "sample_count": 0,
            "samples": [],
        },
    )
    (scratch / "stdout.log").write_bytes(b"")
    (scratch / "stderr.log").write_text(
        f"{type(error).__name__}: {error}\n", encoding="utf-8"
    )
    return {
        "lease": {
            "index": None,
            "uuid": "",
            "name": "unavailable",
            "lock_path": "",
        },
        "authorization": {"state": "unavailable"},
        "command": list(plan["command"]),
        "environment_keys": sorted(plan["environment"]),
        "exit_code": None,
        "timed_out": False,
        "duration_seconds": 0.0,
        "gpu_sampling": _load_json_object(
            scratch / "gpu-sampling.json",
            "AUTO_EXPERIMENT_GPU_SAMPLING_INVALID",
        ),
        "execution_error": {
            "type": type(error).__name__,
            "message": str(error)[:500],
        },
    }


def _execute_trial(
    *,
    plan: Mapping[str, Any],
    destination: Path,
    scratch: Path,
    workspace_root: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path,
    fencing_token: int,
) -> dict[str, object]:
    lease_manager = GpuLeaseManager(lock_root=lock_root)
    with lease_manager.acquire(
        plan["allowed_gpu_indices"],
        wait_seconds=float(plan["gpu_wait_seconds"]),
    ) as lease:
        audit_root = (
            destination / "gpu-audits" / f"{plan['trial_id']}-f{fencing_token:04d}"
        )
        audit_manifest = run_gpu_exclusivity_audit(
            output_root=audit_root,
            requested_gpus=[lease.index],
            archive_db=archive_db,
            cas_root=cas_root,
        )
        authorization = verify_gpu_exclusivity_ready(
            Path(str(audit_manifest["report_path"])).parent / "manifest.json",
            gpu_index=lease.index,
            max_age_seconds=60,
        )
        substitutions = {
            "scratch_dir": str(scratch),
            "workspace_root": str(workspace_root),
            "output_root": str(destination),
            "gpu_index": str(lease.index),
            "gpu_uuid": lease.uuid,
        }
        command = tuple(
            _expand_token(str(token), substitutions) for token in plan["command"]
        )
        environment = dict(os.environ)
        environment.update(
            {
                str(key): _expand_token(str(value), substitutions)
                for key, value in plan["environment"].items()
            }
        )
        environment.update(lease.environment())
        environment.update(
            {
                "VERDIWM_TRIAL_ID": str(plan["trial_id"]),
                "VERDIWM_TRIAL_SCRATCH": str(scratch),
                "PYTHONPATH": _prepend_pythonpath(
                    workspace_root, environment.get("PYTHONPATH")
                ),
            }
        )
        workdir = _resolve_inside(
            workspace_root,
            str(plan["working_directory"]),
            "AUTO_EXPERIMENT_WORKDIR_INVALID",
        )
        sampler = GpuSamplingRecorder(
            gpu_index=lease.index,
            sample_interval_seconds=float(plan["sample_interval_seconds"]),
        )
        backend = LocalSubprocessBackend()
        result = sampler.capture(
            label="experiment",
            callback=lambda: backend.run(
                worktree=workdir,
                command=command,
                environment=environment,
                timeout_seconds=float(plan["timeout_seconds"]),
            ),
        )
        sampling = sampler.to_document()
        _write_json_atomic(scratch / "gpu-sampling.json", sampling)
        (scratch / "stdout.log").write_bytes(result.stdout)
        (scratch / "stderr.log").write_bytes(result.stderr)
        return {
            "lease": lease.to_document(),
            "authorization": authorization,
            "command": list(command),
            "environment_keys": sorted(
                set(plan["environment"]) | set(lease.environment())
            ),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "gpu_sampling": sampling,
        }


def _settle_trial(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    trial_signature: str,
    archive_trial_id: str,
    admission_token: int,
    execution: Mapping[str, Any],
    scratch: Path,
    workspace_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
    budget_db: Path,
    budget_total_gpu_hours: float,
) -> dict[str, object]:
    result_path = _resolve_inside(
        scratch, str(plan["result_path"]), "AUTO_EXPERIMENT_RESULT_PATH_INVALID"
    )
    result = _load_optional_json(result_path)
    metric_gates = tuple(
        item for item in plan["metric_gates"] if isinstance(item, Mapping)
    )
    verdict = verify_auto_experiment_result(
        result=result,
        metric_gates=metric_gates,
        expected_gpu_uuid=str(execution["lease"]["uuid"]),
        gpu_sampling=execution["gpu_sampling"],
        stage=str(plan["stage"]),
    )
    execution_blockers = []
    if execution["timed_out"] is True:
        execution_blockers.append({"code": "EXECUTION_TIMED_OUT"})
    if execution["exit_code"] != 0:
        execution_blockers.append(
            {"code": "EXECUTION_FAILED", "exit_code": execution["exit_code"]}
        )
    if execution.get("execution_error"):
        execution_blockers.append(
            {"code": "EXECUTION_ERROR", **dict(execution["execution_error"])}
        )

    artifact_refs: dict[str, str] = {}
    missing_artifacts = []
    for relative in plan["artifacts"]:
        artifact = _resolve_inside(
            scratch, str(relative), "AUTO_EXPERIMENT_ARTIFACT_PATH_INVALID"
        )
        if not artifact.is_file() or artifact.is_symlink():
            missing_artifacts.append(str(relative))
            continue
        ref = cas.put_bytes(artifact.read_bytes(), media_type=_media_type(artifact)).uri
        archive.record_artifact_reference(ref)
        artifact_refs[str(relative)] = ref
    if missing_artifacts:
        execution_blockers.append(
            {"code": "REQUIRED_ARTIFACTS_MISSING", "paths": missing_artifacts}
        )

    support_refs = {}
    for name, path, media_type in (
        ("stdout", scratch / "stdout.log", "text/plain"),
        ("stderr", scratch / "stderr.log", "text/plain"),
        ("gpu_sampling", scratch / "gpu-sampling.json", "application/json"),
    ):
        ref = cas.put_bytes(path.read_bytes(), media_type=media_type).uri
        archive.record_artifact_reference(ref)
        support_refs[name] = ref
    if execution_blockers:
        verdict = dict(verdict)
        verdict["verdict"] = "VOID"
        verdict["evidence_level"] = "void"
        verdict["blockers"] = [*verdict.get("blockers", []), *execution_blockers]
        verdict["blocker_count"] = len(verdict["blockers"])

    context = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-context",
        "campaign_id": plan["campaign_id"],
        "trial_id": plan["trial_id"],
        "objective": plan["objective"],
        "hypothesis": plan["hypothesis"],
        "selection_reason": plan["selection_reason"],
        "falsification_criterion": plan["falsification_criterion"],
        "stage": plan["stage"],
        "trial_signature": trial_signature,
    }
    context_ref = _put_json(cas=cas, archive=archive, payload=context)
    verdict_ref = _put_json(cas=cas, archive=archive, payload=verdict)
    source_state = _git_state(workspace_root)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-receipt",
        "state": "terminal",
        "settlement_state": "settled",
        "campaign_id": plan["campaign_id"],
        "trial_id": plan["trial_id"],
        "archive_trial_id": archive_trial_id,
        "proposal_id": archive_trial_id,
        "goal_id": plan["campaign_id"],
        "library_version": "auto-experiment-v1",
        "stage": plan["stage"],
        "plan_sha256": plan_sha256,
        "trial_signature": trial_signature,
        "fencing_token": admission_token,
        "source": source_state,
        "execution": dict(execution),
        "cost": {
            "estimated_gpu_hours": float(plan["estimated_gpu_hours"]),
            "actual_gpu_hours": float(execution["duration_seconds"]) / 3600.0,
            "gpu_count": 1,
        },
        "budget": {
            "ledger_path": str(budget_db),
            "campaign_total_gpu_hours": float(plan["total_budget_gpu_hours"]),
            "ledger_total_gpu_hours": budget_total_gpu_hours,
        },
        "artifact_refs": artifact_refs,
        "support_refs": support_refs,
        "failure_context_ref": context_ref,
        "verdict_ref": verdict_ref,
        "verdict": verdict,
        "hypothesis_hash": _sha256_bytes(str(plan["hypothesis"]).encode("utf-8")),
        "implementation_hash": str(source_state["implementation_hash"]),
        "evaluator_hash": _evaluator_hash(metric_gates),
        "created_at": _utc_now(),
    }
    receipt_ref = _put_json(cas=cas, archive=archive, payload=receipt)
    receipt["receipt_ref"] = receipt_ref
    receipt["receipt_hash"] = receipt_ref.rsplit("/", 1)[-1]
    return receipt


def _record_archive_from_receipt(
    *, archive: ArchiveStore, receipt: Mapping[str, Any]
) -> None:
    trial_id = str(receipt["archive_trial_id"])
    if trial_id in archive.visible_settled_trials():
        return
    archive.record_settled_trial(
        SettledTrialRecord(
            trial_id=trial_id,
            proposal_id=str(receipt["proposal_id"]),
            goal_id=str(receipt["goal_id"]),
            library_version=str(receipt["library_version"]),
            failure_context_ref=str(receipt["failure_context_ref"]),
            verdict_ref=str(receipt["verdict_ref"]),
            receipt_ref=str(receipt["receipt_ref"]),
            gpu_hours=float(receipt["cost"]["actual_gpu_hours"]),
            hypothesis_hash=str(receipt["hypothesis_hash"]),
            impl_diff_hash=str(receipt["implementation_hash"]),
            evaluator_hash=str(receipt["evaluator_hash"]),
            settlement_state="settled",
            receipt_hash=str(receipt["receipt_hash"]),
            exploratory=True,
        )
    )


def cleanup_auto_experiment_scratch(
    *,
    run_root: Path,
    older_than_hours: float,
    apply: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Plan or apply deletion of marker-proven, already-archived scratch."""

    root = Path(run_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or older_than_hours < 0:
        raise AutoExperimentError("AUTO_EXPERIMENT_CLEANUP_ROOT_INVALID")
    cutoff = (
        _dt.datetime.now(tz=_dt.timezone.utc).timestamp() - older_than_hours * 3600.0
    )
    candidates = []
    retained = []
    archive = ArchiveStore(Path(archive_db)) if archive_db is not None else None
    cas = ContentAddressedStore(Path(cas_root)) if cas_root is not None else None
    for marker_path in sorted(root.rglob(".verdiwm-scratch.json")):
        if marker_path.is_symlink() or not marker_path.is_file():
            continue
        scratch = marker_path.parent.resolve()
        if not _is_inside(root, scratch) or scratch == root or scratch.is_symlink():
            retained.append({"path": str(scratch), "reason": "PATH_UNSAFE"})
            continue
        marker = _load_optional_json(marker_path)
        eligible = (
            marker.get("state") == "settled"
            and marker.get("cleanup_eligible") is True
            and marker.get("archive_recorded") is True
            and isinstance(marker.get("receipt_ref"), str)
            and str(marker["receipt_ref"]).startswith("cas://sha256/")
            and marker.get("required_artifact_count")
            == marker.get("archived_artifact_count")
        )
        if not eligible:
            retained.append({"path": str(scratch), "reason": "NOT_PROVEN_SETTLED"})
            continue
        proof_error = _verify_cleanup_proof(marker=marker, archive=archive, cas=cas)
        if proof_error is not None:
            retained.append({"path": str(scratch), "reason": proof_error})
            continue
        if marker_path.stat().st_mtime > cutoff:
            retained.append({"path": str(scratch), "reason": "TOO_RECENT"})
            continue
        candidates.append(str(scratch))
        if apply:
            shutil.rmtree(scratch)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-cleanup-report",
        "state": "applied" if apply else "dry_run",
        "run_root": str(root),
        "older_than_hours": older_than_hours,
        "candidate_count": len(candidates),
        "deleted_count": len(candidates) if apply else 0,
        "candidates": candidates,
        "retained": retained,
        "generated_at": _utc_now(),
    }


def _recover_archived_run(
    *,
    receipt_path: Path,
    manifest_path: Path,
    destination: Path,
    plan: Mapping[str, object],
    archive: ArchiveStore,
) -> dict[str, object]:
    """Rebuild local projections after a crash following Archive publication."""

    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise AutoExperimentError("AUTO_EXPERIMENT_SETTLEMENT_RECOVERY_REQUIRED")
    receipt = _load_json_object(receipt_path, "AUTO_EXPERIMENT_RECEIPT_INVALID")
    if (
        receipt.get("settlement_state") != "settled"
        or receipt.get("state") != "terminal"
    ):
        raise AutoExperimentError("AUTO_EXPERIMENT_RECEIPT_NOT_SETTLED")
    if str(receipt.get("archive_trial_id")) not in archive.visible_settled_trials():
        raise AutoExperimentError("AUTO_EXPERIMENT_ARCHIVE_RECEIPT_MISMATCH")
    scratch_root = destination / "scratch" / str(plan["trial_id"])
    attempts = sorted(scratch_root.glob("attempt-*")) if scratch_root.is_dir() else ()
    for attempt in attempts:
        marker_path = attempt / ".verdiwm-scratch.json"
        if not marker_path.is_file() or marker_path.is_symlink():
            continue
        marker = _load_optional_json(marker_path)
        if (
            marker.get("state") != "settled"
            or marker.get("archive_recorded") is not True
        ):
            marker.update(
                {
                    "state": "settled",
                    "settled_at": _utc_now(),
                    "receipt_ref": receipt["receipt_ref"],
                    "archive_recorded": True,
                    "required_artifact_count": len(plan["artifacts"]),
                    "archived_artifact_count": len(receipt.get("artifact_refs", {})),
                }
            )
            _write_json_atomic(marker_path, marker)
        if marker.get("state") == "settled" and marker.get("archive_recorded") is True:
            cleanup = _cleanup_settled_scratch(
                scratch=attempt,
                destination=destination,
                marker=marker,
                apply=plan["cleanup_policy"] == "archive_then_delete",
            )
            receipt["cleanup"] = cleanup
            _write_json_atomic(receipt_path, receipt)
            break
    manifest = _manifest_from_receipt(receipt=receipt, destination=destination)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _verify_cleanup_proof(
    *,
    marker: Mapping[str, object],
    archive: ArchiveStore | None,
    cas: ContentAddressedStore | None,
) -> str | None:
    """Independently verify durable evidence before periodic deletion."""

    if archive is None or cas is None:
        return "DURABLE_STORES_NOT_CONFIGURED"
    receipt_ref = marker.get("receipt_ref")
    if not isinstance(receipt_ref, str):
        return "RECEIPT_REF_INVALID"
    try:
        receipt_core = _load_json_bytes(
            cas.read_bytes(receipt_ref), "RECEIPT_CAS_INVALID"
        )
    except Exception:
        return "RECEIPT_CAS_UNREADABLE"
    if receipt_core.get("artifact_type") != "verdiwm-auto-experiment-receipt":
        return "RECEIPT_CAS_INVALID"
    archive_trial_id = receipt_core.get("archive_trial_id")
    if (
        not isinstance(archive_trial_id, str)
        or archive_trial_id not in archive.visible_settled_trials()
    ):
        return "ARCHIVE_RECORD_MISSING"
    refs: list[object] = []
    for block_name in ("artifact_refs", "support_refs"):
        block = receipt_core.get(block_name)
        if isinstance(block, Mapping):
            refs.extend(block.values())
    refs.extend(
        value
        for value in (
            receipt_core.get("failure_context_ref"),
            receipt_core.get("verdict_ref"),
        )
        if isinstance(value, str)
    )
    for uri in refs:
        try:
            cas.read_bytes(str(uri))
        except Exception:
            return "DECLARED_ARTIFACT_CAS_MISSING"
    return None


def _load_plan(path: Path, *, workspace_root: Path) -> dict[str, object]:
    try:
        plan = _load_json_object(path, "AUTO_EXPERIMENT_PLAN_INVALID")
        validate_document("auto_experiment_plan", plan)
    except (ContractValidationError, AutoExperimentError) as exc:
        raise AutoExperimentError(f"AUTO_EXPERIMENT_PLAN_INVALID:{exc}") from exc
    _validate_plan_semantics(plan, workspace_root=workspace_root)
    return plan


def _validate_plan_semantics(
    plan: Mapping[str, object], *, workspace_root: Path
) -> None:
    control_root = Path(__file__).resolve().parents[2]
    if Path(workspace_root).resolve() != control_root:
        try:
            verify_onboarding_admission(
                plan.get("onboarding_admission"),
                expected_repo_root=workspace_root,
            )
        except OnboardingAdmissionError as exc:
            raise AutoExperimentError(
                f"AUTO_EXPERIMENT_ONBOARDING_ADMISSION_INVALID:{exc}"
            ) from exc
    for field in (
        "objective",
        "hypothesis",
        "selection_reason",
        "falsification_criterion",
    ):
        value = plan.get(field)
        if not isinstance(value, str) or len(value.strip()) < 12:
            raise AutoExperimentError(f"AUTO_EXPERIMENT_RATIONALE_TOO_SHORT:{field}")
    estimated = _finite_float(
        plan.get("estimated_gpu_hours"), "AUTO_EXPERIMENT_ESTIMATE_INVALID"
    )
    total = _finite_float(
        plan.get("total_budget_gpu_hours"), "AUTO_EXPERIMENT_TOTAL_BUDGET_INVALID"
    )
    if estimated > total:
        raise AutoExperimentError("AUTO_EXPERIMENT_ESTIMATE_EXCEEDS_TOTAL_BUDGET")
    if bool(plan.get("human_approved", False)) and _cost_class(estimated) != "high":
        raise AutoExperimentError("AUTO_EXPERIMENT_HUMAN_APPROVAL_ONLY_FOR_HIGH_COST")
    for field, minimum in (
        ("timeout_seconds", 0.0),
        ("gpu_wait_seconds", 0.0),
        ("sample_interval_seconds", 0.0),
    ):
        value = _finite_float(
            plan.get(field), f"AUTO_EXPERIMENT_{field.upper()}_INVALID"
        )
        if value <= minimum if field != "gpu_wait_seconds" else value < minimum:
            raise AutoExperimentError(f"AUTO_EXPERIMENT_{field.upper()}_INVALID")
    workdir = _resolve_inside(
        workspace_root,
        str(plan["working_directory"]),
        "AUTO_EXPERIMENT_WORKDIR_INVALID",
    )
    if not workdir.is_dir():
        raise AutoExperimentError("AUTO_EXPERIMENT_WORKDIR_MISSING")
    for field in ("result_path", *[str(item) for item in plan["artifacts"]]):
        _relative_path(field, "AUTO_EXPERIMENT_RELATIVE_PATH_REQUIRED")
    if str(plan["result_path"]) not in {str(item) for item in plan["artifacts"]}:
        raise AutoExperimentError("AUTO_EXPERIMENT_RESULT_MUST_BE_DECLARED_ARTIFACT")
    for key, value in plan["environment"].items():
        if not isinstance(key, str) or _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise AutoExperimentError("AUTO_EXPERIMENT_ENVIRONMENT_KEY_INVALID")
        if not isinstance(value, str) or "\x00" in value:
            raise AutoExperimentError("AUTO_EXPERIMENT_ENVIRONMENT_VALUE_INVALID")
    for gate in plan["metric_gates"]:
        if not isinstance(gate, Mapping):
            raise AutoExperimentError("AUTO_EXPERIMENT_METRIC_GATE_INVALID")
        if set(gate) - {"metric", "role", "operator", "threshold"}:
            raise AutoExperimentError("AUTO_EXPERIMENT_METRIC_GATE_EXTRA_FIELD")
        if (
            not isinstance(gate.get("metric"), str)
            or not str(gate["metric"])
            or gate.get("operator") not in _METRIC_OPERATORS
            or not isinstance(gate.get("threshold"), (int, float))
            or isinstance(gate.get("threshold"), bool)
            or not math.isfinite(float(gate["threshold"]))
        ):
            raise AutoExperimentError("AUTO_EXPERIMENT_METRIC_GATE_INVALID")
    for token in plan["command"]:
        if "\x00" in token:
            raise AutoExperimentError("AUTO_EXPERIMENT_COMMAND_INVALID")
    if len(set(plan["allowed_gpu_indices"])) != len(plan["allowed_gpu_indices"]):
        raise AutoExperimentError("AUTO_EXPERIMENT_GPU_ALLOWLIST_DUPLICATE")


def _initialize_output(
    *, destination: Path, plan: Mapping[str, object], plan_sha256: str
) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise AutoExperimentError("AUTO_EXPERIMENT_OUTPUT_ROOT_INVALID")
        lock_path = destination / "plan.lock.json"
        if not lock_path.is_file():
            raise AutoExperimentError("AUTO_EXPERIMENT_OUTPUT_ROOT_UNBOUND")
        lock = _load_json_object(lock_path, "AUTO_EXPERIMENT_PLAN_LOCK_INVALID")
        if lock.get("plan_sha256") != plan_sha256 or lock.get("trial_id") != plan.get(
            "trial_id"
        ):
            raise AutoExperimentError("AUTO_EXPERIMENT_OUTPUT_PLAN_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True)
    for child in ("running", "receipts", "verdicts", "scratch", "gpu-audits"):
        (destination / child).mkdir(mode=0o700)
    _write_json_atomic(
        destination / "plan.lock.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-auto-experiment-plan-lock",
            "campaign_id": plan["campaign_id"],
            "trial_id": plan["trial_id"],
            "plan_sha256": plan_sha256,
            "created_at": _utc_now(),
        },
    )


def _trial_signature(*, plan: Mapping[str, object], workspace_root: Path) -> str:
    source = _git_state(workspace_root)
    payload = {
        "campaign_id": plan["campaign_id"],
        "trial_id": plan["trial_id"],
        "objective": plan["objective"],
        "hypothesis": plan["hypothesis"],
        "stage": plan["stage"],
        "command": plan["command"],
        "metric_gates": plan["metric_gates"],
        "source_revision": source["revision"],
        "implementation_hash": source["implementation_hash"],
    }
    return _sha256_bytes(_canonical_json(payload))


def _campaign_budget_path(output_parent: Path, campaign_id: str) -> Path:
    digest = _sha256_bytes(campaign_id.encode("utf-8"))[:24]
    return Path(output_parent).resolve() / "budgets" / f"campaign-{digest}.db"


def _cleanup_settled_scratch(
    *,
    scratch: Path,
    destination: Path,
    marker: Mapping[str, object],
    apply: bool,
) -> dict[str, object]:
    expected_parent = (destination / "scratch").resolve()
    resolved = scratch.resolve()
    if scratch.is_symlink() or not _is_inside(expected_parent, resolved):
        raise AutoExperimentError("AUTO_EXPERIMENT_CLEANUP_PATH_INVALID")
    result = {
        "state": "applied" if apply else "retained",
        "path": str(scratch),
        "reason": (
            "ARCHIVED_TERMINAL"
            if marker.get("archive_recorded") is True
            else "ARCHIVE_REQUIRED"
        ),
    }
    if apply:
        shutil.rmtree(scratch)
    return result


def _manifest_from_receipt(
    *, receipt: Mapping[str, object], destination: Path
) -> dict[str, object]:
    verdict = receipt.get("verdict")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-manifest",
        "state": "ready",
        "settlement_state": receipt.get("settlement_state"),
        "campaign_id": receipt.get("campaign_id"),
        "trial_id": receipt.get("trial_id"),
        "stage": receipt.get("stage"),
        "verdict": verdict.get("verdict") if isinstance(verdict, Mapping) else None,
        "evidence_level": (
            verdict.get("evidence_level") if isinstance(verdict, Mapping) else "void"
        ),
        "receipt_ref": receipt.get("receipt_ref"),
        "receipt_path": str(destination / "receipts" / f"{receipt['trial_id']}.json"),
        "verdict_path": str(destination / "verdicts" / f"{receipt['trial_id']}.json"),
        "cost": receipt.get("cost", {}),
        "budget": receipt.get("budget", {}),
        "artifact_refs": receipt.get("artifact_refs", {}),
        "cleanup": receipt.get("cleanup", {}),
        "archive_trial_id": receipt.get("archive_trial_id"),
    }


def _live_running_marker(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    payload = _load_optional_json(path)
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _resolve_inside(root: Path, value: str, error_code: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (Path(root).resolve() / candidate).resolve()
    if not _is_inside(Path(root).resolve(), resolved):
        raise AutoExperimentError(error_code)
    return resolved


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _relative_path(value: str, error_code: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value or value.endswith("/"):
        raise AutoExperimentError(error_code)


def _expand_token(value: str, substitutions: Mapping[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda match: substitutions[match.group(1)], value)


def _prepend_pythonpath(workspace_root: Path, existing: str | None) -> str:
    values = [str(workspace_root)]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def _load_json_object(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoExperimentError(code) from exc
    if not isinstance(value, dict):
        raise AutoExperimentError(code)
    return value


def _load_json_bytes(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AutoExperimentError(code) from exc
    if not isinstance(value, dict):
        raise AutoExperimentError(code)
    return value


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        if not path.is_file() or path.is_symlink():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _put_json(
    *, cas: ContentAddressedStore, archive: ArchiveStore, payload: Mapping[str, object]
) -> str:
    reference = cas.put_bytes(
        _canonical_json(payload), media_type="application/json"
    ).uri
    archive.record_artifact_reference(reference)
    return reference


def _media_type(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "application/json"
    if path.suffix.lower() in {".log", ".txt", ".md", ".csv"}:
        return "text/plain"
    return "application/octet-stream"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_state(workspace_root: Path) -> dict[str, object]:
    revision_result = subprocess.run(
        ("git", "-C", str(workspace_root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    revision = (
        revision_result.stdout.strip() if revision_result.returncode == 0 else "unknown"
    )
    diff_result = subprocess.run(
        ("git", "-C", str(workspace_root), "diff", "--no-ext-diff", "--binary", "HEAD"),
        check=False,
        capture_output=True,
    )
    diff = diff_result.stdout if diff_result.returncode == 0 else b""
    status_result = subprocess.run(
        ("git", "-C", str(workspace_root), "status", "--porcelain"),
        check=False,
        capture_output=True,
        text=True,
    )
    untracked_sources, untracked_payload = _untracked_source_state(workspace_root)
    return {
        "revision": revision,
        "dirty": (
            bool(status_result.stdout.strip())
            if status_result.returncode == 0
            else True
        ),
        "untracked_source_files": untracked_sources,
        "implementation_hash": _sha256_bytes(
            (revision + "\n").encode("utf-8") + diff + untracked_payload
        ),
    }


def _untracked_source_state(workspace_root: Path) -> tuple[list[dict[str, str]], bytes]:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(workspace_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "wmloop",
            "configs",
            "scripts",
        ),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return [], b"UNTRACKED_SOURCE_QUERY_FAILED\n"
    records: list[dict[str, str]] = []
    payload = bytearray()
    for raw_path in sorted(item for item in result.stdout.split(b"\0") if item):
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            payload.extend(b"UNTRACKED_SOURCE_PATH_INVALID\0" + raw_path + b"\0")
            continue
        source = (workspace_root / relative).resolve()
        if (
            not _is_inside(workspace_root, source)
            or not source.is_file()
            or source.is_symlink()
        ):
            payload.extend(b"UNTRACKED_SOURCE_MEMBER_INVALID\0" + raw_path + b"\0")
            continue
        content = source.read_bytes()
        digest = _sha256_bytes(content)
        records.append({"path": relative, "sha256": digest})
        payload.extend(
            raw_path + b"\0" + digest.encode("ascii") + b"\0" + content + b"\0"
        )
    return records, bytes(payload)


def _evaluator_hash(metric_gates: Sequence[Mapping[str, object]]) -> str:
    source = Path(auto_experiment_verifier.__file__).read_bytes()
    return _sha256_bytes(source + _canonical_json(list(metric_gates)))


def _finite_float(value: object, error_code: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AutoExperimentError(error_code)
    return float(value)


def _cost_class(estimated_gpu_hours: float) -> str:
    for cap, name in _COST_CAPS:
        if estimated_gpu_hours <= cap:
            return name
    raise AutoExperimentError("AUTO_EXPERIMENT_COST_CLASS_UNSUPPORTED")


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or resume one bounded plan")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--archive-db", type=Path)
    run_parser.add_argument("--cas-root", type=Path)
    run_parser.add_argument(
        "--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases")
    )
    run_parser.add_argument("--budget-db", type=Path)
    run_parser.add_argument("--budget-total-gpu-hours", type=float)
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="dry-run or apply proven scratch cleanup"
    )
    cleanup_parser.add_argument("--run-root", type=Path, required=True)
    cleanup_parser.add_argument("--older-than-hours", type=float, default=168.0)
    cleanup_parser.add_argument("--archive-db", type=Path)
    cleanup_parser.add_argument("--cas-root", type=Path)
    cleanup_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            output = Path(args.output_root).resolve()
            archive_db = (
                Path(args.archive_db)
                if args.archive_db
                else output.parent / "archive.db"
            )
            cas_root = Path(args.cas_root) if args.cas_root else output.parent
            manifest = run_auto_experiment(
                plan_path=args.plan,
                output_root=output,
                workspace_root=args.workspace_root,
                archive_db=archive_db,
                cas_root=cas_root,
                lock_root=args.lock_root,
                budget_db=args.budget_db,
                budget_total_gpu_hours=args.budget_total_gpu_hours,
            )
        else:
            manifest = cleanup_auto_experiment_scratch(
                run_root=args.run_root,
                older_than_hours=args.older_than_hours,
                apply=args.apply,
                archive_db=args.archive_db
                or Path(args.run_root).resolve().parent / "archive.db",
                cas_root=args.cas_root or Path(args.run_root).resolve().parent,
            )
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return 0
    except (AutoExperimentError, GpuLeaseError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
