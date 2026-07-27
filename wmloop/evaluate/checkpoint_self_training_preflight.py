"""Fail-closed preflight for the checkpoint self-training recovery branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.execute.gpu_exclusivity_audit import GpuExclusivityAuditError, verify_gpu_exclusivity_ready


class CheckpointSelfTrainingPreflightError(RuntimeError):
    """Checkpoint self-training preflight failed before a durable packet existed."""


def run_checkpoint_self_training_preflight(
    *,
    checkpoint_recovery_packet_manifest: Path,
    m4_unblock_plan_manifest: Path,
    goal_config: Path,
    output_root: Path,
    environment: str = "cloth_move",
    expected_step: int = 100000,
    estimated_gpu_hours: float = 120.0,
    max_allowed_gpu_hours: float = 120.0,
    requested_gpus: Sequence[int] = (),
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_audit_max_age_seconds: float = 300.0,
    confirm_human_approved_self_training: bool = False,
    quarantine_root: Path = Path("results/quarantine/checkpoints"),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only packet for a possible checkpoint continuation-training branch."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_PREFLIGHT_OUTPUT_EXISTS")
    if not environment:
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_ENVIRONMENT_INVALID")
    if expected_step < 1:
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_EXPECTED_STEP_INVALID")
    if estimated_gpu_hours <= 0 or max_allowed_gpu_hours <= 0:
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_BUDGET_INVALID")
    if gpu_audit_max_age_seconds < 0:
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_GPU_AUDIT_MAX_AGE_INVALID")

    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    sources = {
        "checkpoint_recovery_packet": _load_source_with_report(
            checkpoint_recovery_packet_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-recovery-packet-manifest",
            report_artifact_type="acwm-m0-checkpoint-recovery-packet",
            cas=cas,
            archive=archive,
        ),
        "m4_unblock_plan": _load_source_with_report(
            m4_unblock_plan_manifest,
            manifest_artifact_type="wmloop-m4-unblock-dependency-plan-manifest",
            report_artifact_type="wmloop-m4-unblock-dependency-plan",
            cas=cas,
            archive=archive,
        ),
    }
    goal_source = Path(goal_config).resolve(strict=True)
    goal_bytes = goal_source.read_bytes()
    goal = _load_goal(goal_source)
    goal_ref = cas.put_bytes(goal_bytes, media_type="application/yaml").uri
    if archive is not None:
        archive.record_artifact_reference(goal_ref)
    sources["goal_config"] = {
        "payload": goal,
        "summary": {
            "path": str(goal_source),
            "sha256": hashlib.sha256(goal_bytes).hexdigest(),
            "cas_ref": goal_ref,
            "artifact_type": "goal_spec",
            "state": "ready",
        },
    }

    requested = _normalize_gpu_indices(requested_gpus)
    recovery = _report_or_payload(sources, "checkpoint_recovery_packet")
    unblock_plan = _report_or_payload(sources, "m4_unblock_plan")
    current_checkpoint, source_blockers = _current_checkpoint(
        recovery=recovery,
        environment=environment,
        expected_step=expected_step,
    )
    self_training_branch, branch_blockers = _self_training_branch(
        unblock_plan=unblock_plan,
        environment=environment,
    )
    budget = _budget_summary(
        goal=goal,
        estimated_gpu_hours=estimated_gpu_hours,
        max_allowed_gpu_hours=max_allowed_gpu_hours,
    )
    gpu_preflight = _gpu_preflight(
        manifest_path=gpu_exclusivity_audit_manifest,
        requested_gpus=requested,
        max_age_seconds=gpu_audit_max_age_seconds,
    )

    blockers: list[dict[str, object]] = []
    blockers.extend(source_blockers)
    blockers.extend(branch_blockers)
    blockers.extend(budget["blockers"])
    if not confirm_human_approved_self_training:
        blockers.append(
            {
                "surface": "human_approval",
                "reason": "SELF_TRAINING_BRANCH_REQUIRES_EXPLICIT_HUMAN_APPROVAL",
            }
        )
    blockers.extend(gpu_preflight["blockers"])

    state = _state(
        blockers=blockers,
        human_approved=confirm_human_approved_self_training,
        gpu_ready=gpu_preflight["ready"] is True,
        budget_ready=budget["ready"] is True,
    )
    planning_ready = (
        state == "staged_for_manual_launch_planning"
        and confirm_human_approved_self_training
        and gpu_preflight["ready"] is True
        and budget["ready"] is True
    )
    side_effects = {
        "gpu_execution_started": False,
        "training_process_started": False,
        "training_budget_debited": False,
        "active_checkpoint_mutated": False,
        "quarantine_checkpoint_mutated": False,
        "active_goal_config_mutated": False,
        "m4_launch_receipt_written": False,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-self-training-preflight",
        "state": state,
        "scope": "M0 checkpoint source recovery branch preflight",
        "environment": environment,
        "expected_step": expected_step,
        "human_approval_provided": confirm_human_approved_self_training,
        "self_training_launch_planning_ready": planning_ready,
        "packet_grants_training_launch_permission": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "current_checkpoint": current_checkpoint,
        "self_training_branch": self_training_branch,
        "budget": budget["summary"],
        "gpu_preflight": gpu_preflight["summary"],
        "quarantine_contract": {
            "candidate_path_template": str(Path(quarantine_root) / environment / "latest.pt"),
            "active_checkpoint_must_not_be_overwritten_by_training": True,
            "validate_state_required": "ready_for_manual_install",
            "install_requires_explicit_confirmation": True,
        },
        "required_post_training_evidence": [
            "checkpoint_quarantine validate reports ready_for_manual_install",
            "explicit human confirmation before active checkpoint install",
            "checkpoint step audit reaches expected_step",
            "checkpoint source/candidate inventory audits regenerate cleanly",
            "strict launch guard smoke clears",
            "M0 baseline reproduction regenerates under the accepted checkpoint",
            "M1/M3 evidence and strict M4 phase gate are regenerated before formal M4 training",
        ],
        "sources": {name: source["summary"] for name, source in sources.items()},
        "side_effects": side_effects,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "next_actions": _next_actions(state=state, environment=environment, requested_gpus=requested),
        "limitations": [
            "This packet is read-only and launches no training process.",
            "This packet never grants formal M4 launch permission.",
            "A ready packet only stages manual launch planning for checkpoint source remediation; the output checkpoint must still enter quarantine and pass validation.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _load_source_with_report(
    path: Path,
    *,
    manifest_artifact_type: str,
    report_artifact_type: str,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    manifest, manifest_bytes, manifest_path = _read_json(path, "CHECKPOINT_SELF_TRAINING_SOURCE_INVALID")
    if manifest.get("artifact_type") != manifest_artifact_type:
        raise CheckpointSelfTrainingPreflightError(f"CHECKPOINT_SELF_TRAINING_SOURCE_INVALID:{manifest_path}")
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise CheckpointSelfTrainingPreflightError(f"CHECKPOINT_SELF_TRAINING_REPORT_MISSING:{manifest_path}")
    report, report_bytes, resolved_report = _read_json(Path(report_path), "CHECKPOINT_SELF_TRAINING_REPORT_INVALID")
    if report.get("artifact_type") != report_artifact_type:
        raise CheckpointSelfTrainingPreflightError(f"CHECKPOINT_SELF_TRAINING_REPORT_INVALID:{resolved_report}")
    manifest_ref = cas.put_bytes(manifest_bytes, media_type="application/json").uri
    report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(manifest_ref)
        archive.record_artifact_reference(report_ref)
    return {
        "payload": manifest,
        "report": report,
        "summary": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "cas_ref": manifest_ref,
            "artifact_type": manifest.get("artifact_type"),
            "state": manifest.get("state"),
            "report_path": str(resolved_report),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "report_cas_ref": report_ref,
        },
    }


def _read_json(path: Path, code: str) -> tuple[Mapping[str, Any], bytes, Path]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointSelfTrainingPreflightError(f"{code}:{Path(path)}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointSelfTrainingPreflightError(f"{code}:{resolved}")
    return payload, payload_bytes, resolved


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        goal = load_yaml_document(path)
        validate_document("goal_spec", goal)
    except (OSError, ContractValidationError) as exc:
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_GOAL_INVALID") from exc
    return goal


def _report_or_payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    source = sources.get(name, {})
    report = source.get("report")
    if isinstance(report, Mapping):
        return report
    payload = source.get("payload")
    if isinstance(payload, Mapping):
        return payload
    raise CheckpointSelfTrainingPreflightError(f"CHECKPOINT_SELF_TRAINING_SOURCE_MISSING:{name}")


def _normalize_gpu_indices(indices: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in indices:
        value = int(index)
        if value < 0:
            raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_GPU_INDEX_INVALID")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _current_checkpoint(
    *,
    recovery: Mapping[str, Any],
    environment: str,
    expected_step: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    mismatches = recovery.get("mismatches")
    selected: Mapping[str, Any] | None = None
    if isinstance(mismatches, list):
        for item in mismatches:
            if isinstance(item, Mapping) and item.get("environment") == environment:
                selected = item
                break
    if selected is None:
        blockers.append({"surface": "checkpoint_recovery", "reason": "CHECKPOINT_MISMATCH_NOT_ACTIVE", "environment": environment})
        return {"environment": environment, "expected_step": expected_step, "observed_step": None}, blockers
    observed_step = selected.get("observed_step")
    if selected.get("expected_step") != expected_step:
        blockers.append(
            {
                "surface": "checkpoint_recovery",
                "reason": "EXPECTED_STEP_MISMATCH",
                "observed_expected_step": selected.get("expected_step"),
                "required_expected_step": expected_step,
            }
        )
    if not isinstance(observed_step, int) or observed_step >= expected_step:
        blockers.append(
            {
                "surface": "checkpoint_recovery",
                "reason": "CHECKPOINT_ALREADY_AT_OR_ABOVE_EXPECTED_STEP",
                "observed_step": observed_step,
                "expected_step": expected_step,
            }
        )
    return {
        "environment": environment,
        "checkpoint_relative_path": selected.get("checkpoint_relative_path"),
        "checkpoint_path": selected.get("checkpoint_path"),
        "expected_step": selected.get("expected_step"),
        "observed_step": observed_step,
        "remaining_steps": expected_step - observed_step if isinstance(observed_step, int) else None,
        "status": selected.get("status"),
        "recovery_state": recovery.get("state"),
        "active_checkpoint_mutated": recovery.get("active_checkpoint_mutated") is True,
    }, blockers


def _self_training_branch(
    *,
    unblock_plan: Mapping[str, Any],
    environment: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    checkpoint_step = None
    dependencies = unblock_plan.get("dependencies")
    if isinstance(dependencies, list):
        for item in dependencies:
            if isinstance(item, Mapping) and item.get("step_id") == "resolve_checkpoint_source":
                checkpoint_step = item
                break
    if not isinstance(checkpoint_step, Mapping):
        blockers.append({"surface": "m4_unblock_plan", "reason": "CHECKPOINT_RESOLUTION_STEP_MISSING"})
        return {"branch_id": "continue_or_retrain_cloth_move_to_expected_step", "status": "missing"}, blockers
    branch = None
    branches = checkpoint_step.get("resolution_branches")
    if isinstance(branches, list):
        for item in branches:
            if isinstance(item, Mapping) and item.get("branch_id") == "continue_or_retrain_cloth_move_to_expected_step":
                branch = item
                break
    if not isinstance(branch, Mapping):
        blockers.append({"surface": "m4_unblock_plan", "reason": "SELF_TRAINING_BRANCH_MISSING"})
        return {"branch_id": "continue_or_retrain_cloth_move_to_expected_step", "status": "missing"}, blockers
    if branch.get("environment") != environment:
        blockers.append(
            {
                "surface": "m4_unblock_plan",
                "reason": "SELF_TRAINING_BRANCH_ENVIRONMENT_MISMATCH",
                "branch_environment": branch.get("environment"),
                "required_environment": environment,
            }
        )
    if branch.get("human_approval_required") is not True:
        blockers.append({"surface": "m4_unblock_plan", "reason": "SELF_TRAINING_BRANCH_APPROVAL_CONTRACT_MISSING"})
    if branch.get("fresh_gpu_exclusivity_audit_required") is not True:
        blockers.append({"surface": "m4_unblock_plan", "reason": "SELF_TRAINING_BRANCH_GPU_AUDIT_CONTRACT_MISSING"})
    return {
        "branch_id": branch.get("branch_id"),
        "status": branch.get("status"),
        "authority": branch.get("authority"),
        "preferred": branch.get("preferred"),
        "human_approval_required": branch.get("human_approval_required"),
        "fresh_gpu_exclusivity_audit_required": branch.get("fresh_gpu_exclusivity_audit_required"),
        "m4_launch_allowed_after_branch": branch.get("m4_launch_allowed_after_branch"),
        "required_evidence": branch.get("required_evidence", []),
    }, blockers


def _budget_summary(
    *,
    goal: Mapping[str, Any],
    estimated_gpu_hours: float,
    max_allowed_gpu_hours: float,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    goal_budget = goal.get("budget") if isinstance(goal.get("budget"), Mapping) else {}
    total_budget = _number(goal_budget.get("total_gpu_hours")) if isinstance(goal_budget, Mapping) else None
    per_trial_cap = _number(goal_budget.get("per_trial_max_gpu_hours")) if isinstance(goal_budget, Mapping) else None
    if estimated_gpu_hours > max_allowed_gpu_hours:
        blockers.append(
            {
                "surface": "budget",
                "reason": "ESTIMATED_GPU_HOURS_EXCEEDS_SELF_TRAINING_CAP",
                "estimated_gpu_hours": estimated_gpu_hours,
                "max_allowed_gpu_hours": max_allowed_gpu_hours,
            }
        )
    if total_budget is not None and estimated_gpu_hours > total_budget:
        blockers.append(
            {
                "surface": "budget",
                "reason": "ESTIMATED_GPU_HOURS_EXCEEDS_GOAL_TOTAL_BUDGET",
                "estimated_gpu_hours": estimated_gpu_hours,
                "goal_total_gpu_hours": total_budget,
            }
        )
    return {
        "ready": not blockers,
        "summary": {
            "estimated_gpu_hours": estimated_gpu_hours,
            "max_allowed_gpu_hours": max_allowed_gpu_hours,
            "goal_total_gpu_hours": total_budget,
            "goal_per_trial_max_gpu_hours": per_trial_cap,
            "high_cost_branch": True,
            "training_budget_debited": False,
            "budget_authority": "human_protocol_and_gpu_budget",
        },
        "blockers": blockers,
    }


def _gpu_preflight(
    *,
    manifest_path: Path | None,
    requested_gpus: tuple[int, ...],
    max_age_seconds: float,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    if not requested_gpus:
        blockers.append({"surface": "gpu_exclusivity", "reason": "REQUESTED_GPUS_REQUIRED_FOR_SELF_TRAINING"})
        return {
            "ready": False,
            "summary": {
                "state": "not_provided",
                "requested_gpus": [],
                "manifest_path": None,
                "max_age_seconds": max_age_seconds,
            },
            "blockers": blockers,
        }
    if manifest_path is None:
        blockers.append({"surface": "gpu_exclusivity", "reason": "FRESH_GPU_EXCLUSIVITY_AUDIT_REQUIRED"})
        return {
            "ready": False,
            "summary": {
                "state": "not_provided",
                "requested_gpus": list(requested_gpus),
                "manifest_path": None,
                "max_age_seconds": max_age_seconds,
            },
            "blockers": blockers,
        }
    try:
        authorization = verify_gpu_exclusivity_ready(
            manifest_path,
            requested_gpus=requested_gpus,
            max_age_seconds=max_age_seconds,
        )
    except GpuExclusivityAuditError as exc:
        blockers.append(
            {
                "surface": "gpu_exclusivity",
                "reason": "GPU_EXCLUSIVITY_AUDIT_NOT_READY",
                "detail": str(exc),
            }
        )
        return {
            "ready": False,
            "summary": {
                "state": "blocked",
                "requested_gpus": list(requested_gpus),
                "manifest_path": str(manifest_path),
                "max_age_seconds": max_age_seconds,
                "error": str(exc),
            },
            "blockers": blockers,
        }
    return {
        "ready": True,
        "summary": {
            "state": "ready",
            "requested_gpus": list(requested_gpus),
            "manifest_path": authorization["manifest_path"],
            "manifest_sha256": authorization["manifest_sha256"],
            "report_path": authorization["report_path"],
            "age_seconds": authorization["age_seconds"],
            "max_age_seconds": max_age_seconds,
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
        },
        "blockers": [],
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _state(
    *,
    blockers: Sequence[Mapping[str, object]],
    human_approved: bool,
    gpu_ready: bool,
    budget_ready: bool,
) -> str:
    if not human_approved:
        return "awaiting_human_approval"
    if not budget_ready:
        return "awaiting_budget_resolution"
    if not gpu_ready:
        return "awaiting_fresh_gpu_audit"
    if blockers:
        return "blocked"
    return "staged_for_manual_launch_planning"


def _next_actions(*, state: str, environment: str, requested_gpus: Sequence[int]) -> list[str]:
    if state == "staged_for_manual_launch_planning":
        return [
            f"Create a separate launch plan for checkpoint continuation training on requested GPUs {list(requested_gpus)}.",
            f"Write training output to a non-active staging path, then place the verified candidate under results/quarantine/checkpoints/{environment}/latest.pt.",
            "Run checkpoint_quarantine validate before any active checkpoint install.",
        ]
    if state == "awaiting_human_approval":
        return [
            "Obtain explicit human approval that self-training is an accepted checkpoint-source recovery path.",
            "Do not start checkpoint continuation training from this packet.",
        ]
    if state == "awaiting_fresh_gpu_audit":
        return [
            "Regenerate a fresh GPU exclusivity audit covering the requested non-user-occupied GPUs.",
            "Rerun this preflight with the fresh audit before launch planning.",
        ]
    if state == "awaiting_budget_resolution":
        return ["Resolve GPU-hour budget approval for the high-cost checkpoint recovery branch."]
    return ["Resolve blockers and rerun this preflight before any checkpoint self-training launch planning."]


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-self-training-preflight.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-self-training-preflight.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-self-training-preflight-manifest",
            "state": report["state"],
            "scope": report["scope"],
            "environment": report["environment"],
            "expected_step": report["expected_step"],
            "human_approval_provided": report["human_approval_provided"],
            "self_training_launch_planning_ready": report["self_training_launch_planning_ready"],
            "packet_grants_training_launch_permission": False,
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "blocker_count": report["blocker_count"],
            "blockers": report["blockers"],
            "side_effects": report["side_effects"],
            "report_path": str(destination / "checkpoint-self-training-preflight.json"),
            "markdown_path": str(destination / "checkpoint-self-training-preflight.md"),
            "cas_refs": {
                "checkpoint_self_training_preflight_json": report_ref,
                "checkpoint_self_training_preflight_markdown": markdown_ref,
            },
            "next_actions": report["next_actions"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Checkpoint Self-Training Preflight",
        "",
        f"State: `{report['state']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Human approval provided: `{report['human_approval_provided']}`",
        f"Launch planning ready: `{report['self_training_launch_planning_ready']}`",
        f"Packet grants training launch permission: `{report['packet_grants_training_launch_permission']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Budget",
        "",
    ]
    budget = report.get("budget")
    if isinstance(budget, Mapping):
        for key in ("estimated_gpu_hours", "max_allowed_gpu_hours", "goal_total_gpu_hours", "high_cost_branch"):
            lines.append(f"- {key}: `{budget.get(key)}`")
    gpu = report.get("gpu_preflight")
    if isinstance(gpu, Mapping):
        lines.extend(["", "## GPU Preflight", ""])
        for key in ("state", "requested_gpus", "manifest_path", "age_seconds", "max_age_seconds"):
            lines.append(f"- {key}: `{gpu.get(key)}`")
    blockers = report.get("blockers")
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_PREFLIGHT_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write a checkpoint self-training branch preflight packet")
    run.add_argument("--checkpoint-recovery-packet-manifest", type=Path, required=True)
    run.add_argument("--m4-unblock-plan-manifest", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--environment", default="cloth_move")
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--estimated-gpu-hours", type=float, default=120.0)
    run.add_argument("--max-allowed-gpu-hours", type=float, default=120.0)
    run.add_argument("--requested-gpu", type=int, action="append", default=[])
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-audit-max-age-seconds", type=float, default=300.0)
    run.add_argument("--confirm-human-approved-self-training", action="store_true")
    run.add_argument("--quarantine-root", type=Path, default=Path("results/quarantine/checkpoints"))
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_checkpoint_self_training_preflight(
                checkpoint_recovery_packet_manifest=args.checkpoint_recovery_packet_manifest,
                m4_unblock_plan_manifest=args.m4_unblock_plan_manifest,
                goal_config=args.goal_config,
                output_root=args.output_root,
                environment=args.environment,
                expected_step=args.expected_step,
                estimated_gpu_hours=args.estimated_gpu_hours,
                max_allowed_gpu_hours=args.max_allowed_gpu_hours,
                requested_gpus=args.requested_gpu,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_audit_max_age_seconds=args.gpu_audit_max_age_seconds,
                confirm_human_approved_self_training=args.confirm_human_approved_self_training,
                quarantine_root=args.quarantine_root,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
            )
            print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            return 0
        raise CheckpointSelfTrainingPreflightError("CHECKPOINT_SELF_TRAINING_COMMAND_INVALID")
    except CheckpointSelfTrainingPreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
