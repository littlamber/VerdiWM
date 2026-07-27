"""Read-only execution contracts and materialization CLI for exact M0 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.evaluate.cohort_view import cohort_split
from wmloop.evaluate.plan import BaselineEvaluationPlan, EvaluationSelection, build_baseline_evaluation_plan
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.freeze import verify_acwm_dataset_freeze, verify_acwm_heldout_protocol, verify_evaluator_freeze
from wmloop.runtime_env import runtime_subprocess_env
from wmloop.vendor import verify_vendor_checkout


class BaselineLaunchError(RuntimeError):
    """An exact baseline task could not be planned or executed safely."""


@dataclass(frozen=True)
class BaselineLaunchTask:
    task_id: str
    environment: str
    cohort: str
    split: str
    gpu: int
    selection: EvaluationSelection
    source_revision: str
    evaluator_freeze_sha256: str
    repository_root: Path
    vendor_root: Path
    runtime_python: Path
    data_root: Path
    checkpoint_path: Path
    checkpoint_expected_step: int | None
    checkpoint_observed_step: int | None
    vae_path: Path
    task_root: Path
    selection_path: Path
    cohort_view_root: Path
    output_root: Path
    view_command: tuple[str, ...]
    evaluation_command: tuple[str, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "environment": self.environment,
            "cohort": self.cohort,
            "split": self.split,
            "gpu": self.gpu,
            "trajectory_ids": list(self.selection.trajectory_ids),
            "source_revision": self.source_revision,
            "evaluator_freeze_sha256": self.evaluator_freeze_sha256,
            "checkpoint_expected_step": self.checkpoint_expected_step,
            "checkpoint_observed_step": self.checkpoint_observed_step,
            "cohort_view_root": str(self.cohort_view_root),
            "output_root": str(self.output_root),
            "view_command": list(self.view_command),
            "evaluation_command": list(self.evaluation_command),
        }


@dataclass(frozen=True)
class BaselineLaunchPlan:
    dataset_freeze_sha256: str
    heldout_protocol_sha256: str
    source_revision: str
    evaluator_freeze_sha256: str
    gpus: tuple[int, int, int]
    run_root: Path
    tasks: tuple[BaselineLaunchTask, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "acwm-m0-baseline-launch-plan",
            "dataset_freeze_sha256": self.dataset_freeze_sha256,
            "heldout_protocol_sha256": self.heldout_protocol_sha256,
            "source_revision": self.source_revision,
            "evaluator_freeze_sha256": self.evaluator_freeze_sha256,
            "gpus": list(self.gpus),
            "tasks": [task.to_document() for task in self.tasks],
        }


@dataclass(frozen=True)
class BaselineTaskReceipt:
    task_id: str
    state: str
    view_returncode: int | None
    evaluation_returncode: int | None
    receipt_path: Path
    planned_gpu: int
    actual_gpu: int
    gpu_exclusivity_audit: Mapping[str, object] | None = None

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "task_id": self.task_id,
            "state": self.state,
            "view_returncode": self.view_returncode,
            "evaluation_returncode": self.evaluation_returncode,
            "receipt_path": str(self.receipt_path),
            "planned_gpu": self.planned_gpu,
            "actual_gpu": self.actual_gpu,
        }
        if self.gpu_exclusivity_audit is not None:
            document["gpu_exclusivity_audit"] = dict(self.gpu_exclusivity_audit)
        return document


@dataclass(frozen=True)
class BaselineTaskStatus:
    task_id: str
    environment: str
    cohort: str
    gpu: int
    planned_gpu: int
    actual_gpu: int | None
    state: str
    receipt_path: Path
    output_root: Path
    cohort_view_root: Path

    def to_document(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "environment": self.environment,
            "cohort": self.cohort,
            "gpu": self.gpu,
            "planned_gpu": self.planned_gpu,
            "actual_gpu": self.actual_gpu,
            "state": self.state,
            "receipt_path": str(self.receipt_path),
            "output_root": str(self.output_root),
            "cohort_view_root": str(self.cohort_view_root),
        }


def build_baseline_launch_plan(
    evaluation_plan: BaselineEvaluationPlan,
    *,
    repo_root: Path,
    data_root: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    run_root: Path,
    steps: int = 50,
    test_cuts: int = 10,
    batch_size: int = 2,
    num_workers: int = 4,
    gpus: tuple[int, int, int] = (0, 1, 2),
    expected_checkpoint_step: int | None = None,
) -> BaselineLaunchPlan:
    """Bind every exact cohort to a fresh metadata view and upstream command."""

    if min(steps, test_cuts, batch_size, num_workers) < 1:
        raise BaselineLaunchError("BASELINE_LAUNCH_ARGUMENT_INVALID")
    if expected_checkpoint_step is not None and (
        not isinstance(expected_checkpoint_step, int) or isinstance(expected_checkpoint_step, bool) or expected_checkpoint_step < 1
    ):
        raise BaselineLaunchError("BASELINE_LAUNCH_ARGUMENT_INVALID")
    root = Path(repo_root).resolve(strict=True)
    vendor = root / "vendor" / "ACWM-Phys"
    source_revision = verify_vendor_checkout(root)
    evaluator_manifest = _load_evaluator_manifest(root)
    verify_evaluator_freeze(vendor, evaluator_manifest)
    evaluator_digest = _canonical_sha256(evaluator_manifest)
    runtime = _runtime_executable(runtime_python)
    data = Path(data_root).resolve(strict=True)
    checkpoints = Path(checkpoint_root).resolve(strict=True)
    vae = _regular_file(checkpoints / "Wan2.1_VAE.pth", "BASELINE_LAUNCH_VAE_MISSING")
    destination = Path(run_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BaselineLaunchError("BASELINE_LAUNCH_OUTPUT_EXISTS")
    gpu_by_environment = _gpu_assignment(evaluation_plan, gpus=gpus)
    specs = {spec.environment: spec for spec in CANONICAL_ACWM_ENVIRONMENTS}
    checkpoint_steps: dict[Path, int] = {}
    tasks: list[BaselineLaunchTask] = []
    for selection in evaluation_plan.selections:
        spec = specs.get(selection.environment)
        if spec is None:
            raise BaselineLaunchError("BASELINE_LAUNCH_ENVIRONMENT_UNKNOWN")
        checkpoint = _regular_file(checkpoints / spec.checkpoint_relative_path, "BASELINE_LAUNCH_CHECKPOINT_MISSING")
        observed_checkpoint_step: int | None = None
        if expected_checkpoint_step is not None:
            observed_checkpoint_step = checkpoint_steps.get(checkpoint)
            if observed_checkpoint_step is None:
                observed_checkpoint_step = _checkpoint_step_with_runtime(runtime, checkpoint)
                checkpoint_steps[checkpoint] = observed_checkpoint_step
            if observed_checkpoint_step != expected_checkpoint_step:
                raise BaselineLaunchError(
                    f"BASELINE_LAUNCH_CHECKPOINT_STEP_MISMATCH:{selection.environment}:{observed_checkpoint_step}:{expected_checkpoint_step}"
                )
        task_id = f"m0-{selection.environment}-{selection.cohort}"
        task_root = destination / "tasks" / task_id
        selection_path = task_root / "trajectory-ids.json"
        cohort_view_root = destination / "cohort-views" / task_id
        output_root = destination / "outputs" / task_id
        split = cohort_split(selection.cohort)
        view_command = (
            str(runtime), "-m", "wmloop.evaluate.cohort_view_runtime", "create",
            "--data-root", str(data), "--output-root", str(cohort_view_root),
            "--environment", selection.environment, "--cohort", selection.cohort,
            "--trajectory-ids-json", str(selection_path),
            "--dataset-freeze-sha256", evaluation_plan.dataset_freeze_sha256,
        )
        eval_command = (
            str(runtime), "eval.py", "--env", selection.vendor_environment,
            "--ckpt", str(checkpoint), "--steps", str(steps), "--split", split,
            "--max_trajs", str(len(selection.trajectory_ids)), "--test_cuts", str(test_cuts),
            "--batch_size", str(batch_size), "--num_workers", str(num_workers),
            "--output_root", str(output_root),
        )
        tasks.append(BaselineLaunchTask(
            task_id=task_id, environment=selection.environment, cohort=selection.cohort, split=split,
            gpu=gpu_by_environment[selection.environment], selection=selection,
            source_revision=source_revision, evaluator_freeze_sha256=evaluator_digest, repository_root=root,
            vendor_root=vendor, runtime_python=runtime, data_root=data, checkpoint_path=checkpoint,
            checkpoint_expected_step=expected_checkpoint_step,
            checkpoint_observed_step=observed_checkpoint_step,
            vae_path=vae, task_root=task_root, selection_path=selection_path,
            cohort_view_root=cohort_view_root, output_root=output_root,
            view_command=view_command, evaluation_command=eval_command,
        ))
    return BaselineLaunchPlan(
        dataset_freeze_sha256=evaluation_plan.dataset_freeze_sha256,
        heldout_protocol_sha256=evaluation_plan.heldout_protocol_sha256,
        source_revision=source_revision,
        evaluator_freeze_sha256=evaluator_digest,
        gpus=gpus,
        run_root=destination,
        tasks=tuple(tasks),
    )


def materialize_baseline_launch_plan(plan: BaselineLaunchPlan) -> Path:
    """Atomically publish selection files and the launch manifest for one M0 run."""

    root = plan.run_root
    if root.exists() or root.is_symlink():
        raise BaselineLaunchError("BASELINE_LAUNCH_OUTPUT_EXISTS")
    temporary = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        for task in plan.tasks:
            relative = task.selection_path.relative_to(root)
            target = temporary / relative
            target.parent.mkdir(mode=0o700, parents=True)
            _write_json(target, list(task.selection.trajectory_ids))
        _write_json(temporary / "launch-plan.json", plan.to_document())
        os.replace(temporary, root)
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil
            shutil.rmtree(temporary)
        raise
    return root / "launch-plan.json"


def create_m0_baseline_launch(
    *,
    data_root: Path,
    dataset_freeze_path: Path,
    heldout_protocol_path: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    run_root: Path,
    steps: int = 50,
    test_cuts: int = 10,
    batch_size: int = 2,
    num_workers: int = 4,
    gpus: tuple[int, int, int] = (0, 1, 2),
    expected_checkpoint_step: int | None = None,
) -> Path:
    """Verify frozen evaluation inputs, then atomically publish a no-GPU launch plan."""

    dataset_freeze = _load_json_mapping(dataset_freeze_path, "BASELINE_LAUNCH_DATASET_FREEZE_INVALID")
    heldout_protocol = _load_json_mapping(heldout_protocol_path, "BASELINE_LAUNCH_HELDOUT_PROTOCOL_INVALID")
    verify_acwm_dataset_freeze(data_root, dataset_freeze, required_splits=("ind_test", "ood_test"))
    verify_acwm_heldout_protocol(dataset_freeze, heldout_protocol)
    evaluation_plan = build_baseline_evaluation_plan(dataset_freeze, heldout_protocol, gpus=gpus)
    plan = build_baseline_launch_plan(
        evaluation_plan,
        repo_root=Path(__file__).resolve().parents[2],
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        runtime_python=runtime_python,
        run_root=run_root,
        steps=steps,
        test_cuts=test_cuts,
        batch_size=batch_size,
        num_workers=num_workers,
        gpus=gpus,
        expected_checkpoint_step=expected_checkpoint_step,
    )
    return materialize_baseline_launch_plan(plan)


def load_baseline_launch_plan(launch_plan_path: Path, *, repo_root: Path | None = None) -> BaselineLaunchPlan:
    """Load a materialized launch plan without widening any frozen selection."""

    plan_path = Path(launch_plan_path).resolve(strict=True)
    document = _load_json_mapping(plan_path, "BASELINE_LAUNCH_PLAN_INVALID")
    if document.get("schema_version") != 1 or document.get("artifact_type") != "acwm-m0-baseline-launch-plan":
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    dataset_digest = _sha256_field(document, "dataset_freeze_sha256", "BASELINE_LAUNCH_PLAN_INVALID")
    heldout_digest = _sha256_field(document, "heldout_protocol_sha256", "BASELINE_LAUNCH_PLAN_INVALID")
    source_revision = _git_revision_field(document, "source_revision", "BASELINE_LAUNCH_PLAN_INVALID")
    evaluator_digest = _sha256_field(document, "evaluator_freeze_sha256", "BASELINE_LAUNCH_PLAN_INVALID")
    gpus = _gpu_tuple(document.get("gpus"), "BASELINE_LAUNCH_PLAN_INVALID")
    root = plan_path.parent
    repository_root = Path(repo_root).resolve(strict=True) if repo_root is not None else Path(__file__).resolve().parents[2]
    vendor_root = repository_root / "vendor" / "ACWM-Phys"
    specs = {spec.environment: spec for spec in CANONICAL_ACWM_ENVIRONMENTS}
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    tasks: list[BaselineLaunchTask] = []
    seen: set[str] = set()
    for item in raw_tasks:
        if not isinstance(item, Mapping):
            raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
        task = _load_baseline_launch_task(
            item,
            run_root=root,
            repository_root=repository_root,
            vendor_root=vendor_root,
            expected_dataset_digest=dataset_digest,
            expected_source_revision=source_revision,
            expected_evaluator_digest=evaluator_digest,
            specs=specs,
        )
        if task.task_id in seen:
            raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
        if task.gpu not in gpus:
            raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
        seen.add(task.task_id)
        tasks.append(task)
    return BaselineLaunchPlan(
        dataset_freeze_sha256=dataset_digest,
        heldout_protocol_sha256=heldout_digest,
        source_revision=source_revision,
        evaluator_freeze_sha256=evaluator_digest,
        gpus=gpus,
        run_root=root,
        tasks=tuple(tasks),
    )


def baseline_task_status(task: BaselineLaunchTask) -> BaselineTaskStatus:
    """Return the durable state of one materialized task without launching it."""

    receipt = read_baseline_task_receipt(task)
    if receipt is not None:
        state = receipt.state
        actual_gpu: int | None = receipt.actual_gpu
    elif not task.selection_path.is_file():
        state = "missing_selection"
        actual_gpu = None
    elif task.cohort_view_root.exists() or task.output_root.exists():
        active = _task_has_active_process(task)
        active_gpus = _task_active_process_gpus(task) if active else set()
        state = "running" if active else "partial_without_receipt"
        actual_gpu = min(active_gpus) if len(active_gpus) == 1 else None
    else:
        state = "pending"
        actual_gpu = None
    return BaselineTaskStatus(
        task_id=task.task_id,
        environment=task.environment,
        cohort=task.cohort,
        gpu=task.gpu,
        planned_gpu=task.gpu,
        actual_gpu=actual_gpu,
        state=state,
        receipt_path=task.task_root / "receipt.json",
        output_root=task.output_root,
        cohort_view_root=task.cohort_view_root,
    )


def baseline_launch_status(plan: BaselineLaunchPlan) -> dict[str, object]:
    """Summarize a materialized launch plan for resume-oriented operation."""

    tasks = [baseline_task_status(task).to_document() for task in plan.tasks]
    by_state: dict[str, int] = {}
    for task in tasks:
        state = str(task["state"])
        by_state[state] = by_state.get(state, 0) + 1
    return {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-launch-status",
        "run_root": str(plan.run_root),
        "total_tasks": len(plan.tasks),
        "summary": by_state,
        "tasks": tasks,
    }


def read_baseline_task_receipt(task: BaselineLaunchTask) -> BaselineTaskReceipt | None:
    """Read a terminal task receipt, or return ``None`` when it is absent."""

    receipt_path = task.task_root / "receipt.json"
    if not receipt_path.exists():
        return None
    payload = _load_json_mapping(receipt_path, "BASELINE_LAUNCH_RECEIPT_INVALID")
    task_id = _string_field(payload, "task_id", "BASELINE_LAUNCH_RECEIPT_INVALID")
    if task_id != task.task_id:
        raise BaselineLaunchError("BASELINE_LAUNCH_RECEIPT_INVALID")
    state = _string_field(payload, "state", "BASELINE_LAUNCH_RECEIPT_INVALID")
    if state not in {"completed", "view_failed", "evaluation_failed"}:
        raise BaselineLaunchError("BASELINE_LAUNCH_RECEIPT_INVALID")
    planned_gpu = _optional_int_field(payload, "planned_gpu", "BASELINE_LAUNCH_RECEIPT_INVALID")
    actual_gpu = _optional_int_field(payload, "actual_gpu", "BASELINE_LAUNCH_RECEIPT_INVALID")
    gpu_exclusivity = payload.get("gpu_exclusivity_audit")
    if gpu_exclusivity is not None and not isinstance(gpu_exclusivity, Mapping):
        raise BaselineLaunchError("BASELINE_LAUNCH_RECEIPT_INVALID")
    return BaselineTaskReceipt(
        task_id=task_id,
        state=state,
        view_returncode=_optional_int_field(payload, "view_returncode", "BASELINE_LAUNCH_RECEIPT_INVALID"),
        evaluation_returncode=_optional_int_field(payload, "evaluation_returncode", "BASELINE_LAUNCH_RECEIPT_INVALID"),
        receipt_path=receipt_path,
        planned_gpu=task.gpu if planned_gpu is None else planned_gpu,
        actual_gpu=task.gpu if actual_gpu is None else actual_gpu,
        gpu_exclusivity_audit=None if gpu_exclusivity is None else dict(gpu_exclusivity),
    )


def execute_baseline_task_from_plan(
    plan: BaselineLaunchPlan,
    *,
    task_id: str,
    retry_failed: bool = False,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> BaselineTaskReceipt:
    """Execute one planned task, refusing accidental duplicate execution."""

    task = _task_by_id(plan, task_id)
    status = baseline_task_status(task)
    if status.state == "completed":
        receipt = read_baseline_task_receipt(task)
        if receipt is None:
            raise BaselineLaunchError("BASELINE_LAUNCH_RECEIPT_INVALID")
        return receipt
    if status.state in {"view_failed", "evaluation_failed"} and not retry_failed:
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_RECEIPT_EXISTS")
    if status.state not in {"pending", "view_failed", "evaluation_failed"}:
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_NOT_RUNNABLE")
    if status.state in {"view_failed", "evaluation_failed"}:
        _archive_failed_task_attempt(task)
    return execute_baseline_task(
        task,
        gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
        gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
        runner=runner,
    )


def execute_next_baseline_task(
    plan: BaselineLaunchPlan,
    *,
    gpu: int | None = None,
    execution_gpu: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> BaselineTaskReceipt | None:
    """Execute the first pending task in plan order, optionally pinned to one GPU."""

    if gpu is not None and (not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0):
        raise BaselineLaunchError("BASELINE_LAUNCH_GPU_PROFILE_INVALID")
    if execution_gpu is not None and (not isinstance(execution_gpu, int) or isinstance(execution_gpu, bool) or execution_gpu < 0):
        raise BaselineLaunchError("BASELINE_LAUNCH_GPU_PROFILE_INVALID")
    for task in plan.tasks:
        if gpu is not None and task.gpu != gpu:
            continue
        if baseline_task_status(task).state == "pending":
            return execute_baseline_task(
                task,
                execution_gpu=execution_gpu,
                gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
                runner=runner,
            )
    return None


def drain_baseline_tasks_for_gpu(
    plan: BaselineLaunchPlan,
    *,
    gpu: int,
    max_tasks: int | None = None,
    poll_seconds: float = 30.0,
    allow_work_stealing: bool = False,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run assigned tasks serially for one GPU without overlapping active work."""

    if gpu not in plan.gpus:
        raise BaselineLaunchError("BASELINE_LAUNCH_GPU_PROFILE_INVALID")
    if max_tasks is not None and (not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks < 1):
        raise BaselineLaunchError("BASELINE_LAUNCH_ARGUMENT_INVALID")
    if poll_seconds < 0:
        raise BaselineLaunchError("BASELINE_LAUNCH_ARGUMENT_INVALID")
    if not isinstance(allow_work_stealing, bool):
        raise BaselineLaunchError("BASELINE_LAUNCH_ARGUMENT_INVALID")
    receipts: list[BaselineTaskReceipt] = []
    while max_tasks is None or len(receipts) < max_tasks:
        statuses = [baseline_task_status(task) for task in plan.tasks]
        target_statuses = [
            status for status in statuses
            if status.gpu == gpu or status.actual_gpu == gpu
        ]
        blockers = [status for status in target_statuses if status.state in {"partial_without_receipt", "missing_selection"}]
        if blockers:
            raise BaselineLaunchError(f"BASELINE_LAUNCH_GPU_BLOCKED:{blockers[0].state}:{blockers[0].task_id}")
        running_on_target = [
            status for status in statuses
            if status.state == "running" and (status.actual_gpu == gpu or (status.actual_gpu is None and status.gpu == gpu))
        ]
        if running_on_target:
            if poll_seconds == 0:
                return _drain_result(gpu, receipts, "running", allow_work_stealing=allow_work_stealing)
            sleeper(poll_seconds)
            continue
        receipt = execute_next_baseline_task(
            plan,
            gpu=None if allow_work_stealing else gpu,
            execution_gpu=gpu if allow_work_stealing else None,
            gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
            runner=runner,
        )
        if receipt is None:
            return _drain_result(gpu, receipts, "no_pending_task", allow_work_stealing=allow_work_stealing)
        receipts.append(receipt)
    return _drain_result(gpu, receipts, "max_tasks_reached", allow_work_stealing=allow_work_stealing)


def archive_orphaned_partial_task_from_plan(plan: BaselineLaunchPlan, *, task_id: str) -> Path:
    """Move an unreceipted, inactive partial task aside so it can be retried."""

    task = _task_by_id(plan, task_id)
    if baseline_task_status(task).state != "partial_without_receipt":
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_NOT_ORPHANED")
    return _archive_orphaned_partial_task_attempt(task)


def execute_baseline_task(
    task: BaselineLaunchTask,
    *,
    execution_gpu: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> BaselineTaskReceipt:
    """Run one view-build then one immutable upstream evaluator invocation."""

    actual_gpu = _execution_gpu(task, execution_gpu)
    gpu_exclusivity = verify_gpu_exclusivity_ready(
        gpu_exclusivity_audit_manifest,
        gpu_index=actual_gpu,
        max_age_seconds=gpu_exclusivity_max_age_seconds,
    )
    _verify_task_source(task)
    if not task.selection_path.is_file() or task.cohort_view_root.exists() or task.output_root.exists():
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_STATE_INVALID")
    view = runner(
        list(task.view_command), cwd=task.repository_root,
        env=runtime_subprocess_env(task.runtime_python),
        capture_output=True, check=False,
    )
    _write_command_output(task.task_root, "view", view)
    if view.returncode != 0:
        return _write_task_receipt(
            task,
            "view_failed",
            view.returncode,
            None,
            actual_gpu=actual_gpu,
            gpu_exclusivity_audit=gpu_exclusivity,
        )
    environment = runtime_subprocess_env(
        task.runtime_python,
        extra={
            "ACWM_DATA_ROOT": str(task.cohort_view_root),
            "WAN_VAE_PATH": str(task.vae_path),
            "CUDA_VISIBLE_DEVICES": str(actual_gpu),
        },
    )
    evaluation = runner(list(task.evaluation_command), cwd=task.vendor_root, env=environment, capture_output=True, check=False)
    _write_command_output(task.task_root, "evaluation", evaluation)
    return _write_task_receipt(
        task,
        "completed" if evaluation.returncode == 0 else "evaluation_failed",
        view.returncode,
        evaluation.returncode,
        actual_gpu=actual_gpu,
        gpu_exclusivity_audit=gpu_exclusivity,
    )


def _load_baseline_launch_task(
    payload: Mapping[str, Any],
    *,
    run_root: Path,
    repository_root: Path,
    vendor_root: Path,
    expected_dataset_digest: str,
    expected_source_revision: str,
    expected_evaluator_digest: str,
    specs: Mapping[str, object],
) -> BaselineLaunchTask:
    task_id = _string_field(payload, "task_id", "BASELINE_LAUNCH_PLAN_INVALID")
    environment = _string_field(payload, "environment", "BASELINE_LAUNCH_PLAN_INVALID")
    cohort = _string_field(payload, "cohort", "BASELINE_LAUNCH_PLAN_INVALID")
    split = _string_field(payload, "split", "BASELINE_LAUNCH_PLAN_INVALID")
    source_revision = _git_revision_field(payload, "source_revision", "BASELINE_LAUNCH_PLAN_INVALID")
    evaluator_digest = _sha256_field(payload, "evaluator_freeze_sha256", "BASELINE_LAUNCH_PLAN_INVALID")
    if source_revision != expected_source_revision or evaluator_digest != expected_evaluator_digest:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if task_id != f"m0-{environment}-{cohort}":
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if cohort_split(cohort) != split:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    spec = specs.get(environment)
    if spec is None:
        raise BaselineLaunchError("BASELINE_LAUNCH_ENVIRONMENT_UNKNOWN")
    view_command = _string_tuple_field(payload, "view_command", "BASELINE_LAUNCH_PLAN_INVALID")
    evaluation_command = _normalize_evaluation_command(
        _string_tuple_field(payload, "evaluation_command", "BASELINE_LAUNCH_PLAN_INVALID")
    )
    if _command_option(view_command, "--environment", "BASELINE_LAUNCH_PLAN_INVALID") != environment:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if _command_option(view_command, "--cohort", "BASELINE_LAUNCH_PLAN_INVALID") != cohort:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if _command_option(view_command, "--dataset-freeze-sha256", "BASELINE_LAUNCH_PLAN_INVALID") != expected_dataset_digest:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if _command_option(evaluation_command, "--split", "BASELINE_LAUNCH_PLAN_INVALID") != split:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    data_root = Path(_command_option(view_command, "--data-root", "BASELINE_LAUNCH_PLAN_INVALID"))
    selection_path = Path(_command_option(view_command, "--trajectory-ids-json", "BASELINE_LAUNCH_PLAN_INVALID"))
    cohort_view_root = Path(_string_field(payload, "cohort_view_root", "BASELINE_LAUNCH_PLAN_INVALID"))
    if Path(_command_option(view_command, "--output-root", "BASELINE_LAUNCH_PLAN_INVALID")) != cohort_view_root:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    output_root = Path(_string_field(payload, "output_root", "BASELINE_LAUNCH_PLAN_INVALID"))
    if Path(_command_option(evaluation_command, "--output_root", "BASELINE_LAUNCH_PLAN_INVALID")) != output_root:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    runtime = Path(view_command[0])
    if Path(evaluation_command[0]) != runtime:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    checkpoint_path = Path(_command_option(evaluation_command, "--ckpt", "BASELINE_LAUNCH_PLAN_INVALID"))
    checkpoint_root = _checkpoint_root_from_path(checkpoint_path, getattr(spec, "checkpoint_relative_path"))
    checkpoint_expected_step = _optional_checkpoint_step_field(payload, "checkpoint_expected_step", "BASELINE_LAUNCH_PLAN_INVALID")
    checkpoint_observed_step = _optional_checkpoint_step_field(payload, "checkpoint_observed_step", "BASELINE_LAUNCH_PLAN_INVALID")
    if (checkpoint_expected_step is None) != (checkpoint_observed_step is None):
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    if checkpoint_expected_step is not None and checkpoint_expected_step != checkpoint_observed_step:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    max_trajs = _positive_int_text(_command_option(evaluation_command, "--max_trajs", "BASELINE_LAUNCH_PLAN_INVALID"))
    trajectory_ids = _string_tuple_field(payload, "trajectory_ids", "BASELINE_LAUNCH_PLAN_INVALID")
    if max_trajs != len(trajectory_ids):
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    vendor_environment = _command_option(evaluation_command, "--env", "BASELINE_LAUNCH_PLAN_INVALID")
    task_root = selection_path.parent
    if not _is_relative_to(task_root, run_root) or not _is_relative_to(cohort_view_root, run_root) or not _is_relative_to(output_root, run_root):
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    return BaselineLaunchTask(
        task_id=task_id,
        environment=environment,
        cohort=cohort,
        split=split,
        gpu=_int_field(payload, "gpu", "BASELINE_LAUNCH_PLAN_INVALID"),
        selection=EvaluationSelection(environment, vendor_environment, cohort, trajectory_ids),
        source_revision=source_revision,
        evaluator_freeze_sha256=evaluator_digest,
        repository_root=repository_root,
        vendor_root=vendor_root,
        runtime_python=runtime,
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        checkpoint_expected_step=checkpoint_expected_step,
        checkpoint_observed_step=checkpoint_observed_step,
        vae_path=checkpoint_root / "Wan2.1_VAE.pth",
        task_root=task_root,
        selection_path=selection_path,
        cohort_view_root=cohort_view_root,
        output_root=output_root,
        view_command=view_command,
        evaluation_command=evaluation_command,
    )


def _task_by_id(plan: BaselineLaunchPlan, task_id: str) -> BaselineLaunchTask:
    for task in plan.tasks:
        if task.task_id == task_id:
            return task
    raise BaselineLaunchError("BASELINE_LAUNCH_TASK_UNKNOWN")


def _drain_result(
    gpu: int,
    receipts: Sequence[BaselineTaskReceipt],
    reason: str,
    *,
    allow_work_stealing: bool = False,
) -> dict[str, object]:
    return {
        "gpu": gpu,
        "ran": len(receipts),
        "reason": reason,
        "allow_work_stealing": allow_work_stealing,
        "receipts": [receipt.to_document() for receipt in receipts],
    }


def _task_has_active_process(task: BaselineLaunchTask) -> bool:
    """Best-effort Linux process check for unreceipted task directories."""

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    needles = (str(task.output_root).encode(), str(task.cohort_view_root).encode())
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        cmdline = candidate / "cmdline"
        try:
            payload = cmdline.read_bytes()
        except OSError:
            continue
        if any(needle in payload for needle in needles):
            return True
    return False


def _task_active_process_gpus(task: BaselineLaunchTask) -> set[int]:
    """Return CUDA devices used by active processes for this task when visible."""

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    needles = (str(task.output_root).encode(), str(task.cohort_view_root).encode())
    gpus: set[int] = set()
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            cmdline = (candidate / "cmdline").read_bytes()
            environ = (candidate / "environ").read_bytes()
        except OSError:
            continue
        if not any(needle in cmdline for needle in needles):
            continue
        gpus.update(_cuda_visible_devices_from_environ(environ))
    return gpus


def _cuda_visible_devices_from_environ(environ: bytes) -> set[int]:
    prefix = b"CUDA_VISIBLE_DEVICES="
    for item in environ.split(b"\0"):
        if not item.startswith(prefix):
            continue
        value = item[len(prefix):].decode("utf-8", errors="ignore")
        gpus: set[int] = set()
        for token in value.split(","):
            token = token.strip()
            if token.isdecimal():
                gpus.add(int(token))
        return gpus
    return set()


def _archive_failed_task_attempt(task: BaselineLaunchTask) -> None:
    if task.cohort_view_root.exists() or task.output_root.exists():
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_NOT_RUNNABLE")
    history_root = task.task_root / "retry-history"
    history_root.mkdir(mode=0o700, exist_ok=True)
    destination = history_root / uuid.uuid4().hex
    destination.mkdir(mode=0o700)
    moved = False
    for name in ("receipt.json", "view.stdout", "view.stderr", "evaluation.stdout", "evaluation.stderr"):
        source = task.task_root / name
        if source.exists() or source.is_symlink():
            if source.is_symlink() or not source.is_file():
                raise BaselineLaunchError("BASELINE_LAUNCH_TASK_STATE_INVALID")
            os.replace(source, destination / name)
            moved = True
    if not moved:
        raise BaselineLaunchError("BASELINE_LAUNCH_RECEIPT_INVALID")


def _archive_orphaned_partial_task_attempt(task: BaselineLaunchTask) -> Path:
    history_root = task.task_root / "retry-history"
    history_root.mkdir(mode=0o700, exist_ok=True)
    destination = history_root / uuid.uuid4().hex
    destination.mkdir(mode=0o700)
    moved: list[str] = []
    for path, archive_name in (
        (task.cohort_view_root, "cohort-view"),
        (task.output_root, "output"),
    ):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise BaselineLaunchError("BASELINE_LAUNCH_TASK_STATE_INVALID")
            os.replace(path, destination / archive_name)
            moved.append(str(path))
    for name in ("view.stdout", "view.stderr", "evaluation.stdout", "evaluation.stderr"):
        source = task.task_root / name
        if source.exists() or source.is_symlink():
            if source.is_symlink() or not source.is_file():
                raise BaselineLaunchError("BASELINE_LAUNCH_TASK_STATE_INVALID")
            os.replace(source, destination / name)
            moved.append(str(source))
    if not moved:
        raise BaselineLaunchError("BASELINE_LAUNCH_TASK_NOT_ORPHANED")
    _write_json(destination / "orphan-summary.json", {"task_id": task.task_id, "archived_paths": moved})
    return destination


def _verify_task_source(task: BaselineLaunchTask) -> None:
    if verify_vendor_checkout(task.repository_root) != task.source_revision:
        raise BaselineLaunchError("BASELINE_LAUNCH_SOURCE_REVISION_CHANGED")
    manifest = _load_evaluator_manifest(task.repository_root)
    if not _is_sha256(task.evaluator_freeze_sha256):
        raise BaselineLaunchError("BASELINE_LAUNCH_EVALUATOR_FREEZE_INVALID")
    if _canonical_sha256(manifest) != task.evaluator_freeze_sha256:
        raise BaselineLaunchError("BASELINE_LAUNCH_EVALUATOR_FREEZE_CHANGED")
    verify_evaluator_freeze(task.vendor_root, manifest)


def _gpu_assignment(plan: BaselineEvaluationPlan, *, gpus: tuple[int, int, int]) -> dict[str, int]:
    if len(gpus) != 3 or len(set(gpus)) != 3 or any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in gpus):
        raise BaselineLaunchError("BASELINE_LAUNCH_GPU_PROFILE_INVALID")
    assignment: dict[str, int] = {}
    for wave in plan.waves:
        if not wave or len(wave) > len(gpus):
            raise BaselineLaunchError("BASELINE_LAUNCH_WAVES_INVALID")
        for position, environment in enumerate(wave):
            assignment[environment] = gpus[position]
    if set(assignment) != {selection.environment for selection in plan.selections}:
        raise BaselineLaunchError("BASELINE_LAUNCH_WAVES_INVALID")
    return assignment


def _execution_gpu(task: BaselineLaunchTask, execution_gpu: int | None) -> int:
    if execution_gpu is None:
        return task.gpu
    if not isinstance(execution_gpu, int) or isinstance(execution_gpu, bool) or execution_gpu < 0:
        raise BaselineLaunchError("BASELINE_LAUNCH_GPU_PROFILE_INVALID")
    return execution_gpu


def _load_evaluator_manifest(repo_root: Path) -> Mapping[str, object]:
    try:
        payload = json.loads((repo_root / "configs" / "eval_frozen.sha256").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineLaunchError("BASELINE_LAUNCH_EVALUATOR_FREEZE_MISSING") from exc
    if not isinstance(payload, Mapping):
        raise BaselineLaunchError("BASELINE_LAUNCH_EVALUATOR_FREEZE_INVALID")
    return payload


def _load_json_mapping(path: Path, code: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineLaunchError(code) from exc
    if not isinstance(payload, Mapping):
        raise BaselineLaunchError(code)
    return payload


def _string_field(payload: Mapping[str, Any], name: str, code: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise BaselineLaunchError(code)
    return value


def _sha256_field(payload: Mapping[str, Any], name: str, code: str) -> str:
    value = _string_field(payload, name, code)
    if not _is_sha256(value):
        raise BaselineLaunchError(code)
    return value


def _git_revision_field(payload: Mapping[str, Any], name: str, code: str) -> str:
    value = _string_field(payload, name, code)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise BaselineLaunchError(code)
    return value


def _int_field(payload: Mapping[str, Any], name: str, code: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BaselineLaunchError(code)
    return value


def _optional_int_field(payload: Mapping[str, Any], name: str, code: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineLaunchError(code)
    return value


def _optional_checkpoint_step_field(payload: Mapping[str, Any], name: str, code: str) -> int | None:
    value = _optional_int_field(payload, name, code)
    if value is not None and value < 0:
        raise BaselineLaunchError(code)
    return value


def _string_tuple_field(payload: Mapping[str, Any], name: str, code: str) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise BaselineLaunchError(code)
    return tuple(value)


def _gpu_tuple(value: object, code: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise BaselineLaunchError(code)
    gpus: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise BaselineLaunchError(code)
        gpus.append(item)
    if len(set(gpus)) != 3:
        raise BaselineLaunchError(code)
    return (gpus[0], gpus[1], gpus[2])


def _command_option(command: Sequence[str], option: str, code: str) -> str:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise BaselineLaunchError(code) from exc
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise BaselineLaunchError(code)
    return command[index + 1]


def _normalize_evaluation_command(command: tuple[str, ...]) -> tuple[str, ...]:
    aliases = {
        "--max-trajs": "--max_trajs",
        "--test-cuts": "--test_cuts",
        "--batch-size": "--batch_size",
        "--num-workers": "--num_workers",
        "--output-root": "--output_root",
    }
    return tuple(aliases.get(item, item) for item in command)


def _positive_int_text(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID") from exc
    if parsed < 1:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    return parsed


def _checkpoint_root_from_path(checkpoint_path: Path, checkpoint_relative_path: str) -> Path:
    relative_parts = Path(checkpoint_relative_path).parts
    actual_parts = checkpoint_path.parts
    if len(actual_parts) <= len(relative_parts) or tuple(actual_parts[-len(relative_parts):]) != relative_parts:
        raise BaselineLaunchError("BASELINE_LAUNCH_PLAN_INVALID")
    return Path(*actual_parts[:-len(relative_parts)])


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise BaselineLaunchError(code)
    return candidate.resolve(strict=True)


def _runtime_executable(path: Path) -> Path:
    """Resolve an interpreter symlink, then require a regular executable target.

    Python virtual environments commonly expose ``python`` as a symlink.  This
    is an explicit runtime dependency supplied by the launch operator, unlike
    data and checkpoint assets, which remain symlink-free provenance inputs.
    """

    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise BaselineLaunchError("BASELINE_LAUNCH_RUNTIME_MISSING") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise BaselineLaunchError("BASELINE_LAUNCH_RUNTIME_MISSING")
    return resolved


def _checkpoint_step_with_runtime(runtime_python: Path, checkpoint_path: Path) -> int:
    source = (
        "import json, sys, torch\n"
        "payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "step = payload.get('step') if isinstance(payload, dict) else None\n"
        "print(json.dumps({'step': step}))\n"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", source, str(checkpoint_path)],
        env=runtime_subprocess_env(runtime_python),
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise BaselineLaunchError("BASELINE_LAUNCH_CHECKPOINT_STEP_UNREADABLE")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineLaunchError("BASELINE_LAUNCH_CHECKPOINT_STEP_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise BaselineLaunchError("BASELINE_LAUNCH_CHECKPOINT_STEP_UNREADABLE")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise BaselineLaunchError("BASELINE_LAUNCH_CHECKPOINT_STEP_UNREADABLE")
    return step


def _write_command_output(task_root: Path, name: str, completed: subprocess.CompletedProcess[bytes]) -> None:
    task_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_bytes(task_root / f"{name}.stdout", completed.stdout or b"")
    _write_bytes(task_root / f"{name}.stderr", completed.stderr or b"")


def _write_task_receipt(
    task: BaselineLaunchTask,
    state: str,
    view_returncode: int | None,
    evaluation_returncode: int | None,
    *,
    actual_gpu: int | None = None,
    gpu_exclusivity_audit: Mapping[str, object] | None = None,
) -> BaselineTaskReceipt:
    receipt_path = task.task_root / "receipt.json"
    resolved_actual_gpu = task.gpu if actual_gpu is None else actual_gpu
    document: dict[str, object] = {
        "task_id": task.task_id, "state": state,
        "view_returncode": view_returncode, "evaluation_returncode": evaluation_returncode,
        "planned_gpu": task.gpu, "actual_gpu": resolved_actual_gpu,
    }
    if gpu_exclusivity_audit is not None:
        document["gpu_exclusivity_audit"] = dict(gpu_exclusivity_audit)
    _write_json(receipt_path, document)
    return BaselineTaskReceipt(
        task.task_id,
        state,
        view_returncode,
        evaluation_returncode,
        receipt_path,
        task.gpu,
        resolved_actual_gpu,
        None if gpu_exclusivity_audit is None else dict(gpu_exclusivity_audit),
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")


def _write_bytes(path: Path, payload: bytes) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="verify frozen M0 inputs and publish an exact no-GPU launch plan")
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--dataset-freeze", type=Path, required=True)
    create.add_argument("--heldout-protocol", type=Path, required=True)
    create.add_argument("--checkpoint-root", type=Path, required=True)
    create.add_argument("--runtime-python", type=Path, required=True)
    create.add_argument("--run-root", type=Path, required=True)
    create.add_argument("--steps", type=int, default=50)
    create.add_argument("--test-cuts", type=int, default=10)
    create.add_argument("--batch-size", type=int, default=2)
    create.add_argument("--num-workers", type=int, default=4)
    create.add_argument("--expected-checkpoint-step", type=int)
    create.add_argument("--gpus", type=int, nargs=3, default=(0, 1, 2), metavar=("GPU0", "GPU1", "GPU2"))
    status = commands.add_parser("status", help="summarize a materialized M0 launch plan without running tasks")
    status.add_argument("--launch-plan", type=Path, required=True)
    status.add_argument("--repo-root", type=Path)
    run_task = commands.add_parser("run-task", help="run one materialized M0 task and write its receipt")
    run_task.add_argument("--launch-plan", type=Path, required=True)
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--repo-root", type=Path)
    run_task.add_argument("--retry-failed", action="store_true")
    run_task.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run_task.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    run_next = commands.add_parser("run-next", help="run the first pending M0 task, optionally for one assigned GPU")
    run_next.add_argument("--launch-plan", type=Path, required=True)
    run_next.add_argument("--repo-root", type=Path)
    run_next.add_argument("--gpu", type=int)
    run_next.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run_next.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    drain_gpu = commands.add_parser("drain-gpu", help="serially drain M0 tasks for one GPU")
    drain_gpu.add_argument("--launch-plan", type=Path, required=True)
    drain_gpu.add_argument("--repo-root", type=Path)
    drain_gpu.add_argument("--gpu", type=int, required=True)
    drain_gpu.add_argument("--max-tasks", type=int)
    drain_gpu.add_argument("--poll-seconds", type=float, default=30.0)
    drain_gpu.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    drain_gpu.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    drain_gpu.add_argument(
        "--allow-work-stealing",
        action="store_true",
        help="allow this GPU to run the first globally pending task and record planned/actual GPU in the receipt",
    )
    archive_orphan = commands.add_parser("archive-orphan", help="archive an inactive partial task so it can be retried")
    archive_orphan.add_argument("--launch-plan", type=Path, required=True)
    archive_orphan.add_argument("--task-id", required=True)
    archive_orphan.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        manifest = create_m0_baseline_launch(
            data_root=args.data_root,
            dataset_freeze_path=args.dataset_freeze,
            heldout_protocol_path=args.heldout_protocol,
            checkpoint_root=args.checkpoint_root,
            runtime_python=args.runtime_python,
            run_root=args.run_root,
            steps=args.steps,
            test_cuts=args.test_cuts,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gpus=tuple(args.gpus),
            expected_checkpoint_step=args.expected_checkpoint_step,
        )
        print(json.dumps({"ready": True, "launch_plan": str(manifest), "gpu_execution_started": False}, sort_keys=True))
        return 0
    if args.command == "status":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        print(json.dumps(baseline_launch_status(plan), sort_keys=True))
        return 0
    if args.command == "run-task":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        receipt = execute_baseline_task_from_plan(
            plan,
            task_id=args.task_id,
            retry_failed=args.retry_failed,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
        )
        print(json.dumps({"ran": True, "receipt": receipt.to_document()}, sort_keys=True))
        return 0
    if args.command == "run-next":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        receipt = execute_next_baseline_task(
            plan,
            gpu=args.gpu,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
        )
        if receipt is None:
            print(json.dumps({"ran": False, "reason": "no_pending_task"}, sort_keys=True))
        else:
            print(json.dumps({"ran": True, "receipt": receipt.to_document()}, sort_keys=True))
        return 0
    if args.command == "drain-gpu":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        print(
            json.dumps(
                drain_baseline_tasks_for_gpu(
                    plan,
                    gpu=args.gpu,
                    max_tasks=args.max_tasks,
                    poll_seconds=args.poll_seconds,
                    allow_work_stealing=args.allow_work_stealing,
                    gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                    gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "archive-orphan":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        archive_path = archive_orphaned_partial_task_from_plan(plan, task_id=args.task_id)
        print(json.dumps({"archived": True, "archive_path": str(archive_path)}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
