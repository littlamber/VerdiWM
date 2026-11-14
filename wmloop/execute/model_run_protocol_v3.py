"""Protocol-v3 Agibot execution orchestration and compatibility exports."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from wmloop.execute import gpu_lease
from wmloop.execute.gpu_lease import GpuLeaseCancelled, GpuLeaseError, GpuLeaseManager
from wmloop.execute.model_run_budget import ModelRunBudget, ModelRunBudgetError
from wmloop.execute.model_runner import (
    ModelRunnerError,
    ProcessRunner,
    _StageCancelled,
    _lease_environment,
    _run,
    _stage_record,
    _write_timeout_receipt,
    _append_torch_elastic_failure_diagnostics,
    _append_torch_elastic_timeout_diagnostics,
    _configure_torch_elastic_logging,
)
from wmloop.execute._agibot_runtime_isolation import agibot_runtime_context
from wmloop.execute._agibot_native_config_preflight import (
    AgibotNativeConfigPreflightError,
    preflight_agibot_native_config,
)
from wmloop.execute._model_run_protocol_v3_trace import (
    _agibot_bound_receipt_fields,
    _agibot_receipt_bindings,
    _build_agibot_bridge_command,
    _build_agibot_checkpoint_evaluator_command,
    _merge_agibot_training_trace,
    _validate_agibot_training_trace,
    _validate_trace_config_binding,
    _write_agibot_trace_config,
    validate_agibot_support_at_execution,
)
from wmloop.execute._model_run_protocol_v3_receipts import (
    _derive_agibot_checkpoint_receipt,
    _derive_agibot_diagnostic_receipt,
    _mark_launch_attempted,
    _persist_protocol_v3_receipt,
    _protocol_v3_receipt,
    _timeout_stage_record,
    _validate_agibot_checkpoint_handoff,
    _validate_agibot_diagnostic_handoff,
    _validate_agibot_training_progress,
    _write_protocol_receipt,
)
from wmloop.execute.agibot_psnr_contract import (
    build_agibot_psnr_execution_status,
)
from wmloop.storage import checked_path

def execute_protocol_v3(
    manifest: Mapping[str, object],
    *,
    output: Path,
    manager: GpuLeaseManager,
    indices: Sequence[int],
    world_size: int,
    train: Sequence[str],
    evaluate: Sequence[str],
    process_runner: ProcessRunner | None,
    lease_wait_seconds: float,
    cancel_requested: Callable[[], bool],
) -> dict[str, object]:
    """Run a bounded Agibot diagnostic with crash-safe budget settlement."""

    started = time.monotonic()
    receipt_path = output / "execution-receipt.json"
    if not (receipt_path.exists() or receipt_path.is_symlink()) and not cancel_requested():
        try:
            preflight_agibot_native_config(manifest)
        except AgibotNativeConfigPreflightError as exc:
            raise ModelRunnerError(str(exc)) from exc

    try:
        budget = ModelRunBudget.admit(manifest)
    except ModelRunBudgetError as exc:
        raise ModelRunnerError(str(exc)) from exc
    try:
        recovered = budget.recover_terminal(receipt_path)
    except ModelRunBudgetError as exc:
        raise ModelRunnerError(str(exc)) from exc
    if recovered is not None:
        # Older terminal receipts predate the explicit paired-PSNR contract.
        # Add the capability status to the returned projection so recovery
        # cannot hide that the engineering diagnostic is not a quality run.
        recovered = dict(recovered)
        recovered.setdefault(
            "agibot_psnr_execution",
            build_agibot_psnr_execution_status(manifest),
        )
        if recovered.get("state") == "completed":
            training_trace_path, training_trace_sha256 = _validate_agibot_training_trace(
                manifest, output
            )
            expected_checkpoint_receipt = _derive_agibot_checkpoint_receipt(
                manifest,
                output,
                training_trace_path=training_trace_path,
                training_trace_sha256=training_trace_sha256,
            )
            try:
                persisted_checkpoint_receipt = json.loads(
                    Path(str(manifest["training"]["checkpoint_receipt"])).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_RECEIPT_INVALID") from exc
            if persisted_checkpoint_receipt != expected_checkpoint_receipt:
                raise ModelRunnerError("MODEL_RUN_RECOVERY_CHECKPOINT_BINDING_MISMATCH")
            checkpoint, checkpoint_sha256, checkpoint_training_progress = (
                _validate_agibot_checkpoint_handoff(
                    Path(str(manifest["training"]["checkpoint_receipt"])),
                    manifest,
                    output,
                )
            )
            evidence_path = Path(str(manifest["evaluation"]["evidence_receipt"]))
            raw_evaluation_path = output / "receipts" / "checkpoint-evaluation-raw.json"
            expected_evidence = _derive_agibot_diagnostic_receipt(
                raw_evaluation_path,
                manifest,
                checkpoint,
                checkpoint_sha256,
                checkpoint_training_progress,
            )
            try:
                persisted_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_INVALID") from exc
            if persisted_evidence != expected_evidence:
                raise ModelRunnerError("MODEL_RUN_RECOVERY_DIAGNOSTIC_BINDING_MISMATCH")
            _validate_agibot_diagnostic_handoff(
                evidence_path,
                manifest,
                checkpoint,
                checkpoint_sha256,
                checkpoint_training_progress,
            )
            if (
                recovered.get("trained_checkpoint_path") != str(checkpoint)
                or recovered.get("trained_checkpoint_sha256") != checkpoint_sha256
                or recovered.get("engineering_diagnostic_receipt") != str(evidence_path)
                or recovered.get("training_trace_path") != str(training_trace_path)
                or recovered.get("training_trace_sha256") != training_trace_sha256
            ):
                raise ModelRunnerError("MODEL_RUN_RECOVERY_RECEIPT_BINDING_MISMATCH")
        return recovered

    native_config = manifest["training"]["agibot_native_config"]
    config_path = Path(str(native_config["path"]))
    allowed_existing = {output / "model-run.json", output / "config", config_path}
    existing = [path for path in output.rglob("*") if path not in allowed_existing]
    if existing:
        receipt = _protocol_v3_receipt(
            manifest,
            state="failed_before_launch",
            stages=[],
            leases=[],
            elapsed_seconds=time.monotonic() - started,
            launch_attempted=False,
            budget=budget,
            failure_code="MODEL_RUN_OUTPUT_NOT_FRESH",
        )
        _persist_protocol_v3_receipt(budget, receipt_path, receipt)
        raise ModelRunnerError("MODEL_RUN_OUTPUT_NOT_FRESH")

    stages: list[dict[str, object]] = []
    try:
        if cancel_requested():
            receipt = _protocol_v3_receipt(
                manifest,
                state="cancelled_before_launch",
                stages=[_stage_record("train", train, None),
                        _stage_record("engineering_diagnostic", evaluate, None)],
                leases=[],
                elapsed_seconds=time.monotonic() - started,
                launch_attempted=False,
                budget=budget,
            )
            _persist_protocol_v3_receipt(budget, receipt_path, receipt)
            return receipt
        if type(manager) is gpu_lease.GpuLeaseManager:
            leases = manager.acquire_many(
                indices,
                world_size,
                wait_seconds=lease_wait_seconds,
                cancel_requested=cancel_requested,
            )
        else:
            leases = manager.acquire_many(indices, world_size, wait_seconds=lease_wait_seconds)
    except GpuLeaseCancelled:
        receipt = _protocol_v3_receipt(
            manifest,
            state="cancelled_before_launch",
            stages=[_stage_record("train", train, None),
                    _stage_record("engineering_diagnostic", evaluate, None)],
            leases=[],
            elapsed_seconds=time.monotonic() - started,
            launch_attempted=False,
            budget=budget,
        )
        _persist_protocol_v3_receipt(budget, receipt_path, receipt)
        return receipt
    except GpuLeaseError as exc:
        receipt = _protocol_v3_receipt(
            manifest,
            state="failed_before_launch",
            stages=[],
            leases=[],
            elapsed_seconds=time.monotonic() - started,
            launch_attempted=False,
            budget=budget,
            failure_code=str(exc),
        )
        _persist_protocol_v3_receipt(budget, receipt_path, receipt)
        raise ModelRunnerError(str(exc)) from exc
    except BaseException as exc:
        receipt = _protocol_v3_receipt(
            manifest,
            state="failed_before_launch",
            stages=[],
            leases=[],
            elapsed_seconds=time.monotonic() - started,
            launch_attempted=False,
            budget=budget,
            failure_code="GPU_LEASE_ACQUISITION_FAILED",
            failure_message=str(exc),
        )
        _persist_protocol_v3_receipt(budget, receipt_path, receipt)
        raise

    lease_started = time.monotonic()
    launch_state = {"attempted": False}
    launch_attempted = False
    state = "completed"
    execution_error: BaseException | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    checkpoint_training_progress: Mapping[str, object] | None = None
    training_trace_path: Path | None = None
    training_trace_sha256: str | None = None
    success_fields: dict[str, object] = {}
    timeout_seconds: float | None = None
    timed_out_stage: str | None = None
    runtime_cleanup: Callable[[], None] | None = None
    try:
        try:
            cwd, env, runtime_cleanup = agibot_runtime_context(
                manifest,
                output,
                lease_environment=_lease_environment(leases),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRunnerError("MODEL_RUN_RUNTIME_ISOLATION_INVALID") from exc
        trace_config_path = _write_agibot_trace_config(manifest, output)
        checkpoint_raw_trace_path = output / "receipts" / "checkpoint-evaluation-raw.json"
        execution_policy = manifest.get("execution_policy", {})
        stage_timeouts = (
            execution_policy.get("stage_timeouts_seconds")
            if isinstance(execution_policy, Mapping)
            else None
        )
        if not isinstance(stage_timeouts, Mapping):
            raise ModelRunnerError("MODEL_RUN_TIMEOUT_INVALID")
        for stage, base_command in (
            ("train", train),
            ("engineering_diagnostic", evaluate),
        ):
            command = list(base_command)
            if stage == "train":
                command = _build_agibot_bridge_command(
                    manifest,
                    command,
                    trace_config_path=trace_config_path,
                )
            stage_timeout = float(stage_timeouts[stage])
            if not math.isfinite(stage_timeout) or stage_timeout <= 0:
                raise ModelRunnerError("MODEL_RUN_TIMEOUT_INVALID")
            if stage == "engineering_diagnostic":
                if checkpoint is None or checkpoint_sha256 is None:
                    raise ModelRunnerError("MODEL_RUN_CHECKPOINT_HANDOFF_MISSING")
                command = _build_agibot_checkpoint_evaluator_command(
                    manifest,
                    trace_config_path=trace_config_path,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    raw_output=checkpoint_raw_trace_path,
                )
                env["VERDIWM_TRAINED_CHECKPOINT"] = str(checkpoint)
            command, child_log_root = _configure_torch_elastic_logging(
                command, output=output, stage=stage
            )
            try:
                result = _run(
                    process_runner,
                    command,
                    cwd=cwd,
                    env=env,
                    timeout_seconds=stage_timeout,
                    cancel_requested=cancel_requested,
                    on_launch_attempt=lambda: _mark_launch_attempted(launch_state),
                )
            except subprocess.TimeoutExpired as exc:
                exc = _append_torch_elastic_timeout_diagnostics(exc, child_log_root)
                state = "timed_out"
                timed_out_stage = stage
                timeout_seconds = stage_timeout
                stages.append(_timeout_stage_record(stage, command, exc))
                _write_timeout_receipt(
                    output,
                    manifest,
                    stage,
                    stage_timeout,
                    exc,
                    completed_stages=stages[:-1],
                    leases=leases,
                    elapsed_seconds=time.monotonic() - started,
                )
                raise ModelRunnerError(f"MODEL_RUN_{stage.upper()}_TIMEOUT") from exc
            except _StageCancelled as exc:
                stages.append(_stage_record(stage, command, exc.result))
                state = "cancelled"
                return_error = ModelRunnerError("MODEL_RUN_CANCELLED")
                execution_error = return_error
                break
            result = _append_torch_elastic_failure_diagnostics(result, child_log_root)
            stages.append(_stage_record(stage, command, result))
            if result.returncode != 0:
                raise ModelRunnerError(f"MODEL_RUN_{stage.upper()}_FAILED")
            if stage == "train":
                training_trace_path, training_trace_sha256 = _merge_agibot_training_trace(
                    manifest,
                    output,
                    trace_config_path=trace_config_path,
                )
                checkpoint_receipt = _derive_agibot_checkpoint_receipt(
                    manifest,
                    output,
                    training_trace_path=training_trace_path,
                    training_trace_sha256=training_trace_sha256,
                )
                _write_protocol_receipt(
                    Path(str(manifest["training"]["checkpoint_receipt"])),
                    checkpoint_receipt,
                )
                checkpoint, checkpoint_sha256, checkpoint_training_progress = _validate_agibot_checkpoint_handoff(
                    Path(str(manifest["training"]["checkpoint_receipt"])),
                    manifest,
                    output,
                )
        if execution_error is None:
            if checkpoint is None or checkpoint_sha256 is None:
                raise ModelRunnerError("MODEL_RUN_CHECKPOINT_HANDOFF_MISSING")
            if checkpoint_training_progress is None:
                raise ModelRunnerError("MODEL_RUN_CHECKPOINT_TRAINING_PROGRESS_MISSING")
            evidence_path = Path(str(manifest["evaluation"]["evidence_receipt"]))
            raw_evaluation = checked_path(
                checkpoint_raw_trace_path,
                code="MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_INVALID",
                error=ModelRunnerError,
            )
            evidence = _derive_agibot_diagnostic_receipt(
                raw_evaluation,
                manifest,
                checkpoint,
                checkpoint_sha256,
                checkpoint_training_progress,
            )
            _write_protocol_receipt(evidence_path, evidence)
            _validate_agibot_diagnostic_handoff(
                evidence_path,
                manifest,
                checkpoint,
                checkpoint_sha256,
                checkpoint_training_progress,
            )
            success_fields = {
                "checkpoint_receipt": str(Path(str(manifest["training"]["checkpoint_receipt"]))),
                "trained_checkpoint_path": str(checkpoint),
                "trained_checkpoint_sha256": checkpoint_sha256,
                "engineering_diagnostic_receipt": str(evidence_path),
                "training_trace_path": str(training_trace_path),
                "training_trace_sha256": training_trace_sha256,
                "settlement_class": "engineering_diagnostic",
                "evidence_class": "runtime_only",
                "quality_eligible": False,
            }
    except BaseException as exc:
        if execution_error is None:
            execution_error = exc
            if state != "timed_out":
                state = "failed"
    finally:
        launch_attempted = launch_state["attempted"]
        release_error: BaseException | None = None
        if runtime_cleanup is not None:
            try:
                runtime_cleanup()
            except BaseException as exc:
                if execution_error is None:
                    execution_error = ModelRunnerError(
                        "MODEL_RUN_RUNTIME_TMP_CLEANUP_FAILED"
                    )
                    state = "failed"
        for lease in leases:
            try:
                lease.release()
            except BaseException as exc:  # still settle the time spent holding the other leases
                if release_error is None:
                    release_error = exc
        if release_error is not None and execution_error is None:
            execution_error = ModelRunnerError("MODEL_RUN_GPU_LEASE_RELEASE_FAILED")
            state = "failed"

    allocated_gpu_hours = (
        len(leases) * (time.monotonic() - lease_started) / 3600.0
        if launch_attempted
        else None
    )
    if not launch_attempted and state in {"failed", "timed_out", "cancelled"}:
        state = "failed_before_launch" if state == "failed" else "cancelled_before_launch"
    receipt = _protocol_v3_receipt(
        manifest,
        state=state,
        stages=stages,
        leases=leases,
        elapsed_seconds=time.monotonic() - started,
        launch_attempted=launch_attempted,
        budget=budget,
        allocated_gpu_hours=allocated_gpu_hours,
        success_fields=success_fields if state == "completed" else None,
        failure_code=(
            str(execution_error) if execution_error is not None else None
        ),
        timed_out_stage=timed_out_stage,
    )
    _persist_protocol_v3_receipt(budget, receipt_path, receipt)
    if execution_error is not None:
        if isinstance(execution_error, ModelRunnerError):
            raise execution_error
        raise ModelRunnerError("MODEL_RUN_EXECUTION_FAILED") from execution_error
    return receipt
