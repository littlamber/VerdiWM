"""Agibot protocol-v3 receipts and handoff validation."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute import gpu_lease
from wmloop.execute.gpu_lease import GpuLease
from wmloop.execute.model_run_budget import ModelRunBudget, ModelRunBudgetError
from wmloop.execute.model_runner import (
    ModelRunnerError,
    _receipt,
    _stage_record,
    _stage_output_fields,
)
from wmloop.execute._model_run_protocol_v3_trace import (
    _agibot_bound_receipt_fields,
    _agibot_receipt_bindings,
)
from wmloop.execute.agibot_psnr_contract import (
    build_agibot_psnr_execution_status,
)
from wmloop.storage import atomic_write, canonical_bytes, checked_path, sha256_bytes, sha256_file

def _mark_launch_attempted(state: dict[str, bool]) -> None:
    state["attempted"] = True


def _protocol_v3_receipt(
    manifest: Mapping[str, object],
    *,
    state: str,
    stages: Sequence[Mapping[str, object]],
    leases: Sequence[GpuLease],
    elapsed_seconds: float,
    launch_attempted: bool,
    budget: ModelRunBudget,
    allocated_gpu_hours: float | None = None,
    success_fields: Mapping[str, object] | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
    timed_out_stage: str | None = None,
) -> dict[str, object]:
    admission = budget.admission
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-run-execution-receipt",
        "state": state,
        "model_run_id": manifest["model_run_id"],
        "stages": [dict(row) for row in stages],
        "leased_gpus": [lease.to_document() for lease in leases],
        "elapsed_seconds": round(elapsed_seconds, 6),
        "launch_attempted": launch_attempted,
        "estimated_gpu_hours": budget.binding["estimated_gpu_hours"],
        "budget_trial_id": budget.binding["trial_id"],
        "budget_reservation": {
            "trial_id": budget.binding["trial_id"],
            "fencing_token": admission.fencing_token,
            "plan_sha256": budget.binding["plan_sha256"],
        },
        "budget_state": "settled" if launch_attempted else "released",
        "evidence_class": "runtime_only",
        "quality_eligible": False,
        "claim_boundary": (
            "Execution and engineering-diagnostic provenance only. This receipt is not a model-effect result or formal verdict."
        ),
        "agibot_psnr_execution": build_agibot_psnr_execution_status(manifest),
    }
    if launch_attempted:
        receipt["allocated_gpu_count"] = len(leases)
        receipt["allocated_gpu_hours"] = float(allocated_gpu_hours or 0.0)
    if success_fields:
        receipt.update(success_fields)
    if failure_code:
        receipt["failure_code"] = failure_code[:240]
    if failure_message:
        receipt["failure_message"] = failure_message[:500]
    if timed_out_stage:
        receipt["timed_out_stage"] = timed_out_stage
    return receipt


def _persist_protocol_v3_receipt(
    budget: ModelRunBudget, receipt_path: Path, receipt: Mapping[str, object]
) -> None:
    try:
        budget.persist_terminal(receipt_path, receipt)
    except ModelRunBudgetError as exc:
        raise ModelRunnerError(str(exc)) from exc


def _timeout_stage_record(
    stage: str, command: Sequence[str], error: subprocess.TimeoutExpired
) -> dict[str, object]:
    return {
        "stage": stage,
        "command": list(command),
        "returncode": None,
        **_stage_output_fields(error.output, error.stderr, include_head=True),
        "state": "timed_out",
    }


def _write_protocol_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Persist protocol-owned evidence atomically below the run output."""

    try:
        atomic_write(path, canonical_bytes(dict(receipt)) + b"\n")
    except OSError as exc:
        raise ModelRunnerError("MODEL_RUN_PROTOCOL_RECEIPT_WRITE_FAILED") from exc


def _derive_agibot_checkpoint_receipt(
    manifest: Mapping[str, object],
    output_root: Path,
    *,
    training_trace_path: Path,
    training_trace_sha256: str,
) -> dict[str, object]:
    """Derive progress and checkpoint identity from native outputs and trace."""

    try:
        training = manifest["training"]
        plan = training["scale_plan"]
        parallelism = plan["parallelism"]
        updates = int(plan["updates"]["planned_steps"])
        microsteps = updates * int(parallelism["gradient_accumulation"])
        world_size = int(parallelism["world_size"])
        seed = int(plan["replication"]["seeds"][0])
        trace = json.loads(training_trace_path.read_text(encoding="utf-8"))
        events = trace.get("events") if isinstance(trace, Mapping) else None
        if not isinstance(events, list):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_TRACE_INVALID")
        native_config = training["agibot_native_config"]
        config_path = Path(str(native_config["path"])).resolve()
        checkpoint_dir = output_root / "checkpoints" / f"checkpoint-{microsteps}"
        weights_path = checked_path(
            checkpoint_dir / "dit_model.safetensors",
            code="MODEL_RUN_AGIBOT_CHECKPOINT_INVALID",
            error=ModelRunnerError,
        )
        meta_path = checked_path(
            checkpoint_dir / "meta.json",
            code="MODEL_RUN_AGIBOT_CHECKPOINT_INVALID",
            error=ModelRunnerError,
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, Mapping):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_INVALID")
        model_config = meta.get("model_config_dict")
        training_config = meta.get("training_config_dict")
        if (
            meta.get("checkpoint_contract_version") != 2
            or meta.get("step") != microsteps
            or meta.get("world_size") != world_size
            or meta.get("format") != "safetensors+optim_shards"
            or meta.get("safetensors_exported") is not True
            or not isinstance(model_config, Mapping)
            or model_config.get("max_train_steps") != microsteps
            or model_config.get("batch_size") != int(parallelism["batch_size"])
            or model_config.get("grad_accum_steps")
            != int(parallelism["gradient_accumulation"])
            or not isinstance(training_config, Mapping)
            or training_config.get("save_interval") != microsteps
            or training_config.get("val_interval") != microsteps
            or not config_path.is_file()
            or sha256_file(config_path) != native_config.get("sha256")
            or not weights_path.is_file()
            or weights_path.stat().st_size <= 0
        ):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_MISMATCH")

        train_events = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("phase") == "train"
        ]
        progress_rows: list[dict[str, object]] = []
        for update in range(1, updates + 1):
            current = [
                event
                for event in train_events
                if event.get("optimizer_update") == update
            ]
            if not current:
                raise ModelRunnerError("MODEL_RUN_AGIBOT_TRAINING_PROGRESS_INVALID")
            ordered = sorted(
                current,
                key=lambda row: (
                    int(row["rank"]),
                    int(row["microstep"]),
                    int(row["batch_slot"]),
                ),
            )
            losses = [float(row["batch_loss"]) for row in ordered]
            ids = [str(row["sample_window_id"]) for row in ordered]
            progress_rows.append(
                {
                    "optimizer_update": update,
                    "microstep": update * int(parallelism["gradient_accumulation"]),
                    "loss": sum(losses) / len(losses),
                    "sample_count": len(ordered),
                    "train_window_ids_sha256": sha256_bytes(canonical_bytes(ids)),
                }
            )
        progress = {
            "seed": seed,
            "planned_optimizer_updates": updates,
            "completed_optimizer_updates": trace.get("optimizer_update_count"),
            "planned_microsteps": microsteps,
            "completed_microsteps": microsteps,
            "training_loss_rows": progress_rows,
        }
        checkpoint_identity = {
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "optimizer_updates": updates,
            "microsteps": microsteps,
            "seed": seed,
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
            "step": microsteps,
            "world_size": world_size,
        }
        receipt = {
            "schema_version": 1,
            "artifact_type": "verdiwm-agibot-engineering-diagnostic-checkpoint-receipt",
            "state": "complete",
            "model_run_id": manifest["model_run_id"],
            "scale_plan_digest": training["scale_plan_digest"],
            "training_trace_path": str(training_trace_path.resolve()),
            "training_trace_sha256": training_trace_sha256,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_training_progress": progress,
            "support_bindings": _agibot_bound_receipt_fields(manifest),
            "claim_boundary": (
                "Native checkpoint metadata and traced optimizer progress only; this receipt does not establish model improvement."
            ),
        }
        validate_document(
            "agibot_engineering_checkpoint_receipt",
            receipt,
            root=Path(__file__).resolve().parents[2],
        )
        return receipt
    except ModelRunnerError:
        raise
    except (ContractValidationError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_DERIVATION_FAILED") from exc


def _derive_agibot_diagnostic_receipt(
    raw_trace_path: Path,
    manifest: Mapping[str, object],
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_training_progress: Mapping[str, object],
) -> dict[str, object]:
    """Validate evaluator raw output, then write the authoritative receipt."""

    try:
        raw = json.loads(raw_trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_INVALID") from exc
    if not isinstance(raw, Mapping):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_INVALID")
    try:
        validate_document(
            "agibot_checkpoint_evaluation_trace",
            raw,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_INVALID") from exc
    bindings = _agibot_receipt_bindings(manifest)
    sample_ids = bindings.get("validation_sample_window_ids")
    rows = raw.get("consumption_trace")
    metadata_path = checked_path(
        checkpoint.parent / "meta.json",
        code="MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_INVALID",
        error=ModelRunnerError,
    )
    actual_metadata_digest = sha256_file(metadata_path)
    if (
        raw.get("schema_version") != 1
        or raw.get("artifact_type") != "verdiwm-agibot-checkpoint-evaluation-trace"
        or raw.get("state") != "complete"
        or raw.get("model_run_id") != manifest.get("model_run_id")
        or raw.get("checkpoint_identity")
        != {"path": str(checkpoint), "sha256": checkpoint_sha256}
        or raw.get("checkpoint_meta_sha256") != actual_metadata_digest
        or raw.get("checkpoint_reload_verified") is not True
        or type(raw.get("checkpoint_state_key_count")) is not int
        or int(raw["checkpoint_state_key_count"]) < 1
        or raw.get("native_config_sha256")
        != manifest["training"]["agibot_native_config"]["sha256"]  # type: ignore[index]
        or raw.get("support_bindings") != _agibot_bound_receipt_fields(manifest)
        or not isinstance(rows, list)
        or not isinstance(sample_ids, list)
        or len(rows) != len(sample_ids)
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_BINDING_MISMATCH")
    trace_rows: list[dict[str, object]] = []
    for expected_id, row in zip(sample_ids, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or isinstance(row.get("loss"), bool)
            or not isinstance(row.get("loss"), (int, float))
            or not math.isfinite(float(row["loss"]))
        ):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_INVALID")
        if row.get("window_id") != expected_id:
            raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_BINDING_MISMATCH")
        trace_rows.append({"window_id": str(expected_id), "loss": float(row["loss"])})
    mean_loss = sum(float(row["loss"]) for row in trace_rows) / len(trace_rows)
    if not math.isclose(
        float(raw.get("native_validation_loss", math.nan)),
        mean_loss,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_MEAN_MISMATCH")
    trace_digest = sha256_bytes(canonical_bytes(trace_rows))
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-engineering-diagnostic-evaluation-receipt",
        "state": "complete",
        "stage": "engineering_diagnostic",
        "evidence_class": "runtime_only",
        "quality_eligible": False,
        "model_run_id": manifest["model_run_id"],
        "checkpoint_identity": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "checkpoint_training_progress": dict(checkpoint_training_progress),
        "checkpoint_reload": {
            "verified": True,
            "state_key_count": raw["checkpoint_state_key_count"],
            "checkpoint_meta_sha256": raw["checkpoint_meta_sha256"],
            "native_config_sha256": raw["native_config_sha256"],
        },
        "support_bindings": _agibot_bound_receipt_fields(manifest),
        "consumed_sample_window_ids": list(sample_ids),
        "consumption_trace": {"rows": trace_rows, "sha256": trace_digest},
        "metrics": [
            {
                "name": "native_validation_loss",
                "value": mean_loss,
                "units": "loss",
                "aggregation": "mean",
                "sample_count": len(trace_rows),
                "window_ids_sha256": bindings["validation_sample_window_ids_sha256"],
            }
        ],
        "claim_boundary": (
            "Independent native WMTrainer reload and frozen-panel validation only; no model-improvement or promotion claim."
        ),
    }
    try:
        validate_document(
            "agibot_engineering_diagnostic_receipt",
            receipt,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_INVALID") from exc
    return receipt


def _validate_agibot_checkpoint_handoff(
    receipt_path: Path, manifest: Mapping[str, object], output_root: Path
) -> tuple[Path, str, Mapping[str, object]]:
    receipt_file = _receipt(receipt_path)
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_RECEIPT_INVALID")
        validate_document(
            "agibot_engineering_checkpoint_receipt",
            receipt,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_RECEIPT_INVALID") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_RECEIPT_INVALID") from exc
    training = manifest["training"]  # type: ignore[index]
    plan = training["scale_plan"]
    checkpoint_identity = receipt["checkpoint_identity"]
    training_progress = receipt["checkpoint_training_progress"]
    expected_seed = plan["replication"]["seeds"][0]
    expected_optimizer_updates = plan["updates"]["planned_steps"]
    expected_microsteps = (
        expected_optimizer_updates
        * plan["parallelism"]["gradient_accumulation"]
    )
    if (
        receipt.get("model_run_id") != manifest.get("model_run_id")
        or receipt.get("scale_plan_digest") != training.get("scale_plan_digest")
        or checkpoint_identity.get("optimizer_updates") != expected_optimizer_updates
        or checkpoint_identity.get("microsteps") != expected_microsteps
        or checkpoint_identity.get("seed") != expected_seed
        or not isinstance(training_progress, Mapping)
        or receipt.get("support_bindings") != _agibot_bound_receipt_fields(manifest)
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_RECEIPT_BINDING_MISMATCH")
    _validate_agibot_training_progress(
        training_progress,
        seed=expected_seed,
        optimizer_updates=expected_optimizer_updates,
        microsteps=expected_microsteps,
    )
    output = checked_path(output_root, code="MODEL_RUN_OUTPUT_INVALID", error=ModelRunnerError)
    checkpoint = checked_path(
        Path(str(checkpoint_identity.get("path"))),
        code="MODEL_RUN_AGIBOT_CHECKPOINT_INVALID",
        error=ModelRunnerError,
    )
    try:
        checkpoint.relative_to(output)
    except ValueError as exc:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_OUTSIDE_OUTPUT") from exc
    if not checkpoint.is_file():
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_INVALID")
    actual_digest = sha256_file(checkpoint)
    if checkpoint_identity.get("sha256") != actual_digest:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_DIGEST_MISMATCH")
    metadata_path = checked_path(
        Path(str(checkpoint_identity.get("metadata_path"))),
        code="MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_INVALID",
        error=ModelRunnerError,
    )
    if (
        metadata_path != checkpoint.parent / "meta.json"
        or not metadata_path.is_file()
        or checkpoint_identity.get("metadata_sha256") != sha256_file(metadata_path)
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_MISMATCH")
    return checkpoint, actual_digest, training_progress


def _validate_agibot_training_progress(
    progress: Mapping[str, object],
    *,
    seed: int,
    optimizer_updates: int,
    microsteps: int,
) -> None:
    """Require complete, finite, ordered training progress for the frozen plan."""

    rows = progress.get("training_loss_rows")
    if (
        progress.get("seed") != seed
        or progress.get("planned_optimizer_updates") != optimizer_updates
        or progress.get("completed_optimizer_updates") != optimizer_updates
        or progress.get("planned_microsteps") != microsteps
        or progress.get("completed_microsteps") != microsteps
        or not isinstance(rows, list)
        or len(rows) != optimizer_updates
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_TRAINING_PROGRESS_BINDING_MISMATCH")

    previous_microstep = 0
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_TRAINING_PROGRESS_INVALID")
        loss = row.get("loss")
        current_microstep = row.get("microstep")
        if (
            row.get("optimizer_update") != index
            or isinstance(current_microstep, bool)
            or not isinstance(current_microstep, int)
            or current_microstep <= previous_microstep
            or current_microstep > microsteps
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_TRAINING_PROGRESS_INVALID")
        previous_microstep = current_microstep
    if rows[-1].get("microstep") != microsteps:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_TRAINING_PROGRESS_BINDING_MISMATCH")


def _validate_agibot_diagnostic_handoff(
    receipt_path: Path,
    manifest: Mapping[str, object],
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_training_progress: Mapping[str, object],
) -> None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_INVALID")
        validate_document(
            "agibot_engineering_diagnostic_receipt",
            receipt,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_INVALID") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_INVALID") from exc
    training = manifest["training"]  # type: ignore[index]
    bindings = _agibot_receipt_bindings(manifest)
    expected_sample_count = len(bindings.get("validation_sample_window_ids", []))
    expected_sample_ids = bindings.get("validation_sample_window_ids")
    trace = receipt.get("consumption_trace")
    trace_rows = trace.get("rows") if isinstance(trace, Mapping) else None
    trace_digest = trace.get("sha256") if isinstance(trace, Mapping) else None
    metric_rows = receipt.get("metrics", [])
    native_loss_rows = [
        metric
        for metric in metric_rows
        if isinstance(metric, Mapping) and metric.get("name") == "native_validation_loss"
    ] if isinstance(metric_rows, list) else []
    trace_valid = (
        isinstance(trace_rows, list)
        and len(trace_rows) == expected_sample_count
        and [row.get("window_id") for row in trace_rows if isinstance(row, Mapping)]
        == expected_sample_ids
    )
    if trace_valid:
        trace_losses = [row.get("loss") for row in trace_rows if isinstance(row, Mapping)]
        trace_valid = all(
            not isinstance(loss, bool)
            and isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            for loss in trace_losses
        )
    else:
        trace_losses = []
    try:
        actual_trace_digest = sha256_bytes(canonical_bytes(trace_rows)) if trace_valid else None
    except (TypeError, ValueError):
        actual_trace_digest = None
    expected_native_loss = (
        sum(float(loss) for loss in trace_losses) / expected_sample_count
        if trace_valid and expected_sample_count
        else None
    )
    native_loss_valid = False
    if len(native_loss_rows) == 1 and expected_native_loss is not None:
        native_loss_value = native_loss_rows[0].get("value")
        native_loss_valid = (
            not isinstance(native_loss_value, bool)
            and isinstance(native_loss_value, (int, float))
            and math.isfinite(float(native_loss_value))
            and math.isclose(
                float(native_loss_value),
                expected_native_loss,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
    if (
        receipt.get("model_run_id") != manifest.get("model_run_id")
        or receipt.get("stage") != "engineering_diagnostic"
        or receipt.get("evidence_class") != "runtime_only"
        or receipt.get("quality_eligible") is not False
        or receipt.get("checkpoint_training_progress") != checkpoint_training_progress
        or receipt.get("checkpoint_identity")
        != {"path": str(checkpoint), "sha256": checkpoint_sha256}
        or receipt.get("support_bindings") != _agibot_bound_receipt_fields(manifest)
        or receipt.get("consumed_sample_window_ids") != expected_sample_ids
        or not trace_valid
        or trace_digest != actual_trace_digest
        or not native_loss_valid
        or any(
            metric.get("sample_count") != expected_sample_count
            or metric.get("window_ids_sha256")
            != bindings.get("validation_sample_window_ids_sha256")
            for metric in metric_rows if isinstance(metric, Mapping)
        )
    ):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_DIAGNOSTIC_RECEIPT_BINDING_MISMATCH")
