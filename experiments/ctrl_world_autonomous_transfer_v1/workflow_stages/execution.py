"""Execution stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from experiments.ctrl_world_hybrid_memory_transfer_v1 import run as hybrid_campaign
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.control.acwm_materialized_campaign import bind_training_to_batch, validate_materialized_candidate_batch
from wmloop.experiments.materialized_transfer_evidence import project_materialized_transfer_evidence
from wmloop.experiments.training_resource_planner import method_class_from_candidate, plan_training_resources, TrainingResourcePlanningError
from wmloop.verify.acwm_materialized_frozen_verifier import run_materialized_frozen_verifier
from .planning import _load_bound_resource_receipt
from .materialization import _materializer_registration
from .common import AutonomousTransferWorkflowError, StageResult, _attempt_number, _canonical_bytes, _load, _paths, _require_directory, _require_file, _utc_now, _write_json_idempotent


def run_gpu_stage(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    stage: str,
    attempt_root: Path,
) -> StageResult:
    if stage not in {"screen", "confirm"}:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GPU_STAGE_INVALID")
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    resource_receipt = None
    if isinstance(config.get("resource_portfolio"), Mapping):
        resource_receipt = _load_bound_resource_receipt(
            context,
            phase=(
                "screen_admission"
                if stage == "screen"
                else "confirm_reallocation"
            ),
            project_root=_paths(config)["project_root"],
        )
        allocation = resource_receipt["allocation"]
        if allocation.get("stage") != stage:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_RESOURCE_PORTFOLIO_STAGE_MISMATCH"
            )
        requested_gpu_count = int(allocation["requested_gpu_count"])
        if requested_gpu_count < 1 or requested_gpu_count > len(allocation["allowed_gpu_indices"]):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_RESOURCE_GPU_COUNT_INVALID")
    catalog = _require_file(
        Path(str(context.get("candidate_catalog_path") or "")),
        "AUTONOMOUS_CANDIDATE_CATALOG_INVALID",
    )
    assessment = _require_file(
        Path(str(work["assessment_path"])), "AUTONOMOUS_ASSESSMENT_INVALID"
    )
    idea = _load(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID")
    paths = _paths(config)
    contract = _load(paths["contract"], "AUTONOMOUS_CONTRACT_INVALID")
    registration = _materializer_registration(config, work=work)
    evaluator = Path(str(registration.get("evaluator") or paths["evaluator"])).expanduser().resolve()
    _require_file(evaluator, "AUTONOMOUS_EVALUATOR_INVALID")
    stage_rows = [
        row
        for row in contract.get("stages", [])
        if isinstance(row, Mapping) and row.get("stage") == stage
    ]
    if len(stage_rows) != 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CONTRACT_STAGE_INVALID")
    campaign_root = attempt_root.parent.parent
    candidate_id = str(context["candidate_id"])
    batch_path = campaign_root / f"{stage}-batch.json"
    compilation_path = campaign_root / f"{stage}-compilation-report.json"
    compile_args = argparse.Namespace(
        catalog=catalog,
        assessment=assessment,
        contract=paths["contract"],
        stage=stage,
        batch_id=f"{candidate_id}-{stage}-v1",
        objective=str(idea["objective"]),
        hypothesis=str(idea["hypothesis"]),
        falsification_criterion=str(idea["falsification_criterion"]),
        selection_reason=(
            "Source classification, target capability checks, isolated materialization, "
            "and immutable candidate compilation all passed without compromise."
            if stage == "screen"
            else "The exact receipt-bound candidate passed screen without parameter or implementation changes."
        ),
        expected_gpu_hours=float(stage_rows[0]["max_gpu_hours_per_candidate"]),
        output=batch_path,
        compilation_report=compilation_path,
    )
    hybrid_campaign.compile_batch(compile_args)
    resource_plan = _training_resource_plan_for_batch(
        batch_path, allowed_gpu_indices=(
            [int(value) for value in resource_receipt["allocation"]["allowed_gpu_indices"]]
            if resource_receipt is not None
            else [int(value) for value in config["gpu_indices"]]
        ),
    )

    lock_root = Path(str(config["gpu_lock_root"])).expanduser().resolve()
    lease_manager = GpuLeaseManager(lock_root=lock_root)
    allowed = (
        [int(value) for value in resource_receipt["allocation"]["allowed_gpu_indices"]]
        if resource_receipt is not None
        else [int(value) for value in config["gpu_indices"]]
    )
    requested_gpu_count = int(
        resource_receipt["allocation"]["requested_gpu_count"]
        if resource_receipt is not None
        else 1
    )
    if requested_gpu_count > 1:
        acquire_many = getattr(lease_manager, "acquire_many", None)
        if not callable(acquire_many):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_DISTRIBUTED_EXECUTOR_NOT_BOUND")
        leases = acquire_many(
            allowed,
            requested_gpu_count,
            wait_seconds=float(config["gpu_wait_seconds"]),
            poll_seconds=min(5.0, max(1.0, float(config["poll_seconds"]))),
        )
    else:
        leases = [
            lease_manager.acquire(
                allowed,
                wait_seconds=float(config["gpu_wait_seconds"]),
                poll_seconds=min(5.0, max(1.0, float(config["poll_seconds"]))),
            )
        ]
    admission_path = (
        campaign_root
        / "gpu-admissions"
        / f"{stage}-attempt-{_attempt_number(attempt_root):03d}.json"
    )
    admission = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-gpu-admission",
        "state": "admitted",
        "work_id": work["work_id"],
        "stage": stage,
        "candidate_id": candidate_id,
        "config_digest": config["config_digest"],
        "allowed_gpu_indices": allowed,
        "global_gpu_limit": int(config["max_parallel_gpu_jobs"]),
        "resource_portfolio_id": (
            resource_receipt["receipt_id"] if resource_receipt is not None else None
        ),
        "resource_portfolio_digest": (
            resource_receipt["receipt_digest"]
            if resource_receipt is not None
            else None
        ),
        "resource_trial_id": (
            resource_receipt["allocation"]["selected_trial_id"]
            if resource_receipt is not None
            else None
        ),
        "lease": {
            "world_size": len(leases),
            "members": [lease.to_document() for lease in leases],
        },
        "training_resource_plan": resource_plan,
        "admitted_at": _utc_now(),
        "claim_boundary": (
            "This receipt grants one physical GPU lease for one frozen stage only; "
            "it grants no scientific or promotion authority."
        ),
    }
    _write_json_idempotent(admission_path, admission)
    baseline = paths["screen_baseline"] if stage == "screen" else paths["confirm_baseline"]
    training_payload: dict[str, object] = {}
    campaign_error: Exception | None = None
    lease_release_error: Exception | None = None
    lease_released = False
    summary: dict[str, object] | None = None
    try:
        training_payload = _ensure_masked_adapter_training(
            config,
            registration=registration,
            batch_path=batch_path,
            campaign_root=campaign_root,
            stage=stage,
            attempt_root=attempt_root,
            contract=contract,
            gpu_indices=[lease.index for lease in leases],
        )
        run_args = argparse.Namespace(
            contract=paths["contract"],
            batch=batch_path,
            baseline=baseline,
            evaluator=evaluator,
            base_evaluator=paths["base_evaluator"],
            runtime_python=paths["runtime_python"],
            ctrl_world_root=paths["ctrl_world_root"],
            dataset_root=paths["dataset_root"],
            data_stat=paths["data_stat"],
            checkpoint=paths["checkpoint"],
            svd_model=paths["svd_model"],
            clip_model=paths["clip_model"],
            output_root=attempt_root,
            gpu_indices=",".join(str(lease.index) for lease in leases),
            activity_probe_seconds=int(config["activity_probe_seconds"]),
            worker_timeout_seconds=float(config["worker_timeout_seconds"]),
            resume=attempt_root.exists(),
            dry_run=False,
        )
        summary = hybrid_campaign.run_campaign(run_args)
    except Exception as exc:
        campaign_error = exc
    finally:
        lease_document = {
            "world_size": len(leases),
            "members": [lease.to_document() for lease in leases],
        }
        try:
            for lease in leases:
                lease.release()
            lease_released = True
        except Exception as exc:
            lease_release_error = exc
        _write_json_idempotent(
            admission_path.with_name(admission_path.stem + "-release.json"),
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-gpu-release",
                "state": "released" if lease_released else "release_failed",
                "work_id": work["work_id"],
                "stage": stage,
                "lease": lease_document,
                "released_at": _utc_now(),
                "error": (
                    f"{type(lease_release_error).__name__}:"
                    f"{str(lease_release_error)[:500]}"
                    if lease_release_error is not None
                    else None
                ),
            },
        )
    runtime_receipt_path = _write_gpu_stage_execution_receipt(
        attempt_root=attempt_root,
        work=work,
        stage=stage,
        candidate_id=candidate_id,
        lease=lease_document,
        admission_path=admission_path,
        resource_receipt=resource_receipt,
        summary=summary,
        error=campaign_error,
        lease_released=lease_released,
        lease_release_error=lease_release_error,
    )
    if campaign_error is not None:
        raise campaign_error
    if lease_release_error is not None:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_GPU_LEASE_RELEASE_FAILED"
        ) from lease_release_error
    assert summary is not None
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CAMPAIGN_SUMMARY_INVALID")
    row = candidates[0]
    accepted = bool(row.get("accepted"))
    payload = {
        f"{stage}_root": str(attempt_root),
        f"{stage}_batch_path": str(batch_path),
        f"{stage}_accepted": accepted,
        f"{stage}_settlement_state": row.get("state"),
        f"{stage}_gpu_admission_path": str(admission_path),
        f"{stage}_gpu_execution_receipt_path": str(runtime_receipt_path),
        f"{stage}_gpu_execution_receipt_sha256": hashlib.sha256(
            runtime_receipt_path.read_bytes()
        ).hexdigest(),
        **training_payload,
        "training_resource_plan": resource_plan,
    }
    return StageResult(
        state="completed",
        outcome=f"{stage}_{'accepted' if accepted else 'rejected'}",
        payload=payload,
        receipt_path=attempt_root / "campaign-summary.json",
    )


def _write_gpu_stage_execution_receipt(
    *,
    attempt_root: Path,
    work: Mapping[str, object],
    stage: str,
    candidate_id: str,
    lease: Mapping[str, object],
    admission_path: Path,
    resource_receipt: Mapping[str, object] | None,
    summary: Mapping[str, object] | None,
    error: Exception | None,
    lease_released: bool,
    lease_release_error: Exception | None,
) -> Path:
    candidate_root = attempt_root / "candidates" / candidate_id
    worker_path = candidate_root / "worker-receipt.json"
    settlement_path = candidate_root / "settlement.json"
    worker = (
        _load(worker_path, "AUTONOMOUS_GPU_WORKER_RECEIPT_INVALID")
        if worker_path.is_file()
        else {}
    )
    artifacts = []
    for path in (
        worker_path,
        settlement_path,
        attempt_root / "campaign-summary.json",
    ):
        if path.is_file():
            artifacts.append(
                {
                    "artifact": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-gpu-stage-execution",
        "state": "failed" if error is not None else "settled",
        "work_id": work["work_id"],
        "candidate_id": candidate_id,
        "stage": stage,
        "resource_portfolio_id": (
            resource_receipt.get("receipt_id") if resource_receipt else None
        ),
        "resource_portfolio_digest": (
            resource_receipt.get("receipt_digest") if resource_receipt else None
        ),
        "resource_trial_id": (
            resource_receipt["allocation"]["selected_trial_id"]
            if resource_receipt
            else None
        ),
        "physical_gpus": lease,
        "physical_gpu": (
            lease["members"][0]
            if int(lease.get("world_size", 1)) == 1
            and isinstance(lease.get("members"), list)
            and lease.get("members")
            else None
        ),
        "activity_observation": worker.get("gpu_observation"),
        "exit_status": {
            "worker_state": worker.get("state"),
            "exit_code": worker.get("exit_code"),
            "timed_out": worker.get("timed_out"),
            "error": (
                f"{type(error).__name__}:{str(error)[:500]}"
                if error is not None
                else None
            ),
        },
        "artifacts": artifacts,
        "summary_state": summary.get("state") if summary else None,
        "cleanup": {
            "gpu_lease_released": lease_released,
            "release_receipt": admission_path.with_name(
                admission_path.stem + "-release.json"
            ).name,
            "release_error": (
                f"{type(lease_release_error).__name__}:"
                f"{str(lease_release_error)[:500]}"
                if lease_release_error is not None
                else None
            ),
            "scratch_state": "retained_pending_content_addressed_archive",
            "cleanup_policy": "after_content_addressed_receipt",
        },
        "recorded_at": _utc_now(),
        "claim_boundary": (
            "This receipt records physical GPU identity, observed activity, exit status, "
            "artifacts, and lease cleanup. Scientific authority remains with frozen verification."
        ),
    }
    path = attempt_root / "gpu-stage-execution.json"
    _write_json_idempotent(path, receipt)
    return path


def _training_resource_plan_for_batch(
    batch_path: Path, *, allowed_gpu_indices: Sequence[int]
) -> dict[str, object]:
    """Record the planner's method/scale decision alongside every GPU stage."""
    batch = _load(batch_path, "AUTONOMOUS_BATCH_INVALID")
    candidates = batch.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    if not isinstance(candidate, Mapping):
        return {"state": "blocked", "reason": "candidate_metadata_missing"}
    method_class = method_class_from_candidate(candidate)
    parameters = candidate.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    hidden = int(parameters.get("hidden_dim", 0) or 0)
    action = int(parameters.get("action_dim", 0) or 0)
    trainable = int(candidate.get("estimated_trainable_parameters", 0) or 0)
    if trainable < 1 and method_class == "adapter_training":
        # Conservative estimate for the generated mask and residual projections.
        trainable = max(1, hidden * max(action, 1) * 8)
    try:
        return plan_training_resources(
            method_class=method_class,
            trainable_parameters=trainable,
            train_examples=int(candidate.get("training_examples", 1) or 1),
            sequence_length=int(candidate.get("sequence_length", 32) or 32),
            batch_size=int(candidate.get("batch_size", 2) or 2),
            planned_steps=int(candidate.get("training_steps", 100) or 100),
            available_gpus=allowed_gpu_indices,
            competing_candidates=1,
        )
    except (TypeError, ValueError, TrainingResourcePlanningError) as exc:
        return {
            "state": "metadata_required",
            "method_class": method_class,
            "reason": f"{type(exc).__name__}:{str(exc)}",
            "requested_gpu_count": 1,
        }


def _ensure_masked_adapter_training(
    config: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    batch_path: Path,
    campaign_root: Path,
    stage: str,
    attempt_root: Path,
    contract: Mapping[str, object],
    gpu_indices: Sequence[int] | None = None,
    gpu_index: int | None = None,
) -> dict[str, object]:
    """Fit and bind a masked adapter before any screen/confirm evaluator runs."""

    if gpu_indices is None:
        if gpu_index is None:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_TRAINING_GPU_BINDING_INVALID")
        gpu_indices = [gpu_index]
    gpu_indices = [int(index) for index in gpu_indices]
    if not gpu_indices or len(set(gpu_indices)) != len(gpu_indices):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_TRAINING_GPU_BINDING_INVALID")
    batch = _load(batch_path, "AUTONOMOUS_BATCH_INVALID")
    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_CANDIDATE_INVALID")
    candidate = candidates[0]
    if candidate.get("candidate_kind") != "materialized_masked_intermediate_action_adapter":
        return {}
    paths = _paths(config)
    # A bound batch is already durable. Re-validate it and never retrain on resume.
    provenance = candidate.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("training_binding") is not None:
        validate_materialized_candidate_batch(batch, contract, root=paths["project_root"])
        binding = provenance["training_binding"]
        assert isinstance(binding, Mapping)
        return {
            "training_receipt_path": binding.get("training_receipt_path"),
            "adapter_state_path": binding.get("adapter_state_path"),
            "training_reused": True,
        }
    trainer_value = registration.get("trainer")
    if not isinstance(trainer_value, str) or not trainer_value:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINER_MISSING")
    trainer = _require_file(Path(trainer_value), "AUTONOMOUS_MASKED_ADAPTER_TRAINER_INVALID")
    training_config = registration.get("training", {})
    if not isinstance(training_config, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_CONFIG_INVALID")
    candidate_id = str(candidate["candidate_id"])
    training_root = (
        campaign_root.parent
        / "adapter-training"
        / candidate_id
        / "training-v1"
    ).resolve()
    receipt_path = training_root / "training-receipt.json"
    if not receipt_path.is_file():
        command = [
            str(paths["runtime_python"]),
            str(trainer),
            "--candidate", str(batch_path.resolve()),
            "--ctrl-world-root", str(paths["ctrl_world_root"]),
            "--dataset-root", str(paths["dataset_root"]),
            "--data-stat", str(paths["data_stat"]),
            "--checkpoint", str(paths["checkpoint"]),
            "--svd-model", str(paths["svd_model"]),
            "--clip-model", str(paths["clip_model"]),
            "--output-root", str(training_root),
        ]
        option_map = (
            ("steps", "--steps"),
            ("batch_size", "--batch-size"),
            ("learning_rate", "--learning-rate"),
            ("max_grad_norm", "--max-grad-norm"),
            ("seed", "--seed"),
            ("num_history", "--num-history"),
            ("num_frames", "--num-frames"),
            ("split_fingerprint", "--training-split-fingerprint"),
            ("device", "--device"),
        )
        for key, flag in option_map:
            if key in training_config:
                command.extend([flag, str(training_config[key])])
        training_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path = training_root / "training.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        environment["PYTHONUNBUFFERED"] = "1"
        if len(gpu_indices) > 1:
            # torchrun owns rank/world-size setup; the trainer writes one receipt
            # from rank zero after all ranks finish.
            command = [
                str(paths["runtime_python"]), "-m", "torch.distributed.run",
                "--standalone", "--nproc_per_node", str(len(gpu_indices)),
                *command[1:],
            ]
        try:
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=str(paths["project_root"]),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(config["worker_timeout_seconds"]),
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_TIMEOUT") from exc
        if completed.returncode != 0:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MASKED_ADAPTER_TRAINING_FAILED:{completed.returncode}"
            )
    if not receipt_path.is_file():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_RECEIPT_MISSING")
    updated = bind_training_to_batch(
        batch,
        receipt_path=receipt_path,
        contract=contract,
        root=paths["project_root"],
    )
    _replace_compiled_batch(batch_path, expected_digest=str(batch["batch_digest"]), updated=updated)
    binding = updated["candidates"][0]["provenance"]["training_binding"]
    assert isinstance(binding, Mapping)
    return {
        "training_receipt_path": binding["training_receipt_path"],
        "adapter_state_path": binding["adapter_state_path"],
        "training_reused": False,
    }


def _replace_compiled_batch(
    path: Path, *, expected_digest: str, updated: Mapping[str, object]
) -> None:
    """Allow exactly one controlled transition from pending to trained batch."""

    current = _load(path, "AUTONOMOUS_BATCH_INVALID")
    if current.get("batch_digest") != expected_digest:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_BINDING_RACE")
    raw = _canonical_bytes(updated)
    temporary = path.with_name(f".{path.name}.training-{os.getpid()}.tmp")
    if path.is_symlink() or not path.is_file():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_INVALID")
    with temporary.open("xb") as handle:
        handle.write(raw)
    os.replace(temporary, path)


def verify(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    screen_root = _require_directory(
        Path(str(context.get("screen_root") or "")), "AUTONOMOUS_SCREEN_ROOT_INVALID"
    )
    confirm_value = context.get("confirm_root")
    confirm_root = (
        _require_directory(Path(str(confirm_value)), "AUTONOMOUS_CONFIRM_ROOT_INVALID")
        if confirm_value
        else None
    )
    paths = _paths(config)
    manifest = run_materialized_frozen_verifier(
        policy_path=paths["verifier_policy"],
        contract_path=paths["contract"],
        screen_root=screen_root,
        confirm_root=confirm_root,
        output_root=attempt_root,
        project_root=paths["project_root"],
    )
    evidence_path = attempt_root / "verified-evidence.jsonl"
    projection = project_materialized_transfer_evidence(
        verifier_root=attempt_root,
        output_path=evidence_path,
        project_root=paths["project_root"],
    )
    return StageResult(
        state="completed",
        outcome=str(manifest["decision"]),
        payload={
            "verifier_root": str(attempt_root),
            "decision": manifest["decision"],
            "verdict_ref": manifest["verdict_ref"],
            "verifier_ref": projection["verifier_ref"],
            "verified_evidence_path": str(evidence_path),
            "evidence_projection": projection,
        },
        receipt_path=attempt_root / "verification-manifest.json",
    )
