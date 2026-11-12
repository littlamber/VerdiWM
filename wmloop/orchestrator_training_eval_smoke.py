"""M3 environment smoke with training plus frozen metric evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import load_yaml_document
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.primitive_runtime_smoke import (
    _validate_training_assets,
)
from wmloop.execute.training_smoke.bundle import (
    _bundle_action_following_files as _bundle_action_following_files,
    _bundle_candidate_checkpoint as _bundle_candidate_checkpoint,
    _bundle_candidate_runtime as _bundle_candidate_runtime,
    _bundle_checkpoint_ladder as _bundle_checkpoint_ladder,
    _bundle_evaluation_files as _bundle_evaluation_files,
    _file_sha256 as _file_sha256,
    _link_or_copy_checkpoint as _link_or_copy_checkpoint,
    _render_markdown as _render_markdown,
    _write_report_bundle as _write_report_bundle,
)
from wmloop.execute.training_smoke.errors import (
    OrchestratorTrainingEvalSmokeError as OrchestratorTrainingEvalSmokeError,
)
from wmloop.execute.training_smoke.evidence import (
    _action_following_observed as _action_following_observed,
    _action_following_records_with_refs as _action_following_records_with_refs,
    _action_rows as _action_rows,
    _evaluation_summary as _evaluation_summary,
    _load_action_following_measurement as _load_action_following_measurement,
    _load_action_following_probe_pair as _load_action_following_probe_pair,
    _load_probe_replications as _load_probe_replications,
    _replication_records_with_refs as _replication_records_with_refs,
    _verification_evidence_from_probe_manifests as _verification_evidence_from_probe_manifests,
    _verification_evidence_from_probe_replications as _verification_evidence_from_probe_replications,
    _write_accept_cohort_view_inputs as _write_accept_cohort_view_inputs,
    _write_accept_replicate_inputs as _write_accept_replicate_inputs,
)
from wmloop.execute.training_smoke.executor import (
    _in_trial_gpu_exclusivity_max_age_seconds as _in_trial_gpu_exclusivity_max_age_seconds,
    _run_required_check as _run_required_check,
    _TrainingEvalSmokeExecutor as _TrainingEvalSmokeExecutor,
)
from wmloop.execute.training_smoke.io import (
    _canonical_json_bytes as _canonical_json_bytes,
    _load_json_object as _load_json_object,
    _write_bytes_atomic as _write_bytes_atomic,
)
from wmloop.execute.training_smoke.metrics import (
    _auc_metric_name as _auc_metric_name,
    _auc_value as _auc_value,
    _environment_spec as _environment_spec,
    _finite_float as _finite_float,
    _goal_action_following_threshold as _goal_action_following_threshold,
    _goal_id as _goal_id,
    _goal_metric_name as _goal_metric_name,
    _has_required_horizons as _has_required_horizons,
    _max_observed_horizon as _max_observed_horizon,
    _positive_int as _positive_int,
    _receipt_gpu_hours as _receipt_gpu_hours,
    _replication_delta as _replication_delta,
    _replication_value as _replication_value,
    _vendor_environment_name as _vendor_environment_name,
    _verify_goal_environment as _verify_goal_environment,
)
from wmloop.execute.training_smoke.planning import (
    _bind_primitive_training_config as _bind_primitive_training_config,
    _checkpoint_frequency as _checkpoint_frequency,
    _official_resume_training_config as _official_resume_training_config,
    _proposal_params as _proposal_params,
    _training_eval_budget_estimate_gpu_hours as _training_eval_budget_estimate_gpu_hours,
    _TrainingEvalSmokeProposalClient as _TrainingEvalSmokeProposalClient,
)
from wmloop.execute.training_smoke.routing import (
    _augment_failure_with_signature_bank_routes as _augment_failure_with_signature_bank_routes,
    _positive_transfer_routed_failure_families as _positive_transfer_routed_failure_families,
    _signature_bank_routed_failure_families as _signature_bank_routed_failure_families,
    _targeted_plan_routed_failure_families as _targeted_plan_routed_failure_families,
)
from wmloop.execute.training_smoke.runtime import (
    _candidate_checkpoint_path as _candidate_checkpoint_path,
    _checkpoint_resolution_suffix as _checkpoint_resolution_suffix,
    _checkpoint_step as _checkpoint_step,
    _copy_candidate_runtime as _copy_candidate_runtime,
    _copy_training_sidecar as _copy_training_sidecar,
    _copy_training_sidecars as _copy_training_sidecars,
    _horizon_probe_command as _horizon_probe_command,
    _load_yaml_with_pyyaml as _load_yaml_with_pyyaml,
    _raw_probe_measure_command as _raw_probe_measure_command,
    _safe_run_name as _safe_run_name,
    _training_entrypoint_command as _training_entrypoint_command,
    _validate_probe_runtime_binding as _validate_probe_runtime_binding,
)
from wmloop.execute.training_smoke.settlement import (
    _digest_from_cas_ref as _digest_from_cas_ref,
    _evaluator_hash as _evaluator_hash,
    _formal_trial_cell as _formal_trial_cell,
    _impl_diff_hash as _impl_diff_hash,
    _is_sha256 as _is_sha256,
    _primary_delta as _primary_delta,
    _record_formal_settled_trial as _record_formal_settled_trial,
)
from wmloop.freeze import (
    dataset_freeze_sha256,
    verify_acwm_dataset_freeze,
    verify_acwm_heldout_protocol,
)
from wmloop.orchestrator import ResearchLoop
from wmloop.orchestrator_training_smoke import _put_json
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.generator import ProposalContext, ProposalGenerator
from wmloop.propose.llm_logging import seal_llm_call_log
from wmloop.vendor import verify_vendor_checkout
from wmloop.verify.m4_launch_guard import phase_gate_guard
from wmloop.verify.round_start_guard import round_start_guard


def run_push_cube_training_eval_smoke(**kwargs: Any) -> dict[str, object]:
    """Compatibility wrapper for the original Push Cube M3 smoke entrypoint."""

    return run_environment_training_eval_smoke(environment="push_cube", **kwargs)


def run_environment_training_eval_smoke(
    *,
    environment: str,
    repo_root: Path,
    failure_report: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze_path: Path,
    heldout_protocol_path: Path,
    gpu_index: int = 1,
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
    train_steps: int = 1,
    train_batch_size: int = 1,
    train_val_batch_size: int | None = None,
    train_size: int = 1,
    train_num_workers: int = 0,
    eval_horizons: Sequence[int] = (16, 32, 48, 64),
    max_accept_trajectories: int = 1,
    replication_count: int = 1,
    eval_inference_steps: int = 1,
    hook_timeout_seconds: float = 60.0,
    training_timeout_seconds: float = 1800.0,
    eval_timeout_seconds: float = 900.0,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    keep_temp_on_failure: bool = False,
    m4_phase_gate_manifest: Path | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    publish_settled_trial: bool = False,
) -> dict[str, object]:
    """Run one proposal through training, held-out metric collection, and judge."""

    root = Path(repo_root).resolve()
    env_spec = _environment_spec(environment)
    environment_name = env_spec.environment
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_OUTPUT_EXISTS")
    if publish_settled_trial and archive_db is None:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_ARCHIVE_REQUIRED"
        )
    if publish_settled_trial and m4_phase_gate_manifest is None:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_M4_GATE_REQUIRED"
        )
    m4_guard = (
        phase_gate_guard(m4_phase_gate_manifest)
        if m4_phase_gate_manifest is not None
        else None
    )
    m4_authorization = m4_guard() if m4_guard is not None else None
    runtime = Path(runtime_python).expanduser().absolute()
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    freeze_path = Path(dataset_freeze_path).resolve()
    protocol_path = Path(heldout_protocol_path).resolve()
    if (
        gpu_index < 0
        or train_steps < 1
        or train_batch_size < 1
        or (train_val_batch_size is not None and train_val_batch_size < 1)
        or train_size < 1
        or train_num_workers < 0
        or max_accept_trajectories < 1
        or replication_count < 1
        or eval_inference_steps < 1
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    if trial_arm is not None and trial_arm not in {
        "prior",
        "cold_start",
        "shuffled_prior",
    }:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_TRIAL_ARM_INVALID"
        )
    if trial_seed is not None and trial_seed < 1:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_TRIAL_SEED_INVALID"
        )
    gpu_exclusivity = verify_gpu_exclusivity_ready(
        gpu_exclusivity_audit_manifest,
        gpu_index=gpu_index,
        max_age_seconds=gpu_exclusivity_max_age_seconds,
    )
    if gpu_exclusivity_audit_manifest is None:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_GPU_EXCLUSIVITY_AUDIT_REQUIRED"
        )
    gpu_exclusivity_manifest = Path(gpu_exclusivity_audit_manifest).resolve(strict=True)
    _validate_training_assets(
        runtime=runtime, data_root=data, checkpoint_root=checkpoints
    )
    horizons = tuple(int(value) for value in eval_horizons)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(value < 2 for value in horizons)
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    failure = _augment_failure_with_signature_bank_routes(
        root=root,
        failure=_load_json_object(failure_report),
        requested_primitive=proposal_primitive,
        proposal_routing_plan=proposal_routing_plan,
    )
    goal = load_yaml_document(goal_config)
    goal_id = _goal_id(goal)
    _verify_goal_environment(goal, environment_name)
    if failure.get("env") != environment_name or failure.get("goal_id") != goal_id:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FAILURE_SCOPE_INVALID"
        )
    primitive_params = _proposal_params(
        primitive=proposal_primitive,
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
    )
    primary_metric = _goal_metric_name(goal=goal, horizons=horizons)
    dataset_freeze = _load_json_object(freeze_path)
    heldout_protocol = _load_json_object(protocol_path)
    verify_acwm_dataset_freeze(
        data, dataset_freeze, required_splits=("ind_test", "ood_test")
    )
    verify_acwm_heldout_protocol(dataset_freeze, heldout_protocol)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    failed = False
    try:
        temporary.mkdir(mode=0o700, parents=True)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_storage_root = (
            cas_root
            if cas_root is not None
            else (
                Path(archive_db).resolve().parent
                if archive_db is not None
                else destination.parent
            )
        )
        cas = ContentAddressedStore(cas_storage_root)
        executor = _TrainingEvalSmokeExecutor(
            repo_root=root,
            environment=environment_name,
            runs_root=temporary / "sandbox-runs",
            runtime_python=runtime,
            data_root=data,
            checkpoint_root=checkpoints,
            dataset_freeze=dataset_freeze,
            heldout_protocol=heldout_protocol,
            gpu_index=gpu_index,
            gpu_exclusivity_audit_manifest=gpu_exclusivity_manifest,
            gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
            weight=weight,
            intervention_lr={
                "next_forcing": next_forcing_lr,
                "self_forcing_finetune": self_forcing_lr,
                "wmsd_self_distill": wmsd_lr,
            }.get(proposal_primitive),
            train_steps=train_steps,
            train_batch_size=train_batch_size,
            train_val_batch_size=train_val_batch_size,
            train_size=train_size,
            train_num_workers=train_num_workers,
            eval_horizons=horizons,
            max_accept_trajectories=max_accept_trajectories,
            replication_count=replication_count,
            eval_inference_steps=eval_inference_steps,
            action_following_threshold=_goal_action_following_threshold(goal),
            primary_metric=primary_metric,
            output_root=temporary,
            cas=cas,
            archive=archive,
            hook_timeout_seconds=hook_timeout_seconds,
            training_timeout_seconds=training_timeout_seconds,
            eval_timeout_seconds=eval_timeout_seconds,
            proposal_primitive=proposal_primitive,
            proposal_params=primitive_params,
            trial_arm=trial_arm,
            trial_seed=trial_seed,
        )
        budget = BudgetLedger(
            temporary / "budget.db",
            BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])),
        )
        loop = ResearchLoop(
            proposal_generator=ProposalGenerator(
                _TrainingEvalSmokeProposalClient(
                    environment=environment_name,
                    primitive=proposal_primitive,
                    params=primitive_params,
                    metric_name=primary_metric,
                    budget_estimate_gpu_hours=_training_eval_budget_estimate_gpu_hours(
                        registry=registry,
                        primitive=proposal_primitive,
                        train_steps=train_steps,
                        replication_count=replication_count,
                        horizon_count=len(horizons),
                        action_following_probe_count=2,
                    ),
                    trial_arm=trial_arm,
                    trial_seed=trial_seed,
                )
            ),
            budget_ledger=budget,
            executor=executor,
            vendor_verifier=lambda: verify_vendor_checkout(root),
            m4_launch_guard=m4_guard,
            round_start_guard=invariant_guard,
        )
        context = ProposalContext(
            failure_report={**failure, "round": 0},
            goal_spec=goal,
            archive_statistics=archive.archive_statistics()
            if archive is not None
            else {},
            registry=registry,
        )
        result = loop.run_round(context)
        proposal = dict(result.generated_proposal.proposal)
        verdict = result.verdict.to_dict()
        proposal_ref = _put_json(cas, proposal, archive=archive)
        llm_call_log = seal_llm_call_log(
            result.generated_proposal, cas=cas, archive=archive
        )
        failure_context_ref = _put_json(cas, failure, archive=archive)
        verdict_ref = _put_json(cas, verdict, archive=archive)
        execution_receipt = executor.receipt(proposal_id=str(proposal["proposal_id"]))
        formal_settled_trial_published = False
        if publish_settled_trial:
            if archive is None:
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_FORMAL_ARCHIVE_REQUIRED"
                )
            formal_settled_trial_published = _record_formal_settled_trial(
                archive=archive,
                proposal=proposal,
                verdict=verdict,
                receipt=execution_receipt,
                receipt_ref=result.execution.receipt_ref,
                failure_context_ref=failure_context_ref,
                verdict_ref=verdict_ref,
                gpu_hours=result.execution.actual_gpu_hours,
                settlement_state=result.settlement.state,
                round_start_verification=result.round_start_verification,
            )
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-smoke-report",
            "state": "ready"
            if execution_receipt["state"] == "ready"
            else "checks_failed",
            "environment": environment_name,
            "goal_id": goal_id,
            "primary_metric": primary_metric,
            "proposal_primitive": proposal_primitive,
            "proposal_params": primitive_params,
            "proposal_routing": {
                "dominant_failure": failure.get("dominant_failure"),
                "routed_failure_families": failure.get("routed_failure_families", []),
                "provenance": failure.get("proposal_routing_provenance", []),
            },
            "proposal_id": proposal["proposal_id"],
            **({"trial_arm": trial_arm} if trial_arm is not None else {}),
            **({"seed": trial_seed} if trial_seed is not None else {}),
            "failure_context_ref": failure_context_ref,
            "proposal_ref": proposal_ref,
            "prompt_ref": llm_call_log["prompt_ref"],
            "llm_call_log": llm_call_log,
            "receipt_ref": result.execution.receipt_ref,
            "verdict_ref": verdict_ref,
            "verdict": verdict["verdict"],
            "violation": verdict["violation"],
            "gates": verdict["gates"],
            "action_following_gate": verdict["action_following_gate"],
            "delta_m_ver": verdict["delta_m_ver"],
            "settlement_state": result.settlement.state,
            "actual_gpu_hours": result.execution.actual_gpu_hours,
            "round_start_verification": result.round_start_verification,
            "runtime_python": str(runtime),
            "data_root": str(data),
            "checkpoint_root": str(checkpoints),
            "dataset_freeze_path": str(freeze_path),
            "heldout_protocol_path": str(protocol_path),
            "dataset_freeze_sha256": dataset_freeze_sha256(dataset_freeze),
            "gpu_index": gpu_index,
            "gpu_exclusivity_audit": gpu_exclusivity,
            "train_steps": train_steps,
            "train_batch_size": train_batch_size,
            "train_val_batch_size": train_val_batch_size
            if train_val_batch_size is not None
            else train_batch_size,
            "train_size": train_size,
            "train_num_workers": train_num_workers,
            "eval_horizons": list(horizons),
            "eval_inference_steps": eval_inference_steps,
            "max_accept_trajectories": max_accept_trajectories,
            "replication_count": replication_count,
            "receipt": execution_receipt,
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "formal_settled_trial_published": formal_settled_trial_published,
            "archive_db": str(Path(archive_db).resolve())
            if archive_db is not None
            else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "m4_launch_gate": m4_authorization,
            "limitations": [
                f"This is a {train_steps}-step {environment_name} smoke for orchestration and metric-evidence wiring, not a formal model-quality claim.",
                "It evaluates one or more frozen ind_accept trajectory batches; small-count smoke evidence is not a formal model-quality claim.",
                "The evaluator runs from the frozen vendor checkout; the trial worktree patch cannot modify evaluator files.",
                "The candidate checkpoint is evaluated before sandbox cleanup and is not retained as a reusable model artifact.",
            ],
        }
        return _write_report_bundle(
            report=report,
            output_root=destination,
            cas=cas,
            archive=archive,
            budget_db=temporary / "budget.db",
        )
    except Exception:
        failed = True
        raise
    finally:
        if (temporary.exists() or temporary.is_symlink()) and not (
            failed and keep_temp_on_failure
        ):
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one training+eval environment M3 smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--environment", default="push_cube")
    run.add_argument("--failure-report", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--dataset-freeze", type=Path, required=True)
    run.add_argument("--heldout-protocol", type=Path, required=True)
    run.add_argument("--gpu-index", type=int, default=1)
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
    run.add_argument("--train-steps", type=int, default=1)
    run.add_argument("--train-batch-size", type=int, default=1)
    run.add_argument("--train-val-batch-size", type=int)
    run.add_argument("--train-size", type=int, default=1)
    run.add_argument("--train-num-workers", type=int, default=0)
    run.add_argument("--eval-horizons", type=int, nargs="+", default=[16, 32, 48, 64])
    run.add_argument("--max-accept-trajectories", type=int, default=1)
    run.add_argument("--replication-count", type=int, default=1)
    run.add_argument("--eval-inference-steps", type=int, default=1)
    run.add_argument("--hook-timeout-seconds", type=float, default=60.0)
    run.add_argument("--training-timeout-seconds", type=float, default=1800.0)
    run.add_argument("--eval-timeout-seconds", type=float, default=900.0)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--keep-temp-on-failure", action="store_true")
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    run.add_argument("--publish-settled-trial", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_environment_training_eval_smoke(
            environment=args.environment,
            repo_root=args.repo_root,
            failure_report=args.failure_report,
            goal_config=args.goal_config,
            output_root=args.output_root,
            runtime_python=args.runtime_python,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            dataset_freeze_path=args.dataset_freeze,
            heldout_protocol_path=args.heldout_protocol,
            gpu_index=args.gpu_index,
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
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
            train_val_batch_size=args.train_val_batch_size,
            train_size=args.train_size,
            train_num_workers=args.train_num_workers,
            eval_horizons=tuple(args.eval_horizons),
            max_accept_trajectories=args.max_accept_trajectories,
            replication_count=args.replication_count,
            eval_inference_steps=args.eval_inference_steps,
            hook_timeout_seconds=args.hook_timeout_seconds,
            training_timeout_seconds=args.training_timeout_seconds,
            eval_timeout_seconds=args.eval_timeout_seconds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            keep_temp_on_failure=args.keep_temp_on_failure,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            publish_settled_trial=args.publish_settled_trial,
        )
        print(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
