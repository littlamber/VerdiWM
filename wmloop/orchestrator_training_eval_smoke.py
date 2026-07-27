"""M3 environment smoke with training plus frozen metric evidence."""

from __future__ import annotations

import argparse
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

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.contracts import load_yaml_document
from wmloop.diagnose.probes.action_following import no_action_delta_psnr, per_frame_inverse_dynamics_accuracy
from wmloop.evaluate.plan import build_baseline_evaluation_plan
from wmloop.execute.agent_staging import AgentRepairSession, CommandReceipt
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.gpu_sampling import GpuSamplingRecorder
from wmloop.execute.primitive_runtime_smoke import (
    _runtime_environment,
    _validate_training_assets,
    hook_unit_script,
)
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.execute.training_monitor_policy import checkpoint_eval_ladder
from wmloop.freeze import dataset_freeze_sha256, verify_acwm_dataset_freeze, verify_acwm_heldout_protocol
from wmloop.orchestrator import ExecutionOutcome, ResearchLoop
from wmloop.orchestrator_training_smoke import _cas_attempts, _put_file, _put_json
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION, ProposalContext, ProposalGenerator
from wmloop.propose.llm_logging import seal_llm_call_log
from wmloop.propose.scheduler import InterventionCell
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.verify.judge import VerificationEvidence
from wmloop.verify.m4_launch_guard import phase_gate_guard
from wmloop.verify.round_start_guard import round_start_guard
from wmloop.vendor import verify_vendor_checkout


class OrchestratorTrainingEvalSmokeError(RuntimeError):
    """The M3 training+evaluation smoke failed closed."""


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
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_ARCHIVE_REQUIRED")
    if publish_settled_trial and m4_phase_gate_manifest is None:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_M4_GATE_REQUIRED")
    m4_guard = phase_gate_guard(m4_phase_gate_manifest) if m4_phase_gate_manifest is not None else None
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
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
    if trial_arm is not None and trial_arm not in {"prior", "cold_start", "shuffled_prior"}:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_TRIAL_ARM_INVALID")
    if trial_seed is not None and trial_seed < 1:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_TRIAL_SEED_INVALID")
    gpu_exclusivity = verify_gpu_exclusivity_ready(
        gpu_exclusivity_audit_manifest,
        gpu_index=gpu_index,
        max_age_seconds=gpu_exclusivity_max_age_seconds,
    )
    if gpu_exclusivity_audit_manifest is None:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GPU_EXCLUSIVITY_AUDIT_REQUIRED")
    gpu_exclusivity_manifest = Path(gpu_exclusivity_audit_manifest).resolve(strict=True)
    _validate_training_assets(runtime=runtime, data_root=data, checkpoint_root=checkpoints)
    horizons = tuple(int(value) for value in eval_horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(value < 2 for value in horizons):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
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
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FAILURE_SCOPE_INVALID")
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
    verify_acwm_dataset_freeze(data, dataset_freeze, required_splits=("ind_test", "ood_test"))
    verify_acwm_heldout_protocol(dataset_freeze, heldout_protocol)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    failed = False
    try:
        temporary.mkdir(mode=0o700, parents=True)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
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
        budget = BudgetLedger(temporary / "budget.db", BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])))
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
            archive_statistics=archive.archive_statistics() if archive is not None else {},
            registry=registry,
        )
        result = loop.run_round(context)
        proposal = dict(result.generated_proposal.proposal)
        verdict = result.verdict.to_dict()
        proposal_ref = _put_json(cas, proposal, archive=archive)
        llm_call_log = seal_llm_call_log(result.generated_proposal, cas=cas, archive=archive)
        failure_context_ref = _put_json(cas, failure, archive=archive)
        verdict_ref = _put_json(cas, verdict, archive=archive)
        execution_receipt = executor.receipt(proposal_id=str(proposal["proposal_id"]))
        formal_settled_trial_published = False
        if publish_settled_trial:
            if archive is None:
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_ARCHIVE_REQUIRED")
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
            "state": "ready" if execution_receipt["state"] == "ready" else "checks_failed",
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
            "train_val_batch_size": train_val_batch_size if train_val_batch_size is not None else train_batch_size,
            "train_size": train_size,
            "train_num_workers": train_num_workers,
            "eval_horizons": list(horizons),
            "eval_inference_steps": eval_inference_steps,
            "max_accept_trajectories": max_accept_trajectories,
            "replication_count": replication_count,
            "receipt": execution_receipt,
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "formal_settled_trial_published": formal_settled_trial_published,
            "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "m4_launch_gate": m4_authorization,
            "limitations": [
                f"This is a {train_steps}-step {environment_name} smoke for orchestration and metric-evidence wiring, not a formal model-quality claim.",
                "It evaluates one or more frozen ind_accept trajectory batches; small-count smoke evidence is not a formal model-quality claim.",
                "The evaluator runs from the frozen vendor checkout; the trial worktree patch cannot modify evaluator files.",
                "The candidate checkpoint is evaluated before sandbox cleanup and is not retained as a reusable model artifact.",
            ],
        }
        return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive, budget_db=temporary / "budget.db")
    except Exception:
        failed = True
        raise
    finally:
        if (temporary.exists() or temporary.is_symlink()) and not (failed and keep_temp_on_failure):
            shutil.rmtree(temporary, ignore_errors=True)


class _TrainingEvalSmokeExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        environment: str,
        runs_root: Path,
        runtime_python: Path,
        data_root: Path,
        checkpoint_root: Path,
        dataset_freeze: Mapping[str, object],
        heldout_protocol: Mapping[str, object],
        gpu_index: int,
        gpu_exclusivity_audit_manifest: Path,
        gpu_exclusivity_max_age_seconds: float | None,
        weight: float,
        intervention_lr: float | None,
        train_steps: int,
        train_batch_size: int,
        train_val_batch_size: int | None,
        train_size: int,
        train_num_workers: int,
        eval_horizons: tuple[int, ...],
        max_accept_trajectories: int,
        replication_count: int,
        eval_inference_steps: int,
        action_following_threshold: float | None,
        primary_metric: str,
        output_root: Path,
        cas: ContentAddressedStore,
        archive: ArchiveStore | None,
        hook_timeout_seconds: float,
        training_timeout_seconds: float,
        eval_timeout_seconds: float,
        proposal_primitive: str,
        proposal_params: Mapping[str, object],
        trial_arm: str | None,
        trial_seed: int | None,
    ) -> None:
        self._repo_root = repo_root
        self._environment = environment
        self._runs_root = runs_root
        self._runtime_python = runtime_python
        self._data_root = data_root
        self._checkpoint_root = checkpoint_root
        self._dataset_freeze = dataset_freeze
        self._heldout_protocol = heldout_protocol
        self._gpu_index = gpu_index
        self._gpu_exclusivity_audit_manifest = gpu_exclusivity_audit_manifest
        self._gpu_exclusivity_launch_max_age_seconds = gpu_exclusivity_max_age_seconds
        self._gpu_exclusivity_trial_max_age_seconds = _in_trial_gpu_exclusivity_max_age_seconds(
            launch_max_age_seconds=gpu_exclusivity_max_age_seconds,
            training_timeout_seconds=training_timeout_seconds,
            eval_timeout_seconds=eval_timeout_seconds,
            hook_timeout_seconds=hook_timeout_seconds,
            replication_count=replication_count,
        )
        self._weight = weight
        self._intervention_lr = intervention_lr
        self._train_steps = train_steps
        self._train_batch_size = train_batch_size
        self._train_val_batch_size = train_val_batch_size if train_val_batch_size is not None else train_batch_size
        self._train_size = train_size
        self._train_num_workers = train_num_workers
        self._eval_horizons = eval_horizons
        self._max_accept_trajectories = max_accept_trajectories
        self._replication_count = replication_count
        self._eval_inference_steps = eval_inference_steps
        self._action_following_threshold = action_following_threshold
        self._primary_metric = primary_metric
        self._output_root = output_root
        self._cas = cas
        self._archive = archive
        self._hook_timeout = hook_timeout_seconds
        self._training_timeout = training_timeout_seconds
        self._eval_timeout = eval_timeout_seconds
        self._proposal_primitive = proposal_primitive
        self._proposal_params = dict(proposal_params)
        self._trial_arm = trial_arm
        self._trial_seed = trial_seed
        self._receipts: dict[str, dict[str, object]] = {}
        self._gpu_sampler = GpuSamplingRecorder(gpu_index=gpu_index, sample_interval_seconds=2.0)

    def execute(self, proposal: dict[str, object], fencing_token: int) -> ExecutionOutcome:
        source_revision = verify_vendor_checkout(self._repo_root)
        registry = PrimitiveRegistry.from_root(self._repo_root)
        renderer = PrimitiveRenderer(registry)
        proposal_id = str(proposal["proposal_id"])
        sandbox = WorktreeSandbox(vendor_root=self._repo_root / "vendor" / "ACWM-Phys", runs_root=self._runs_root)
        lease: SandboxLease | None = None
        worktree_removed = False
        try:
            lease = sandbox.create(trial_id=proposal_id, expected_revision=source_revision)
            interventions = proposal.get("interventions")
            if not isinstance(interventions, list):
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_INTERVENTIONS_INVALID")
            rendered = renderer.render_checked(worktree=lease.worktree, interventions=interventions, hook_ios=_default_hook_ios())
            for item in rendered:
                _apply_diff(lease.worktree, item.diff)
            spec = _environment_spec(self._environment)
            official_checkpoint = self._checkpoint_root / spec.checkpoint_relative_path
            resume_step = _checkpoint_step(runtime_python=self._runtime_python, checkpoint_path=official_checkpoint)
            run_name = _safe_run_name(proposal_id)
            training_config_path = self._output_root / f"{proposal_id}-official-resume-train.yaml"
            training_config = _official_resume_training_config(
                vendor_root=self._repo_root / "vendor" / "ACWM-Phys",
                checkpoint_root=self._checkpoint_root,
                environment=self._environment,
                run_name=run_name,
                learning_rate=self._intervention_lr,
                total_steps=resume_step + self._train_steps,
                train_steps=self._train_steps,
                batch_size=self._train_batch_size,
                val_batch_size=self._train_val_batch_size,
                train_size=self._train_size,
                num_workers=self._train_num_workers,
            )
            training_config = _bind_primitive_training_config(
                training_config,
                primitive=self._proposal_primitive,
                params=self._proposal_params,
            )
            _write_bytes_atomic(training_config_path, _canonical_json_bytes(training_config))
            replicates = _write_accept_replicate_inputs(
                output_root=self._output_root,
                heldout_protocol=self._heldout_protocol,
                dataset_freeze=self._dataset_freeze,
                environment=self._environment,
                max_accept_trajectories=self._max_accept_trajectories,
                replication_count=self._replication_count,
            )
            training_run_root = self._output_root / "training-run"
            training_run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            training_sidecars = _copy_training_sidecars(
                source_root=lease.worktree / "wmloop_interventions",
                run_root=training_run_root,
                primitive_names=[item.name for item in rendered],
            )
            candidate_checkpoint = _candidate_checkpoint_path(
                training_run_root=training_run_root,
                run_name=run_name,
                training_config=training_config,
            )
            action_probe_data_root = Path(str(replicates[0]["view_root"]))
            baseline_action_probe_root = self._output_root / "baseline-action-following-probe"
            candidate_action_probe_root = self._output_root / "candidate-action-following-probe"
            required = [
                "runtime_hook_unit",
                "acwm_train_smoke",
                "candidate_checkpoint_present",
                "baseline_action_following_probe",
                "candidate_action_following_probe",
            ]
            for replicate in replicates:
                label = str(replicate["label"])
                required.extend(
                    [
                        f"accept_cohort_view_{label}",
                        f"baseline_horizon_probe_{label}",
                        f"candidate_horizon_probe_{label}",
                    ]
                )
            session = AgentRepairSession(
                worktree=lease.worktree,
                staging_root=self._runs_root / proposal_id / "staging",
                candidate_id="m3-training-eval-smoke",
                source_revision=source_revision,
                registry_digest=registry.digest(),
                required_check_labels=required,
                environment=_runtime_environment(
                    runtime_python=self._runtime_python,
                    worktree=lease.worktree,
                    repo_root=self._repo_root,
                    output_root=self._output_root,
                    data_root=self._data_root,
                    checkpoint_root=self._checkpoint_root,
                    gpu_index=self._gpu_index,
                ),
                max_command_timeout_seconds=max(86_400.0, self._training_timeout),
            )
            for replicate in replicates:
                self._gpu_sampler.capture(
                    f"accept_cohort_view_{replicate['label']}",
                    lambda replicate=replicate: _run_required_check(
                        session,
                        label=f"accept_cohort_view_{replicate['label']}",
                        argv=(
                            str(self._runtime_python),
                            "-m",
                            "wmloop.evaluate.cohort_view_runtime",
                            "create",
                            "--data-root",
                            str(self._data_root),
                            "--output-root",
                            str(replicate["view_root"]),
                            "--environment",
                            self._environment,
                            "--cohort",
                            "ind_accept",
                            "--trajectory-ids-json",
                            str(replicate["trajectory_ids_path"]),
                            "--dataset-freeze-sha256",
                            str(replicate["dataset_freeze_sha256"]),
                        ),
                        timeout_seconds=120.0,
                        required_files=(Path(str(replicate["view_root"])) / "cohort-view-manifest.json",),
                    ),
                )
            self._gpu_sampler.capture(
                "runtime_hook_unit",
                lambda: _run_required_check(
                    session,
                    label="runtime_hook_unit",
                    argv=(str(self._runtime_python), "-c", hook_unit_script(self._proposal_primitive)),
                    timeout_seconds=self._hook_timeout,
                ),
            )
            self._gpu_sampler.capture(
                "acwm_train_smoke",
                lambda: _run_required_check(
                    session,
                    label="acwm_train_smoke",
                    argv=_training_entrypoint_command(
                        runtime=self._runtime_python,
                        worktree=lease.worktree,
                        run_root=training_run_root,
                        config_path=training_config_path,
                        checkpoint_path=official_checkpoint,
                    ),
                    timeout_seconds=self._training_timeout,
                ),
            )
            self._gpu_sampler.capture(
                "candidate_checkpoint_present",
                lambda: _run_required_check(
                    session,
                    label="candidate_checkpoint_present",
                    argv=(
                        str(self._runtime_python),
                        "-c",
                        "import sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if p.is_file() and not p.is_symlink() else 2)",
                        str(candidate_checkpoint),
                    ),
                    timeout_seconds=30.0,
                    required_files=(candidate_checkpoint,),
                ),
            )
            self._gpu_sampler.capture(
                "baseline_action_following_probe",
                lambda: _run_required_check(
                    session,
                    label="baseline_action_following_probe",
                    argv=_raw_probe_measure_command(
                        runtime=self._runtime_python,
                        repo_root=self._repo_root,
                        data_root=action_probe_data_root,
                        checkpoint_root=self._checkpoint_root,
                        checkpoint_path=official_checkpoint,
                        inverse_summary_path=self._repo_root / "results" / "m1" / "inverse-dynamics" / "summary-r1.json",
                        environment=self._environment,
                        output_root=baseline_action_probe_root,
                        horizons=self._eval_horizons,
                        primary_horizon=max(self._eval_horizons),
                        max_trajectories=self._max_accept_trajectories,
                        inference_steps=self._eval_inference_steps,
                        seed=101,
                        action_only=True,
                        gpu_index=self._gpu_index,
                        gpu_exclusivity_audit_manifest=self._gpu_exclusivity_audit_manifest,
                        gpu_exclusivity_max_age_seconds=self._gpu_exclusivity_trial_max_age_seconds,
                    ),
                    timeout_seconds=self._eval_timeout,
                    required_files=(
                        baseline_action_probe_root / "manifest.json",
                        baseline_action_probe_root / "measurements.json",
                    ),
                ),
            )
            self._gpu_sampler.capture(
                "candidate_action_following_probe",
                lambda: _run_required_check(
                    session,
                    label="candidate_action_following_probe",
                    argv=_raw_probe_measure_command(
                        runtime=self._runtime_python,
                        repo_root=self._repo_root,
                        vendor_root=lease.worktree,
                        data_root=action_probe_data_root,
                        checkpoint_root=self._checkpoint_root,
                        checkpoint_path=candidate_checkpoint,
                        inverse_summary_path=self._repo_root / "results" / "m1" / "inverse-dynamics" / "summary-r1.json",
                        environment=self._environment,
                        output_root=candidate_action_probe_root,
                        horizons=self._eval_horizons,
                        primary_horizon=max(self._eval_horizons),
                        max_trajectories=self._max_accept_trajectories,
                        inference_steps=self._eval_inference_steps,
                        seed=101,
                        action_only=True,
                        gpu_index=self._gpu_index,
                        gpu_exclusivity_audit_manifest=self._gpu_exclusivity_audit_manifest,
                        gpu_exclusivity_max_age_seconds=self._gpu_exclusivity_trial_max_age_seconds,
                    ),
                    timeout_seconds=self._eval_timeout,
                    required_files=(
                        candidate_action_probe_root / "manifest.json",
                        candidate_action_probe_root / "measurements.json",
                    ),
                ),
            )
            for replicate in replicates:
                label = str(replicate["label"])
                self._gpu_sampler.capture(
                    f"baseline_horizon_probe_{label}",
                    lambda replicate=replicate, label=label: _run_required_check(
                        session,
                        label=f"baseline_horizon_probe_{label}",
                        argv=_horizon_probe_command(
                            runtime=self._runtime_python,
                            repo_root=self._repo_root,
                            data_root=Path(str(replicate["view_root"])),
                            checkpoint_root=self._checkpoint_root,
                            checkpoint_path=official_checkpoint,
                            environment=self._environment,
                            output_root=Path(str(replicate["baseline_probe_root"])),
                            horizons=self._eval_horizons,
                            max_trajectories=self._max_accept_trajectories,
                            inference_steps=self._eval_inference_steps,
                            seed=int(replicate["seed"]),
                            gpu_index=self._gpu_index,
                            gpu_exclusivity_audit_manifest=self._gpu_exclusivity_audit_manifest,
                            gpu_exclusivity_max_age_seconds=self._gpu_exclusivity_trial_max_age_seconds,
                        ),
                        timeout_seconds=self._eval_timeout,
                        required_files=(
                            Path(str(replicate["baseline_probe_root"])) / "manifest.json",
                            Path(str(replicate["baseline_probe_root"])) / "metrics.json",
                        ),
                    ),
                )
                self._gpu_sampler.capture(
                    f"candidate_horizon_probe_{label}",
                    lambda replicate=replicate, label=label: _run_required_check(
                        session,
                        label=f"candidate_horizon_probe_{label}",
                        argv=_horizon_probe_command(
                            runtime=self._runtime_python,
                            repo_root=self._repo_root,
                            vendor_root=lease.worktree,
                            data_root=Path(str(replicate["view_root"])),
                            checkpoint_root=self._checkpoint_root,
                            checkpoint_path=candidate_checkpoint,
                            environment=self._environment,
                            output_root=Path(str(replicate["candidate_probe_root"])),
                            horizons=self._eval_horizons,
                            max_trajectories=self._max_accept_trajectories,
                            inference_steps=self._eval_inference_steps,
                            seed=int(replicate["seed"]),
                            gpu_index=self._gpu_index,
                            gpu_exclusivity_audit_manifest=self._gpu_exclusivity_audit_manifest,
                            gpu_exclusivity_max_age_seconds=self._gpu_exclusivity_trial_max_age_seconds,
                        ),
                        timeout_seconds=self._eval_timeout,
                        required_files=(
                            Path(str(replicate["candidate_probe_root"])) / "manifest.json",
                            Path(str(replicate["candidate_probe_root"])) / "metrics.json",
                        ),
                    ),
                )
            candidate = session.seal()
            candidate_runtime = _copy_candidate_runtime(
                source=lease.worktree,
                destination=self._output_root / "candidate-runtime",
                source_revision=source_revision,
                rendered_primitives=[{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
            )
            runtime_binding = _validate_probe_runtime_binding(
                baseline_runtime_sha256=runtime_tree_sha256(self._repo_root / "vendor" / "ACWM-Phys"),
                candidate_runtime_sha256=candidate_runtime["tree_sha256"],
                baseline_manifest_paths=[
                    baseline_action_probe_root / "manifest.json",
                    *(Path(str(item["baseline_probe_root"])) / "manifest.json" for item in replicates),
                ],
                candidate_manifest_paths=[
                    candidate_action_probe_root / "manifest.json",
                    *(Path(str(item["candidate_probe_root"])) / "manifest.json" for item in replicates),
                ],
            )
            replication_records = _load_probe_replications(
                replicates=replicates,
                required_horizons=self._eval_horizons,
                metric_name=self._primary_metric,
            )
            action_following = _load_action_following_probe_pair(
                baseline_root=baseline_action_probe_root,
                candidate_root=candidate_action_probe_root,
            )
            action_following_with_refs = _action_following_records_with_refs(
                action_following=action_following,
                cas=self._cas,
                archive=self._archive,
            )
            evidence = _verification_evidence_from_probe_replications(
                proposal_id=proposal_id,
                replications=replication_records,
                diff_audit_passed=candidate.ready_for_promotion,
                required_horizons=self._eval_horizons,
                metric_name=self._primary_metric,
                action_following=action_following_with_refs,
                action_following_threshold=self._action_following_threshold,
            )
            candidate_checkpoint_present = candidate_checkpoint.is_file()
            sandbox.remove(lease)
            worktree_removed = True
            attempts = _cas_attempts(candidate.manifest_path.parent / "attempts", cas=self._cas, archive=self._archive)
            gpu_sampling = self._gpu_sampler.to_document()
            receipt = {
                "schema_version": 1,
                "artifact_type": "wmloop-m3-training-eval-smoke-execution-receipt",
                "proposal_id": proposal_id,
                "environment": self._environment,
                "primary_metric": self._primary_metric,
                "fencing_token": fencing_token,
                "source_revision": source_revision,
                "registry_digest": registry.digest(),
                "runtime_python": str(self._runtime_python),
                "gpu_index": self._gpu_index,
                "gpu_exclusivity_launch_max_age_seconds": self._gpu_exclusivity_launch_max_age_seconds,
                "gpu_exclusivity_trial_max_age_seconds": self._gpu_exclusivity_trial_max_age_seconds,
                "proposal_primitive": self._proposal_primitive,
                "proposal_params": self._proposal_params,
                **({"trial_arm": self._trial_arm} if self._trial_arm is not None else {}),
                **({"seed": self._trial_seed} if self._trial_seed is not None else {}),
                "training_scale": {
                    "train_steps": self._train_steps,
                    "batch_size": self._train_batch_size,
                    "val_batch_size": self._train_val_batch_size,
                    "train_size": self._train_size,
                    "num_workers": self._train_num_workers,
                },
                "worktree_removed": worktree_removed,
                "rendered_primitives": [{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
                "training_config_ref": _put_file(self._cas, training_config_path, archive=self._archive, media_type="application/json"),
                "training_sidecars": training_sidecars,
                "official_checkpoint_path": str(official_checkpoint),
                "candidate_checkpoint_path": str(candidate_checkpoint),
                "candidate_runtime_path": candidate_runtime["runtime_path"],
                "candidate_runtime_manifest_path": candidate_runtime["manifest_path"],
                "candidate_runtime_sha256": candidate_runtime["tree_sha256"],
                "runtime_binding": runtime_binding,
                "candidate_checkpoint_exists_before_cleanup": candidate_checkpoint_present,
                "candidate_checkpoint_retained": False,
                "training_run_root": str(training_run_root),
                "accept_cohort_root": str(self._output_root / "accept-cohorts"),
                "accept_trajectory_count": sum(int(item["trajectory_count"]) for item in replicates),
                "replication_count": len(replication_records),
                "replications": _replication_records_with_refs(
                    replications=replication_records,
                    cas=self._cas,
                    archive=self._archive,
                ),
                "candidate": candidate.to_document(),
                "candidate_manifest_ref": _put_file(self._cas, candidate.manifest_path, archive=self._archive, media_type="application/json"),
                "candidate_diff_ref": _put_file(self._cas, candidate.diff_path, archive=self._archive, media_type="text/plain"),
                "attempt_refs": attempts,
                "gpu_sampling": gpu_sampling,
                "gpu_sampling_ref": _put_json(self._cas, gpu_sampling, archive=self._archive),
                "evaluation": _evaluation_summary(
                    replications=replication_records,
                    evidence=evidence,
                    metric_name=self._primary_metric,
                    action_following=action_following_with_refs,
                ),
                "state": "ready" if candidate.ready_for_promotion else "checks_failed",
                "actual_gpu_hours": _receipt_gpu_hours(candidate.to_document()),
            }
            receipt_ref = _put_json(self._cas, receipt, archive=self._archive)
            self._receipts[proposal_id] = {**receipt, "receipt_ref": receipt_ref}
            return ExecutionOutcome(
                actual_gpu_hours=float(receipt["actual_gpu_hours"]),
                receipt_ref=receipt_ref,
                verification_evidence=evidence,
            )
        finally:
            if lease is not None and not worktree_removed:
                try:
                    sandbox.remove(lease)
                except Exception:
                    pass

    def receipt(self, *, proposal_id: str) -> dict[str, object]:
        try:
            return self._receipts[proposal_id]
        except KeyError as exc:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_RECEIPT_MISSING") from exc


def _run_required_check(
    session: AgentRepairSession,
    *,
    label: str,
    argv: Sequence[str],
    timeout_seconds: float,
    required_files: Sequence[Path] = (),
) -> CommandReceipt:
    receipt = session.run(label=label, argv=argv, timeout_seconds=timeout_seconds)
    if not receipt.passed:
        status = "timeout" if receipt.timed_out else str(receipt.exit_code)
        raise OrchestratorTrainingEvalSmokeError(f"M3_TRAINING_EVAL_SMOKE_CHECK_FAILED:{label}:{status}")
    missing = [str(Path(path)) for path in required_files if not Path(path).is_file()]
    if missing:
        joined = ",".join(missing)
        raise OrchestratorTrainingEvalSmokeError(f"M3_TRAINING_EVAL_SMOKE_CHECK_ARTIFACT_MISSING:{label}:{joined}")
    return receipt


def _in_trial_gpu_exclusivity_max_age_seconds(
    *,
    launch_max_age_seconds: float | None,
    training_timeout_seconds: float,
    eval_timeout_seconds: float,
    hook_timeout_seconds: float,
    replication_count: int,
) -> float | None:
    """Keep launch freshness strict while covering one admitted trial's full lifecycle."""

    if launch_max_age_seconds is None:
        return None
    if min(training_timeout_seconds, eval_timeout_seconds, hook_timeout_seconds) < 0 or replication_count < 1:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GPU_AUDIT_LIFETIME_INVALID")
    probe_count = 2 + (2 * replication_count)
    lifecycle_seconds = (
        training_timeout_seconds
        + hook_timeout_seconds
        + (probe_count * eval_timeout_seconds)
        + 600.0
    )
    return max(float(launch_max_age_seconds), lifecycle_seconds)


class _TrainingEvalSmokeProposalClient:
    def __init__(
        self,
        *,
        environment: str,
        primitive: str,
        params: Mapping[str, object],
        metric_name: str,
        budget_estimate_gpu_hours: float | None = None,
        trial_arm: str | None = None,
        trial_seed: int | None = None,
    ) -> None:
        self._environment = environment
        self._primitive = primitive
        self._params = dict(params)
        self._metric_name = metric_name
        self._budget_estimate_gpu_hours = budget_estimate_gpu_hours
        self._trial_arm = trial_arm
        self._trial_seed = trial_seed

    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        manifest = allowed.get(self._primitive)
        if manifest is None:
            raise OrchestratorTrainingEvalSmokeError(f"M3_TRAINING_EVAL_SMOKE_PRIMITIVE_UNAVAILABLE:{self._primitive}")
        proposal = {
            "proposal_id": self._proposal_id(),
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [
                {"layer": manifest["layer"], "primitive": self._primitive, "params": self._params}
            ],
            "falsifiable_prediction": {
                "metric": self._metric_name,
                "horizon": _max_observed_horizon(failure),
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(
                self._budget_estimate_gpu_hours
                if self._budget_estimate_gpu_hours is not None
                else manifest["estimated_gpu_hours"]
            ),
            "rationale_ref": f"training_eval_m3_smoke#{failure['dominant_failure']}->{self._primitive}",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def _proposal_id(self) -> str:
        suffix = ""
        if self._trial_arm is not None or self._trial_seed is not None:
            arm = _safe_run_name(self._trial_arm or "unlabeled")
            seed = "unknown" if self._trial_seed is None else str(self._trial_seed)
            suffix = f"-{arm}-s{seed}"
        return f"m3-{_safe_run_name(self._environment)}-training-eval-smoke-{self._primitive}{suffix}-r1"


def _training_eval_budget_estimate_gpu_hours(
    *,
    registry: PrimitiveRegistry,
    primitive: str,
    train_steps: int,
    replication_count: int,
    horizon_count: int,
    action_following_probe_count: int,
) -> float:
    """Estimate the whole training+eval trial, not only the selected primitive hook."""

    if train_steps < 1 or replication_count < 1 or horizon_count < 1 or action_following_probe_count < 0:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_BUDGET_ESTIMATE_INVALID")
    manifest_estimate = float(registry.manifest(primitive).estimated_gpu_hours)
    training_hours = max(0.25, train_steps / 1024.0)
    probe_count = 2 * replication_count + action_following_probe_count
    eval_hours = probe_count * max(0.05, 0.01 * horizon_count)
    overhead_hours = 0.25
    estimated = max(manifest_estimate, training_hours + eval_hours + overhead_hours)
    goal_cap = 48.0
    if estimated > goal_cap:
        estimated = goal_cap
    return round(estimated, 3)


def _official_resume_training_config(
    *,
    vendor_root: Path,
    checkpoint_root: Path,
    environment: str = "push_cube",
    run_name: str,
    total_steps: int,
    batch_size: int = 1,
    val_batch_size: int | None = None,
    train_size: int = 1,
    num_workers: int = 0,
    learning_rate: float | None = None,
    train_steps: int = 1,
) -> dict[str, object]:
    if batch_size < 1 or (val_batch_size is not None and val_batch_size < 1) or train_size < 1 or num_workers < 0:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
    if train_steps < 1:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
    if learning_rate is not None and (not math.isfinite(float(learning_rate)) or learning_rate <= 0.0):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
    _environment_spec(environment)
    base = _load_yaml_with_pyyaml(Path(vendor_root) / "configs" / "envs" / f"{_vendor_environment_name(environment)}.yaml")
    model_type = str(base.get("model_type") or "")
    if model_type:
        model_spec = _load_yaml_with_pyyaml(Path(vendor_root) / "configs" / "model" / f"{model_type}.yaml")
        model_config = dict(base.get("model_config") or {})
        model_config.update(dict(model_spec.get("model_config") or {}))
        base["model_config"] = model_config
    model_config = dict(base["model_config"])  # type: ignore[index]
    model_config["vae_config"] = [str(Path(checkpoint_root).resolve() / "Wan2.1_VAE.pth")]
    model_config["use_flash_attn"] = False
    dataset = dict(base["dataset"])  # type: ignore[index]
    dataset.update(
        {
            "train_size": int(train_size),
            "ind_test_size": 1,
            "ood_test_size": 1,
            "test_cuts": 1,
            "cache_size": 2,
        }
    )
    training = dict(base["training"])  # type: ignore[index]
    training.update(
        {
            "batch_size": int(batch_size),
            "val_batch_size": int(val_batch_size if val_batch_size is not None else batch_size),
            **({"learning_rate": float(learning_rate)} if learning_rate is not None else {}),
            "num_epochs": max(1, int(base.get("training", {}).get("num_epochs", 1)) if isinstance(base.get("training"), Mapping) else 1),
            "total_steps": int(total_steps),
            "num_workers": int(num_workers),
            "log_freq": 1,
            "val_freq": 999999,
            "checkpoint_freq": _checkpoint_frequency(train_steps),
            "gen_mode": "parallel",
            "inference_steps": 1,
        }
    )
    return {
        **base,
        "model_config": model_config,
        "dataset": dataset,
        "training": training,
        "wandb": {"project": "wmloop-training-eval-smoke", "run_name": run_name},
        "distributed": {"use_fsdp": False},
    }


def _checkpoint_frequency(train_steps: int) -> int:
    """Keep smoke runs debuggable without making long confirmation runs I/O-bound."""

    if train_steps <= 32:
        return 1
    if train_steps <= 250:
        return 25
    if train_steps <= 1000:
        return 100
    return 500


def _bind_primitive_training_config(
    config: Mapping[str, object],
    *,
    primitive: str,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Make config-facing primitive intent explicit before the runtime hook runs."""

    output = dict(config)
    model_config = dict(output.get("model_config") or {})
    # This key is owned exclusively by latent_motion_prior. Scrub any stale
    # value before binding the currently reviewed intervention.
    model_config.pop("wmloop_latent_motion_prior_weight", None)
    if primitive == "latent_motion_prior":
        value = params.get("weight")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_LATENT_MOTION_WEIGHT_INVALID")
        latent_weight = float(value)
        if not math.isfinite(latent_weight) or latent_weight <= 0.0 or latent_weight > 4.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_LATENT_MOTION_WEIGHT_INVALID")
        model_config["wmloop_latent_motion_prior_weight"] = latent_weight
    if primitive == "motion_region_reweight":
        value = params.get("weight")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_MOTION_REGION_WEIGHT_INVALID")
        gamma = float(value)
        if not math.isfinite(gamma) or gamma <= 0.0 or gamma > 4.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_MOTION_REGION_WEIGHT_INVALID")
        model_config["motion_weighting_gamma"] = gamma
    output["model_config"] = model_config
    return output


def _proposal_params(
    *,
    primitive: str,
    weight: float,
    action_balance_blend: float = 0.5,
    action_balance_max_gain: float = 4.0,
    history_noise: float,
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
) -> dict[str, object]:
    if primitive in {"latent_motion_prior", "motion_region_reweight"}:
        if not math.isfinite(float(weight)) or weight <= 0.0 or (primitive == "motion_region_reweight" and weight > 4.0):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_WEIGHT_INVALID")
        return {"weight": weight}
    if primitive == "dino_rep_injection":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_DINO_WEIGHT_INVALID")
        return {"injection_weight": float(weight)}
    if primitive == "action_contrastive_finetune":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_CONTRASTIVE_WEIGHT_INVALID")
        return {"weight": float(weight)}
    if primitive == "action_dimension_balancing":
        if not math.isfinite(float(action_balance_blend)) or action_balance_blend <= 0.0 or action_balance_blend > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_BALANCE_BLEND_INVALID")
        if not math.isfinite(float(action_balance_max_gain)) or action_balance_max_gain < 1.0 or action_balance_max_gain > 8.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_BALANCE_MAX_GAIN_INVALID")
        return {"blend": float(action_balance_blend), "max_gain": float(action_balance_max_gain)}
    if primitive == "history_noise_schedule":
        if not math.isfinite(float(history_noise)) or history_noise < 0.0 or history_noise > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_HISTORY_NOISE_INVALID")
        return {"history_noise": float(history_noise)}
    if primitive == "drift_token_trim":
        if keep_tokens < 1:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_KEEP_TOKENS_INVALID")
        return {"keep_tokens": int(keep_tokens)}
    if primitive == "mixture_reweight":
        if not math.isfinite(float(frontier_weight)) or frontier_weight < 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FRONTIER_WEIGHT_INVALID")
        return {"frontier_weight": float(frontier_weight)}
    if primitive == "event_window_reweight":
        if not math.isfinite(float(event_weight)) or event_weight <= 0.0 or event_weight > 16.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVENT_WEIGHT_INVALID")
        if not math.isfinite(float(event_quantile)) or event_quantile < 0.5 or event_quantile > 0.95:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVENT_QUANTILE_INVALID")
        if not math.isfinite(float(event_visual_blend)) or event_visual_blend < 0.0 or event_visual_blend > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVENT_VISUAL_BLEND_INVALID")
        return {
            "event_weight": float(event_weight),
            "event_quantile": float(event_quantile),
            "visual_motion_blend": float(event_visual_blend),
        }
    if primitive == "next_forcing":
        if next_forcing_chunks < 2 or next_forcing_chunks > 8:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_CHUNKS_INVALID")
        if next_forcing_steps < 1:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_STEPS_INVALID")
        if not math.isfinite(float(next_forcing_lr)) or next_forcing_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_LR_INVALID")
        return {
            "chunks": int(next_forcing_chunks),
            "steps": int(next_forcing_steps),
            "lr": float(next_forcing_lr),
        }
    if primitive == "inv_dyn_reward_finetune":
        if not math.isfinite(float(reward_weight)) or reward_weight <= 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_INV_DYN_REWARD_WEIGHT_INVALID")
        if inv_dyn_steps < 1:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_INV_DYN_STEPS_INVALID")
        if not math.isfinite(float(inv_dyn_lr)) or inv_dyn_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_INV_DYN_LR_INVALID")
        return {
            "reward_weight": float(reward_weight),
            "steps": int(inv_dyn_steps),
            "lr": float(inv_dyn_lr),
        }
    if primitive == "latent_spatial_memory":
        if memory_slots < 1 or memory_slots > 128:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_MEMORY_SLOTS_INVALID")
        if not math.isfinite(float(memory_weight)) or memory_weight <= 0.0 or memory_weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_MEMORY_WEIGHT_INVALID")
        return {"memory_slots": int(memory_slots), "memory_weight": float(memory_weight)}
    if primitive == "first_frame_anchor":
        if anchor_every < 4 or anchor_every > 32:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ANCHOR_EVERY_INVALID")
        if not math.isfinite(float(anchor_weight)) or anchor_weight < 0.0 or anchor_weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ANCHOR_WEIGHT_INVALID")
        return {"anchor_every": int(anchor_every), "anchor_weight": float(anchor_weight)}
    if primitive == "cfg_guidance_schedule":
        if not math.isfinite(float(guidance_start)) or guidance_start < 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GUIDANCE_START_INVALID")
        if not math.isfinite(float(guidance_end)) or guidance_end < 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GUIDANCE_END_INVALID")
        return {"guidance_start": float(guidance_start), "guidance_end": float(guidance_end)}
    if primitive == "wmsd_self_distill":
        if not math.isfinite(float(wmsd_teacher_ema)) or wmsd_teacher_ema <= 0.0 or wmsd_teacher_ema > 1.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_WMSD_TEACHER_EMA_INVALID")
        if wmsd_steps < 1:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_WMSD_STEPS_INVALID")
        if not math.isfinite(float(wmsd_lr)) or wmsd_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_WMSD_LR_INVALID")
        return {"teacher_ema": float(wmsd_teacher_ema), "steps": int(wmsd_steps), "lr": float(wmsd_lr)}
    if primitive == "self_forcing_finetune":
        if self_forcing_rollout_horizon < 2:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SELF_FORCING_HORIZON_INVALID")
        if self_forcing_steps < 1:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SELF_FORCING_STEPS_INVALID")
        if not math.isfinite(float(self_forcing_lr)) or self_forcing_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SELF_FORCING_LR_INVALID")
        return {
            "rollout_horizon": int(self_forcing_rollout_horizon),
            "steps": int(self_forcing_steps),
            "lr": float(self_forcing_lr),
        }
    raise OrchestratorTrainingEvalSmokeError(f"M3_TRAINING_EVAL_SMOKE_PRIMITIVE_UNSUPPORTED:{primitive}")


def _augment_failure_with_signature_bank_routes(
    *,
    root: Path,
    failure: Mapping[str, Any],
    requested_primitive: str,
    proposal_routing_plan: Path | None = None,
) -> dict[str, Any]:
    augmented = dict(failure)
    signature_routed = _signature_bank_routed_failure_families(
        root=root,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    transfer_routed = _positive_transfer_routed_failure_families(
        root=root,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    targeted_routed, targeted_provenance = _targeted_plan_routed_failure_families(
        root=root,
        plan_path=proposal_routing_plan,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    routed = tuple(dict.fromkeys((*signature_routed, *transfer_routed, *targeted_routed)))
    if routed:
        existing = augmented.get("routed_failure_families")
        merged: list[str] = []
        if isinstance(existing, list):
            merged.extend(str(item) for item in existing if isinstance(item, str))
        merged.extend(routed)
        augmented["routed_failure_families"] = list(dict.fromkeys(item for item in merged if item and item != "mixed"))
        provenance = augmented.get("proposal_routing_provenance")
        provenance_rows = [dict(item) for item in provenance if isinstance(item, Mapping)] if isinstance(provenance, list) else []
        if signature_routed:
            augmented["signature_bank_routed_primitive"] = requested_primitive
            provenance_rows.append(
                {
                    "source": "failure_signature_bank",
                    "primitive": requested_primitive,
                    "routed_failure_families": list(signature_routed),
                }
            )
        if transfer_routed:
            augmented["positive_transfer_routed_primitive"] = requested_primitive
            provenance_rows.append(
                {
                    "source": "positive_transfer_exemplar",
                    "primitive": requested_primitive,
                    "routed_failure_families": list(transfer_routed),
                }
            )
        if targeted_routed:
            augmented["targeted_gap_plan_routed_primitive"] = requested_primitive
            provenance_rows.append(targeted_provenance)
        augmented["proposal_routing_provenance"] = provenance_rows
    return augmented


def _targeted_plan_routed_failure_families(
    *,
    root: Path,
    plan_path: Path | None,
    environment: str,
    primitive: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Compile an explicit targeted exploration route without rewriting diagnosis."""

    if plan_path is None:
        return (), {}
    path = Path(plan_path).resolve(strict=True)
    payload = _load_json_object(path)
    if payload.get("artifact_type") != "wmloop-acwm-targeted-gap-plan" or payload.get("state") != "ready":
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID")
    claim_boundary = payload.get("claim_boundary")
    records = payload.get("environment_records")
    if not isinstance(claim_boundary, str) or not claim_boundary or not isinstance(records, list):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("environment") == environment
        and primitive in (record.get("recommended_existing_primitives") or [])
    ]
    if len(matches) != 1:
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_SCOPE_MISMATCH:{environment}:{primitive}"
        )
    record = matches[0]
    hypothesis = record.get("mechanism_hypothesis")
    routed = record.get("routed_failure_families")
    probes = record.get("diagnostic_probe_candidates")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis
        or not isinstance(routed, list)
        or not routed
        or not all(isinstance(item, str) and item and item != "mixed" for item in routed)
        or not isinstance(probes, list)
        or not probes
        or not all(isinstance(item, str) and item for item in probes)
    ):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID")
    try:
        targets = set(PrimitiveRegistry.from_root(root).manifest(primitive).targets_failures)
    except (OSError, ValueError, KeyError) as exc:
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_PRIMITIVE_INVALID:{primitive}"
        ) from exc
    normalized = tuple(dict.fromkeys(str(item) for item in routed))
    if not set(normalized).issubset(targets):
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_TARGET_MISMATCH:{primitive}"
        )
    return normalized, {
        "source": "targeted_gap_plan",
        "path": str(path),
        "primitive": primitive,
        "routed_failure_families": list(normalized),
        "diagnostic_probe_candidates": list(probes),
        "mechanism_hypothesis": hypothesis,
        "claim_boundary": claim_boundary,
    }


def _positive_transfer_routed_failure_families(
    *, root: Path, environment: str, primitive: str
) -> tuple[str, ...]:
    """Admit a cross-environment route only after a formal positive exemplar exists."""

    if not environment or not primitive:
        return ()
    reports = Path(root) / "results" / "reports"
    has_positive_source = False
    for path in reports.glob("acwm-autoloop-official-gate-*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gate = manifest.get("official_quality_gate") if isinstance(manifest, Mapping) else None
        if (
            isinstance(manifest, Mapping)
            and manifest.get("state") == "ready"
            and manifest.get("primitive") == primitive
            and manifest.get("environment") != environment
            and isinstance(gate, Mapping)
            and gate.get("pass") is True
        ):
            has_positive_source = True
            break
    if not has_positive_source:
        return ()
    try:
        manifest = PrimitiveRegistry.from_root(root).manifest(primitive)
    except (OSError, ValueError, KeyError):
        return ()
    return tuple(failure for failure in manifest.targets_failures if failure != "mixed")


def _signature_bank_routed_failure_families(*, root: Path, environment: str, primitive: str) -> tuple[str, ...]:
    if not environment or not primitive:
        return ()
    bank_path = Path(root) / "results" / "reports" / "failure-signature-bank-r1" / "failure-signature-bank.json"
    if not bank_path.is_file() or bank_path.is_symlink():
        return ()
    try:
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-failure-signature-bank":
        return ()
    rows = payload.get("primitive_routing")
    if not isinstance(rows, list):
        return ()
    routed: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("environment") != environment or row.get("primitive") != primitive:
            continue
        decision = row.get("routing_decision")
        if decision not in {"stage_canary_after_diagnostic_probe", "retain_as_source_exemplar"}:
            continue
        failures = row.get("target_failures")
        if isinstance(failures, list):
            routed.extend(str(item) for item in failures if isinstance(item, str) and item != "mixed")
    return tuple(dict.fromkeys(routed))


def _verification_evidence_from_probe_manifests(
    *,
    proposal_id: str,
    baseline_manifest: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    diff_audit_passed: bool,
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
) -> VerificationEvidence:
    baseline_auc = _auc_value(baseline_manifest, horizons=required_horizons)
    candidate_auc = _auc_value(candidate_manifest, horizons=required_horizons)
    delta = candidate_auc - baseline_auc
    extended = _has_required_horizons(baseline_manifest, required_horizons) and _has_required_horizons(candidate_manifest, required_horizons)
    complete = (
        baseline_manifest.get("state") == "ready"
        and candidate_manifest.get("state") == "ready"
        and extended
        and math.isfinite(delta)
    )
    return VerificationEvidence(
        proposal_id=proposal_id,
        readonly_evaluator_verified=True,
        accept_split_verified=True,
        extended_horizon_verified=extended,
        diff_audit_passed=diff_audit_passed,
        evidence_complete=complete,
        accept_metric_deltas={metric_name: delta},
        replication_deltas=[delta],
        action_following_observed=0.0,
        action_following_threshold=0.0,
    )


def _verification_evidence_from_probe_replications(
    *,
    proposal_id: str,
    replications: Sequence[Mapping[str, object]],
    diff_audit_passed: bool,
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
    action_following: Mapping[str, object] | None = None,
    action_following_threshold: float | None = 0.0,
) -> VerificationEvidence:
    if not replications:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPLICATIONS_EMPTY")
    deltas = [_replication_delta(item, metric_name=metric_name) for item in replications]
    mean_delta = sum(deltas) / len(deltas)
    extended = all(bool(item.get("required_horizons_complete")) for item in replications)
    complete = extended and all(math.isfinite(value) for value in deltas)
    action_enabled = action_following_threshold is not None
    action_observed = _action_following_observed(action_following) if action_enabled else None
    return VerificationEvidence(
        proposal_id=proposal_id,
        readonly_evaluator_verified=True,
        accept_split_verified=True,
        extended_horizon_verified=extended,
        diff_audit_passed=diff_audit_passed,
        evidence_complete=complete,
        accept_metric_deltas={metric_name: mean_delta},
        replication_deltas=deltas,
        action_following_observed=action_observed,
        action_following_threshold=action_following_threshold,
        action_following_gate_enabled=action_enabled,
    )


def _write_accept_cohort_view_inputs(
    *,
    output_root: Path,
    heldout_protocol: Mapping[str, object],
    dataset_freeze: Mapping[str, object],
    environment: str = "push_cube",
    max_accept_trajectories: int,
) -> dict[str, object]:
    _environment_spec(environment)
    plan = build_baseline_evaluation_plan(dataset_freeze, heldout_protocol, gpus=(0, 1, 2))
    selection = plan.selection(environment, "ind_accept")
    selected = list(selection.trajectory_ids[:max_accept_trajectories])
    if not selected:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACCEPT_SELECTION_EMPTY")
    root = Path(output_root) / "accept-cohort"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ids_path = root / "trajectory-ids.json"
    _write_bytes_atomic(
        ids_path,
        json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n",
    )
    return {
        "view_root": root / "view",
        "trajectory_ids_path": ids_path,
        "trajectory_count": len(selected),
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
    }


def _write_accept_replicate_inputs(
    *,
    output_root: Path,
    heldout_protocol: Mapping[str, object],
    dataset_freeze: Mapping[str, object],
    environment: str = "push_cube",
    max_accept_trajectories: int,
    replication_count: int,
) -> tuple[dict[str, object], ...]:
    _environment_spec(environment)
    plan = build_baseline_evaluation_plan(dataset_freeze, heldout_protocol, gpus=(0, 1, 2))
    selection = plan.selection(environment, "ind_accept")
    required = max_accept_trajectories * replication_count
    selected_all = list(selection.trajectory_ids[:required])
    if len(selected_all) != required:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACCEPT_SELECTION_INSUFFICIENT")
    root = Path(output_root) / "accept-cohorts"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    replicates: list[dict[str, object]] = []
    for index in range(replication_count):
        label = f"r{index + 1:02d}"
        start = index * max_accept_trajectories
        selected = selected_all[start : start + max_accept_trajectories]
        replicate_root = root / label
        replicate_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        ids_path = replicate_root / "trajectory-ids.json"
        _write_bytes_atomic(
            ids_path,
            json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n",
        )
        replicates.append(
            {
                "index": index + 1,
                "label": label,
                "seed": 101 + index,
                "trajectory_ids": selected,
                "trajectory_ids_path": ids_path,
                "trajectory_count": len(selected),
                "view_root": replicate_root / "view",
                "baseline_probe_root": Path(output_root) / f"baseline-horizon-probe-{label}",
                "candidate_probe_root": Path(output_root) / f"candidate-horizon-probe-{label}",
                "dataset_freeze_sha256": plan.dataset_freeze_sha256,
                "heldout_protocol_sha256": plan.heldout_protocol_sha256,
            }
        )
    return tuple(replicates)


def _training_entrypoint_command(
    *,
    runtime: Path,
    worktree: Path,
    run_root: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    wrapper = (
        "import os, runpy, sys; "
        "train_path, run_root = sys.argv[1], sys.argv[2]; "
        "os.chdir(run_root); "
        "sys.argv = [train_path, *sys.argv[3:]]; "
        "namespace = runpy.run_path(train_path, run_name='__main__'); "
        "entrypoint = namespace.get('main'); "
        "state = getattr(entrypoint, '__globals__', namespace); "
        "step = int(state.get('_step', 0)); "
        "checkpoint_dir = state.get('_ckpt_dir', ''); "
        "save = state.get('save_checkpoint'); "
        "model = state.get('_model'); "
        "optimizer = state.get('_optimizer'); "
        "assert step > 0 and checkpoint_dir and callable(save) and model is not None and optimizer is not None; "
        "final_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pt'); "
        "save(model, optimizer, step, state.get('_epoch', 0), final_path, "
        "wandb_run_id=state.get('_wandb_run_id'))"
    )
    return (
        str(runtime),
        "-c",
        wrapper,
        str(Path(worktree) / "train.py"),
        str(run_root),
        "--config",
        str(config_path),
        "--ckpt_path",
        str(checkpoint_path),
    )


def _copy_training_sidecar(*, source: Path, run_root: Path) -> Path:
    if source.is_symlink() or not source.is_file():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SIDECAR_MISSING")
    target = Path(run_root) / "wmloop_interventions" / source.name
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _copy_training_sidecars(*, source_root: Path, run_root: Path, primitive_names: Sequence[str]) -> list[str]:
    copied: list[str] = []
    for primitive in dict.fromkeys(str(name) for name in primitive_names):
        if not primitive or "/" in primitive or "\\" in primitive:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SIDECAR_INVALID")
        target = _copy_training_sidecar(source=Path(source_root) / f"{primitive}.json", run_root=run_root)
        copied.append(str(target))
    if not copied:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_SIDECAR_MISSING")
    return copied


def _copy_candidate_runtime(
    *,
    source: Path,
    destination: Path,
    source_revision: str,
    rendered_primitives: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_dir() or destination.exists() or destination.is_symlink():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_INVALID")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
    )
    tree_sha256 = runtime_tree_sha256(destination)
    manifest_path = destination / "wmloop-runtime-manifest.json"
    _write_bytes_atomic(
        manifest_path,
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_type": "wmloop-materialized-candidate-runtime",
                "state": "ready",
                "source_revision": source_revision,
                "runtime_path": str(destination),
                "tree_sha256": tree_sha256,
                "rendered_primitives": [dict(item) for item in rendered_primitives],
            }
        ),
    )
    return {
        "runtime_path": str(destination),
        "manifest_path": str(manifest_path),
        "tree_sha256": tree_sha256,
    }


def _validate_probe_runtime_binding(
    *,
    baseline_runtime_sha256: str,
    candidate_runtime_sha256: str,
    baseline_manifest_paths: Sequence[Path],
    candidate_manifest_paths: Sequence[Path],
) -> dict[str, object]:
    if not baseline_manifest_paths or not candidate_manifest_paths:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_RUNTIME_BINDING_EMPTY")
    records: list[dict[str, str]] = []
    for side, expected, paths in (
        ("baseline", baseline_runtime_sha256, baseline_manifest_paths),
        ("candidate", candidate_runtime_sha256, candidate_manifest_paths),
    ):
        for path in paths:
            manifest = _load_json_object(Path(path))
            observed = manifest.get("vendor_runtime_sha256")
            if observed != expected:
                raise OrchestratorTrainingEvalSmokeError(
                    f"M3_TRAINING_EVAL_SMOKE_RUNTIME_BINDING_MISMATCH:{side}:{path}"
                )
            records.append(
                {
                    "side": side,
                    "manifest_path": str(Path(path)),
                    "vendor_root": str(manifest.get("vendor_root") or ""),
                    "runtime_sha256": expected,
                }
            )
    return {
        "state": "ready",
        "baseline_runtime_sha256": baseline_runtime_sha256,
        "candidate_runtime_sha256": candidate_runtime_sha256,
        "records": records,
    }


def _horizon_probe_command(
    *,
    runtime: Path,
    repo_root: Path,
    vendor_root: Path | None = None,
    data_root: Path,
    checkpoint_root: Path,
    checkpoint_path: Path,
    environment: str = "push_cube",
    output_root: Path,
    horizons: Sequence[int],
    max_trajectories: int,
    inference_steps: int,
    gpu_index: int,
    gpu_exclusivity_audit_manifest: Path,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    seed: int = 101,
) -> tuple[str, ...]:
    command = [
        str(runtime),
        "-m",
        "wmloop.diagnose.horizon_runtime",
        "run",
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-path",
        str(checkpoint_path),
        "--environment",
        environment,
        "--split",
        "ind_test",
        "--output-root",
        str(output_root),
        "--horizons",
        *(str(value) for value in horizons),
        "--max-trajectories",
        str(max_trajectories),
        "--num-inference-steps",
        str(inference_steps),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--mode",
        "autoregressive",
        "--max-evidence",
        "1",
        "--gpu-index",
        str(gpu_index),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_exclusivity_audit_manifest),
    ]
    if gpu_exclusivity_max_age_seconds is not None:
        command.extend(["--gpu-exclusivity-max-age-seconds", str(gpu_exclusivity_max_age_seconds)])
    if vendor_root is not None:
        command.extend(["--vendor-root", str(vendor_root)])
    return tuple(command)


def _raw_probe_measure_command(
    *,
    runtime: Path,
    repo_root: Path,
    vendor_root: Path | None = None,
    data_root: Path,
    checkpoint_root: Path,
    checkpoint_path: Path,
    inverse_summary_path: Path,
    environment: str = "push_cube",
    output_root: Path,
    horizons: Sequence[int],
    primary_horizon: int,
    max_trajectories: int,
    inference_steps: int,
    gpu_index: int,
    gpu_exclusivity_audit_manifest: Path,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    seed: int = 101,
    action_only: bool = False,
) -> tuple[str, ...]:
    command = [
        str(runtime),
        "-m",
        "wmloop.diagnose.raw_probe_measure_runtime",
        "run",
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-path",
        str(checkpoint_path),
        "--inverse-summary",
        str(inverse_summary_path),
        "--environment",
        environment,
        "--output-root",
        str(output_root),
        "--horizons",
        *(str(value) for value in horizons),
        "--primary-horizon",
        str(primary_horizon),
        "--max-trajectories",
        str(max_trajectories),
        "--num-inference-steps",
        str(inference_steps),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--mode",
        "autoregressive",
        "--gpu-index",
        str(gpu_index),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_exclusivity_audit_manifest),
    ]
    if gpu_exclusivity_max_age_seconds is not None:
        command.extend(["--gpu-exclusivity-max-age-seconds", str(gpu_exclusivity_max_age_seconds)])
    if vendor_root is not None:
        command.extend(["--vendor-root", str(vendor_root)])
    if action_only:
        command.append("--action-only")
    return tuple(command)


def _load_probe_replications(
    *,
    replicates: Sequence[Mapping[str, object]],
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for replicate in replicates:
        baseline_root = Path(str(replicate["baseline_probe_root"]))
        candidate_root = Path(str(replicate["candidate_probe_root"]))
        baseline_manifest = _load_json_object(baseline_root / "manifest.json")
        candidate_manifest = _load_json_object(candidate_root / "manifest.json")
        baseline_auc = _auc_value(baseline_manifest, horizons=required_horizons)
        candidate_auc = _auc_value(candidate_manifest, horizons=required_horizons)
        delta = candidate_auc - baseline_auc
        records.append(
            {
                "index": int(replicate["index"]),
                "label": str(replicate["label"]),
                "seed": int(replicate["seed"]),
                "trajectory_ids": list(replicate["trajectory_ids"]),  # type: ignore[arg-type]
                "trajectory_count": int(replicate["trajectory_count"]),
                "primary_metric": metric_name,
                "baseline_primary_metric": baseline_auc,
                "candidate_primary_metric": candidate_auc,
                "delta_primary_metric": delta,
                f"baseline_{metric_name}": baseline_auc,
                f"candidate_{metric_name}": candidate_auc,
                f"delta_{metric_name}": delta,
                "baseline_auc_psnr_16_64": baseline_auc,
                "candidate_auc_psnr_16_64": candidate_auc,
                "delta_auc_psnr_16_64": delta,
                "required_horizons_complete": _has_required_horizons(baseline_manifest, required_horizons)
                and _has_required_horizons(candidate_manifest, required_horizons),
                "accept_cohort_view_manifest_path": str(Path(str(replicate["view_root"])) / "cohort-view-manifest.json"),
                "baseline_manifest_path": str(baseline_root / "manifest.json"),
                "candidate_manifest_path": str(candidate_root / "manifest.json"),
                "baseline_metrics_path": str(baseline_root / "metrics.json"),
                "candidate_metrics_path": str(candidate_root / "metrics.json"),
            }
        )
    return tuple(records)


def _load_action_following_probe_pair(*, baseline_root: Path, candidate_root: Path) -> dict[str, object]:
    baseline = _load_action_following_measurement(Path(baseline_root) / "measurements.json")
    candidate = _load_action_following_measurement(Path(candidate_root) / "measurements.json")
    return {
        "baseline": {
            **baseline,
            "manifest_path": str(Path(baseline_root) / "manifest.json"),
            "measurements_path": str(Path(baseline_root) / "measurements.json"),
        },
        "candidate": {
            **candidate,
            "manifest_path": str(Path(candidate_root) / "manifest.json"),
            "measurements_path": str(Path(candidate_root) / "measurements.json"),
        },
        "delta_inv_dyn_acc_perframe": float(candidate["inv_dyn_acc_perframe"]) - float(baseline["inv_dyn_acc_perframe"]),
        "delta_no_action_delta_psnr": float(candidate["no_action_delta_psnr"]) - float(baseline["no_action_delta_psnr"]),
    }


def _load_action_following_measurement(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if payload.get("artifact_type") not in {
        "wmloop-raw-probe-measurement-input",
        "wmloop-action-following-measurement-input",
    } or payload.get("source_kind") != "measured":
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    record = records[0]
    action = record.get("action_following")
    if not isinstance(action, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    predicted = _action_rows(action.get("predicted_actions"))
    target = _action_rows(action.get("target_actions"))
    tolerance = _finite_float(action.get("tolerance"), "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    action_conditioned_psnr = _finite_float(
        action.get("action_conditioned_psnr"),
        "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID",
    )
    no_action_psnr = _finite_float(action.get("no_action_psnr"), "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    inverse_dynamics_r2 = _finite_float(
        action.get("inverse_dynamics_r2"),
        "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID",
    )
    low_confidence = action.get("low_confidence")
    if not isinstance(low_confidence, bool):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    try:
        accuracy = per_frame_inverse_dynamics_accuracy(
            predicted_actions=predicted,
            target_actions=target,
            tolerance=tolerance,
        )
        delta = no_action_delta_psnr(
            action_conditioned_psnr=action_conditioned_psnr,
            no_action_psnr=no_action_psnr,
        )
    except ValueError as exc:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID") from exc
    return {
        "state": "measured",
        "inv_dyn_acc_perframe": accuracy,
        "no_action_delta_psnr": delta,
        "frame_count": len(target),
        "tolerance": tolerance,
        "action_conditioned_psnr": action_conditioned_psnr,
        "no_action_psnr": no_action_psnr,
        "inverse_dynamics_r2": inverse_dynamics_r2,
        "low_confidence": low_confidence,
    }


def _action_rows(raw: object) -> list[list[float]]:
    if not isinstance(raw, list) or not raw:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    rows: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list) or not row:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
        rows.append([_finite_float(value, "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID") for value in row])
    return rows


def _action_following_observed(action_following: Mapping[str, object] | None) -> float:
    if action_following is None:
        return 0.0
    candidate = action_following.get("candidate")
    if not isinstance(candidate, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
    return _finite_float(candidate.get("inv_dyn_acc_perframe"), "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")


def _replication_records_with_refs(
    *,
    replications: Sequence[Mapping[str, object]],
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    path_media = {
        "accept_cohort_view_manifest_path": "application/json",
        "baseline_manifest_path": "application/json",
        "candidate_manifest_path": "application/json",
        "baseline_metrics_path": "application/json",
        "candidate_metrics_path": "application/json",
    }
    for replication in replications:
        item = dict(replication)
        refs: dict[str, str] = {}
        for key, media_type in path_media.items():
            value = item.get(key)
            if not isinstance(value, str):
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPLICATION_ARTIFACT_INVALID")
            refs[key.replace("_path", "_ref")] = _put_file(cas, Path(value), archive=archive, media_type=media_type)
        item["cas_refs"] = refs
        output.append(item)
    return output


def _action_following_records_with_refs(
    *,
    action_following: Mapping[str, object],
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    output = json.loads(_canonical_json_bytes(action_following))
    for side in ("baseline", "candidate"):
        payload = output.get(side)
        if not isinstance(payload, dict):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
        refs: dict[str, str] = {}
        for key in ("manifest_path", "measurements_path"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
            refs[key.replace("_path", "_ref")] = _put_file(cas, Path(value), archive=archive, media_type="application/json")
        payload["cas_refs"] = refs
    return output


def _record_formal_settled_trial(
    *,
    archive: ArchiveStore,
    proposal: Mapping[str, Any],
    verdict: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_ref: str,
    failure_context_ref: str,
    verdict_ref: str,
    gpu_hours: float,
    settlement_state: str,
    round_start_verification: object,
) -> bool:
    interventions = proposal.get("interventions")
    if not isinstance(interventions, list) or not interventions or not all(isinstance(item, Mapping) for item in interventions):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_INTERVENTIONS_INVALID")
    single = len(interventions) == 1
    cell = _formal_trial_cell(proposal=proposal, interventions=interventions)
    verdict_label = verdict.get("verdict")
    exploratory = not single or verdict_label == "VOID"
    verified_gain = None if exploratory else _primary_delta(verdict)
    archive.record_settled_trial(
        SettledTrialRecord(
            trial_id=str(proposal["proposal_id"]),
            proposal_id=str(proposal["proposal_id"]),
            goal_id=str(proposal["goal_id"]),
            library_version=str(proposal.get("library_version", "v1.0")),
            failure_context_ref=failure_context_ref,
            verdict_ref=verdict_ref,
            receipt_ref=receipt_ref,
            gpu_hours=float(gpu_hours),
            hypothesis_hash=hashlib.sha256(_canonical_json_bytes(proposal)).hexdigest(),
            impl_diff_hash=_impl_diff_hash(receipt),
            evaluator_hash=_evaluator_hash(round_start_verification),
            settlement_state=settlement_state,
            receipt_hash=_digest_from_cas_ref(receipt_ref),
            cell=cell,
            verified_gain=verified_gain,
            exploratory=exploratory,
        )
    )
    return True


def _formal_trial_cell(*, proposal: Mapping[str, Any], interventions: Sequence[Mapping[str, Any]]) -> InterventionCell:
    if len(interventions) == 1:
        first = interventions[0]
        layer = first.get("layer")
        primitive = first.get("primitive")
        params = first.get("params")
        if not isinstance(layer, str) or not isinstance(primitive, str) or not isinstance(params, Mapping):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_INTERVENTIONS_INVALID")
        return InterventionCell(
            environment=str(proposal["env"]),
            layer=layer,
            primitive_family=primitive,
            parameter_bucket=json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
    return InterventionCell(
        environment=str(proposal["env"]),
        layer="combo",
        primitive_family="combo",
        parameter_bucket=json.dumps(list(interventions), sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _primary_delta(verdict: Mapping[str, Any]) -> float:
    deltas = verdict.get("delta_m_ver")
    if not isinstance(deltas, Mapping) or not deltas:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_DELTA_INVALID")
    for value in deltas.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_DELTA_INVALID")


def _impl_diff_hash(receipt: Mapping[str, Any]) -> str:
    candidate = receipt.get("candidate")
    if isinstance(candidate, Mapping):
        digest = candidate.get("worktree_diff_sha256")
        if isinstance(digest, str) and _is_sha256(digest):
            return digest
    diff_ref = receipt.get("candidate_diff_ref")
    if isinstance(diff_ref, str):
        return _digest_from_cas_ref(diff_ref)
    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_DIFF_HASH_INVALID")


def _evaluator_hash(round_start: object) -> str:
    if isinstance(round_start, Mapping):
        evaluator = round_start.get("evaluator_freeze")
        if isinstance(evaluator, Mapping):
            digest = evaluator.get("sha256")
            if isinstance(digest, str) and _is_sha256(digest):
                return digest
    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_EVALUATOR_HASH_INVALID")


def _digest_from_cas_ref(ref: str) -> str:
    prefix = "cas://sha256/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_CAS_REF_INVALID")
    digest = ref[len(prefix) :]
    if not _is_sha256(digest):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FORMAL_CAS_REF_INVALID")
    return digest


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _evaluation_summary(
    *,
    replications: Sequence[Mapping[str, object]],
    evidence: VerificationEvidence,
    metric_name: str = "auc_psnr_16_64",
    action_following: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not replications:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPLICATIONS_EMPTY")
    baseline_values = [_replication_value(item, prefix="baseline", metric_name=metric_name) for item in replications]
    candidate_values = [_replication_value(item, prefix="candidate", metric_name=metric_name) for item in replications]
    deltas = [_replication_delta(item, metric_name=metric_name) for item in replications]
    first = replications[0]
    summary: dict[str, object] = {
        "primary_metric": metric_name,
        "baseline_primary_metric": sum(baseline_values) / len(baseline_values),
        "candidate_primary_metric": sum(candidate_values) / len(candidate_values),
        "delta_primary_metric": evidence.accept_metric_deltas[metric_name],
        "baseline_auc_psnr_16_64": sum(baseline_values) / len(baseline_values),
        "candidate_auc_psnr_16_64": sum(candidate_values) / len(candidate_values),
        "delta_auc_psnr_16_64": evidence.accept_metric_deltas[metric_name],
        "replication_deltas": deltas,
        "replication_count": len(replications),
        "required_horizons_complete": all(bool(item.get("required_horizons_complete")) for item in replications),
        "replications": [dict(item) for item in replications],
    }
    for key in (
        "accept_cohort_view_manifest_path",
        "baseline_manifest_path",
        "candidate_manifest_path",
        "baseline_metrics_path",
        "candidate_metrics_path",
    ):
        summary[key] = first[key]
    if action_following is not None:
        summary["action_following"] = {
            **dict(action_following),
            "gate_enabled": evidence.action_following_gate_enabled,
            "gate_threshold": evidence.action_following_threshold,
            "gate_observed": evidence.action_following_observed,
        }
    return summary


def _checkpoint_step(*, runtime_python: Path, checkpoint_path: Path) -> int:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_OFFICIAL_CHECKPOINT_MISSING")
    code = "import json, sys, torch; p=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(json.dumps({'step': int(p.get('step', 0))}))"
    completed = subprocess.run(
        [str(runtime_python), "-c", code, str(checkpoint_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CHECKPOINT_STEP_UNREADABLE")
    payload = json.loads(completed.stdout)
    step = payload.get("step")
    if not isinstance(step, int) or step < 0:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CHECKPOINT_STEP_INVALID")
    return step


def _load_yaml_with_pyyaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_YAML_UNAVAILABLE") from exc
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # type: ignore[attr-defined]
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CONFIG_INVALID") from exc
    if not isinstance(payload, dict):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CONFIG_INVALID")
    return payload


def _safe_run_name(proposal_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", proposal_id).strip("_")
    if not name:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_RUN_NAME_INVALID")
    return name[:96]


def _candidate_checkpoint_path(*, training_run_root: Path, run_name: str, training_config: Mapping[str, object]) -> Path:
    return Path(training_run_root) / "checkpoints" / f"{run_name}_{_checkpoint_resolution_suffix(training_config)}" / "latest.pt"


def _checkpoint_resolution_suffix(training_config: Mapping[str, object]) -> str:
    dataset = training_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID")
    obs_shape = dataset.get("obs_shape")
    if not isinstance(obs_shape, list) or len(obs_shape) != 3:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID")
    height = _positive_int(obs_shape[1])
    width = _positive_int(obs_shape[2])
    return f"{height}x{width}"


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID")
    return value


def _environment_spec(environment: str) -> Any:
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise OrchestratorTrainingEvalSmokeError(f"M3_TRAINING_EVAL_SMOKE_ENVIRONMENT_UNKNOWN:{environment}")


def _goal_action_following_threshold(goal: Mapping[str, object]) -> float | None:
    raw = goal.get("action_following_gate")
    if not isinstance(raw, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GOAL_AF_GATE_INVALID")
    tau = _finite_float(raw.get("tau_af"), "M3_TRAINING_EVAL_SMOKE_GOAL_AF_GATE_INVALID")
    return None if tau < 0.0 else tau


def _goal_id(goal: Mapping[str, object]) -> str:
    value = goal.get("goal_id")
    if not isinstance(value, str) or not value:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GOAL_INVALID")
    return value


def _verify_goal_environment(goal: Mapping[str, object], environment: str) -> None:
    raw = goal.get("envs")
    if raw is None:
        return
    if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GOAL_INVALID")
    if environment not in raw:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_FAILURE_SCOPE_INVALID")


def _goal_metric_name(*, goal: Mapping[str, object], horizons: Sequence[int]) -> str:
    primary = goal.get("primary_objective")
    if isinstance(primary, str) and primary:
        return primary
    return _auc_metric_name(horizons)


def _auc_metric_name(horizons: Sequence[int]) -> str:
    points = sorted(int(value) for value in horizons)
    if len(points) < 2:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID")
    return f"auc_psnr_{points[0]}_{points[-1]}"


def _max_observed_horizon(failure: Mapping[str, object]) -> int:
    curve = failure.get("horizon_curve")
    if not isinstance(curve, Mapping):
        return 64
    psnr = curve.get("psnr")
    if not isinstance(psnr, Mapping):
        return 64
    horizons = []
    for key in psnr:
        try:
            horizons.append(int(key))
        except (TypeError, ValueError):
            continue
    return max(horizons) if horizons else 64


def _vendor_environment_name(environment: str) -> str:
    return "clothmove" if environment == "cloth_move" else environment


def _auc_value(manifest: Mapping[str, object], *, horizons: Sequence[int] | None = None) -> float:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_PROBE_MANIFEST_INVALID")
    curve = aggregate.get("horizon_curve")
    if not isinstance(curve, Mapping):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_PROBE_MANIFEST_INVALID")
    candidate_keys: list[str] = []
    if horizons is not None:
        candidate_keys.append(_auc_metric_name(horizons))
    candidate_keys.extend(["auc_psnr_16_64", "auc_psnr_envmax"])
    value = None
    for key in candidate_keys:
        if key in curve:
            value = curve.get(key)
            break
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_AUC_MISSING")
    return float(value)


def _replication_value(replication: Mapping[str, object], *, prefix: str, metric_name: str) -> float:
    for key in (f"{prefix}_{metric_name}", f"{prefix}_primary_metric", f"{prefix}_auc_psnr_16_64"):
        value = replication.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPLICATION_METRIC_MISSING")


def _replication_delta(replication: Mapping[str, object], *, metric_name: str) -> float:
    for key in (f"delta_{metric_name}", "delta_primary_metric", "delta_auc_psnr_16_64"):
        value = replication.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPLICATION_METRIC_MISSING")


def _finite_float(value: object, error_code: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise OrchestratorTrainingEvalSmokeError(error_code)
    return float(value)


def _has_required_horizons(manifest: Mapping[str, object], required_horizons: Sequence[int]) -> bool:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return False
    metrics = aggregate.get("horizon_metrics")
    if not isinstance(metrics, Mapping):
        return False
    return all(str(horizon) in metrics for horizon in required_horizons)


def _receipt_gpu_hours(candidate: Mapping[str, object]) -> float:
    receipts = candidate.get("receipts")
    if not isinstance(receipts, list):
        return 0.0
    seconds = 0.0
    for receipt in receipts:
        if isinstance(receipt, Mapping):
            duration = receipt.get("duration_seconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(float(duration)):
                seconds += float(duration)
    return seconds / 3600.0


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
    budget_db: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        public_report = _bundle_evaluation_files(report=report, temporary=temporary, destination=destination)
        report_bytes = _canonical_json_bytes(public_report)
        markdown_bytes = _render_markdown(public_report).encode("utf-8")
        _write_bytes_atomic(temporary / "orchestrator-training-eval-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "orchestrator-training-eval-smoke.md", markdown_bytes)
        shutil.copy2(budget_db, temporary / "budget.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(budget_db.read_bytes(), media_type="application/x-sqlite3").uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, budget_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-smoke-manifest",
            "state": public_report["state"],
            "environment": public_report["environment"],
            "goal_id": public_report["goal_id"],
            "primary_metric": public_report.get("primary_metric"),
            "proposal_id": public_report["proposal_id"],
            **({"trial_arm": public_report["trial_arm"]} if "trial_arm" in public_report else {}),
            **({"seed": public_report["seed"]} if "seed" in public_report else {}),
            "verdict": public_report["verdict"],
            "action_following_gate": public_report["action_following_gate"],
            "delta_m_ver": public_report["delta_m_ver"],
            "gpu_exclusivity_audit": public_report.get("gpu_exclusivity_audit"),
            "formal_settled_trial_published": public_report.get("formal_settled_trial_published", False),
            "candidate_checkpoint_retained": public_report.get("receipt", {}).get("candidate_checkpoint_retained", False),
            "candidate_checkpoint_retained_path": public_report.get("receipt", {}).get("candidate_checkpoint_retained_path"),
            "candidate_checkpoint_sha256": public_report.get("receipt", {}).get("candidate_checkpoint_sha256"),
            "candidate_runtime_retained": public_report.get("receipt", {}).get("candidate_runtime_retained", False),
            "candidate_runtime_root": public_report.get("receipt", {}).get("candidate_runtime_root"),
            "candidate_runtime_sha256": public_report.get("receipt", {}).get("candidate_runtime_sha256"),
            "report_path": str(destination / "orchestrator-training-eval-smoke.json"),
            "markdown_path": str(destination / "orchestrator-training-eval-smoke.md"),
            "budget_db_path": str(destination / "budget.db"),
            "evaluation_dir": str(destination / "evaluation"),
            "retained_training_dir": str(destination / "retained_training"),
            "retained_runtime_dir": str(destination / "retained_runtime"),
            "cas_refs": {
                "orchestrator_training_eval_smoke_json": report_ref,
                "orchestrator_training_eval_smoke_markdown": markdown_ref,
                "budget_db": budget_ref,
            },
            "limitations": public_report["limitations"],
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


def _bundle_evaluation_files(
    *,
    report: Mapping[str, object],
    temporary: Path,
    destination: Path,
) -> dict[str, Any]:
    public = json.loads(_canonical_json_bytes(report))
    receipt = public.get("receipt")
    if not isinstance(receipt, dict):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, dict):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
    final_dir = destination / "evaluation"
    bundle_dir = temporary / "evaluation"
    bundle_dir.mkdir(mode=0o700, parents=True)
    path_keys = {
        "accept_cohort_view_manifest_path": "accept-cohort-view-manifest.json",
        "baseline_manifest_path": "baseline-horizon-probe-manifest.json",
        "candidate_manifest_path": "candidate-horizon-probe-manifest.json",
        "baseline_metrics_path": "baseline-horizon-probe-metrics.json",
        "candidate_metrics_path": "candidate-horizon-probe-metrics.json",
    }
    for key, name in path_keys.items():
        value = evaluation.get(key)
        if not isinstance(value, str) or not value:
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
        source = Path(value)
        if source.is_symlink() or not source.is_file():
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING")
        shutil.copy2(source, bundle_dir / name)
        evaluation[key] = str(final_dir / name)
    replications = evaluation.get("replications")
    if isinstance(replications, list):
        for index, replication in enumerate(replications, start=1):
            if not isinstance(replication, dict):
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
            label = str(replication.get("label") or f"r{index:02d}")
            safe_label = _safe_run_name(label)
            replicate_bundle_dir = bundle_dir / safe_label
            replicate_final_dir = final_dir / safe_label
            replicate_bundle_dir.mkdir(mode=0o700)
            for key, name in path_keys.items():
                value = replication.get(key)
                if not isinstance(value, str) or not value:
                    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
                source = Path(value)
                if source.is_symlink() or not source.is_file():
                    raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING")
                shutil.copy2(source, replicate_bundle_dir / name)
                replication[key] = str(replicate_final_dir / name)
    action_following = evaluation.get("action_following")
    if isinstance(action_following, dict):
        _bundle_action_following_files(action_following=action_following, bundle_dir=bundle_dir, final_dir=final_dir)
    _bundle_candidate_checkpoint(receipt=receipt, temporary=temporary, destination=destination, public=public)
    # Older synthetic receipts in the unit suite predate runtime-bound evidence.
    # Real training/eval receipts always carry both runtime fields and therefore
    # still fail closed if the candidate runtime cannot be retained.
    if "candidate_runtime_path" in receipt or "candidate_runtime_sha256" in receipt:
        _bundle_candidate_runtime(receipt=receipt, temporary=temporary, destination=destination)
    return public


def _bundle_action_following_files(*, action_following: dict[str, object], bundle_dir: Path, final_dir: Path) -> None:
    path_keys = {
        "manifest_path": "manifest.json",
        "measurements_path": "measurements.json",
    }
    for side in ("baseline", "candidate"):
        payload = action_following.get(side)
        if not isinstance(payload, dict):
            raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
        side_bundle = bundle_dir / "action-following" / side
        side_final = final_dir / "action-following" / side
        side_bundle.mkdir(mode=0o700, parents=True)
        for key, name in path_keys.items():
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
            source = Path(value)
            if source.is_symlink() or not source.is_file():
                raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING")
            shutil.copy2(source, side_bundle / name)
            payload[key] = str(side_final / name)


def _bundle_candidate_checkpoint(
    *,
    receipt: dict[str, object],
    temporary: Path,
    destination: Path,
    public: dict[str, Any],
) -> None:
    source_value = receipt.get("candidate_checkpoint_path")
    if not isinstance(source_value, str) or not source_value:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_REPORT_INVALID")
    source = Path(source_value)
    if source.is_symlink() or not source.is_file():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CANDIDATE_CHECKPOINT_MISSING")
    retained_dir = temporary / "retained_training"
    retained_dir.mkdir(mode=0o700, parents=True)
    retained_path = retained_dir / "latest.pt"
    latest_storage_mode = _link_or_copy_checkpoint(source, retained_path)
    digest = _file_sha256(retained_path)
    byte_count = retained_path.stat().st_size
    final_path = destination / "retained_training" / "latest.pt"
    training_scale = receipt.get("training_scale")
    train_steps = training_scale.get("train_steps") if isinstance(training_scale, Mapping) else None
    ladder = _bundle_checkpoint_ladder(
        source=source,
        retained_dir=retained_dir,
        destination=destination,
        train_steps=train_steps,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-retained-training-checkpoint",
        "source_path": str(source),
        "retained_path": str(final_path),
        "sha256": digest,
        "bytes": byte_count,
        "storage_mode": latest_storage_mode,
        "checkpoint_ladder": ladder,
    }
    _write_bytes_atomic(retained_dir / "manifest.json", _canonical_json_bytes(manifest))
    receipt["candidate_checkpoint_retained"] = True
    receipt["candidate_checkpoint_retained_path"] = str(final_path)
    receipt["candidate_checkpoint_retained_manifest_path"] = str(destination / "retained_training" / "manifest.json")
    receipt["candidate_checkpoint_sha256"] = digest
    receipt["candidate_checkpoint_bytes"] = byte_count
    receipt["candidate_checkpoint_ladder"] = ladder
    limitations = public.get("limitations")
    if isinstance(limitations, list):
        public["limitations"] = [
            item
            for item in limitations
            if not (isinstance(item, str) and "candidate checkpoint" in item and "not retained" in item)
        ]
        public["limitations"].append(
            "The final candidate and configured checkpoint ladder are retained under retained_training; formal claims still require frozen official evaluator receipts."
        )


def _bundle_candidate_runtime(*, receipt: dict[str, object], temporary: Path, destination: Path) -> None:
    source_value = receipt.get("candidate_runtime_path")
    expected_sha = receipt.get("candidate_runtime_sha256")
    if not isinstance(source_value, str) or not source_value or not isinstance(expected_sha, str) or not expected_sha:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_MISSING")
    source = Path(source_value)
    if source.is_symlink() or not source.is_dir():
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_MISSING")
    retained = temporary / "retained_runtime"
    shutil.copytree(source, retained)
    actual_sha = runtime_tree_sha256(retained)
    if actual_sha != expected_sha:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_HASH_MISMATCH")
    manifest_path = retained / "wmloop-runtime-manifest.json"
    manifest = _load_json_object(manifest_path)
    manifest["runtime_path"] = str(destination / "retained_runtime")
    manifest["tree_sha256"] = actual_sha
    _write_bytes_atomic(manifest_path, _canonical_json_bytes(manifest))
    receipt["candidate_runtime_retained"] = True
    receipt["candidate_runtime_root"] = str(destination / "retained_runtime")
    receipt["candidate_runtime_manifest_path"] = str(
        destination / "retained_runtime" / "wmloop-runtime-manifest.json"
    )
    receipt["candidate_runtime_sha256"] = actual_sha


def _bundle_checkpoint_ladder(
    *,
    source: Path,
    retained_dir: Path,
    destination: Path,
    train_steps: object,
) -> list[dict[str, object]]:
    if isinstance(train_steps, bool) or not isinstance(train_steps, int) or train_steps < 1:
        return []
    checkpoint_files: dict[int, Path] = {}
    for checkpoint in source.parent.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", checkpoint.name)
        if match is not None and checkpoint.is_file() and not checkpoint.is_symlink():
            checkpoint_files[int(match.group(1))] = checkpoint
    final_absolute_step = max(checkpoint_files, default=-1)
    base_step = final_absolute_step - train_steps if final_absolute_step >= train_steps else None
    records: list[dict[str, object]] = []
    ladder_dir = retained_dir / "checkpoints"
    ladder_dir.mkdir(mode=0o700)
    for relative_step in checkpoint_eval_ladder(train_steps):
        absolute_step = base_step + relative_step if base_step is not None else None
        checkpoint_source = checkpoint_files.get(absolute_step) if absolute_step is not None else None
        if checkpoint_source is None and relative_step == train_steps:
            checkpoint_source = source
        if checkpoint_source is None:
            records.append(
                {
                    "relative_step": relative_step,
                    "absolute_step": absolute_step,
                    "state": "checkpoint_missing",
                    "retained_path": None,
                }
            )
            continue
        retained_name = f"relative_step_{relative_step:06d}.pt"
        retained_checkpoint = ladder_dir / retained_name
        storage_mode = _link_or_copy_checkpoint(checkpoint_source, retained_checkpoint)
        records.append(
            {
                "relative_step": relative_step,
                "absolute_step": absolute_step,
                "state": "retained",
                "source_path": str(checkpoint_source),
                "retained_path": str(destination / "retained_training" / "checkpoints" / retained_name),
                "bytes": retained_checkpoint.stat().st_size,
                "storage_mode": storage_mode,
                "sha256": None,
                "hash_state": "deferred_until_official_gate",
            }
        )
    return records


def _link_or_copy_checkpoint(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(report: Mapping[str, Any]) -> str:
    receipt = report["receipt"]
    evaluation = receipt["evaluation"]
    metric_name = str(evaluation.get("primary_metric") or "auc_psnr_16_64")
    rows = []
    for attempt in receipt["candidate"]["receipts"]:
        rows.append(
            f"| {attempt['label']} | {attempt['passed']} | {attempt['timed_out']} | "
            f"{attempt['exit_code']} | {attempt['duration_seconds']:.3f} |"
        )
    lines = [
        f"# M3 {report['environment']} Training+Eval Smoke",
        "",
        f"State: `{report['state']}`",
        f"Proposal: `{report['proposal_id']}`",
        f"GPU index: `{report['gpu_index']}`",
        f"Verdict: `{report['verdict']}`",
        f"Primary metric: `{metric_name}`",
        f"Delta {metric_name}: `{evaluation.get('delta_primary_metric', evaluation['delta_auc_psnr_16_64'])}`",
        f"Replication count: `{evaluation.get('replication_count', 1)}`",
        f"Replication deltas: `{evaluation.get('replication_deltas', [evaluation['delta_auc_psnr_16_64']])}`",
        f"Action following observed: `{evaluation.get('action_following', {}).get('gate_observed')}`",
        f"Action following threshold: `{evaluation.get('action_following', {}).get('gate_threshold')}`",
        f"Settlement: `{report['settlement_state']}`",
        f"Worktree removed: `{receipt['worktree_removed']}`",
        "",
        "| Check | Passed | Timed out | Exit code | Seconds |",
        "|:--|:--|:--|:--|--:|",
        *rows,
        "",
        "## Limitations",
        "",
    ]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_JSON_INVALID")
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
