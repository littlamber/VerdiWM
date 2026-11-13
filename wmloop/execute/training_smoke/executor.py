"""Training and frozen evaluation smoke components: executor."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.agent_staging import AgentRepairSession, CommandReceipt
from wmloop.execute.gpu_sampling import GpuSamplingRecorder
from wmloop.execute.primitive_runtime_smoke import (
    _runtime_environment,
    hook_unit_script,
)
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.evidence import (
    _action_following_records_with_refs,
    _evaluation_summary,
    _load_action_following_probe_pair,
    _load_probe_replications,
    _replication_records_with_refs,
    _verification_evidence_from_probe_replications,
    _write_accept_replicate_inputs,
)
from wmloop.execute.training_smoke.io import _canonical_json_bytes, _write_bytes_atomic
from wmloop.execute.training_smoke.metrics import _environment_spec, _receipt_gpu_hours
from wmloop.execute.training_smoke.planning import (
    _bind_primitive_training_config,
    _official_resume_training_config,
)
from wmloop.execute.training_smoke.runtime import (
    _candidate_checkpoint_path,
    _checkpoint_step,
    _copy_candidate_runtime,
    _copy_training_sidecars,
    _horizon_probe_command,
    _raw_probe_measure_command,
    _safe_run_name,
    _training_entrypoint_command,
    _validate_probe_runtime_binding,
)
from wmloop.orchestrator import ExecutionOutcome
from wmloop.orchestrator_training_smoke import _cas_attempts, _put_file, _put_json
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.vendor import verify_vendor_checkout
from wmloop.verify.acceptance_policy import VerificationPolicy


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
        verification_policy: VerificationPolicy | None = None,
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
        self._gpu_exclusivity_trial_max_age_seconds = (
            _in_trial_gpu_exclusivity_max_age_seconds(
                launch_max_age_seconds=gpu_exclusivity_max_age_seconds,
                training_timeout_seconds=training_timeout_seconds,
                eval_timeout_seconds=eval_timeout_seconds,
                hook_timeout_seconds=hook_timeout_seconds,
                replication_count=replication_count,
            )
        )
        self._weight = weight
        self._intervention_lr = intervention_lr
        self._train_steps = train_steps
        self._train_batch_size = train_batch_size
        self._train_val_batch_size = (
            train_val_batch_size
            if train_val_batch_size is not None
            else train_batch_size
        )
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
        self._verification_policy = verification_policy
        self._receipts: dict[str, dict[str, object]] = {}
        self._gpu_sampler = GpuSamplingRecorder(
            gpu_index=gpu_index, sample_interval_seconds=2.0
        )

    def execute(
        self, proposal: dict[str, object], fencing_token: int
    ) -> ExecutionOutcome:
        source_revision = verify_vendor_checkout(self._repo_root)
        registry = PrimitiveRegistry.from_root(self._repo_root)
        renderer = PrimitiveRenderer(registry)
        proposal_id = str(proposal["proposal_id"])
        sandbox = WorktreeSandbox(
            vendor_root=self._repo_root / "vendor" / "ACWM-Phys",
            runs_root=self._runs_root,
        )
        lease: SandboxLease | None = None
        worktree_removed = False
        try:
            lease = sandbox.create(
                trial_id=proposal_id, expected_revision=source_revision
            )
            interventions = proposal.get("interventions")
            if not isinstance(interventions, list):
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_INTERVENTIONS_INVALID"
                )
            rendered = renderer.render_checked(
                worktree=lease.worktree,
                interventions=interventions,
                hook_ios=_default_hook_ios(),
            )
            for item in rendered:
                _apply_diff(lease.worktree, item.diff)
            spec = _environment_spec(self._environment)
            official_checkpoint = self._checkpoint_root / spec.checkpoint_relative_path
            resume_step = _checkpoint_step(
                runtime_python=self._runtime_python, checkpoint_path=official_checkpoint
            )
            run_name = _safe_run_name(proposal_id)
            training_config_path = (
                self._output_root / f"{proposal_id}-official-resume-train.yaml"
            )
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
            _write_bytes_atomic(
                training_config_path, _canonical_json_bytes(training_config)
            )
            replicates = _write_accept_replicate_inputs(
                output_root=self._output_root,
                heldout_protocol=self._heldout_protocol,
                dataset_freeze=self._dataset_freeze,
                environment=self._environment,
                max_accept_trajectories=self._max_accept_trajectories,
                replication_count=self._replication_count,
                replication_seeds=self._verification_policy.seeds_accept if self._verification_policy is not None else None,
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
            baseline_action_probe_root = (
                self._output_root / "baseline-action-following-probe"
            )
            candidate_action_probe_root = (
                self._output_root / "candidate-action-following-probe"
            )
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
                        required_files=(
                            Path(str(replicate["view_root"]))
                            / "cohort-view-manifest.json",
                        ),
                    ),
                )
            self._gpu_sampler.capture(
                "runtime_hook_unit",
                lambda: _run_required_check(
                    session,
                    label="runtime_hook_unit",
                    argv=(
                        str(self._runtime_python),
                        "-c",
                        hook_unit_script(self._proposal_primitive),
                    ),
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
                        inverse_summary_path=self._repo_root
                        / "results"
                        / "m1"
                        / "inverse-dynamics"
                        / "summary-r1.json",
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
                        inverse_summary_path=self._repo_root
                        / "results"
                        / "m1"
                        / "inverse-dynamics"
                        / "summary-r1.json",
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
                            Path(str(replicate["baseline_probe_root"]))
                            / "manifest.json",
                            Path(str(replicate["baseline_probe_root"]))
                            / "metrics.json",
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
                            Path(str(replicate["candidate_probe_root"]))
                            / "manifest.json",
                            Path(str(replicate["candidate_probe_root"]))
                            / "metrics.json",
                        ),
                    ),
                )
            candidate = session.seal()
            candidate_runtime = _copy_candidate_runtime(
                source=lease.worktree,
                destination=self._output_root / "candidate-runtime",
                source_revision=source_revision,
                rendered_primitives=[
                    {"name": item.name, "diff_sha256": item.sha256} for item in rendered
                ],
            )
            runtime_binding = _validate_probe_runtime_binding(
                baseline_runtime_sha256=runtime_tree_sha256(
                    self._repo_root / "vendor" / "ACWM-Phys"
                ),
                candidate_runtime_sha256=candidate_runtime["tree_sha256"],
                baseline_manifest_paths=[
                    baseline_action_probe_root / "manifest.json",
                    *(
                        Path(str(item["baseline_probe_root"])) / "manifest.json"
                        for item in replicates
                    ),
                ],
                candidate_manifest_paths=[
                    candidate_action_probe_root / "manifest.json",
                    *(
                        Path(str(item["candidate_probe_root"])) / "manifest.json"
                        for item in replicates
                    ),
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
            attempts = _cas_attempts(
                candidate.manifest_path.parent / "attempts",
                cas=self._cas,
                archive=self._archive,
            )
            gpu_sampling = self._gpu_sampler.to_document()
            receipt = {
                "schema_version": 1,
                "artifact_type": "wmloop-m3-training-eval-smoke-execution-receipt",
                "proposal_id": proposal_id,
                "environment": self._environment,
                "primary_metric": self._primary_metric,
                "fencing_token": fencing_token,
                **({
                    "verification_policy_ref": f"cas://sha256/{self._verification_policy.sha256}",
                    "frozen_goal_ref": f"cas://sha256/{self._verification_policy.goal_sha256}",
                } if self._verification_policy is not None else {}),
                "source_revision": source_revision,
                "registry_digest": registry.digest(),
                "runtime_python": str(self._runtime_python),
                "gpu_index": self._gpu_index,
                "gpu_exclusivity_launch_max_age_seconds": self._gpu_exclusivity_launch_max_age_seconds,
                "gpu_exclusivity_trial_max_age_seconds": self._gpu_exclusivity_trial_max_age_seconds,
                "proposal_primitive": self._proposal_primitive,
                "proposal_params": self._proposal_params,
                **(
                    {"trial_arm": self._trial_arm}
                    if self._trial_arm is not None
                    else {}
                ),
                **({"seed": self._trial_seed} if self._trial_seed is not None else {}),
                "training_scale": {
                    "train_steps": self._train_steps,
                    "batch_size": self._train_batch_size,
                    "val_batch_size": self._train_val_batch_size,
                    "train_size": self._train_size,
                    "num_workers": self._train_num_workers,
                },
                "worktree_removed": worktree_removed,
                "rendered_primitives": [
                    {"name": item.name, "diff_sha256": item.sha256} for item in rendered
                ],
                "training_config_ref": _put_file(
                    self._cas,
                    training_config_path,
                    archive=self._archive,
                    media_type="application/json",
                ),
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
                "accept_trajectory_count": sum(
                    int(item["trajectory_count"]) for item in replicates
                ),
                "replication_count": len(replication_records),
                "replications": _replication_records_with_refs(
                    replications=replication_records,
                    cas=self._cas,
                    archive=self._archive,
                ),
                "candidate": candidate.to_document(),
                "candidate_manifest_ref": _put_file(
                    self._cas,
                    candidate.manifest_path,
                    archive=self._archive,
                    media_type="application/json",
                ),
                "candidate_diff_ref": _put_file(
                    self._cas,
                    candidate.diff_path,
                    archive=self._archive,
                    media_type="text/plain",
                ),
                "attempt_refs": attempts,
                "gpu_sampling": gpu_sampling,
                "gpu_sampling_ref": _put_json(
                    self._cas, gpu_sampling, archive=self._archive
                ),
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
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_RECEIPT_MISSING"
            ) from exc


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
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_CHECK_FAILED:{label}:{status}"
        )
    missing = [str(Path(path)) for path in required_files if not Path(path).is_file()]
    if missing:
        joined = ",".join(missing)
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_CHECK_ARTIFACT_MISSING:{label}:{joined}"
        )
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
    if (
        min(training_timeout_seconds, eval_timeout_seconds, hook_timeout_seconds) < 0
        or replication_count < 1
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_GPU_AUDIT_LIFETIME_INVALID"
        )
    probe_count = 2 + (2 * replication_count)
    lifecycle_seconds = (
        training_timeout_seconds
        + hook_timeout_seconds
        + (probe_count * eval_timeout_seconds)
        + 600.0
    )
    return max(float(launch_max_age_seconds), lifecycle_seconds)
