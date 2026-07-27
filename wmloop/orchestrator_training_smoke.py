"""M3 one-round orchestrator smoke that launches ACWM training in the sandbox."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import load_yaml_document
from wmloop.execute.agent_staging import AgentRepairSession
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.gpu_sampling import GpuSamplingRecorder
from wmloop.execute.primitive_runtime_smoke import (
    _HOOK_UNIT_SCRIPT,
    _runtime_environment,
    _training_config,
    _validate_training_assets,
)
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.orchestrator import ExecutionOutcome, ResearchLoop
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION, ProposalContext, ProposalGenerator
from wmloop.propose.llm_logging import seal_llm_call_log
from wmloop.verify.judge import VerificationEvidence
from wmloop.verify.m4_launch_guard import phase_gate_guard
from wmloop.verify.round_start_guard import round_start_guard
from wmloop.vendor import verify_vendor_checkout


class OrchestratorTrainingSmokeError(RuntimeError):
    """The M3 training-backed smoke could not produce trustworthy evidence."""


def run_push_cube_training_smoke(
    *,
    repo_root: Path,
    failure_report: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu_index: int = 1,
    weight: float = 0.2,
    run_training: bool = True,
    hook_timeout_seconds: float = 60.0,
    training_timeout_seconds: float = 1200.0,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    m4_phase_gate_manifest: Path | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_OUTPUT_EXISTS")
    m4_guard = phase_gate_guard(m4_phase_gate_manifest) if m4_phase_gate_manifest is not None else None
    m4_authorization = m4_guard() if m4_guard is not None else None
    runtime = Path(runtime_python).expanduser().absolute()
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    if gpu_index < 0:
        raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_GPU_INVALID")
    gpu_exclusivity = None
    if run_training:
        gpu_exclusivity = verify_gpu_exclusivity_ready(
            gpu_exclusivity_audit_manifest,
            gpu_index=gpu_index,
            max_age_seconds=gpu_exclusivity_max_age_seconds,
        )
        _validate_training_assets(runtime=runtime, data_root=data, checkpoint_root=checkpoints)
    failure = _load_json_object(failure_report)
    if failure.get("env") != "push_cube" or failure.get("goal_id") != "g1_long_horizon":
        raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_FAILURE_SCOPE_INVALID")
    goal = load_yaml_document(goal_config)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
        cas = ContentAddressedStore(cas_storage_root)
        executor = _TrainingSmokeExecutor(
            repo_root=root,
            runs_root=temporary / "sandbox-runs",
            runtime_python=runtime,
            data_root=data,
            checkpoint_root=checkpoints,
            gpu_index=gpu_index,
            weight=weight,
            run_training=run_training,
            output_root=temporary,
            cas=cas,
            archive=archive,
            hook_timeout_seconds=hook_timeout_seconds,
            training_timeout_seconds=training_timeout_seconds,
        )
        budget = BudgetLedger(temporary / "budget.db", BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])))
        loop = ResearchLoop(
            proposal_generator=ProposalGenerator(_TrainingSmokeProposalClient(weight=weight)),
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
        verdict_ref = _put_json(cas, verdict, archive=archive)
        execution_receipt = executor.receipt(proposal_id=str(proposal["proposal_id"]))
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-training-smoke-report",
            "state": "ready" if execution_receipt["state"] == "ready" else "checks_failed",
            "environment": "push_cube",
            "goal_id": goal["goal_id"],
            "proposal_id": proposal["proposal_id"],
            "proposal_ref": proposal_ref,
            "prompt_ref": llm_call_log["prompt_ref"],
            "llm_call_log": llm_call_log,
            "receipt_ref": result.execution.receipt_ref,
            "verdict_ref": verdict_ref,
            "verdict": verdict["verdict"],
            "violation": verdict["violation"],
            "settlement_state": result.settlement.state,
            "actual_gpu_hours": result.execution.actual_gpu_hours,
            "round_start_verification": result.round_start_verification,
            "run_training": run_training,
            "runtime_python": str(runtime),
            "data_root": str(data),
            "checkpoint_root": str(checkpoints),
            "gpu_index": gpu_index,
            "gpu_exclusivity_audit": gpu_exclusivity,
            "receipt": execution_receipt,
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "m4_launch_gate": m4_authorization,
            "limitations": [
                "This is a one-step provider training smoke for orchestration connectivity, not a model-quality fine-tune.",
                "The verifier evidence intentionally reports zero metric improvement, so the round is expected to be REJECT.",
                "Smoke artifacts are CAS-indexed but are not published as model-quality settled trials in the main archive.",
            ],
        }
        return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive, budget_db=temporary / "budget.db")
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)


class _TrainingSmokeExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        runs_root: Path,
        runtime_python: Path,
        data_root: Path,
        checkpoint_root: Path,
        gpu_index: int,
        weight: float,
        run_training: bool,
        output_root: Path,
        cas: ContentAddressedStore,
        archive: ArchiveStore | None,
        hook_timeout_seconds: float,
        training_timeout_seconds: float,
    ) -> None:
        self._repo_root = repo_root
        self._runs_root = runs_root
        self._runtime_python = runtime_python
        self._data_root = data_root
        self._checkpoint_root = checkpoint_root
        self._gpu_index = gpu_index
        self._weight = weight
        self._run_training = run_training
        self._output_root = output_root
        self._cas = cas
        self._archive = archive
        self._hook_timeout = hook_timeout_seconds
        self._training_timeout = training_timeout_seconds
        self._receipts: dict[str, dict[str, object]] = {}
        self._gpu_sampler = GpuSamplingRecorder(gpu_index=gpu_index, sample_interval_seconds=1.0)

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
                raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_INTERVENTIONS_INVALID")
            rendered = renderer.render_checked(worktree=lease.worktree, interventions=interventions, hook_ios=_default_hook_ios())
            for item in rendered:
                _apply_diff(lease.worktree, item.diff)
            training_config_path = self._output_root / f"{proposal_id}-train-smoke.yaml"
            _write_bytes_atomic(
                training_config_path,
                _canonical_json_bytes(
                    _training_config(
                        self._checkpoint_root,
                        primitive="latent_motion_prior",
                        weight=self._weight,
                    )
                ),
            )
            required = ["runtime_hook_unit"]
            if self._run_training:
                required.append("acwm_train_smoke")
            session = AgentRepairSession(
                worktree=lease.worktree,
                staging_root=self._runs_root / proposal_id / "staging",
                candidate_id="m3-training-smoke",
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
            )
            self._gpu_sampler.capture(
                "runtime_hook_unit",
                lambda: session.run(
                    label="runtime_hook_unit",
                    argv=(str(self._runtime_python), "-c", _HOOK_UNIT_SCRIPT),
                    timeout_seconds=self._hook_timeout,
                ),
            )
            if self._run_training:
                self._gpu_sampler.capture(
                    "acwm_train_smoke",
                    lambda: session.run(
                        label="acwm_train_smoke",
                        argv=(str(self._runtime_python), "train.py", "--config", str(training_config_path)),
                        timeout_seconds=self._training_timeout,
                    ),
                )
            candidate = session.seal()
            sandbox.remove(lease)
            worktree_removed = True
            attempts = _cas_attempts(candidate.manifest_path.parent / "attempts", cas=self._cas, archive=self._archive)
            gpu_sampling = self._gpu_sampler.to_document()
            receipt = {
                "schema_version": 1,
                "artifact_type": "wmloop-m3-training-smoke-execution-receipt",
                "proposal_id": proposal_id,
                "fencing_token": fencing_token,
                "source_revision": source_revision,
                "registry_digest": registry.digest(),
                "runtime_python": str(self._runtime_python),
                "gpu_index": self._gpu_index,
                "run_training": self._run_training,
                "worktree_removed": worktree_removed,
                "rendered_primitives": [{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
                "training_config_ref": _put_file(self._cas, training_config_path, archive=self._archive, media_type="application/json"),
                "candidate": candidate.to_document(),
                "candidate_manifest_ref": _put_file(self._cas, candidate.manifest_path, archive=self._archive, media_type="application/json"),
                "candidate_diff_ref": _put_file(self._cas, candidate.diff_path, archive=self._archive, media_type="text/plain"),
                "attempt_refs": attempts,
                "gpu_sampling": gpu_sampling,
                "gpu_sampling_ref": _put_json(self._cas, gpu_sampling, archive=self._archive),
                "state": "ready" if candidate.ready_for_promotion else "checks_failed",
                "actual_gpu_hours": _receipt_gpu_hours(candidate.to_document()) if self._run_training else 0.0,
            }
            receipt_ref = _put_json(self._cas, receipt, archive=self._archive)
            self._receipts[proposal_id] = {**receipt, "receipt_ref": receipt_ref}
            return ExecutionOutcome(
                actual_gpu_hours=float(receipt["actual_gpu_hours"]),
                receipt_ref=receipt_ref,
                verification_evidence=VerificationEvidence(
                    proposal_id=proposal_id,
                    readonly_evaluator_verified=True,
                    accept_split_verified=True,
                    extended_horizon_verified=True,
                    diff_audit_passed=candidate.ready_for_promotion,
                    evidence_complete=True,
                    accept_metric_deltas={"auc_psnr_16_64": 0.0},
                    replication_deltas=[0.0, 0.0, 0.0],
                    action_following_observed=0.0,
                    action_following_threshold=0.0,
                ),
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
            raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_RECEIPT_MISSING") from exc


class _TrainingSmokeProposalClient:
    def __init__(self, *, weight: float) -> None:
        self._weight = weight

    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        manifest = allowed.get("latent_motion_prior")
        if manifest is None:
            raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_LATENT_MOTION_PRIOR_UNAVAILABLE")
        proposal = {
            "proposal_id": "m3-push-cube-training-smoke-latent_motion_prior-r1",
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [
                {"layer": manifest["layer"], "primitive": "latent_motion_prior", "params": {"weight": self._weight}}
            ],
            "falsifiable_prediction": {
                "metric": "auc_psnr_16_64",
                "horizon": 64,
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(manifest["estimated_gpu_hours"]),
            "rationale_ref": f"training_m3_smoke#{failure['dominant_failure']}->latent_motion_prior",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


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
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "orchestrator-training-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "orchestrator-training-smoke.md", markdown_bytes)
        shutil.copy2(budget_db, temporary / "budget.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(budget_db.read_bytes(), media_type="application/x-sqlite3").uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, budget_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-training-smoke-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "proposal_id": report["proposal_id"],
            "run_training": report["run_training"],
            "verdict": report["verdict"],
            "gpu_exclusivity_audit": report.get("gpu_exclusivity_audit"),
            "report_path": str(destination / "orchestrator-training-smoke.json"),
            "markdown_path": str(destination / "orchestrator-training-smoke.md"),
            "budget_db_path": str(destination / "budget.db"),
            "cas_refs": {
                "orchestrator_training_smoke_json": report_ref,
                "orchestrator_training_smoke_markdown": markdown_ref,
                "budget_db": budget_ref,
            },
            "limitations": report["limitations"],
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
    receipt = report["receipt"]
    rows = []
    for attempt in receipt["candidate"]["receipts"]:
        rows.append(
            f"| {attempt['label']} | {attempt['passed']} | {attempt['timed_out']} | "
            f"{attempt['exit_code']} | {attempt['duration_seconds']:.3f} |"
        )
    lines = [
        "# M3 Push Cube Training Orchestrator Smoke",
        "",
        f"State: `{report['state']}`",
        f"Proposal: `{report['proposal_id']}`",
        f"Run training: `{report['run_training']}`",
        f"GPU index: `{report['gpu_index']}`",
        f"Verdict: `{report['verdict']}`",
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
        raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise OrchestratorTrainingSmokeError("M3_TRAINING_SMOKE_JSON_INVALID")
    return payload


def _cas_attempts(attempts_root: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, str]:
    refs: dict[str, str] = {}
    if not attempts_root.is_dir():
        return refs
    for member in sorted(path for path in attempts_root.rglob("*") if path.is_file()):
        key = member.relative_to(attempts_root).as_posix().replace("/", "_").replace(".", "_")
        media_type = "application/json" if member.suffix == ".json" else "text/plain"
        refs[key] = _put_file(cas, member, archive=archive, media_type=media_type)
    return refs


def _receipt_gpu_hours(candidate: Mapping[str, object]) -> float:
    receipts = candidate.get("receipts")
    if not isinstance(receipts, list):
        return 0.0
    seconds = 0.0
    for receipt in receipts:
        if isinstance(receipt, Mapping):
            duration = receipt.get("duration_seconds")
            if isinstance(duration, (float, int)):
                seconds += float(duration)
    return seconds / 3600.0


def _put_json(cas: ContentAddressedStore, payload: Mapping[str, object], *, archive: ArchiveStore | None) -> str:
    ref = cas.put_bytes(_canonical_json_bytes(payload), media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref


def _put_file(cas: ContentAddressedStore, path: Path, *, archive: ArchiveStore | None, media_type: str) -> str:
    ref = cas.put_bytes(path.read_bytes(), media_type=media_type).uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref


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
    run = commands.add_parser("run", help="run one training-backed push_cube M3 smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--failure-report", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--gpu-index", type=int, default=1)
    run.add_argument("--weight", type=float, default=0.2)
    run.add_argument("--run-training", action="store_true")
    run.add_argument("--hook-timeout-seconds", type=float, default=60.0)
    run.add_argument("--training-timeout-seconds", type=float, default=1200.0)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_push_cube_training_smoke(
            repo_root=args.repo_root,
            failure_report=args.failure_report,
            goal_config=args.goal_config,
            output_root=args.output_root,
            runtime_python=args.runtime_python,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            gpu_index=args.gpu_index,
            weight=args.weight,
            run_training=args.run_training,
            hook_timeout_seconds=args.hook_timeout_seconds,
            training_timeout_seconds=args.training_timeout_seconds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
