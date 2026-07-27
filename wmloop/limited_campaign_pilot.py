"""Scoped multi-environment closed-loop pilot for non-formal campaigns.

The pilot is intentionally not an M4 launcher.  It consumes a ready
``limited_campaign_gate`` manifest and exercises the same proposal -> admission
-> sandbox receipt -> settlement -> judge path used by formal rounds.  Model-quality claims remain
out of scope; every pilot trial is recorded in a separate pilot archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.contracts import load_yaml_document
from wmloop.execute.agent_staging import AgentRepairSession
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios, _sidecar_check_script
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.orchestrator import CampaignOrchestrator, ExecutionOutcome, ResearchLoop
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION, ProposalContext, ProposalGenerator
from wmloop.propose.llm_logging import seal_llm_call_log
from wmloop.propose.scheduler import InterventionCell
from wmloop.verify.judge import VerificationEvidence
from wmloop.verify.round_start_guard import round_start_guard
from wmloop.vendor import verify_vendor_checkout


class LimitedCampaignPilotError(RuntimeError):
    """The scoped limited campaign pilot failed closed."""


def run_limited_campaign_pilot(
    *,
    repo_root: Path,
    failure_report_manifest: Path,
    limited_gate_manifest: Path,
    goal_config: Path,
    output_root: Path,
    rounds_per_env: int = 1,
    parallel_slots: int = 1,
    cas_root: Path | None = None,
    archive_db: Path | None = None,
    command_timeout_seconds: float = 30.0,
    campaign_id: str | None = None,
) -> dict[str, object]:
    """Run a sandbox-backed closed-loop pilot over the gate's included envs."""

    if rounds_per_env < 1 or parallel_slots < 1:
        raise LimitedCampaignPilotError("LIMITED_PILOT_ARGUMENT_INVALID")
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise LimitedCampaignPilotError("LIMITED_PILOT_OUTPUT_EXISTS")
    gate = _load_json_object(limited_gate_manifest, "LIMITED_PILOT_GATE_INVALID")
    included_envs, excluded_envs = _validate_limited_gate(gate)
    checkpoint_policy = _checkpoint_policy(gate)
    goal = load_yaml_document(goal_config)
    goal_id = _required_string(goal.get("goal_id"), "LIMITED_PILOT_GOAL_INVALID")
    failure_manifest = _load_json_object(failure_report_manifest, "LIMITED_PILOT_FAILURE_MANIFEST_INVALID")
    failure_paths = _failure_report_paths(
        manifest=failure_manifest,
        included_envs=included_envs,
        goal_id=goal_id,
    )
    pilot_id = _campaign_id(campaign_id or destination.name)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        cas_storage_root = Path(cas_root).resolve() if cas_root is not None else destination.parent
        cas = ContentAddressedStore(cas_storage_root)
        artifact_archive = ArchiveStore(archive_db) if archive_db is not None else None
        pilot_archive_path = temporary / "pilot-archive.db"
        pilot_archive = ArchiveStore(pilot_archive_path)
        executor = _LimitedSandboxExecutor(
            repo_root=root,
            runs_root=temporary / "sandbox-runs",
            cas=cas,
            artifact_archive=artifact_archive,
            command_timeout_seconds=command_timeout_seconds,
            campaign_id=pilot_id,
        )
        budget = BudgetLedger(temporary / "budget.db", BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])))
        loop = ResearchLoop(
            proposal_generator=ProposalGenerator(_LimitedCampaignProposalClient(campaign_id=pilot_id)),
            budget_ledger=budget,
            executor=executor,
            vendor_verifier=lambda: verify_vendor_checkout(root),
            round_start_guard=invariant_guard,
        )
        context_by_key: dict[tuple[str, int], ProposalContext] = {}
        contexts: list[ProposalContext] = []
        for env in included_envs:
            base_failure = _load_failure_report(failure_paths[env], env=env, goal_id=goal_id)
            for round_index in range(rounds_per_env):
                context = ProposalContext(
                    failure_report={**base_failure, "round": round_index},
                    goal_spec=goal,
                    archive_statistics=pilot_archive.archive_statistics(),
                    registry=registry,
                )
                contexts.append(context)
                context_by_key[(env, round_index)] = context
        campaign = CampaignOrchestrator(loop, parallel_slots=parallel_slots).run(contexts)
        if campaign.paused or campaign.remaining_contexts or len(campaign.completed) != len(contexts):
            raise LimitedCampaignPilotError("LIMITED_PILOT_CAMPAIGN_INCOMPLETE")
        round_records = []
        for ordinal, result in enumerate(campaign.completed, start=1):
            proposal = dict(result.generated_proposal.proposal)
            verdict = result.verdict.to_dict()
            env = str(proposal["env"])
            round_index = int(proposal["round"])
            context = context_by_key[(env, round_index)]
            failure_context_ref = _put_json(cas, context.failure_report, archive=artifact_archive)
            proposal_ref = _put_json(cas, proposal, archive=artifact_archive)
            llm_call_log = seal_llm_call_log(result.generated_proposal, cas=cas, archive=artifact_archive)
            verdict_ref = _put_json(cas, verdict, archive=artifact_archive)
            receipt = executor.receipt(proposal_id=str(proposal["proposal_id"]))
            record = {
                "ordinal": ordinal,
                "environment": env,
                "proposal_id": proposal["proposal_id"],
                "failure_context_ref": failure_context_ref,
                "proposal_ref": proposal_ref,
                "prompt_ref": llm_call_log["prompt_ref"],
                "llm_call_log": llm_call_log,
                "receipt_ref": result.execution.receipt_ref,
                "receipt": receipt,
                "verdict_ref": verdict_ref,
                "verdict": verdict["verdict"],
                "violation": verdict["violation"],
                "gates": verdict["gates"],
                "action_following_gate": verdict["action_following_gate"],
                "delta_m_ver": verdict["delta_m_ver"],
                "fencing_token": result.admission.fencing_token,
                "settlement_state": result.settlement.state,
                "actual_gpu_hours": result.execution.actual_gpu_hours,
                "round_start_verification": result.round_start_verification,
                "interventions": proposal["interventions"],
                "attempts": result.generated_proposal.attempts,
                "pilot_archive_trial_recorded": _record_pilot_trial(
                    pilot_archive=pilot_archive,
                    proposal=proposal,
                    verdict=verdict,
                    receipt=receipt,
                    receipt_ref=result.execution.receipt_ref,
                    failure_context_ref=failure_context_ref,
                    verdict_ref=verdict_ref,
                    gpu_hours=result.execution.actual_gpu_hours,
                    settlement_state=result.settlement.state,
                    evaluator_hash=_evaluator_hash(result.round_start_verification),
                ),
            }
            _record_artifact_refs(
                pilot_archive,
                refs=(
                    failure_context_ref,
                    proposal_ref,
                    str(llm_call_log["prompt_ref"]),
                    result.execution.receipt_ref,
                    verdict_ref,
                ),
            )
            round_records.append(record)
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-limited-campaign-pilot-report",
            "state": "ready",
            "scope_type": "limited_pilot",
            "campaign_id": pilot_id,
            "goal_id": goal_id,
            "included_envs": list(included_envs),
            "excluded_envs": list(excluded_envs),
            "checkpoint_policy": checkpoint_policy,
            "rounds_per_env": rounds_per_env,
            "rounds_requested": len(contexts),
            "completed_round_count": len(round_records),
            "parallel_slots": parallel_slots,
            "verdict_counts": _counts(record["verdict"] for record in round_records),
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "pilot_archive_visible_settled_trials": list(pilot_archive.visible_settled_trials()),
            "pilot_archive_statistics": pilot_archive.archive_statistics(),
            "artifact_archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(cas_storage_root),
            "limited_gate_manifest": str(Path(limited_gate_manifest).resolve()),
            "failure_report_manifest": str(Path(failure_report_manifest).resolve()),
            "goal_config": str(Path(goal_config).resolve()),
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "gpu_execution_started": False,
            "rounds": round_records,
            "limitations": [
                "This is a scoped limited pilot, not a formal M4 launch authorization.",
                "The sandbox executor applies and seals primitive patches but does not run model training or formal evaluation.",
                "All verdicts use zero-improvement evidence and are expected to be REJECT, so no model-quality claim is made.",
                "Settled pilot trials are written only to the copied pilot archive DB, not to the formal campaign trial projection.",
            ],
        }
        return _write_report_bundle(
            report=report,
            output_root=destination,
            cas=cas,
            artifact_archive=artifact_archive,
            pilot_archive=pilot_archive,
            budget_db=temporary / "budget.db",
            pilot_archive_db=pilot_archive_path,
        )
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)


class _LimitedCampaignProposalClient:
    def __init__(self, *, campaign_id: str) -> None:
        self._campaign_id = campaign_id

    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        primitive, params = _choose_primitive(failure=failure, allowed=allowed)
        manifest = allowed[primitive]
        proposal = {
            "proposal_id": f"{self._campaign_id}-{failure['env']}-r{int(failure['round']) + 1:03d}-{primitive}",
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": packet["goal_spec"]["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [{"layer": manifest["layer"], "primitive": primitive, "params": params}],
            "falsifiable_prediction": {
                "metric": packet["goal_spec"].get("primary_objective", "ladder_auc_psnr_envmax"),
                "horizon": _max_observed_horizon(failure),
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(manifest["estimated_gpu_hours"]),
            "rationale_ref": f"limited_campaign_pilot#{failure['dominant_failure']}->{primitive}",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class _LimitedSandboxExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        runs_root: Path,
        cas: ContentAddressedStore,
        artifact_archive: ArchiveStore | None,
        command_timeout_seconds: float,
        campaign_id: str,
    ) -> None:
        self._repo_root = repo_root
        self._runs_root = runs_root
        self._cas = cas
        self._artifact_archive = artifact_archive
        self._timeout = command_timeout_seconds
        self._campaign_id = campaign_id
        self._receipts: dict[str, dict[str, object]] = {}

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
                raise LimitedCampaignPilotError("LIMITED_PILOT_INTERVENTIONS_INVALID")
            rendered = renderer.render_checked(
                worktree=lease.worktree,
                interventions=interventions,
                hook_ios=_default_hook_ios(),
            )
            for item in rendered:
                _apply_diff(lease.worktree, item.diff)
            names = [item.name for item in rendered]
            session = AgentRepairSession(
                worktree=lease.worktree,
                staging_root=self._runs_root / proposal_id / "staging",
                candidate_id=f"{self._campaign_id}-candidate",
                source_revision=source_revision,
                registry_digest=registry.digest(),
                required_check_labels=("git_status", "sidecar_json"),
            )
            session.run(label="git_status", argv=("git", "status", "--short"), timeout_seconds=self._timeout)
            session.run(
                label="sidecar_json",
                argv=(sys.executable, "-c", _sidecar_check_script(names)),
                timeout_seconds=self._timeout,
            )
            candidate = session.seal()
            sandbox.remove(lease)
            worktree_removed = True
            attempt_refs = _cas_attempts(candidate.manifest_path.parent / "attempts", cas=self._cas, archive=self._artifact_archive)
            receipt = {
                "schema_version": 1,
                "artifact_type": "wmloop-limited-campaign-sandbox-receipt",
                "proposal_id": proposal_id,
                "fencing_token": fencing_token,
                "source_revision": source_revision,
                "registry_digest": registry.digest(),
                "worktree_removed": worktree_removed,
                "rendered_primitives": [{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
                "candidate": candidate.to_document(),
                "candidate_manifest_ref": _put_file(self._cas, candidate.manifest_path, archive=self._artifact_archive, media_type="application/json"),
                "candidate_diff_ref": _put_file(self._cas, candidate.diff_path, archive=self._artifact_archive, media_type="text/plain"),
                "attempt_refs": attempt_refs,
                "state": "ready" if candidate.ready_for_promotion else "checks_failed",
                "actual_gpu_hours": 0.0,
                "side_effects": ["sandbox_worktree_created", "primitive_patch_applied", "candidate_sealed", "sandbox_worktree_removed"],
            }
            receipt_ref = _put_json(self._cas, receipt, archive=self._artifact_archive)
            self._receipts[proposal_id] = {**receipt, "receipt_ref": receipt_ref}
            return ExecutionOutcome(
                actual_gpu_hours=0.0,
                receipt_ref=receipt_ref,
                verification_evidence=VerificationEvidence(
                    proposal_id=proposal_id,
                    readonly_evaluator_verified=True,
                    accept_split_verified=True,
                    extended_horizon_verified=True,
                    diff_audit_passed=candidate.ready_for_promotion,
                    evidence_complete=True,
                    accept_metric_deltas={"ladder_auc_psnr_envmax": 0.0},
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
            raise LimitedCampaignPilotError("LIMITED_PILOT_RECEIPT_MISSING") from exc


def _validate_limited_gate(gate: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if (
        gate.get("artifact_type") != "wmloop-limited-campaign-gate-manifest"
        or gate.get("state") != "ready"
        or gate.get("scope_type") != "limited_pilot"
        or gate.get("limited_campaign_allowed") is not True
        or gate.get("m4_launch_allowed") is not False
        or gate.get("formal_training_allowed") is not False
    ):
        raise LimitedCampaignPilotError("LIMITED_PILOT_GATE_NOT_READY")
    included = _string_tuple(gate.get("included_envs"), "LIMITED_PILOT_INCLUDED_ENVS_INVALID")
    excluded = _string_tuple(gate.get("excluded_envs", []), "LIMITED_PILOT_EXCLUDED_ENVS_INVALID")
    if not included or set(included) & set(excluded):
        raise LimitedCampaignPilotError("LIMITED_PILOT_SCOPE_INVALID")
    return included, excluded


def _checkpoint_policy(gate: Mapping[str, Any]) -> dict[str, object]:
    value = gate.get("checkpoint_policy")
    if value is None:
        return {
            "allow_official_current_checkpoint_warning": False,
            "claim_boundary": "Every included environment must pass the strict checkpoint-step audit.",
        }
    if not isinstance(value, Mapping):
        raise LimitedCampaignPilotError("LIMITED_PILOT_CHECKPOINT_POLICY_INVALID")
    allow = value.get("allow_official_current_checkpoint_warning", False)
    if not isinstance(allow, bool):
        raise LimitedCampaignPilotError("LIMITED_PILOT_CHECKPOINT_POLICY_INVALID")
    policy = dict(value)
    policy["allow_official_current_checkpoint_warning"] = allow
    claim_boundary = policy.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        policy["claim_boundary"] = (
            "Paired-delta closed-loop trials may use official-current checkpoints with provenance warnings; "
            "official 100k reproduction claims remain disallowed for warned environments."
            if allow
            else "Every included environment must pass the strict checkpoint-step audit."
        )
    return policy


def _failure_report_paths(
    *,
    manifest: Mapping[str, Any],
    included_envs: Sequence[str],
    goal_id: str,
) -> dict[str, Path]:
    if manifest.get("state") != "ready" or manifest.get("goal_id") != goal_id:
        raise LimitedCampaignPilotError("LIMITED_PILOT_FAILURE_MANIFEST_NOT_READY")
    reports = manifest.get("reports")
    if not isinstance(reports, list):
        raise LimitedCampaignPilotError("LIMITED_PILOT_FAILURE_MANIFEST_INVALID")
    paths: dict[str, Path] = {}
    for item in reports:
        if not isinstance(item, Mapping):
            raise LimitedCampaignPilotError("LIMITED_PILOT_FAILURE_MANIFEST_INVALID")
        env = item.get("environment")
        path = item.get("failure_report_path")
        if isinstance(env, str) and isinstance(path, str):
            paths[env] = Path(path).resolve(strict=True)
    missing = sorted(set(included_envs) - set(paths))
    if missing:
        raise LimitedCampaignPilotError(f"LIMITED_PILOT_FAILURE_REPORT_MISSING:{','.join(missing)}")
    return {env: paths[env] for env in included_envs}


def _load_failure_report(path: Path, *, env: str, goal_id: str) -> dict[str, Any]:
    payload = _load_json_object(path, "LIMITED_PILOT_FAILURE_REPORT_INVALID")
    if payload.get("env") != env or payload.get("goal_id") != goal_id:
        raise LimitedCampaignPilotError("LIMITED_PILOT_FAILURE_REPORT_SCOPE_INVALID")
    return payload


def _choose_primitive(
    *,
    failure: Mapping[str, Any],
    allowed: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, object]]:
    worst_ood = "ood_test"
    ood_profile = failure.get("ood_profile")
    if isinstance(ood_profile, Mapping) and isinstance(ood_profile.get("worst_ood_condition"), str):
        worst_ood = str(ood_profile["worst_ood_condition"])
    preferences: tuple[tuple[str, Mapping[str, object]], ...] = (
        ("frontier_collection", {"condition": worst_ood, "n_episodes": 1}),
        ("cfg_guidance_schedule", {"guidance_start": 0.5, "guidance_end": 1.0}),
        ("drift_token_trim", {"keep_tokens": 64}),
        ("mixture_reweight", {"frontier_weight": 0.1}),
        ("first_frame_anchor", {"anchor_every": 8, "anchor_weight": 0.2}),
        ("history_noise_schedule", {"history_noise": 0.1}),
        ("dino_rep_injection", {"injection_weight": 0.1}),
        ("latent_spatial_memory", {"memory_slots": 1, "memory_weight": 0.1}),
        ("latent_motion_prior", {"weight": 0.1}),
        ("inv_dyn_reward_finetune", {"reward_weight": 0.1, "steps": 1, "lr": 0.00001}),
        ("self_forcing_finetune", {"rollout_horizon": 16, "steps": 1, "lr": 0.00001}),
        ("next_forcing", {"chunks": 2, "steps": 1, "lr": 0.00001}),
        ("wmsd_self_distill", {"teacher_ema": 0.9, "steps": 1, "lr": 0.00001}),
    )
    for name, params in preferences:
        if name in allowed:
            return name, params
    raise LimitedCampaignPilotError("LIMITED_PILOT_NO_ALLOWED_PRIMITIVE")


def _record_pilot_trial(
    *,
    pilot_archive: ArchiveStore,
    proposal: Mapping[str, Any],
    verdict: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_ref: str,
    failure_context_ref: str,
    verdict_ref: str,
    gpu_hours: float,
    settlement_state: str,
    evaluator_hash: str,
) -> bool:
    first = proposal["interventions"][0]
    cell = InterventionCell(
        environment=str(proposal["env"]),
        layer=str(first["layer"]),
        primitive_family=str(first["primitive"]),
        parameter_bucket=json.dumps(first["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )
    candidate = receipt.get("candidate")
    impl_diff_hash = ""
    if isinstance(candidate, Mapping) and isinstance(candidate.get("worktree_diff_sha256"), str):
        impl_diff_hash = str(candidate["worktree_diff_sha256"])
    if not _is_sha256(impl_diff_hash):
        impl_diff_hash = _digest_from_cas_ref(str(receipt["candidate_diff_ref"]))
    pilot_archive.record_settled_trial(
        SettledTrialRecord(
            trial_id=str(proposal["proposal_id"]),
            proposal_id=str(proposal["proposal_id"]),
            goal_id=str(proposal["goal_id"]),
            library_version=str(proposal["library_version"]),
            failure_context_ref=failure_context_ref,
            verdict_ref=verdict_ref,
            receipt_ref=receipt_ref,
            gpu_hours=float(gpu_hours),
            hypothesis_hash=hashlib.sha256(
                json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest(),
            impl_diff_hash=impl_diff_hash,
            evaluator_hash=evaluator_hash,
            settlement_state=settlement_state,
            receipt_hash=_digest_from_cas_ref(receipt_ref),
            cell=cell,
            verified_gain=None,
            exploratory=True,
        )
    )
    _ = verdict
    return True


def _record_artifact_refs(archive: ArchiveStore, *, refs: Sequence[str]) -> None:
    for ref in refs:
        archive.record_artifact_reference(ref)


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    artifact_archive: ArchiveStore | None,
    pilot_archive: ArchiveStore,
    budget_db: Path,
    pilot_archive_db: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "limited-campaign-pilot.json", report_bytes)
        _write_bytes_atomic(temporary / "limited-campaign-pilot.md", markdown_bytes)
        shutil.copy2(budget_db, temporary / "budget.db")
        shutil.copy2(pilot_archive_db, temporary / "pilot-archive.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(budget_db.read_bytes(), media_type="application/x-sqlite3").uri
        pilot_archive_ref = cas.put_bytes(pilot_archive_db.read_bytes(), media_type="application/x-sqlite3").uri
        for archive in (artifact_archive, pilot_archive):
            if archive is not None:
                for ref in (report_ref, markdown_ref, budget_ref, pilot_archive_ref):
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-limited-campaign-pilot-manifest",
            "state": report["state"],
            "scope_type": report["scope_type"],
            "campaign_id": report["campaign_id"],
            "goal_id": report["goal_id"],
            "included_envs": report["included_envs"],
            "excluded_envs": report["excluded_envs"],
            "checkpoint_policy": report["checkpoint_policy"],
            "completed_round_count": report["completed_round_count"],
            "verdict_counts": report["verdict_counts"],
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "gpu_execution_started": False,
            "report_path": str(destination / "limited-campaign-pilot.json"),
            "markdown_path": str(destination / "limited-campaign-pilot.md"),
            "budget_db_path": str(destination / "budget.db"),
            "pilot_archive_db_path": str(destination / "pilot-archive.db"),
            "cas_refs": {
                "limited_campaign_pilot_json": report_ref,
                "limited_campaign_pilot_markdown": markdown_ref,
                "budget_db": budget_ref,
                "pilot_archive_db": pilot_archive_ref,
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
    lines = [
        "# Limited Campaign Pilot",
        "",
        f"State: `{report['state']}`",
        f"Scope: `{report['scope_type']}`",
        f"Goal: `{report['goal_id']}`",
        f"Completed rounds: `{report['completed_round_count']}/{report['rounds_requested']}`",
        f"Verdicts: `{report['verdict_counts']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        f"Formal training allowed: `{report['formal_training_allowed']}`",
        "",
        "| Round | Env | Proposal | Primitive | Candidate Ready | Worktree Removed | Verdict |",
        "|--:|:--|:--|:--|:--|:--|:--|",
    ]
    for record in report["rounds"]:
        primitive = record["interventions"][0]["primitive"]
        receipt = record["receipt"]
        candidate = receipt["candidate"]
        lines.append(
            f"| {record['ordinal']} | {record['environment']} | {record['proposal_id']} | "
            f"{primitive} | {candidate['ready_for_promotion']} | {receipt['worktree_removed']} | {record['verdict']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _cas_attempts(attempt_root: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, str]:
    refs: dict[str, str] = {}
    for path in sorted(attempt_root.glob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lstrip(".")
        stem = path.stem
        media_type = "application/json" if suffix == "json" else "text/plain"
        refs[f"{stem}_{suffix}"] = _put_file(cas, path, archive=archive, media_type=media_type)
    return refs


def _put_file(cas: ContentAddressedStore, path: Path, *, archive: ArchiveStore | None, media_type: str) -> str:
    ref = cas.put_bytes(path.read_bytes(), media_type=media_type).uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref


def _put_json(cas: ContentAddressedStore, payload: Mapping[str, object], *, archive: ArchiveStore | None) -> str:
    ref = cas.put_bytes(_canonical_json_bytes(payload), media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref


def _counts(values: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


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


def _load_json_object(path: Path, error_code: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LimitedCampaignPilotError(error_code) from exc
    if not isinstance(payload, dict):
        raise LimitedCampaignPilotError(error_code)
    return payload


def _string_tuple(value: object, error_code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise LimitedCampaignPilotError(error_code)
    return tuple(value)


def _required_string(value: object, error_code: str) -> str:
    if not isinstance(value, str) or not value:
        raise LimitedCampaignPilotError(error_code)
    return value


def _campaign_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"limited-{cleaned}"
    return cleaned[:96]


def _max_observed_horizon(failure: Mapping[str, Any]) -> int:
    curve = failure.get("horizon_curve")
    psnr = curve.get("psnr") if isinstance(curve, Mapping) else None
    if isinstance(psnr, Mapping):
        values = []
        for key in psnr:
            try:
                values.append(int(key))
            except (TypeError, ValueError):
                continue
        if values:
            return max(values)
    return 32


def _evaluator_hash(round_start: object) -> str:
    if isinstance(round_start, Mapping):
        evaluator = round_start.get("evaluator_freeze")
        if isinstance(evaluator, Mapping):
            digest = evaluator.get("sha256")
            if isinstance(digest, str) and _is_sha256(digest):
                return digest
    return hashlib.sha256(b"limited-campaign-pilot:evaluator").hexdigest()


def _digest_from_cas_ref(ref: str) -> str:
    prefix = "cas://sha256/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise LimitedCampaignPilotError("LIMITED_PILOT_CAS_REF_INVALID")
    digest = ref[len(prefix) :]
    if not _is_sha256(digest):
        raise LimitedCampaignPilotError("LIMITED_PILOT_CAS_REF_INVALID")
    return digest


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a scoped limited campaign pilot")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--failure-report-manifest", type=Path, required=True)
    run.add_argument("--limited-gate-manifest", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--rounds-per-env", type=int, default=1)
    run.add_argument("--parallel-slots", type=int, default=1)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--command-timeout-seconds", type=float, default=30.0)
    run.add_argument("--campaign-id")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_limited_campaign_pilot(
            repo_root=args.repo_root,
            failure_report_manifest=args.failure_report_manifest,
            limited_gate_manifest=args.limited_gate_manifest,
            goal_config=args.goal_config,
            output_root=args.output_root,
            rounds_per_env=args.rounds_per_env,
            parallel_slots=args.parallel_slots,
            cas_root=args.cas_root,
            archive_db=args.archive_db,
            command_timeout_seconds=args.command_timeout_seconds,
            campaign_id=args.campaign_id,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
