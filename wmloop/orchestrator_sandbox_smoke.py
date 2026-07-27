"""M3 Push Cube orchestrator smoke with a real git-worktree sandbox executor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
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
from wmloop.verify.judge import VerificationEvidence
from wmloop.verify.m4_launch_guard import phase_gate_guard
from wmloop.verify.round_start_guard import round_start_guard
from wmloop.vendor import verify_vendor_checkout


class OrchestratorSandboxSmokeError(RuntimeError):
    """Sandbox-backed M3 smoke failed closed."""


def run_push_cube_sandbox_smoke(
    *,
    repo_root: Path,
    failure_report: Path,
    goal_config: Path,
    output_root: Path,
    rounds: int = 5,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    command_timeout_seconds: float = 30.0,
    m4_phase_gate_manifest: Path | None = None,
) -> dict[str, object]:
    if rounds < 5:
        raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_ROUNDS_TOO_SMALL")
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_OUTPUT_EXISTS")
    m4_guard = phase_gate_guard(m4_phase_gate_manifest) if m4_phase_gate_manifest is not None else None
    m4_authorization = m4_guard() if m4_guard is not None else None
    base_failure = _load_json_object(failure_report)
    if base_failure.get("env") != "push_cube" or base_failure.get("goal_id") != "g1_long_horizon":
        raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_FAILURE_REPORT_SCOPE_INVALID")
    goal = load_yaml_document(goal_config)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
        cas = ContentAddressedStore(cas_storage_root)
        executor = _SandboxSmokeExecutor(
            repo_root=root,
            runs_root=temporary / "sandbox-runs",
            cas=cas,
            archive=archive,
            command_timeout_seconds=command_timeout_seconds,
        )
        budget = BudgetLedger(temporary / "budget.db", BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])))
        loop = ResearchLoop(
            proposal_generator=ProposalGenerator(_SandboxSmokeProposalClient()),
            budget_ledger=budget,
            executor=executor,
            vendor_verifier=lambda: verify_vendor_checkout(root),
            m4_launch_guard=m4_guard,
            round_start_guard=invariant_guard,
        )
        contexts = [
            ProposalContext(
                failure_report={**base_failure, "round": index},
                goal_spec=goal,
                archive_statistics=_archive_statistics(archive),
                registry=registry,
            )
            for index in range(rounds)
        ]
        campaign = CampaignOrchestrator(loop, parallel_slots=1, m4_launch_guard=m4_guard).run(contexts)
        if campaign.paused or campaign.remaining_contexts or len(campaign.completed) != rounds:
            raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_CAMPAIGN_INCOMPLETE")
        round_records = []
        for ordinal, result in enumerate(campaign.completed, start=1):
            proposal = dict(result.generated_proposal.proposal)
            verdict = result.verdict.to_dict()
            proposal_ref = _put_json(cas, proposal, archive=archive)
            llm_call_log = seal_llm_call_log(result.generated_proposal, cas=cas, archive=archive)
            verdict_ref = _put_json(cas, verdict, archive=archive)
            execution_receipt = executor.receipt(proposal_id=str(proposal["proposal_id"]))
            round_records.append(
                {
                    "ordinal": ordinal,
                    "proposal_id": proposal["proposal_id"],
                    "proposal_ref": proposal_ref,
                    "prompt_ref": llm_call_log["prompt_ref"],
                    "llm_call_log": llm_call_log,
                    "receipt_ref": result.execution.receipt_ref,
                    "receipt": execution_receipt,
                    "verdict_ref": verdict_ref,
                    "verdict": verdict["verdict"],
                    "violation": verdict["violation"],
                    "fencing_token": result.admission.fencing_token,
                    "settlement_state": result.settlement.state,
                    "actual_gpu_hours": result.execution.actual_gpu_hours,
                    "round_start_verification": result.round_start_verification,
                    "interventions": proposal["interventions"],
                    "attempts": result.generated_proposal.attempts,
                }
            )
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-sandbox-smoke-report",
            "state": "ready",
            "environment": "push_cube",
            "goal_id": goal["goal_id"],
            "rounds_requested": rounds,
            "completed_round_count": len(round_records),
            "verdict_counts": _counts(record["verdict"] for record in round_records),
            "m4_launch_gate": m4_authorization,
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "rounds": round_records,
            "limitations": [
                "This smoke executes real git-worktree sandbox apply/check/seal/remove flow, but does not launch provider training or evaluation.",
                "The executor returns zero metric improvement, so every round is expected to be REJECT.",
                "Smoke artifacts are CAS-indexed but are not published as model-quality settled trials in the main archive.",
            ],
        }
        return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive, budget_db=temporary / "budget.db")
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)


class _SandboxSmokeExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        runs_root: Path,
        cas: ContentAddressedStore,
        archive: ArchiveStore | None,
        command_timeout_seconds: float,
    ) -> None:
        self._repo_root = repo_root
        self._runs_root = runs_root
        self._cas = cas
        self._archive = archive
        self._timeout = command_timeout_seconds
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
                raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_INTERVENTIONS_INVALID")
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
                candidate_id="m3-sandbox-smoke",
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
            receipt = {
                "schema_version": 1,
                "artifact_type": "wmloop-m3-sandbox-smoke-execution-receipt",
                "proposal_id": proposal_id,
                "fencing_token": fencing_token,
                "source_revision": source_revision,
                "registry_digest": registry.digest(),
                "worktree_removed": worktree_removed,
                "rendered_primitives": [{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
                "candidate": candidate.to_document(),
                "candidate_manifest_ref": _put_file(self._cas, candidate.manifest_path, archive=self._archive, media_type="application/json"),
                "candidate_diff_ref": _put_file(self._cas, candidate.diff_path, archive=self._archive, media_type="text/plain"),
                "state": "ready" if candidate.ready_for_promotion else "checks_failed",
                "actual_gpu_hours": 0.0,
            }
            receipt_ref = _put_json(self._cas, receipt, archive=self._archive)
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
            raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_RECEIPT_MISSING") from exc


class _SandboxSmokeProposalClient:
    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        preferred = (
            ("frontier_collection", {"condition": "ood_test", "n_episodes": 1}),
            ("mixture_reweight", {"frontier_weight": 0.1}),
            ("dino_rep_injection", {"injection_weight": 0.1}),
            ("latent_spatial_memory", {"memory_slots": 1, "memory_weight": 0.1}),
            ("latent_motion_prior", {"weight": 0.1}),
        )
        choices = [(name, params) for name, params in preferred if name in allowed]
        if not choices:
            raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_NO_ALLOWED_PRIMITIVE")
        index = int(failure["round"]) % len(choices)
        primitive, params = choices[index]
        manifest = allowed[primitive]
        proposal = {
            "proposal_id": f"m3-push-cube-sandbox-r{int(failure['round']) + 1:02d}-{primitive}",
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [{"layer": manifest["layer"], "primitive": primitive, "params": params}],
            "falsifiable_prediction": {
                "metric": "auc_psnr_16_64",
                "horizon": 64,
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(manifest["estimated_gpu_hours"]),
            "rationale_ref": f"sandbox_m3_smoke#{failure['dominant_failure']}->{primitive}",
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
        _write_bytes_atomic(temporary / "orchestrator-sandbox-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "orchestrator-sandbox-smoke.md", markdown_bytes)
        shutil.copy2(budget_db, temporary / "budget.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(budget_db.read_bytes(), media_type="application/x-sqlite3").uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, budget_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-sandbox-smoke-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "completed_round_count": report["completed_round_count"],
            "verdict_counts": report["verdict_counts"],
            "report_path": str(destination / "orchestrator-sandbox-smoke.json"),
            "markdown_path": str(destination / "orchestrator-sandbox-smoke.md"),
            "budget_db_path": str(destination / "budget.db"),
            "cas_refs": {
                "orchestrator_sandbox_smoke_json": report_ref,
                "orchestrator_sandbox_smoke_markdown": markdown_ref,
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
    lines = [
        "# M3 Push Cube Sandbox Orchestrator Smoke",
        "",
        f"State: `{report['state']}`",
        f"Completed rounds: `{report['completed_round_count']}/{report['rounds_requested']}`",
        f"Verdicts: `{report['verdict_counts']}`",
        "",
        "| Round | Proposal | Primitive | Candidate Ready | Worktree Removed | Verdict |",
        "|--:|:--|:--|:--|:--|:--|",
    ]
    for record in report["rounds"]:
        primitive = record["interventions"][0]["primitive"]
        receipt = record["receipt"]
        candidate = receipt["candidate"]
        lines.append(
            f"| {record['ordinal']} | {record['proposal_id']} | {primitive} | "
            f"{candidate['ready_for_promotion']} | {receipt['worktree_removed']} | {record['verdict']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _archive_statistics(archive: ArchiveStore | None) -> Mapping[str, int]:
    return archive.archive_statistics() if archive is not None else {}


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise OrchestratorSandboxSmokeError("M3_SANDBOX_SMOKE_JSON_INVALID")
    return payload


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run sandbox-backed push_cube M3 smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--failure-report", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--command-timeout-seconds", type=float, default=30.0)
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_push_cube_sandbox_smoke(
            repo_root=args.repo_root,
            failure_report=args.failure_report,
            goal_config=args.goal_config,
            output_root=args.output_root,
            rounds=args.rounds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            command_timeout_seconds=args.command_timeout_seconds,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
