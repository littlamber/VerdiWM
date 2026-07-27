"""M3 deterministic Push Cube orchestrator smoke.

This runner exercises the generator -> admission/fencing -> executor receipt
-> settlement -> independent judge ordering for at least five autonomous
rounds.  It deliberately returns REJECT-quality evidence so the smoke cannot be
mistaken for a model improvement claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import load_yaml_document
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.orchestrator import CampaignOrchestrator, ExecutionOutcome, ResearchLoop
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.llm_logging import seal_llm_call_log
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION, ProposalContext, ProposalGenerator
from wmloop.verify.judge import VerificationEvidence
from wmloop.verify.m4_launch_guard import phase_gate_guard
from wmloop.verify.round_start_guard import round_start_guard
from wmloop.vendor import verify_vendor_checkout


class OrchestratorSmokeError(RuntimeError):
    """M3 orchestrator smoke evidence could not be generated."""


def run_push_cube_orchestrator_smoke(
    *,
    repo_root: Path,
    failure_report: Path,
    goal_config: Path,
    output_root: Path,
    rounds: int = 5,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    m4_phase_gate_manifest: Path | None = None,
) -> dict[str, object]:
    if rounds < 5:
        raise OrchestratorSmokeError("M3_SMOKE_ROUNDS_TOO_SMALL")
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OrchestratorSmokeError("M3_SMOKE_OUTPUT_EXISTS")
    m4_guard = phase_gate_guard(m4_phase_gate_manifest) if m4_phase_gate_manifest is not None else None
    m4_authorization = m4_guard() if m4_guard is not None else None
    goal = load_yaml_document(goal_config)
    goal_id = _goal_id(goal)
    base_failure = _load_json_object(failure_report)
    if base_failure.get("env") != "push_cube" or base_failure.get("goal_id") != goal_id:
        raise OrchestratorSmokeError("M3_SMOKE_FAILURE_REPORT_SCOPE_INVALID")
    metric_name = _primary_metric(goal)
    prediction_horizon = _prediction_horizon(goal)
    registry = PrimitiveRegistry.from_root(root)
    invariant_guard = round_start_guard(root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
        cas = ContentAddressedStore(cas_storage_root)
        executor = _DeterministicRejectExecutor(cas=cas, archive=archive, metric_name=metric_name)
        budget = BudgetLedger(temporary / "budget.db", BudgetPolicy(total_gpu_hours=float(goal["budget"]["total_gpu_hours"])))
        loop = ResearchLoop(
            proposal_generator=ProposalGenerator(
                _DeterministicProposalClient(metric_name=metric_name, prediction_horizon=prediction_horizon)
            ),
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
        campaign = CampaignOrchestrator(loop, parallel_slots=2, m4_launch_guard=m4_guard).run(contexts)
        if campaign.paused or campaign.remaining_contexts or len(campaign.completed) != rounds:
            raise OrchestratorSmokeError("M3_SMOKE_CAMPAIGN_INCOMPLETE")
        round_records = []
        for ordinal, result in enumerate(campaign.completed, start=1):
            proposal = dict(result.generated_proposal.proposal)
            verdict = result.verdict.to_dict()
            proposal_ref = _put_json(cas, proposal, archive=archive)
            llm_call_log = seal_llm_call_log(result.generated_proposal, cas=cas, archive=archive)
            verdict_ref = _put_json(cas, verdict, archive=archive)
            round_records.append(
                {
                    "ordinal": ordinal,
                    "proposal_id": proposal["proposal_id"],
                    "proposal_ref": proposal_ref,
                    "prompt_ref": llm_call_log["prompt_ref"],
                    "llm_call_log": llm_call_log,
                    "receipt_ref": result.execution.receipt_ref,
                    "verdict_ref": verdict_ref,
                    "verdict": verdict["verdict"],
                    "violation": verdict["violation"],
                    "fencing_token": result.admission.fencing_token,
                    "settlement_state": result.settlement.state,
                    "actual_gpu_hours": result.execution.actual_gpu_hours,
                    "round_start_verification": result.round_start_verification,
                    "interventions": proposal["interventions"],
                    "raw_response_count": llm_call_log["raw_response_count"],
                    "attempts": result.generated_proposal.attempts,
                }
            )
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-orchestrator-smoke-report",
            "state": "ready",
            "environment": "push_cube",
            "goal_id": goal["goal_id"],
            "rounds_requested": rounds,
            "completed_round_count": len(round_records),
            "verdict_counts": _counts(record["verdict"] for record in round_records),
            "budget_visible_settled_trials": list(budget.visible_settled_trial_ids()),
            "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
            "cas_root": str(Path(cas_storage_root).resolve()),
            "m4_launch_gate": m4_authorization,
            "rounds": round_records,
            "limitations": [
                "Deterministic smoke executor returns zero improvement, so all rounds are expected to be REJECT.",
                "Smoke artifacts are CAS-indexed but are not published as model-quality settled trials in the main archive.",
                "This uses the provided push_cube ready raw failure report; M1 coverage is enforced by the separate raw-failure batch input.",
            ],
        }
        return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive, budget_db=temporary / "budget.db")
    except Exception:
        raise
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)


class _DeterministicProposalClient:
    def __init__(self, *, metric_name: str, prediction_horizon: int) -> None:
        self._metric_name = metric_name
        self._prediction_horizon = prediction_horizon

    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        preferred = (
            ("frontier_collection", {"condition": "ood_test", "n_episodes": 1}),
            ("mixture_reweight", {"frontier_weight": 0.1}),
            ("latent_motion_prior", {"weight": 0.1}),
            ("dino_rep_injection", {"injection_weight": 0.1}),
            ("wmsd_self_distill", {"teacher_ema": 0.9, "steps": 1, "lr": 0.00001}),
            ("latent_spatial_memory", {"memory_slots": 1, "memory_weight": 0.1}),
        )
        choices = [(name, params) for name, params in preferred if name in allowed]
        if not choices:
            raise OrchestratorSmokeError("M3_SMOKE_NO_ALLOWED_PRIMITIVE")
        index = int(failure["round"]) % len(choices)
        primitive, params = choices[index]
        manifest = allowed[primitive]
        proposal = {
            "proposal_id": f"m3-push-cube-smoke-r{int(failure['round']) + 1:02d}-{primitive}",
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [{"layer": manifest["layer"], "primitive": primitive, "params": params}],
            "falsifiable_prediction": {
                "metric": self._metric_name,
                "horizon": self._prediction_horizon,
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(manifest["estimated_gpu_hours"]),
            "rationale_ref": f"deterministic_m3_smoke#{failure['dominant_failure']}->{primitive}",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class _DeterministicRejectExecutor:
    def __init__(self, *, cas: ContentAddressedStore, archive: ArchiveStore | None, metric_name: str) -> None:
        self._cas = cas
        self._archive = archive
        self._metric_name = metric_name

    def execute(self, proposal: dict[str, object], fencing_token: int) -> ExecutionOutcome:
        proposal_id = str(proposal["proposal_id"])
        receipt = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-smoke-execution-receipt",
            "proposal_id": proposal_id,
            "fencing_token": fencing_token,
            "state": "settled_smoke_reject",
            "actual_gpu_hours": 0.0,
            "side_effects": [],
            "note": "deterministic orchestrator smoke; no provider training launched",
        }
        receipt_ref = _put_json(self._cas, receipt, archive=self._archive)
        return ExecutionOutcome(
            actual_gpu_hours=0.0,
            receipt_ref=receipt_ref,
            verification_evidence=VerificationEvidence(
                proposal_id=proposal_id,
                readonly_evaluator_verified=True,
                accept_split_verified=True,
                extended_horizon_verified=True,
                diff_audit_passed=True,
                evidence_complete=True,
                accept_metric_deltas={self._metric_name: 0.0},
                replication_deltas=[0.0, 0.0, 0.0],
                action_following_observed=0.0,
                action_following_threshold=0.0,
            ),
        )


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
        _write_bytes_atomic(temporary / "orchestrator-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "orchestrator-smoke.md", markdown_bytes)
        shutil.copy2(budget_db, temporary / "budget.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(budget_db.read_bytes(), media_type="application/x-sqlite3").uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, budget_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-push-cube-orchestrator-smoke-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "completed_round_count": report["completed_round_count"],
            "verdict_counts": report["verdict_counts"],
            "report_path": str(destination / "orchestrator-smoke.json"),
            "markdown_path": str(destination / "orchestrator-smoke.md"),
            "budget_db_path": str(destination / "budget.db"),
            "cas_refs": {
                "orchestrator_smoke_json": report_ref,
                "orchestrator_smoke_markdown": markdown_ref,
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
        "# M3 Push Cube Orchestrator Smoke",
        "",
        f"State: `{report['state']}`",
        f"Completed rounds: `{report['completed_round_count']}/{report['rounds_requested']}`",
        f"Verdicts: `{report['verdict_counts']}`",
        "",
        "| Round | Proposal | Verdict | Primitive | Settlement |",
        "|--:|:--|:--|:--|:--|",
    ]
    for record in report["rounds"]:
        primitive = record["interventions"][0]["primitive"]
        lines.append(
            f"| {record['ordinal']} | {record['proposal_id']} | {record['verdict']} | "
            f"{primitive} | {record['settlement_state']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _archive_statistics(archive: ArchiveStore | None) -> Mapping[str, int]:
    if archive is None:
        return {}
    return archive.archive_statistics()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorSmokeError("M3_SMOKE_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise OrchestratorSmokeError("M3_SMOKE_JSON_INVALID")
    return payload


def _goal_id(goal: Mapping[str, Any]) -> str:
    value = goal.get("goal_id")
    if not isinstance(value, str) or not value:
        raise OrchestratorSmokeError("M3_SMOKE_GOAL_INVALID")
    return value


def _primary_metric(goal: Mapping[str, Any]) -> str:
    value = goal.get("primary_objective")
    if not isinstance(value, str) or not value:
        raise OrchestratorSmokeError("M3_SMOKE_GOAL_INVALID")
    return value


def _prediction_horizon(goal: Mapping[str, Any]) -> int:
    values = goal.get("horizons")
    if not isinstance(values, list) or not values:
        return 64
    parsed = [int(value) for value in values]
    if any(value < 1 for value in parsed):
        raise OrchestratorSmokeError("M3_SMOKE_GOAL_INVALID")
    return max(parsed)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run deterministic push_cube M3 smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--failure-report", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_push_cube_orchestrator_smoke(
            repo_root=args.repo_root,
            failure_report=args.failure_report,
            goal_config=args.goal_config,
            output_root=args.output_root,
            rounds=args.rounds,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
