"""T4.2 manual-proposal preflight packet.

The packet stages a human-supplied proposal against the same proposal contract,
registry, goal config, and M4 launch gate used by the automated loop.  It is
deliberately read-only: it never admits, executes, settles, or debits a trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveSelection, PrimitiveValidationError
from wmloop.propose.generator import (
    CURRENT_LIBRARY_VERSION,
    ProposalContext,
    ProposalGenerationError,
    validate_proposal_for_context,
)
from wmloop.verify.m4_launch_guard import M4LaunchGuardError, verify_m4_launch_allowed


class ManualProposalPacketError(RuntimeError):
    """Manual proposal packet generation failed before a durable packet existed."""


def run_manual_proposal_packet(
    *,
    repo_root: Path,
    proposal_path: Path,
    failure_report: Path,
    goal_config: Path,
    phase_gate_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    library_version: str = CURRENT_LIBRARY_VERSION,
) -> dict[str, object]:
    """Validate and stage one manual proposal without launching work."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ManualProposalPacketError("MANUAL_PROPOSAL_PACKET_OUTPUT_EXISTS")
    if not library_version:
        raise ManualProposalPacketError("MANUAL_PROPOSAL_LIBRARY_VERSION_INVALID")

    proposal_source, proposal, proposal_bytes = _load_json_mapping(
        proposal_path,
        "MANUAL_PROPOSAL_JSON_INVALID",
    )
    failure_source, failure, failure_bytes = _load_json_mapping(
        failure_report,
        "MANUAL_PROPOSAL_FAILURE_REPORT_INVALID",
    )
    goal_source = Path(goal_config).resolve(strict=True)
    goal_bytes = goal_source.read_bytes()
    phase_gate_source = Path(phase_gate_manifest).resolve(strict=True)
    phase_gate_bytes = phase_gate_source.read_bytes()

    goal = _load_goal(goal_source)
    registry = PrimitiveRegistry.from_root(root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"

    validation = _validate_manual_proposal(
        proposal=proposal,
        failure=failure,
        goal=goal,
        registry=registry,
        library_version=library_version,
    )
    phase_gate = _phase_gate_status(phase_gate_source)
    blockers = list(validation["blockers"])
    if phase_gate["m4_launch_allowed"] is not True:
        blockers.append(
            {
                "stage": "phase_gate",
                "reason": phase_gate.get("error") or "M4_LAUNCH_GATE_NOT_READY",
                "phase_gate_manifest": str(phase_gate_source),
            }
        )

    proposal_valid = validation["proposal_validation_passed"] is True
    m4_ready = phase_gate["m4_launch_allowed"] is True
    if proposal_valid and m4_ready:
        state = "staged_ready"
        manual_proposal_executor_ready = True
    elif proposal_valid:
        state = "staged_blocked"
        manual_proposal_executor_ready = False
    else:
        state = "blocked"
        manual_proposal_executor_ready = False

    side_effects = {
        "llm_or_autosearch_proposal": False,
        "automatic_search_budget_debited": False,
        "trial_admitted": False,
        "trial_executed": False,
        "trial_settled": False,
        "gpu_execution_started": False,
        "packet_grants_m4_launch_permission": False,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-t4-2-manual-proposal-packet",
        "state": state,
        "scope": "T4.2/E8 manual-proposal preflight",
        "manual_proposal_executor_ready": manual_proposal_executor_ready,
        "proposal_validation_passed": proposal_valid,
        "m4_launch_allowed_by_phase_gate": m4_ready,
        "packet_grants_m4_launch_permission": False,
        "proposal_id": proposal.get("proposal_id"),
        "env": proposal.get("env"),
        "round": proposal.get("round"),
        "goal_id": proposal.get("goal_id"),
        "based_on_failure": proposal.get("based_on_failure"),
        "library_version": proposal.get("library_version"),
        "registry_digest": registry.digest(),
        "input_paths": {
            "proposal": str(proposal_source),
            "failure_report": str(failure_source),
            "goal_config": str(goal_source),
            "phase_gate_manifest": str(phase_gate_source),
        },
        "input_sha256": {
            "proposal": hashlib.sha256(proposal_bytes).hexdigest(),
            "failure_report": hashlib.sha256(failure_bytes).hexdigest(),
            "goal_config": hashlib.sha256(goal_bytes).hexdigest(),
            "phase_gate_manifest": hashlib.sha256(phase_gate_bytes).hexdigest(),
        },
        "validation": validation,
        "phase_gate": phase_gate,
        "side_effects": side_effects,
        "blockers": blockers,
        "limitations": [
            "This packet validates a human-supplied proposal; it is not an LLM or automatic-search proposal.",
            "The packet is not counted against automatic search budget and records no trial admission or settlement.",
            "The packet performs no GPU work and launches no training, evaluation, or M4 campaign process.",
            "The packet does not grant M4 launch permission; executors must verify the strict M4 guard separately.",
        ],
    }

    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir(mode=0o700)

        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        refs = {
            "manual_proposal_packet_json": cas.put_bytes(report_bytes, media_type="application/json").uri,
            "manual_proposal_packet_markdown": cas.put_bytes(markdown_bytes, media_type="text/markdown").uri,
            "proposal": cas.put_bytes(proposal_bytes, media_type="application/json").uri,
            "failure_report": cas.put_bytes(failure_bytes, media_type="application/json").uri,
            "goal_config": cas.put_bytes(goal_bytes, media_type="application/yaml").uri,
            "phase_gate_manifest": cas.put_bytes(phase_gate_bytes, media_type="application/json").uri,
        }
        if archive is not None:
            for ref in refs.values():
                archive.record_artifact_reference(ref)

        _write_bytes_atomic(temporary / "manual-proposal-packet.json", report_bytes)
        _write_bytes_atomic(temporary / "manual-proposal-packet.md", markdown_bytes)
        _write_bytes_atomic(inputs_dir / "proposal.json", proposal_bytes)
        _write_bytes_atomic(inputs_dir / "failure-report.json", failure_bytes)
        _write_bytes_atomic(inputs_dir / "goal-config.yaml", goal_bytes)
        _write_bytes_atomic(inputs_dir / "phase-gate-manifest.json", phase_gate_bytes)

        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-t4-2-manual-proposal-packet-manifest",
            "state": state,
            "scope": report["scope"],
            "manual_proposal_executor_ready": manual_proposal_executor_ready,
            "proposal_validation_passed": proposal_valid,
            "m4_launch_allowed_by_phase_gate": m4_ready,
            "packet_grants_m4_launch_permission": False,
            "proposal_id": proposal.get("proposal_id"),
            "env": proposal.get("env"),
            "round": proposal.get("round"),
            "report_path": str(destination / "manual-proposal-packet.json"),
            "markdown_path": str(destination / "manual-proposal-packet.md"),
            "input_snapshot_dir": str(destination / "inputs"),
            "cas_refs": refs,
            "side_effects": side_effects,
            "blockers": blockers,
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


def _validate_manual_proposal(
    *,
    proposal: Mapping[str, Any],
    failure: Mapping[str, Any],
    goal: Mapping[str, Any],
    registry: PrimitiveRegistry,
    library_version: str,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    selections: tuple[PrimitiveSelection, ...] = ()
    registered_minimum = 0.0
    max_cost_class = None
    try:
        validate_document("failure_report", failure)
        validate_document("goal_spec", goal)
        validate_proposal_for_context(
            proposal,
            ProposalContext(
                failure_report=failure,
                goal_spec=goal,
                archive_statistics={},
                registry=registry,
                library_version=library_version,
            ),
        )
        _require_goal_environment(proposal=proposal, goal=goal)
        selections = registry.validate_combination(
            [(str(row["primitive"]), row["params"]) for row in proposal["interventions"]]
        )
        registered_minimum = sum(selection.estimated_gpu_hours for selection in selections)
        max_cost_class = _max_cost_class(selections)
        _require_goal_budget(proposal=proposal, goal=goal)
    except (
        ContractValidationError,
        PrimitiveValidationError,
        ProposalGenerationError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        blockers.append(
            {
                "stage": "proposal_validation",
                "reason": _reason(exc),
                "detail": str(exc),
            }
        )

    return {
        "proposal_validation_passed": not blockers,
        "registered_minimum_gpu_hours": registered_minimum,
        "proposal_budget_estimate_gpu_hours": _numeric(proposal.get("budget_estimate_gpu_hours")),
        "goal_per_trial_max_gpu_hours": _goal_per_trial_max(goal),
        "cost_classes": [selection.cost_class for selection in selections],
        "max_cost_class": max_cost_class,
        "primitive_names": [selection.name for selection in selections],
        "hook_bindings": {
            selection.name: {hook: int(selection.hook_order[hook]) for hook in selection.hooks}
            for selection in selections
        },
        "blockers": blockers,
    }


def _phase_gate_status(path: Path) -> dict[str, object]:
    try:
        authorization = verify_m4_launch_allowed(path)
    except M4LaunchGuardError as exc:
        return {
            "state": "blocked",
            "m4_launch_allowed": False,
            "error": str(exc),
            "phase_gate_manifest": str(path),
        }
    return {
        "state": "ready",
        "m4_launch_allowed": True,
        "error": None,
        "authorization": authorization.to_document(),
        "phase_gate_manifest": str(path),
    }


def _require_goal_environment(*, proposal: Mapping[str, Any], goal: Mapping[str, Any]) -> None:
    envs = goal.get("envs")
    if not isinstance(envs, list) or proposal.get("env") not in envs:
        raise ManualProposalPacketError("MANUAL_PROPOSAL_ENV_NOT_IN_ACTIVE_GOAL")


def _require_goal_budget(*, proposal: Mapping[str, Any], goal: Mapping[str, Any]) -> None:
    estimate = _numeric(proposal.get("budget_estimate_gpu_hours"))
    cap = _goal_per_trial_max(goal)
    if estimate is None or cap is None or estimate > cap:
        raise ManualProposalPacketError("MANUAL_PROPOSAL_GOAL_BUDGET_CAP_EXCEEDED")


def _goal_per_trial_max(goal: Mapping[str, Any]) -> float | None:
    budget = goal.get("budget")
    if not isinstance(budget, Mapping):
        return None
    return _numeric(budget.get("per_trial_max_gpu_hours"))


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _max_cost_class(selections: Sequence[PrimitiveSelection]) -> str | None:
    rank = {"very_low": 0, "low": 1, "medium": 2, "high": 3}
    if not selections:
        return None
    return max((selection.cost_class for selection in selections), key=lambda value: rank[value])


def _load_json_mapping(path: Path, error_code: str) -> tuple[Path, Mapping[str, Any], bytes]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualProposalPacketError(error_code) from exc
    if not isinstance(payload, Mapping):
        raise ManualProposalPacketError(error_code)
    return resolved, payload, payload_bytes


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        return load_yaml_document(path)
    except (OSError, ContractValidationError) as exc:
        raise ManualProposalPacketError("MANUAL_PROPOSAL_GOAL_CONFIG_INVALID") from exc


def _reason(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return message.split(":", 1)[0]
    return type(exc).__name__


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# T4.2 Manual Proposal Packet",
        "",
        f"State: `{report['state']}`",
        f"Proposal: `{report.get('proposal_id')}`",
        f"Environment: `{report.get('env')}`",
        f"Round: `{report.get('round')}`",
        f"Proposal validation passed: `{report['proposal_validation_passed']}`",
        f"M4 launch allowed by phase gate: `{report['m4_launch_allowed_by_phase_gate']}`",
        f"Manual proposal executor ready: `{report['manual_proposal_executor_ready']}`",
        f"Packet grants M4 launch permission: `{report['packet_grants_m4_launch_permission']}`",
        "",
        "## Side Effects",
        "",
    ]
    side_effects = report.get("side_effects")
    if isinstance(side_effects, Mapping):
        for key, value in side_effects.items():
            lines.append(f"- {key}: `{value}`")
    blockers = report.get("blockers")
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:  # type: ignore[assignment]
            lines.append(f"- `{blocker}`")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:  # type: ignore[index]
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ManualProposalPacketError("MANUAL_PROPOSAL_PACKET_OUTPUT_EXISTS")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write a T4.2 manual-proposal preflight packet")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--proposal", type=Path, required=True)
    run.add_argument("--failure-report", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--library-version", default=CURRENT_LIBRARY_VERSION)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_manual_proposal_packet(
                repo_root=args.repo_root,
                proposal_path=args.proposal,
                failure_report=args.failure_report,
                goal_config=args.goal_config,
                phase_gate_manifest=args.phase_gate_manifest,
                output_root=args.output_root,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                library_version=args.library_version,
            )
            print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            return 0
        raise ManualProposalPacketError("MANUAL_PROPOSAL_PACKET_COMMAND_INVALID")
    except ManualProposalPacketError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
