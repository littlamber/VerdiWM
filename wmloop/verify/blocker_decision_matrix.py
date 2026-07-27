"""Route phase-gate blockers into safe next actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class BlockerDecisionMatrixError(RuntimeError):
    """A blocker decision matrix could not be produced."""


def run_blocker_decision_matrix(
    *,
    phase_gate_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only routing report for a strict phase-gate result."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BlockerDecisionMatrixError("BLOCKER_DECISION_MATRIX_OUTPUT_EXISTS")
    phase_manifest, phase_manifest_bytes, phase_manifest_path = _read_json(
        phase_gate_manifest,
        "BLOCKER_DECISION_MATRIX_PHASE_MANIFEST_INVALID",
    )
    phase_report = _load_report_from_manifest(
        phase_manifest,
        "BLOCKER_DECISION_MATRIX_PHASE_REPORT_INVALID",
    )
    blockers = _blockers(phase_report)
    routes = [_route_blocker(blocker) for blocker in blockers]
    m4_launch_allowed = (
        phase_report.get("artifact_type") == "wmloop-strict-phase-gate"
        and phase_report.get("state") == "ready"
        and phase_report.get("m4_launch_allowed") is True
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-blocker-decision-matrix",
        "state": "ready",
        "phase_gate_state": phase_report.get("state"),
        "m4_launch_allowed": m4_launch_allowed,
        "formal_training_policy": _formal_training_policy(m4_launch_allowed=m4_launch_allowed),
        "blocker_count": len(blockers),
        "routes": routes,
        "route_counts": _route_counts(routes),
        "next_safe_actions": _next_safe_actions(routes=routes, m4_launch_allowed=m4_launch_allowed),
        "source_phase_gate": {
            "manifest_path": str(phase_manifest_path),
            "manifest_sha256": hashlib.sha256(phase_manifest_bytes).hexdigest(),
            "report_path": str(Path(str(phase_manifest["report_path"])).resolve(strict=True)),
            "state": phase_manifest.get("state"),
            "m4_launch_allowed": phase_manifest.get("m4_launch_allowed"),
        },
        "limitations": [
            "This matrix is read-only and does not launch training, evaluation, or protocol changes.",
            "GPU availability is intentionally not treated as M4 authorization; the phase gate is authoritative.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _formal_training_policy(*, m4_launch_allowed: bool) -> dict[str, object]:
    if m4_launch_allowed:
        return {
            "formal_m4_training_allowed": True,
            "gpu_policy": "formal_m4_gpu_training_allowed_by_phase_gate",
            "reason": "The strict phase gate reports ready and m4_launch_allowed=true.",
        }
    return {
        "formal_m4_training_allowed": False,
        "gpu_policy": "formal_m4_gpu_training_forbidden_until_phase_gate_ready",
        "reason": "The strict phase gate is not ready; free GPUs may only be used by explicitly allowed smoke or diagnostic commands.",
    }


def _route_blocker(blocker: Mapping[str, Any]) -> dict[str, object]:
    requirement = str(blocker.get("requirement", "UNKNOWN_REQUIREMENT"))
    observed = blocker.get("observed") if isinstance(blocker.get("observed"), Mapping) else {}
    route = _route_for(requirement=requirement, observed=observed)
    return {
        "requirement": requirement,
        "blocking_authority": route["blocking_authority"],
        "codex_can_advance_without_human": route["codex_can_advance_without_human"],
        "requires_human_approval": route["requires_human_approval"],
        "requires_external_asset": route["requires_external_asset"],
        "formal_training_allowed_before_resolution": False,
        "gpu_policy": route["gpu_policy"],
        "next_step": route["next_step"],
        "observed_summary": _observed_summary(observed),
    }


def _route_for(*, requirement: str, observed: Mapping[str, Any]) -> dict[str, object]:
    if requirement == "M0_baseline_reproduction":
        if observed.get("state") == "checkpoint_step_mismatch":
            return _route(
                "external_checkpoint_required",
                False,
                False,
                True,
                "no_gpu_until_checkpoint_quarantine_passes",
                "Find a verified 100k-step cloth_move checkpoint, validate it in quarantine, then rerun M0 reproduction.",
            )
        return _route(
            "codex_reproduction_audit",
            True,
            False,
            False,
            "cpu_or_guarded_eval_only",
            "Inspect the M0 reproduction discrepancy and rerun only the guarded reproduction audit once inputs are correct.",
        )
    if requirement == "M0_checkpoint_source_resolution":
        return _route(
            "external_checkpoint_required",
            False,
            False,
            True,
            "no_gpu_until_checkpoint_quarantine_passes",
            "Provide or locate an alternate verified 100k-step checkpoint candidate and run checkpoint_quarantine before install.",
        )
    if requirement == "M0_checkpoint_launch_guard_clear":
        return _route(
            "upstream_checkpoint_guard",
            False,
            False,
            True,
            "baseline_and_m4_gpu_launch_forbidden",
            "Keep launch blocked until checkpoint source resolution is re-audited clean.",
        )
    if requirement == "M1_raw_failure_reports":
        if _has_horizon_protocol_blocker(observed):
            return _route(
                "human_protocol_decision",
                False,
                True,
                False,
                "no_gpu_for_dataset_length_limited_horizons",
                "Choose a protocol/data path before trying to turn the blocked environments into formal failure reports.",
            )
        return _route(
            "codex_evidence_rerun",
            True,
            False,
            False,
            "diagnostic_gpu_allowed_only_under_m1_commands",
            "Regenerate missing raw failure reports under the frozen protocol and archive the schema-checked outputs.",
        )
    if requirement == "M1_original_g1_data_feasibility":
        return _route(
            "human_protocol_decision",
            False,
            True,
            False,
            "no_gpu_until_protocol_path_selected",
            "Obtain human approval for a data/protocol path or keep the affected G1 records formally blocked.",
        )
    if requirement == "M1_attribution_review":
        return _route(
            "human_review_required",
            False,
            True,
            False,
            "no_gpu_needed",
            "Complete the required attribution review/signoff under the chosen protocol.",
        )
    if requirement == "M3_strict_acceptance":
        return _route(
            "upstream_m1_evidence_required",
            False,
            False,
            False,
            "no_direct_gpu_rerun",
            "Rerun M3 acceptance only after M1 supplies enough real failure_report inputs.",
        )
    if requirement == "M0_generation_zero_archive":
        return _route(
            "codex_archive_repair",
            True,
            False,
            False,
            "cpu_only",
            "Repair or import generation-zero archive records with receipts before rerunning the phase gate.",
        )
    if requirement == "M2_vendor_and_registry_freeze":
        return _route(
            "codex_freeze_audit",
            True,
            False,
            False,
            "cpu_only",
            "Repair vendor/registry freeze evidence without mutating vendor or active campaign semantics.",
        )
    return _route(
        "manual_triage_required",
        False,
        True,
        False,
        "no_gpu_until_triaged",
        "Inspect the blocker and classify it before launching further GPU work.",
    )


def _route(
    blocking_authority: str,
    codex_can_advance_without_human: bool,
    requires_human_approval: bool,
    requires_external_asset: bool,
    gpu_policy: str,
    next_step: str,
) -> dict[str, object]:
    return {
        "blocking_authority": blocking_authority,
        "codex_can_advance_without_human": codex_can_advance_without_human,
        "requires_human_approval": requires_human_approval,
        "requires_external_asset": requires_external_asset,
        "gpu_policy": gpu_policy,
        "next_step": next_step,
    }


def _has_horizon_protocol_blocker(observed: Mapping[str, Any]) -> bool:
    blocked_records = observed.get("blocked_records")
    if not isinstance(blocked_records, list):
        return False
    for record in blocked_records:
        if not isinstance(record, Mapping):
            continue
        blockers = record.get("blockers")
        if isinstance(blockers, list) and "horizon_unavailable_by_protocol" in blockers:
            return True
    return False


def _observed_summary(observed: Mapping[str, Any]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in (
        "state",
        "strict_m0_t03_pass",
        "report_count",
        "blocked_count",
        "reviewed_count",
        "expectation_mismatch_count",
        "mismatch_count",
        "replacement_candidate_count",
        "remote_unreachable_count",
        "strict_m3_pass",
    ):
        if key in observed:
            summary[key] = observed[key]
    if "unfilled_blocked_environments" in observed:
        summary["unfilled_blocked_environments"] = observed["unfilled_blocked_environments"]
    if "protocol_application" in observed:
        protocol_application = observed["protocol_application"]
        if isinstance(protocol_application, Mapping):
            summary["protocol_application_state"] = protocol_application.get("state")
    if "remediation" in observed:
        remediation = observed["remediation"]
        if isinstance(remediation, Mapping):
            summary["remediation_state"] = remediation.get("state")
            summary["remediation_signoff_allowed"] = remediation.get("signoff_allowed")
            summary["remediation_blocker_count"] = remediation.get("blocker_count")
            summary["remediation_source_hash_matches_current"] = remediation.get("source_hash_matches_current")
    if "local_candidate_inventory" in observed:
        inventory = observed["local_candidate_inventory"]
        if isinstance(inventory, Mapping):
            summary["local_candidate_inventory_state"] = inventory.get("state")
            summary["local_candidate_count"] = inventory.get("candidate_count")
    if "requirements" in observed:
        requirements = observed["requirements"]
        if isinstance(requirements, Mapping):
            summary["m3_requirement_passes"] = {
                str(name): item.get("passed")
                for name, item in requirements.items()
                if isinstance(item, Mapping)
            }
    return summary


def _route_counts(routes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        authority = str(route.get("blocking_authority", "unknown"))
        counts[authority] = counts.get(authority, 0) + 1
    return counts


def _next_safe_actions(*, routes: Sequence[Mapping[str, Any]], m4_launch_allowed: bool) -> list[str]:
    if m4_launch_allowed:
        return ["M4 gate is open; launch formal settled trials through guarded training/orchestrator entrypoints."]
    actions = ["Do not launch M4 formal training until a regenerated phase gate reports m4_launch_allowed=true."]
    authorities = {str(route.get("blocking_authority")) for route in routes}
    if "external_checkpoint_required" in authorities or "upstream_checkpoint_guard" in authorities:
        actions.append("Resolve the cloth_move checkpoint source with a verified 100k candidate and quarantine validation before any M0/M4 GPU evaluator launch.")
    if "human_protocol_decision" in authorities:
        actions.append("Get a human protocol/data decision before trying to convert dataset-length-limited records into formal M1/M3 evidence.")
    if "human_review_required" in authorities:
        actions.append("Complete attribution review/signoff after the protocol path is chosen.")
    if "upstream_m1_evidence_required" in authorities:
        actions.append("Rerun M3 acceptance and phase gate only after M1 evidence is regenerated under the chosen protocol.")
    if any(route.get("codex_can_advance_without_human") is True for route in routes):
        actions.append("Codex may continue CPU/read-only audits or guarded reruns for blockers marked codex_can_advance_without_human=true.")
    return actions


def _blockers(phase_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blockers = phase_report.get("blockers")
    if not isinstance(blockers, list):
        raise BlockerDecisionMatrixError("BLOCKER_DECISION_MATRIX_PHASE_BLOCKERS_INVALID")
    result: list[Mapping[str, Any]] = []
    for blocker in blockers:
        if not isinstance(blocker, Mapping) or not isinstance(blocker.get("requirement"), str):
            raise BlockerDecisionMatrixError("BLOCKER_DECISION_MATRIX_PHASE_BLOCKERS_INVALID")
        result.append(blocker)
    return result


def _load_report_from_manifest(manifest: Mapping[str, Any], code: str) -> Mapping[str, Any]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise BlockerDecisionMatrixError(code)
    report, _, _ = _read_json(Path(report_path), code)
    return report


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockerDecisionMatrixError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise BlockerDecisionMatrixError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BlockerDecisionMatrixError("BLOCKER_DECISION_MATRIX_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "blocker-decision-matrix.json", report_bytes)
        _write_bytes_atomic(temporary / "blocker-decision-matrix.md", markdown_bytes)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("blocker_decision_matrix_json", report_bytes, "application/json"),
                ("blocker_decision_matrix_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-blocker-decision-matrix-manifest",
            "state": report["state"],
            "phase_gate_state": report["phase_gate_state"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_policy": report["formal_training_policy"],
            "blocker_count": report["blocker_count"],
            "route_counts": report["route_counts"],
            "report_path": str(destination / "blocker-decision-matrix.json"),
            "markdown_path": str(destination / "blocker-decision-matrix.md"),
            "cas_refs": cas_refs,
            "next_safe_actions": report["next_safe_actions"],
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
        "# Blocker Decision Matrix",
        "",
        f"Phase gate state: `{report['phase_gate_state']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        f"Formal training GPU policy: `{report['formal_training_policy']['gpu_policy']}`",
        "",
        "| Requirement | Authority | Codex Can Advance | GPU Policy | Next Step |",
        "|:--|:--|:--|:--|:--|",
    ]
    for route in report["routes"]:
        lines.append(
            f"| `{route['requirement']}` | `{route['blocking_authority']}` | "
            f"`{route['codex_can_advance_without_human']}` | `{route['gpu_policy']}` | {route['next_step']} |"
        )
    lines.extend(["", "## Next Safe Actions", ""])
    lines.extend(f"- {item}" for item in report["next_safe_actions"])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BlockerDecisionMatrixError("BLOCKER_DECISION_MATRIX_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="route phase-gate blockers")
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_blocker_decision_matrix(
            phase_gate_manifest=args.phase_gate_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
