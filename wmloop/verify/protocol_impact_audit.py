"""Audit protocol candidate impact without applying a protocol change."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ProtocolImpactAuditError(RuntimeError):
    """Protocol impact audit could not be generated."""


NON_PROTOCOL_BLOCKERS = {
    "M0_baseline_reproduction",
    "M0_checkpoint_source_resolution",
    "M0_checkpoint_launch_guard_clear",
    "M2_vendor_and_registry_freeze",
}
PROTOCOL_BLOCKERS = {
    "M1_raw_failure_reports",
    "M1_original_g1_data_feasibility",
    "M3_strict_acceptance",
}
HUMAN_REVIEW_BLOCKERS = {
    "M1_attribution_review",
}


def run_protocol_impact_audit(
    *,
    protocol_staging_manifest: Path,
    phase_gate_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only impact report for staged protocol candidates."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolImpactAuditError("PROTOCOL_IMPACT_OUTPUT_EXISTS")
    staging_manifest_payload, staging_manifest_bytes, staging_manifest_path = _read_json(
        protocol_staging_manifest,
        "PROTOCOL_IMPACT_STAGING_MANIFEST_INVALID",
    )
    staging_report = _load_report_from_manifest(
        staging_manifest_payload,
        "PROTOCOL_IMPACT_STAGING_REPORT_INVALID",
    )
    phase_manifest_payload, phase_manifest_bytes, phase_manifest_path = _read_json(
        phase_gate_manifest,
        "PROTOCOL_IMPACT_PHASE_GATE_INVALID",
    )
    phase_report = _load_report_from_manifest(
        phase_manifest_payload,
        "PROTOCOL_IMPACT_PHASE_GATE_REPORT_INVALID",
    )
    candidates = _candidate_impacts(staging_report=staging_report, phase_report=phase_report)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-protocol-impact-audit",
        "state": "ready",
        "active_protocol_changed": False,
        "human_approval_required": True,
        "m4_launch_allowed_now": phase_report.get("m4_launch_allowed") is True,
        "immediate_m4_launch_allowed_by_any_candidate": any(
            item["m4_launch_allowed_after_only_selecting_option"] is True for item in candidates
        ),
        "candidate_count": len(candidates),
        "candidate_impacts": candidates,
        "current_gate": {
            "manifest_path": str(phase_manifest_path),
            "manifest_sha256": hashlib.sha256(phase_manifest_bytes).hexdigest(),
            "report_path": str(Path(str(phase_manifest_payload["report_path"])).resolve(strict=True)),
            "state": phase_manifest_payload.get("state"),
            "m4_launch_allowed": phase_manifest_payload.get("m4_launch_allowed"),
            "blockers": _blocker_names(phase_report),
            "next_actions": phase_report.get("next_actions", []),
        },
        "protocol_staging": {
            "manifest_path": str(staging_manifest_path),
            "manifest_sha256": hashlib.sha256(staging_manifest_bytes).hexdigest(),
            "report_path": str(Path(str(staging_manifest_payload["report_path"])).resolve(strict=True)),
            "state": staging_manifest_payload.get("state"),
            "active_protocol_changed": staging_manifest_payload.get("active_protocol_changed"),
            "selected_option": staging_manifest_payload.get("selected_option"),
        },
        "notes": [
            "This audit does not apply candidate configs and does not mutate configs/goal.",
            "A protocol candidate can only remove protocol-scope blockers after human approval and reruns.",
            "Non-protocol M0 checkpoint blockers remain blockers regardless of horizon protocol choice.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _candidate_impacts(*, staging_report: Mapping[str, Any], phase_report: Mapping[str, Any]) -> list[dict[str, object]]:
    raw_candidates = staging_report.get("staged_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ProtocolImpactAuditError("PROTOCOL_IMPACT_STAGING_CANDIDATES_INVALID")
    current_blockers = _blocker_names(phase_report)
    non_protocol = [item for item in current_blockers if item in NON_PROTOCOL_BLOCKERS]
    human_review = [item for item in current_blockers if item in HUMAN_REVIEW_BLOCKERS]
    impacts: list[dict[str, object]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("option_id"), str):
            raise ProtocolImpactAuditError("PROTOCOL_IMPACT_STAGING_CANDIDATES_INVALID")
        option_id = str(candidate["option_id"])
        protocol_resolved = _protocol_blockers_addressed(option_id=option_id, candidate=candidate, current_blockers=current_blockers)
        unresolved_protocol = [
            item for item in current_blockers if item in PROTOCOL_BLOCKERS and item not in protocol_resolved
        ]
        extra_scope_blockers = _scope_blockers(option_id=option_id)
        unresolved_after_choice = non_protocol + human_review + unresolved_protocol + extra_scope_blockers
        m4_without_further_work = (
            not unresolved_after_choice
            and candidate.get("can_open_m4_gate_after_rerun") is True
            and candidate.get("requires_human_approval_to_apply") is not True
        )
        impacts.append(
            {
                "option_id": option_id,
                "stage_kind": candidate.get("stage_kind"),
                "claim_scope": candidate.get("claim_scope"),
                "requires_human_approval_to_apply": candidate.get("requires_human_approval_to_apply") is True,
                "candidate_goal_path": candidate.get("goal_filename"),
                "can_unblock_t3_1_after_approval": candidate.get("can_unblock_t3_1_after_approval") is True,
                "staging_claims_can_open_m4_after_rerun": candidate.get("can_open_m4_gate_after_rerun") is True,
                "m4_launch_allowed_after_only_selecting_option": m4_without_further_work,
                "protocol_blockers_addressable_after_approval_and_rerun": protocol_resolved,
                "unresolved_non_protocol_blockers": non_protocol,
                "unresolved_human_review_blockers": human_review,
                "unresolved_protocol_or_scope_blockers": unresolved_protocol + extra_scope_blockers,
                "remaining_blockers_before_m4_launch": unresolved_after_choice,
                "impact_summary": _impact_summary(
                    option_id=option_id,
                    unresolved_after_choice=unresolved_after_choice,
                    protocol_resolved=protocol_resolved,
                    candidate=candidate,
                ),
                "required_followup": _required_followup(
                    option_id=option_id,
                    non_protocol=non_protocol,
                    human_review=human_review,
                    unresolved_protocol=unresolved_protocol,
                    extra_scope_blockers=extra_scope_blockers,
                ),
            }
        )
    return impacts


def _protocol_blockers_addressed(
    *,
    option_id: str,
    candidate: Mapping[str, Any],
    current_blockers: Sequence[str],
) -> list[str]:
    if option_id == "keep_current_frozen_protocol":
        return []
    if option_id == "data_extension_preserve_g1":
        return [] if candidate.get("can_unblock_t3_1_after_approval") is not True else [
            item for item in current_blockers if item in PROTOCOL_BLOCKERS
        ]
    if option_id == "common_horizon_all_environments":
        return [item for item in current_blockers if item in PROTOCOL_BLOCKERS]
    if option_id == "longest_common_prefix_meeting_t3_1":
        return [item for item in current_blockers if item in {"M3_strict_acceptance"}]
    if option_id == "per_environment_max_horizon":
        return []
    return []


def _scope_blockers(option_id: str) -> list[str]:
    if option_id == "longest_common_prefix_meeting_t3_1":
        return ["M4_8_environment_scope_not_satisfied"]
    if option_id == "per_environment_max_horizon":
        return ["protocol_schema_reporting_changes_required"]
    if option_id == "data_extension_preserve_g1":
        return ["original_g1_long_data_not_available"]
    return []


def _impact_summary(
    *,
    option_id: str,
    unresolved_after_choice: Sequence[str],
    protocol_resolved: Sequence[str],
    candidate: Mapping[str, Any],
) -> str:
    if option_id == "keep_current_frozen_protocol":
        return "Keeps the active protocol and all current blockers in place."
    if unresolved_after_choice:
        return (
            "Can address "
            + (",".join(protocol_resolved) if protocol_resolved else "no current protocol blockers")
            + " after approval/rerun, but M4 remains blocked by "
            + ",".join(unresolved_after_choice)
            + "."
        )
    if candidate.get("can_open_m4_gate_after_rerun") is True:
        return "Could open M4 only after human approval, protocol application, and full evidence reruns."
    return "Does not open M4 under the current gate semantics."


def _required_followup(
    *,
    option_id: str,
    non_protocol: Sequence[str],
    human_review: Sequence[str],
    unresolved_protocol: Sequence[str],
    extra_scope_blockers: Sequence[str],
) -> list[str]:
    followup = ["Do not launch M4 until a regenerated phase gate reports m4_launch_allowed=true."]
    if non_protocol:
        followup.append("Resolve non-protocol blockers first: " + ",".join(non_protocol) + ".")
    if human_review:
        followup.append("Rerun attribution review under the selected protocol and obtain human sign-off.")
    if unresolved_protocol:
        followup.append("Protocol selection does not address: " + ",".join(unresolved_protocol) + ".")
    if extra_scope_blockers:
        followup.append("Resolve scope/schema blockers: " + ",".join(extra_scope_blockers) + ".")
    if option_id == "common_horizon_all_environments":
        followup.append("After human approval, regenerate all M1/M3 evidence under the 16/32 eight-environment claim scope.")
    if option_id == "longest_common_prefix_meeting_t3_1":
        followup.append("Use this option for T3.1/M3 only unless an explicit seven-environment claim is accepted.")
    return followup


def _blocker_names(phase_report: Mapping[str, Any]) -> list[str]:
    blockers = phase_report.get("blockers")
    if not isinstance(blockers, list):
        raise ProtocolImpactAuditError("PROTOCOL_IMPACT_PHASE_GATE_BLOCKERS_INVALID")
    names: list[str] = []
    for item in blockers:
        if not isinstance(item, Mapping) or not isinstance(item.get("requirement"), str):
            raise ProtocolImpactAuditError("PROTOCOL_IMPACT_PHASE_GATE_BLOCKERS_INVALID")
        names.append(str(item["requirement"]))
    return names


def _load_report_from_manifest(manifest: Mapping[str, Any], code: str) -> Mapping[str, Any]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise ProtocolImpactAuditError(code)
    report, _, _ = _read_json(Path(report_path), code)
    return report


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolImpactAuditError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolImpactAuditError(f"{error}:{resolved}")
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
        raise ProtocolImpactAuditError("PROTOCOL_IMPACT_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "protocol-impact-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "protocol-impact-audit.md", markdown_bytes)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("protocol_impact_audit_json", report_bytes, "application/json"),
                ("protocol_impact_audit_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-protocol-impact-audit-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "human_approval_required": report["human_approval_required"],
            "m4_launch_allowed_now": report["m4_launch_allowed_now"],
            "immediate_m4_launch_allowed_by_any_candidate": report["immediate_m4_launch_allowed_by_any_candidate"],
            "candidate_count": report["candidate_count"],
            "report_path": str(destination / "protocol-impact-audit.json"),
            "markdown_path": str(destination / "protocol-impact-audit.md"),
            "cas_refs": cas_refs,
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol Impact Audit",
        "",
        f"State: `{report['state']}`",
        f"M4 launch allowed now: `{report['m4_launch_allowed_now']}`",
        f"Immediate M4 launch by any candidate: `{report['immediate_m4_launch_allowed_by_any_candidate']}`",
        "",
        "| Option | M4 After Selection Only | Addressable Protocol Blockers | Remaining Blockers |",
        "|:--|:--|:--|:--|",
    ]
    for item in report["candidate_impacts"]:
        lines.append(
            f"| `{item['option_id']}` | `{item['m4_launch_allowed_after_only_selecting_option']}` | "
            f"`{','.join(item['protocol_blockers_addressable_after_approval_and_rerun'])}` | "
            f"`{','.join(item['remaining_blockers_before_m4_launch'])}` |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in report["notes"])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolImpactAuditError("PROTOCOL_IMPACT_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit staged protocol candidate impact")
    run.add_argument("--protocol-staging-manifest", type=Path, required=True)
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_protocol_impact_audit(
            protocol_staging_manifest=args.protocol_staging_manifest,
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
