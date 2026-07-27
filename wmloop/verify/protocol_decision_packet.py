"""Human decision packet for resolving the active G1 horizon blocker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ProtocolDecisionPacketError(RuntimeError):
    """A protocol decision packet could not be generated."""


def run_protocol_decision_packet(
    *,
    horizon_protocol_amendment_manifest: Path,
    data_extension_audit_manifest: Path,
    phase_gate_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a consolidated human-review packet without changing protocol."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolDecisionPacketError("PROTOCOL_DECISION_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(Path(cas_storage_root))
    sources = {
        "horizon_protocol_amendment": _load_source_with_report(horizon_protocol_amendment_manifest, cas=cas, archive=archive),
        "data_extension_audit": _load_source_with_report(data_extension_audit_manifest, cas=cas, archive=archive),
        "phase_gate": _load_source_with_report(phase_gate_manifest, cas=cas, archive=archive),
    }
    amendment = _report_or_payload(sources, "horizon_protocol_amendment")
    data_extension_manifest = _payload(sources, "data_extension_audit")
    phase_gate_manifest_payload = _payload(sources, "phase_gate")
    data_extension_report = _report_or_payload(sources, "data_extension_audit")
    options = _decision_options(
        amendment=amendment,
        data_extension_manifest=data_extension_manifest,
        data_extension_report=data_extension_report,
        phase_gate_manifest=phase_gate_manifest_payload,
    )
    recommendations = _recommendations(options)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-protocol-decision-packet",
        "state": "awaiting_human_decision",
        "active_protocol_changed": False,
        "human_approval_required": True,
        "selected_option": None,
        "current_gate_state": {
            "phase": phase_gate_manifest_payload.get("phase"),
            "state": phase_gate_manifest_payload.get("state"),
            "m4_launch_allowed": phase_gate_manifest_payload.get("m4_launch_allowed"),
            "blockers": phase_gate_manifest_payload.get("blockers", []),
            "next_actions": phase_gate_manifest_payload.get("next_actions", []),
        },
        "decision_options": options,
        "recommendations": recommendations,
        "post_decision_contract": [
            "No active goal config or verdict protocol is modified by this packet.",
            "Any selected protocol-changing option must be applied in a new human-approved version boundary.",
            "After approval, rerun M0/M1/M3 evidence and regenerate the M4 phase gate before launching M4.",
        ],
        "sources": {name: source["summary"] for name, source in sources.items()},
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _decision_options(
    *,
    amendment: Mapping[str, Any],
    data_extension_manifest: Mapping[str, Any],
    data_extension_report: Mapping[str, Any],
    phase_gate_manifest: Mapping[str, Any],
) -> list[dict[str, object]]:
    raw_options = amendment.get("candidate_options")
    if not isinstance(raw_options, list) or not raw_options:
        raise ProtocolDecisionPacketError("PROTOCOL_DECISION_AMENDMENT_OPTIONS_INVALID")
    phase_blockers = {str(item.get("requirement")) for item in phase_gate_manifest.get("blockers", []) if isinstance(item, Mapping)}
    data_preserves_now = data_extension_manifest.get("preserve_original_g1_feasible_with_current_data") is True
    unfilled = data_extension_manifest.get("unfilled_blocked_environments", [])
    fill_opportunities = data_extension_report.get("fill_opportunities", [])
    result: list[dict[str, object]] = []
    for option in raw_options:
        if not isinstance(option, Mapping) or not isinstance(option.get("option_id"), str):
            raise ProtocolDecisionPacketError("PROTOCOL_DECISION_AMENDMENT_OPTIONS_INVALID")
        option_id = str(option["option_id"])
        claim_scope = str(option.get("claim_scope") or "")
        active_protocol_changed = option.get("active_protocol_changed") is True
        supported_environment_count = _int(option.get("supported_environment_count"))
        strict_t31_feasible = option.get("strict_t3_1_input_feasible") is True
        preserves_original = not active_protocol_changed and "original_g1" in claim_scope
        m4_effect = _m4_effect(
            option_id=option_id,
            strict_t31_feasible=strict_t31_feasible,
            supported_environment_count=supported_environment_count,
            preserves_original=preserves_original,
            data_preserves_now=data_preserves_now,
            phase_blockers=phase_blockers,
        )
        result.append(
            {
                "option_id": option_id,
                "human_action": option.get("human_action"),
                "active_protocol_changed": active_protocol_changed,
                "preserves_original_g1_metric": preserves_original,
                "claim_scope": claim_scope,
                "horizons": option.get("horizons"),
                "horizons_by_environment": option.get("horizons_by_environment"),
                "primary_objective": option.get("primary_objective"),
                "supported_environment_count": supported_environment_count,
                "supported_environments": option.get("supported_environments", []),
                "blocked_environments": option.get("blocked_environments", []),
                "strict_t3_1_input_feasible": strict_t31_feasible,
                "can_unblock_t3_1_after_approval": _can_unblock_t31(option_id, strict_t31_feasible, data_preserves_now),
                "can_open_m4_gate_after_rerun": m4_effect["can_open_m4_gate_after_rerun"],
                "m4_gate_effect": m4_effect["effect"],
                "requires_schema_change": False,
                "requires_new_data": option_id == "data_extension_preserve_g1",
                "current_local_data_sufficient": _current_local_data_sufficient(option_id, data_preserves_now),
                "risk": option.get("risk"),
                "post_approval_workflow": _post_approval_workflow(option_id, unfilled=unfilled, fill_opportunities=fill_opportunities),
            }
        )
    return result


def _m4_effect(
    *,
    option_id: str,
    strict_t31_feasible: bool,
    supported_environment_count: int,
    preserves_original: bool,
    data_preserves_now: bool,
    phase_blockers: set[str],
) -> dict[str, object]:
    if option_id == "keep_current_frozen_protocol":
        return {"can_open_m4_gate_after_rerun": False, "effect": "keeps_current_m1_m3_m4_blockers"}
    if option_id == "data_extension_preserve_g1":
        return {
            "can_open_m4_gate_after_rerun": data_preserves_now,
            "effect": "requires_external_long_data_before_rerun" if not data_preserves_now else "can_preserve_original_g1_after_refreeze_and_rerun",
        }
    if option_id == "common_horizon_all_environments":
        return {
            "can_open_m4_gate_after_rerun": strict_t31_feasible and supported_environment_count >= 8,
            "effect": "can_unblock_8_env_m1_m3_after_human_protocol_revision_but_changes_claim_scope",
        }
    if option_id == "longest_common_prefix_meeting_t3_1":
        return {
            "can_open_m4_gate_after_rerun": False,
            "effect": "can_unblock_t3_1_but_not_8_env_m4_gate_without_scope_change",
        }
    if option_id == "per_environment_max_horizon":
        return {
            "can_open_m4_gate_after_rerun": strict_t31_feasible and supported_environment_count >= 8,
            "effect": "can_unblock_8_env_horizon_coverage_after_ladder_protocol_application_and_full_rerun",
        }
    return {
        "can_open_m4_gate_after_rerun": False,
        "effect": "unknown_option_requires_manual_analysis",
    }


def _can_unblock_t31(option_id: str, strict_t31_feasible: bool, data_preserves_now: bool) -> bool:
    if option_id == "data_extension_preserve_g1":
        return data_preserves_now
    return strict_t31_feasible


def _current_local_data_sufficient(option_id: str, data_preserves_now: bool) -> bool:
    if option_id == "data_extension_preserve_g1":
        return data_preserves_now
    return option_id != "keep_current_frozen_protocol"


def _post_approval_workflow(option_id: str, *, unfilled: object, fill_opportunities: object) -> list[str]:
    if option_id == "keep_current_frozen_protocol":
        return ["No config change; keep formal failure_report records blocked and do not launch M4."]
    if option_id == "data_extension_preserve_g1":
        rows = ["Collect or locate longer trajectories for every unfilled blocked environment."]
        if fill_opportunities:
            rows.append("Review noncanonical fill opportunities before any refreeze.")
        if unfilled:
            rows.append(f"Still missing local long-data candidates for: {','.join(str(item) for item in unfilled)}.")
        rows.extend(
            [
                "Create a new dataset freeze and held-out protocol after data is approved.",
                "Rerun horizon availability, raw probes, raw failure reports, attribution review, M3 acceptance audit, and M4 phase gate.",
            ]
        )
        return rows
    if option_id == "common_horizon_all_environments":
        return [
            "Create a human-approved revised goal config with horizons [16,32] and primary objective auc_psnr_16_32.",
            "Regenerate M1 raw failure reports for all 8 environments under the revised claim scope.",
            "Rerun attribution review, M3 proposal readiness, M3 acceptance audit, and M4 phase gate before M4 launch.",
        ]
    if option_id == "longest_common_prefix_meeting_t3_1":
        return [
            "Create a human-approved revised goal config with horizons [16,32,48] and explicit 7-environment claim scope excluding reacher.",
            "Use it to unblock T3.1/M3 only; do not treat it as full 8-environment M4 launch evidence.",
            "Regenerate M3 proposal readiness and M3 acceptance audit under the revised scope.",
        ]
    if option_id == "per_environment_max_horizon":
        return [
            "Create a human-approved ladder goal config with primary objective ladder_auc_psnr_envmax and per-environment horizon_ladder_path.",
            "Freeze the ladder goal and horizon_ladder artifact as a new constitutional version boundary.",
            "Regenerate M1 raw failure reports for all 8 environments under each environment's declared horizon set.",
            "Regenerate attribution review, M3 proposal readiness, M3 acceptance audit, and M4 phase gate before M4 launch.",
        ]
    return ["Manual analysis required before applying this option."]


def _recommendations(options: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_id = {str(option["option_id"]): option for option in options}
    return {
        "do_not_auto_apply": True,
        "preserve_original_g1": "data_extension_preserve_g1",
        "fastest_t3_1_unblock": "longest_common_prefix_meeting_t3_1",
        "fastest_8_env_m4_unblock_current_data": _fastest_8_env_current_data(by_id),
        "current_original_g1_local_data_sufficient": by_id.get("data_extension_preserve_g1", {}).get("current_local_data_sufficient"),
    }


def _fastest_8_env_current_data(by_id: Mapping[str, Mapping[str, object]]) -> str | None:
    if by_id.get("per_environment_max_horizon", {}).get("can_open_m4_gate_after_rerun") is True:
        return "per_environment_max_horizon"
    if by_id.get("common_horizon_all_environments", {}).get("can_open_m4_gate_after_rerun") is True:
        return "common_horizon_all_environments"
    return None


def _load_source_with_report(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    payload, payload_bytes, resolved = _read_json(path, "PROTOCOL_DECISION_SOURCE_INVALID")
    payload_ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(payload_ref)
    report_payload: Mapping[str, Any] | None = None
    report_ref = None
    report_path = payload.get("report_path")
    if isinstance(report_path, str) and report_path:
        report_payload, report_bytes, resolved_report = _read_json(Path(report_path), "PROTOCOL_DECISION_REPORT_INVALID")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
    else:
        resolved_report = None
    return {
        "payload": payload,
        "report": report_payload,
        "summary": {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "cas_ref": payload_ref,
            "report_path": str(resolved_report) if resolved_report is not None else None,
            "report_cas_ref": report_ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolDecisionPacketError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolDecisionPacketError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    payload = sources.get(name, {}).get("payload")
    if not isinstance(payload, Mapping):
        raise ProtocolDecisionPacketError(f"PROTOCOL_DECISION_SOURCE_MISSING:{name}")
    return payload


def _report_or_payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    source = sources.get(name, {})
    report = source.get("report")
    if isinstance(report, Mapping):
        return report
    return _payload(sources, name)


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "protocol-decision-packet.json", report_bytes)
        _write_bytes_atomic(temporary / "protocol-decision-packet.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-protocol-decision-packet-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "human_approval_required": report["human_approval_required"],
            "selected_option": report["selected_option"],
            "recommendations": report["recommendations"],
            "current_gate_state": report["current_gate_state"],
            "report_path": str(destination / "protocol-decision-packet.json"),
            "markdown_path": str(destination / "protocol-decision-packet.md"),
            "cas_refs": {
                "protocol_decision_packet_json": report_ref,
                "protocol_decision_packet_markdown": markdown_ref,
            },
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
        "# Protocol Decision Packet",
        "",
        f"State: `{report['state']}`",
        f"Active protocol changed: `{report['active_protocol_changed']}`",
        f"Human approval required: `{report['human_approval_required']}`",
        f"M4 launch allowed now: `{report['current_gate_state'].get('m4_launch_allowed')}`",
        "",
        "## Recommendations",
        "",
    ]
    for key, value in report["recommendations"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Options",
            "",
            "| Option | T3.1 After Approval | M4 After Rerun | Original G1 | Current Data | Scope |",
            "|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for option in report["decision_options"]:
        lines.append(
            f"| `{option['option_id']}` | {option['can_unblock_t3_1_after_approval']} | "
            f"{option['can_open_m4_gate_after_rerun']} | {option['preserves_original_g1_metric']} | "
            f"{option['current_local_data_sufficient']} | `{option['claim_scope']}` |"
        )
    lines.extend(["", "## Current Gate Blockers", ""])
    blockers = report["current_gate_state"].get("blockers", [])
    if blockers:
        for item in blockers:
            if isinstance(item, Mapping):
                lines.append(f"- `{item.get('requirement')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Contract", ""])
    lines.extend(f"- {item}" for item in report["post_decision_contract"])
    return "\n".join(lines) + "\n"


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolDecisionPacketError("PROTOCOL_DECISION_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="generate a human protocol decision packet")
    run.add_argument("--horizon-protocol-amendment-manifest", type=Path, required=True)
    run.add_argument("--data-extension-audit-manifest", type=Path, required=True)
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_protocol_decision_packet(
            horizon_protocol_amendment_manifest=args.horizon_protocol_amendment_manifest,
            data_extension_audit_manifest=args.data_extension_audit_manifest,
            phase_gate_manifest=args.phase_gate_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    raise ProtocolDecisionPacketError("PROTOCOL_DECISION_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
