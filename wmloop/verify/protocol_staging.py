"""Stage inactive goal-config previews for human-approved protocol choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class ProtocolStagingError(RuntimeError):
    """Protocol staging could not be generated."""


def run_protocol_staging(
    *,
    protocol_decision_packet_manifest: Path,
    active_goal_config: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write inactive candidate configs and rerun plans without applying them."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolStagingError("PROTOCOL_STAGING_OUTPUT_EXISTS")
    active_goal = _load_goal(active_goal_config)
    packet, packet_report, packet_summary = _load_packet(protocol_decision_packet_manifest)
    staged = _stage_candidates(active_goal=active_goal, packet=packet_report, candidate_root=destination / "candidates")
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-protocol-staging-preview",
        "state": "ready",
        "active_protocol_changed": False,
        "human_approval_required": True,
        "selected_option": None,
        "active_goal_config_path": str(Path(active_goal_config).resolve(strict=True)),
        "active_goal_sha256": _sha256_file(Path(active_goal_config).resolve(strict=True)),
        "protocol_decision_packet": packet_summary,
        "staged_candidate_count": len(staged),
        "staged_candidates": staged,
        "notes": [
            "Candidate goal configs are written under this report directory only; configs/goal is not modified.",
            "Each candidate goal spec validates against goal_spec.schema.json, but it is not active until a human-approved version boundary applies it.",
            "After applying any candidate, regenerate M1/M3 evidence and the M4 phase gate before launch.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _stage_candidates(*, active_goal: Mapping[str, Any], packet: Mapping[str, Any], candidate_root: Path) -> list[dict[str, object]]:
    options = packet.get("decision_options")
    if not isinstance(options, list) or not options:
        raise ProtocolStagingError("PROTOCOL_STAGING_PACKET_OPTIONS_INVALID")
    staged: list[dict[str, object]] = []
    for option in options:
        if not isinstance(option, Mapping) or not isinstance(option.get("option_id"), str):
            raise ProtocolStagingError("PROTOCOL_STAGING_PACKET_OPTIONS_INVALID")
        option_id = str(option["option_id"])
        goal = _candidate_goal(active_goal=active_goal, option=option)
        if goal is None:
            staged.append(_non_config_candidate(option))
            continue
        try:
            validate_document("goal_spec", goal)
        except ContractValidationError as exc:
            raise ProtocolStagingError(f"PROTOCOL_STAGING_GOAL_INVALID:{option_id}") from exc
        goal_filename = f"{_safe_name(option_id)}.goal.json"
        staged.append(
            {
                "option_id": option_id,
                "stage_kind": "inactive_goal_config_preview",
                "goal_filename": goal_filename,
                "goal_spec": goal,
                "active_protocol_changed": False,
                "requires_human_approval_to_apply": True,
                "claim_scope": option.get("claim_scope"),
                "can_unblock_t3_1_after_approval": option.get("can_unblock_t3_1_after_approval"),
                "can_open_m4_gate_after_rerun": option.get("can_open_m4_gate_after_rerun"),
                "rerun_plan": _rerun_plan(option_id=option_id, candidate_goal_path=candidate_root / goal_filename),
            }
        )
    return staged


def _candidate_goal(*, active_goal: Mapping[str, Any], option: Mapping[str, Any]) -> dict[str, object] | None:
    option_id = str(option["option_id"])
    if option_id in {"keep_current_frozen_protocol", "data_extension_preserve_g1"}:
        return None
    if option_id == "per_environment_max_horizon":
        return _per_environment_ladder_goal(active_goal=active_goal, option=option)
    horizons = option.get("horizons")
    primary = option.get("primary_objective")
    supported_envs = option.get("supported_environments")
    if not isinstance(horizons, list) or not horizons or not isinstance(primary, str) or not isinstance(supported_envs, list):
        raise ProtocolStagingError(f"PROTOCOL_STAGING_OPTION_CONFIG_INVALID:{option_id}")
    goal = deepcopy(dict(active_goal))
    goal["goal_id"] = _goal_id_for_option(option_id)
    goal["horizons"] = [int(item) for item in horizons]
    goal["primary_objective"] = primary
    goal["envs"] = [str(item) for item in supported_envs]
    protocol = deepcopy(goal["eval_protocol"])
    if isinstance(protocol, Mapping):
        protocol = dict(protocol)
        protocol["accept_horizon_extra"] = max(1, min(16, max(goal["horizons"])))  # type: ignore[arg-type]
        goal["eval_protocol"] = protocol
    return goal


def _per_environment_ladder_goal(*, active_goal: Mapping[str, Any], option: Mapping[str, Any]) -> dict[str, object]:
    horizons_by_environment = option.get("horizons_by_environment")
    supported_envs = option.get("supported_environments")
    if not isinstance(horizons_by_environment, Mapping) or not isinstance(supported_envs, list) or not supported_envs:
        raise ProtocolStagingError("PROTOCOL_STAGING_OPTION_CONFIG_INVALID:per_environment_max_horizon")
    parsed = {
        str(environment): [int(horizon) for horizon in horizons]
        for environment, horizons in horizons_by_environment.items()
        if isinstance(horizons, list) and horizons
    }
    if set(parsed) != {str(item) for item in supported_envs}:
        raise ProtocolStagingError("PROTOCOL_STAGING_OPTION_CONFIG_INVALID:per_environment_max_horizon")
    common = sorted(set.intersection(*(set(values) for values in parsed.values())))
    if not common:
        raise ProtocolStagingError("PROTOCOL_STAGING_OPTION_CONFIG_INVALID:per_environment_max_horizon")
    goal = deepcopy(dict(active_goal))
    goal["goal_id"] = _goal_id_for_option("per_environment_max_horizon")
    goal["primary_objective"] = "ladder_auc_psnr_envmax"
    goal["envs"] = [str(item) for item in supported_envs]
    protocol = deepcopy(goal["eval_protocol"])
    if not isinstance(protocol, Mapping):
        raise ProtocolStagingError("PROTOCOL_STAGING_OPTION_CONFIG_INVALID:per_environment_max_horizon")
    protocol = dict(protocol)
    protocol["mode"] = "per_environment_horizon_ladder"
    protocol["horizon_ladder_path"] = "configs/goal/horizon_ladder_g1_acwm_phys_v1.yaml"
    protocol["cross_environment_comparison_horizons"] = common
    protocol["long_horizon_claim_policy"] = "horizon_64_claims_only_for_ladder_long_horizon_64_envs"
    goal["eval_protocol"] = protocol
    return goal


def _non_config_candidate(option: Mapping[str, Any]) -> dict[str, object]:
    option_id = str(option["option_id"])
    return {
        "option_id": option_id,
        "stage_kind": "no_goal_config_preview",
        "active_protocol_changed": False,
        "requires_human_approval_to_apply": True,
        "claim_scope": option.get("claim_scope"),
        "reason": _non_config_reason(option_id),
        "can_unblock_t3_1_after_approval": option.get("can_unblock_t3_1_after_approval"),
        "can_open_m4_gate_after_rerun": option.get("can_open_m4_gate_after_rerun"),
        "rerun_plan": option.get("post_approval_workflow", []),
    }


def _non_config_reason(option_id: str) -> str:
    if option_id == "keep_current_frozen_protocol":
        return "keeps the current active config unchanged and leaves current blockers in place"
    if option_id == "data_extension_preserve_g1":
        return "requires external long data before a new freeze can produce a meaningful config preview"
    return "option is not represented as a single goal_spec"


def _rerun_plan(*, option_id: str, candidate_goal_path: Path) -> list[str]:
    goal_ref = str(candidate_goal_path)
    suffix = _safe_name(option_id)
    if option_id == "per_environment_max_horizon":
        return [
            f"Human approval: promote {goal_ref} into a new versioned ladder goal config; do not overwrite the active config silently.",
            "Freeze configs/goal/horizon_ladder_g1_acwm_phys_v1.yaml with the promoted ladder goal in a new constitutional config.",
            f"Rerun raw-probe coverage and raw failure reports under {goal_ref}, respecting each environment's declared horizon set.",
            f"Regenerate attribution review, M3 proposal readiness, M3 acceptance audit, and M4 phase gate with suffix {suffix}-r1.",
        ]
    return [
        f"Human approval: copy or promote {goal_ref} into a new versioned goal config; do not overwrite the active config silently.",
        f"Rerun horizon protocol audit against {goal_ref} and write results/reports/m1-horizon-protocol-decision-{suffix}-r1.",
        f"Regenerate raw-probe coverage and raw failure reports under {goal_ref}.",
        f"Regenerate attribution review, M3 proposal readiness, M3 acceptance audit, and M4 phase gate with suffix {suffix}-r1.",
    ]


def _goal_id_for_option(option_id: str) -> str:
    if option_id == "common_horizon_all_environments":
        return "g1_long_horizon_revised_16_32_all_envs"
    if option_id == "longest_common_prefix_meeting_t3_1":
        return "g1_long_horizon_revised_16_48_7env_t31"
    if option_id == "per_environment_max_horizon":
        return "g1_long_horizon_ladder_v1"
    return f"g1_long_horizon_candidate_{_safe_name(option_id)}"


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise ProtocolStagingError("PROTOCOL_STAGING_ACTIVE_GOAL_INVALID") from exc
    return payload


def _load_packet(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, object]]:
    manifest, manifest_bytes, resolved = _read_json(path, "PROTOCOL_STAGING_PACKET_INVALID")
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise ProtocolStagingError("PROTOCOL_STAGING_PACKET_REPORT_MISSING")
    report, report_bytes, resolved_report = _read_json(Path(report_path), "PROTOCOL_STAGING_PACKET_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-protocol-decision-packet":
        raise ProtocolStagingError("PROTOCOL_STAGING_PACKET_REPORT_INVALID")
    return (
        manifest,
        report,
        {
            "manifest_path": str(resolved),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "report_path": str(resolved_report),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "state": manifest.get("state"),
            "selected_option": manifest.get("selected_option"),
        },
    )


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolStagingError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolStagingError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        candidates_dir = temporary / "candidates"
        candidates_dir.mkdir(mode=0o700)
        candidate_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
        cas = ContentAddressedStore(Path(root))
        for candidate in report["staged_candidates"]:  # type: ignore[index]
            if not isinstance(candidate, Mapping) or candidate.get("stage_kind") != "inactive_goal_config_preview":
                continue
            filename = str(candidate["goal_filename"])
            goal_bytes = _canonical_json_bytes(candidate["goal_spec"])  # type: ignore[arg-type,index]
            _write_bytes_atomic(candidates_dir / filename, goal_bytes)
            ref = cas.put_bytes(goal_bytes, media_type="application/json").uri
            candidate_refs[str(candidate["option_id"])] = ref
            if archive is not None:
                archive.record_artifact_reference(ref)
        _write_bytes_atomic(temporary / "protocol-staging.json", report_bytes)
        _write_bytes_atomic(temporary / "protocol-staging.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-protocol-staging-preview-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "human_approval_required": report["human_approval_required"],
            "selected_option": report["selected_option"],
            "staged_candidate_count": report["staged_candidate_count"],
            "report_path": str(destination / "protocol-staging.json"),
            "markdown_path": str(destination / "protocol-staging.md"),
            "candidate_goal_paths": {
                str(candidate["option_id"]): str(destination / "candidates" / str(candidate["goal_filename"]))
                for candidate in report["staged_candidates"]  # type: ignore[index]
                if isinstance(candidate, Mapping) and candidate.get("stage_kind") == "inactive_goal_config_preview"
            },
            "cas_refs": {
                "protocol_staging_json": report_ref,
                "protocol_staging_markdown": markdown_ref,
                "candidate_goals": candidate_refs,
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
        "# Protocol Staging Preview",
        "",
        f"State: `{report['state']}`",
        f"Active protocol changed: `{report['active_protocol_changed']}`",
        f"Human approval required: `{report['human_approval_required']}`",
        "",
        "| Option | Stage Kind | Goal File | T3.1 | M4 | Scope |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for candidate in report["staged_candidates"]:
        goal_file = candidate.get("goal_filename") or "n/a"
        lines.append(
            f"| `{candidate['option_id']}` | `{candidate['stage_kind']}` | `{goal_file}` | "
            f"{candidate.get('can_unblock_t3_1_after_approval')} | {candidate.get('can_open_m4_gate_after_rerun')} | "
            f"`{candidate.get('claim_scope')}` |"
        )
    lines.extend(["", "Notes:"])
    lines.extend(f"- {item}" for item in report["notes"])
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolStagingError("PROTOCOL_STAGING_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="stage inactive candidate goal configs from a decision packet")
    run.add_argument("--protocol-decision-packet-manifest", type=Path, required=True)
    run.add_argument("--active-goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_protocol_staging(
            protocol_decision_packet_manifest=args.protocol_decision_packet_manifest,
            active_goal_config=args.active_goal_config,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    raise ProtocolStagingError("PROTOCOL_STAGING_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
