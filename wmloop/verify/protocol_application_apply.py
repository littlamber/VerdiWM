"""Apply a human-approved protocol candidate as a new versioned goal config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class ProtocolApplicationApplyError(RuntimeError):
    """Protocol application failed closed."""


def run_protocol_application_apply(
    *,
    protocol_application_preview_manifest: Path,
    active_goal_config: Path,
    selected_option: str,
    output_root: Path,
    target_goal_config: Path | None = None,
    confirm_human_approved_protocol_change: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Promote one ready preview candidate without overwriting the active config."""

    if not confirm_human_approved_protocol_change:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_HUMAN_APPROVAL_REQUIRED")
    if not selected_option:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_SELECTED_OPTION_REQUIRED")

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_OUTPUT_EXISTS")

    active_path = Path(active_goal_config).resolve(strict=True)
    active_goal = _load_goal(active_path, "PROTOCOL_APPLICATION_APPLY_ACTIVE_GOAL_INVALID")
    active_hash = _sha256_file(active_path)

    preview_manifest, preview_manifest_bytes, preview_manifest_path = _read_json(
        protocol_application_preview_manifest,
        "PROTOCOL_APPLICATION_APPLY_PREVIEW_MANIFEST_INVALID",
    )
    preview_report = _load_preview_report(preview_manifest)
    _verify_preview_state(preview_manifest=preview_manifest, preview_report=preview_report)
    _verify_active_goal_unchanged(active_path=active_path, active_hash=active_hash, preview_report=preview_report)

    candidate = _selected_candidate(preview_report, selected_option)
    source_path = Path(str(candidate["candidate_goal_path"])).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_hash = str(candidate["candidate_goal_sha256"])
    if source_hash != expected_hash:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_HASH_DRIFT")
    manifest_ref = candidate.get("candidate_manifest_cas_ref")
    manifest_hash = _hash_from_cas_ref(manifest_ref) if isinstance(manifest_ref, str) else None
    if manifest_hash is not None and manifest_hash != source_hash:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_CAS_MISMATCH")

    candidate_goal = _load_goal(source_path, "PROTOCOL_APPLICATION_APPLY_CANDIDATE_GOAL_INVALID")
    if candidate_goal.get("goal_id") != candidate.get("goal_id"):
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_GOAL_ID_MISMATCH")
    if not _split_fragments_preserved(candidate_goal):
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_SPLIT_FRAGMENTS_MISSING")

    target_path = _resolve_target_path(
        active_goal_config=active_path,
        target_goal_config=target_goal_config,
        candidate_goal=candidate_goal,
    )
    if target_path.exists() or target_path.is_symlink():
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_TARGET_EXISTS")

    report = _application_report(
        preview_manifest=preview_manifest,
        preview_manifest_bytes=preview_manifest_bytes,
        preview_manifest_path=preview_manifest_path,
        preview_report=preview_report,
        active_path=active_path,
        active_goal=active_goal,
        active_hash=active_hash,
        candidate=candidate,
        source_path=source_path,
        source_hash=source_hash,
        target_path=target_path,
        candidate_goal=candidate_goal,
    )
    target_created = False
    try:
        target_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_bytes_atomic(target_path, source_bytes)
        target_created = True
        target_hash = _sha256_file(target_path)
        if target_hash != source_hash:
            raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_TARGET_HASH_MISMATCH")
        _load_goal(target_path, "PROTOCOL_APPLICATION_APPLY_TARGET_GOAL_INVALID")
        return _write_report_bundle(
            report=report,
            output_root=destination,
            promoted_goal_bytes=source_bytes,
            archive_db=archive_db,
            cas_root=cas_root,
        )
    except Exception:
        if target_created and target_path.exists() and not target_path.is_symlink():
            try:
                if _sha256_file(target_path) == source_hash:
                    target_path.unlink()
            except OSError:
                pass
        raise


def _application_report(
    *,
    preview_manifest: Mapping[str, Any],
    preview_manifest_bytes: bytes,
    preview_manifest_path: Path,
    preview_report: Mapping[str, Any],
    active_path: Path,
    active_goal: Mapping[str, Any],
    active_hash: str,
    candidate: Mapping[str, Any],
    source_path: Path,
    source_hash: str,
    target_path: Path,
    candidate_goal: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-protocol-application-receipt",
        "state": "applied",
        "active_protocol_changed": True,
        "active_goal_config_mutated": False,
        "candidate_goal_paths_mutated": False,
        "human_approval_confirmed": True,
        "selected_option": candidate["option_id"],
        "m4_launch_allowed": False,
        "m4_launch_guard": "Regenerate M1/M3 evidence and the M4 phase gate before any M4 launch.",
        "active_goal": {
            "path": str(active_path),
            "sha256": active_hash,
            "goal_id": active_goal.get("goal_id"),
            "mutated": False,
        },
        "protocol_application_preview": {
            "manifest_path": str(preview_manifest_path),
            "manifest_sha256": hashlib.sha256(preview_manifest_bytes).hexdigest(),
            "report_path": str(Path(str(preview_manifest["report_path"])).resolve(strict=True)),
            "state": preview_manifest.get("state"),
            "ready_candidate_count": preview_manifest.get("ready_candidate_count"),
            "selected_option": preview_manifest.get("selected_option"),
            "active_goal_sha256": preview_report.get("active_goal", {}).get("sha256")
            if isinstance(preview_report.get("active_goal"), Mapping)
            else None,
        },
        "source_candidate": {
            "option_id": candidate["option_id"],
            "path": str(source_path),
            "sha256": source_hash,
            "manifest_cas_ref": candidate.get("candidate_manifest_cas_ref"),
            "hash_matches_manifest": True,
            "mutated": False,
        },
        "promoted_goal_config": {
            "path": str(target_path),
            "sha256": source_hash,
            "goal_id": candidate_goal.get("goal_id"),
            "horizons": candidate_goal.get("horizons"),
            "primary_objective": candidate_goal.get("primary_objective"),
            "envs": candidate_goal.get("envs"),
            "split_fragments_preserved": True,
            "created": True,
        },
        "post_apply_required_reruns": candidate.get("rerun_plan", []),
        "post_apply_contract": [
            "The previous active goal config remains unchanged.",
            "The promoted config is a new version boundary, not an in-place protocol mutation.",
            "Regenerate horizon protocol evidence, raw-probe coverage, raw failure reports, attribution review, M3 acceptance audit, and M4 phase gate.",
            "M4 launch remains forbidden until the regenerated phase gate reports m4_launch_allowed=true.",
        ],
    }


def _verify_preview_state(*, preview_manifest: Mapping[str, Any], preview_report: Mapping[str, Any]) -> None:
    if preview_manifest.get("artifact_type") != "wmloop-protocol-application-preview-manifest":
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_MANIFEST_INVALID")
    if preview_manifest.get("state") != "ready" or preview_report.get("state") != "ready":
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_NOT_READY")
    if preview_manifest.get("active_protocol_changed") is not False or preview_report.get("active_protocol_changed") is not False:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_ALREADY_MUTATED")
    if preview_manifest.get("human_approval_required") is not True or preview_report.get("human_approval_required") is not True:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_APPROVAL_CONTRACT_INVALID")


def _verify_active_goal_unchanged(*, active_path: Path, active_hash: str, preview_report: Mapping[str, Any]) -> None:
    active_preview = preview_report.get("active_goal")
    if not isinstance(active_preview, Mapping):
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_ACTIVE_GOAL_PREVIEW_MISSING")
    preview_path = active_preview.get("path")
    if isinstance(preview_path, str) and Path(preview_path).resolve() != active_path:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_ACTIVE_GOAL_PATH_MISMATCH")
    preview_hash = active_preview.get("sha256")
    expected_hash = active_preview.get("expected_sha256_from_staging")
    if active_preview.get("drift_detected") is True or active_hash != preview_hash or active_hash != expected_hash:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_ACTIVE_GOAL_HASH_DRIFT")


def _selected_candidate(preview_report: Mapping[str, Any], selected_option: str) -> Mapping[str, Any]:
    candidates = preview_report.get("candidate_previews")
    if not isinstance(candidates, list):
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATES_INVALID")
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("option_id") == selected_option:
            if candidate.get("ready_for_human_apply") is not True:
                raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_NOT_READY")
            if candidate.get("stage_kind") != "inactive_goal_config_preview":
                raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_NOT_CONFIG")
            for key in ("candidate_goal_path", "candidate_goal_sha256", "goal_id"):
                if not isinstance(candidate.get(key), str) or not candidate.get(key):
                    raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_CANDIDATE_INVALID")
            return candidate
    raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_SELECTED_OPTION_NOT_FOUND")


def _resolve_target_path(
    *,
    active_goal_config: Path,
    target_goal_config: Path | None,
    candidate_goal: Mapping[str, Any],
) -> Path:
    goal_dir = active_goal_config.parent.resolve()
    goal_id = candidate_goal.get("goal_id")
    if not isinstance(goal_id, str) or not goal_id:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_GOAL_ID_INVALID")
    target = Path(target_goal_config) if target_goal_config is not None else goal_dir / f"{goal_id}.yaml"
    resolved = target.resolve()
    if resolved.parent != goal_dir:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_TARGET_OUTSIDE_GOAL_DIR")
    if resolved == active_goal_config:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_TARGET_IS_ACTIVE_GOAL")
    return resolved


def _split_fragments_preserved(goal: Mapping[str, Any]) -> bool:
    protocol = goal.get("eval_protocol")
    if not isinstance(protocol, Mapping):
        return False
    dev = protocol.get("dev_split")
    accept = protocol.get("accept_split")
    return isinstance(dev, str) and "#dev" in dev and isinstance(accept, str) and "#accept" in accept


def _hash_from_cas_ref(ref: str) -> str | None:
    prefix = "cas://sha256/"
    if ref.startswith(prefix) and len(ref) == len(prefix) + 64:
        return ref[len(prefix) :]
    return None


def _load_goal(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(path)
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise ProtocolApplicationApplyError(code) from exc
    return payload


def _load_preview_report(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_REPORT_MISSING")
    report, _, _ = _read_json(Path(report_path), "PROTOCOL_APPLICATION_APPLY_PREVIEW_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-protocol-application-preview":
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_PREVIEW_REPORT_INVALID")
    return report


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolApplicationApplyError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolApplicationApplyError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    promoted_goal_bytes: bytes,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("protocol_application_receipt_json", report_bytes, "application/json"),
                ("protocol_application_receipt_markdown", markdown_bytes, "text/markdown"),
                ("promoted_goal_config", promoted_goal_bytes, "application/json"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        _write_bytes_atomic(temporary / "protocol-application-receipt.json", report_bytes)
        _write_bytes_atomic(temporary / "protocol-application-receipt.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-protocol-application-receipt-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "active_goal_config_mutated": report["active_goal_config_mutated"],
            "candidate_goal_paths_mutated": report["candidate_goal_paths_mutated"],
            "human_approval_confirmed": report["human_approval_confirmed"],
            "selected_option": report["selected_option"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "target_goal_config": report["promoted_goal_config"],
            "report_path": str(destination / "protocol-application-receipt.json"),
            "markdown_path": str(destination / "protocol-application-receipt.md"),
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
    target = report["promoted_goal_config"]
    active = report["active_goal"]
    lines = [
        "# Protocol Application Receipt",
        "",
        f"State: `{report['state']}`",
        f"Selected option: `{report['selected_option']}`",
        f"Active protocol changed: `{report['active_protocol_changed']}`",
        f"Active goal config mutated: `{report['active_goal_config_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Goal Configs",
        "",
        f"- Active: `{active['path']}` (`{active['sha256']}`), mutated=`{active['mutated']}`",
        f"- Promoted: `{target['path']}` (`{target['sha256']}`), goal_id=`{target['goal_id']}`",
        "",
        "## Required Reruns",
        "",
    ]
    reruns = report.get("post_apply_required_reruns", [])
    if isinstance(reruns, list) and reruns:
        lines.extend(f"- {item}" for item in reruns)
    else:
        lines.append("- Regenerate M1/M3 evidence and the M4 phase gate.")
    lines.extend(["", "## Contract", ""])
    lines.extend(f"- {item}" for item in report["post_apply_contract"])
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolApplicationApplyError("PROTOCOL_APPLICATION_APPLY_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="apply one human-approved protocol candidate")
    run.add_argument("--protocol-application-preview-manifest", type=Path, required=True)
    run.add_argument("--active-goal-config", type=Path, required=True)
    run.add_argument("--selected-option", required=True)
    run.add_argument("--target-goal-config", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--confirm-human-approved-protocol-change", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_protocol_application_apply(
            protocol_application_preview_manifest=args.protocol_application_preview_manifest,
            active_goal_config=args.active_goal_config,
            selected_option=args.selected_option,
            target_goal_config=args.target_goal_config,
            output_root=args.output_root,
            confirm_human_approved_protocol_change=args.confirm_human_approved_protocol_change,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
