"""Preview staged protocol application without mutating active goal configs."""

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


class ProtocolApplicationPreviewError(RuntimeError):
    """Protocol application preview could not be generated."""


def run_protocol_application_preview(
    *,
    protocol_staging_manifest: Path,
    active_goal_config: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a dry-run application receipt for staged candidate goal configs."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_OUTPUT_EXISTS")
    active_path = Path(active_goal_config).resolve(strict=True)
    active_goal = _load_goal(active_path, "PROTOCOL_APPLICATION_PREVIEW_ACTIVE_GOAL_INVALID")
    active_hash = _sha256_file(active_path)
    staging_manifest_payload, staging_manifest_bytes, staging_manifest_path = _read_json(
        protocol_staging_manifest,
        "PROTOCOL_APPLICATION_PREVIEW_STAGING_MANIFEST_INVALID",
    )
    staging_report = _load_staging_report(staging_manifest_payload)
    expected_active_hash = staging_report.get("active_goal_sha256")
    active_goal_drift = expected_active_hash != active_hash
    candidate_previews = _candidate_previews(
        staging_manifest=staging_manifest_payload,
        staging_report=staging_report,
        active_goal=active_goal,
        active_goal_drift=active_goal_drift,
    )
    ready_count = sum(1 for item in candidate_previews if item["ready_for_human_apply"] is True)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-protocol-application-preview",
        "state": "ready" if ready_count else "blocked_active_goal_drift" if active_goal_drift else "no_config_candidates",
        "active_protocol_changed": False,
        "active_goal_config_mutated": False,
        "candidate_goal_paths_mutated": False,
        "human_approval_required": True,
        "selected_option": None,
        "ready_candidate_count": ready_count,
        "candidate_count": len(candidate_previews),
        "active_goal": {
            "path": str(active_path),
            "sha256": active_hash,
            "expected_sha256_from_staging": expected_active_hash,
            "drift_detected": active_goal_drift,
            "goal_id": active_goal.get("goal_id"),
        },
        "protocol_staging": {
            "manifest_path": str(staging_manifest_path),
            "manifest_sha256": hashlib.sha256(staging_manifest_bytes).hexdigest(),
            "report_path": str(Path(str(staging_manifest_payload["report_path"])).resolve(strict=True)),
            "state": staging_manifest_payload.get("state"),
            "selected_option": staging_manifest_payload.get("selected_option"),
        },
        "candidate_previews": candidate_previews,
        "post_apply_contract": [
            "This preview does not copy, overwrite, or modify configs/goal.",
            "A human-approved version boundary is required before promoting any candidate goal config.",
            "After promotion, rerun M1/M3 evidence and regenerate the M4 phase gate before launching M4.",
            "M4 launch remains forbidden unless the regenerated phase gate reports m4_launch_allowed=true.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _candidate_previews(
    *,
    staging_manifest: Mapping[str, Any],
    staging_report: Mapping[str, Any],
    active_goal: Mapping[str, Any],
    active_goal_drift: bool,
) -> list[dict[str, object]]:
    candidates = staging_report.get("staged_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_CANDIDATES_INVALID")
    candidate_goal_paths = staging_manifest.get("candidate_goal_paths")
    if not isinstance(candidate_goal_paths, Mapping):
        candidate_goal_paths = {}
    candidate_refs = staging_manifest.get("cas_refs", {}).get("candidate_goals", {}) if isinstance(staging_manifest.get("cas_refs"), Mapping) else {}
    if not isinstance(candidate_refs, Mapping):
        candidate_refs = {}
    previews: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("option_id"), str):
            raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_CANDIDATES_INVALID")
        option_id = str(candidate["option_id"])
        if candidate.get("stage_kind") != "inactive_goal_config_preview":
            previews.append(_non_config_preview(candidate))
            continue
        path_value = candidate_goal_paths.get(option_id)
        if not isinstance(path_value, str) or not path_value:
            raise ProtocolApplicationPreviewError(f"PROTOCOL_APPLICATION_PREVIEW_CANDIDATE_PATH_MISSING:{option_id}")
        candidate_path = Path(path_value).resolve(strict=True)
        goal = _load_goal(candidate_path, f"PROTOCOL_APPLICATION_PREVIEW_CANDIDATE_INVALID:{option_id}")
        candidate_hash = _sha256_file(candidate_path)
        expected_ref = candidate_refs.get(option_id)
        expected_hash = _hash_from_cas_ref(expected_ref) if isinstance(expected_ref, str) else None
        hash_matches_manifest = expected_hash is None or expected_hash == candidate_hash
        split_fragments_preserved = _split_fragments_preserved(goal)
        goal_id_differs = goal.get("goal_id") != active_goal.get("goal_id")
        ready = (
            not active_goal_drift
            and hash_matches_manifest
            and split_fragments_preserved
            and goal_id_differs
        )
        blockers: list[str] = []
        if active_goal_drift:
            blockers.append("active_goal_hash_drift")
        if not hash_matches_manifest:
            blockers.append("candidate_hash_manifest_mismatch")
        if not split_fragments_preserved:
            blockers.append("heldout_split_fragment_missing")
        if not goal_id_differs:
            blockers.append("candidate_goal_id_matches_active")
        previews.append(
            {
                "option_id": option_id,
                "stage_kind": candidate.get("stage_kind"),
                "claim_scope": candidate.get("claim_scope"),
                "ready_for_human_apply": ready,
                "blockers": blockers,
                "candidate_goal_path": str(candidate_path),
                "candidate_goal_sha256": candidate_hash,
                "candidate_manifest_cas_ref": expected_ref,
                "candidate_hash_matches_manifest": hash_matches_manifest,
                "goal_id": goal.get("goal_id"),
                "horizons": goal.get("horizons"),
                "primary_objective": goal.get("primary_objective"),
                "envs": goal.get("envs"),
                "split_fragments_preserved": split_fragments_preserved,
                "suggested_target_path": str(Path("configs") / "goal" / f"{goal.get('goal_id')}.yaml"),
                "active_goal_config_mutated": False,
                "rerun_plan": candidate.get("rerun_plan", []),
            }
        )
    return previews


def _non_config_preview(candidate: Mapping[str, Any]) -> dict[str, object]:
    return {
        "option_id": candidate["option_id"],
        "stage_kind": candidate.get("stage_kind"),
        "claim_scope": candidate.get("claim_scope"),
        "ready_for_human_apply": False,
        "blockers": ["no_goal_config_preview"],
        "reason": candidate.get("reason"),
        "active_goal_config_mutated": False,
        "rerun_plan": candidate.get("rerun_plan", []),
    }


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
        raise ProtocolApplicationPreviewError(code) from exc
    return payload


def _load_staging_report(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_STAGING_REPORT_MISSING")
    report, _, _ = _read_json(Path(report_path), "PROTOCOL_APPLICATION_PREVIEW_STAGING_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-protocol-staging-preview":
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_STAGING_REPORT_INVALID")
    return report


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolApplicationPreviewError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolApplicationPreviewError(f"{error}:{resolved}")
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
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "protocol-application-preview.json", report_bytes)
        _write_bytes_atomic(temporary / "protocol-application-preview.md", markdown_bytes)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("protocol_application_preview_json", report_bytes, "application/json"),
                ("protocol_application_preview_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-protocol-application-preview-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "active_goal_config_mutated": report["active_goal_config_mutated"],
            "candidate_goal_paths_mutated": report["candidate_goal_paths_mutated"],
            "human_approval_required": report["human_approval_required"],
            "selected_option": report["selected_option"],
            "ready_candidate_count": report["ready_candidate_count"],
            "candidate_count": report["candidate_count"],
            "report_path": str(destination / "protocol-application-preview.json"),
            "markdown_path": str(destination / "protocol-application-preview.md"),
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
        "# Protocol Application Preview",
        "",
        f"State: `{report['state']}`",
        f"Active goal config mutated: `{report['active_goal_config_mutated']}`",
        f"Ready candidates: `{report['ready_candidate_count']}/{report['candidate_count']}`",
        "",
        "| Option | Ready | Goal | Target | Blockers |",
        "|:--|:--|:--|:--|:--|",
    ]
    for item in report["candidate_previews"]:
        lines.append(
            f"| `{item['option_id']}` | `{item['ready_for_human_apply']}` | "
            f"`{item.get('goal_id')}` | `{item.get('suggested_target_path')}` | "
            f"`{','.join(item.get('blockers', []))}` |"
        )
    lines.extend(["", "## Post-Apply Contract", ""])
    lines.extend(f"- {item}" for item in report["post_apply_contract"])
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolApplicationPreviewError("PROTOCOL_APPLICATION_PREVIEW_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="preview applying staged protocol candidates")
    run.add_argument("--protocol-staging-manifest", type=Path, required=True)
    run.add_argument("--active-goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_protocol_application_preview(
            protocol_staging_manifest=args.protocol_staging_manifest,
            active_goal_config=args.active_goal_config,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
