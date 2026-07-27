"""Build a read-only recovery packet for blocked M0 checkpoint replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointRecoveryPacketError(RuntimeError):
    """Checkpoint recovery packet evidence could not be produced safely."""


def generate_checkpoint_recovery_packet(
    *,
    checkpoint_step_audit_manifest: Path,
    checkpoint_source_manifest: Path,
    checkpoint_candidate_inventory_manifest: Path,
    checkpoint_launch_guard_manifest: Path,
    phase_gate_manifest: Path,
    output_root: Path,
    checkpoint_remote_watch_manifest: Path | None = None,
    checkpoint_revision_watch_manifest: Path | None = None,
    quarantine_root: Path = Path("results/quarantine/checkpoints"),
    runtime_python: Path | None = None,
    checkpoint_root: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a durable packet describing the exact M0 checkpoint recovery state."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_OUTPUT_EXISTS")
    sources = {
        "checkpoint_step_audit": _load_source_with_report(
            checkpoint_step_audit_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-step-audit-manifest",
            report_artifact_type="acwm-m0-checkpoint-step-audit",
            report_key="report_path",
        ),
        "checkpoint_source": _load_source_with_report(
            checkpoint_source_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-source-audit-manifest",
            report_artifact_type="acwm-m0-checkpoint-source-audit",
            report_key="report_path",
        ),
        "candidate_inventory": _load_source_with_report(
            checkpoint_candidate_inventory_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-candidate-inventory-manifest",
            report_artifact_type="acwm-m0-checkpoint-candidate-inventory",
            report_key="report_path",
        ),
        "launch_guard": _load_json_source(
            checkpoint_launch_guard_manifest,
            expected_artifact_type="acwm-m0-baseline-launch-guard-smoke-manifest",
        ),
        "phase_gate": _load_json_source(
            phase_gate_manifest,
            expected_artifact_type="wmloop-strict-phase-gate-manifest",
        ),
    }
    if checkpoint_remote_watch_manifest is not None:
        sources["checkpoint_remote_watch"] = _load_source_with_report(
            checkpoint_remote_watch_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-remote-watch-manifest",
            report_artifact_type="acwm-m0-checkpoint-remote-watch",
            report_key="report_path",
        )
    if checkpoint_revision_watch_manifest is not None:
        sources["checkpoint_revision_watch"] = _load_source_with_report(
            checkpoint_revision_watch_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-revision-watch-manifest",
            report_artifact_type="acwm-m0-checkpoint-revision-watch",
            report_key="report_path",
        )
    step_report = _payload(sources, "checkpoint_step_audit", report=True)
    source_report = _payload(sources, "checkpoint_source", report=True)
    inventory_report = _payload(sources, "candidate_inventory", report=True)
    remote_watch_report = _optional_payload(sources, "checkpoint_remote_watch", report=True)
    revision_watch_report = _optional_payload(sources, "checkpoint_revision_watch", report=True)
    launch_guard = _payload(sources, "launch_guard")
    phase_gate = _payload(sources, "phase_gate")
    mismatches = _mismatch_records(step_report)
    source_records = _source_records(source_report)
    candidate_count = _int(inventory_report.get("candidate_count"))
    replacement_candidate_count = _int(source_report.get("replacement_candidate_count"))
    expected_step = _int(step_report.get("expected_step"))
    recovery_state = _recovery_state(
        mismatch_count=len(mismatches),
        source_state=str(source_report.get("state")),
        candidate_count=candidate_count,
        replacement_candidate_count=replacement_candidate_count,
        remote_candidate_available=_remote_watch_candidate_available(remote_watch_report),
        revision_candidate_available=_revision_watch_candidate_available(revision_watch_report),
    )
    primary_environment = str(mismatches[0]["environment"]) if mismatches else str(inventory_report.get("environment", "unknown"))
    active_checkpoint_path = _active_checkpoint_path(inventory_report, mismatches)
    resolved_quarantine_root = Path(quarantine_root)
    runtime = Path(runtime_python) if runtime_python is not None else _optional_path(inventory_report.get("runtime_python"))
    checkpoint_base = Path(checkpoint_root) if checkpoint_root is not None else _checkpoint_base(active_checkpoint_path, str(inventory_report.get("checkpoint_relative_path", "")))
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-recovery-packet",
        "state": recovery_state,
        "active_checkpoint_mutated": False,
        "m4_launch_allowed": phase_gate.get("m4_launch_allowed"),
        "checkpoint_blockers_active": _checkpoint_blockers(phase_gate),
        "environment": primary_environment,
        "expected_step": expected_step,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "source_resolution": {
            "state": source_report.get("state"),
            "replacement_candidate_count": replacement_candidate_count,
            "remote_unreachable_count": source_report.get("remote_unreachable_count"),
            "source_records": source_records,
            "remote_watch": _remote_watch_summary(remote_watch_report),
            "revision_watch": _revision_watch_summary(revision_watch_report),
        },
        "local_candidate_inventory": {
            "state": inventory_report.get("state"),
            "candidate_count": candidate_count,
            "rejected_count": inventory_report.get("rejected_count"),
            "searched_roots": inventory_report.get("searched_roots", []),
            "rejected": inventory_report.get("rejected", []),
        },
        "launch_guard": {
            "state": launch_guard.get("state"),
            "strict_launch_guard_pass": launch_guard.get("strict_launch_guard_pass"),
            "observed_error": launch_guard.get("observed_error"),
            "materialized_launch_plan": launch_guard.get("materialized_launch_plan"),
            "gpu_execution_started": launch_guard.get("gpu_execution_started"),
        },
        "required_candidate_contract": {
            "environment": primary_environment,
            "expected_step": expected_step,
            "must_live_under_quarantine_root": True,
            "active_checkpoint_must_not_be_candidate": True,
            "active_checkpoint_path": active_checkpoint_path,
            "candidate_path_template": str(resolved_quarantine_root / primary_environment / "latest.pt"),
            "validate_state_required": "ready_for_manual_install",
            "install_requires_explicit_confirmation": True,
            "post_install_required_checks": [
                "checkpoint step audit",
                "checkpoint source audit",
                "strict launch guard smoke",
                "M0 baseline reproduction",
                "M4 phase gate",
            ],
        },
        "commands": _commands(
            environment=primary_environment,
            expected_step=expected_step,
            quarantine_root=resolved_quarantine_root,
            runtime_python=runtime,
            checkpoint_root=checkpoint_base,
        ),
        "sources": {name: source["summary"] for name, source in sources.items()},
        "next_actions": _next_actions(recovery_state),
        "limitations": [
            "This packet is read-only; it does not download, validate, install, or replace checkpoint bytes.",
            "A recovery packet cannot clear M0. Only a validated candidate install followed by re-audits can clear the checkpoint blockers.",
            "Do not treat the current Hugging Face main cloth_move file as a replacement when source audit reports remote_current_mismatch.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _load_source_with_report(
    path: Path,
    *,
    manifest_artifact_type: str,
    report_artifact_type: str,
    report_key: str,
) -> dict[str, object]:
    manifest_source = _load_json_source(path, expected_artifact_type=manifest_artifact_type)
    manifest = manifest_source["payload"]
    if not isinstance(manifest, Mapping):
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_SOURCE_INVALID")
    report_path = manifest.get(report_key)
    if not isinstance(report_path, str) or not report_path:
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_REPORT_MISSING")
    report_source = _load_json_source(Path(report_path), expected_artifact_type=report_artifact_type)
    return {
        "manifest": manifest_source,
        "report": report_source,
        "summary": {
            "manifest": manifest_source["summary"],
            "report": report_source["summary"],
        },
    }


def _load_json_source(path: Path, *, expected_artifact_type: str) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointRecoveryPacketError(f"CHECKPOINT_RECOVERY_PACKET_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != expected_artifact_type:
        raise CheckpointRecoveryPacketError(f"CHECKPOINT_RECOVERY_PACKET_SOURCE_INVALID:{resolved}")
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _payload(sources: Mapping[str, Mapping[str, object]], name: str, *, report: bool = False) -> Mapping[str, Any]:
    source = sources[name]
    if report:
        payload = source["report"]["payload"]  # type: ignore[index]
    else:
        payload = source["payload"]
    if not isinstance(payload, Mapping):
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_SOURCE_INVALID")
    return payload


def _optional_payload(sources: Mapping[str, Mapping[str, object]], name: str, *, report: bool = False) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _payload(sources, name, report=report)


def _mismatch_records(step_report: Mapping[str, Any]) -> list[dict[str, object]]:
    records = step_report.get("records")
    if not isinstance(records, list):
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_STEP_REPORT_INVALID")
    mismatches: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_STEP_REPORT_INVALID")
        if record.get("status") == "pass":
            continue
        mismatches.append(
            {
                "environment": record.get("environment"),
                "checkpoint_relative_path": record.get("checkpoint_relative_path"),
                "checkpoint_path": record.get("checkpoint_path"),
                "expected_step": record.get("expected_step"),
                "observed_step": record.get("observed_step"),
                "status": record.get("status"),
                "huggingface_download_metadata": record.get("huggingface_download_metadata", {}),
            }
        )
    return mismatches


def _source_records(source_report: Mapping[str, Any]) -> list[dict[str, object]]:
    records = source_report.get("records")
    if not isinstance(records, list):
        return []
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        result.append(
            {
                "environment": record.get("environment"),
                "checkpoint_relative_path": record.get("checkpoint_relative_path"),
                "observed_step": record.get("observed_step"),
                "expected_step": record.get("expected_step"),
                "source_status": record.get("source_status"),
                "local_hf_hash": record.get("local_hf_hash"),
            }
        )
    return result


def _recovery_state(
    *,
    mismatch_count: int,
    source_state: str,
    candidate_count: int,
    replacement_candidate_count: int,
    remote_candidate_available: bool,
    revision_candidate_available: bool,
) -> str:
    if mismatch_count == 0:
        return "not_required"
    if candidate_count > 0:
        return "candidate_validation_required"
    if replacement_candidate_count > 0 or remote_candidate_available or revision_candidate_available:
        return "remote_candidate_download_required"
    if source_state == "remote_current_mismatch":
        return "awaiting_verified_external_candidate"
    return "source_unresolved"


def _remote_watch_candidate_available(remote_watch_report: Mapping[str, Any] | None) -> bool:
    return bool(remote_watch_report is not None and remote_watch_report.get("candidate_available") is True)


def _revision_watch_candidate_available(revision_watch_report: Mapping[str, Any] | None) -> bool:
    return bool(revision_watch_report is not None and revision_watch_report.get("candidate_available") is True)


def _remote_watch_summary(remote_watch_report: Mapping[str, Any] | None) -> dict[str, object]:
    if remote_watch_report is None:
        return {"state": "not_provided"}
    return {
        "state": remote_watch_report.get("state"),
        "observed_at_utc": remote_watch_report.get("observed_at_utc"),
        "candidate_available": remote_watch_report.get("candidate_available"),
        "downloaded_checkpoint_bytes": remote_watch_report.get("downloaded_checkpoint_bytes"),
        "active_checkpoint_mutated": remote_watch_report.get("active_checkpoint_mutated"),
        "remote_hashes": remote_watch_report.get("remote_hashes", []),
    }


def _revision_watch_summary(revision_watch_report: Mapping[str, Any] | None) -> dict[str, object]:
    if revision_watch_report is None:
        return {"state": "not_provided"}
    candidate_revisions = revision_watch_report.get("candidate_revisions")
    if not isinstance(candidate_revisions, list):
        candidate_revisions = []
    return {
        "state": revision_watch_report.get("state"),
        "observed_at_utc": revision_watch_report.get("observed_at_utc"),
        "candidate_available": revision_watch_report.get("candidate_available"),
        "candidate_revision_count": revision_watch_report.get("candidate_revision_count"),
        "scanned_revision_count": revision_watch_report.get("scanned_revision_count"),
        "downloaded_checkpoint_bytes": revision_watch_report.get("downloaded_checkpoint_bytes"),
        "active_checkpoint_mutated": revision_watch_report.get("active_checkpoint_mutated"),
        "candidate_revisions": [
            {
                "revision": item.get("revision"),
                "source": item.get("source"),
                "target_commit": item.get("target_commit"),
                "remote_hashes": item.get("remote_hashes", []),
            }
            for item in candidate_revisions
            if isinstance(item, Mapping)
        ],
    }


def _active_checkpoint_path(inventory_report: Mapping[str, Any], mismatches: Sequence[Mapping[str, object]]) -> str | None:
    value = inventory_report.get("active_checkpoint_path")
    if isinstance(value, str) and value:
        return value
    if mismatches and isinstance(mismatches[0].get("checkpoint_path"), str):
        return str(mismatches[0]["checkpoint_path"])
    return None


def _checkpoint_base(active_checkpoint_path: str | None, checkpoint_relative_path: str) -> Path | None:
    if active_checkpoint_path is None or not checkpoint_relative_path:
        return None
    active = Path(active_checkpoint_path)
    relative = Path(checkpoint_relative_path)
    try:
        parents = len(relative.parts)
        base = active
        for _ in range(parents):
            base = base.parent
        return base
    except Exception:
        return None


def _optional_path(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _checkpoint_blockers(phase_gate: Mapping[str, Any]) -> list[str]:
    blockers = phase_gate.get("blockers")
    if not isinstance(blockers, list):
        return []
    names: list[str] = []
    for item in blockers:
        if not isinstance(item, Mapping):
            continue
        requirement = item.get("requirement")
        if isinstance(requirement, str) and requirement.startswith("M0_checkpoint"):
            names.append(requirement)
    return names


def _commands(
    *,
    environment: str,
    expected_step: int,
    quarantine_root: Path,
    runtime_python: Path | None,
    checkpoint_root: Path | None,
) -> dict[str, object]:
    candidate_path = quarantine_root / environment / "latest.pt"
    validate_parts = [
        "uv",
        "run",
        "python",
        "-m",
        "wmloop.evaluate.checkpoint_quarantine",
        "validate",
        "--environment",
        environment,
        "--candidate-path",
        str(candidate_path),
        "--quarantine-root",
        str(quarantine_root),
        "--expected-step",
        str(expected_step),
    ]
    if checkpoint_root is not None:
        validate_parts.extend(["--checkpoint-root", str(checkpoint_root)])
    if runtime_python is not None:
        validate_parts.extend(["--runtime-python", str(runtime_python)])
    return {
        "candidate_path_template": str(candidate_path),
        "quarantine_validate_template": " ".join(_shell_quote(part) for part in validate_parts),
        "quarantine_install_template": (
            "uv run python -m wmloop.evaluate.checkpoint_quarantine install "
            "--quarantine-manifest <ready_manifest.json> "
            "--checkpoint-root <checkpoint_root> "
            "--output-root <install_report_root> "
            "--confirm-active-replacement"
        ),
    }


def _shell_quote(value: str) -> str:
    if value and all(character.isalnum() or character in "/._:-=+" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _next_actions(state: str) -> list[str]:
    if state == "not_required":
        return ["No checkpoint recovery action is currently required."]
    if state == "candidate_validation_required":
        return [
            "Validate each candidate in quarantine before any active replacement.",
            "Install only a ready_for_manual_install report after explicit human confirmation.",
            "Rerun all M0 checkpoint and phase-gate audits after install.",
        ]
    if state == "remote_candidate_download_required":
        return [
            "Download the differing remote or revision checkpoint into the quarantine root, not over the active checkpoint.",
            "Rerun candidate inventory and checkpoint_quarantine validate before any active replacement.",
            "Install only a ready_for_manual_install report after explicit human confirmation.",
        ]
    if state == "awaiting_verified_external_candidate":
        return [
            "Obtain a verified alternate or publisher-fixed 100k-step cloth_move checkpoint file.",
            "Place the candidate under the quarantine root and rerun candidate inventory.",
            "Do not redownload the current Hugging Face main file as a fix.",
        ]
    return [
        "Resolve checkpoint source access or provide a verified 100k-step candidate file.",
        "Keep M0 checkpoint source resolution and M4 launch blocked until quarantine validation passes.",
    ]


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Recovery Packet",
        "",
        f"State: `{report['state']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Mismatch count: `{report['mismatch_count']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Candidate Contract",
        "",
    ]
    contract = report["required_candidate_contract"]
    lines.extend(
        [
            f"- Candidate path template: `{contract['candidate_path_template']}`",
            f"- Validate state required: `{contract['validate_state_required']}`",
            f"- Install requires explicit confirmation: `{contract['install_requires_explicit_confirmation']}`",
        ]
    )
    lines.extend(["", "## Commands", ""])
    commands = report["commands"]
    lines.append(f"- Validate: `{commands['quarantine_validate_template']}`")
    lines.append(f"- Install: `{commands['quarantine_install_template']}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-recovery-packet.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-recovery-packet.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_recovery_packet_json", report_bytes, "application/json"),
                ("checkpoint_recovery_packet_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-recovery-packet-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "expected_step": report["expected_step"],
            "mismatch_count": report["mismatch_count"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "checkpoint_blockers_active": report["checkpoint_blockers_active"],
            "report_path": str(destination / "checkpoint-recovery-packet.json"),
            "markdown_path": str(destination / "checkpoint-recovery-packet.md"),
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


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CheckpointRecoveryPacketError("CHECKPOINT_RECOVERY_PACKET_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="build a checkpoint recovery packet")
    run.add_argument("--checkpoint-step-audit-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-source-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-candidate-inventory-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-launch-guard-manifest", type=Path, required=True)
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-remote-watch-manifest", type=Path)
    run.add_argument("--checkpoint-revision-watch-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--quarantine-root", type=Path, default=Path("results/quarantine/checkpoints"))
    run.add_argument("--runtime-python", type=Path)
    run.add_argument("--checkpoint-root", type=Path)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_checkpoint_recovery_packet(
            checkpoint_step_audit_manifest=args.checkpoint_step_audit_manifest,
            checkpoint_source_manifest=args.checkpoint_source_manifest,
            checkpoint_candidate_inventory_manifest=args.checkpoint_candidate_inventory_manifest,
            checkpoint_launch_guard_manifest=args.checkpoint_launch_guard_manifest,
            phase_gate_manifest=args.phase_gate_manifest,
            checkpoint_remote_watch_manifest=args.checkpoint_remote_watch_manifest,
            checkpoint_revision_watch_manifest=args.checkpoint_revision_watch_manifest,
            output_root=args.output_root,
            quarantine_root=args.quarantine_root,
            runtime_python=args.runtime_python,
            checkpoint_root=args.checkpoint_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
