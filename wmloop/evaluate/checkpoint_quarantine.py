"""Validate quarantined M0 checkpoint candidates before active replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.runtime_env import runtime_subprocess_env


class CheckpointQuarantineError(RuntimeError):
    """A quarantined checkpoint candidate could not be validated safely."""


def validate_checkpoint_candidate(
    *,
    environment: str,
    candidate_path: Path,
    quarantine_root: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    output_root: Path,
    expected_step: int = 100000,
    expected_sha256: str | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Validate one candidate without mutating the active checkpoint tree."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_EXPECTED_STEP_INVALID")
    normalized_expected_sha256 = _optional_sha256(expected_sha256)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_OUTPUT_EXISTS")
    spec = _environment_spec(environment)
    quarantine = _directory(quarantine_root, "CHECKPOINT_QUARANTINE_ROOT_INVALID")
    candidate = _regular_file(candidate_path, "CHECKPOINT_QUARANTINE_CANDIDATE_MISSING")
    if not _is_relative_to(candidate, quarantine):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_CANDIDATE_OUTSIDE_ROOT")
    root = _directory(checkpoint_root, "CHECKPOINT_QUARANTINE_CHECKPOINT_ROOT_INVALID")
    active = _regular_file(root / spec.checkpoint_relative_path, "CHECKPOINT_QUARANTINE_ACTIVE_MISSING")
    if candidate == active:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_CANDIDATE_IS_ACTIVE")
    runtime = _runtime_executable(runtime_python)
    active_sha256 = _sha256_file(active)
    candidate_sha256 = _sha256_file(candidate)
    observed_step = _checkpoint_step_with_runtime(runtime, candidate)
    step_pass = observed_step == expected_step
    hash_pass = normalized_expected_sha256 is None or candidate_sha256 == normalized_expected_sha256
    if step_pass and hash_pass:
        state = "ready_for_manual_install"
    elif not step_pass:
        state = "candidate_step_mismatch"
    else:
        state = "candidate_hash_mismatch"
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-quarantine-report",
        "state": state,
        "environment": spec.environment,
        "checkpoint_relative_path": spec.checkpoint_relative_path,
        "expected_step": expected_step,
        "active_checkpoint_mutated": False,
        "active_checkpoint": {
            "path": str(active),
            "size_bytes": active.stat().st_size,
            "sha256": active_sha256,
            "huggingface_download_metadata": _huggingface_download_metadata(root, spec.checkpoint_relative_path),
        },
        "candidate": {
            "path": str(candidate),
            "quarantine_root": str(quarantine),
            "size_bytes": candidate.stat().st_size,
            "sha256": candidate_sha256,
            "expected_sha256": normalized_expected_sha256,
            "observed_step": observed_step,
            "step_pass": step_pass,
            "hash_pass": hash_pass,
        },
        "install_plan": _install_plan(active=active, candidate=candidate, candidate_sha256=candidate_sha256)
        if state == "ready_for_manual_install"
        else None,
        "next_actions": _next_actions(state),
        "limitations": [
            "This command never overwrites the active checkpoint.",
            "A ready report is permission to perform a separately reviewed replacement, not proof that M0 baseline reproduction has passed.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def install_validated_checkpoint(
    *,
    quarantine_manifest_path: Path,
    checkpoint_root: Path,
    output_root: Path,
    confirm_active_replacement: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Install a previously validated candidate, preserving an active backup."""

    if not confirm_active_replacement:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_CONFIRMATION_REQUIRED")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_OUTPUT_EXISTS")
    manifest = _load_quarantine_manifest(quarantine_manifest_path)
    report = _load_quarantine_report(Path(str(manifest["report_path"])))
    spec = _environment_spec(str(report["environment"]))
    if manifest["state"] != "ready_for_manual_install" or report["state"] != "ready_for_manual_install":
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_NOT_READY")
    if manifest["checkpoint_relative_path"] != spec.checkpoint_relative_path:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    if report["active_checkpoint_mutated"] is not False:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    install_plan = report.get("install_plan")
    if not isinstance(install_plan, Mapping) or install_plan.get("manual_review_required") is not True:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    root = _directory(checkpoint_root, "CHECKPOINT_QUARANTINE_CHECKPOINT_ROOT_INVALID")
    active = _regular_file(root / spec.checkpoint_relative_path, "CHECKPOINT_QUARANTINE_ACTIVE_MISSING")
    active_report = _mapping_field(report, "active_checkpoint")
    candidate_report = _mapping_field(report, "candidate")
    if Path(str(active_report["path"])).resolve(strict=True) != active:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_ACTIVE_MISMATCH")
    active_sha256_before = _sha256_file(active)
    expected_active_sha256 = _string_field(active_report, "sha256", "CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    if active_sha256_before != expected_active_sha256:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_ACTIVE_CHANGED")
    candidate = _regular_file(Path(str(candidate_report["path"])), "CHECKPOINT_QUARANTINE_CANDIDATE_MISSING")
    quarantine = _directory(Path(str(candidate_report["quarantine_root"])), "CHECKPOINT_QUARANTINE_ROOT_INVALID")
    if not _is_relative_to(candidate, quarantine):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_CANDIDATE_OUTSIDE_ROOT")
    candidate_sha256 = _sha256_file(candidate)
    expected_candidate_sha256 = _string_field(candidate_report, "sha256", "CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    if candidate_sha256 != expected_candidate_sha256:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_CANDIDATE_CHANGED")
    backup = active.with_name(f"{active.name}.backup-before-{candidate_sha256[:12]}")
    if backup.exists() or backup.is_symlink():
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_BACKUP_EXISTS")
    active_parent = _directory(active.parent, "CHECKPOINT_QUARANTINE_ACTIVE_DIRECTORY_INVALID")
    report_payload: dict[str, object]
    try:
        os.replace(active, backup)
        _copy_regular_file_atomic(candidate, active, directory=active_parent)
    except Exception:
        if not active.exists() and backup.is_file() and not backup.is_symlink():
            os.replace(backup, active)
        raise
    active_sha256_after = _sha256_file(active)
    if active_sha256_after != candidate_sha256:
        os.replace(backup, active)
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_DIGEST_MISMATCH")
    report_payload = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-install-report",
        "state": "installed",
        "environment": spec.environment,
        "checkpoint_relative_path": spec.checkpoint_relative_path,
        "active_checkpoint_mutated": True,
        "quarantine_manifest_path": str(Path(quarantine_manifest_path).resolve(strict=True)),
        "active_checkpoint_path": str(active),
        "backup_checkpoint_path": str(backup),
        "candidate_path": str(candidate),
        "active_sha256_before": active_sha256_before,
        "active_sha256_after": active_sha256_after,
        "candidate_sha256": candidate_sha256,
        "post_install_required_checks": list(install_plan.get("post_install_required_checks", ())),
        "next_actions": [
            "Run checkpoint step audit against the active checkpoint root.",
            "Run strict M0 launch creation with expected checkpoint step 100000.",
            "Rerun M0 baseline evaluation, reproduction report, and M4 phase gate.",
        ],
    }
    return _write_install_report_bundle(report=report_payload, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _environment_spec(environment: str):
    if not isinstance(environment, str) or not environment:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_ENVIRONMENT_INVALID")
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_ENVIRONMENT_UNKNOWN")


def _directory(path: Path, code: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise CheckpointQuarantineError(code) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise CheckpointQuarantineError(code)
    return resolved


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise CheckpointQuarantineError(code)
    return candidate.resolve(strict=True)


def _runtime_executable(path: Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_RUNTIME_MISSING") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_RUNTIME_MISSING")
    return resolved


def _checkpoint_step_with_runtime(runtime_python: Path, checkpoint_path: Path) -> int:
    source = (
        "import json, math, sys, torch\n"
        "payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "step = payload.get('step') if isinstance(payload, dict) else None\n"
        "ok = isinstance(step, (int, float)) and not isinstance(step, bool) and math.isfinite(float(step)) and int(step) == float(step)\n"
        "print(json.dumps({'step': int(step) if ok else None}))\n"
        "sys.exit(0 if ok else 2)\n"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", source, str(checkpoint_path)],
        env=runtime_subprocess_env(runtime_python),
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_STEP_UNREADABLE")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_STEP_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_STEP_UNREADABLE")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_STEP_UNREADABLE")
    return step


def _huggingface_download_metadata(checkpoint_root: Path, relative_path: str) -> dict[str, object]:
    metadata_path = checkpoint_root / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return {"status": "missing", "path": str(metadata_path)}
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_HF_METADATA_INVALID") from exc
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_HF_METADATA_INVALID")
    return {
        "status": "available",
        "path": str(metadata_path),
        "download_commit": lines[0],
        "etag_or_lfs_sha256": lines[1],
        "timestamp": lines[2] if len(lines) > 2 else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_EXPECTED_SHA256_INVALID")
    return normalized


def _install_plan(*, active: Path, candidate: Path, candidate_sha256: str) -> dict[str, object]:
    backup = active.with_name(f"{active.name}.backup-before-{candidate_sha256[:12]}")
    return {
        "manual_review_required": True,
        "active_checkpoint_path": str(active),
        "candidate_path": str(candidate),
        "suggested_backup_path": str(backup),
        "post_install_required_checks": [
            "Run checkpoint step audit against the active checkpoint root.",
            "Run strict M0 launch creation with expected checkpoint step 100000.",
            "Rerun M0 baseline evaluation, reproduction report, and M4 phase gate.",
        ],
    }


def _next_actions(state: str) -> list[str]:
    if state == "ready_for_manual_install":
        return [
            "Review the install plan and replace the active checkpoint only after human approval.",
            "After replacement, rerun checkpoint step audit, launch guard smoke, baseline reproduction, and phase gate.",
        ]
    if state == "candidate_step_mismatch":
        return ["Reject this candidate; it does not have the expected checkpoint step."]
    if state == "candidate_hash_mismatch":
        return ["Reject this candidate or rerun validation with the correct expected SHA-256 from a verified source."]
    return ["Keep the active checkpoint unchanged."]


def _render_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["candidate"]
    active = report["active_checkpoint"]
    lines = [
        "# M0 Checkpoint Quarantine",
        "",
        f"State: `{report['state']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        "",
        "## Candidate",
        "",
        f"- Path: `{candidate['path']}`",
        f"- SHA-256: `{candidate['sha256']}`",
        f"- Observed step: `{candidate['observed_step']}`",
        f"- Step pass: `{candidate['step_pass']}`",
        f"- Hash pass: `{candidate['hash_pass']}`",
        "",
        "## Active",
        "",
        f"- Path: `{active['path']}`",
        f"- SHA-256: `{active['sha256']}`",
        "",
        "## Next Actions",
        "",
    ]
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
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
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-quarantine.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-quarantine.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_quarantine_json", report_bytes, "application/json"),
                ("checkpoint_quarantine_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-quarantine-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "checkpoint_relative_path": report["checkpoint_relative_path"],
            "expected_step": report["expected_step"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "report_path": str(destination / "checkpoint-quarantine.json"),
            "markdown_path": str(destination / "checkpoint-quarantine.md"),
            "cas_refs": cas_refs,
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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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


def _write_install_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_install_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-install.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-install.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_install_json", report_bytes, "application/json"),
                ("checkpoint_install_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-install-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "checkpoint_relative_path": report["checkpoint_relative_path"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "active_checkpoint_path": report["active_checkpoint_path"],
            "backup_checkpoint_path": report["backup_checkpoint_path"],
            "report_path": str(destination / "checkpoint-install.json"),
            "markdown_path": str(destination / "checkpoint-install.md"),
            "cas_refs": cas_refs,
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


def _render_install_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Install",
        "",
        f"State: `{report['state']}`",
        f"Environment: `{report['environment']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        f"Active checkpoint: `{report['active_checkpoint_path']}`",
        f"Backup checkpoint: `{report['backup_checkpoint_path']}`",
        f"Active SHA-256 before: `{report['active_sha256_before']}`",
        f"Active SHA-256 after: `{report['active_sha256_after']}`",
        "",
        "## Required Checks",
        "",
    ]
    for item in report["post_install_required_checks"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _load_quarantine_manifest(path: Path) -> Mapping[str, object]:
    payload = _load_json_mapping(path, "CHECKPOINT_QUARANTINE_INSTALL_MANIFEST_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "acwm-m0-checkpoint-quarantine-manifest":
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_MANIFEST_INVALID")
    return payload


def _load_quarantine_report(path: Path) -> Mapping[str, object]:
    payload = _load_json_mapping(path, "CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "acwm-m0-checkpoint-quarantine-report":
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    return payload


def _load_json_mapping(path: Path, code: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointQuarantineError(code) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointQuarantineError(code)
    return payload


def _mapping_field(payload: Mapping[str, Any], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise CheckpointQuarantineError("CHECKPOINT_QUARANTINE_INSTALL_REPORT_INVALID")
    return value


def _string_field(payload: Mapping[str, Any], name: str, code: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise CheckpointQuarantineError(code)
    return value


def _copy_regular_file_atomic(source: Path, destination: Path, *, directory: Path) -> None:
    temporary = directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a quarantined checkpoint candidate")
    validate.add_argument("--environment", required=True)
    validate.add_argument("--candidate-path", type=Path, required=True)
    validate.add_argument("--quarantine-root", type=Path, required=True)
    validate.add_argument("--checkpoint-root", type=Path, required=True)
    validate.add_argument("--runtime-python", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)
    validate.add_argument("--expected-step", type=int, default=100000)
    validate.add_argument("--expected-sha256")
    validate.add_argument("--archive-db", type=Path)
    validate.add_argument("--cas-root", type=Path)
    install = commands.add_parser("install", help="install a ready quarantine manifest after explicit confirmation")
    install.add_argument("--quarantine-manifest", type=Path, required=True)
    install.add_argument("--checkpoint-root", type=Path, required=True)
    install.add_argument("--output-root", type=Path, required=True)
    install.add_argument("--confirm-active-replacement", action="store_true")
    install.add_argument("--archive-db", type=Path)
    install.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        manifest = validate_checkpoint_candidate(
            environment=args.environment,
            candidate_path=args.candidate_path,
            quarantine_root=args.quarantine_root,
            checkpoint_root=args.checkpoint_root,
            runtime_python=args.runtime_python,
            output_root=args.output_root,
            expected_step=args.expected_step,
            expected_sha256=args.expected_sha256,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    if args.command == "install":
        manifest = install_validated_checkpoint(
            quarantine_manifest_path=args.quarantine_manifest,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            confirm_active_replacement=args.confirm_active_replacement,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
