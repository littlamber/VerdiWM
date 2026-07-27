"""Fail-closed runtime guard for M4-scale launch authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class M4LaunchGuardError(RuntimeError):
    """M4 launch authorization is missing, stale, or blocked."""


REQUIRED_M4_REQUIREMENTS = frozenset(
    {
        "M0_generation_zero_archive",
        "M0_baseline_reproduction",
        "M1_raw_failure_reports",
        "M1_attribution_review",
        "M1_original_g1_data_feasibility",
        "M2_vendor_and_registry_freeze",
        "M3_strict_acceptance",
        "M0_checkpoint_source_resolution",
        "M0_checkpoint_launch_guard_clear",
    }
)


@dataclass(frozen=True)
class M4LaunchAuthorization:
    phase_gate_manifest_path: Path
    phase_gate_manifest_sha256: str
    phase_gate_report_path: Path
    phase_gate_report_sha256: str
    blocker_count: int
    requirement_count: int

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-launch-authorization",
            "state": "ready",
            "phase": "M4_launch",
            "m4_launch_allowed": True,
            "blocker_count": self.blocker_count,
            "requirement_count": self.requirement_count,
            "phase_gate_manifest": {
                "path": str(self.phase_gate_manifest_path),
                "sha256": self.phase_gate_manifest_sha256,
            },
            "phase_gate_report": {
                "path": str(self.phase_gate_report_path),
                "sha256": self.phase_gate_report_sha256,
            },
        }


def verify_m4_launch_allowed(phase_gate_manifest: Path) -> M4LaunchAuthorization:
    """Load the strict phase-gate artifact and require explicit M4 readiness."""

    manifest_path, manifest, manifest_bytes = _load_json_mapping(
        phase_gate_manifest,
        "M4_LAUNCH_GATE_MANIFEST_INVALID",
    )
    _require_manifest_ready(manifest)
    report_path = _report_path_from_manifest(manifest, manifest_path=manifest_path)
    resolved_report_path, report, report_bytes = _load_json_mapping(report_path, "M4_LAUNCH_GATE_REPORT_INVALID")
    _require_report_ready(report)
    return M4LaunchAuthorization(
        phase_gate_manifest_path=manifest_path,
        phase_gate_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        phase_gate_report_path=resolved_report_path,
        phase_gate_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        blocker_count=len(_list_field(manifest.get("blockers"), "M4_LAUNCH_GATE_MANIFEST_INVALID")),
        requirement_count=len(_requirements(report)),
    )


def phase_gate_guard(phase_gate_manifest: Path):
    """Return a zero-argument guard suitable for orchestrator injection."""

    def _guard() -> dict[str, object]:
        return verify_m4_launch_allowed(phase_gate_manifest).to_document()

    return _guard


def run_m4_launch_guard(
    *,
    phase_gate_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a launch-guard receipt only after the strict M4 gate is open."""

    authorization = verify_m4_launch_allowed(phase_gate_manifest)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M4LaunchGuardError("M4_LAUNCH_GUARD_OUTPUT_EXISTS")
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else destination.parent
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m4-launch-guard-report",
        "state": "ready",
        "phase": "M4_launch",
        "m4_launch_allowed": True,
        "authorization": authorization.to_document(),
        "guarded_scope": "M4 campaign or training launch preflight",
        "limitations": [
            "This guard verifies the strict phase-gate artifact only; it does not launch training.",
            "Entry points must call this guard before provider, scheduler, or GPU side effects.",
        ],
    }
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "m4-launch-guard.json", report_bytes)
        _write_bytes_atomic(temporary / "m4-launch-guard.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-launch-guard-manifest",
            "state": "ready",
            "phase": "M4_launch",
            "m4_launch_allowed": True,
            "phase_gate_manifest": authorization.to_document()["phase_gate_manifest"],
            "phase_gate_report": authorization.to_document()["phase_gate_report"],
            "report_path": str(destination / "m4-launch-guard.json"),
            "markdown_path": str(destination / "m4-launch-guard.md"),
            "cas_refs": {"m4_launch_guard_json": report_ref, "m4_launch_guard_markdown": markdown_ref},
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


def _require_manifest_ready(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "wmloop-strict-phase-gate-manifest"
        or manifest.get("phase") != "M4_launch"
    ):
        raise M4LaunchGuardError("M4_LAUNCH_GATE_MANIFEST_INVALID")
    blockers = _list_field(manifest.get("blockers"), "M4_LAUNCH_GATE_MANIFEST_INVALID")
    if manifest.get("state") != "ready" or manifest.get("m4_launch_allowed") is not True:
        raise M4LaunchGuardError(
            f"M4_LAUNCH_GATE_NOT_READY:{manifest.get('state')}:{manifest.get('m4_launch_allowed')}"
        )
    if blockers:
        raise M4LaunchGuardError("M4_LAUNCH_GATE_HAS_BLOCKERS")


def _require_report_ready(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != "wmloop-strict-phase-gate"
        or report.get("phase") != "M4_launch"
    ):
        raise M4LaunchGuardError("M4_LAUNCH_GATE_REPORT_INVALID")
    blockers = _list_field(report.get("blockers"), "M4_LAUNCH_GATE_REPORT_INVALID")
    if report.get("state") != "ready" or report.get("m4_launch_allowed") is not True:
        raise M4LaunchGuardError(
            f"M4_LAUNCH_GATE_REPORT_NOT_READY:{report.get('state')}:{report.get('m4_launch_allowed')}"
        )
    if blockers:
        raise M4LaunchGuardError("M4_LAUNCH_GATE_REPORT_HAS_BLOCKERS")
    for name, item in _requirements(report).items():
        if not isinstance(item, Mapping) or item.get("passed") is not True:
            raise M4LaunchGuardError(f"M4_LAUNCH_GATE_REQUIREMENT_BLOCKED:{name}")


def _requirements(report: Mapping[str, Any]) -> Mapping[str, Any]:
    requirements = report.get("requirements")
    if not isinstance(requirements, Mapping) or not requirements:
        raise M4LaunchGuardError("M4_LAUNCH_GATE_REPORT_REQUIREMENTS_INVALID")
    missing = sorted(REQUIRED_M4_REQUIREMENTS.difference(str(name) for name in requirements))
    if missing:
        raise M4LaunchGuardError(f"M4_LAUNCH_GATE_REQUIREMENT_MISSING:{','.join(missing)}")
    return requirements


def _report_path_from_manifest(manifest: Mapping[str, Any], *, manifest_path: Path) -> Path:
    raw = manifest.get("report_path")
    if not isinstance(raw, str) or not raw:
        raise M4LaunchGuardError("M4_LAUNCH_GATE_REPORT_PATH_INVALID")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _list_field(value: object, error_code: str) -> list[object]:
    if not isinstance(value, list):
        raise M4LaunchGuardError(error_code)
    return value


def _load_json_mapping(path: Path, error_code: str) -> tuple[Path, Mapping[str, Any], bytes]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise M4LaunchGuardError(error_code) from exc
    if not isinstance(payload, Mapping):
        raise M4LaunchGuardError(error_code)
    return resolved, payload, payload_bytes


def _render_markdown(report: Mapping[str, Any]) -> str:
    authorization = report["authorization"]
    phase_gate_manifest = authorization["phase_gate_manifest"]
    phase_gate_report = authorization["phase_gate_report"]
    return "\n".join(
        [
            "# M4 Launch Guard",
            "",
            f"State: `{report['state']}`",
            f"M4 launch allowed: `{report['m4_launch_allowed']}`",
            f"Phase gate manifest: `{phase_gate_manifest['path']}`",
            f"Phase gate manifest sha256: `{phase_gate_manifest['sha256']}`",
            f"Phase gate report: `{phase_gate_report['path']}`",
            f"Phase gate report sha256: `{phase_gate_report['sha256']}`",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M4LaunchGuardError("M4_LAUNCH_GUARD_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="verify strict M4 launch authorization and write a guard receipt")
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    check = commands.add_parser("check", help="verify strict M4 launch authorization without writing artifacts")
    check.add_argument("--phase-gate-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            manifest = run_m4_launch_guard(
                phase_gate_manifest=args.phase_gate_manifest,
                output_root=args.output_root,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
            )
            print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "check":
            authorization = verify_m4_launch_allowed(args.phase_gate_manifest)
            print(json.dumps(authorization.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            return 0
        raise M4LaunchGuardError("M4_LAUNCH_GUARD_COMMAND_INVALID")
    except M4LaunchGuardError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
