"""Inventory local M0 checkpoint replacement candidates without installing them."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointCandidateInventoryError(RuntimeError):
    """Checkpoint candidate inventory evidence could not be produced safely."""


def generate_checkpoint_candidate_inventory(
    *,
    environment: str,
    candidate_roots: Sequence[Path],
    checkpoint_root: Path,
    runtime_python: Path,
    output_root: Path,
    expected_step: int = 100000,
    max_depth: int = 8,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only inventory of local candidate checkpoints."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_EXPECTED_STEP_INVALID")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_MAX_DEPTH_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_OUTPUT_EXISTS")
    if not candidate_roots:
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_ROOTS_EMPTY")
    spec = _environment_spec(environment)
    checkpoint_base = _directory(checkpoint_root, "CHECKPOINT_CANDIDATE_INVENTORY_CHECKPOINT_ROOT_INVALID")
    active_checkpoint = _regular_file(
        checkpoint_base / spec.checkpoint_relative_path,
        "CHECKPOINT_CANDIDATE_INVENTORY_ACTIVE_MISSING",
    )
    runtime = _runtime_executable(runtime_python)
    aliases = _candidate_aliases(spec.environment, spec.checkpoint_relative_path)
    root_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for root in candidate_roots:
        root_record, found, rejected = _scan_root(
            root=Path(root),
            active_checkpoint=active_checkpoint,
            aliases=aliases,
            max_depth=max_depth,
        )
        root_records.append(root_record)
        rejected_records.extend(rejected)
        candidates.extend(
            _candidate_record(
                path=item,
                root=Path(str(root_record["resolved_path"])),
                spec_environment=spec.environment,
                quarantine_root=Path(root),
                checkpoint_root=checkpoint_base,
                runtime_python=runtime,
                expected_step=expected_step,
            )
            for item in found
        )
    candidates.sort(key=lambda item: str(item["path"]))
    state = "candidate_found" if candidates else "no_candidates_found"
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-candidate-inventory",
        "state": state,
        "environment": spec.environment,
        "checkpoint_relative_path": spec.checkpoint_relative_path,
        "expected_step": expected_step,
        "active_checkpoint_path": str(active_checkpoint),
        "active_checkpoint_mutated": False,
        "runtime_python": str(runtime),
        "max_depth": max_depth,
        "search_aliases": aliases,
        "searched_roots": root_records,
        "candidate_count": len(candidates),
        "rejected_count": len(rejected_records),
        "candidates": candidates,
        "rejected": rejected_records,
        "next_actions": _next_actions(state),
        "limitations": [
            "This inventory never loads checkpoint tensors and never infers training step.",
            "A listed file is only a candidate. It must pass wmloop.evaluate.checkpoint_quarantine validate before any install.",
            "Active checkpoint paths are excluded so the current mismatched checkpoint cannot be reintroduced as a replacement candidate.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _environment_spec(environment: str):
    if not isinstance(environment, str) or not environment:
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_ENVIRONMENT_INVALID")
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_ENVIRONMENT_UNKNOWN")


def _directory(path: Path, code: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise CheckpointCandidateInventoryError(code) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise CheckpointCandidateInventoryError(code)
    return resolved


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise CheckpointCandidateInventoryError(code)
    return candidate.resolve(strict=True)


def _runtime_executable(path: Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_RUNTIME_MISSING") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_RUNTIME_MISSING")
    return resolved


def _scan_root(
    *,
    root: Path,
    active_checkpoint: Path,
    aliases: tuple[str, ...],
    max_depth: int,
) -> tuple[dict[str, object], list[Path], list[dict[str, object]]]:
    root_display = str(root)
    if root.is_symlink():
        return {"path": root_display, "status": "skipped_symlink", "resolved_path": root_display, "matched_file_count": 0}, [], []
    if not root.exists():
        return {"path": root_display, "status": "missing", "resolved_path": root_display, "matched_file_count": 0}, [], []
    if not root.is_dir():
        return {"path": root_display, "status": "not_directory", "resolved_path": root_display, "matched_file_count": 0}, [], []
    resolved = root.resolve(strict=True)
    found: list[Path] = []
    rejected: list[dict[str, object]] = []
    for current, directories, filenames in os.walk(resolved, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.relative_to(resolved).parts)
        if depth >= max_depth:
            directories[:] = []
        directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                rejected.append({"path": str(path), "reason": "symlink"})
                continue
            if path.suffix != ".pt":
                continue
            try:
                resolved_file = path.resolve(strict=True)
            except OSError:
                rejected.append({"path": str(path), "reason": "changed_during_scan"})
                continue
            if resolved_file == active_checkpoint:
                rejected.append({"path": str(resolved_file), "reason": "active_checkpoint"})
                continue
            if not _matches_alias(resolved_file, aliases):
                continue
            if not _looks_like_checkpoint_candidate(resolved_file):
                rejected.append({"path": str(resolved_file), "reason": "not_checkpoint_shaped"})
                continue
            found.append(resolved_file)
    return (
        {"path": root_display, "status": "scanned", "resolved_path": str(resolved), "matched_file_count": len(found)},
        found,
        rejected,
    )


def _candidate_record(
    *,
    path: Path,
    root: Path,
    spec_environment: str,
    quarantine_root: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    expected_step: int,
) -> dict[str, object]:
    stats = path.stat()
    return {
        "path": str(path),
        "size_bytes": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
        "relative_to_search_root": _relative_or_none(path, root),
        "suggested_quarantine_root": str(Path(quarantine_root).resolve(strict=True)),
        "suggested_validate_command": _suggested_validate_command(
            environment=spec_environment,
            candidate_path=path,
            quarantine_root=Path(quarantine_root).resolve(strict=True),
            checkpoint_root=checkpoint_root,
            runtime_python=runtime_python,
            expected_step=expected_step,
        ),
    }


def _relative_or_none(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _candidate_aliases(environment: str, checkpoint_relative_path: str) -> tuple[str, ...]:
    parent = Path(checkpoint_relative_path).parent.as_posix()
    raw = {
        environment.lower(),
        environment.lower().replace("_", ""),
        parent.lower(),
        _compact_key(parent),
    }
    return tuple(sorted(item for item in raw if item))


def _matches_alias(path: Path, aliases: tuple[str, ...]) -> bool:
    raw = str(path).lower()
    compact = _compact_key(raw)
    return any(alias in raw or _compact_key(alias) in compact for alias in aliases)


def _looks_like_checkpoint_candidate(path: Path) -> bool:
    compact = _compact_key(str(path))
    if path.name == "latest.pt":
        return True
    return any(marker in compact for marker in ("checkpoint", "ckpt", "quarantine", "videodit"))


def _compact_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _suggested_validate_command(
    *,
    environment: str,
    candidate_path: Path,
    quarantine_root: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    expected_step: int,
) -> str:
    parts = [
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
        "--checkpoint-root",
        str(checkpoint_root),
        "--runtime-python",
        str(runtime_python),
        "--expected-step",
        str(expected_step),
    ]
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    if value and all(character.isalnum() or character in "/._:-=+" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _next_actions(state: str) -> list[str]:
    if state == "candidate_found":
        return [
            "Run wmloop.evaluate.checkpoint_quarantine validate for each candidate before considering replacement.",
            "Install only a ready_for_manual_install quarantine report, and only with explicit human confirmation.",
            "After any install, rerun checkpoint step audit, source audit, launch guard smoke, M0 reproduction, and M4 phase gate.",
        ]
    return [
        "No local replacement candidate was found in the searched roots.",
        "Place a verified alternate 100k-step checkpoint under a quarantine root and rerun this inventory.",
        "Keep M0 checkpoint source resolution and M4 launch blocked until a candidate passes quarantine validation.",
    ]


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Candidate Inventory",
        "",
        f"State: `{report['state']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Candidate count: `{report['candidate_count']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        "",
        "| Root | Status | Matches |",
        "|:--|:--|--:|",
    ]
    for root in report["searched_roots"]:
        lines.append(f"| `{root['path']}` | `{root['status']}` | {root['matched_file_count']} |")
    lines.extend(["", "## Candidates", ""])
    if report["candidates"]:
        for candidate in report["candidates"]:
            lines.append(f"- `{candidate['path']}` ({candidate['size_bytes']} bytes)")
    else:
        lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
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
        raise CheckpointCandidateInventoryError("CHECKPOINT_CANDIDATE_INVENTORY_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-candidate-inventory.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-candidate-inventory.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_candidate_inventory_json", report_bytes, "application/json"),
                ("checkpoint_candidate_inventory_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-candidate-inventory-manifest",
            "state": report["state"],
            "environment": report["environment"],
            "checkpoint_relative_path": report["checkpoint_relative_path"],
            "expected_step": report["expected_step"],
            "candidate_count": report["candidate_count"],
            "rejected_count": report["rejected_count"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "report_path": str(destination / "checkpoint-candidate-inventory.json"),
            "markdown_path": str(destination / "checkpoint-candidate-inventory.md"),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="inventory local checkpoint replacement candidates")
    run.add_argument("--environment", required=True)
    run.add_argument("--candidate-root", type=Path, action="append", required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--max-depth", type=int, default=8)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_checkpoint_candidate_inventory(
            environment=args.environment,
            candidate_roots=args.candidate_root,
            checkpoint_root=args.checkpoint_root,
            runtime_python=args.runtime_python,
            output_root=args.output_root,
            expected_step=args.expected_step,
            max_depth=args.max_depth,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
