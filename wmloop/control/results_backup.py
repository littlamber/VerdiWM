"""Content-addressed incremental backups for the results directory."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ResultsBackupError(RuntimeError):
    """A results backup invariant failed closed."""


def run_results_backup(
    *,
    results_root: Path,
    backup_root: Path,
    output_root: Path,
    execute: bool = False,
    backup_id: str | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Create a dry-run or executed incremental backup manifest for ``results/``."""

    source = Path(results_root).resolve(strict=True)
    if not source.is_dir():
        raise ResultsBackupError("RESULTS_BACKUP_SOURCE_NOT_DIRECTORY")
    backup = Path(backup_root).expanduser().resolve()
    if _is_relative_to(backup, source):
        raise ResultsBackupError("RESULTS_BACKUP_ROOT_INSIDE_SOURCE")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ResultsBackupError("RESULTS_BACKUP_OUTPUT_EXISTS")
    identifier = backup_id or _default_backup_id()
    _require_backup_id(identifier)
    entries, skipped = _scan_results(source)
    previous_snapshot = _latest_snapshot(backup)
    copied_count = 0
    reused_count = 0
    copied_bytes = 0
    if execute:
        (backup / "blobs").mkdir(mode=0o700, parents=True, exist_ok=True)
        for entry in entries:
            copied = _ensure_blob(source / str(entry["path"]), backup=backup, digest=str(entry["sha256"]))
            if copied:
                copied_count += 1
                copied_bytes += int(entry["size_bytes"])
            else:
                reused_count += 1
    else:
        for entry in entries:
            if _blob_path(backup, str(entry["sha256"])).is_file():
                reused_count += 1
            else:
                copied_count += 1
                copied_bytes += int(entry["size_bytes"])
    total_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    snapshot = {
        "schema_version": 1,
        "artifact_type": "wmloop-results-incremental-backup-snapshot",
        "state": "ready" if execute else "dry_run",
        "backup_id": identifier,
        "source_root": str(source),
        "backup_root": str(backup),
        "previous_snapshot_path": str(previous_snapshot) if previous_snapshot is not None else None,
        "file_count": len(entries),
        "skipped_count": len(skipped),
        "total_bytes": total_bytes,
        "copied_count": copied_count,
        "reused_count": reused_count,
        "copied_bytes": copied_bytes,
        "tree_sha256": _tree_digest(entries),
        "files": entries,
        "skipped": skipped,
    }
    snapshot_bytes = _canonical_json_bytes(snapshot)
    snapshot_path = None
    if execute:
        snapshot_path = backup / "snapshots" / f"{identifier}.json"
        if snapshot_path.exists() or snapshot_path.is_symlink():
            raise ResultsBackupError("RESULTS_BACKUP_SNAPSHOT_EXISTS")
        snapshot_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_bytes_atomic(snapshot_path, snapshot_bytes, error_code="RESULTS_BACKUP_SNAPSHOT_EXISTS")
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else destination.parent
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-results-incremental-backup-report",
        "state": "ready" if execute else "dry_run",
        "executed": execute,
        "backup_id": identifier,
        "source_root": str(source),
        "backup_root": str(backup),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "previous_snapshot_path": snapshot["previous_snapshot_path"],
        "file_count": len(entries),
        "skipped_count": len(skipped),
        "total_bytes": total_bytes,
        "copied_count": copied_count,
        "reused_count": reused_count,
        "copied_bytes": copied_bytes,
        "tree_sha256": snapshot["tree_sha256"],
        "limitations": [
            "The snapshot is taken before this report directory is written, so the report is not included in its own backup.",
            "Only regular files are copied; symlinks and non-regular files are reported as skipped.",
            "The backup root is forbidden inside the source results tree to avoid recursive backups.",
        ],
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "results-backup.json", report_bytes, error_code="RESULTS_BACKUP_OUTPUT_EXISTS")
        _write_bytes_atomic(temporary / "results-backup.md", markdown_bytes, error_code="RESULTS_BACKUP_OUTPUT_EXISTS")
        snapshot_ref = cas.put_bytes(snapshot_bytes, media_type="application/json").uri
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            for ref in (snapshot_ref, report_ref, markdown_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-results-incremental-backup-manifest",
            "state": report["state"],
            "executed": execute,
            "backup_id": identifier,
            "source_root": str(source),
            "backup_root": str(backup),
            "snapshot_path": report["snapshot_path"],
            "snapshot_sha256": report["snapshot_sha256"],
            "previous_snapshot_path": report["previous_snapshot_path"],
            "file_count": len(entries),
            "skipped_count": len(skipped),
            "total_bytes": total_bytes,
            "copied_count": copied_count,
            "reused_count": reused_count,
            "copied_bytes": copied_bytes,
            "tree_sha256": snapshot["tree_sha256"],
            "report_path": str(destination / "results-backup.json"),
            "markdown_path": str(destination / "results-backup.md"),
            "cas_refs": {
                "results_backup_snapshot": snapshot_ref,
                "results_backup_report_json": report_ref,
                "results_backup_report_markdown": markdown_ref,
            },
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest), error_code="RESULTS_BACKUP_OUTPUT_EXISTS")
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _scan_results(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        try:
            status = path.lstat()
        except OSError as exc:
            skipped.append({"path": relative, "reason": f"lstat_failed:{exc.__class__.__name__}"})
            continue
        if stat.S_ISLNK(status.st_mode):
            skipped.append({"path": relative, "reason": "symlink"})
            continue
        if not stat.S_ISREG(status.st_mode):
            if not stat.S_ISDIR(status.st_mode):
                skipped.append({"path": relative, "reason": "non_regular"})
            continue
        digest, size = _hash_regular_file_stable(path, before=status)
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
                "mode": stat.S_IMODE(status.st_mode),
                "mtime_ns": status.st_mtime_ns,
            }
        )
    return entries, skipped


def _hash_regular_file_stable(path: Path, *, before: os.stat_result) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise ResultsBackupError(f"RESULTS_BACKUP_SOURCE_READ_FAILED:{path}") from exc
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise ResultsBackupError(f"RESULTS_BACKUP_SOURCE_FILE_CHANGED:{path}")
    if size != before.st_size:
        raise ResultsBackupError(f"RESULTS_BACKUP_SOURCE_SIZE_MISMATCH:{path}")
    return digest.hexdigest(), size


def _ensure_blob(source: Path, *, backup: Path, digest: str) -> bool:
    target = _blob_path(backup, digest)
    if target.exists() or target.is_symlink():
        _verify_blob(target, digest)
        return False
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = target.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
    copied_digest = hashlib.sha256()
    try:
        with source.open("rb") as reader:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as writer:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    copied_digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        if copied_digest.hexdigest() != digest:
            raise ResultsBackupError(f"RESULTS_BACKUP_COPY_DIGEST_MISMATCH:{source}")
        os.replace(temporary, target)
        return True
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _verify_blob(path: Path, digest: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ResultsBackupError("RESULTS_BACKUP_BLOB_INVALID")
    observed = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            observed.update(chunk)
    if observed.hexdigest() != digest:
        raise ResultsBackupError("RESULTS_BACKUP_BLOB_DIGEST_MISMATCH")


def _blob_path(backup: Path, digest: str) -> Path:
    _require_digest(digest)
    return backup / "blobs" / digest[:2] / digest


def _latest_snapshot(backup: Path) -> Path | None:
    snapshots = backup / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(path for path in snapshots.glob("*.json") if path.is_file() and not path.is_symlink())
    return candidates[-1].resolve() if candidates else None


def _tree_digest(entries: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        line = f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Results Incremental Backup",
            "",
            f"State: `{report['state']}`",
            f"Executed: `{report['executed']}`",
            f"Backup id: `{report['backup_id']}`",
            f"Files: `{report['file_count']}`",
            f"Total bytes: `{report['total_bytes']}`",
            f"Copied: `{report['copied_count']}` files / `{report['copied_bytes']}` bytes",
            f"Reused: `{report['reused_count']}` files",
            f"Snapshot: `{report['snapshot_path']}`",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _default_backup_id() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _require_backup_id(value: str) -> None:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
        raise ResultsBackupError("RESULTS_BACKUP_ID_INVALID")


def _require_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ResultsBackupError("RESULTS_BACKUP_DIGEST_INVALID")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes, *, error_code: str) -> None:
    if path.exists() or path.is_symlink():
        raise ResultsBackupError(error_code)
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
    run = commands.add_parser("run", help="create a dry-run or executed results backup manifest")
    run.add_argument("--results-root", type=Path, required=True)
    run.add_argument("--backup-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--backup-id")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_results_backup(
            results_root=args.results_root,
            backup_root=args.backup_root,
            output_root=args.output_root,
            execute=args.execute,
            backup_id=args.backup_id,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise ResultsBackupError("RESULTS_BACKUP_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
