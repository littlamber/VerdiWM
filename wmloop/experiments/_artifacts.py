"""Atomic artifact bundles shared by experiment planners and reporters."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ExperimentArtifactError(RuntimeError):
    """An experiment artifact bundle could not be published safely."""


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_bundle(
    *,
    output_root: Path,
    files: Mapping[str, bytes],
    manifest_fields: Mapping[str, object],
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ExperimentArtifactError("EXPERIMENT_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        refs: dict[str, str] = {}
        for relative, payload in sorted(files.items()):
            safe = _safe_relative_path(relative)
            target = temporary / safe
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_bytes_atomic(target, payload)
            media_type = _media_type(safe)
            ref = cas.put_bytes(payload, media_type=media_type).uri
            refs[relative] = ref
            if archive is not None:
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            **manifest_fields,
            "files": sorted(files),
            "cas_refs": refs,
        }
        _write_bytes_atomic(temporary / "manifest.json", canonical_json(manifest))
        os.replace(temporary, destination)
        return {**manifest, "manifest_path": str(destination / "manifest.json")}
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("manifest.json"):
        raise ExperimentArtifactError("EXPERIMENT_BUNDLE_PATH_INVALID")
    return path


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".tex": "text/x-tex",
    }.get(suffix, "application/octet-stream")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
