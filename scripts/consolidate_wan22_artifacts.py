#!/usr/bin/env python3
"""Consolidate WAN22 outputs into one idea-owned, recoverable directory.

The default is a dry run. ``--apply`` only moves explicitly classified
top-level WAN22 directories; it never deletes data and writes a mapping
manifest so paths can be recovered or updated downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Sequence


def plan_layout(parent: Path, *, idea_id: str) -> list[tuple[Path, Path, str]]:
    parent = parent.expanduser().resolve(strict=True)
    destination = parent / "wan22-droid-experiments" / idea_id
    moves: list[tuple[Path, Path, str]] = []
    runs = parent / "wan22-droid-runs"
    if runs.is_dir() and not runs.is_symlink():
        moves.append((runs, destination / "runs", "canonical experiment runs"))
    for source in sorted(parent.glob("wan22-worldarena-input*")):
        if source.is_dir() and not source.is_symlink():
            moves.append((source, destination / "worldarena" / "staging" / source.name, "WorldArena staging"))
    for prefix in (
        "wan22-literature-",
        "wan22-mechanism-",
        "wan22-method-",
    ):
        for source in sorted(parent.glob(prefix + "*")):
            if source.is_dir() and not source.is_symlink():
                moves.append((source, destination / "research" / source.name, "research support artifact"))
    return moves


def apply_layout(
    parent: Path, *, idea_id: str, archive_label: str | None = None
) -> dict[str, object]:
    moves = plan_layout(parent, idea_id=idea_id)
    destination = parent.expanduser().resolve() / "wan22-droid-experiments" / idea_id
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("WAN22_ARTIFACT_LAYOUT_DESTINATION_NOT_EMPTY")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source, target, category in moves:
        if target.exists() or target.is_symlink():
            raise ValueError(f"WAN22_ARTIFACT_LAYOUT_TARGET_EXISTS:{target}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        records.append({"source": str(source), "destination": str(target), "category": category})
        if category == "canonical experiment runs":
            records.extend(_organize_run_children(target))
    # Receipts created before run classification often already point at the
    # canonical ``runs/<name>`` parent. Add aliases for those pre-classified
    # paths so references are repaired after the child is moved under
    # ``runs/pilot`` or ``runs/confirm`` as well.
    repair_mappings = [
        (str(record["source"]), str(record["destination"]))
        for record in records
    ]
    for record in records:
        if record.get("category") != "run classification":
            continue
        source = Path(str(record["source"]))
        destination_path = Path(str(record["destination"]))
        repair_mappings.append(
            (str(destination / "runs" / source.name), str(destination_path))
        )
    reference_repair = repair_references(
        destination,
        repair_mappings,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-artifact-layout-manifest",
        "state": "applied",
        "idea_id": idea_id,
        "created_date": archive_label or date.today().isoformat(),
        "record_count": len(records),
        "moves": records,
        "reference_repair": reference_repair,
        "recoverability": "All operations are directory moves; reverse the source/destination mapping to restore names.",
        "claim_boundary": "Directory consolidation changes storage layout only; it does not change experiment evidence or scientific claims.",
    }
    path = destination / "artifact-layout.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(destination)
    return manifest


def repair_references(
    root: Path, mappings: Sequence[tuple[str, str]]
) -> dict[str, object]:
    """Rewrite moved absolute paths in text receipts without touching provenance.

    ``artifact-layout.json`` is deliberately excluded: its source paths are the
    recoverability record and must continue to describe the pre-move locations.
    Binary media is skipped, while JSON/YAML/log/text receipts are repaired in
    place through a same-directory atomic replacement.
    """

    root = Path(root).expanduser().resolve(strict=True)
    normalized = sorted(
        (
            str(Path(source).expanduser().resolve()),
            str(Path(destination).expanduser().resolve()),
        )
        for source, destination in mappings
        if source and destination and source != destination
    )
    normalized.sort(key=lambda item: len(item[0]), reverse=True)
    changed_files = 0
    replacement_count = 0
    skipped_binary = 0
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name in {"artifact-layout.json", "index.json"}
        ):
            continue
        original = path.read_bytes()
        if b"\x00" in original:
            skipped_binary += 1
            continue
        updated = original
        replacements = 0
        for source, destination in normalized:
            count = updated.count(source.encode("utf-8"))
            if count:
                updated = updated.replace(
                    source.encode("utf-8"), destination.encode("utf-8")
                )
                replacements += count
        if updated == original:
            continue
        temporary = tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        try:
            temporary.write(updated)
            temporary.close()
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        changed_files += 1
        replacement_count += replacements
    return {
        "state": "repaired",
        "root": str(root),
        "changed_file_count": changed_files,
        "replacement_count": replacement_count,
        "skipped_binary_file_count": skipped_binary,
    }


def archive_staging(root: Path) -> dict[str, object]:
    """Move unclassified WorldArena staging under a recoverable legacy area."""

    root = Path(root).expanduser().resolve(strict=True)
    staging = root / "worldarena" / "staging"
    archive = root / "worldarena" / "archive" / "legacy"
    if not staging.is_dir() or staging.is_symlink():
        return {"state": "nothing_to_archive", "moves": []}
    archive.mkdir(mode=0o700, parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for source in sorted(staging.iterdir()):
        if source.name.startswith("."):
            continue
        target = archive / source.name
        if target.exists() or target.is_symlink():
            raise ValueError(f"WAN22_ARTIFACT_LAYOUT_TARGET_EXISTS:{target}")
        shutil.move(str(source), str(target))
        records.append({
            "source": str(source),
            "destination": str(target),
            "category": "WorldArena legacy staging",
        })
    if not any(staging.iterdir()):
        staging.rmdir()
    repair = repair_references(
        root,
        [(item["source"], item["destination"]) for item in records],
    )
    return {"state": "archived", "moves": records, "reference_repair": repair}


def write_index(root: Path) -> dict[str, object]:
    """Write a compact inventory so cleanup never turns into evidence loss."""

    root = Path(root).expanduser().resolve(strict=True)
    files: list[dict[str, object]] = []
    by_digest: dict[str, list[str]] = {}
    total_bytes = 0
    ignored_dirs = {
        "generated_dataset", "gt_dataset", "video", "output",
        "output_action_following", "cas", "checkpoints", "frames",
    }
    candidate_paths: list[Path] = []
    for current, directories, names in __import__("os").walk(root):
        directories[:] = sorted(name for name in directories if name not in ignored_dirs and not name.startswith("."))
        candidate_paths.extend(Path(current) / name for name in sorted(names))
    for path in sorted(candidate_paths):
        if not path.is_file() or path.is_symlink() or path.name in {"index.json"}:
            continue
        size = path.stat().st_size
        total_bytes += size
        # Hash receipts/configuration, while retaining a deterministic metadata
        # fingerprint for large media blobs. This keeps indexing cheap and
        # still makes accidental duplicate JSON artifacts visible.
        digest: str | None = None
        if path.suffix.lower() in {".json", ".yaml", ".yml", ".txt", ".sha256"}:
            digest = _sha256(path)
            by_digest.setdefault(digest, []).append(str(path.relative_to(root)))
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": size,
            "sha256": digest,
            "media_fingerprint": f"{size}:{path.stat().st_mtime_ns}" if digest is None else None,
        })
    duplicates = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(by_digest.items()) if len(paths) > 1
    ]
    index = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-artifact-index",
        "state": "indexed",
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "duplicate_content_groups": duplicates,
        "claim_boundary": "This inventory describes storage and duplicate content only; it does not select or alter scientific evidence.",
    }
    (root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _organize_run_children(runs_root: Path) -> list[dict[str, object]]:
    """Keep active formal runs named, and quarantine probe/failed attempts."""

    records: list[dict[str, object]] = []
    destinations = {
        "long-ema-anchor-balanced-20260902": runs_root / "confirm",
        "long-ema-anchor-20260902": runs_root / "pilot",
        "summaries": runs_root / "summaries",
    }
    archive = runs_root / "archive" / "legacy"
    for child in sorted(runs_root.iterdir()):
        if child.name in {"archive", "pilot", "confirm", "summaries"}:
            continue
        target_root = destinations.get(child.name, archive)
        target = target_root / child.name
        target_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError(f"WAN22_ARTIFACT_LAYOUT_RUN_TARGET_EXISTS:{target}")
        shutil.move(str(child), str(target))
        records.append({"source": str(child), "destination": str(target), "category": "run classification"})
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--idea-id", default="wan22_droid_acwm_v1")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--index-root", type=Path, help="index an already consolidated idea directory")
    parser.add_argument("--repair-root", type=Path, help="repair moved absolute paths using artifact-layout.json")
    parser.add_argument("--archive-staging-root", type=Path, help="move old WorldArena staging below worldarena/archive/legacy")
    args = parser.parse_args(argv)
    if args.archive_staging_root is not None:
        root = args.archive_staging_root.expanduser().resolve(strict=True)
        layout_path = root / "artifact-layout.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        result = archive_staging(root)
        moves = layout.setdefault("worldarena_archive_moves", [])
        if not isinstance(moves, list):
            raise ValueError("WAN22_ARTIFACT_LAYOUT_ARCHIVE_MOVES_INVALID")
        moves.extend(result.get("moves", []))
        layout["worldarena_archive_reference_repair"] = result.get("reference_repair")
        layout_path.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_index(root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.repair_root is not None:
        root = args.repair_root.expanduser().resolve(strict=True)
        layout_path = root / "artifact-layout.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        moves = layout.get("moves")
        if not isinstance(moves, list):
            raise ValueError("WAN22_ARTIFACT_LAYOUT_MOVES_INVALID")
        mappings = [
            (str(item["source"]), str(item["destination"]))
            for item in moves
            if isinstance(item, dict) and "source" in item and "destination" in item
        ]
        result = repair_references(root, mappings)
        layout["reference_repair"] = result
        layout_path.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_index(root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.index_root is not None:
        print(json.dumps(write_index(args.index_root), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.parent is None:
        parser.error("--parent is required unless --index-root is used")
    moves = plan_layout(args.parent, idea_id=args.idea_id)
    if not args.apply:
        print(json.dumps({
            "state": "dry_run",
            "destination": str(args.parent.expanduser().resolve() / "wan22-droid-experiments" / args.idea_id),
            "moves": [{"source": str(source), "destination": str(target), "category": category} for source, target, category in moves],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(json.dumps(apply_layout(args.parent, idea_id=args.idea_id, archive_label=args.label), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
