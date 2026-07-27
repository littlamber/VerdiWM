"""Pure validation logic for an immutable ACWM training metadata projection.

The public ACWM release contains metadata rows that do not always map one to
one to released videos.  A training projection never mutates the release: it
rewrites accepted rows to absolute, read-only source-video paths and records a
digest over every source video it admits.  Missing rows are rejected unless a
caller explicitly allows the named released subset.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class TrainingViewError(ValueError):
    """A training projection would be ambiguous or silently widen its data."""


@dataclass(frozen=True)
class SplitProjectionReport:
    source_split: str
    source_metadata_sha256: str
    source_video_tree_sha256: str
    metadata_entries: int
    emitted_entries: int
    source_video_count: int
    missing_reference_count: int
    rewritten_path_count: int
    released_subset: bool

    def to_document(self) -> dict[str, Any]:
        return asdict(self)


def project_metadata_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_root: Path,
    allow_missing_references: bool,
) -> tuple[list[dict[str, Any]], SplitProjectionReport]:
    """Return metadata rows bound to existing, immutable source videos.

    A row is retained only when its basename names a regular video directly
    inside ``split_root``.  Absolute paths in a published metadata row are
    normalized to their local released counterpart; this is a projection, not
    a patch to the source metadata.
    """

    root = _trusted_split_root(split_root)
    videos = _released_videos(root)
    projected: list[dict[str, Any]] = []
    referenced: set[str] = set()
    missing: list[str] = []
    rewritten = 0
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise TrainingViewError(f"TRAINING_VIEW_METADATA_ROW_INVALID:{index}")
        raw_path = raw_row.get("video_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise TrainingViewError(f"TRAINING_VIEW_VIDEO_PATH_INVALID:{index}")
        name = Path(raw_path).name
        if not name or name != Path(name).name or not name.endswith(".mp4"):
            raise TrainingViewError(f"TRAINING_VIEW_VIDEO_PATH_INVALID:{index}")
        video = videos.get(name)
        if video is None:
            missing.append(name)
            continue
        row = dict(raw_row)
        normalized = str(video)
        if raw_path != normalized:
            rewritten += 1
        row["video_path"] = normalized
        projected.append(row)
        referenced.add(name)
    if missing and not allow_missing_references:
        raise TrainingViewError("TRAINING_VIEW_METADATA_REFERENCES_MISSING_VIDEO")
    extras = set(videos) - referenced
    if extras:
        raise TrainingViewError("TRAINING_VIEW_RELEASED_VIDEO_UNREFERENCED")
    if len({str(row["video_path"]) for row in projected}) != len(projected):
        raise TrainingViewError("TRAINING_VIEW_DUPLICATE_VIDEO_REFERENCE")
    if not projected:
        raise TrainingViewError("TRAINING_VIEW_EMPTY")
    report = SplitProjectionReport(
        source_split=str(root),
        source_metadata_sha256="",
        source_video_tree_sha256=_video_tree_sha256(videos),
        metadata_entries=len(rows),
        emitted_entries=len(projected),
        source_video_count=len(videos),
        missing_reference_count=len(missing),
        rewritten_path_count=rewritten,
        released_subset=bool(missing),
    )
    return projected, report


def with_metadata_digest(report: SplitProjectionReport, metadata_path: Path) -> SplitProjectionReport:
    """Bind a projection report to the exact source metadata bytes."""

    return SplitProjectionReport(
        **{**asdict(report), "source_metadata_sha256": _sha256_regular_file(Path(metadata_path))}
    )


def canonical_report_sha256(reports: Sequence[SplitProjectionReport]) -> str:
    """Stable digest over ordered split reports for the view manifest."""

    payload = [report.to_document() for report in reports]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _trusted_split_root(path: Path) -> Path:
    root = Path(path)
    try:
        metadata = os.lstat(root)
    except FileNotFoundError as exc:
        raise TrainingViewError("TRAINING_VIEW_SPLIT_MISSING") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise TrainingViewError("TRAINING_VIEW_SPLIT_UNTRUSTED")
    return root.resolve(strict=True)


def _released_videos(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.iterdir():
        if path.suffix != ".mp4":
            continue
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise TrainingViewError("TRAINING_VIEW_VIDEO_UNTRUSTED")
        if path.name in result:
            raise TrainingViewError("TRAINING_VIEW_VIDEO_DUPLICATE")
        result[path.name] = path.resolve(strict=True)
    if not result:
        raise TrainingViewError("TRAINING_VIEW_VIDEOSET_EMPTY")
    return result


def _video_tree_sha256(videos: Mapping[str, Path]) -> str:
    entries = [
        {"name": name, "size": path.stat().st_size, "sha256": _sha256_regular_file(path)}
        for name, path in sorted(videos.items())
    ]
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_regular_file(path: Path) -> str:
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise TrainingViewError("TRAINING_VIEW_MEMBER_MISSING") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TrainingViewError("TRAINING_VIEW_MEMBER_UNTRUSTED")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.lstat(path)
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise TrainingViewError("TRAINING_VIEW_MEMBER_CHANGED")
    return digest.hexdigest()
