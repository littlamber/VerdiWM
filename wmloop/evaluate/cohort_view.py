"""Exact held-out metadata projections for the immutable ACWM evaluator.

Upstream ``eval.py`` accepts only a split name and a random ``max_trajs``
limit.  A controller that needs exact held-out cohorts must instead give it a
metadata-only view containing precisely the frozen trajectory IDs.  The view
never copies or alters source videos: accepted metadata rows point to their
absolute, read-only source files.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from wmloop.acwm_data import AcwmEnvironmentSpec
from wmloop.evaluate.plan import EvaluationSelection


class CohortViewError(ValueError):
    """A held-out selection could not be materialized exactly and safely."""


_COHORT_SPLITS = {"ind_dev": "ind_test", "ind_accept": "ind_test", "ood_accept": "ood_test"}


@dataclass(frozen=True)
class CohortProjectionReport:
    environment: str
    cohort: str
    split: str
    source_split: str
    selected_trajectory_ids_sha256: str
    source_metadata_sha256: str
    source_video_tree_sha256: str
    emitted_entries: int
    rewritten_path_count: int

    def to_document(self) -> dict[str, Any]:
        return asdict(self)


def cohort_split(cohort: str) -> str:
    split = _COHORT_SPLITS.get(cohort)
    if split is None:
        raise CohortViewError("COHORT_VIEW_COHORT_INVALID")
    return split


def trusted_source_directory(path: Path) -> Path:
    """Resolve one non-symlink source directory before any runtime deserialization."""

    return _trusted_directory(Path(path))


def trusted_source_file(path: Path) -> Path:
    """Resolve one non-symlink regular source file before any runtime deserialization."""

    return _trusted_regular_file(Path(path))


def project_cohort_metadata_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection: EvaluationSelection,
    environment_spec: AcwmEnvironmentSpec,
    source_data_root: Path,
) -> tuple[list[dict[str, Any]], CohortProjectionReport]:
    """Select exactly one frozen cohort and bind its rows to source videos."""

    if selection.environment != environment_spec.environment:
        raise CohortViewError("COHORT_VIEW_ENVIRONMENT_MISMATCH")
    split = cohort_split(selection.cohort)
    source_root = _trusted_directory(Path(source_data_root))
    expected_prefix = f"{environment_spec.dataset_relative_path}/{split}/"
    ids = tuple(selection.trajectory_ids)
    if not ids or len(ids) != len(set(ids)):
        raise CohortViewError("COHORT_VIEW_SELECTION_INVALID")
    expected_names: list[str] = []
    for trajectory_id in ids:
        if not isinstance(trajectory_id, str) or not trajectory_id.startswith(expected_prefix):
            raise CohortViewError("COHORT_VIEW_SELECTION_SCOPE_INVALID")
        relative = PurePosixPath(trajectory_id)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts) or relative.suffix != ".mp4":
            raise CohortViewError("COHORT_VIEW_SELECTION_SCOPE_INVALID")
        expected_names.append(relative.name)
    if len(expected_names) != len(set(expected_names)):
        raise CohortViewError("COHORT_VIEW_SELECTION_INVALID")
    split_root = _trusted_directory(source_root / environment_spec.dataset_relative_path / split)
    indexed_rows: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise CohortViewError(f"COHORT_VIEW_METADATA_ROW_INVALID:{index}")
        raw_path = raw_row.get("video_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise CohortViewError(f"COHORT_VIEW_VIDEO_PATH_INVALID:{index}")
        name = Path(raw_path).name
        if not name or name != Path(name).name or not name.endswith(".mp4") or name in indexed_rows:
            raise CohortViewError(f"COHORT_VIEW_VIDEO_PATH_INVALID:{index}")
        indexed_rows[name] = raw_row
    projected: list[dict[str, Any]] = []
    videos: dict[str, Path] = {}
    rewritten = 0
    for name in expected_names:
        row = indexed_rows.get(name)
        if row is None:
            raise CohortViewError("COHORT_VIEW_SELECTION_METADATA_MISSING")
        video = _trusted_regular_file(split_root / name)
        normalized = str(video)
        output = dict(row)
        if output["video_path"] != normalized:
            rewritten += 1
        output["video_path"] = normalized
        projected.append(output)
        videos[name] = video
    report = CohortProjectionReport(
        environment=selection.environment,
        cohort=selection.cohort,
        split=split,
        source_split=str(split_root),
        selected_trajectory_ids_sha256=_ids_sha256(ids),
        source_metadata_sha256="",
        source_video_tree_sha256=_video_tree_sha256(videos),
        emitted_entries=len(projected),
        rewritten_path_count=rewritten,
    )
    return projected, report


def with_metadata_digest(report: CohortProjectionReport, metadata_path: Path) -> CohortProjectionReport:
    return CohortProjectionReport(**{**asdict(report), "source_metadata_sha256": _sha256_regular_file(Path(metadata_path))})


def canonical_report_sha256(report: CohortProjectionReport) -> str:
    return hashlib.sha256(
        json.dumps(report.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _trusted_directory(path: Path) -> Path:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise CohortViewError("COHORT_VIEW_SOURCE_DIRECTORY_MISSING") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CohortViewError("COHORT_VIEW_SOURCE_DIRECTORY_UNTRUSTED")
    return path.resolve(strict=True)


def _trusted_regular_file(path: Path) -> Path:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise CohortViewError("COHORT_VIEW_SOURCE_VIDEO_MISSING") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CohortViewError("COHORT_VIEW_SOURCE_VIDEO_UNTRUSTED")
    return path.resolve(strict=True)


def _video_tree_sha256(videos: Mapping[str, Path]) -> str:
    entries = [
        {"name": name, "size": path.stat().st_size, "sha256": _sha256_regular_file(path)}
        for name, path in sorted(videos.items())
    ]
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_regular_file(path: Path) -> str:
    trusted = _trusted_regular_file(path)
    before = os.lstat(trusted)
    digest = hashlib.sha256()
    with trusted.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.lstat(trusted)
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise CohortViewError("COHORT_VIEW_SOURCE_VIDEO_CHANGED")
    return digest.hexdigest()
