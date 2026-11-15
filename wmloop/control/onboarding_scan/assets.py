"""Read-only onboarding scan components: assets."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from wmloop.control.onboarding_scan.files import _relative
from wmloop.control.onboarding_scan.io import _bounded, _safe_size
from wmloop.control.onboarding_scan.types import (
    _CHECKPOINT_SUFFIXES,
    _DATA_METADATA_SUFFIXES,
    _DEPENDENCY_FILENAMES,
    _MAX_ASSETS,
    _MAX_SOURCE_REVISION_FILES,
    _SOURCE_OR_DOCUMENT_SUFFIXES,
)


_DATASET_DIRECTORY_MARKERS = frozenset(
    {
        "annotation",
        "annotations",
        "data",
        "dataset",
        "datasets",
        "droid",
        "latent",
        "latent_videos",
        "latents",
        "meta",
        "metadata",
        "shard",
        "shards",
        "stat",
        "stats",
        "subset",
        "test",
        "train",
        "training",
        "trajectory",
        "trajectories",
        "val",
        "validation",
        "video",
        "videos",
    }
)
_DATASET_FILENAME_MARKERS = frozenset(
    {
        "annotation",
        "annotations",
        "dataset",
        "index",
        "manifest",
        "metadata",
        "meta",
        "stat",
        "stats",
        "test",
        "train",
        "trajectory",
        "val",
        "validation",
    }
)
_CHECKPOINT_DIRECTORY_MARKERS = frozenset(
    {"checkpoint", "checkpoints", "ckpt", "weights", "weight"}
)
_CHECKPOINT_FILENAME_MARKERS = frozenset(
    {"checkpoint", "ckpt", "ema", "model", "pytorch", "weight", "weights"}
)
_CAMERA_CONFIG_MARKERS = frozenset(
    {"calibration", "camera", "cameras", "extrinsics", "intrinsics"}
)
_CONFIG_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})


def _discover_assets(files: Sequence[Path], repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        suffix = path.suffix.lower()
        relative = Path(_relative(path, repo)).as_posix().lower()
        segments = relative.split("/")
        directory_tokens = {
            token
            for segment in segments[:-1]
            for token in re.split(r"[^a-z0-9]+", segment)
            if token
        }
        filename_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", path.stem.lower())
            if token
        }
        path_tokens = directory_tokens | filename_tokens
        latent_shard = bool(
            path_tokens & {"latent", "latents"}
            and path_tokens & {"shard", "shards"}
        )
        dataset_marker = bool(
            directory_tokens & _DATASET_DIRECTORY_MARKERS
            or filename_tokens & _DATASET_FILENAME_MARKERS
        )
        camera_config = (
            suffix in _CONFIG_SUFFIXES
            and bool(path_tokens & _CAMERA_CONFIG_MARKERS)
        )
        checkpoint_directory = bool(
            directory_tokens & _CHECKPOINT_DIRECTORY_MARKERS
        )
        checkpoint_filename = bool(
            filename_tokens & _CHECKPOINT_FILENAME_MARKERS
        )
        checkpoint = not latent_shard and (
            (
                suffix in _CHECKPOINT_SUFFIXES - {".bin"}
                and (checkpoint_directory or checkpoint_filename or not dataset_marker)
            )
            or (
                suffix == ".bin"
                and (checkpoint_directory or checkpoint_filename)
            )
        )
        dataset = (
            latent_shard
            or (
                not camera_config
                and dataset_marker
                and suffix in _DATA_METADATA_SUFFIXES
            )
        )
        if suffix in _SOURCE_OR_DOCUMENT_SUFFIXES:
            dataset = False
        model_dependency = bool(path_tokens & {"vae", "clip", "svd"})
        if not (checkpoint or dataset or model_dependency):
            continue
        if checkpoint:
            kind = "checkpoint"
        elif dataset:
            kind = "dataset_or_metadata"
        else:
            kind = "model_dependency"
        rows.append(
            {
                "path": _relative(path, repo),
                "kind": kind,
                "size_bytes": _safe_size(path),
                "large_file": _safe_size(path) >= 1_000_000_000,
                "hashed": False,
            }
        )
    rows.sort(key=lambda item: str(item["path"]))
    return rows[:_MAX_ASSETS]


def _discover_source_revision(repo: Path, files: Sequence[Path]) -> dict[str, object]:
    git = shutil.which("git")
    if git is not None:
        try:
            result = subprocess.run(
                [git, "-C", str(repo), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None:
            revision = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
                return {
                    "state": "bound",
                    "kind": "git_commit",
                    "revision": revision.lower(),
                    "detail": "git rev-parse HEAD",
                }
    return _source_tree_revision(repo, files)


def _source_tree_revision(repo: Path, files: Sequence[Path]) -> dict[str, object]:
    selected = [path for path in files if _is_source_revision_file(path, repo)]
    selected.sort(key=lambda path: _relative(path, repo))
    if not selected or len(selected) > _MAX_SOURCE_REVISION_FILES:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": "source tree is empty or exceeds the revision file limit",
        }
    digest = hashlib.sha256()
    try:
        for path in selected:
            relative = _relative(path, repo).encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(hashlib.sha256(payload).digest())
    except OSError as exc:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": _bounded(str(exc)),
        }
    return {
        "state": "bound",
        "kind": "source_tree_sha256",
        "revision": digest.hexdigest(),
        "detail": f"content hash over {len(selected)} source and configuration files",
    }


def _is_source_revision_file(path: Path, repo: Path) -> bool:
    # Dataset loaders and trajectory transforms are executable dependencies,
    # even when their module names match data-asset directory markers.
    if path.suffix.lower() in {".py", ".pyi", ".sh"}:
        return True
    relative = _relative(path, repo).lower()
    if path.name.lower() in _DEPENDENCY_FILENAMES:
        return True
    if (set(Path(relative).parts) & {"config", "configs"}
        and path.suffix.lower() in {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}):
        return True
    parts = set(re.split(r"[^a-z0-9]+", relative))
    if parts & {"dataset", "annotation", "trajectory", "videos", "latent_videos"}:
        return False
    return path.suffix.lower() in {
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".ini",
    }
