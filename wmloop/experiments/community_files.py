"""Bounded, symlink-free discovery of local semantic documents."""

from __future__ import annotations

import os
from pathlib import Path


class CommunityExportError(ValueError):
    """A semantic export input or immutable output invariant failed."""


def semantic_files(sources, destination, *, max_files, max_entries=100_000, max_depth=64):
    """Count entries while walking; never materialize the source tree to sort it."""
    files = entries = 0

    def walk(directory, depth):
        nonlocal files, entries
        if directory == destination:
            return
        if depth > max_depth:
            raise CommunityExportError("COMMUNITY_EXPORT_DEPTH_LIMIT_EXCEEDED")
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > max_entries:
                        raise CommunityExportError("COMMUNITY_EXPORT_ENTRY_LIMIT_EXCEEDED")
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if path == destination:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        yield from walk(path, depth + 1)
                    elif entry.is_file(follow_symlinks=False) and path.suffix in {".json", ".jsonl"}:
                        files += 1
                        if files > max_files:
                            raise CommunityExportError("COMMUNITY_EXPORT_FILE_LIMIT_EXCEEDED")
                        yield path
        except OSError as exc:
            raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_READ_FAILED") from exc

    for source in sorted(sources):
        yield from walk(source, 0)


def bounded_read(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        import stat
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_READ_FAILED")
        return source.read(maximum + 1)
