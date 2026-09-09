"""Small filesystem primitives shared by local state and publication adapters."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def checked_path(value: Path, *, code: str, error: type[ValueError] = ValueError) -> Path:
    """Reject symlinks before resolution, including linked parent directories."""
    raw = Path(os.path.abspath(Path(value).expanduser()))
    if any(part.is_symlink() for part in (raw, *raw.parents)):
        raise error(code)
    return raw.resolve()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_file_lock(path: Path):
    """Process lock; kernel releases it after crashes. Never unlink a lock file."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def require_exact_tree(root: Path, members: set[str], *, code: str, error: type[ValueError] = ValueError) -> None:
    expected_dirs = {p.as_posix() for name in members for p in Path(name).parents if p != Path(".")}
    files, directories = set(), set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise error(code)
        name = path.relative_to(root).as_posix()
        if path.is_file():
            files.add(name)
        elif path.is_dir():
            directories.add(name)
        else:
            raise error(code)
    if files != members or directories != expected_dirs:
        raise error(code)
