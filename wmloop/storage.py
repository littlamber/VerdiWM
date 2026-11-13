"""Small filesystem primitives shared by local state and publication adapters."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_bytes(value: object, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_bytes(value: object, *, ensure_ascii: bool = True) -> bytes:
    """Serialize a JSON artifact using the repository-wide canonical form."""

    return canonical_bytes(value, ensure_ascii=ensure_ascii) + b"\n"


def sha256_bytes(value: bytes) -> str:
    """Return the content digest used by CAS and immutable receipts."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("STORAGE_CHUNK_SIZE_INVALID")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    """Load one JSON document; callers own domain-specific error translation."""

    return load_json_bytes(Path(path).read_bytes())


def load_json_bytes(payload: bytes) -> object:
    """Decode one UTF-8 JSON snapshot without rereading its source path."""

    return json.loads(payload.decode("utf-8"))


def checked_path(value: Path, *, code: str, error: type[Exception] = ValueError) -> Path:
    """Reject symlinks before resolution, including linked parent directories."""
    raw = Path(os.path.abspath(Path(value).expanduser()))
    if any(part.is_symlink() for part in (raw, *raw.parents)):
        raise error(code)
    return raw.resolve()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
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
def exclusive_file_lock(path: Path, *, blocking: bool = True):
    """Process lock; kernel releases it after crashes. Never unlink a lock file."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
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
