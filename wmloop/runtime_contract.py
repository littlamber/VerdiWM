from __future__ import annotations

import hashlib
from pathlib import Path


RUNTIME_MANIFEST_NAME = "wmloop-runtime-manifest.json"


def runtime_tree_sha256(root: Path) -> str:
    """Hash executable runtime files while excluding transient metadata."""

    runtime_root = Path(root)
    digest = hashlib.sha256()
    for item in sorted(runtime_root.rglob("*"), key=lambda value: value.relative_to(runtime_root).as_posix()):
        relative = item.relative_to(runtime_root)
        if (
            item.is_symlink()
            or not item.is_file()
            or item.name == RUNTIME_MANIFEST_NAME
            or ".git" in relative.parts
            or "__pycache__" in relative.parts
            or item.suffix in {".pyc", ".pyo"}
        ):
            continue
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
