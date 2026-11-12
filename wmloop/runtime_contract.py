from __future__ import annotations

from pathlib import Path

from wmloop.storage import sha256_bytes, sha256_file


RUNTIME_MANIFEST_NAME = "wmloop-runtime-manifest.json"


def runtime_tree_sha256(root: Path) -> str:
    """Hash executable runtime files while excluding transient metadata."""

    runtime_root = Path(root)
    parts: list[bytes] = []
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
        parts.extend((len(encoded).to_bytes(8, "big"), encoded, bytes.fromhex(sha256_file(item))))
    return sha256_bytes(b"".join(parts))


def _file_sha256(path: Path) -> str:
    """Compatibility alias for older callers; new code uses storage.sha256_file."""

    return sha256_file(path)
