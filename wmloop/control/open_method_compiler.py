"""Compile an open Method IR into a content-addressed candidate overlay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.control.open_method_ir import (
    OpenMethodIRError,
    build_candidate_overlay,
    validate_method_ir,
)


class OpenMethodCompilerError(RuntimeError):
    """An open candidate workspace crossed an isolation or binding boundary."""


def compile_candidate_overlay(
    *,
    method_ir: Mapping[str, object],
    base_revision: Mapping[str, object],
    workspace: Path,
    output_path: Path,
    tests: Sequence[Mapping[str, object]],
    project_root: Path,
    file_roles: Mapping[str, str] | None = None,
    execution_contract_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Hash an isolated workspace and emit an overlay receipt.

    This compiler does not execute generated code. Execution is a later stage
    owned by the experiment compiler and verifier.
    """
    try:
        validate_method_ir(method_ir, root=project_root)
    except OpenMethodIRError as exc:
        raise OpenMethodCompilerError(f"OPEN_METHOD_IR_INVALID:{exc}") from exc
    root = Path(project_root).expanduser().resolve()
    source = Path(workspace).expanduser().resolve()
    if source == root or root in source.parents or source.is_symlink() or not source.is_dir():
        raise OpenMethodCompilerError("OPEN_METHOD_WORKSPACE_NOT_ISOLATED")
    destination = Path(output_path).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise OpenMethodCompilerError("OPEN_METHOD_OUTPUT_INSIDE_SOURCE")
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source).as_posix()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise OpenMethodCompilerError("OPEN_METHOD_WORKSPACE_PATH_INVALID")
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in pure.parts):
            continue
        role = str(file_roles[relative]) if file_roles and relative in file_roles else _role(relative)
        if role not in {"implementation", "configuration", "test", "descriptor"}:
            raise OpenMethodCompilerError("OPEN_METHOD_WORKSPACE_ROLE_INVALID")
        files.append({
            "relative_path": relative,
            "sha256": _sha256(path),
            "role": role,
        })
    if not files:
        raise OpenMethodCompilerError("OPEN_METHOD_WORKSPACE_EMPTY")
    try:
        overlay = build_candidate_overlay(
            method_ir=method_ir,
            base_revision=base_revision,
            files=files,
            tests=tests,
            execution_contract_binding=execution_contract_binding,
            state="proposed",
        )
    except OpenMethodIRError as exc:
        raise OpenMethodCompilerError(f"OPEN_METHOD_OVERLAY_INVALID:{exc}") from exc
    _write_atomic(destination, overlay)
    return overlay


def _role(relative: str) -> str:
    if "test" in Path(relative).name.lower() or "/tests/" in f"/{relative}/":
        return "test"
    suffix = Path(relative).suffix.lower()
    if suffix in {".py", ".cc", ".cpp", ".cu", ".c", ".rs"}:
        return "implementation"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "configuration"
    return "descriptor"


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
