"""Subprocess environment construction for pinned Python runtimes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def runtime_subprocess_env(
    runtime_python: Path | None,
    *,
    base: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the minimal environment needed to execute a runtime interpreter.

    Baseline execution intentionally avoids inheriting the caller's full shell
    environment.  Conda-built binary packages still need the runtime prefix's
    ``lib`` directory ahead of system libraries; otherwise packages such as
    decord can import the host ``libstdc++`` and fail only after preflight.
    """

    source = base if base is not None else os.environ
    env: dict[str, str] = {"PYTHONDONTWRITEBYTECODE": "1"}
    path = source.get("PATH", "")
    library_path = source.get("LD_LIBRARY_PATH", "")
    if runtime_python is not None:
        prefix = _runtime_prefix(runtime_python)
        if prefix is not None:
            path = _prepend_path(prefix / "bin", path)
            library_path = _prepend_path(prefix / "lib", library_path)
            env["CONDA_PREFIX"] = str(prefix)
            env["VIRTUAL_ENV"] = str(prefix)
    env["PATH"] = path
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _runtime_prefix(runtime_python: Path) -> Path | None:
    interpreter = Path(runtime_python).resolve()
    if interpreter.parent.name != "bin":
        return None
    return interpreter.parent.parent


def _prepend_path(path: Path, current: str) -> str:
    value = str(path)
    if not current:
        return value
    parts = [part for part in current.split(os.pathsep) if part]
    return os.pathsep.join([value, *[part for part in parts if part != value]])
