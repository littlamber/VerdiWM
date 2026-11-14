"""Read-only onboarding scan components: runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.control.onboarding_scan.io import _bounded
from wmloop.control.onboarding_scan.types import (
    DEFAULT_RUNTIME_TIMEOUT_SECONDS,
    _MAX_DEPENDENCIES,
    _PACKAGE_IMPORT_ALIASES,
    _PIP_PLATFORM_WARNING,
    validate_runtime_timeout,
)
from wmloop.process_lifecycle import LocalProcessTree, retire_process_group


def _discover_runtime(
    repo: Path,
    dependencies: Sequence[Mapping[str, object]],
    *,
    runtime_python: Path | None,
    probe_imports: bool,
    runtime_timeout_seconds: float = DEFAULT_RUNTIME_TIMEOUT_SECONDS,
) -> dict[str, object]:
    timeout_seconds = validate_runtime_timeout(runtime_timeout_seconds)
    candidates = _runtime_candidates(repo, runtime_python)
    selected = (
        candidates[0]
        if runtime_python is not None and candidates
        else next(
            (candidate for candidate in candidates if candidate["exists"]),
            None,
        )
    )
    if selected is None:
        return {
            "state": "blocked",
            "selected_python": None,
            "candidates": [],
            "required_packages": _required_packages(dependencies),
            "import_probes": [],
            "pip_check": {
                "state": "not_run",
                "returncode": None,
                "detail": "no Python runtime found",
            },
        }
    required = _required_packages(dependencies)
    probes = (
        _probe_imports(selected, required, repo=repo, timeout_seconds=timeout_seconds)
        if probe_imports and selected["exists"]
        else []
    )
    pip_check = (
        _probe_pip(selected, repo=repo, timeout_seconds=timeout_seconds)
        if probe_imports and selected["exists"]
        else {
            "state": "not_run",
            "returncode": None,
            "detail": "disabled or runtime missing",
        }
    )
    failures = [probe for probe in probes if probe["state"] != "ok"]
    state = (
        "ready"
        if selected["exists"]
        and not failures
        and pip_check["state"] in {"ok", "warning", "not_run"}
        else "blocked"
    )
    return {
        "state": state,
        "selected_python": selected["path"] if selected["exists"] else None,
        "candidates": candidates,
        "required_packages": required,
        "import_probes": probes,
        "pip_check": pip_check,
        "cuda_hidden": True,
        "probe_timeout_seconds": timeout_seconds,
    }


def _runtime_candidates(repo: Path, explicit: Path | None) -> list[dict[str, object]]:
    raw: list[tuple[Path, str]] = []
    if explicit is not None:
        # Preserve a venv launcher symlink: resolving it changes ``sys.prefix``
        # and can silently switch to the base interpreter without the venv's
        # site-packages.  We only need an absolute path for durable reports.
        raw.append((Path(explicit).expanduser().absolute(), "explicit"))
    for relative in (".venv/bin/python", "venv/bin/python", "env/bin/python"):
        raw.append((repo / relative, "repository_local"))
    raw.append((Path(sys.executable).resolve(), "current_process"))
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for path, source in raw:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        exists = path.is_file() and os.access(path, os.X_OK)
        version = _python_version(path) if exists else None
        rows.append(
            {"path": key, "source": source, "exists": exists, "version": version}
        )
    return rows


def _python_version(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _bounded((result.stdout or result.stderr).strip(), limit=80)


def _required_packages(dependencies: Sequence[Mapping[str, object]]) -> list[str]:
    values: set[str] = set()
    for dependency in dependencies:
        for package in dependency.get("declared_packages", []):
            values.add(str(package).lower())
    return sorted(values)[:_MAX_DEPENDENCIES]


def _probe_imports(
    path_info: Mapping[str, object], packages: Sequence[str], *, repo: Path,
    timeout_seconds: float = DEFAULT_RUNTIME_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    if not packages:
        return []
    path = Path(str(path_info["path"]))
    rows: list[dict[str, object]] = []
    for package in packages:
        module = _PACKAGE_IMPORT_ALIASES.get(package, package.replace("-", "_"))
        code = "import importlib; importlib.import_module(" + repr(module) + ")"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONPATH": str(repo),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
        }
        try:
            result = _run_runtime_command(
                [str(path), "-c", code],
                timeout_seconds=timeout_seconds,
                cwd=repo,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            rows.append(
                {
                    "package": package,
                    "module": module,
                    "state": "timeout",
                    "returncode": None,
                    "detail": _bounded(str(exc)),
                }
            )
            continue
        except OSError as exc:
            rows.append(
                {
                    "package": package,
                    "module": module,
                    "state": "error",
                    "returncode": None,
                    "detail": _bounded(str(exc)),
                }
            )
            continue
        state = "ok" if result.returncode == 0 else "failed"
        detail = _bounded((result.stderr or result.stdout).strip())
        rows.append(
            {
                "package": package,
                "module": module,
                "state": state,
                "returncode": result.returncode,
                "detail": detail,
            }
        )
    return rows


def _probe_pip(
    path_info: Mapping[str, object], *, repo: Path,
    timeout_seconds: float = DEFAULT_RUNTIME_TIMEOUT_SECONDS,
) -> dict[str, object]:
    path = Path(str(path_info["path"]))
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "disabled",
    }
    try:
        result = _run_runtime_command(
            [str(path), "-m", "pip", "check"],
            timeout_seconds=timeout_seconds,
            cwd=repo,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "error", "returncode": None, "detail": _bounded(str(exc))}
    detail = _bounded(
        "\n".join(
            line.strip()
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if line.strip()
        )
    )
    if result.returncode == 0:
        state = "ok"
    else:
        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        state = (
            "warning"
            if lines and all(_PIP_PLATFORM_WARNING.fullmatch(line) for line in lines)
            else "failed"
        )
    return {"state": state, "returncode": result.returncode, "detail": detail}


def _run_runtime_command(
    command: Sequence[str], *, timeout_seconds: float, cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Retire the complete probe group on timeout, cancellation and success."""
    timeout_seconds = validate_runtime_timeout(timeout_seconds)
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout, tempfile.TemporaryFile(
        mode='w+', encoding='utf-8'
    ) as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, env=dict(env), stdout=stdout,
            stderr=stderr, text=True, start_new_session=True,
        )
        tree = None
        try:
            tree = LocalProcessTree(process)
            tree.wait(timeout=timeout_seconds)
        finally:
            retire_process_group(process, tree=tree)
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(command, process.returncode, stdout.read(), stderr.read())
