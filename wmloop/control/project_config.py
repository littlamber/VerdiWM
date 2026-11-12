"""Resolve user-owned project facts for intent-first product entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wmloop.runtime_env import runtime_python_path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


class ProjectConfigError(ValueError):
    """A project configuration cannot be interpreted safely."""


@dataclass(frozen=True)
class ProjectConfig:
    source: Path | None
    values: dict[str, Any]


_ALLOWED_KEYS = frozenset(
    {
        "model",
        "source",
        "data",
        "dataset",
        "goal",
        "metric",
        "metrics",
        "target_metrics",
        "budget",
        "adapter",
        "mode",
        "adapter_profile",
        "runtime_python",
        "evaluator_contract",
        "state_root",
        "campaign_id",
    }
)
_PATH_KEYS = frozenset(
    {"model", "source", "data", "dataset", "adapter_profile", "runtime_python", "evaluator_contract", "state_root"}
)


def load_project_config(*, cwd: Path | None = None) -> ProjectConfig:
    """Load the nearest project file, resolving its paths against that file."""

    source = find_project_config(cwd=cwd)
    if source is None:
        return ProjectConfig(source=None, values={})
    try:
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectConfigError("PROJECT_CONFIG_INVALID") from exc
    project = payload.get("project", payload)
    if not isinstance(project, Mapping):
        raise ProjectConfigError("PROJECT_CONFIG_INVALID")
    unknown = sorted(set(project) - _ALLOWED_KEYS)
    if unknown:
        raise ProjectConfigError(f"PROJECT_CONFIG_KEY_UNKNOWN:{','.join(unknown)}")
    values = dict(project)
    _validate_values(values)
    for key in _PATH_KEYS.intersection(values):
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise ProjectConfigError(f"PROJECT_CONFIG_VALUE_INVALID:{key}")
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else source.parent / path
        # Resolving bin/python symlinks switches a virtualenv to its base Python.
        values[key] = str(runtime_python_path(candidate) if key == "runtime_python" else candidate.resolve())
    return ProjectConfig(source=source, values=values)


def _validate_values(values: Mapping[str, Any]) -> None:
    """Reject mistyped user configuration before command-specific processing."""
    for key in (_PATH_KEYS | {"goal", "adapter", "campaign_id", "mode"}).intersection(values):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ProjectConfigError(f"PROJECT_CONFIG_VALUE_INVALID:{key}")
    for key in {"metric", "metrics", "target_metrics"}.intersection(values):
        value = values[key]
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ProjectConfigError(f"PROJECT_CONFIG_VALUE_INVALID:{key}")
    if "mode" in values:
        from wmloop.control.research_modes import ResearchModeError, normalize_research_mode

        try:
            normalize_research_mode(values["mode"])
        except ResearchModeError as exc:
            raise ProjectConfigError("PROJECT_CONFIG_VALUE_INVALID:mode") from exc
    if "budget" in values:
        from wmloop.control.adapter_profiles import AdapterProfileError, parse_gpu_budget

        try:
            parse_gpu_budget(values["budget"])
        except AdapterProfileError as exc:
            raise ProjectConfigError("PROJECT_CONFIG_VALUE_INVALID:budget") from exc


def find_project_config(*, cwd: Path | None = None) -> Path | None:
    configured = os.environ.get("VERDIWM_PROJECT_CONFIG")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ProjectConfigError("PROJECT_CONFIG_NOT_FOUND")
        return path
    current = (cwd or Path.cwd()).expanduser().resolve()
    for parent in (current, *current.parents):
        candidate = parent / "verdiwm.toml"
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    fallback = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser() / "verdiwm" / "project.toml"
    return fallback if fallback.is_file() and not fallback.is_symlink() else None
