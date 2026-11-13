"""CLI commands and configuration boundaries: inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.first_contact import infer_source_root
from wmloop.control.project_config import ProjectConfigError, load_project_config


def _default_state_root() -> Path:
    configured = os.environ.get("VERDIWM_STATE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "verdiwm"


def _load_project_intent() -> dict[str, Any]:
    try:
        return load_project_config().values
    except ProjectConfigError as exc:
        raise CampaignAPIError(str(exc)) from exc


def resolve_state_root(explicit: Path | None, configured: dict[str, Any] | None = None) -> Path:
    """Use an explicit CLI path, then project configuration, then user default."""
    if explicit is not None:
        return explicit
    values = _load_project_intent() if configured is None else configured
    return Path(str(values["state_root"])) if values.get("state_root") else _default_state_root()


def campaign_store(
    explicit: Path | None = None,
    configured: dict[str, Any] | None = None,
    *,
    read_only: bool = False,
) -> CampaignStore:
    """Open the shared campaign namespace without creating a second path rule."""

    return CampaignStore(resolve_state_root(explicit, configured) / "campaigns", read_only=read_only)


def _resolve_run_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve human intent and project defaults into the legacy run shape."""

    configured = _load_project_intent()
    model = args.model or configured.get("model")
    data = args.data or configured.get("data", configured.get("dataset"))
    if model is None:
        candidate = Path.cwd() / "model"
        model = str(candidate) if candidate.is_dir() else None
    if data is None:
        for name in ("data", "dataset"):
            candidate = Path.cwd() / name
            if candidate.exists():
                data = str(candidate)
                break
    goal = args.goal or args.intent or configured.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise CampaignAPIError("GOAL_REQUIRED")
    if not model:
        raise CampaignAPIError(
            "MODEL_PATH_REQUIRED:configure project.model or pass --model"
        )
    if not data:
        raise CampaignAPIError(
            "DATA_PATH_REQUIRED:configure project.data or pass --data"
        )
    source = args.source or configured.get("source")
    if source is None:
        source_path = infer_source_root(Path(str(model)))
        source = str(source_path) if source_path is not None else None
    return {
        "model": str(model),
        "source": str(source) if source else None,
        "data": str(data),
        "goal": goal.strip(),
        "target_metrics": (
            args.target_metrics
            if getattr(args, "target_metrics", None) is not None
            else configured.get(
                "target_metrics", configured.get("metrics", configured.get("metric"))
            )
        ),
        "budget": args.budget or configured.get("budget", "1gpu-hour"),
        "adapter": args.adapter
        if args.adapter != "auto"
        else configured.get("adapter", "auto"),
        "mode": args.mode or configured.get("mode"),
        "adapter_profile": args.adapter_profile
        or _path_configured(configured.get("adapter_profile")),
        "runtime_python": args.runtime_python
        or _path_configured(configured.get("runtime_python")),
        "state_root": resolve_state_root(args.state_root, configured),
        "campaign_id": args.campaign_id or configured.get("campaign_id"),
    }


def _path_configured(value: object) -> Path | None:
    return Path(str(value)).expanduser() if value is not None else None
