"""CLI commands and configuration boundaries: configuration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adapter_profiles import AdapterProfileError
from wmloop.execute.configured_llm_broker import ConfiguredBrokerError, load_config
from wmloop.experiments.training_scale import TrainingScaleError


def _configured_llm_adapter(config_path: Path | None) -> dict[str, object]:
    source = _resolve_llm_config(config_path)
    config = load_config(source)
    return {
        "command": [
            sys.executable,
            "-m",
            "wmloop.execute.configured_llm_broker",
            "{request_path}",
            "{response_path}",
            "--config",
            str(source),
        ],
        "timeout_seconds": float(config["timeout_seconds"]),
        "max_output_bytes": int(config["maximum_bytes"]),
        "provider_alias": "configured-openai-compatible",
        "model_alias": str(config["model"]),
        "credential_environment_keys": [],
    }


def _resolve_llm_config(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    configured = os.environ.get("VERDIWM_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    for name in ("verdiwm.toml", "verdiwm.config.toml", "config.toml"):
        candidate = Path.cwd() / name
        if candidate.is_file() and not candidate.is_symlink():
            try:
                load_config(candidate)
            except ConfiguredBrokerError:
                continue
            return candidate.resolve()
    return (Path.home() / ".config" / "verdiwm" / "config.toml").resolve()


def _is_repairable_campaign_error(error: BaseException) -> bool:
    """Follow the domain-error chain for a provider-neutral repair decision."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AdapterProfileError) and current.repairable:
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_training_scale_plan(path: Path, *, root: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID") from exc
    if not isinstance(value, dict):
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID")
    try:
        validate_document(
            "training_scale_plan", value, root=root.expanduser().resolve()
        )
    except ContractValidationError as exc:
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID") from exc
    if value.get("state") != "ready":
        raise TrainingScaleError("TRAINING_SCALE_PLAN_NOT_READY")
    return value
