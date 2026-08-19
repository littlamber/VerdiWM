#!/usr/bin/env python3
"""Run the OpenAI-compatible broker from a project-owned config and auth file."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10, supported by this project
    import tomli as tomllib  # type: ignore[no-redef]

from wmloop.execute.openai_compatible_llm_broker import (
    OpenAICompatibleBrokerError,
    forward,
)


class ConfiguredBrokerError(RuntimeError):
    """The VerdiWM LLM config is missing or invalid."""


def default_config_path() -> Path:
    configured = os.environ.get("VERDIWM_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "verdiwm" / "config.toml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = (path or default_config_path()).expanduser()
    try:
        info = source.lstat()
        if source.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            raise ConfiguredBrokerError("VERDIWM_CONFIG_INVALID")
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    except ConfiguredBrokerError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfiguredBrokerError("VERDIWM_CONFIG_INVALID") from exc
    llm = payload.get("llm") if isinstance(payload, Mapping) else None
    if not isinstance(llm, Mapping):
        raise ConfiguredBrokerError("VERDIWM_CONFIG_LLM_SECTION_MISSING")
    return _normalize_config(dict(llm), config_path=source)


def _normalize_config(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    endpoint = raw.get("endpoint")
    base_url = raw.get("base_url")
    if (endpoint is None) == (base_url is None):
        raise ConfiguredBrokerError("VERDIWM_CONFIG_ENDPOINT_INVALID")
    if (endpoint is not None and not isinstance(endpoint, str)) or (
        base_url is not None and not isinstance(base_url, str)
    ):
        raise ConfiguredBrokerError("VERDIWM_CONFIG_ENDPOINT_INVALID")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfiguredBrokerError("VERDIWM_CONFIG_MODEL_INVALID")
    api_style = raw.get("api_style", "responses")
    if api_style not in {"responses", "chat_completions"}:
        raise ConfiguredBrokerError("VERDIWM_CONFIG_API_STYLE_INVALID")
    token_file = raw.get("token_file", "auth")
    if not isinstance(token_file, str) or not token_file.strip():
        raise ConfiguredBrokerError("VERDIWM_CONFIG_TOKEN_FILE_INVALID")
    token_path = Path(token_file).expanduser()
    if not token_path.is_absolute():
        token_path = config_path.parent / token_path
    token_environment_key = raw.get(
        "token_environment_key", "VERDIWM_LLM_BROKER_TOKEN"
    )
    if not isinstance(token_environment_key, str):
        raise ConfiguredBrokerError("VERDIWM_CONFIG_TOKEN_KEY_INVALID")
    return {
        "endpoint": endpoint,
        "base_url": base_url,
        "model": model,
        "api_style": api_style,
        "reasoning_effort": raw.get("reasoning_effort"),
        "token_environment_key": token_environment_key,
        "token_file": token_path,
        "timeout_seconds": raw.get("timeout_seconds", 180.0),
        "maximum_bytes": raw.get("maximum_bytes", 524288),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("response_path", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    forward(
        request_path=args.request_path,
        response_path=args.response_path,
        endpoint=config["endpoint"],
        base_url=config["base_url"],
        model=config["model"],
        token_environment_key=config["token_environment_key"],
        token_file=config["token_file"],
        api_style=config["api_style"],
        reasoning_effort=config["reasoning_effort"],
        timeout_seconds=float(config["timeout_seconds"]),
        maximum_bytes=int(config["maximum_bytes"]),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
