"""Optional HTTP providers; no vendor SDK is required by the Kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tomllib
import urllib.parse
import urllib.request
import time
from dataclasses import dataclass
from typing import Any

from .retrieval import SearchHit


@dataclass
class OpenAICompatibleProvider:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    provider_id: str = "openai-compatible"
    role_models: dict[str, str] | None = None
    max_retries: int = 2
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider | None":
        settings = _load_local_settings()
        base_url = os.getenv("VERDI_AI_BASE_URL") or settings.get("base_url")
        model = os.getenv("VERDI_AI_MODEL") or settings.get("model")
        if not base_url or not model:
            return None
        role_models = None
        raw_role_models = os.getenv("VERDI_AI_ROLE_MODELS")
        if raw_role_models:
            try:
                parsed = json.loads(raw_role_models)
                role_models = {str(key): str(value) for key, value in parsed.items()}
            except (json.JSONDecodeError, AttributeError):
                role_models = None
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.getenv("VERDI_AI_API_KEY") or settings.get("api_key"),
            role_models=role_models,
            reasoning_effort=os.getenv("VERDI_AI_REASONING_EFFORT") or settings.get("reasoning_effort"),
        )

    def complete(self, *, role: str, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {"model": (self.role_models or {}).get(role, self.model), "messages": [{"role": "system", "content": role}, {"role": "user", "content": prompt}], "temperature": 0}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "User-Agent": "VerdiWM/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                content = value["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("AI response content is empty")
                return content
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.25 * (2 ** attempt))
        raise RuntimeError(f"OpenAI-compatible provider failed after retries: {last_error}")


def _load_local_settings() -> dict[str, str]:
    """Load opt-in user settings without requiring a shell to source a file.

    Environment variables remain authoritative. The parser intentionally accepts
    only simple ``export NAME=value`` assignments from ``ai.env`` and the small
    ``[llm]`` table from ``config.toml``; arbitrary shell code is never executed.
    """
    result: dict[str, str] = {}
    config_dir = Path(os.getenv("VERDI_CONFIG_DIR", Path.home() / ".config" / "verdiwm"))
    env_file = config_dir / "ai.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(?:export\s+)?(VERDI_AI_[A-Z_]+)=(?:\"([^\"]*)\"|'([^']*)'|(.+?))\s*$", line)
            if match:
                value = next((item for item in match.groups()[1:] if item is not None), "")
                result[match.group(1)] = value
    toml_file = config_dir / "config.toml"
    if toml_file.is_file():
        try:
            llm = tomllib.loads(toml_file.read_text(encoding="utf-8")).get("llm", {})
            if isinstance(llm, dict):
                result.setdefault("base_url", str(llm.get("base_url", "")))
                result.setdefault("model", str(llm.get("model", "")))
                result.setdefault("reasoning_effort", str(llm.get("reasoning_effort", "")))
                token_file = llm.get("token_file")
                if token_file and "api_key" not in result:
                    token_path = config_dir / str(token_file)
                    if token_path.is_file():
                        result["api_key"] = token_path.read_text(encoding="utf-8").strip()
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return {
        "base_url": result.get("VERDI_AI_BASE_URL", result.get("base_url", "")),
        "model": result.get("VERDI_AI_MODEL", result.get("model", "")),
        "api_key": result.get("VERDI_AI_API_KEY", result.get("api_key", "")),
        "reasoning_effort": result.get("VERDI_AI_REASONING_EFFORT", result.get("reasoning_effort", "")),
    }


@dataclass
class JsonSearchBackend:
    """Adapter for a user-selected search service returning JSON hits."""

    endpoint: str
    api_key: str | None = None
    timeout: float = 20.0

    def search(self, query: str) -> list[SearchHit]:
        url = self.endpoint + ("&" if "?" in self.endpoint else "?") + urllib.parse.urlencode({"q": query})
        headers = {"Accept": "application/json", "User-Agent": "VerdiWM/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value: Any = json.loads(response.read().decode("utf-8"))
        rows = value.get("results", value) if isinstance(value, dict) else value
        return [SearchHit(str(row["url"]), str(row.get("title", "")), str(row.get("source", "web")), str(row.get("snippet", "")), row.get("pdf_url")) for row in rows if isinstance(row, dict) and row.get("url")]
