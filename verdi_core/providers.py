"""Optional HTTP providers; no vendor SDK is required by the Kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 remains a supported runtime.
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]
import urllib.parse
import urllib.request
import time
import threading
from queue import Queue
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
    # ``urllib``'s timeout is an inactivity timeout. A peer that sends small
    # chunks forever can therefore keep a request alive indefinitely. Keep a
    # separate wall-clock budget so autonomous supervisors can always recover.
    total_timeout: float = 120.0

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
        try:
            total_timeout = float(os.getenv("VERDI_AI_TOTAL_TIMEOUT", settings.get("total_timeout", "120")))
        except ValueError:
            total_timeout = 120.0
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.getenv("VERDI_AI_API_KEY") or settings.get("api_key"),
            role_models=role_models,
            reasoning_effort=os.getenv("VERDI_AI_REASONING_EFFORT") or settings.get("reasoning_effort"),
            total_timeout=max(1.0, total_timeout),
        )

    def complete(self, *, role: str, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {"model": (self.role_models or {}).get(role, self.model), "messages": [{"role": "system", "content": role}, {"role": "user", "content": prompt}], "temperature": 0}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "User-Agent": "VerdiWM/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            remaining = self.total_timeout - (time.monotonic() - started)
            if remaining <= 0:
                break
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                value = self._request_json(request, timeout=min(self.timeout, remaining), join_timeout=remaining)
                content = value["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("AI response content is empty")
                return content
            except (OSError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = min(0.25 * (2 ** attempt), max(0.0, self.total_timeout - (time.monotonic() - started)))
                    if delay:
                        time.sleep(delay)
        if last_error is None:
            last_error = TimeoutError(f"request exceeded total timeout ({self.total_timeout:.1f}s)")
        elif time.monotonic() - started >= self.total_timeout:
            last_error = TimeoutError(f"request exceeded total timeout ({self.total_timeout:.1f}s): {last_error}")
        raise RuntimeError(f"OpenAI-compatible provider failed after retries: {last_error}")

    @staticmethod
    def _request_json(request: urllib.request.Request, *, timeout: float, join_timeout: float) -> Any:
        """Run a blocking urllib read behind a hard wall-clock deadline.

        The worker is daemonized because Python cannot safely interrupt an
        arbitrary socket read. The caller returns on deadline, allowing the
        campaign to persist an abstention and continue; the bounded worker is
        discarded when the owning process exits.
        """
        result: Queue[tuple[bool, Any]] = Queue(maxsize=1)

        def fetch() -> None:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result.put((True, json.loads(response.read().decode("utf-8"))))
            except Exception as exc:  # propagated to the caller below
                result.put((False, exc))

        worker = threading.Thread(target=fetch, name="verdi-ai-request", daemon=True)
        worker.start()
        worker.join(max(0.0, join_timeout))
        if worker.is_alive():
            raise TimeoutError(f"request exceeded total timeout ({join_timeout:.1f}s)")
        if result.empty():
            raise RuntimeError("AI request worker exited without a result")
        ok, value = result.get()
        if not ok:
            raise value
        return value


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
    if toml_file.is_file() and tomllib is not None:
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
