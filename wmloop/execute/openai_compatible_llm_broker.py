#!/usr/bin/env python3
"""Call an OpenAI-compatible JSON API for one bounded research task.

The broker is deliberately provider-neutral.  It supports the Responses API
and Chat Completions wire shapes, but never discovers credentials or endpoint
configuration from the surrounding Codex/bridge process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


class OpenAICompatibleBrokerError(RuntimeError):
    """The configured OpenAI-compatible service returned an invalid response."""


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_INTERACTIVE_CREDENTIAL_PREFIXES = ("CODEX_", "LARK_", "LARKSUITE_")


def forward(
    *,
    request_path: Path,
    response_path: Path,
    endpoint: str | None,
    model: str,
    token_environment_key: str,
    token_file: Path | None = None,
    api_style: str = "responses",
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    disable_response_storage: bool = True,
    timeout_seconds: float = 180.0,
    maximum_bytes: int = 524288,
) -> None:
    request = _load_request(request_path, maximum_bytes)
    endpoint = _resolve_endpoint(endpoint=endpoint, base_url=base_url, api_style=api_style)
    _validate_endpoint(endpoint)
    if not model or "\x00" in model:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_MODEL_INVALID")
    _validate_token_environment_key(token_environment_key)
    token = _resolve_token(token_environment_key=token_environment_key, token_file=token_file)
    if not token:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_MISSING")
    if api_style not in {"responses", "chat_completions"}:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_API_STYLE_INVALID")
    if reasoning_effort is not None and (
        not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reasoning_effort)
    ):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REASONING_EFFORT_INVALID")

    prompt = _prompt(request)
    if api_style == "responses":
        payload = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "store": not disable_response_storage,
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
    raw = _post_json(
        endpoint=endpoint,
        payload=payload,
        token=token,
        timeout_seconds=timeout_seconds,
        maximum_bytes=maximum_bytes,
    )
    content = _extract_content(raw, api_style=api_style)
    output = _parse_json_object(content)
    response = {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task-response",
        "task_id": request["task_id"],
        "task_type": request["task_type"],
        "state": "completed",
        "output": output,
    }
    _write_response(response_path, response)


def _load_request(path: Path, maximum_bytes: int) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size > maximum_bytes:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REQUEST_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REQUEST_INVALID") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("task_id"), str):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REQUEST_INVALID")
    return payload


def _prompt(request: Mapping[str, object]) -> str:
    return (
        "Return only one JSON object matching the requested output schema. "
        "Do not use Markdown fences. Do not claim permissions not present in the request.\n\n"
        + json.dumps(request, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def _post_json(
    *, endpoint: str, payload: Mapping[str, object], token: str, timeout_seconds: float, maximum_bytes: int
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(body) > maximum_bytes:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REQUEST_TOO_LARGE")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(maximum_bytes + 1)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_REQUEST_FAILED") from exc
    if len(raw) > maximum_bytes:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_RESPONSE_TOO_LARGE")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_RESPONSE_INVALID") from exc
    if not isinstance(parsed, dict):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_RESPONSE_INVALID")
    return parsed


def _extract_content(response: Mapping[str, object], *, api_style: str) -> str:
    if api_style == "responses":
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        output = response.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                        parts.append(str(block["text"]))
            if parts:
                return "".join(parts)
    else:
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return str(message["content"])
    raise OpenAICompatibleBrokerError("OPENAI_BROKER_RESPONSE_CONTENT_MISSING")


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(content)
        if match is None:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_OUTPUT_JSON_INVALID")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_OUTPUT_JSON_INVALID") from exc
    if not isinstance(parsed, dict):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_OUTPUT_JSON_INVALID")
    return parsed


def _write_response(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    temporary = Path(name)
    try:
        os.close(fd)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_endpoint(endpoint: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    if not parsed.hostname or not (
        parsed.scheme == "https"
        or (parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS)
    ):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_ENDPOINT_INVALID")


def _resolve_endpoint(*, endpoint: str | None, base_url: str | None, api_style: str) -> str:
    if (endpoint is None) == (base_url is None):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_ENDPOINT_CONFIGURATION_INVALID")
    if endpoint is not None:
        return endpoint
    assert base_url is not None
    normalized = base_url.rstrip("/")
    suffix = "/v1/responses" if api_style == "responses" else "/v1/chat/completions"
    if normalized.endswith(suffix):
        return normalized
    return normalized + suffix


def _validate_token_environment_key(key: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_KEY_INVALID")
    if key.upper().startswith(_INTERACTIVE_CREDENTIAL_PREFIXES):
        raise OpenAICompatibleBrokerError("OPENAI_BROKER_INTERACTIVE_CREDENTIAL_FORBIDDEN")


def _resolve_token(*, token_environment_key: str, token_file: Path | None) -> str:
    """Read an explicitly configured deployment credential without discovery.

    A token file is an opt-in convenience for headless deployments.  It must be
    a regular, non-symlink file with owner-only permissions; this prevents an
    accidentally shared auth file from becoming a silent credential leak.
    """
    if token_file is not None:
        source = Path(token_file).expanduser()
        try:
            stat = source.stat()
        except OSError as exc:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_FILE_INVALID") from exc
        if source.is_symlink() or not source.is_file() or stat.st_size > 16384:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_FILE_INVALID")
        if stat.st_mode & 0o077:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_FILE_PERMISSIONS")
        try:
            token = source.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_FILE_INVALID") from exc
        if not token or "\x00" in token or "\n" in token or "\r" in token:
            raise OpenAICompatibleBrokerError("OPENAI_BROKER_TOKEN_FILE_INVALID")
        return token
    return os.environ.get(token_environment_key, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("response_path", type=Path)
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--endpoint")
    endpoint.add_argument("--base-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-environment-key", default="VERDIWM_LLM_BROKER_TOKEN")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--api-style", choices=("responses", "chat_completions"), default="responses")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--allow-response-storage", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--maximum-bytes", type=int, default=524288)
    args = parser.parse_args()
    forward(
        request_path=args.request_path,
        response_path=args.response_path,
        endpoint=args.endpoint,
        base_url=args.base_url,
        model=args.model,
        token_environment_key=args.token_environment_key,
        token_file=args.token_file,
        api_style=args.api_style,
        reasoning_effort=args.reasoning_effort,
        disable_response_storage=not args.allow_response_storage,
        timeout_seconds=args.timeout_seconds,
        maximum_bytes=args.maximum_bytes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
