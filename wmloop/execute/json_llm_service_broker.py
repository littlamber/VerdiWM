#!/usr/bin/env python3
"""Forward one bounded research task to a trusted JSON LLM service."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class JsonLLMServiceBrokerError(RuntimeError):
    """The trusted JSON service did not return a bounded task response."""


def forward(
    *,
    request_path: Path,
    response_path: Path,
    endpoint: str,
    token_environment_key: str | None,
    timeout_seconds: float,
    maximum_response_bytes: int,
) -> None:
    source = request_path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_REQUEST_INVALID")
    request = source.read_bytes()
    if len(request) > maximum_response_bytes:
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_REQUEST_TOO_LARGE")
    try:
        parsed_request = json.loads(request)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_REQUEST_INVALID") from exc
    if not isinstance(parsed_request, dict):
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_REQUEST_INVALID")
    _validate_endpoint(endpoint)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token_environment_key is not None:
        token = os.environ.get(token_environment_key)
        if not token:
            raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_TOKEN_MISSING")
        headers["Authorization"] = f"Bearer {token}"
    outbound = urllib.request.Request(endpoint, data=request, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(outbound, timeout=timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > maximum_response_bytes:
                raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_RESPONSE_TOO_LARGE")
            payload = response.read(maximum_response_bytes + 1)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_REQUEST_FAILED") from exc
    if len(payload) > maximum_response_bytes:
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_RESPONSE_TOO_LARGE")
    try:
        parsed_response = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_RESPONSE_INVALID") from exc
    if not isinstance(parsed_response, dict):
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_RESPONSE_INVALID")
    destination = response_path.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(
            json.dumps(
                parsed_response,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    local = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if not parsed.hostname or (
        parsed.scheme != "https" and not (local and parsed.scheme == "http")
    ):
        raise JsonLLMServiceBrokerError("JSON_LLM_BROKER_ENDPOINT_INVALID")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("response_path", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-environment-key")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--maximum-response-bytes", type=int, default=262144)
    args = parser.parse_args()
    forward(
        request_path=args.request_path,
        response_path=args.response_path,
        endpoint=args.endpoint,
        token_environment_key=args.token_environment_key,
        timeout_seconds=args.timeout_seconds,
        maximum_response_bytes=args.maximum_response_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
