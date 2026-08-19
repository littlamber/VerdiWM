"""Bounded, provider-neutral adapter for schema-backed LLM research tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document


class LLMTaskAdapterError(RuntimeError):
    """A trusted LLM adapter configuration or transaction is invalid."""


_INTERACTIVE_CREDENTIAL_PREFIXES = ("CODEX_", "LARK_", "LARKSUITE_")
_HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
    ),
)


def task_request_digest(request: Mapping[str, object]) -> str:
    return _sha256(_canonical_json(request))


def run_llm_task(
    *,
    request: Mapping[str, object],
    adapter: Mapping[str, object],
    output_root: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Execute one external LLM task without granting shell or GPU authority.

    The configured command is a trusted deployment adapter or broker. It sees a
    JSON request and writes a JSON response. The generated candidate later runs
    in a different environment that never receives the adapter credentials.
    """

    root = Path(project_root).resolve() if project_root is not None else None
    _validate_document("llm_research_task", request, root=root)
    command = _adapter_command(adapter)
    timeout = _positive_number(adapter.get("timeout_seconds"), "LLM_ADAPTER_TIMEOUT_INVALID")
    maximum = _positive_integer(
        adapter.get("max_output_bytes"), "LLM_ADAPTER_OUTPUT_LIMIT_INVALID"
    )
    provider_alias = _required_text(adapter.get("provider_alias"), "LLM_ADAPTER_PROVIDER_INVALID")
    model_alias = _required_text(adapter.get("model_alias"), "LLM_ADAPTER_MODEL_INVALID")
    credential_keys = _credential_keys(adapter.get("credential_environment_keys", []))
    transaction_input = {
        "request_digest": task_request_digest(request),
        "provider_alias": provider_alias,
        "model_alias": model_alias,
        "command": list(command),
        "timeout_seconds": timeout,
        "max_output_bytes": maximum,
        "credential_environment_keys": list(credential_keys),
    }
    input_digest = _sha256(_canonical_json(transaction_input))
    destination = Path(output_root).expanduser().resolve()
    resumed = _resume(destination, input_digest=input_digest)
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise LLMTaskAdapterError("LLM_ADAPTER_OUTPUT_INVALID")
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "input-lock.json", {**transaction_input, "input_digest": input_digest})
    request_path = destination / "request.json"
    _write_json(request_path, dict(request))

    response_path = destination / ".provider-response.json"
    argv = _expand_argv(
        command,
        {
            "{request_path}": str(request_path),
            "{response_path}": str(response_path),
        },
    )
    environment, credential_values = _adapter_environment(
        destination=destination, credential_keys=credential_keys
    )
    started = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    stdout = b""
    stderr = b""
    blockers: list[dict[str, object]] = []
    raw_response = b""
    try:
        process = subprocess.Popen(
            argv,
            cwd=destination,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            blockers.append({"code": "LLM_ADAPTER_TIMEOUT"})
        exit_code = process.returncode
    except OSError:
        blockers.append({"code": "LLM_ADAPTER_START_FAILED"})
    duration = time.monotonic() - started
    if not timed_out and exit_code != 0:
        blockers.append({"code": "LLM_ADAPTER_COMMAND_FAILED", "exit_code": exit_code})
    if len(stdout) > maximum or len(stderr) > maximum:
        blockers.append({"code": "LLM_ADAPTER_DIAGNOSTIC_OUTPUT_TOO_LARGE"})
    if not blockers:
        try:
            if response_path.is_symlink() or not response_path.is_file():
                raise OSError
            if response_path.stat().st_size > maximum:
                blockers.append({"code": "LLM_ADAPTER_RESPONSE_TOO_LARGE"})
            else:
                raw_response = response_path.read_bytes()
        except OSError:
            blockers.append({"code": "LLM_ADAPTER_RESPONSE_MISSING"})

    response: dict[str, object] | None = None
    if raw_response and not blockers:
        if _contains_secret(raw_response, credential_values=credential_values):
            blockers.append({"code": "LLM_ADAPTER_SECRET_LIKE_OUTPUT"})
        else:
            try:
                parsed = json.loads(raw_response)
            except (UnicodeDecodeError, json.JSONDecodeError):
                blockers.append({"code": "LLM_ADAPTER_RESPONSE_JSON_INVALID"})
            else:
                if not isinstance(parsed, dict):
                    blockers.append({"code": "LLM_ADAPTER_RESPONSE_JSON_INVALID"})
                else:
                    response = parsed
    if response is not None and not blockers:
        try:
            _validate_document("llm_research_task_response", response, root=root)
            _validate_response_binding(request=request, response=response)
            output = response.get("output")
            assert isinstance(output, Mapping)
            _validate_document(str(request["output_schema"]), output, root=root)
        except (LLMTaskAdapterError, ContractValidationError) as exc:
            blockers.append({"code": "LLM_ADAPTER_RESPONSE_CONTRACT_INVALID", "detail": str(exc)})

    if response_path.exists() and not response_path.is_symlink():
        response_path.unlink()
    state = "completed" if not blockers else "blocked"
    normalized_response_path: Path | None = None
    response_sha256: str | None = None
    if state == "completed":
        assert response is not None
        normalized_response_path = destination / "response.json"
        _write_json(normalized_response_path, response)
        response_sha256 = _sha256(normalized_response_path.read_bytes())
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-task-receipt",
        "state": state,
        "task_id": request["task_id"],
        "task_type": request["task_type"],
        "provider_alias": provider_alias,
        "model_alias": model_alias,
        "prompt_template_digest": request["prompt_template_digest"],
        "input_digest": input_digest,
        "request_digest": task_request_digest(request),
        "response_sha256": response_sha256,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "blockers": blockers,
        "side_effects": {
            "source_mutated": False,
            "gpu_execution_started": False,
            "gpu_scheduling_authority": False,
            "evaluator_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "This receipt records one bounded LLM task. A completed response is a draft "
            "artifact only and grants no implementation, GPU, evaluator, or promotion authority."
        ),
    }
    _validate_document("llm_task_receipt", receipt, root=root)
    receipt_path = destination / "receipt.json"
    _write_json(receipt_path, receipt)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-task-manifest",
        "state": state,
        "task_id": request["task_id"],
        "input_digest": input_digest,
        "response_path": str(normalized_response_path) if normalized_response_path else None,
        "response_sha256": response_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path.read_bytes()),
        "blockers": blockers,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _validate_response_binding(
    *, request: Mapping[str, object], response: Mapping[str, object]
) -> None:
    if response.get("task_id") != request.get("task_id"):
        raise LLMTaskAdapterError("LLM_ADAPTER_TASK_ID_MISMATCH")
    if response.get("task_type") != request.get("task_type"):
        raise LLMTaskAdapterError("LLM_ADAPTER_TASK_TYPE_MISMATCH")
    if response.get("state") != "completed":
        raise LLMTaskAdapterError("LLM_ADAPTER_TASK_NOT_COMPLETED")


def _adapter_command(adapter: Mapping[str, object]) -> tuple[str, ...]:
    raw = adapter.get("command")
    if not isinstance(raw, list) or not raw:
        raise LLMTaskAdapterError("LLM_ADAPTER_COMMAND_INVALID")
    command = tuple(str(value) for value in raw)
    if any(not value or "\x00" in value for value in command):
        raise LLMTaskAdapterError("LLM_ADAPTER_COMMAND_INVALID")
    return command


def _credential_keys(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise LLMTaskAdapterError("LLM_ADAPTER_CREDENTIAL_KEYS_INVALID")
    values = tuple(str(value) for value in raw)
    if len(values) != len(set(values)):
        raise LLMTaskAdapterError("LLM_ADAPTER_CREDENTIAL_KEYS_INVALID")
    for value in values:
        upper = value.upper()
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
            or upper.startswith(_INTERACTIVE_CREDENTIAL_PREFIXES)
        ):
            raise LLMTaskAdapterError("LLM_ADAPTER_INTERACTIVE_CREDENTIAL_FORBIDDEN")
    return values


def _adapter_environment(
    *, destination: Path, credential_keys: Sequence[str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    home = destination / "adapter-home"
    home.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(destination / "tmp"),
    }
    Path(environment["TMPDIR"]).mkdir(mode=0o700)
    values = []
    for key in credential_keys:
        value = os.environ.get(key)
        if value is None:
            raise LLMTaskAdapterError(f"LLM_ADAPTER_CREDENTIAL_MISSING:{key}")
        environment[key] = value
        if len(value) >= 8:
            values.append(value)
    return environment, tuple(values)


def _contains_secret(payload: bytes, *, credential_values: Sequence[str]) -> bool:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return True
    if any(value in text for value in credential_values):
        return True
    return any(pattern.search(text) is not None for pattern in _HIGH_CONFIDENCE_SECRET_PATTERNS)


def _expand_argv(command: Sequence[str], placeholders: Mapping[str, str]) -> list[str]:
    expanded = []
    for raw in command:
        value = raw
        for token, replacement in placeholders.items():
            value = value.replace(token, replacement)
        if re.search(r"\{[A-Za-z][A-Za-z0-9_]*\}", value):
            raise LLMTaskAdapterError("LLM_ADAPTER_COMMAND_PLACEHOLDER_UNBOUND")
        expanded.append(value)
    return expanded


def _resume(destination: Path, *, input_digest: str) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise LLMTaskAdapterError("LLM_ADAPTER_OUTPUT_INVALID")
    lock = _load(destination / "input-lock.json", "LLM_ADAPTER_INPUT_LOCK_INVALID")
    if lock.get("input_digest") != input_digest:
        raise LLMTaskAdapterError("LLM_ADAPTER_INPUT_LOCK_MISMATCH")
    return _load(destination / "manifest.json", "LLM_ADAPTER_MANIFEST_INVALID")


def _positive_number(raw: object, code: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise LLMTaskAdapterError(code)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise LLMTaskAdapterError(code)
    return value


def _positive_integer(raw: object, code: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise LLMTaskAdapterError(code)
    return raw


def _required_text(raw: object, code: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise LLMTaskAdapterError(code)
    return raw.strip()


def _validate_document(name: str, document: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_document(name, document, root=root)
    except ContractValidationError as exc:
        raise LLMTaskAdapterError(f"LLM_ADAPTER_CONTRACT_INVALID:{name}:{exc}") from exc


def _load(path: Path, code: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LLMTaskAdapterError(code)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMTaskAdapterError(code) from exc
    if not isinstance(payload, dict):
        raise LLMTaskAdapterError(code)
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(body, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
