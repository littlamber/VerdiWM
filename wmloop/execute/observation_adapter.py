"""Bounded request/response execution for admitted observation ABIs."""

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


class ObservationAdapterError(RuntimeError):
    """An observation adapter transaction violated its durable boundary."""


_RESERVED_ENVIRONMENT = {
    "HOME",
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "TMPDIR",
}
_AUTHORITY = {
    "source_mutated": False,
    "intervention_executed": False,
    "active_metric_mutated": False,
    "active_evaluator_mutated": False,
    "verdict_exposed": False,
    "promotion_authority": False,
}


def request_digest(request: Mapping[str, object]) -> str:
    """Return the canonical identity of one observation request."""

    return _sha256(_canonical_bytes(request))


def run_observation_adapter(
    *,
    request: Mapping[str, object],
    adapter: Mapping[str, object],
    output_root: Path,
    project_root: Path | None = None,
    gpu_environment: Mapping[str, str] | None = None,
    gpu_lease: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run one trusted adapter and seal its path-free observation response."""

    root = Path(project_root).resolve() if project_root is not None else None
    _validate("observation_execution_request", request, root=root)
    command = _command(adapter)
    timeout = _positive_number(
        adapter.get("timeout_seconds"), "OBSERVATION_ADAPTER_TIMEOUT_INVALID"
    )
    maximum = _positive_integer(
        adapter.get("max_output_bytes"), "OBSERVATION_ADAPTER_OUTPUT_LIMIT_INVALID"
    )
    resource_class = str(adapter.get("resource_class") or "")
    if resource_class not in {"cpu_only", "diagnostic_gpu"}:
        raise ObservationAdapterError("OBSERVATION_ADAPTER_RESOURCE_CLASS_INVALID")
    if resource_class == "cpu_only" and (gpu_environment or gpu_lease):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_GPU_FORBIDDEN")
    if resource_class == "diagnostic_gpu" and (not gpu_environment or not gpu_lease):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_GPU_LEASE_REQUIRED")
    configured_environment = _environment_mapping(adapter.get("environment", {}))
    adapter_identity = {
        "command": list(command),
        "timeout_seconds": timeout,
        "max_output_bytes": maximum,
        "resource_class": resource_class,
        "environment": configured_environment,
    }
    adapter_digest = _sha256(_canonical_bytes(adapter_identity))
    input_lock = {
        "request_digest": request_digest(request),
        "adapter_digest": adapter_digest,
        "resource_class": resource_class,
        "gpu_lease": dict(gpu_lease) if gpu_lease is not None else None,
    }
    input_digest = _sha256(_canonical_bytes(input_lock))
    raw_destination = Path(output_root).expanduser()
    if raw_destination.is_symlink():
        raise ObservationAdapterError("OBSERVATION_ADAPTER_OUTPUT_INVALID")
    destination = raw_destination.resolve()
    resumed = _resume(
        destination,
        request=request,
        input_lock=input_lock,
        adapter_digest=adapter_digest,
        root=root,
    )
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise ObservationAdapterError("OBSERVATION_ADAPTER_OUTPUT_INVALID")
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "input-lock.json", {**input_lock, "input_digest": input_digest})
    request_path = destination / "request.json"
    _write_json(request_path, request)
    response_path = destination / ".adapter-response.json"
    argv = _expand(
        command,
        {
            "{request_path}": str(request_path),
            "{response_path}": str(response_path),
        },
    )
    environment = _execution_environment(
        destination=destination,
        configured=configured_environment,
        gpu_environment=gpu_environment or {},
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
            blockers.append({"code": "OBSERVATION_ADAPTER_TIMEOUT"})
        exit_code = process.returncode
    except OSError:
        blockers.append({"code": "OBSERVATION_ADAPTER_START_FAILED"})
    duration = time.monotonic() - started
    if not timed_out and exit_code != 0:
        blockers.append(
            {"code": "OBSERVATION_ADAPTER_COMMAND_FAILED", "exit_code": exit_code}
        )
    if len(stdout) > maximum or len(stderr) > maximum:
        blockers.append({"code": "OBSERVATION_ADAPTER_DIAGNOSTIC_OUTPUT_TOO_LARGE"})
    if not blockers:
        if response_path.is_symlink() or not response_path.is_file():
            blockers.append({"code": "OBSERVATION_ADAPTER_RESPONSE_MISSING"})
        elif response_path.stat().st_size > maximum:
            blockers.append({"code": "OBSERVATION_ADAPTER_RESPONSE_TOO_LARGE"})
        else:
            raw_response = response_path.read_bytes()

    response: dict[str, object] | None = None
    if raw_response and not blockers:
        try:
            parsed = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append({"code": "OBSERVATION_ADAPTER_RESPONSE_JSON_INVALID"})
        else:
            if not isinstance(parsed, dict):
                blockers.append({"code": "OBSERVATION_ADAPTER_RESPONSE_JSON_INVALID"})
            else:
                response = parsed
    if response is not None and not blockers:
        try:
            _validate("observation_execution_result", response, root=root)
            _validate_response_binding(request=request, response=response)
        except ObservationAdapterError as exc:
            blockers.append(
                {
                    "code": "OBSERVATION_ADAPTER_RESPONSE_CONTRACT_INVALID",
                    "detail": str(exc),
                }
            )
        else:
            if response.get("state") == "blocked":
                response_blockers = response.get("blockers")
                if isinstance(response_blockers, list):
                    blockers.extend(
                        dict(row)
                        for row in response_blockers
                        if isinstance(row, Mapping)
                    )
    response_path.unlink(missing_ok=True)
    state = "completed" if not blockers else "blocked"
    normalized_response: Path | None = None
    response_sha256: str | None = None
    response_ref: str | None = None
    if state == "completed":
        assert response is not None
        normalized_response = destination / "response.json"
        _write_json(normalized_response, response)
        response_sha256 = _sha256(normalized_response.read_bytes())
        response_ref = "sha256:" + response_sha256
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-observation-execution-receipt",
        "state": state,
        "request_id": request["request_id"],
        "task_id": request["task_id"],
        "abi_id": request["abi_id"],
        "adapter_digest": adapter_digest,
        "input_digest": input_digest,
        "request_sha256": _sha256(request_path.read_bytes()),
        "response_sha256": response_sha256,
        "response_ref": response_ref,
        "resource_class": resource_class,
        "gpu_lease": dict(gpu_lease) if gpu_lease is not None else None,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "blockers": blockers,
        "side_effects": dict(_AUTHORITY),
        "claim_boundary": (
            "This receipt records one controller-owned observation execution. It grants "
            "no intervention, evaluator, verdict, metric, or promotion authority."
        ),
    }
    _validate("observation_execution_receipt", receipt, root=root)
    receipt_path = destination / "receipt.json"
    _write_json(receipt_path, receipt)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-observation-execution-manifest",
        "state": state,
        "request_id": request["request_id"],
        "task_id": request["task_id"],
        "abi_id": request["abi_id"],
        "adapter_digest": adapter_digest,
        "input_digest": input_digest,
        "resource_class": resource_class,
        "gpu_lease": dict(gpu_lease) if gpu_lease is not None else None,
        "request_sha256": _sha256(request_path.read_bytes()),
        "response_path": str(normalized_response) if normalized_response else None,
        "response_sha256": response_sha256,
        "response_ref": response_ref,
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
        raise ObservationAdapterError("OBSERVATION_ADAPTER_TASK_ID_MISMATCH")
    if response.get("abi_id") != request.get("abi_id"):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_ABI_ID_MISMATCH")
    state = response.get("state")
    kind = response.get("observation_kind")
    probe = response.get("probe_observation")
    structural = response.get("structural_observation")
    blockers = response.get("blockers")
    if state == "completed":
        if blockers or kind not in {"probe_fingerprint", "structural_surface"}:
            raise ObservationAdapterError("OBSERVATION_ADAPTER_COMPLETED_RESULT_INVALID")
        if kind == "probe_fingerprint" and (not isinstance(probe, Mapping) or structural is not None):
            raise ObservationAdapterError("OBSERVATION_ADAPTER_PROBE_RESULT_INVALID")
        if kind == "structural_surface" and (not isinstance(structural, Mapping) or probe is not None):
            raise ObservationAdapterError("OBSERVATION_ADAPTER_STRUCTURAL_RESULT_INVALID")
    elif kind != "none" or probe is not None or structural is not None or not blockers:
        raise ObservationAdapterError("OBSERVATION_ADAPTER_BLOCKED_RESULT_INVALID")
    if response.get("authority") != _AUTHORITY:
        raise ObservationAdapterError("OBSERVATION_ADAPTER_AUTHORITY_INVALID")


def _command(adapter: Mapping[str, object]) -> tuple[str, ...]:
    raw = adapter.get("command")
    if not isinstance(raw, list) or not raw:
        raise ObservationAdapterError("OBSERVATION_ADAPTER_COMMAND_INVALID")
    command = tuple(str(value) for value in raw)
    if any(not value or "\x00" in value for value in command):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_COMMAND_INVALID")
    return command


def _environment_mapping(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_ENVIRONMENT_INVALID")
    result: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key)
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name in _RESERVED_ENVIRONMENT
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ObservationAdapterError("OBSERVATION_ADAPTER_ENVIRONMENT_INVALID")
        result[name] = value
    return dict(sorted(result.items()))


def _execution_environment(
    *,
    destination: Path,
    configured: Mapping[str, str],
    gpu_environment: Mapping[str, str],
) -> dict[str, str]:
    home = destination / "adapter-home"
    temporary = destination / "tmp"
    home.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        **configured,
    }
    for key, value in gpu_environment.items():
        if key not in {
            "CUDA_VISIBLE_DEVICES",
            "VERDIWM_PHYSICAL_GPU_INDEX",
            "VERDIWM_PHYSICAL_GPU_UUID",
        }:
            raise ObservationAdapterError("OBSERVATION_ADAPTER_GPU_ENVIRONMENT_INVALID")
        environment[key] = str(value)
    return environment


def _expand(command: Sequence[str], placeholders: Mapping[str, str]) -> list[str]:
    expanded = []
    for raw in command:
        value = raw
        for token, replacement in placeholders.items():
            value = value.replace(token, replacement)
        if re.search(r"\{[A-Za-z][A-Za-z0-9_]*\}", value):
            raise ObservationAdapterError("OBSERVATION_ADAPTER_PLACEHOLDER_UNBOUND")
        expanded.append(value)
    return expanded


def _resume(
    destination: Path,
    *,
    request: Mapping[str, object],
    input_lock: Mapping[str, object],
    adapter_digest: str,
    root: Path | None,
) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ObservationAdapterError("OBSERVATION_ADAPTER_OUTPUT_INVALID")
    lock_path = _fixed_file(
        destination / "input-lock.json", "OBSERVATION_ADAPTER_INPUT_LOCK_INVALID"
    )
    lock = _load(lock_path, "OBSERVATION_ADAPTER_INPUT_LOCK_INVALID")
    expected_input_digest = _sha256(_canonical_bytes(input_lock))
    expected_lock = {**dict(input_lock), "input_digest": expected_input_digest}
    if lock != expected_lock or lock.get("input_digest") != expected_input_digest:
        raise ObservationAdapterError("OBSERVATION_ADAPTER_INPUT_LOCK_MISMATCH")
    request_path = _fixed_file(
        destination / "request.json", "OBSERVATION_ADAPTER_REQUEST_INVALID"
    )
    request_document = _load(request_path, "OBSERVATION_ADAPTER_REQUEST_INVALID")
    if _canonical_bytes(request_document) != _canonical_bytes(dict(request)):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_REQUEST_MISMATCH")
    manifest_path = _fixed_file(
        destination / "manifest.json", "OBSERVATION_ADAPTER_MANIFEST_INVALID"
    )
    manifest = _load(manifest_path, "OBSERVATION_ADAPTER_MANIFEST_INVALID")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "verdiwm-observation-execution-manifest"
        or manifest.get("state") not in {"completed", "blocked"}
        or manifest.get("request_id") != request.get("request_id")
        or manifest.get("task_id") != request.get("task_id")
        or manifest.get("abi_id") != request.get("abi_id")
        or manifest.get("adapter_digest") != adapter_digest
        or manifest.get("input_digest") != expected_input_digest
        or manifest.get("resource_class") != input_lock.get("resource_class")
        or manifest.get("gpu_lease") != input_lock.get("gpu_lease")
        or manifest.get("request_sha256") != _sha256(request_path.read_bytes())
    ):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_MANIFEST_BINDING_MISMATCH")
    receipt_path = _fixed_file(
        destination / "receipt.json", "OBSERVATION_ADAPTER_RECEIPT_INVALID"
    )
    _assert_path_field(
        manifest.get("receipt_path"), receipt_path, "OBSERVATION_ADAPTER_RECEIPT_INVALID"
    )
    if manifest.get("receipt_sha256") != _sha256(receipt_path.read_bytes()):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_RECEIPT_HASH_MISMATCH")
    receipt = _load(receipt_path, "OBSERVATION_ADAPTER_RECEIPT_INVALID")
    _validate("observation_execution_receipt", receipt, root=root)
    if (
        receipt.get("state") != manifest.get("state")
        or receipt.get("request_id") != request.get("request_id")
        or receipt.get("task_id") != request.get("task_id")
        or receipt.get("abi_id") != request.get("abi_id")
        or receipt.get("adapter_digest") != adapter_digest
        or receipt.get("input_digest") != expected_input_digest
        or receipt.get("resource_class") != input_lock.get("resource_class")
        or receipt.get("gpu_lease") != input_lock.get("gpu_lease")
        or receipt.get("request_sha256") != _sha256(request_path.read_bytes())
        or manifest.get("blockers") != receipt.get("blockers")
    ):
        raise ObservationAdapterError("OBSERVATION_ADAPTER_RECEIPT_BINDING_MISMATCH")
    response_value = manifest.get("response_path")
    response_sha256 = manifest.get("response_sha256")
    response_ref = manifest.get("response_ref")
    if manifest.get("state") == "completed":
        response_path = _fixed_file(
            destination / "response.json", "OBSERVATION_ADAPTER_RESPONSE_INVALID"
        )
        _assert_path_field(
            response_value, response_path, "OBSERVATION_ADAPTER_RESPONSE_INVALID"
        )
        actual_response_sha256 = _sha256(response_path.read_bytes())
        if (
            response_sha256 != actual_response_sha256
            or response_ref != "sha256:" + actual_response_sha256
            or receipt.get("response_sha256") != actual_response_sha256
            or receipt.get("response_ref") != response_ref
        ):
            raise ObservationAdapterError("OBSERVATION_ADAPTER_RESPONSE_HASH_MISMATCH")
        response = _load(response_path, "OBSERVATION_ADAPTER_RESPONSE_INVALID")
        _validate("observation_execution_result", response, root=root)
        _validate_response_binding(request=request, response=response)
    else:
        if response_value is not None or response_sha256 is not None or response_ref is not None:
            raise ObservationAdapterError("OBSERVATION_ADAPTER_BLOCKED_RESPONSE_INVALID")
        response_path = destination / "response.json"
        if response_path.exists() or response_path.is_symlink():
            raise ObservationAdapterError("OBSERVATION_ADAPTER_BLOCKED_RESPONSE_INVALID")
    return manifest


def _positive_number(raw: object, code: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ObservationAdapterError(code)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ObservationAdapterError(code)
    return value


def _positive_integer(raw: object, code: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ObservationAdapterError(code)
    return raw


def _validate(name: str, document: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_document(name, document, root=root)
    except ContractValidationError as exc:
        raise ObservationAdapterError(
            f"OBSERVATION_ADAPTER_CONTRACT_INVALID:{name}:{exc}"
        ) from exc


def _load(path: Path, code: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ObservationAdapterError(code)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationAdapterError(code) from exc
    if not isinstance(payload, dict):
        raise ObservationAdapterError(code)
    return payload


def _fixed_file(path: Path, code: str) -> Path:
    """Require a regular, non-symlink file at a transaction-owned path."""

    if path.is_symlink() or not path.is_file():
        raise ObservationAdapterError(code)
    return path


def _assert_path_field(value: object, expected: Path, code: str) -> None:
    if not isinstance(value, str) or not value:
        raise ObservationAdapterError(code)
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise ObservationAdapterError(code)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = _canonical_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
