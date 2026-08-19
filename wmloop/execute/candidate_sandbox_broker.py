#!/usr/bin/env python3
"""Run one candidate entrypoint in a restricted Docker/Podman container.

The broker is a deployment boundary. The request may select an operation and
data manifests, but the deployment selects the runtime, image, workspace root,
credentials (none are forwarded), and optional GPU lease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.control.candidate_execution import (
    CandidateExecutionContractError,
    validate_candidate_execution_contract,
)


class CandidateSandboxBrokerError(RuntimeError):
    """The candidate request crossed a sandbox boundary or could not run."""


_OPERATIONS = {"calibrate", "train", "infer"}
_IMAGE_DIGEST = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"\{[A-Za-z][A-Za-z0-9_]*\}")


def run_request(
    *,
    request_path: Path,
    response_path: Path,
    workspace_root: Path,
    image: str,
    runtime: str = "docker",
    timeout_seconds: float = 900.0,
    memory: str = "8g",
    cpus: str = "2",
    pids_limit: int = 256,
    gpu_devices: Sequence[str] = (),
    allow_unpinned_image: bool = False,
) -> None:
    request = _load_json(request_path)
    _validate_image(image, allow_unpinned_image=allow_unpinned_image)
    if runtime not in {"docker", "podman"}:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_RUNTIME_INVALID")
    if timeout_seconds <= 0 or pids_limit <= 0:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_LIMIT_INVALID")
    operation = request.get("operation")
    if operation not in _OPERATIONS:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_OPERATION_INVALID")
    contract = request.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_CONTRACT_MISSING")
    try:
        validate_candidate_execution_contract(contract)
    except CandidateExecutionContractError as exc:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_CONTRACT_INVALID") from exc

    root = _directory(workspace_root, must_exist=True)
    candidate = _contained_directory(root, request.get("candidate_root"), must_exist=True)
    output = _contained_directory(root, request.get("output_root"), must_exist=False)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    input_manifest = request.get("input_manifest")
    input_host: Path | None = None
    if input_manifest is not None:
        input_host = _contained_file(root, input_manifest)

    command = _entrypoint(contract, str(operation))
    if any("{input_manifest}" in token for token in command) and input_host is None:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_INPUT_REQUIRED")
    container_command = _expand_command(
        command,
        candidate_root="/workspace",
        output_root="/output",
        input_manifest="/input/manifest.json" if input_host else "",
        world_size=_positive_int(request.get("world_size", 1), "WORLD_SIZE"),
        rank=_positive_int(request.get("rank", 0), "RANK", allow_zero=True),
    )
    argv = _runtime_command(
        runtime=runtime,
        image=image,
        candidate=candidate,
        output=output,
        input_manifest=input_host,
        command=container_command,
        memory=memory,
        cpus=cpus,
        pids_limit=pids_limit,
        gpu_devices=gpu_devices,
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        exit_code = completed.returncode
        timed_out = False
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        timed_out = True
        stdout = _bytes(exc.stdout)
        stderr = _bytes(exc.stderr)
    except OSError as exc:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_RUNTIME_UNAVAILABLE") from exc
    blockers = []
    if timed_out:
        blockers.append("CANDIDATE_SANDBOX_TIMEOUT")
    elif exit_code != 0:
        blockers.append("CANDIDATE_SANDBOX_COMMAND_FAILED")
    response = {
        "schema_version": 1,
        "artifact_type": "verdiwm-candidate-sandbox-response",
        "state": "completed" if not blockers else "blocked",
        "operation": operation,
        "execution_id": contract["execution_id"],
        "candidate_root": str(request["candidate_root"]),
        "output_root": str(request["output_root"]),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "blockers": blockers,
        "isolation": {
            "network_access": False,
            "candidate_read_only": True,
            "credentials_forwarded": False,
            "evaluator_access": False,
        },
    }
    _write_response(response_path, response)
    if blockers:
        raise CandidateSandboxBrokerError(blockers[0])


def _load_json(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_REQUEST_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_REQUEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_REQUEST_INVALID")
    return payload


def _validate_image(image: str, *, allow_unpinned_image: bool) -> None:
    if not image or "\x00" in image or any(char.isspace() for char in image):
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_IMAGE_INVALID")
    if not allow_unpinned_image and _IMAGE_DIGEST.fullmatch(image) is None:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_IMAGE_NOT_PINNED")


def _directory(path: Path, *, must_exist: bool) -> Path:
    resolved = Path(path).expanduser().resolve(strict=must_exist)
    if resolved.is_symlink() or (must_exist and not resolved.is_dir()):
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_DIRECTORY_INVALID")
    return resolved


def _contained_directory(root: Path, raw: object, *, must_exist: bool) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_PATH_INVALID")
    unresolved = (root / raw) if not Path(raw).is_absolute() else Path(raw)
    if unresolved.is_symlink():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_PATH_INVALID")
    path = unresolved.resolve()
    _require_contained(root, path)
    if must_exist and not path.is_dir():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_CANDIDATE_MISSING")
    return path


def _contained_file(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_INPUT_INVALID")
    unresolved = (root / raw) if not Path(raw).is_absolute() else Path(raw)
    if unresolved.is_symlink():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_INPUT_INVALID")
    path = unresolved.resolve()
    _require_contained(root, path)
    if path.is_symlink() or not path.is_file():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_INPUT_INVALID")
    return path


def _require_contained(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_PATH_ESCAPE") from exc


def _entrypoint(contract: Mapping[str, object], operation: str) -> list[str]:
    entrypoints = contract.get("entrypoints")
    if not isinstance(entrypoints, Mapping) or not isinstance(entrypoints.get(operation), list):
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_ENTRYPOINT_MISSING")
    return [str(value) for value in entrypoints[operation]]


def _expand_command(command: Sequence[str], **values: object) -> list[str]:
    expanded = []
    for token in command:
        if not isinstance(token, str) or not token:
            raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_ENTRYPOINT_INVALID")
        value = token
        for name, replacement in values.items():
            value = value.replace("{" + name + "}", str(replacement))
        if _PLACEHOLDER.search(value):
            raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_PLACEHOLDER_UNBOUND")
        expanded.append(value)
    return expanded


def _runtime_command(
    *,
    runtime: str,
    image: str,
    candidate: Path,
    output: Path,
    input_manifest: Path | None,
    command: Sequence[str],
    memory: str,
    cpus: str,
    pids_limit: int,
    gpu_devices: Sequence[str],
) -> list[str]:
    argv = [
        runtime,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        f"--pids-limit={pids_limit}",
        f"--memory={memory}",
        f"--cpus={cpus}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=512m",
        "--mount",
        f"type=bind,src={candidate},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={output},dst=/output",
    ]
    if input_manifest is not None:
        argv.extend(["--mount", f"type=bind,src={input_manifest},dst=/input/manifest.json,readonly"])
    if gpu_devices:
        devices = ",".join(str(value) for value in gpu_devices)
        argv.extend(["--gpus", f"device={devices}"])
    argv.extend([image, *command])
    return argv


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise CandidateSandboxBrokerError(f"CANDIDATE_SANDBOX_{name}_INVALID")
    return value


def _bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value.encode() if isinstance(value, str) else value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_response(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise CandidateSandboxBrokerError("CANDIDATE_SANDBOX_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("response_path", type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime", choices=("docker", "podman"), default="docker")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--pids-limit", type=int, default=256)
    parser.add_argument("--gpu-device", action="append", default=[])
    parser.add_argument("--allow-unpinned-image", action="store_true")
    args = parser.parse_args()
    run_request(
        request_path=args.request_path,
        response_path=args.response_path,
        workspace_root=args.workspace_root,
        image=args.image,
        runtime=args.runtime,
        timeout_seconds=args.timeout_seconds,
        memory=args.memory,
        cpus=args.cpus,
        pids_limit=args.pids_limit,
        gpu_devices=args.gpu_device,
        allow_unpinned_image=args.allow_unpinned_image,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
