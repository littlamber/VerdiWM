#!/usr/bin/env python3
"""Run a candidate in a guarded worktree process without requiring Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from wmloop.control.candidate_execution import (
    CandidateExecutionContractError,
    validate_candidate_execution_contract,
)
from wmloop.execute.candidate_sandbox_broker import (
    CandidateSandboxBrokerError,
    _contained_directory,
    _contained_file,
    _directory,
    _entrypoint,
    _expand_command,
    _load_json,
    _positive_int,
    _write_response,
)


class CandidateLocalDevBrokerError(RuntimeError):
    """A local development smoke was unsafe or failed."""


_DEV_OPERATIONS = {"calibrate", "train", "infer"}


def run_request(
    *,
    request_path: Path,
    response_path: Path,
    workspace_root: Path,
    timeout_seconds: float = 300.0,
    memory_bytes: int = 32 * 1024 * 1024 * 1024,
    maximum_output_bytes: int = 256 * 1024 * 1024,
    maximum_processes: int = 256,
    gpu_devices: tuple[str, ...] = (),
) -> None:
    """Execute candidate code with process guards and explicit assurance metadata."""

    request = _load_json(request_path)
    operation = request.get("operation")
    if operation not in _DEV_OPERATIONS:
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_OPERATION_FORBIDDEN")
    contract = request.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_CONTRACT_MISSING")
    try:
        validate_candidate_execution_contract(contract)
    except CandidateExecutionContractError as exc:
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_CONTRACT_INVALID") from exc

    root = _directory(workspace_root, must_exist=True)
    candidate = _contained_directory(root, request.get("candidate_root"), must_exist=True)
    output = _contained_directory(root, request.get("output_root"), must_exist=False)
    if output == candidate or candidate in output.parents or output in candidate.parents:
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_PATH_OVERLAP")
    if any(path.is_symlink() for path in candidate.rglob("*")):
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_SYMLINK_FORBIDDEN")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    input_manifest = request.get("input_manifest")
    input_host = _contained_file(root, input_manifest) if input_manifest is not None else None

    command = _entrypoint(contract, str(operation))
    if any("{input_manifest}" in token for token in command) and input_host is None:
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_INPUT_REQUIRED")
    workspace_copy = output / "workspace-copy"
    if workspace_copy.exists() or workspace_copy.is_symlink():
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_WORKSPACE_EXISTS")
    shutil.copytree(candidate, workspace_copy, symlinks=False)
    command = _expand_command(
        command,
        candidate_root=str(workspace_copy),
        output_root=str(output),
        input_manifest=str(input_host) if input_host else "",
        world_size=_positive_int(request.get("world_size", 1), "WORLD_SIZE"),
        rank=_positive_int(request.get("rank", 0), "RANK", allow_zero=True),
    )
    home = output / "dev-home"
    scratch = output / "tmp"
    home.mkdir(mode=0o700)
    scratch.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_devices),
    }
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace_copy,
            env=environment,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            preexec_fn=lambda: _set_limits(
                timeout_seconds=timeout_seconds,
                memory_bytes=memory_bytes,
                maximum_output_bytes=maximum_output_bytes,
                maximum_processes=maximum_processes,
            ),
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
    except (OSError, ValueError, CandidateSandboxBrokerError) as exc:
        raise CandidateLocalDevBrokerError("CANDIDATE_LOCAL_DEV_START_FAILED") from exc

    blockers = []
    if timed_out:
        blockers.append("CANDIDATE_LOCAL_DEV_TIMEOUT")
    elif exit_code != 0:
        blockers.append("CANDIDATE_LOCAL_DEV_COMMAND_FAILED")
    response = {
        "schema_version": 1,
        "artifact_type": "verdiwm-candidate-local-dev-response",
        "state": "completed" if not blockers else "blocked",
        "operation": operation,
        "execution_id": contract["execution_id"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "blockers": blockers,
        "isolation": {
            "execution_backend": "worktree_process",
            "assurance_level": "process_guarded",
            "containerized": False,
            "worktree_copy": True,
            "environment_sanitized": True,
            "network_isolation_enforced": False,
            "filesystem_isolation_enforced": False,
            "credentials_forwarded": False,
            "gpu_devices": list(gpu_devices),
        },
        "authority": {
            "local_experiment": True,
            "local_admission": True,
            "screen_or_confirm": True,
            "community_projection": False,
        },
        "claim_boundary": (
            "Worktree-process execution supports local experiments and evaluation. It records "
            "weaker host isolation than a container and requires independent strong-isolation "
            "verification before community promotion."
        ),
    }
    _write_response(response_path, response)
    if blockers:
        raise CandidateLocalDevBrokerError(blockers[0])


def _set_limits(
    *,
    timeout_seconds: float,
    memory_bytes: int,
    maximum_output_bytes: int,
    maximum_processes: int,
) -> None:
    cpu_seconds = max(1, int(timeout_seconds) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum_output_bytes, maximum_output_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (maximum_processes, maximum_processes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value.encode() if isinstance(value, str) else value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("response_path", type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--memory-bytes", type=int, default=32 * 1024 * 1024 * 1024)
    parser.add_argument("--maximum-output-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--maximum-processes", type=int, default=256)
    parser.add_argument("--gpu-device", action="append", default=[])
    args = parser.parse_args()
    run_request(
        request_path=args.request_path,
        response_path=args.response_path,
        workspace_root=args.workspace_root,
        timeout_seconds=args.timeout_seconds,
        memory_bytes=args.memory_bytes,
        maximum_output_bytes=args.maximum_output_bytes,
        maximum_processes=args.maximum_processes,
        gpu_devices=tuple(args.gpu_device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
