"""Fail-closed Docker command backend for future agent repair execution."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.execute.backends import CommandExecutionResult


class DockerBackendError(RuntimeError):
    """Docker execution was requested before the runtime contract was proven."""


_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}\Z")


@dataclass(frozen=True)
class DockerRuntimeReceipt:
    socket_path: Path
    image: str
    command: tuple[str, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "wm-loop-docker-runtime-receipt",
            "socket_path": str(self.socket_path),
            "image": self.image,
            "command": list(self.command),
        }


class DockerExecutionBackend:
    """Run commands in the dedicated wm-loop Docker daemon after verification."""

    def __init__(
        self,
        *,
        image: str,
        socket_path: Path = Path("/run/wm-loop-docker/docker.sock"),
        memory: str = "8g",
        cpus: int = 8,
        pids_limit: int = 512,
        user: str = "65532:65532",
    ) -> None:
        if not _IMAGE.fullmatch(image):
            raise DockerBackendError("DOCKER_BACKEND_IMAGE_INVALID")
        if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1:
            raise DockerBackendError("DOCKER_BACKEND_RESOURCE_INVALID")
        if not isinstance(pids_limit, int) or isinstance(pids_limit, bool) or pids_limit < 1:
            raise DockerBackendError("DOCKER_BACKEND_RESOURCE_INVALID")
        self._image = image
        self._socket_path = Path(socket_path)
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._user = user
        self._verified: DockerRuntimeReceipt | None = None

    @property
    def verified(self) -> bool:
        return self._verified is not None

    def verify_runtime(
        self,
        *,
        runner=subprocess.run,
        require_socket: bool = True,
    ) -> DockerRuntimeReceipt:
        if require_socket and (not self._socket_path.is_socket() or self._socket_path.is_symlink()):
            raise DockerBackendError("DOCKER_BACKEND_SOCKET_MISSING")
        info = runner(
            self._docker_base() + ("info",),
            check=False,
            capture_output=True,
        )
        if info.returncode != 0:
            raise DockerBackendError("DOCKER_BACKEND_DAEMON_UNREACHABLE")
        command = self._verified_probe_command()
        probe = runner(command, check=False, capture_output=True)
        if probe.returncode != 0:
            raise DockerBackendError("DOCKER_BACKEND_RUNTIME_UNVERIFIED")
        receipt = DockerRuntimeReceipt(socket_path=self._socket_path, image=self._image, command=command)
        self._verified = receipt
        return receipt

    def run(
        self,
        *,
        worktree: Path,
        command: Sequence[str],
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        if self._verified is None:
            raise DockerBackendError("DOCKER_BACKEND_RUNTIME_UNVERIFIED")
        if not command or any(not isinstance(item, str) or "\x00" in item for item in command):
            raise DockerBackendError("DOCKER_BACKEND_COMMAND_INVALID")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise DockerBackendError("DOCKER_BACKEND_TIMEOUT_INVALID")
        root = Path(worktree).resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise DockerBackendError("DOCKER_BACKEND_WORKTREE_INVALID")
        with tempfile.TemporaryDirectory(prefix="wm-loop-docker-") as temporary:
            cidfile = Path(temporary) / "container.cid"
            docker_command = self._run_command(root, tuple(command), environment, cidfile)
            return self._run_docker_command(docker_command, cidfile=cidfile, timeout_seconds=timeout_seconds)

    def _docker_base(self) -> tuple[str, ...]:
        return ("docker", "--host", f"unix://{self._socket_path}")

    def _verified_probe_command(self) -> tuple[str, ...]:
        return self._docker_base() + (
            "run",
            "--pull=never",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self._pids_limit),
            "--memory",
            self._memory,
            "--cpus",
            str(self._cpus),
            "--user",
            self._user,
            self._image,
            "true",
        )

    def _run_command(
        self,
        worktree: Path,
        command: tuple[str, ...],
        environment: Mapping[str, str],
        cidfile: Path,
    ) -> tuple[str, ...]:
        args = self._docker_base() + (
            "run",
            "--pull=never",
            "--rm",
            "--cidfile",
            str(cidfile),
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self._pids_limit),
            "--memory",
            self._memory,
            "--cpus",
            str(self._cpus),
            "--user",
            self._user,
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,src={worktree},dst=/workspace,rw",
        )
        for key, value in sorted(environment.items()):
            if not _safe_env(key, value):
                raise DockerBackendError("DOCKER_BACKEND_ENVIRONMENT_INVALID")
            args += ("--env", f"{key}={value}")
        return args + (self._image, *command)

    def _run_docker_command(self, command: tuple[str, ...], *, cidfile: Path, timeout_seconds: float) -> CommandExecutionResult:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code: int | None = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_docker_container(self._docker_base(), cidfile)
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = process.returncode
        return CommandExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )


def _safe_env(key: str, value: str) -> bool:
    return (
        isinstance(key, str)
        and isinstance(value, str)
        and key
        and "=" not in key
        and "\x00" not in key
        and "\x00" not in value
    )


def _kill_docker_container(docker_base: tuple[str, ...], cidfile: Path) -> None:
    if not cidfile.is_file() or cidfile.is_symlink():
        return
    container_id = cidfile.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{12,64}", container_id):
        return
    subprocess.run((*docker_base, "rm", "-f", container_id), check=False, capture_output=True)
