"""Command execution backends for auditable repair sessions."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class CommandExecutionResult:
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    timed_out: bool
    duration_seconds: float


class CommandBackend(Protocol):
    def run(
        self,
        *,
        worktree: Path,
        command: Sequence[str],
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        """Run one command and return captured output."""


class LocalSubprocessBackend:
    """Current local execution backend used when no stronger sandbox is wired."""

    def run(
        self,
        *,
        worktree: Path,
        command: Sequence[str],
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        started = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            tuple(command),
            cwd=worktree,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code: int | None = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
            stdout, stderr = process.communicate()
            exit_code = process.returncode
        return CommandExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
