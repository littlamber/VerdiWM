"""Command execution backends for auditable repair sessions."""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from wmloop.process_lifecycle import LocalCommandSignals, LocalProcessTree, retire_process_group


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
        signal_scope: LocalCommandSignals | None = None,
    ) -> CommandExecutionResult:
        if signal_scope is None:
            with LocalCommandSignals() as scope:
                return self.run(worktree=worktree, command=command,
                    environment=environment, timeout_seconds=timeout_seconds, signal_scope=scope)
        signal_scope.check()
        started = time.monotonic()
        timed_out = False
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = None
            tree = None
            try:
                signal_scope.check()
                process = subprocess.Popen(
                    tuple(command),
                    cwd=worktree,
                    env=dict(environment),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                tree = LocalProcessTree(process)
                deadline = started + timeout_seconds
                while True:
                    signal_scope.check()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        tree.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process is not None:
                    retire_process_group(process, tree=tree)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout, stderr = stdout_file.read(), stderr_file.read()
        signal_scope.check()
        return CommandExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )
