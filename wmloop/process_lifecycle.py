"""Signal-aware cancellation and retirement shared across local CPU runners."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


class ProcessLifecycleError(ValueError):
    """Local execution cannot preserve its cancellation or cleanup contract."""


class LocalProcessCancelled(KeyboardInterrupt):
    """Cancellation propagates past ordinary trial failure handlers."""


def _read_proc_stat(pid: int) -> tuple[str, str, int] | None:
    """Read identity, state and group without decoding an arbitrary comm.

    Linux permits non-UTF-8 bytes and closing parentheses in process names.
    Only the kernel's ASCII fields after the final parenthesis are text.
    Permission failures remain visible for tracked processes; the full-proc
    discovery scan separately skips entries it is not allowed to inspect.
    """
    try:
        record = (Path('/proc') / str(pid) / 'stat').read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        fields = record.rsplit(b')', 1)[1].split()
        return fields[19].decode('ascii'), fields[0].decode('ascii'), int(fields[2])
    except (IndexError, ValueError) as exc:
        raise ProcessLifecycleError(f'LOCAL_PROCESS_INVALID_PROC_STAT:{pid}') from exc


class LocalCommandSignals:
    """Defer cancellation until each command has retired its process group.

    Enter on the main thread, then pass the scope explicitly to worker calls.
    The handler only records a request: raising inside Popen can lose the
    child between its creation and assignment to the caller's process handle.
    """

    def __init__(self) -> None:
        self._previous: dict[int, object] = {}
        self._signum: int | None = None
        self._active = False

    def __enter__(self) -> LocalCommandSignals:
        if threading.current_thread() is not threading.main_thread():
            raise ProcessLifecycleError('LOCAL_PROCESS_MAIN_THREAD_SIGNAL_SCOPE_REQUIRED')
        if self._active:
            raise ProcessLifecycleError('LOCAL_PROCESS_SIGNAL_SCOPE_ALREADY_ACTIVE')
        self._signum = None
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous = signal.getsignal(signum)
                self._previous[signum] = previous
                signal.signal(signum, self._request_stop)
        except BaseException:
            self._restore()
            raise
        self._active = True
        return self

    def _request_stop(self, signum: int, _frame: object) -> None:
        if self._previous[signum] != signal.SIG_IGN and self._signum is None:
            self._signum = signum

    def check(self) -> None:
        if not self._active:
            raise ProcessLifecycleError('LOCAL_PROCESS_SIGNAL_SCOPE_INACTIVE')
        if self._signum is not None:
            raise LocalProcessCancelled('LOCAL_PROCESS_CANCELLED')

    @property
    def stop_requested(self) -> bool:
        """Let workers retire their commands before the main thread raises."""
        return self._signum is not None

    def _restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()
        self._active = False

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        signum = self._signum
        previous = self._previous.get(signum)
        self._restore()
        if signum is not None:
            if callable(previous):
                previous(signum, None)
            if signum == signal.SIGTERM:
                raise SystemExit(128 + signum)
            raise KeyboardInterrupt
        return False


class LocalProcessTree:
    """Track Linux descendants that start sessions outside the command group.

    Observe while the leader is alive so that its children remain identifiable
    after reparenting. Start times prevent a recycled PID from being signalled.
    This is cooperative local cleanup, not isolation against hostile forks.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self._identities: dict[int, str] = {}
        self.observe()

    @staticmethod
    def _identity(pid: int) -> tuple[str, str] | None:
        stat = _read_proc_stat(pid)
        return None if stat is None else stat[:2]

    def observe(self) -> None:
        pending = [self.process.pid, *self._identities]
        visited: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            identity = self._identity(pid)
            if identity is None or (
                pid in self._identities and self._identities[pid] != identity[0]
            ):
                continue
            self._identities[pid] = identity[0]
            try:
                for task in (Path('/proc') / str(pid) / 'task').glob('*'):
                    try:
                        pending.extend(
                            int(value)
                            for value in (task / 'children').read_text().split()
                        )
                    except (FileNotFoundError, ProcessLookupError):
                        continue
            except (FileNotFoundError, ProcessLookupError):
                # The process can exit after its identity check but before
                # /proc/<pid>/task is enumerated (or while glob advances).
                continue

    def poll(self) -> int | None:
        """Inspect completion without releasing the leader's PID identity.

        Keeping an exited leader waitable anchors its original process group
        until retirement. ``Popen.poll`` would reap it before an unobserved
        fast child could be discovered, making a remembered PGID unsafe.
        """
        if self.process.returncode is not None:
            return self.process.returncode
        result = os.waitid(
            os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
        if result is None:
            return None
        return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status

    def wait(self, timeout: float | None = None) -> int:
        """Wait for leader completion while retaining its cleanup identity."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while (returncode := self.poll()) is None:
            self.observe()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self.process.args, timeout)
                time.sleep(min(0.05, remaining))
            else:
                time.sleep(0.05)
        return returncode

    def _observe_anchored_groups(self) -> set[int]:
        groups: set[int] = set()
        for pid, started in tuple(self._identities.items()):
            identity = self._identity(pid)
            if identity is None or identity[0] != started:
                continue
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if group in {pid, self.process.pid}:
                # A waitable zombie leader still reserves its PID/PGID. Its
                # identity is an anchor even when no live member was observed.
                groups.add(group)
        if not groups:
            return groups
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = _read_proc_stat(int(entry.name))
            except PermissionError:
                continue
            if stat is not None and stat[2] in groups:
                self._identities[int(entry.name)] = stat[0]
        return groups

    def _live_pids(self) -> list[int]:
        return [
            pid for pid, started in self._identities.items()
            if (identity := self._identity(pid)) is not None
            and identity[0] == started and identity[1] not in {'Z', 'X'}
        ]

    def _group_is_anchored(self, group: int) -> bool:
        for pid, started in self._identities.items():
            identity = self._identity(pid)
            if identity is None or identity[0] != started:
                continue
            try:
                if os.getpgid(pid) == group:
                    return True
            except ProcessLookupError:
                continue
        return False

    def _signal(self, signum: int) -> None:
        self.observe()
        groups = self._observe_anchored_groups()
        members: dict[int, int] = {}
        for pid in self._live_pids():
            try:
                # Each observed session leader also owns any children spawned
                # since the last observation in that process group.
                group = os.getpgid(pid)
                if group in {pid, self.process.pid}:
                    groups.add(group)
                members[pid] = group
            except ProcessLookupError:
                continue
        # A current, identity-checked member (including our waitable leader)
        # binds each group; an old PGID alone never authorizes signalling.
        for group in groups:
            if not self._group_is_anchored(group):
                continue
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                continue
        for pid, group in members.items():
            if group not in groups:
                identity = self._identity(pid)
                if identity is None or identity[0] != self._identities[pid]:
                    continue
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    continue

    def _wait_retired(self) -> None:
        deadline = time.monotonic() + 5.0
        while self._live_pids():
            if time.monotonic() >= deadline:
                raise ProcessLifecycleError('LOCAL_PROCESS_DESCENDANTS_STILL_RUNNING')
            time.sleep(0.01)

    def terminate(self, *, grace_seconds: float) -> str:
        """Request graceful shutdown, then kill every surviving descendant."""
        self._signal(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while self._live_pids() and time.monotonic() < deadline:
            self.observe()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        termination = 'SIGKILL' if self._live_pids() else 'SIGTERM'
        retire_process_group(self.process, tree=self)
        return termination


def retire_process_group(
    process: subprocess.Popen, *, tree: LocalProcessTree | None = None
) -> None:
    """Retire an identity-bound tree, then reap its command leader.

    Construct a tree before waiting, and use its non-reaping poll/wait. Legacy
    callers may omit the tree only while their leader remains waitable.
    """
    if tree is None:
        if process.returncode is not None:
            raise ProcessLifecycleError('LOCAL_PROCESS_LEADER_ALREADY_REAPED')
        tree = LocalProcessTree(process)
    tree._signal(signal.SIGKILL)
    process.wait()
    tree._wait_retired()
