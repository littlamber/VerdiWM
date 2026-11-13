"""Signal-aware cancellation and retirement shared across local CPU runners."""

from __future__ import annotations

import os
import signal
import subprocess
import threading


class ProcessLifecycleError(ValueError):
    """Local execution lacks a valid main-thread cancellation scope."""


class LocalProcessCancelled(KeyboardInterrupt):
    """Cancellation propagates past ordinary trial failure handlers."""


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


def retire_process_group(process: subprocess.Popen) -> None:
    """Kill the command's session group, including after its leader exits."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
