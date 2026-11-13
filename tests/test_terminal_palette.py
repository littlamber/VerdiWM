from __future__ import annotations

import os
import select
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(os.name != "posix", reason="raw terminal picker requires POSIX termios")


def _read_until(fd: int, expected: bytes, received: bytearray) -> None:
    deadline = time.monotonic() + 8
    while expected not in received:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            pytest.fail(f"terminal did not produce {expected!r}: {bytes(received)!r}")
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            pytest.fail(f"terminal closed before {expected!r}: {bytes(received)!r}")
        received.extend(chunk)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (b"sta\x1b[B\r", "status"),
        ("版本\r".encode(), "version"),
        (b"\x1b[A\r", "exit"),
        (b"\x04", "None"),
    ],
)
def test_real_terminal_palette_consumes_keys_and_restores_mode(keys: bytes, expected: str) -> None:
    import fcntl
    import struct
    import termios

    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 16, 50, 0, 0))
    # Exercise real fd reads and TextIOWrapper, rather than a StringIO stand-in.
    code = """
import sys, termios
from wmloop.control.interactive_cli import _matching_commands
from wmloop.control.terminal_palette import select_command
from wmloop.control.terminal_ui import Theme
before = termios.tcgetattr(sys.stdin.fileno())
selected = select_command(lambda query: tuple(_matching_commands(query)), stdin=sys.stdin, stdout=sys.stdout, theme=Theme(True))
assert termios.tcgetattr(sys.stdin.fileno()) == before
sys.stdout.write('SELECTED=' + str(selected) + '\\n')
sys.stdout.flush()
"""
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code], stdin=slave, stdout=slave, stderr=slave,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, close_fds=True,
        )
        os.close(slave)
        slave = -1
        received = bytearray()
        _read_until(master, b"Command palette", received)
        os.write(master, keys)
        _read_until(master, f"SELECTED={expected}".encode(), received)
        assert process.wait(timeout=5) == 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        if slave >= 0:
            os.close(slave)
