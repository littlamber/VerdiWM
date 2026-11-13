"""Bounded, keyboard-driven command picker for a real terminal."""

from __future__ import annotations

import codecs
import os
import select
from typing import Callable, Sequence, TextIO

from wmloop.control.terminal_ui import CommandEntry, Theme, clip_text, terminal_size


def palette_lines(
    matches: Sequence[CommandEntry], query: str, selected: int, *, theme: Theme, columns: int, rows: int
) -> list[str]:
    """Always keep the selected command visible, including on small screens."""

    width = max(columns - 2, 8)
    capacity = min(10, max(rows - 6, 1))
    selected = min(max(selected, 0), max(len(matches) - 1, 0))
    start = min(max(selected - capacity // 2, 0), max(len(matches) - capacity, 0))
    lines = [
        theme.title(clip_text("  Command palette / 命令面板", width)),
        theme.muted(clip_text("  ↑↓ 选择 · Enter 执行 · 输入筛选 · Esc 返回", width)),
        theme.accent(clip_text("  /" + query, width)),
    ]
    if matches:
        for index in range(start, min(start + capacity, len(matches))):
            command = matches[index]
            label = f"  {'>' if index == selected else ' '} /{command.name:<19} {command.summary}"
            lines.append((theme.accent if index == selected else str)(clip_text(label, width)))
        lines.append(theme.muted(clip_text(f"  {selected + 1}/{len(matches)} · {matches[selected].usage}", width)))
    else:
        lines.append(theme.warn(clip_text("  没有匹配命令；Esc 返回。", width)))
    return lines


def draw_palette(
    matches: Sequence[CommandEntry], query: str, selected: int, *, stdout: TextIO, theme: Theme, previous_lines: int
) -> int:
    size = terminal_size(stdout)
    lines = palette_lines(matches, query, selected, theme=theme, columns=size.columns, rows=size.lines)
    if previous_lines:
        stdout.write(f"\033[{previous_lines}A")
    for line in lines:
        stdout.write("\r\033[2K" + line + "\n")
    surplus = max(previous_lines - len(lines), 0)
    for _ in range(surplus):
        stdout.write("\r\033[2K\n")
    if surplus:
        stdout.write(f"\033[{surplus}A")
    stdout.flush()
    return len(lines)


def _escape_key(fd: int) -> str:
    """Read bytes from the fd, without TextIOWrapper's invisible read-ahead."""

    sequence = b""
    for _ in range(5):
        ready, _, _ = select.select([fd], [], [], 0.08)
        if not ready:
            break
        value = os.read(fd, 1)
        if not value:
            break
        sequence += value
        if len(sequence) >= 2 and (sequence[-1:] in b"ABHF~" or sequence[:1] not in {b"[", b"O"}):
            break
    return {b"[A": "up", b"OA": "up", b"[B": "down", b"OB": "down", b"[H": "home", b"[F": "end"}.get(sequence, "escape")


def select_command(
    matching_commands: Callable[[str], Sequence[CommandEntry]], *, stdin: TextIO, stdout: TextIO, theme: Theme
) -> str | None:
    try:
        import termios
        import tty

        fd = stdin.fileno()
        previous_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, ImportError, ValueError):
        return None
    query, selected, previous_lines = "", 0, 0
    decoder = codecs.getincrementaldecoder(getattr(stdin, "encoding", None) or "utf-8")(errors="replace")
    try:
        tty.setcbreak(fd)
        while True:
            matches = list(matching_commands(query))
            selected = min(selected, max(len(matches) - 1, 0))
            previous_lines = draw_palette(matches, query, selected, stdout=stdout, theme=theme, previous_lines=previous_lines)
            value = os.read(fd, 1)
            if not value or value in {b"\x03", b"\x04"}:
                return None
            if value in {b"\r", b"\n"}:
                return matches[selected].name if matches else None
            if value == b"\x1b":
                key = _escape_key(fd)
                if key == "escape":
                    return None
                if key in {"up", "down"}:
                    selected = (selected + (1 if key == "down" else -1)) % max(len(matches), 1)
                elif key == "home":
                    selected = 0
                elif key == "end":
                    selected = max(len(matches) - 1, 0)
                continue
            if value in {b"\x7f", b"\b"}:
                query, selected = query[:-1], 0
                decoder.reset()
            elif value == b"\t" and matches:
                selected = (selected + 1) % len(matches)
            else:
                char = decoder.decode(value)
                if char and char.isprintable():
                    query, selected = query + char, 0
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)
        if previous_lines:
            stdout.write(f"\033[{previous_lines}A")
            for _ in range(previous_lines):
                stdout.write("\r\033[2K\n")
            stdout.write(f"\033[{previous_lines}A")
        stdout.flush()
