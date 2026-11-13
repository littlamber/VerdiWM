"""Terminal presentation primitives; project and experiment rules stay in services."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Callable, Mapping, Protocol, Sequence, TextIO
import unicodedata


_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")


def plain_text(value: object) -> str:
    """Keep resource names and error messages from issuing terminal controls."""

    text = _ESCAPES.sub("", str(value)).replace("\t", "    ")
    return "".join(char for char in text if char == "\n" or not unicodedata.category(char).startswith("C"))


def display_width(value: str) -> int:
    return sum(
        0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in plain_text(value)
    )


def wrap_lines(value: object, width: int) -> list[str]:
    """Wrap long paths and CJK text without dropping any of their characters."""

    width = max(width, 2)
    result: list[str] = []
    for paragraph in plain_text(value).split("\n"):
        line, used = "", 0
        for char in paragraph:
            cells = display_width(char)
            if used + cells > width and line:
                result.append(line)
                line, used = "", 0
            line += char
            used += cells
        result.append(line)
    return result


def clip_text(value: object, width: int) -> str:
    text = plain_text(value).replace("\n", " ")
    if display_width(text) <= width:
        return text
    return wrap_lines(text, max(width - 1, 2))[0] + "…" if width > 2 else ""


def terminal_size(stream: TextIO) -> os.terminal_size:
    try:
        size = os.get_terminal_size(stream.fileno())
    except (AttributeError, OSError, ValueError):
        size = shutil.get_terminal_size(fallback=(88, 24))
    return os.terminal_size((max(size.columns, 12), max(size.lines, 6)))


class Theme:
    """ANSI styling without forcing the user's terminal background or palette."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str, value: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.enabled else value

    def title(self, value: str) -> str:
        return self._c("1;96", value)

    def accent(self, value: str) -> str:
        return self._c("36", value)

    def muted(self, value: str) -> str:
        return self._c("90", value)

    def good(self, value: str) -> str:
        return self._c("32", value)

    def warn(self, value: str) -> str:
        return self._c("33", value)

    def bad(self, value: str) -> str:
        return self._c("31", value)


class CommandEntry(Protocol):
    name: str
    summary: str
    usage: str


@dataclass(frozen=True)
class Dashboard:
    version: str
    project_root: Path
    config: Mapping[str, object] | None
    readiness: Mapping[str, object] | None
    llm: Mapping[str, object]
    next_label: str
    next_command: str
    plan_path: Path | None = None
    latest: Mapping[str, object] | None = None
    config_error: str | None = None
    campaign_error: str | None = None


class _Panel:
    def __init__(self, stdout: TextIO, theme: Theme, width: int) -> None:
        self.stdout, self.theme = stdout, theme
        self.width = max(width, 10)
        self.content_width = self.width - 4
        encoding = getattr(stdout, "encoding", None) or "utf-8"
        try:
            "╭─╮│╰╯".encode(encoding)
            self.glyphs = "╭─╮│╰╯" if os.environ.get("TERM") != "dumb" else "+-+|++"
        except (UnicodeError, LookupError):
            self.glyphs = "+-+|++"

    def edge(self, *, bottom: bool = False) -> None:
        left, horizontal, right, _, lower_left, lower_right = self.glyphs
        if bottom:
            left, right = lower_left, lower_right
        self.stdout.write(" " + self.theme.muted(left + horizontal * (self.width - 2) + right) + "\n")

    def line(self, value: object = "", painter: Callable[[str], str] | None = None) -> None:
        side = self.theme.muted(self.glyphs[3])
        for part in wrap_lines(value, self.content_width):
            padded = part + " " * max(self.content_width - display_width(part), 0)
            self.stdout.write(" " + side + " " + (painter(padded) if painter else padded) + " " + side + "\n")

    def field(self, label: str, value: object, painter: Callable[[str], str] | None = None) -> None:
        prefix = label.ljust(9)
        if self.content_width < 24:
            self.line(label, self.theme.muted)
            self.line(value, painter)
            return
        for index, part in enumerate(wrap_lines(value, self.content_width - len(prefix))):
            self.line((prefix if index == 0 else " " * len(prefix)) + part, painter)


_NEXT_HELP = {
    "/setup": "绑定模型与数据，填写研究目标；回车接受发现的默认路径。",
    "/check": "查看每个阻塞项和解决步骤，补齐后重新检查。",
    "/run": "先查看计划与预算，再决定是否执行。",
    "/progress": "查看任务进度；/recent 可找回任务与计划。",
    "/status": "恢复原状态目录、任务库或读取权限后重新查询；不要删除已有研究记录。",
    "/start": "按当前项目配置生成下一份研究计划。",
}


def render_dashboard(snapshot: Dashboard, *, stdout: TextIO, theme: Theme, width: int | None = None) -> None:
    """Render read-only snapshots; the caller owns readiness and next-step policy."""

    width = min(width or terminal_size(stdout).columns - 2, 104)
    panel = _Panel(stdout, theme, width)
    config = snapshot.config or {}
    stdout.write("\n")
    panel.edge()
    panel.line(f"VERDI   /   v{snapshot.version}", theme.title)
    panel.line("WORLD MODEL RESEARCH CONSOLE", theme.muted)
    panel.line()
    panel.line("01  PROJECT / 项目接入", theme.accent)
    panel.field("Project", snapshot.project_root)
    panel.field("Status", snapshot.next_label, theme.bad if snapshot.config_error or snapshot.campaign_error or "BLOCKED" in snapshot.next_label else theme.warn if snapshot.next_label in {"SETUP", "GOAL REQUIRED", "CHECK REQUIRED"} else theme.good)
    if snapshot.config_error:
        panel.line(snapshot.config_error, theme.bad)
        panel.line("修正配置文件后输入 /check；/exit 可随时退出。", theme.warn)
    else:
        panel.field("Model", config.get("model") or "未绑定 · /setup")
        panel.field("Data", config.get("data", config.get("dataset")) or "未绑定 · /setup")
        panel.field("Goal", config.get("goal") or "未填写 · 例如：提升长程一致性")
    llm_state = snapshot.llm.get("state", "missing")
    llm_label = "READY · 研究助手已配置" if llm_state == "ready" else "未配置 · /llm 查看接入说明" if llm_state == "missing" else f"{llm_state} · /llm status 查看原因"
    panel.field("LLM", llm_label, theme.good if llm_state == "ready" else theme.muted)
    if snapshot.plan_path is not None:
        panel.field("Plan", snapshot.plan_path)
    if snapshot.latest:
        panel.field("Latest", f"{snapshot.latest.get('campaign_id', '?')} · {snapshot.latest.get('status', '?')}")
    if snapshot.campaign_error:
        panel.line()
        panel.line("任务记录暂不可读取", theme.bad)
        panel.line(snapshot.campaign_error, theme.bad)
    blockers = snapshot.readiness.get("blockers") if snapshot.readiness else None
    if isinstance(blockers, list) and blockers:
        first = blockers[0]
        panel.line()
        panel.line(f"接入待解决 · {len(blockers)} 项", theme.warn)
        if isinstance(first, dict):
            panel.line(first.get("message", first.get("code", "请运行 /check")), theme.warn)
            if first.get("action"):
                panel.line(first["action"])
    panel.line()
    panel.line("02  NEXT / 下一步", theme.accent)
    panel.line(snapshot.next_command, theme.title)
    panel.line(_NEXT_HELP.get(snapshot.next_command, "按上面的建议继续；/help 可查看各命令的用法。"))
    panel.edge(bottom=True)
    footer = [
        "/ + 回车  命令面板    /start  继续    /recent  最近任务",
        "方法路径：刻画 → 检索 → 验证 → 积累",
        "可直接描述研究目标；计划先审阅，正式实验需明确确认。",
    ]
    for item in footer:
        for part in wrap_lines(item, max(width - 2, 2)):
            stdout.write("  " + theme.muted(part) + "\n")
    stdout.write("\n")


_COMMAND_GROUPS = (
    ("开始 / START", {"start", "setup", "check", "plan", "run", "research"}),
    ("任务 / WORK", {"recent", "progress", "status", "resume", "cancel"}),
    ("接入与诊断 / CONNECT", {"models", "guide", "doctor", "llm", "evaluator", "diagnose", "evaluate-irg", "evaluate-predictor"}),
    ("会话 / SESSION", {"help", "clear", "version", "exit"}),
)


def render_commands(commands: Sequence[CommandEntry], *, query: str, stdout: TextIO, theme: Theme) -> None:
    panel = _Panel(stdout, theme, min(terminal_size(stdout).columns - 2, 104))
    stdout.write("\n")
    panel.edge()
    panel.line("Command palette / 命令面板", theme.title)
    if query.strip():
        panel.line(f"搜索  {query}", theme.muted)
    panel.line("输入 /命令 执行；/help plan 查看单个命令。", theme.muted)
    if not commands:
        panel.line("没有匹配命令。输入 /help 查看全部命令。", theme.warn)
    for group, names in _COMMAND_GROUPS:
        items = [command for command in commands if command.name in names]
        if not items:
            continue
        panel.line()
        panel.line(group, theme.accent)
        for command in items:
            if panel.content_width >= 52:
                panel.field("/" + command.name.ljust(19), command.summary)
            else:
                panel.line("/" + command.name, theme.accent)
                panel.line(command.summary)
    panel.edge(bottom=True)
    stdout.write("\n")
