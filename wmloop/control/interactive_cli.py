"""Small, dependency-free interactive shell for the public ``verdi`` CLI.

The shell deliberately stays a thin presentation layer over the existing CLI
handlers.  It never starts a campaign from natural-language input; users must
review and confirm a generated research plan before expensive work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import select
import shlex
import sys
import threading
import time
from typing import Callable, Iterable, TextIO


Dispatch = Callable[[list[str]], int]


@dataclass(frozen=True)
class InteractiveCommand:
    name: str
    aliases: tuple[str, ...]
    summary: str
    usage: str


@dataclass
class SessionState:
    """Ephemeral context that makes consecutive shell commands composable."""

    last_goal: str | None = None
    last_plan: str | None = None
    last_campaign_id: str | None = None
    last_state: str | None = None
    last_command: str | None = None
    last_error: str | None = None


COMMANDS: tuple[InteractiveCommand, ...] = (
    InteractiveCommand("help", ("h", "?"), "显示命令面板和使用说明", "/help"),
    InteractiveCommand("research", ("r",), "进入研究计划或执行流程", "/research plan ..."),
    InteractiveCommand("start", ("go", "next"), "按当前项目状态继续下一步", "/start"),
    InteractiveCommand("plan", (), "生成可审阅的研究计划（只读）", "/plan \"研究目标\""),
    InteractiveCommand("run", (), "预览并确认后执行研究计划", "/run  或  /run --plan PATH --confirm"),
    InteractiveCommand("status", ("s",), "查看 campaign 状态", "/status [CAMPAIGN_ID]"),
    InteractiveCommand("recent", (), "查看最近计划、任务和下一步", "/recent"),
    InteractiveCommand("resume", (), "恢复最近任务或审阅默认计划", "/resume"),
    InteractiveCommand("progress", ("p",), "查看最近任务的进度", "/progress [CAMPAIGN_ID]"),
    InteractiveCommand("cancel", (), "取消一个排队或运行中的任务", "/cancel CAMPAIGN_ID"),
    InteractiveCommand("check", (), "检查项目接入和本地 readiness", "/check"),
    InteractiveCommand("doctor", (), "检查 Verdi 本地安装", "/doctor"),
    InteractiveCommand("diagnose", (), "只读诊断模型项目", "/diagnose"),
    InteractiveCommand("evaluator", ("eval",), "发现并查看评测候选（需确认后才能冻结）", "/evaluator discover --source-root PATH"),
    InteractiveCommand("setup", ("configure",), "首次接入模型、数据和目标", "/setup"),
    InteractiveCommand("guide", (), "生成模型接入问卷", "/guide"),
    InteractiveCommand("models", (), "查看当前项目绑定", "/models"),
    InteractiveCommand("clear", (), "清屏并重新显示欢迎界面", "/clear"),
    InteractiveCommand("version", (), "显示 Verdi 版本", "/version"),
    InteractiveCommand("exit", ("quit", "q"), "退出交互会话", "/exit"),
)


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").casefold() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


class _Theme:
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


def interactive_supported(stdin: TextIO, stdout: TextIO) -> bool:
    """Return whether it is safe to enter a blocking interactive session."""

    try:
        return bool(stdin.isatty() and stdout.isatty())
    except (AttributeError, OSError):
        return False


def _version() -> str:
    for package in ("verdiwm", "verdi"):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "source"


def _project_snapshot(project_root: Path | None = None) -> dict[str, object] | None:
    try:
        from wmloop.control.project_config import load_project_config

        return dict(load_project_config(cwd=project_root).values)
    except Exception:  # a status panel should never make the shell unusable
        return None


def _discovered_inputs() -> tuple[str | None, str | None]:
    """Discover conventional paths for the setup wizard without scanning code."""

    try:
        from wmloop.control.first_contact import discover_project_inputs

        model, data = discover_project_inputs(Path.cwd())
        return (str(model) if model else None, str(data) if data else None)
    except Exception:
        return None, None


def _default_plan_path(project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).expanduser().resolve()
    return root / ".verdiwm" / "research-plan.json"


def _state_root(config: dict[str, object] | None, project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).expanduser().resolve()
    configured = config.get("state_root") if config else None
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return root / ".verdiwm" / "state"


def _readiness_snapshot(config: dict[str, object] | None, project_root: Path) -> dict[str, object] | None:
    """Run the bounded, read-only onboarding check for welcome/start panels."""

    if not config:
        return None
    try:
        from wmloop.control.first_contact import inspect_project

        return inspect_project(
            root=project_root,
            model=str(config.get("model")) if config.get("model") else None,
            source=str(config.get("source")) if config.get("source") else None,
            data=str(config.get("data", config.get("dataset"))) if config.get("data", config.get("dataset")) else None,
            evaluator_contract=str(config.get("evaluator_contract")) if config.get("evaluator_contract") else None,
            runtime_python=str(config.get("runtime_python")) if config.get("runtime_python") else None,
        )
    except Exception:
        return None


def _recent_campaigns(config: dict[str, object] | None, project_root: Path | None = None, *, limit: int = 5) -> list[dict[str, object]]:
    try:
        from wmloop.control.campaign_api import CampaignStore

        records = CampaignStore(_state_root(config, project_root), read_only=True).list(limit=1000)
    except Exception:
        return []
    records.sort(key=lambda item: str(item.get("updated_at", item.get("created_at", ""))), reverse=True)
    return records[:limit]


def _next_step(
    config: dict[str, object] | None,
    *,
    readiness: dict[str, object] | None = None,
    plan_path: Path | None = None,
    campaigns: Iterable[dict[str, object]] = (),
) -> tuple[str, str]:
    if not config:
        return "SETUP", "/setup"
    if not config.get("goal"):
        return "GOAL REQUIRED", "/plan \"你的研究目标\""
    blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
    if isinstance(blockers, list) and blockers:
        return "BLOCKED", "/check"
    for campaign in campaigns:
        if campaign.get("status") in {"queued", "running"}:
            return "IN PROGRESS", "/progress"
    if plan_path is not None and plan_path.is_file():
        return "PLAN READY", "/run"
    return "READY", "/start"


def render_welcome(*, stdout: TextIO = sys.stdout, project_root: Path | None = None, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    root = (project_root or Path.cwd()).expanduser().resolve()
    config = _project_snapshot(root)
    model = config.get("model") if config else None
    data = config.get("data", config.get("dataset")) if config else None
    goal = config.get("goal") if config else None
    readiness = _readiness_snapshot(config, root)
    campaigns = _recent_campaigns(config, root)
    blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
    ready = bool(config and model and data and goal and not blockers)
    status = theme.good("READY") if ready else theme.bad("BLOCKED") if blockers else theme.warn("NEEDS SETUP")
    plan_path = _default_plan_path(root)
    next_label, next_command = _next_step(config, readiness=readiness, plan_path=plan_path, campaigns=campaigns)
    stdout.write("\n")
    stdout.write(theme.title(f"  VERDI  v{_version()}\n"))
    stdout.write(theme.muted("  Evidence-driven world-model research\n\n"))
    stdout.write(f"  Project  {root}\n")
    stdout.write(f"  Status   {status}\n")
    stdout.write(f"  Model    {model or '未绑定（输入 /setup）'}\n")
    stdout.write(f"  Data     {data or '未绑定（输入 /setup）'}\n")
    if goal:
        stdout.write(f"  Goal     {goal}\n")
    if plan_path.is_file():
        stdout.write(f"  Plan     {plan_path}\n")
    if campaigns:
        latest = campaigns[0]
        stdout.write(f"  Latest   {latest.get('campaign_id', '?')}  {latest.get('status', '?')}\n")
    if isinstance(blockers, list) and blockers:
        first = blockers[0]
        if isinstance(first, dict):
            stdout.write(theme.warn(f"  Blocker  {first.get('code', 'BLOCKED')}: {first.get('message', '')}\n"))
    stdout.write(f"  Next     {theme.accent(next_label)}  {next_command}\n")
    stdout.write("\n")
    stdout.write(theme.accent("  输入 / 查看命令，或直接描述一个研究目标。\n"))
    stdout.write(theme.muted("  计划生成是只读的；正式实验始终需要显式确认。\n\n"))


def _matching_commands(query: str = "") -> Iterable[InteractiveCommand]:
    needle = query.strip().lstrip("/").casefold()
    if not needle:
        return COMMANDS
    return tuple(
        command
        for command in COMMANDS
        if needle in command.name.casefold()
        or any(needle in alias.casefold() for alias in command.aliases)
        or needle in command.summary.casefold()
    )


def render_command_palette(query: str = "", *, stdout: TextIO = sys.stdout, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    matches = tuple(_matching_commands(query))
    label = query.strip() or "/"
    stdout.write(f"\n{theme.title('  Command palette')}  {theme.muted(label)}\n")
    if not matches:
        stdout.write(theme.warn("  没有匹配命令。输入 /help 查看全部命令。\n\n"))
        return
    for command in matches:
        aliases = f" ({', '.join('/' + alias for alias in command.aliases)})" if command.aliases else ""
        label_text = ("/" + command.name).ljust(18)
        stdout.write(f"  {theme.accent(label_text)} {command.summary}{aliases}\n")
        stdout.write(f"  {theme.muted(' ' * 18 + command.usage)}\n")
    stdout.write("\n")


def _draw_live_palette(
    matches: list[InteractiveCommand],
    query: str,
    selected: int,
    *,
    stdout: TextIO,
    theme: _Theme,
    previous_lines: int,
) -> int:
    """Draw a small terminal-native palette used after typing ``/``.

    This uses only ANSI cursor movement and the standard terminal APIs.  It is
    deliberately enabled only for the real stdin/stdout pair, so redirected
    streams and tests retain deterministic line output.
    """

    if previous_lines:
        stdout.write(f"\033[{previous_lines}A")
    visible = matches[:10]
    lines = [
        f"  {theme.title('Command palette')}  {theme.muted('/' + query)}",
        theme.muted("  ↑/↓ choose   Enter run   type to filter   Esc close"),
    ]
    if visible:
        for index, command in enumerate(visible):
            marker = ">" if index == selected else " "
            label = f"/{command.name}".ljust(18)
            line = f"  {marker} {label} {command.summary}"
            lines.append(theme.accent(line) if index == selected else line)
    else:
        lines.append(theme.warn("  没有匹配命令；Esc 关闭面板。"))
    for line in lines:
        stdout.write("\033[2K\r" + line + "\n")
    stdout.flush()
    return len(lines)


def _live_palette_selection(*, stdin: TextIO, stdout: TextIO, theme: _Theme) -> str | None:
    """Let a real TTY select a slash command with arrows and live filtering."""

    if stdin is not sys.stdin or stdout is not sys.stdout or not theme.enabled:
        return None
    try:
        import termios
        import tty

        fd = stdin.fileno()
        previous_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, ImportError):
        return None
    query = ""
    selected = 0
    previous_lines = 0
    try:
        tty.setcbreak(fd)
        while True:
            matches = list(_matching_commands(query))
            if matches:
                selected = min(selected, len(matches) - 1)
            else:
                selected = 0
            previous_lines = _draw_live_palette(
                matches,
                query,
                selected,
                stdout=stdout,
                theme=theme,
                previous_lines=previous_lines,
            )
            char = stdin.read(1)
            if char == "":
                return None
            if char in {"\r", "\n"}:
                return matches[selected].name if matches else None
            if char in {"\x03", "\x1b"}:
                if char == "\x1b":
                    # Arrow keys send ESC [ A/B.  A bare ESC closes the panel.
                    sequence = ""
                    for _ in range(2):
                        ready, _, _ = select.select([stdin], [], [], 0.05)
                        if not ready:
                            break
                        sequence += stdin.read(1)
                    if sequence == "[A":
                        selected = (selected - 1) % max(len(matches), 1)
                        continue
                    if sequence == "[B":
                        selected = (selected + 1) % max(len(matches), 1)
                        continue
                return None
            if char in {"\x7f", "\b"}:
                query = query[:-1]
                selected = 0
            elif char == "\t" and matches:
                selected = (selected + 1) % len(matches)
            elif char.isprintable():
                query += char
                selected = 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)
        # Clear the live menu and leave a clean prompt line behind.
        if previous_lines:
            stdout.write(f"\033[{previous_lines}A")
            for _ in range(previous_lines):
                stdout.write("\033[2K\r\n")
            stdout.write(f"\033[{previous_lines}A")
        stdout.flush()


def _clear_screen(stdout: TextIO, theme: _Theme) -> None:
    if theme.enabled:
        stdout.write("\033[2J\033[H")
    render_welcome(stdout=stdout, color=theme.enabled)


def _prompt_value(
    stdin: TextIO,
    stdout: TextIO,
    prompt: str,
    *,
    default: str | None = None,
) -> str | None:
    suffix = f" [{default}]" if default else ""
    stdout.write(f"  {prompt}{suffix}: ")
    stdout.flush()
    try:
        if stdin is sys.stdin and stdout is sys.stdout:
            value = input()
        else:
            value = stdin.readline()
    except (EOFError, KeyboardInterrupt):
        stdout.write("\n")
        return None
    if value == "":
        return None
    value = value.strip()
    return value or default


def _prompt_confirmation(stdin: TextIO, stdout: TextIO, theme: _Theme, prompt: str) -> bool:
    stdout.write(f"  {prompt} [y/N]: ")
    stdout.flush()
    try:
        value = input() if stdin is sys.stdin and stdout is sys.stdout else stdin.readline()
    except (EOFError, KeyboardInterrupt):
        stdout.write("\n")
        return False
    return value.strip().casefold() in {"y", "yes"}


def _render_command_help(command: InteractiveCommand, *, stdout: TextIO, theme: _Theme) -> None:
    stdout.write(f"\n{theme.title('  /' + command.name)}  {command.summary}\n")
    stdout.write(f"  用法  {command.usage}\n")
    if command.name == "start":
        stdout.write("  说明  自动发现 model/ 和 data/；项目已准备好时直接生成研究计划。\n")
    elif command.name == "plan":
        stdout.write("  说明  只读生成计划；目标可直接作为第一个参数，不会启动 GPU。\n")
    elif command.name == "run":
        stdout.write("  说明  无参数时会预览默认计划并询问确认；正式 campaign 仍必须带 --confirm。\n")
    stdout.write("\n")


def _guided_setup(
    stdin: TextIO,
    stdout: TextIO,
    dispatch: Dispatch,
    theme: _Theme,
) -> bool:
    """Collect only the three facts needed for a first project configuration."""

    model_default, data_default = _discovered_inputs()
    stdout.write("\n" + theme.title("  首次接入 Verdi") + "\n")
    stdout.write(theme.muted("  只需回答三个问题，系统会生成本地 verdiwm.toml。\n"))
    model = _prompt_value(stdin, stdout, "模型目录", default=model_default)
    if model is None:
        stdout.write(theme.warn("  已取消设置。输入 /setup --help 查看完整参数。\n"))
        return True
    data = _prompt_value(stdin, stdout, "数据目录", default=data_default)
    if data is None:
        stdout.write(theme.warn("  已取消设置。\n"))
        return True
    goal = _prompt_value(stdin, stdout, "研究目标（例如：提升分钟级长程一致性）")
    if goal is None:
        stdout.write(theme.warn("  还需要一句研究目标；设置未执行。\n"))
        return True
    result = _dispatch_slash(
        "setup",
        ["--model", model, "--data", data, "--goal", goal],
        dispatch,
        stdout=stdout,
        theme=theme,
    )
    if result:
        render_welcome(stdout=stdout, color=theme.enabled)
        stdout.write(theme.accent("  接入完成后可以直接输入 /plan 生成只读研究计划。\n"))
    return result


def _plan_summary(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if path.is_symlink() or not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "计划文件无法读取，请重新生成 /plan。"
    if not isinstance(payload, dict):
        return None, "计划文件格式无效，请重新生成 /plan。"
    return payload, None


def _render_plan_preview(payload: dict[str, object], path: Path, *, stdout: TextIO, theme: _Theme) -> None:
    stdout.write(theme.title("  研究计划") + "\n")
    stdout.write(f"    Goal     {payload.get('goal', '未填写')}\n")
    stdout.write(f"    Model    {payload.get('model', '未绑定')}\n")
    stdout.write(f"    Budget   {payload.get('budget', '未填写')}\n")
    stdout.write(f"    Mode     {payload.get('mode', '未填写')}\n")
    stdout.write(f"    State    {payload.get('state', 'unknown')}\n")
    stdout.write(f"    File     {path}\n")


def _render_recent(
    *,
    stdout: TextIO,
    theme: _Theme,
    project_root: Path | None = None,
    state: SessionState | None = None,
) -> bool:
    root = (project_root or Path.cwd()).expanduser().resolve()
    config = _project_snapshot(root)
    plan_path = _default_plan_path(root)
    campaigns = _recent_campaigns(config, root, limit=8)
    stdout.write("\n" + theme.title("  Recent Verdi work") + "\n")
    if plan_path.is_file():
        payload, error = _plan_summary(plan_path)
        if payload is not None:
            stdout.write(f"  Plan     {plan_path}\n")
            stdout.write(f"  Goal     {payload.get('goal', '未填写')}\n")
            stdout.write(f"  State    {payload.get('state', 'unknown')}\n")
        elif error:
            stdout.write(theme.warn(f"  Plan     {error}\n"))
    else:
        stdout.write(theme.muted("  Plan     尚未生成（输入 /plan）。\n"))
    if campaigns:
        stdout.write("  Campaigns\n")
        for item in campaigns:
            stdout.write(f"    {item.get('campaign_id', '?')}  {item.get('status', '?')}  {item.get('goal', '')}\n")
    else:
        stdout.write(theme.muted("  Campaigns 尚无记录。\n"))
    active = next((item for item in campaigns if item.get("status") in {"queued", "running"}), None)
    if active:
        stdout.write(theme.accent(f"  Next     /progress {active.get('campaign_id')}\n"))
    elif plan_path.is_file():
        stdout.write(theme.accent("  Next     /run\n"))
    elif config and config.get("goal"):
        stdout.write(theme.accent("  Next     /plan\n"))
    else:
        stdout.write(theme.accent("  Next     /setup\n"))
    stdout.write("\n")
    if state is not None and campaigns:
        state.last_campaign_id = str(campaigns[0].get("campaign_id"))
        state.last_state = str(campaigns[0].get("status"))
    return True


def _configure_readline(stdin: TextIO, stdout: TextIO) -> tuple[object | None, object | None, object | None, str | None]:
    if stdin is not sys.stdin or stdout is not sys.stdout:
        return None, None, None, None
    try:
        import readline  # type: ignore
    except ImportError:
        return None, None, None, None

    names = sorted({command.name for command in COMMANDS} | {alias for command in COMMANDS for alias in command.aliases})

    def completion_candidates(line: str, text: str) -> list[str]:
        if not line.startswith("/"):
            return []
        body = line[1:]
        try:
            tokens = shlex.split(body)
        except ValueError:
            tokens = body.split()
        command = tokens[0].casefold() if tokens else ""
        if command in {"run", "resume"} and len(tokens) <= 1:
            plan = _default_plan_path()
            return [str(plan)] if plan.is_file() else []
        if command in {"status", "progress", "p", "cancel"} and len(tokens) <= 1:
            campaigns = _recent_campaigns(_project_snapshot(), Path.cwd(), limit=20)
            return [str(item.get("campaign_id")) for item in campaigns if item.get("campaign_id")]
        if len(tokens) <= 1:
            return ["/" + name for name in names if ("/" + name).startswith(text)]
        return []

    def completer(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        candidates = completion_candidates(line, text)
        return candidates[state] if state < len(candidates) else None

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()
    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        history = Path(os.environ.get("VERDI_HISTORY_FILE", "~/.local/state/verdiwm/cli-history")).expanduser()
        history.parent.mkdir(parents=True, exist_ok=True)
        if history.exists():
            try:
                readline.read_history_file(str(history))
            except OSError:
                pass
        return readline, history, old_completer, old_delims
    except OSError:
        return readline, None, old_completer, old_delims


def _restore_readline(
    readline: object | None,
    history: object | None,
    old_completer: object | None,
    old_delims: str | None,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    if readline is None:
        return
    try:
        module = readline  # type: ignore[assignment]
        if history is not None:
            try:
                # Keep only slash commands in the durable history.  Natural
                # language goals may contain private project details, and
                # command arguments can point at sensitive local resources.
                entries = [
                    module.get_history_item(index)
                    for index in range(1, module.get_current_history_length() + 1)
                ]
                safe_entries = [entry for entry in entries if entry and entry.lstrip().startswith("/")]
                module.clear_history()
                for entry in safe_entries[-500:]:
                    module.add_history(entry)
                module.write_history_file(str(history))
                os.chmod(str(history), 0o600)
            except OSError:
                pass
    finally:
        # Keep the user's shell readline configuration intact after exit.
        try:
            module.set_completer(old_completer)
            if old_delims is not None:
                module.set_completer_delims(old_delims)
        except Exception:
            pass


def _dispatch_slash(
    command: str,
    args: list[str],
    dispatch: Dispatch,
    *,
    stdout: TextIO,
    theme: _Theme,
    stdin: TextIO | None = None,
    session: SessionState | None = None,
) -> bool:
    name = command.casefold()
    if name in {"exit", "quit", "q"}:
        stdout.write(theme.muted("\n  已退出 Verdi。\n"))
        return False
    if name in {"help", "h", "?"}:
        render_command_palette(args[0] if args else "", stdout=stdout, color=theme.enabled)
        return True
    if name == "clear":
        _clear_screen(stdout, theme)
        return True
    if name == "version":
        stdout.write(f"  verdi {_version()}\n")
        return True
    if name == "recent":
        return _render_recent(stdout=stdout, theme=theme, state=session)
    if name == "resume":
        root = Path.cwd().expanduser().resolve()
        config = _project_snapshot(root)
        campaigns = _recent_campaigns(config, root)
        active = next((item for item in campaigns if item.get("status") in {"queued", "running"}), None)
        if active:
            campaign_id = str(active.get("campaign_id"))
            stdout.write(theme.accent(f"  发现进行中的 campaign：{campaign_id}\n"))
            return _dispatch_slash("progress", [campaign_id], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
        return _dispatch_slash("run", [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
    if name == "progress":
        campaign_id = args[0] if args else (session.last_campaign_id if session else None)
        if campaign_id:
            args = [campaign_id]
        else:
            return _dispatch_slash("recent", [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
        return _dispatch_command(["status", *args], dispatch, stdout=stdout, theme=theme, label="/progress", session=session)
    if name == "cancel":
        campaign_id = args[0] if args else None
        if not campaign_id:
            stdout.write(theme.warn("  请明确指定 campaign ID，例如 /cancel CAMPAIGN_ID；系统不会猜测任务。\n"))
            return True
        return _dispatch_command(["cancel", campaign_id], dispatch, stdout=stdout, theme=theme, label="/cancel", session=session)
    if name in {"start", "go", "next"}:
        root = Path.cwd().expanduser().resolve()
        config = _project_snapshot(root)
        if not config:
            if stdin is None:
                stdout.write(theme.warn("  还没有项目配置；输入 /setup 开始接入。\n"))
                return True
            return _guided_setup(stdin, stdout, dispatch, theme)
        if not config.get("goal"):
            stdout.write(theme.warn("  还没有研究目标；输入 /plan \"你的目标\"。\n"))
            return True
        readiness = _readiness_snapshot(config, root)
        blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
        if isinstance(blockers, list) and blockers:
            stdout.write(theme.bad("  当前接入仍有阻塞项；先运行 /check 查看可执行的修复步骤。\n"))
            return True
        plan_path = _default_plan_path(root)
        if plan_path.is_file():
            stdout.write(theme.accent(f"  已发现研究计划：{plan_path}\n"))
            return _dispatch_slash("run", [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
        stdout.write("  当前项目已配置，下一步生成研究计划：/plan\n")
        return _dispatch_slash(
            "plan",
            ["--goal", str(config["goal"])],
            dispatch,
            stdout=stdout,
            theme=theme,
            stdin=stdin,
            session=session,
        )
    if name == "doctor":
        return _dispatch_command(["doctor", *args], dispatch, stdout=stdout, theme=theme, label="/doctor", session=session)
    if name == "models":
        config = _project_snapshot()
        if not config:
            stdout.write(theme.warn("  尚未发现 verdiwm.toml；输入 /setup 开始接入。\n"))
            return True
        stdout.write("  当前项目绑定：\n")
        for key in ("model", "source", "data", "goal", "target_metrics", "budget", "mode"):
            value = config.get(key)
            if value is not None:
                stdout.write(f"    {key:<15} {value}\n")
        return True
    if name in {"guide", "guide-model"}:
        argv = ["guide-model", *args]
        return _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label="/guide", session=session)
    if name in {"evaluator", "eval"}:
        if not args:
            config = _project_snapshot()
            source = config.get("source") if config else None
            if not source:
                stdout.write(theme.warn("  尚未绑定源码目录；输入 /setup 或显式指定 --source-root PATH。\n"))
                return True
            args = ["discover", "--source-root", str(source)]
        return _dispatch_command(["evaluator", *args], dispatch, stdout=stdout, theme=theme, label="/evaluator", session=session)
    if name == "setup" and not args:
        config = _project_snapshot()
        if config:
            stdout.write(theme.warn("  当前目录已经有 verdiwm.toml；输入 /models 查看，或使用 /setup --help。\n"))
            return True
        if stdin is None:
            stdout.write(theme.warn("  输入 /setup 后会启动首次接入向导。\n"))
            return True
        return _guided_setup(stdin, stdout, dispatch, theme)
    if name == "research" and not args:
        return _dispatch_slash("start", [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
    mapping = {
        "check": ["check", *args],
        "status": ["status", *args],
        "diagnose": ["diagnose", *args],
        "setup": ["setup", *args],
        "init": ["init", *args],
        "plan": ["research", "plan", *args],
        "run": ["research", "run", *args],
        "research": ["research", *args],
    }
    argv = mapping.get(name)
    if argv is None:
        matches = tuple(_matching_commands(name))
        if matches:
            stdout.write(theme.warn(f"  未知命令 /{command}；你可能想输入：\n"))
            render_command_palette(name, stdout=stdout, color=theme.enabled)
        else:
            stdout.write(theme.warn(f"  未知命令 /{command}。输入 / 查看可用命令。\n"))
        return True
    if name == "run" and not args:
        plan_path = _default_plan_path(Path.cwd())
        payload, error = _plan_summary(plan_path)
        if payload is None:
            stdout.write(theme.warn(f"  {error or '还没有默认研究计划；输入 /plan 先生成。'}\n"))
            return True
        _render_plan_preview(payload, plan_path, stdout=stdout, theme=theme)
        if payload.get("state") not in {"ready", "ready_with_deferred_discovery"}:
            stdout.write(theme.bad("  计划当前不可执行；请先解决 blocker，再输入 /plan 重新生成。\n"))
            return True
        if stdin is None or not _prompt_confirmation(stdin, stdout, theme, "确认创建并执行这个 campaign？"):
            stdout.write(theme.muted("  已取消执行；计划仍保留在原路径。\n"))
            return True
        args = ["--plan", str(plan_path), "--confirm"]
        argv = ["research", "run", *args]
    if name == "run" and "--confirm" in args:
        stdout.write(theme.warn("  即将执行已确认计划；Verdi 会继续遵守计划和证据门禁。\n"))
    # Friendly shorthand: ``/plan \"goal\"`` and ``/run PLAN`` are expanded
    # into the explicit argparse forms while retaining the same safety gates.
    if name == "plan" and argv[2:] and not argv[2].startswith("-") and "--goal" not in argv[2:]:
        # Accept both ``/plan \"goal\"`` and ``/plan \"goal\" --budget 4gpu-hours``.
        first_option = next(
            (index for index, token in enumerate(argv[2:], start=2) if token.startswith("-")),
            len(argv),
        )
        argv = ["research", "plan", "--goal", " ".join(argv[2:first_option]), *argv[first_option:]]
    elif name == "run" and len(argv) > 2 and not argv[2].startswith("-"):
        argv = ["research", "run", "--plan", argv[2], *argv[3:]]
    return _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label=f"/{name}", session=session)


def _render_result(raw: str, *, stdout: TextIO, theme: _Theme) -> None:
    """Turn machine JSON into a compact human-facing result card."""

    text = raw.strip()
    if not text:
        return
    payload: object | None = None
    try:
        payload = json.loads(text.splitlines()[-1])
    except (json.JSONDecodeError, TypeError):
        stdout.write(text + ("\n" if not text.endswith("\n") else ""))
        return
    if not isinstance(payload, dict):
        stdout.write(text + "\n")
        return
    state = payload.get("state")
    if state is not None:
        state_text = str(state)
        painter = theme.good if state_text in {"ready", "queued", "completed", "verified"} else theme.warn if state_text in {"blocked", "needs_input", "awaiting_confirmation", "running"} else theme.bad
        stdout.write(f"  {theme.muted('State')}  {painter(state_text)}\n")
    for key, label in (("plan_path", "Plan"), ("project_file", "Project"), ("campaign_id", "Campaign"), ("job_id", "Job"), ("output", "Output")):
        if payload.get(key) is not None:
            stdout.write(f"  {label:<8} {payload[key]}\n")
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        stdout.write(theme.warn(f"  Blockers ({len(blockers)})\n"))
        for blocker in blockers[:5]:
            if isinstance(blocker, dict):
                stdout.write(f"    - {blocker.get('code', 'BLOCKED')}: {blocker.get('message', blocker.get('detail', ''))}\n")
    items = payload.get("items")
    if isinstance(items, list):
        stdout.write(f"  Campaigns {len(items)}\n")
        for item in items[:8]:
            if isinstance(item, dict):
                stdout.write(f"    {item.get('campaign_id', '?')}  {item.get('status', item.get('state', '?'))}\n")
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        stdout.write(f"  Evaluators {len(candidates)} candidates (none frozen)\n")
        for candidate in candidates[:8]:
            if not isinstance(candidate, dict):
                continue
            horizon = candidate.get("horizon_seconds")
            horizon_text = f"{horizon}s" if horizon is not None else "horizon?"
            marker = "minute-ready" if candidate.get("minute_level_supported") else "short/unknown"
            stdout.write(
                f"    {candidate.get('candidate_id', '?')}  {horizon_text}  {marker}  {candidate.get('relative_path', '')}\n"
            )
        hint = payload.get("selection_hint")
        if isinstance(hint, dict) and hint.get("candidate_id"):
            stdout.write(theme.accent(f"  Hint     review {hint['candidate_id']} first; confirmation is still required\n"))
    stages = payload.get("stages")
    if isinstance(stages, list):
        stdout.write(f"  Stages    {len(stages)}\n")
    if state is None and not any(key in payload for key in ("plan_path", "project_file", "campaign_id", "items", "blockers")):
        stdout.write(text + "\n")


def _dispatch_command(
    argv: list[str],
    dispatch: Dispatch,
    *,
    stdout: TextIO,
    theme: _Theme,
    label: str,
    session: SessionState | None = None,
) -> bool:
    """Run one existing CLI command and keep the shell responsive on errors."""

    stdout.write(theme.muted(f"  执行 {label} …\n"))
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    started = time.monotonic()
    heartbeat_stop = threading.Event()
    heartbeat_visible = threading.Event()

    def heartbeat() -> None:
        # Keep long-running commands visibly alive without touching their
        # stdout capture.  The worker writes only to the caller's terminal.
        while not heartbeat_stop.wait(1.0):
            elapsed = time.monotonic() - started
            try:
                heartbeat_visible.set()
                stdout.write(theme.muted(f"\r  {label} 进行中 … {elapsed:.0f}s"))
                stdout.flush()
            except (OSError, ValueError):
                return

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            result = int(dispatch(argv))
    except SystemExit as exc:
        result = int(exc.code or 0)
    except KeyboardInterrupt:
        stdout.write(theme.warn("  已中断当前命令，交互会话仍保持打开。\n"))
        return True
    except Exception as exc:  # keep one bad command from tearing down the shell
        result = 1
        captured_err.write(f"{type(exc).__name__}: {exc}")
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=0.2)
        # Remove a possible in-place heartbeat before rendering the result.
        if heartbeat_visible.is_set():
            stdout.write("\r\033[2K" if theme.enabled else "\r" + (" " * 48) + "\r")
            stdout.flush()
    raw_output = captured_out.getvalue()
    _render_result(raw_output, stdout=stdout, theme=theme)
    if session is not None:
        session.last_command = label
        session.last_error = None
        try:
            payload = json.loads(raw_output.strip().splitlines()[-1]) if raw_output.strip() else None
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            if payload.get("plan_path") is not None:
                session.last_plan = str(payload["plan_path"])
            if payload.get("campaign_id") is not None:
                session.last_campaign_id = str(payload["campaign_id"])
            campaign = payload.get("campaign")
            if isinstance(campaign, dict):
                if campaign.get("campaign_id") is not None:
                    session.last_campaign_id = str(campaign["campaign_id"])
                if campaign.get("status") is not None:
                    session.last_state = str(campaign["status"])
            items = payload.get("items")
            if isinstance(items, list) and items and isinstance(items[0], dict):
                latest = items[0]
                if latest.get("campaign_id") is not None:
                    session.last_campaign_id = str(latest["campaign_id"])
                if latest.get("status") is not None:
                    session.last_state = str(latest["status"])
            if payload.get("state") is not None:
                session.last_state = str(payload["state"])
    error_text = captured_err.getvalue().strip()
    if error_text:
        if session is not None:
            session.last_error = error_text
        stdout.write(theme.bad("  " + error_text.replace("\n", "\n  ") + "\n"))
    elapsed = time.monotonic() - started
    if result == 0:
        stdout.write(theme.good(f"  命令完成（{elapsed:.1f}s）。\n"))
    else:
        stdout.write(theme.bad(f"  命令返回状态 {result}。\n"))
    return True


def run_interactive_session(*, stdin: TextIO | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None, dispatch: Dispatch | None = None) -> int:
    """Run the line-oriented shell.  ``dispatch`` is injectable for tests."""

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    if not interactive_supported(stdin, stdout):
        stdout.write("verdi 需要交互式终端；在脚本或管道中请使用 verdi --help。\n")
        return 0
    if dispatch is None:
        from wmloop.cli import main

        dispatch = lambda argv: main(argv)
    theme = _Theme(_supports_color(stdout))
    render_welcome(stdout=stdout, color=theme.enabled)
    readline, history, old_completer, old_delims = _configure_readline(stdin, stdout)
    use_builtin_input = stdin is sys.stdin and stdout is sys.stdout
    session = SessionState()
    try:
        while True:
            try:
                if use_builtin_input:
                    # ``input`` is what activates Python's optional readline
                    # editor, history, and completion hooks.
                    line = input(theme.accent("verdi> "))
                else:
                    stdout.write(theme.accent("verdi> "))
                    stdout.flush()
                    line = stdin.readline()
            except KeyboardInterrupt:
                stdout.write("\n" + theme.warn("  ^C 已清除当前输入。\n"))
                continue
            except EOFError:
                stdout.write("\n" + theme.muted("  已退出 Verdi。\n"))
                return 0
            if line == "":
                stdout.write("\n" + theme.muted("  已退出 Verdi。\n"))
                return 0
            text = line.strip()
            if not text:
                continue
            if text == "/":
                selected = _live_palette_selection(stdin=stdin, stdout=stdout, theme=theme)
                if selected:
                    _dispatch_slash(selected, [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session)
                elif stdin is not sys.stdin or stdout is not sys.stdout or not theme.enabled:
                    render_command_palette(stdout=stdout, color=theme.enabled)
                continue
            if text.startswith("/"):
                try:
                    tokens = shlex.split(text[1:])
                except ValueError as exc:
                    stdout.write(theme.bad(f"  命令解析失败：{exc}\n"))
                    continue
                if not tokens:
                    render_command_palette(stdout=stdout, color=theme.enabled)
                    continue
                if tokens[0].casefold() == "help" and len(tokens) == 2:
                    # ``/help plan`` is a concise contextual help view.
                    command = next((item for item in COMMANDS if item.name == tokens[1].casefold()), None)
                    if command is not None:
                        _render_command_help(command, stdout=stdout, theme=theme)
                        continue
                if tokens[0].casefold() in {"plan", "research"} and len(tokens) == 1 and session.last_goal:
                    tokens.extend(["--goal", session.last_goal])
                if not _dispatch_slash(tokens[0], tokens[1:], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session):
                    return 0
                continue
            # Natural-language input is intentionally advisory.  It gives the
            # user the next safe command without silently allocating a GPU.
            goal = text
            session.last_goal = goal
            quoted = shlex.join([goal])
            stdout.write(theme.accent("  已收到研究目标：") + goal + "\n")
            stdout.write("  " + theme.muted(f"下一步可输入 /plan --goal {quoted}" ) + "\n")
            stdout.write("  " + theme.muted("计划会先生成并等待你的审阅；输入 /run --plan PATH --confirm 才会执行。\n"))
    finally:
        _restore_readline(readline, history, old_completer, old_delims, stdin, stdout)
