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
import time
from typing import Callable, Iterable, TextIO


Dispatch = Callable[[list[str]], int]


@dataclass(frozen=True)
class InteractiveCommand:
    name: str
    aliases: tuple[str, ...]
    summary: str
    usage: str


COMMANDS: tuple[InteractiveCommand, ...] = (
    InteractiveCommand("help", ("h", "?"), "显示命令面板和使用说明", "/help"),
    InteractiveCommand("research", ("r",), "进入研究计划或执行流程", "/research plan ..."),
    InteractiveCommand("start", ("go", "next"), "按当前项目状态继续下一步", "/start"),
    InteractiveCommand("plan", (), "生成可审阅的研究计划（只读）", "/plan \"研究目标\""),
    InteractiveCommand("run", (), "执行已确认的研究计划", "/run --plan PATH --confirm"),
    InteractiveCommand("status", ("s",), "查看 campaign 状态", "/status [CAMPAIGN_ID]"),
    InteractiveCommand("check", (), "检查项目接入和本地 readiness", "/check"),
    InteractiveCommand("doctor", (), "检查 Verdi 本地安装", "/doctor"),
    InteractiveCommand("diagnose", (), "只读诊断模型项目", "/diagnose"),
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


def _project_snapshot() -> dict[str, object] | None:
    try:
        from wmloop.control.project_config import load_project_config

        return dict(load_project_config().values)
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


def _next_step(config: dict[str, object] | None) -> tuple[str, str]:
    if not config:
        return "SETUP", "/setup"
    if not config.get("goal"):
        return "GOAL REQUIRED", "/plan \"你的研究目标\""
    return "READY", "/start"


def render_welcome(*, stdout: TextIO = sys.stdout, project_root: Path | None = None, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    root = (project_root or Path.cwd()).expanduser().resolve()
    config = _project_snapshot()
    model = config.get("model") if config else None
    data = config.get("data", config.get("dataset")) if config else None
    goal = config.get("goal") if config else None
    ready = bool(config and model and data and goal)
    status = theme.good("READY") if ready else theme.warn("NEEDS SETUP")
    next_label, next_command = _next_step(config)
    stdout.write("\n")
    stdout.write(theme.title(f"  VERDI  v{_version()}\n"))
    stdout.write(theme.muted("  Evidence-driven world-model research\n\n"))
    stdout.write(f"  Project  {root}\n")
    stdout.write(f"  Status   {status}\n")
    stdout.write(f"  Model    {model or '未绑定（输入 /setup）'}\n")
    stdout.write(f"  Data     {data or '未绑定（输入 /setup）'}\n")
    if goal:
        stdout.write(f"  Goal     {goal}\n")
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


def _render_command_help(command: InteractiveCommand, *, stdout: TextIO, theme: _Theme) -> None:
    stdout.write(f"\n{theme.title('  /' + command.name)}  {command.summary}\n")
    stdout.write(f"  用法  {command.usage}\n")
    if command.name == "start":
        stdout.write("  说明  自动发现 model/ 和 data/；项目已准备好时直接生成研究计划。\n")
    elif command.name == "plan":
        stdout.write("  说明  只读生成计划；目标可直接作为第一个参数，不会启动 GPU。\n")
    elif command.name == "run":
        stdout.write("  说明  只有带 --confirm 的计划才会进入正式 campaign。\n")
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


def _configure_readline(stdin: TextIO, stdout: TextIO) -> tuple[object | None, object | None, object | None, str | None]:
    if stdin is not sys.stdin or stdout is not sys.stdout:
        return None, None, None, None
    try:
        import readline  # type: ignore
    except ImportError:
        return None, None, None, None

    names = sorted({command.name for command in COMMANDS} | {alias for command in COMMANDS for alias in command.aliases})

    def completer(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        if not line.startswith("/"):
            return None
        prefix = line[1:].split(maxsplit=1)[0] if line[1:] else ""
        candidates = ["/" + name for name in names if name.startswith(prefix)]
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
    if name in {"start", "go", "next"}:
        config = _project_snapshot()
        if not config:
            if stdin is None:
                stdout.write(theme.warn("  还没有项目配置；输入 /setup 开始接入。\n"))
                return True
            return _guided_setup(stdin, stdout, dispatch, theme)
        if not config.get("goal"):
            stdout.write(theme.warn("  还没有研究目标；输入 /plan \"你的目标\"。\n"))
            return True
        stdout.write("  当前项目已配置，下一步生成研究计划：/plan\n")
        return _dispatch_slash(
            "plan",
            ["--goal", str(config["goal"])],
            dispatch,
            stdout=stdout,
            theme=theme,
            stdin=stdin,
        )
    if name == "doctor":
        return _dispatch_command(["doctor", *args], dispatch, stdout=stdout, theme=theme, label="/doctor")
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
        return _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label="/guide")
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
        return _dispatch_slash("start", [], dispatch, stdout=stdout, theme=theme, stdin=stdin)
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
    return _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label=f"/{name}")


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
) -> bool:
    """Run one existing CLI command and keep the shell responsive on errors."""

    stdout.write(theme.muted(f"  执行 {label} …\n"))
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    started = time.monotonic()
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
    _render_result(captured_out.getvalue(), stdout=stdout, theme=theme)
    error_text = captured_err.getvalue().strip()
    if error_text:
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
    last_goal: str | None = None
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
                        _dispatch_slash(selected, [], dispatch, stdout=stdout, theme=theme, stdin=stdin)
                    elif (
                        stdin is not sys.stdin
                        or stdout is not sys.stdout
                        or not theme.enabled
                    ):
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
                if tokens[0].casefold() in {"plan", "research"} and len(tokens) == 1 and last_goal:
                    tokens.extend(["--goal", last_goal])
                if not _dispatch_slash(tokens[0], tokens[1:], dispatch, stdout=stdout, theme=theme, stdin=stdin):
                    return 0
                continue
            # Natural-language input is intentionally advisory.  It gives the
            # user the next safe command without silently allocating a GPU.
            goal = text
            last_goal = goal
            quoted = shlex.join([goal])
            stdout.write(theme.accent("  已收到研究目标：") + goal + "\n")
            stdout.write("  " + theme.muted(f"下一步可输入 /plan --goal {quoted}" ) + "\n")
            stdout.write("  " + theme.muted("计划会先生成并等待你的审阅；输入 /run --plan PATH --confirm 才会执行。\n"))
    finally:
        _restore_readline(readline, history, old_completer, old_delims, stdin, stdout)
