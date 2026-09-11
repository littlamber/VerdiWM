"""Small, dependency-free interactive shell for the public ``verdi`` CLI.

The shell deliberately stays a thin presentation layer over the existing CLI
handlers.  It never starts a campaign from natural-language input; users must
review and confirm a generated research plan before expensive work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import shlex
import sys
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
    InteractiveCommand("plan", (), "生成可审阅的研究计划（只读）", "/plan --goal \"...\""),
    InteractiveCommand("run", (), "执行已确认的研究计划", "/run --plan PATH --confirm"),
    InteractiveCommand("status", ("s",), "查看 campaign 状态", "/status [CAMPAIGN_ID]"),
    InteractiveCommand("check", (), "检查项目接入和本地 readiness", "/check"),
    InteractiveCommand("diagnose", (), "只读诊断模型项目", "/diagnose"),
    InteractiveCommand("setup", (), "首次接入模型、数据和目标", "/setup --model PATH --data PATH --goal \"...\""),
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


def render_welcome(*, stdout: TextIO = sys.stdout, project_root: Path | None = None, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    root = (project_root or Path.cwd()).expanduser().resolve()
    config = _project_snapshot()
    model = config.get("model") if config else None
    data = config.get("data", config.get("dataset")) if config else None
    goal = config.get("goal") if config else None
    ready = bool(config and model and data and goal)
    status = theme.good("READY") if ready else theme.warn("NEEDS SETUP")
    stdout.write("\n")
    stdout.write(theme.title(f"  VERDI  v{_version()}\n"))
    stdout.write(theme.muted("  Evidence-driven world-model research\n\n"))
    stdout.write(f"  Project  {root}\n")
    stdout.write(f"  Status   {status}\n")
    stdout.write(f"  Model    {model or '未绑定（输入 /setup）'}\n")
    stdout.write(f"  Data     {data or '未绑定（输入 /setup）'}\n")
    if goal:
        stdout.write(f"  Goal     {goal}\n")
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


def _clear_screen(stdout: TextIO, theme: _Theme) -> None:
    if theme.enabled:
        stdout.write("\033[2J\033[H")
    render_welcome(stdout=stdout, color=theme.enabled)


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


def _dispatch_slash(command: str, args: list[str], dispatch: Dispatch, *, stdout: TextIO, theme: _Theme) -> bool:
    name = command.casefold()
    if name in {"exit", "quit", "q"}:
        stdout.write(theme.muted("\n  已退出 Verdi。\n"))
        return False
    if name in {"help", "h", "?"}:
        render_command_palette(stdout=stdout, color=theme.enabled)
        return True
    if name == "clear":
        _clear_screen(stdout, theme)
        return True
    if name == "version":
        stdout.write(f"  verdi {_version()}\n")
        return True
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
    if name == "research" and not args:
        stdout.write("  研究闭环：先用 /plan 生成并审阅计划，再用 /run --plan PATH --confirm 执行。\n")
        stdout.write("  示例：/plan --goal \"提升分钟级长程一致性\"\n")
        return True
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
        stdout.write(theme.warn(f"  未知命令 /{command}。输入 / 查看可用命令。\n"))
        return True
    if name == "run" and "--confirm" in args:
        stdout.write(theme.warn("  即将执行已确认计划；Verdi 会继续遵守计划和证据门禁。\n"))
    try:
        result = int(dispatch(argv))
    except SystemExit as exc:
        result = int(exc.code or 0)
    except KeyboardInterrupt:
        stdout.write(theme.warn("  已中断当前命令，交互会话仍保持打开。\n"))
        return True
    if result == 0:
        stdout.write(theme.good("  命令完成。\n"))
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
                if not _dispatch_slash(tokens[0], tokens[1:], dispatch, stdout=stdout, theme=theme):
                    return 0
                continue
            # Natural-language input is intentionally advisory.  It gives the
            # user the next safe command without silently allocating a GPU.
            goal = text
            quoted = shlex.join([goal])
            stdout.write(theme.accent("  已收到研究目标：") + goal + "\n")
            stdout.write("  " + theme.muted(f"下一步可输入 /plan --goal {quoted}" ) + "\n")
            stdout.write("  " + theme.muted("计划会先生成并等待你的审阅；输入 /run --plan PATH --confirm 才会执行。\n"))
    finally:
        _restore_readline(readline, history, old_completer, old_delims, stdin, stdout)
