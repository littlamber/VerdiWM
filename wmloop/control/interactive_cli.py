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
import re
import shlex
import sys
import threading
import time
from typing import Callable, Iterable, TextIO

from wmloop.control.campaign_api import CampaignAPIError
from wmloop.control.interactive_intent import parse_round_intent
from wmloop.control.llm_setup import inspect_llm_config
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_request import matching_terminal_research_result
from wmloop.control.terminal_ui import Dashboard, Theme as _Theme, render_commands, render_dashboard


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
    last_rounds: int | None = None
    last_plan: str | None = None
    last_campaign_id: str | None = None
    last_state: str | None = None
    last_command: str | None = None
    last_error: str | None = None


COMMANDS: tuple[InteractiveCommand, ...] = (
    InteractiveCommand("help", ("h", "?"), "显示命令面板和使用说明", "/help"),
    InteractiveCommand("research", ("r",), "进入研究计划或执行流程", "/research plan ..."),
    InteractiveCommand("start", ("go", "next"), "按当前项目状态继续下一步", "/start"),
    InteractiveCommand("plan", (), "生成可审阅的研究计划（只读）", "/plan \"研究目标\" --rounds 2 [--stop-on-success]"),
    InteractiveCommand("run", (), "选择轮数、预览并确认研究计划", "/run [--rounds 2] [--stop-on-success]"),
    InteractiveCommand("status", ("s",), "查看 campaign 状态", "/status [CAMPAIGN_ID]"),
    InteractiveCommand("recent", (), "查看最近计划、任务和下一步", "/recent"),
    InteractiveCommand("resume", (), "查看活动任务或审阅默认计划", "/resume"),
    InteractiveCommand("progress", ("p",), "查看最近任务的进度", "/progress [CAMPAIGN_ID]"),
    InteractiveCommand("cancel", (), "取消一个排队或运行中的任务", "/cancel CAMPAIGN_ID"),
    InteractiveCommand("check", (), "检查项目接入和本地 readiness", "/check"),
    InteractiveCommand("doctor", (), "检查 Verdi 本地安装", "/doctor"),
    InteractiveCommand("llm", ("api", "provider"), "配置或检查研究 LLM API", "/llm [status]"),
    InteractiveCommand("diagnose", (), "只读诊断模型项目", "/diagnose"),
    InteractiveCommand("evaluate-irg", ("irg",), "比较 IRG 与基线 selector 的外层回放结果", "/evaluate-irg --plan ..."),
    InteractiveCommand("evaluate-predictor", ("predictor",), "回放修复收益预测与支持度判断", "/evaluate-predictor --examples ... --plan ..."),
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
    return dict(load_project_config(cwd=project_root).values)


def _render_project_config_error(error: ProjectConfigError, *, stdout: TextIO, theme: _Theme) -> None:
    """Keep an invalid project actionable without treating it as a new setup."""

    stdout.write(theme.bad(f"  配置错误  {error}\n"))
    if error.__cause__ is not None:
        stdout.write(f"  原因      {error.__cause__}\n")
    stdout.write(theme.accent("  Next      修复 verdiwm.toml 或 VERDIWM_PROJECT_CONFIG 指向的配置，再运行 /check。\n"))


def _render_campaign_store_error(error: CampaignAPIError, *, stdout: TextIO, theme: _Theme) -> None:
    """Preserve storage failures instead of presenting an empty task history."""

    stdout.write(theme.bad(f"  任务库错误  {error}\n"))
    if error.__cause__ is not None:
        stdout.write(f"  原因        {type(error.__cause__).__name__}: {error.__cause__}\n")
    stdout.write(theme.accent("  Next        检查 state_root 或 VERDIWM_STATE_ROOT 指向的原任务目录、权限及数据库；恢复后运行 /status 重查。\n"))


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
    loaded = load_project_config(cwd=root)
    if loaded.source is not None:
        root = loaded.source.parent
    return root / ".verdiwm" / "research-plan.json"


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
            adapter_profile=str(config.get("adapter_profile")) if config.get("adapter_profile") else None,
            adapter=str(config.get("adapter")) if config.get("adapter") else None,
            goal=str(config.get("goal")) if config.get("goal") else None,
        )
    except Exception as exc:
        return {
            "state": "blocked",
            "blockers": [{
                "code": "READINESS_CHECK_FAILED",
                "message": "项目接入检查未能完成。",
                "detail": f"{type(exc).__name__}: {exc}",
                "action": "运行 /check 查看根因；修复后重新检查接入状态。",
            }],
        }


def _recent_campaigns(config: dict[str, object] | None, project_root: Path | None = None, *, limit: int = 5) -> list[dict[str, object]]:
    from wmloop.control.commands.inputs import campaign_store

    records = campaign_store(configured=config or {}, read_only=True).list(limit=1000)
    records.sort(key=lambda item: str(item.get("updated_at", item.get("created_at", ""))), reverse=True)
    return records[:limit]


def _next_step(
    config: dict[str, object] | None,
    *,
    readiness: dict[str, object] | None = None,
    plan_path: Path | None = None,
    campaigns: Iterable[dict[str, object]] = (),
) -> tuple[str, str]:
    campaigns = tuple(campaigns)
    if not config:
        return "SETUP", "/setup"
    if not config.get("goal"):
        return "GOAL REQUIRED", "/plan \"你的研究目标\""
    blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
    if isinstance(blockers, list) and blockers:
        return "BLOCKED", "/check"
    readiness_state = readiness.get("state") if isinstance(readiness, dict) else None
    if readiness_state not in {"ready", "ready_for_conformance", "ready_for_execution"}:
        label = "CONFIRMATION REQUIRED" if readiness_state == "needs_confirmation" else "CHECK REQUIRED"
        return label, "/check"
    for campaign in campaigns:
        if campaign.get("status") in {"queued", "running"}:
            return "IN PROGRESS", "/progress"
    if plan_path is not None and plan_path.is_file():
        payload, _ = _plan_summary(plan_path)
        if payload is not None and payload.get("state") in {"ready", "ready_with_deferred_discovery"}:
            terminal = matching_terminal_research_result(payload, list(campaigns))
            if terminal is not None:
                return "PREVIOUS RUN " + str(terminal["state"]).upper(), "/resume"
            return "PLAN READY", "/run"
        return "PLAN BLOCKED", "/check"
    return "READY", "/start"


def render_welcome(*, stdout: TextIO = sys.stdout, project_root: Path | None = None, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    root = (project_root or Path.cwd()).expanduser().resolve()
    config_error = None
    try:
        config = _project_snapshot(root)
        plan_path = _default_plan_path(root)
    except ProjectConfigError as exc:
        config = None
        plan_path = None
        config_error = str(exc)
        if exc.__cause__ is not None:
            config_error += f" · {exc.__cause__}"
    readiness = _readiness_snapshot(config, root)
    campaigns = []
    campaign_error = None
    if not config_error:
        try:
            campaigns = _recent_campaigns(config, root)
        except CampaignAPIError as exc:
            campaign_error = str(exc)
            if exc.__cause__ is not None:
                campaign_error += f" · {exc.__cause__}"
    if config_error:
        next_label, next_command = "CONFIG BLOCKED", "/check"
    elif campaign_error:
        next_label, next_command = "STATE BLOCKED", "/status"
    else:
        next_label, next_command = _next_step(
            config, readiness=readiness, plan_path=plan_path, campaigns=campaigns
        )
    render_dashboard(
        Dashboard(
            version=_version(), project_root=root, config=config, readiness=readiness,
            llm=inspect_llm_config(), next_label=next_label, next_command=next_command,
            plan_path=plan_path if plan_path is not None and plan_path.is_file() else None,
            latest=campaigns[0] if campaigns else None, config_error=config_error,
            campaign_error=campaign_error,
        ),
        stdout=stdout, theme=theme,
    )


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


def _canonical_command(name: str) -> str:
    normalized = name.casefold()
    return next(
        (command.name for command in COMMANDS if normalized == command.name or normalized in command.aliases),
        normalized,
    )


def render_command_palette(query: str = "", *, stdout: TextIO = sys.stdout, color: bool | None = None) -> None:
    theme = _Theme(_supports_color(stdout) if color is None else color)
    render_commands(tuple(_matching_commands(query)), query=query, stdout=stdout, theme=theme)


def _live_palette_selection(*, stdin: TextIO, stdout: TextIO, theme: _Theme) -> str | None:
    """Use the terminal picker only for the real, color-capable TTY pair."""

    if stdin is not sys.stdin or stdout is not sys.stdout or not theme.enabled:
        return None
    from wmloop.control.terminal_palette import select_command

    return select_command(lambda query: tuple(_matching_commands(query)), stdin=stdin, stdout=stdout, theme=theme)


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
        use_builtin_input = stdin is sys.stdin and stdout is sys.stdout
        if use_builtin_input:
            value = input()
        else:
            value = stdin.readline()
    except (EOFError, KeyboardInterrupt):
        stdout.write("\n")
        return None
    if value == "" and not use_builtin_input:
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
        stdout.write("  轮数  --rounds N 设置最多研究轮数，默认 1；--budget 是所有轮次共享的总预算。\n")
        stdout.write("  目标  --stop-on-success 仅在冻结验证器正式确认目标收益后提前停止。\n")
    elif command.name == "run":
        stdout.write("  说明  无参数时预览最近计划并选择轮数，回车沿用计划值；确认后执行。\n")
        stdout.write("  轮数  /run --rounds 2 先生成并预览派生计划，再询问一次确认；所有轮次共享总预算。\n")
        stdout.write("  目标  --stop-on-success 仅由冻结验证器正式确认后提前停；筛选通过或诊断重复都不能代替。\n")
        stdout.write("  范围  多轮仍属一个 campaign，统一使用 /status 与 /cancel；独立 daemon 入口也继续兼容。\n")
        stdout.write("  停止  另一个终端可用 /cancel CAMPAIGN_ID 请求取消，再用 /status 核对；详见 docs/RUN_LIFECYCLE.md。\n")
    elif command.name == "cancel":
        stdout.write("  说明  取消请求写入持久状态；用 /status 核对执行器是否已退出。独立 daemon 需向控制器发送停止信号。\n")
    elif command.name == "resume":
        stdout.write("  说明  有活动 campaign 时显示进度，否则回到计划确认；不会自动重启失败训练或独立 daemon。\n")
    elif command.name == "llm":
        stdout.write("  说明  研究助手的 API 配置教程；/llm status 只读检查，不会发送网络请求或显示密钥。\n")
    stdout.write("\n")


def _guided_setup(
    stdin: TextIO,
    stdout: TextIO,
    dispatch: Dispatch,
    theme: _Theme,
) -> bool:
    """Collect first-project inputs and delegate validation to public setup."""

    from wmloop.control.first_contact import infer_source_root

    model_default, data_default = _discovered_inputs()
    stdout.write("\n" + theme.title("  首次接入 Verdi") + "\n")
    stdout.write(theme.muted("  填写模型、源码、数据和目标，系统会生成本地 verdiwm.toml；回车接受默认值，Ctrl-C / Ctrl-D 取消。\n"))
    model = _prompt_value(stdin, stdout, "模型目录", default=model_default)
    if model is None:
        stdout.write(theme.warn("  已取消设置。输入 /setup --help 查看完整参数。\n"))
        return True
    inferred_source = infer_source_root(Path(model))
    source_default = str(inferred_source or model)
    source = _prompt_value(stdin, stdout, "源码目录（默认推断或复用模型位置；权重单独存放时填写训练/评测源码）", default=source_default)
    if source is None:
        stdout.write(theme.warn("  已取消设置。\n"))
        return True
    data = _prompt_value(stdin, stdout, "数据目录", default=data_default)
    if data is None:
        stdout.write(theme.warn("  已取消设置。\n"))
        return True
    goal = _prompt_value(stdin, stdout, "研究目标（例如：提升分钟级长程一致性）")
    if goal is None:
        stdout.write(theme.warn("  还需要一句研究目标；设置未执行。\n"))
        return True
    args = ["--model", model, "--data", data, "--goal", goal]
    # Preserve setup's implicit-source behavior for the model-path fallback.
    if inferred_source is not None or source != model:
        args.extend(["--source", source])
    result = _dispatch_slash(
        "setup",
        args,
        dispatch,
        stdout=stdout,
        theme=theme,
    )
    if result:
        stdout.write(theme.muted("  按上面的 Action / Next 继续；输入 /clear 可刷新项目首页。\n"))
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
    from wmloop.control.terminal_results import render_execution_policy

    stdout.write(theme.title("  研究计划") + "\n")
    stdout.write(f"    Goal     {payload.get('goal', '未填写')}\n")
    stdout.write(f"    Model    {payload.get('model', '未绑定')}\n")
    stdout.write(f"    Budget   {payload.get('budget', '未填写')}（所有轮次共享总预算）\n")
    render_execution_policy(payload, stdout=stdout)
    stdout.write(f"    Mode     {payload.get('mode', '未填写')}\n")
    stdout.write(f"    State    {payload.get('state', 'unknown')}\n")
    stdout.write(f"    File     {path}\n")


def _review_run(
    args: list[str], dispatch: Dispatch, *, stdin: TextIO | None,
    stdout: TextIO, theme: _Theme, session: SessionState | None,
) -> bool:
    """Review the exact persisted policy before the terminal grants execution."""

    plan_path = Path(session.last_plan) if session and session.last_plan else _default_plan_path()
    confirmed = False
    if not args:
        payload, error = _plan_summary(plan_path)
        if payload is None:
            stdout.write(theme.warn(f"  {error or '还没有默认研究计划；输入 /plan 先生成。'}\n"))
            return True
        _render_plan_preview(payload, plan_path, stdout=stdout, theme=theme)
        if payload.get("state") not in {"ready", "ready_with_deferred_discovery"}:
            stdout.write(theme.bad("  计划当前不可执行；请先解决 blocker，再输入 /plan 重新生成。\n"))
            return True
        policy = payload.get("execution_policy")
        planned_rounds = policy.get("rounds", 1) if isinstance(policy, dict) else 1
        choice = _prompt_value(
            stdin, stdout, "最多运行轮数（回车沿用；y 确认创建并执行当前计划；n 取消）", default=str(planned_rounds),
        ) if stdin is not None else None
        if choice is None or choice.casefold() in {"n", "no"}:
            stdout.write(theme.muted("  已取消执行；计划仍保留在原路径。\n"))
            return True
        confirmed = choice.casefold() in {"y", "yes"}
        if not confirmed:
            if not re.fullmatch(r"[1-9]\d*", choice):
                stdout.write(theme.warn("  轮数必须是正整数；未创建 campaign。输入 /run 可重新选择。\n"))
                return True
            if int(choice) != planned_rounds:
                args = ["--rounds", choice]
    if args:
        if not any(item == "--plan" or item.startswith("--plan=") for item in args):
            args = ["--plan", str(plan_path), *args]
        preview_state = SessionState()
        _dispatch_command(
            ["research", "run", *args], dispatch,
            stdout=stdout, theme=theme, label="/run 预览", session=preview_state,
        )
        if preview_state.last_state != "awaiting_confirmation" or not preview_state.last_plan:
            return True
        plan_path = Path(preview_state.last_plan)
        payload, error = _plan_summary(plan_path)
        if payload is None:
            stdout.write(theme.bad(f"  {error or '预览未返回可读取的计划；未创建 campaign。'}\n"))
            return True
        _render_plan_preview(payload, plan_path, stdout=stdout, theme=theme)
        if payload.get("state") not in {"ready", "ready_with_deferred_discovery"}:
            stdout.write(theme.bad("  计划当前不可执行；请先解决 blocker，再重新生成计划。\n"))
            return True
        if session is not None:
            session.last_plan = str(plan_path)
            session.last_goal = preview_state.last_goal or session.last_goal
            session.last_rounds = preview_state.last_rounds or session.last_rounds
    if not confirmed and (stdin is None or not _prompt_confirmation(stdin, stdout, theme, "确认创建并执行这个 campaign？")):
        stdout.write(theme.muted("  已取消执行；审阅过的计划仍保留，可用 /run 继续。\n"))
        return True
    # Keep execution options, but send the reviewed plan as the sole policy
    # authority.  Backend argparse has already validated the preview arguments.
    retained: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--plan", "--rounds"}:
            index += 2
            continue
        if token.startswith(("--plan=", "--rounds=")) or token in {"--stop-on-success", "--no-stop-on-success"}:
            index += 1
            continue
        retained.append(token)
        index += 1
    return _dispatch_command(
        ["research", "run", "--plan", str(plan_path), *retained, "--confirm"], dispatch,
        stdout=stdout, theme=theme, label="/run", session=session,
    )


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
        stdout.write(theme.muted("  Plan     尚未生成。\n"))
    if campaigns:
        stdout.write("  Campaigns\n")
        for item in campaigns:
            stdout.write(f"    {item.get('campaign_id', '?')}  {item.get('status', '?')}  {item.get('goal', '')}\n")
    else:
        stdout.write(theme.muted("  Campaigns 尚无记录。\n"))
    _, command = _next_step(
        config,
        readiness=_readiness_snapshot(config, root),
        plan_path=plan_path,
        campaigns=campaigns,
    )
    stdout.write(theme.accent(f"  Next     {command}\n"))
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
        command = _canonical_command(tokens[0]) if tokens else ""
        if command in {"run", "resume"} and len(tokens) <= 1:
            plan = _default_plan_path()
            return [str(plan)] if plan.is_file() else []
        if command in {"status", "progress", "cancel"} and len(tokens) <= 1:
            campaigns = _recent_campaigns(_project_snapshot(), Path.cwd(), limit=20)
            return [str(item.get("campaign_id")) for item in campaigns if item.get("campaign_id")]
        if len(tokens) <= 1:
            return ["/" + name for name in names if ("/" + name).startswith(text)]
        return []

    def completer(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        try:
            candidates = completion_candidates(line, text)
        except (ProjectConfigError, CampaignAPIError):
            # The command boundary presents this error. Tab completion must
            # neither hide it as a new project nor interrupt line editing.
            return None
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
    try:
        return _dispatch_slash_command(
            _canonical_command(command), args, dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session,
        )
    except ProjectConfigError as exc:
        _render_project_config_error(exc, stdout=stdout, theme=theme)
        if session is not None:
            session.last_error = str(exc)
        return True
    except CampaignAPIError as exc:
        _render_campaign_store_error(exc, stdout=stdout, theme=theme)
        if session is not None:
            session.last_error = str(exc)
        return True


def _dispatch_slash_command(
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
    if name == "exit":
        stdout.write(theme.muted("\n  已退出 Verdi。\n"))
        return False
    if name == "help":
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
        payload, _ = _plan_summary(_default_plan_path(root))
        terminal = matching_terminal_research_result(payload, campaigns) if payload is not None else None
        if terminal is not None:
            _render_result(json.dumps(terminal, ensure_ascii=False), stdout=stdout, theme=theme)
            if session is not None:
                session.last_campaign_id = str(terminal["campaign"]["campaign_id"])
                session.last_state = str(terminal["state"])
            return True
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
    if name == "start":
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
        if _next_step(config, readiness=readiness)[1] == "/check":
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
    if name == "llm":
        action = args[0] if args and args[0].casefold() == "status" else None
        argv = ["llm"] + (["status"] if action else [])
        remaining = args[1:] if action else args
        argv.extend(remaining)
        return _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label="/llm", session=session)
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
    if name == "evaluator":
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
        "evaluate-irg": ["research", "evaluate-irg", *args],
        "evaluate-predictor": ["research", "evaluate-predictor", *args],
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
    if name == "plan" and session is not None:
        if "--goal" not in argv and not any(item.startswith("--goal=") for item in argv) and session.last_goal:
            argv.extend(["--goal", session.last_goal])
        if "--rounds" not in argv and not any(item.startswith("--rounds=") for item in argv) and session.last_rounds is not None:
            argv.extend(["--rounds", str(session.last_rounds)])
    if name == "run":
        if "--confirm" not in argv and not {"--help", "-h"}.intersection(argv):
            return _review_run(argv[2:], dispatch, stdin=stdin, stdout=stdout, theme=theme, session=session)
        if "--confirm" in argv:
            if not any(item == "--plan" or item.startswith("--plan=") for item in argv):
                plan_path = session.last_plan if session and session.last_plan else str(_default_plan_path())
                argv.extend(["--plan", plan_path])
            stdout.write(theme.warn("  即将执行已确认计划；Verdi 会继续遵守计划和证据门禁。\n"))
    result = _dispatch_command(argv, dispatch, stdout=stdout, theme=theme, label=f"/{name}", session=session)
    if name == "plan" and session is not None and session.last_state in {"ready", "ready_with_deferred_discovery"}:
        stdout.write(theme.accent("  下一步输入 /run 选择轮数并确认执行，无需复制计划路径。\n"))
    return result


def _render_result(raw: str, *, stdout: TextIO, theme: _Theme) -> None:
    from wmloop.control.terminal_results import render_result

    render_result(raw, stdout=stdout, theme=theme)


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
            document = payload.get("plan", payload)
            if isinstance(document, dict):
                if isinstance(document.get("goal"), str):
                    session.last_goal = document["goal"]
                policy = document.get("execution_policy")
                if isinstance(policy, dict) and type(policy.get("rounds")) is int:
                    session.last_rounds = policy["rounds"]
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


def _prepare_round_request(text: str, *, dispatch: Dispatch, stdout: TextIO, theme: _Theme, session: SessionState) -> bool:
    """Convert explicit natural-language round requests into read-only plans."""

    try:
        intent = parse_round_intent(text)
    except ValueError as exc:
        stdout.write(theme.warn(f"  {exc}\n"))
        return True
    if intent is None:
        return False
    try:
        plan_path = Path(session.last_plan) if session.last_plan else _default_plan_path()
        existing, error = _plan_summary(plan_path)
        if error:
            stdout.write(theme.warn(f"  {error}\n"))
            return True
        options: list[str] = []
        if intent.rounds is not None:
            session.last_rounds = intent.rounds
            options.extend(["--rounds", str(intent.rounds)])
            stdout.write(theme.accent(f"  最多运行 {intent.rounds} 轮；先准备计划，尚未启动 GPU。\n"))
        if intent.stop_on_success is True:
            options.append("--stop-on-success")
            stdout.write(theme.accent("  冻结验证器正式确认目标收益后提前停；先准备计划，尚未启动 GPU。\n"))
        if intent.goal is None and existing is not None and (session.last_goal is None or session.last_goal == existing.get("goal")):
            stdout.write(f"  沿用研究目标：{existing.get('goal', '未填写')}\n")
            result = _dispatch_command(
                ["research", "run", "--plan", str(plan_path), *options], dispatch,
                stdout=stdout, theme=theme, label="/plan 轮数", session=session,
            )
            if session.last_state == "awaiting_confirmation":
                stdout.write(theme.accent("  下一步输入 /run 审阅并确认，无需复制计划路径。\n"))
            return result
        goal = intent.goal or session.last_goal
        if goal is None:
            config = _project_snapshot()
            goal = str(config["goal"]) if config and config.get("goal") else None
        if goal is None:
            stdout.write(theme.warn(f'  还需要研究目标；输入 /plan "你的目标" {shlex.join(options)}。\n'))
            return True
        session.last_goal = goal
        return _dispatch_slash(
            "plan", ["--goal", goal, *options], dispatch,
            stdout=stdout, theme=theme, session=session,
        )
    except ProjectConfigError as exc:
        _render_project_config_error(exc, stdout=stdout, theme=theme)
        session.last_error = str(exc)
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
                    prompt = theme.accent("verdi> ")
                    if readline is not None:
                        prompt = re.sub(r"\x1b\[[0-9;]*m", lambda match: "\001" + match.group() + "\002", prompt)
                    line = input(prompt)
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
            if line == "" and not use_builtin_input:
                stdout.write("\n" + theme.muted("  已退出 Verdi。\n"))
                return 0
            text = line.strip()
            if not text:
                continue
            if text == "/":
                selected = _live_palette_selection(stdin=stdin, stdout=stdout, theme=theme)
                if selected:
                    if not _dispatch_slash(selected, [], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session):
                        return 0
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
                tokens[0] = _canonical_command(tokens[0])
                if tokens[0] == "help" and len(tokens) == 2:
                    # ``/help plan`` is a concise contextual help view.
                    command = next((item for item in COMMANDS if item.name == _canonical_command(tokens[1])), None)
                    if command is not None:
                        _render_command_help(command, stdout=stdout, theme=theme)
                        continue
                if tokens[0] in {"plan", "research"} and len(tokens) == 1 and session.last_goal:
                    tokens = ["plan", "--goal", session.last_goal]
                if not _dispatch_slash(tokens[0], tokens[1:], dispatch, stdout=stdout, theme=theme, stdin=stdin, session=session):
                    return 0
                continue
            if _prepare_round_request(text, dispatch=dispatch, stdout=stdout, theme=theme, session=session):
                continue
            # Other natural-language input stays advisory; round requests
            # above can prepare a plan but never grant GPU execution.
            goal = text
            session.last_goal = goal
            quoted = shlex.join([goal])
            stdout.write(theme.accent("  已收到研究目标：") + goal + "\n")
            stdout.write("  " + theme.muted(f"下一步可输入 /plan --goal {quoted}" ) + "\n")
            stdout.write("  " + theme.muted("计划会先生成并等待你的审阅；输入 /run --plan PATH --confirm 才会执行。\n"))
    finally:
        _restore_readline(readline, history, old_completer, old_delims, stdin, stdout)
