from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

from wmloop.control import interactive_cli


class FakeTerminal(StringIO):
    def isatty(self) -> bool:
        return True


class FakePipe(StringIO):
    def isatty(self) -> bool:
        return False


def test_palette_for_slash_lists_commands() -> None:
    output = FakeTerminal()
    interactive_cli.render_command_palette(stdout=output, color=False)
    text = output.getvalue()
    assert "/research" in text
    assert "/status" in text
    assert "/exit" in text


def test_palette_query_filters_commands() -> None:
    output = FakeTerminal()
    interactive_cli.render_command_palette("/res", stdout=output, color=False)
    text = output.getvalue()
    assert "/research" in text
    assert "/status" not in text


def test_contextual_help_describes_one_command() -> None:
    stdin = FakeTerminal("/help plan\n/exit\n")
    stdout = FakeTerminal()
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: 0)
    text = stdout.getvalue()
    assert "/plan" in text
    assert "不会启动 GPU" in text


def test_session_dispatches_shortcuts_and_exits() -> None:
    stdin = FakeTerminal("/check\n/status demo\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    assert interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=dispatch
    ) == 0
    assert calls == [["check"], ["status", "demo"]]
    assert "已退出 Verdi" in stdout.getvalue()


def test_slash_alone_opens_palette_inside_session() -> None:
    stdin = FakeTerminal("/\n/exit\n")
    stdout = FakeTerminal()
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: 0)
    text = stdout.getvalue()
    assert "Command palette" in text
    assert "/research" in text


def test_plan_shortcut_preserves_quoted_goal() -> None:
    stdin = FakeTerminal('/plan --goal "improve long horizon"\n/exit\n')
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["research", "plan", "--goal", "improve long horizon"]]


def test_plan_shortcut_allows_options_after_goal() -> None:
    stdin = FakeTerminal('/plan "improve consistency" --budget 2gpu-hours\n/exit\n')
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["research", "plan", "--goal", "improve consistency", "--budget", "2gpu-hours"]]


def test_plan_without_arguments_reuses_last_natural_language_goal() -> None:
    stdin = FakeTerminal("improve consistency\n/plan\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["research", "plan", "--goal", "improve consistency"]]


def test_unknown_slash_command_does_not_dispatch() -> None:
    stdin = FakeTerminal("/does-not-exist\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == []
    assert "未知命令" in stdout.getvalue()


def test_unknown_prefix_suggests_matching_command() -> None:
    stdin = FakeTerminal("/sta\n/exit\n")
    stdout = FakeTerminal()
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: 0)
    assert "你可能想输入" in stdout.getvalue()
    assert "/status" in stdout.getvalue()


def test_command_result_is_rendered_as_readable_card() -> None:
    stdin = FakeTerminal("/check\n/exit\n")
    stdout = FakeTerminal()

    def dispatch(argv: list[str]) -> int:
        print(json.dumps({"state": "blocked", "blockers": [{"code": "MODEL_REQUIRED", "message": "需要模型"}]}))
        return 2

    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=dispatch)
    text = stdout.getvalue()
    assert "State" in text
    assert "MODEL_REQUIRED" in text
    assert "命令返回状态 2" in text


def test_dispatch_exception_is_reported_without_closing_shell() -> None:
    stdin = FakeTerminal("/check\n/exit\n")
    stdout = FakeTerminal()

    def dispatch(argv: list[str]) -> int:
        raise RuntimeError("temporary failure")

    assert interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=dispatch) == 0
    text = stdout.getvalue()
    assert "RuntimeError: temporary failure" in text
    assert "已退出 Verdi" in text


def test_natural_language_is_advisory_and_does_not_dispatch() -> None:
    stdin = FakeTerminal("提升分钟级长程一致性\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    text = stdout.getvalue()
    assert calls == []
    assert "/plan --goal" in text


def test_non_tty_does_not_block_or_render_session() -> None:
    stdout = FakePipe()
    assert interactive_cli.run_interactive_session(
        stdin=FakePipe(), stdout=stdout
    ) == 0
    assert "需要交互式终端" in stdout.getvalue()


def test_no_color_disables_ansi(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    output = FakeTerminal()
    interactive_cli.render_welcome(stdout=output)
    assert "\x1b[" not in output.getvalue()


def test_tty_welcome_uses_ansi(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    output = FakeTerminal()
    interactive_cli.render_welcome(stdout=output)
    assert "\x1b[" in output.getvalue()


def test_eof_exits_cleanly() -> None:
    stdout = FakeTerminal()
    assert interactive_cli.run_interactive_session(
        stdin=FakeTerminal(""), stdout=stdout, dispatch=lambda argv: 0
    ) == 0
    assert "已退出 Verdi" in stdout.getvalue()


class InterruptOnceTerminal(FakeTerminal):
    def __init__(self) -> None:
        super().__init__("/exit\n")
        self.interrupted = False

    def readline(self, *args, **kwargs):
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return super().readline(*args, **kwargs)


def test_ctrl_c_keeps_session_open() -> None:
    stdout = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=InterruptOnceTerminal(), stdout=stdout, dispatch=lambda argv: 0
    )
    assert "已清除当前输入" in stdout.getvalue()
    assert "已退出 Verdi" in stdout.getvalue()


def test_guided_setup_collects_three_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    stdin = FakeTerminal("/start\n\n\n提高长程一致性\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin,
        stdout=stdout,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [[
        "setup",
        "--model",
        str(model),
        "--data",
        str(data),
        "--goal",
        "提高长程一致性",
    ]]
    assert "首次接入 Verdi" in stdout.getvalue()


def test_explicit_chat_parser_is_available(monkeypatch) -> None:
    import wmloop.cli as cli

    called = []
    monkeypatch.setattr(
        interactive_cli,
        "run_interactive_session",
        lambda **kwargs: called.append(True) or 0,
    )
    monkeypatch.setattr(cli.sys, "argv", ["verdi"])
    assert cli.main(["chat"]) == 0
    assert called == [True]


def test_bare_verdi_enters_session_only_on_tty(monkeypatch) -> None:
    import sys
    import wmloop.cli as cli

    stdin = FakeTerminal("/exit\n")
    stdout = FakeTerminal()
    monkeypatch.setattr(sys, "argv", ["/tmp/bin/verdi"])
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    assert cli.main([]) == 0
    assert "VERDI" in stdout.getvalue()
    assert "已退出 Verdi" in stdout.getvalue()


def test_recent_without_project_explains_next_step(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    stdin = FakeTerminal("/recent\n/exit\n")
    stdout = FakeTerminal()
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: 0)
    text = stdout.getvalue()
    assert "尚未生成" in text
    assert "Next     /setup" in text


def test_run_without_arguments_previews_default_plan_and_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "ready", "goal": "improve consistency", "model": "model", "budget": {"gpu_hours": 1}, "mode": "hybrid"}), encoding="utf-8")
    stdin = FakeTerminal("/run\ny\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0)
    assert calls == [["research", "run", "--plan", str(plan), "--confirm"]]
    assert "确认创建并执行" in stdout.getvalue()


def test_run_rejection_does_not_dispatch_or_consume_a_campaign(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "ready", "goal": "improve consistency"}), encoding="utf-8")
    stdin = FakeTerminal("/run\nno\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0)
    assert calls == []
    assert "已取消执行" in stdout.getvalue()


def test_progress_reuses_last_campaign_id_from_status_result() -> None:
    stdin = FakeTerminal("/status\n/progress\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        if argv == ["status"]:
            print(json.dumps({"items": [{"campaign_id": "demo", "status": "running"}]}))
        else:
            print(json.dumps({"campaign_id": "demo", "status": "running"}))
        return 0

    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=dispatch)
    assert calls == [["status"], ["status", "demo"]]


def test_cancel_requires_explicit_campaign_id() -> None:
    stdin = FakeTerminal("/cancel\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0)
    assert calls == []
    assert "明确指定 campaign ID" in stdout.getvalue()


def test_welcome_marks_configured_but_blocked_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "verdiwm.toml").write_text(
        "[project]\nmodel = \"model\"\ndata = \"data\"\ngoal = \"improve consistency\"\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    output = FakeTerminal()
    interactive_cli.render_welcome(stdout=output, project_root=tmp_path, color=False)
    text = output.getvalue()
    assert "BLOCKED" in text
    assert "Next     BLOCKED  /check" in text
