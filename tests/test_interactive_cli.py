from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pytest

from wmloop.control import interactive_cli


class FakeTerminal(StringIO):
    def isatty(self) -> bool:
        return True


class FakePipe(StringIO):
    def isatty(self) -> bool:
        return False


def test_builtin_input_empty_line_does_not_exit_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stdin, stdout = FakeTerminal(), FakeTerminal()
    monkeypatch.setattr(interactive_cli.sys, "stdin", stdin)
    monkeypatch.setattr(interactive_cli.sys, "stdout", stdout)
    monkeypatch.setattr(interactive_cli, "_configure_readline", lambda *args: (None, None, None, None))
    entries = iter(["", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(entries))
    calls = []
    assert interactive_cli.run_interactive_session(dispatch=lambda argv: calls.append(argv) or 0) == 0
    assert calls == [["status"]]


def test_palette_for_slash_lists_commands() -> None:
    output = FakeTerminal()
    interactive_cli.render_command_palette(stdout=output, color=False)
    text = output.getvalue()
    assert "/research" in text
    assert "/status" in text
    assert "/llm" in text
    assert "/exit" in text
    assert "/evaluate-irg" in text


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


def test_llm_command_dispatches_help_and_status() -> None:
    stdin = FakeTerminal("/llm\n/llm status\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["llm"], ["llm", "status"]]


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


def test_irg_alias_dispatches_evaluation_command() -> None:
    stdin = FakeTerminal("/irg --plan plan.json --projections p.jsonl --labels l.json --output out\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["research", "evaluate-irg", "--plan", "plan.json", "--projections", "p.jsonl", "--labels", "l.json", "--output", "out"]]


def test_predictor_alias_dispatches_evaluation_command() -> None:
    stdin = FakeTerminal("/predictor --examples e.json --plan p.json --labels l.json --output out\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0
    )
    assert calls == [["research", "evaluate-predictor", "--examples", "e.json", "--plan", "p.json", "--labels", "l.json", "--output", "out"]]


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


@pytest.mark.parametrize("command", ["plan", "research", "r"])
def test_plan_without_arguments_reuses_last_natural_language_goal(command: str) -> None:
    stdin = FakeTerminal(f"improve consistency\n/{command}\n/exit\n")
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


def test_blocked_default_plan_is_versioned_when_new_plan_is_generated(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "blocked", "goal": "old goal"}), encoding="utf-8")
    stdin = FakeTerminal("/plan new goal\n/exit\n")
    stdout = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0)
    # The command is dispatched to the normal planner; the planner owns
    # versioning after it returns a newly compiled plan.
    assert calls and calls[0][:3] == ["research", "plan", "--goal"]


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
    assert "02  NEXT" in text
    assert "/check" in text


def test_builtin_setup_empty_enter_accepts_discovered_default(monkeypatch) -> None:
    stdin, stdout = FakeTerminal(), FakeTerminal()
    monkeypatch.setattr(interactive_cli.sys, "stdin", stdin)
    monkeypatch.setattr(interactive_cli.sys, "stdout", stdout)
    monkeypatch.setattr("builtins.input", lambda: "")
    assert interactive_cli._prompt_value(stdin, stdout, "模型目录", default="./model") == "./model"


def test_builtin_setup_eof_cancels_even_with_default(monkeypatch) -> None:
    stdin, stdout = FakeTerminal(), FakeTerminal()
    monkeypatch.setattr(interactive_cli.sys, "stdin", stdin)
    monkeypatch.setattr(interactive_cli.sys, "stdout", stdout)

    def end_of_input():
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)
    assert interactive_cli._prompt_value(stdin, stdout, "模型目录", default="./model") is None
    assert interactive_cli._prompt_value(FakeTerminal(""), stdout, "模型目录", default="./model") is None


def test_palette_exit_returns_without_reading_another_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(interactive_cli, "_live_palette_selection", lambda **kwargs: "exit")
    stdin, stdout = FakeTerminal("/\n/version\n"), FakeTerminal()
    calls = []
    assert interactive_cli.run_interactive_session(
        stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0,
    ) == 0
    assert calls == []
    assert stdout.getvalue().count("verdi> ") == 1
    assert "已退出 Verdi" in stdout.getvalue()


def test_readline_prompt_marks_ansi_as_zero_width(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stdin, stdout = FakeTerminal(), FakeTerminal()
    monkeypatch.setattr(interactive_cli.sys, "stdin", stdin)
    monkeypatch.setattr(interactive_cli.sys, "stdout", stdout)
    monkeypatch.setattr(interactive_cli, "_configure_readline", lambda *args: (object(), None, None, None))
    monkeypatch.setattr(interactive_cli, "_restore_readline", lambda *args: None)
    prompts = []

    def read_exit(prompt):
        prompts.append(prompt)
        return "/exit"

    monkeypatch.setattr("builtins.input", read_exit)
    interactive_cli.run_interactive_session(dispatch=lambda argv: 0)
    assert len(prompts) == 1
    assert "\001\033[" in prompts[0]
    assert "m\002" in prompts[0]


def test_default_plan_follows_nearest_parent_configuration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    config = tmp_path / "verdiwm.toml"
    config.write_text('[project]\ngoal = "improve consistency"\n', encoding="utf-8")
    child = tmp_path / "nested" / "working"
    child.mkdir(parents=True)
    assert interactive_cli._default_plan_path(child) == tmp_path / ".verdiwm" / "research-plan.json"


def test_default_plan_follows_explicit_configuration_file(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "custom.toml"
    config.write_text('[project]\ngoal = "improve consistency"\n', encoding="utf-8")
    monkeypatch.setenv("VERDIWM_PROJECT_CONFIG", str(config))
    assert interactive_cli._default_plan_path(tmp_path) == project / ".verdiwm" / "research-plan.json"


@pytest.mark.parametrize("command", ["setup", "models", "start", "resume", "recent", "evaluator", "run"])
def test_invalid_project_is_actionable_without_starting_or_overwriting(monkeypatch, tmp_path: Path, command: str) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "verdiwm.toml"
    malformed = "[project\nmodel = \"model\"\n"
    config.write_text(malformed, encoding="utf-8")
    output = FakeTerminal()
    calls = []
    assert interactive_cli._dispatch_slash(
        command, [], lambda argv: calls.append(argv) or 0,
        stdout=output, theme=interactive_cli._Theme(False), stdin=FakeTerminal(""),
    )
    assert calls == []
    assert config.read_text(encoding="utf-8") == malformed
    text = output.getvalue()
    assert "PROJECT_CONFIG_INVALID" in text
    assert "原因" in text
    assert "/check" in text
    assert "尚未发现" not in text
    assert "首次接入 Verdi" not in text


def test_readiness_failure_stays_blocked_and_retains_cause(monkeypatch, tmp_path: Path) -> None:
    from wmloop.control import first_contact

    def fail_inspection(**kwargs):
        raise OSError("cannot read source metadata")

    monkeypatch.setattr(first_contact, "inspect_project", fail_inspection)
    config = {"goal": "improve consistency", "model": "model", "data": "data"}
    readiness = interactive_cli._readiness_snapshot(config, tmp_path)
    assert readiness["state"] == "blocked"
    assert "OSError: cannot read source metadata" in readiness["blockers"][0]["detail"]
    assert interactive_cli._next_step(config, readiness=readiness) == ("BLOCKED", "/check")


@pytest.mark.parametrize("readiness", [None, {}, {"state": "blocked"}, {"state": "needs_confirmation"}])
def test_uncertain_or_unconfirmed_readiness_cannot_recommend_execution(readiness) -> None:
    label, command = interactive_cli._next_step({"goal": "improve consistency"}, readiness=readiness)
    assert command == "/check"
    assert label != "READY"


def test_recent_reuses_readiness_and_recommends_check_for_blocked_project(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "verdiwm.toml").write_text('[project]\ngoal = "improve consistency"\n', encoding="utf-8")
    monkeypatch.setattr(interactive_cli, "_readiness_snapshot", lambda *args: {"state": "blocked", "blockers": [{"code": "MISSING_RUNTIME"}]})
    output = FakeTerminal()
    interactive_cli._render_recent(stdout=output, theme=interactive_cli._Theme(False), project_root=tmp_path)
    assert "Next     /check" in output.getvalue()
    assert "Next     /plan" not in output.getvalue()
    assert "输入 /plan" not in output.getvalue()


def test_result_retains_repair_actions_readiness_errors_and_all_blockers() -> None:
    payload = {
        "state": "needs_input",
        "project_file": "verdiwm.toml",
        "errors": [{"field": "data", "message": "Data path does not exist"}],
        "readiness": {
            "state": "blocked",
            "blockers": [{"code": f"MISSING_{i}", "message": "missing input", "detail": f"input {i}", "action": f"repair {i}"} for i in range(6)],
            "next_step": "Run /check after repairing inputs",
        },
        "next_steps": ["Inspect /guide", "Select a runtime"],
    }
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    for expected in ("verdiwm.toml", "Data path does not exist", "Readiness", "MISSING_5", "input 5", "repair 5", "Run /check after repairing inputs", "Inspect /guide", "Select a runtime"):
        assert expected in text


def test_real_guide_command_presents_questions_and_output_path(monkeypatch, tmp_path: Path) -> None:
    from wmloop.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    questionnaire = tmp_path / "questions.json"
    output = FakeTerminal()
    assert interactive_cli._dispatch_slash(
        "guide", ["--output", str(questionnaire)], main,
        stdout=output, theme=interactive_cli._Theme(False),
    )
    payload = json.loads(questionnaire.read_text(encoding="utf-8"))
    text = output.getvalue()
    continuous = text.replace("\n", "")
    assert str(questionnaire) in continuous
    for question in payload["questions"]:
        assert question["title"] in continuous
        assert question["prompt"] in continuous
        assert question["example"] in continuous
    assert payload["next_step"] in continuous


def test_llm_result_exposes_configuration_error_and_action() -> None:
    output = FakeTerminal()
    payload = {"state": "invalid", "config_path": "config.toml", "error": "OPENAI_BROKER_ENDPOINT_INVALID", "action": "Use HTTPS for the provider"}
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert "config.toml" in text
    assert "OPENAI_BROKER_ENDPOINT_INVALID" in text
    assert "Use HTTPS for the provider" in text


def test_structured_results_wrap_cjk_and_paths_without_terminal_controls(monkeypatch) -> None:
    from wmloop.control.terminal_ui import display_width

    monkeypatch.setenv("COLUMNS", "50")
    payload = {
        "state": "blocked",
        "plan_path": "project/" + "very-long-directory/" * 8 + "plan.json",
        "blockers": [{
            "code": "NEEDS_INPUT",
            "message": "检查源码目录与模型运行时的依赖关系" * 8,
            "action": "run \033[2J/check\033]0;unexpected-title\a",
        }],
    }
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert "\033" not in text
    assert "unexpected-title" not in text
    assert "run /check" in text
    assert all(display_width(line) < 50 for line in text.splitlines())
    assert payload["plan_path"] in text.replace("\n", "")


def test_result_summarizes_passing_checks_and_keeps_failed_check_detail() -> None:
    payload = {
        "state": "blocked",
        "checks": [
            {"name": "python", "state": "pass", "detail": "verbose successful Python discovery"},
            {"name": "package", "state": "pass", "detail": "verbose successful package discovery"},
            {"name": "model_runtime", "state": "missing", "required": True, "detail": "runtime Python not found", "action": "choose an installed runtime"},
        ],
    }
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert "2/3 通过" in text
    assert "verbose successful" not in text
    assert "runtime Python not found" in text
    assert "choose an installed runtime" in text


def test_setup_result_deduplicates_repair_actions_and_readiness_next_steps() -> None:
    repair = "Select an existing model runtime"
    next_step = "Run /check after fixing the runtime"
    payload = {
        "state": "ready",
        "readiness": {"state": "blocked", "blockers": [{"code": "MISSING_RUNTIME", "action": repair}], "next_step": next_step},
        "next_steps": [repair, next_step, "Read /guide for onboarding questions"],
    }
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert text.count(repair) == 1
    assert text.count(next_step) == 1
    assert "Read /guide for onboarding questions" in text


def test_guide_without_output_explains_how_to_save_questions() -> None:
    output = FakeTerminal()
    payload = {"state": "awaiting_answers", "questions": [{"id": "runtime", "title": "Choose a runtime", "prompt": "Which Python should run the model?"}]}
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    assert "/guide --output .verdiwm/onboarding-questions.json" in output.getvalue().replace("\n", "")


@pytest.mark.parametrize("state_source", ["environment", "project_config"])
def test_recent_and_public_status_share_campaigns_across_working_directories(monkeypatch, tmp_path: Path, capsys, state_source: str) -> None:
    from wmloop.cli import main
    from wmloop.control.commands.inputs import campaign_store

    state_root = tmp_path / "user-state"
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.delenv("VERDIWM_STATE_ROOT", raising=False)
    if state_source == "environment":
        monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state_root))
    else:
        project_config = tmp_path / "verdiwm.toml"
        project_config.write_text('[project]\nstate_root = "user-state"\n', encoding="utf-8")
        monkeypatch.setenv("VERDIWM_PROJECT_CONFIG", str(project_config))
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}', encoding="utf-8")
    store = campaign_store()
    store.create({
        "campaign_id": "terminal-fixture", "goal": "improve score", "model": str(model),
        "dataset": str(data), "budget": 1, "resource_request": {"gpu_indices": []},
        "execution": {"kind": "pipeline", "repo_root": str(model), "output_root": str(tmp_path / "output"),
                      "evaluator_contract": str(evaluator), "budget_total_gpu_hours": 1,
                      "asset_bindings": {"--checkpoint": str(model)}},
    })
    for name in ("first-working-directory", "second-working-directory"):
        unrelated_cwd = tmp_path / name
        unrelated_cwd.mkdir()
        monkeypatch.chdir(unrelated_cwd)
        assert main(["status"]) == 0
        public_records = json.loads(capsys.readouterr().out)["items"]
        assert [(record["campaign_id"], record["status"]) for record in public_records] == [("terminal-fixture", "created")]
        output = FakeTerminal()
        interactive_cli._dispatch_slash("recent", [], main, stdout=output, theme=interactive_cli._Theme(False))
        assert "terminal-fixture  created  improve score" in output.getvalue()
        assert "尚无记录" not in output.getvalue()
    assert (state_root / "campaigns" / "campaigns.sqlite3").is_file()
    assert not (state_root / "campaigns.sqlite3").exists()
    assert not (tmp_path / "output").exists()


def test_recent_does_not_create_a_missing_state_store(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / "missing-state"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state_root))
    output = FakeTerminal()
    interactive_cli._dispatch_slash("recent", [], lambda argv: 0, stdout=output, theme=interactive_cli._Theme(False))
    assert "尚无记录" in output.getvalue()
    assert not state_root.exists()


@pytest.mark.parametrize("failure", ["corrupt_database", "missing_database", "root_is_file", "symlink"])
def test_recent_reports_storage_failures_without_hiding_or_rebuilding_history(monkeypatch, tmp_path: Path, failure: str) -> None:
    state_root = tmp_path / "user-state"
    state_root.mkdir()
    campaigns = state_root / "campaigns"
    if failure == "root_is_file":
        campaigns.write_text("wrong-path-kind", encoding="utf-8")
    elif failure == "symlink":
        original = tmp_path / "original-campaigns"
        original.mkdir()
        campaigns.symlink_to(original, target_is_directory=True)
    else:
        campaigns.mkdir()
        if failure == "corrupt_database":
            (campaigns / "campaigns.sqlite3").write_bytes(b"invalid-sqlite-content")
        else:
            (campaigns / ".sqlite-campaign-store").write_text("existing-store", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state_root))
    output = FakeTerminal()
    session = interactive_cli.SessionState()
    interactive_cli._dispatch_slash("recent", [], lambda argv: 0, stdout=output, theme=interactive_cli._Theme(False), session=session)
    text = output.getvalue()
    expected = "CAMPAIGN_ROOT_INVALID" if failure == "symlink" else "CAMPAIGN_STORE_UNAVAILABLE"
    assert expected in text
    assert session.last_error == expected
    assert "/status" in text
    assert "尚无记录" not in text
    if failure in {"corrupt_database", "root_is_file"}:
        assert "原因" in text
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_recent_reports_an_unreadable_campaign_directory(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / "user-state"
    campaigns = state_root / "campaigns"
    campaigns.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state_root))
    original_iterdir = Path.iterdir

    def denied_iterdir(path: Path):
        if path == campaigns:
            raise PermissionError("test permission denial")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied_iterdir)
    output = FakeTerminal()
    interactive_cli._dispatch_slash("recent", [], lambda argv: 0, stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert "CAMPAIGN_STORE_UNAVAILABLE" in text
    assert "PermissionError: test permission denial" in text
    assert "尚无记录" not in text
    assert not (campaigns / "campaigns.sqlite3").exists()


@pytest.mark.parametrize(("command", "expected"), [
    ('/r plan --goal "improve consistency"', ["research", "plan", "--goal", "improve consistency"]),
    ("/s", ["status"]),
    ("/p demo", ["status", "demo"]),
    ('/configure --model model --data data --goal "improve consistency"', ["setup", "--model", "model", "--data", "data", "--goal", "improve consistency"]),
])
def test_registered_short_aliases_use_the_public_command_handlers(monkeypatch, tmp_path: Path, command: str, expected: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    stdin, stdout = FakeTerminal(command + "\n/exit\n"), FakeTerminal()
    calls = []
    interactive_cli.run_interactive_session(stdin=stdin, stdout=stdout, dispatch=lambda argv: calls.append(argv) or 0)
    assert calls == [expected]
    assert "未知命令" not in stdout.getvalue()


def test_help_accepts_registered_command_alias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    output = FakeTerminal()
    calls = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/h configure\n/exit\n"), stdout=output,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == []
    assert "用法  /setup" in output.getvalue()
    assert "Command palette" not in output.getvalue()


def test_setup_saved_configuration_does_not_claim_experiment_readiness(monkeypatch, tmp_path: Path) -> None:
    from wmloop.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    output = FakeTerminal()
    interactive_cli._dispatch_slash(
        "setup", ["--model", str(model), "--data", str(data), "--goal", "improve consistency"], main,
        stdout=output, theme=interactive_cli._Theme(False),
    )
    assert (tmp_path / "verdiwm.toml").is_file()
    text = output.getvalue()
    assert "配置已保存" in text
    assert "Readiness  needs_input" in text
    assert "State" not in text
    assert "ready" not in text
