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


@pytest.mark.parametrize(("utterance", "rounds", "goal"), [
    ("提升长程一致性，跑两轮", "2", "提升长程一致性"),
    ("运行 two rounds to improve consistency", "2", "improve consistency"),
    ("improve consistency, run 3 rounds", "3", "improve consistency"),
])
def test_explicit_natural_round_request_only_prepares_plan(monkeypatch, tmp_path: Path, utterance: str, rounds: str, goal: str) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal(utterance + "\n/exit\n"), stdout=output,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", goal, "--rounds", rounds]]
    assert "尚未启动 GPU" in output.getvalue()


@pytest.mark.parametrize(("utterance", "rounds"), [("我想跑两轮完整研究", "2"), ("最多跑3轮", "3"), ("run two rounds", "2"), ("只跑两轮", "2"), ("那就跑两轮吧", "2"), ("跑两轮就停止", "2")])
def test_round_only_request_preserves_last_research_goal(monkeypatch, tmp_path: Path, utterance: str, rounds: str) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("improve consistency\n" + utterance + "\n/exit\n"), stdout=FakeTerminal(),
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "improve consistency", "--rounds", rounds]]


@pytest.mark.parametrize("utterance", ["跑两轮还是三轮", "跑0轮", "跑-2轮", "跑2.5轮", "run several rounds", "不要跑两轮", "跑两轮，不要达标就停", "run two rounds, do not stop on success"])
def test_ambiguous_or_invalid_rounds_do_not_replace_goal_or_dispatch(monkeypatch, tmp_path: Path, utterance: str) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("improve consistency\n" + utterance + "\n/plan\n/exit\n"), stdout=FakeTerminal(),
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "improve consistency"]]


def test_round_request_without_goal_waits_for_goal_and_retains_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal('跑两轮\n/plan "improve consistency"\n/exit\n'), stdout=output,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "improve consistency", "--rounds", "2"]]
    assert "还需要研究目标" in output.getvalue()


def test_round_only_request_derives_existing_plan_without_execution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    original = {"state": "ready", "goal": "preserve custom evaluator goal", "budget": {"gpu_hours": 3}}
    plan.write_text(json.dumps(original), encoding="utf-8")
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("我想跑两轮完整研究\n/exit\n"), stdout=FakeTerminal(),
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "run", "--plan", str(plan), "--rounds", "2"]]
    assert json.loads(plan.read_text(encoding="utf-8")) == original


def test_round_mention_in_objective_is_not_an_execution_request(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("提升两轮交互的稳定性\n/plan\n/exit\n"), stdout=FakeTerminal(),
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "提升两轮交互的稳定性"]]


@pytest.mark.parametrize(("utterance", "options"), [
    ("跑两轮，达到目标就停", ["--rounds", "2", "--stop-on-success"]),
    ("最多跑3轮，达标就停", ["--rounds", "3", "--stop-on-success"]),
    ("run two rounds and stop on success", ["--rounds", "2", "--stop-on-success"]),
    ("达到目标就停", ["--stop-on-success"]),
])
def test_explicit_success_request_preserves_goal_and_only_prepares_plan(monkeypatch, tmp_path: Path, utterance: str, options: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("improve consistency\n" + utterance + "\n/exit\n"), stdout=output,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "improve consistency", *options]]
    assert "冻结验证器正式确认目标收益" in output.getvalue()


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


@pytest.mark.parametrize("command", ["start", "setup"])
def test_guided_setup_accepts_discovered_path_defaults(monkeypatch, tmp_path: Path, command: str) -> None:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    stdin = FakeTerminal(f"/{command}\n\n\n\n提高长程一致性\n/exit\n")
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


def test_guided_setup_defaults_source_to_inferred_checkout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    model = source / "checkpoints" / "checkpoint-1"
    model.mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname = 'tiny-model'\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    calls: list[list[str]] = []
    interactive_cli._guided_setup(
        FakeTerminal(f"{model}\n\n\n提高长程一致性\n"), FakeTerminal(),
        lambda argv: calls.append(argv) or 0, interactive_cli._Theme(False),
    )
    assert calls == [[
        "setup", "--model", str(model), "--data", str(data),
        "--goal", "提高长程一致性", "--source", str(source),
    ]]


@pytest.mark.parametrize("interrupt", [None, EOFError, KeyboardInterrupt])
def test_guided_setup_cancels_at_source_prompt_without_dispatch(monkeypatch, tmp_path: Path, interrupt) -> None:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()

    class StopAtSource(FakeTerminal):
        def readline(self, *args, **kwargs):
            value = super().readline(*args, **kwargs)
            if not value and interrupt is not None:
                raise interrupt
            return value

    output = FakeTerminal()
    calls: list[list[str]] = []
    interactive_cli._guided_setup(
        StopAtSource(f"{model}\n"), output, lambda argv: calls.append(argv) or 0,
        interactive_cli._Theme(False),
    )
    assert calls == []
    assert "源码目录" in output.getvalue()
    assert "已取消设置" in output.getvalue()
    assert not (tmp_path / "verdiwm.toml").exists()


@pytest.mark.parametrize("separate_source", [False, True])
def test_guided_setup_saves_source_choice_through_public_setup(monkeypatch, tmp_path: Path, separate_source: bool) -> None:
    from wmloop.cli import main
    from wmloop.control.first_contact import inspect_project
    from wmloop.control.project_config import load_project_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    model, data = tmp_path / "weights", tmp_path / "data"
    source = tmp_path / "source" if separate_source else model
    for directory in {model, source, data}:
        directory.mkdir()
    (model / "model.safetensors").write_bytes(b"tiny-test-weights")
    for entrypoint in ("train.py", "evaluate.py"):
        (source / entrypoint).write_text("raise RuntimeError('setup must not execute model code')\n", encoding="utf-8")
    output = FakeTerminal()
    source_answer = str(source) if separate_source else ""
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal(f"/setup\n{model}\n{source_answer}\n{data}\n提高长程一致性\n/exit\n"),
        stdout=output, dispatch=main,
    )
    config = load_project_config(cwd=tmp_path)
    assert Path(config.values["model"]) == model
    if separate_source:
        assert Path(config.values["source"]) == source
    else:
        assert "source" not in config.values
    readiness = inspect_project(root=tmp_path, **{
        key: config.values[key] for key in ("model", "source", "data") if key in config.values
    })
    assert "MODEL_SOURCE_NOT_SEPARATE" not in {item["code"] for item in readiness["blockers"]}
    assert Path(config.values["data"]) == data
    assert config.values["goal"] == "提高长程一致性"
    assert "配置已保存" in output.getvalue()
    assert "Readiness  needs_input" in output.getvalue()
    assert "setup must not execute model code" not in output.getvalue()


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


@pytest.mark.parametrize("command", ["/run --rounds 2 --stop-on-success", "/run"])
def test_run_round_change_reviews_persisted_derivation_and_confirms_once(monkeypatch, tmp_path: Path, command: str) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    original = {"state": "ready", "goal": "improve consistency", "budget": {"gpu_hours": 2}, "execution_policy": {"rounds": 1, "stop_on_success": False, "budget_scope": "all_rounds"}}
    plan.write_text(json.dumps(original), encoding="utf-8")
    before = plan.read_bytes()
    derived = plan.parent / "reviewed-rounds.json"
    calls: list[list[str]] = []

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        if "--confirm" not in argv:
            document = {**original, "execution_policy": {"rounds": 2, "stop_on_success": "--stop-on-success" in argv, "budget_scope": "all_rounds"}}
            derived.write_text(json.dumps(document), encoding="utf-8")
            print(json.dumps({"state": "awaiting_confirmation", "plan_path": str(derived), "plan": document, "policy_changed": True}))
        else:
            assert json.loads(Path(argv[argv.index("--plan") + 1]).read_text())["execution_policy"]["rounds"] == 2
            print(json.dumps({"state": "completed", "campaign_id": "reviewed"}))
        return 0

    selection = "2\n" if command == "/run" else ""
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal(command + "\n" + selection + "y\n/exit\n"), stdout=output, dispatch=dispatch,
    )
    assert len(calls) == 2
    assert "--rounds" in calls[0] and "--confirm" not in calls[0]
    assert calls[1] == ["research", "run", "--plan", str(derived), "--confirm"]
    assert output.getvalue().count("确认创建并执行这个 campaign？") == 1
    assert "最多 2 轮" in output.getvalue()
    assert "所有轮次共享" in output.getvalue()
    assert plan.read_bytes() == before


def test_run_enter_retains_planned_rounds_before_confirming(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "ready", "goal": "improve consistency", "execution_policy": {"rounds": 4, "stop_on_success": True}}), encoding="utf-8")
    calls: list[list[str]] = []
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/run\n\ny\n/exit\n"), stdout=output,
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "run", "--plan", str(plan), "--confirm"]]
    assert "[4]" in output.getvalue()
    assert "冻结验证器正式确认" in output.getvalue()


@pytest.mark.parametrize("input_text", ["/run\n", "/run\ninvalid\n/exit\n", "/run\n0\n/exit\n"])
def test_round_selection_eof_or_invalid_input_cannot_create_campaign(monkeypatch, tmp_path: Path, input_text: str) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "ready", "goal": "goal"}), encoding="utf-8")
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal(input_text), stdout=FakeTerminal(), dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == []


def test_rejected_derived_plan_is_reused_by_next_run_without_path_copying(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / ".verdiwm" / "research-plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"state": "ready", "goal": "goal"}), encoding="utf-8")
    derived = plan.parent / "derived.json"
    calls: list[list[str]] = []

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        if "--confirm" not in argv:
            document = {"state": "ready", "goal": "goal", "execution_policy": {"rounds": 3, "stop_on_success": False}}
            derived.write_text(json.dumps(document), encoding="utf-8")
            print(json.dumps({"state": "awaiting_confirmation", "plan_path": str(derived), "plan": document}))
        return 0

    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/run --rounds 3\nn\n/run\ny\n/exit\n"), stdout=FakeTerminal(), dispatch=dispatch,
    )
    assert len(calls) == 2
    assert calls[1] == ["research", "run", "--plan", str(derived), "--confirm"]


def test_failed_run_preview_cannot_reuse_previous_confirmation_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    output = FakeTerminal()

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        print(json.dumps({"state": "blocked", "blockers": [{"code": "MISSING_MODEL"}]}))
        return 2

    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/run --rounds 2\n/exit\n"), stdout=output, dispatch=dispatch,
    )
    assert len(calls) == 1 and "--confirm" not in calls[0]
    assert "确认创建并执行这个 campaign？" not in output.getvalue()


def test_explicit_plan_rounds_and_stop_policy_reach_public_handler() -> None:
    calls: list[list[str]] = []
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal('/plan "improve consistency" --rounds 2 --stop-on-success\n/exit\n'), stdout=FakeTerminal(),
        dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert calls == [["research", "plan", "--goal", "improve consistency", "--rounds", "2", "--stop-on-success"]]


@pytest.mark.parametrize("command", ["/run --rounds 2 --confirm", "/run --rounds 2 --confirm --plan explicit.json"])
def test_explicit_confirmation_executes_once_without_preview_or_second_prompt(monkeypatch, tmp_path: Path, command: str) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal(command + "\n/exit\n"), stdout=output, dispatch=lambda argv: calls.append(argv) or 0,
    )
    assert len(calls) == 1
    assert calls[0][:2] == ["research", "run"]
    assert calls[0].count("--confirm") == 1
    assert calls[0][calls[0].index("--rounds") + 1] == "2"
    expected_plan = "explicit.json" if "explicit.json" in command else str(tmp_path / ".verdiwm" / "research-plan.json")
    assert calls[0][calls[0].index("--plan") + 1] == expected_plan
    assert "即将执行已确认计划" in output.getvalue()
    assert "预览" not in output.getvalue()
    assert "确认创建并执行这个 campaign？" not in output.getvalue()


def test_explicit_confirmation_reuses_last_reviewed_derived_plan(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    derived = tmp_path / "derived.json"
    calls: list[list[str]] = []

    def dispatch(argv: list[str]) -> int:
        calls.append(argv)
        if "--confirm" not in argv:
            document = {"state": "ready", "goal": "goal", "execution_policy": {"rounds": 2, "stop_on_success": False}}
            derived.write_text(json.dumps(document), encoding="utf-8")
            print(json.dumps({"state": "awaiting_confirmation", "plan_path": str(derived), "plan": document}))
        return 0

    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/run --rounds 2\nn\n/run --confirm\n/exit\n"), stdout=FakeTerminal(), dispatch=dispatch,
    )
    assert len(calls) == 2
    assert "--confirm" not in calls[0]
    assert calls[1] == ["research", "run", "--confirm", "--plan", str(derived)]


def test_real_multiround_plan_displays_limits_and_cannot_run_without_resources(monkeypatch, tmp_path: Path) -> None:
    from wmloop.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(tmp_path / "user-state"))
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    output = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal('/plan "improve consistency" --model model --data data --rounds 2 --budget 3gpu-hours --stop-on-success\n/run\n/exit\n'),
        stdout=output, dispatch=main,
    )
    document = json.loads((tmp_path / ".verdiwm" / "research-plan.json").read_text(encoding="utf-8"))
    assert document["execution_policy"] == {"rounds": 2, "stop_on_success": True, "budget_scope": "all_rounds"}
    assert document["state"] == "blocked"
    assert "最多 2 轮" in output.getvalue()
    assert "3.0（所有轮次共享）" in output.getvalue()
    assert "冻结验证器正式确认收益后提前停止" in output.getvalue()
    assert "确认创建并执行这个 campaign？" not in output.getvalue()
    assert not (tmp_path / "user-state" / "campaigns").exists()


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


def test_terminal_matching_plan_guides_resume_but_new_plan_stays_runnable(tmp_path):
    plan_path = tmp_path / "plan.json"
    old_digest, new_digest = "a" * 64, "b" * 64
    plan_path.write_text(json.dumps({"schema_version": 3, "plan_digest": old_digest, "state": "ready"}))
    campaign = {"campaign_id": "research-" + old_digest[:24], "status": "blocked"}
    arguments = {"readiness": {"state": "ready"}, "plan_path": plan_path, "campaigns": [campaign]}
    assert interactive_cli._next_step({"goal": "improve"}, **arguments) == ("PREVIOUS RUN BLOCKED", "/resume")
    plan_path.write_text(json.dumps({"schema_version": 3, "plan_digest": new_digest, "state": "ready"}))
    assert interactive_cli._next_step({"goal": "improve"}, **arguments) == ("PLAN READY", "/run")


def test_resume_of_terminal_plan_shows_result_without_execution_confirmation(monkeypatch, tmp_path):
    digest = "a" * 64
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": 3, "plan_digest": digest, "state": "ready"}))
    campaign = {"campaign_id": "research-" + digest[:24], "status": "blocked",
                "execution_error": {"type": "FixtureBoundary", "message": "FIXTURE_REQUIRES_REPAIR"}}
    monkeypatch.setattr(interactive_cli, "_project_snapshot", lambda _: {"goal": "improve"})
    monkeypatch.setattr(interactive_cli, "_recent_campaigns", lambda *_: [campaign])
    monkeypatch.setattr(interactive_cli, "_default_plan_path", lambda *_: plan_path)
    stdin, stdout = StringIO("y\n"), FakeTerminal()
    session = interactive_cli.SessionState()
    interactive_cli._dispatch_slash("resume", [], lambda _: pytest.fail("resume must remain read-only"),
                                    stdout=stdout, theme=interactive_cli._Theme(False), stdin=stdin, session=session)
    assert stdin.tell() == 0
    assert "blocked" in stdout.getvalue()
    assert "/plan" in stdout.getvalue()
    assert session.last_campaign_id == campaign["campaign_id"]
    assert session.last_state == "blocked"


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


@pytest.mark.parametrize("envelope", ["status", "research_run", "campaign_list"])
def test_campaign_results_show_round_completion_and_unconfirmed_goal(envelope: str) -> None:
    record = {
        "campaign_id": "two-rounds", "status": "completed",
        "execution_result": {
            "stop_reason": "MAX_ITERATIONS_REACHED", "rounds_completed": 2,
            "evolution_manifest": {"goal_success": {"state": "not_confirmed", "reason": "FROZEN_GOAL_NOT_CONFIGURED"}},
        },
    }
    payload = record if envelope == "status" else {"state": "completed", "campaign": record} if envelope == "research_run" else {"items": [record]}
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue().replace("\n", "")
    for expected in ("已完成 2 轮", "达到设定轮数上限", "MAX_ITERATIONS_REACHED", "尚未正式确认目标收益", "FROZEN_GOAL_NOT_CONFIGURED", "未绑定 evaluator 的 frozen_goal"):
        assert expected in text
    assert "冻结验证器已正式确认目标收益" not in text
    assert text.count("已完成 2 轮") == 1


def test_failed_campaign_status_shows_actionable_summary_without_model_path() -> None:
    model_path = "/private/models/wan5b/checkpoint-63020"
    receipt_path = "/private/state/runs/campaign-1/execution-receipt.json"
    payload = {
        "campaign_id": "campaign-1",
        "status": "failed",
        "model": model_path,
        "execution": {"output_root": "/private/state/runs/campaign-1"},
        "execution_error": {"type": "ModelRunnerError", "message": "MODEL_RUN_TRAIN_FAILED"},
        "revision": {"large": "private execution manifest"},
    }
    output = FakeTerminal()

    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))

    text = output.getvalue()
    assert "Failed at" in text and "train" in text
    assert "MODEL_RUN_TRAIN_FAILED" in text
    assert receipt_path in text
    assert "stderr" in text and "/run" in text
    assert "未能从阶段摘要提取根因" in text
    assert model_path not in text
    assert "private execution manifest" not in text


def test_failed_campaign_summary_reads_and_safely_bounds_receipt_root_cause(
    tmp_path: Path,
) -> None:
    from wmloop.control.terminal_results import render_result

    output_root = tmp_path / "campaign-output"
    output_root.mkdir()
    receipt_path = output_root / "execution-receipt.json"
    private_path = "/".join(("", "cs", "wjy", "private", "trainer.py"))
    receipt_path.write_text(json.dumps({
        "artifact_type": "verdiwm-model-run-execution-receipt",
        "stages": [{
            "stage": "train",
            "returncode": 1,
            "stderr_tail": (
                "ChildFailedError\nerror_file=N/A\n"
                f"RuntimeError: failed at {private_path} api_key=secret-value"
            ),
        }],
    }), encoding="utf-8")
    payload = {
        "campaign_id": "campaign-root-cause",
        "status": "failed",
        "execution": {"output_root": str(output_root)},
        "execution_error": {"message": "MODEL_RUN_TRAIN_FAILED"},
    }
    output = FakeTerminal()

    render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))

    text = output.getvalue()
    assert "RuntimeError: failed at [path] api_key=[redacted]" in text
    assert private_path not in text
    assert "secret-value" not in text
    assert str(receipt_path) in text.replace("\n", "")


def test_nonfailed_campaign_status_keeps_compatible_summary_without_failure_guidance() -> None:
    output = FakeTerminal()
    payload = {"campaign_id": "campaign-2", "status": "completed", "execution_result": {"rounds_completed": 1}}

    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))

    text = output.getvalue()
    assert "completed" in text
    assert "已完成 1 轮" in text
    assert "Failed at" not in text
    assert "Error code" not in text
    assert "Receipt" not in text


def test_public_campaign_status_remains_parseable_machine_json_with_core_fields(monkeypatch, tmp_path: Path, capsys) -> None:
    from wmloop.cli import main
    from wmloop.control.campaign_api import CampaignStore

    state_root = tmp_path / "user-state"
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}', encoding="utf-8")
    store = CampaignStore(state_root / "campaigns", project_root=tmp_path)
    store.create({
        "campaign_id": "machine-json-fixture", "goal": "improve score",
        "model": str(model), "dataset": str(data), "budget": 1,
        "resource_request": {"gpu_indices": []},
        "execution": {
            "kind": "pipeline", "repo_root": str(model), "output_root": str(tmp_path / "output"),
            "evaluator_contract": str(evaluator), "budget_total_gpu_hours": 1,
            "asset_bindings": {"--checkpoint": str(model)},
        },
    })
    store.confirm("machine-json-fixture")
    store.record_dispatch_result("machine-json-fixture", status="running")
    store.record_dispatch_result(
        "machine-json-fixture", status="failed",
        error={"type": "ModelRunnerError", "message": "MODEL_RUN_TRAIN_FAILED"},
    )

    assert main(["status", "machine-json-fixture", "--state-root", str(state_root)]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["campaign_id"] == "machine-json-fixture"
    assert payload["status"] == "failed"
    assert payload["model"] == str(model)
    assert payload["execution_error"]["message"] == "MODEL_RUN_TRAIN_FAILED"
    assert payload["execution"]["output_root"] == str(tmp_path / "output")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state_root))
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    terminal = FakeTerminal()
    interactive_cli.run_interactive_session(
        stdin=FakeTerminal("/status machine-json-fixture\n/exit\n"),
        stdout=terminal, dispatch=main,
    )
    rendered = terminal.getvalue()
    rendered_text = rendered.replace("\n", "")
    assert "MODEL_RUN_TRAIN_FAILED" in rendered
    assert "Failed at" in rendered and "train" in rendered
    assert str(tmp_path / "output" / "execution-receipt.json") in rendered_text
    assert str(model) not in rendered


def test_confirmed_goal_is_distinct_from_successful_execution() -> None:
    output = FakeTerminal()
    payload = {"status": "completed", "execution_result": {
        "stop_reason": "GOAL_CONFIRMED", "rounds_completed": 1,
        "evolution_manifest": {"goal_success": {"state": "confirmed", "reason": "FROZEN_GOAL_CONFIRMED"}},
    }}
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue().replace("\n", "")
    assert "冻结验证器已正式确认目标收益（confirmed）" in text
    assert "提前停止（GOAL_CONFIRMED）" in text
    assert "FROZEN_GOAL_CONFIRMED" in text


def test_stop_reason_alone_cannot_claim_goal_success() -> None:
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps({"status": "completed", "execution_result": {
        "stop_reason": "MAX_ITERATIONS_REACHED", "rounds_completed": 2,
    }}), stdout=output, theme=interactive_cli._Theme(False))
    assert "执行完成不代表模型达标" in output.getvalue()
    assert "冻结验证器已正式确认目标收益" not in output.getvalue()


def test_cancelled_execution_shows_zero_completed_rounds_and_stable_reason() -> None:
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps({"status": "cancelled", "execution_result": {
        "stop_reason": "SIGNAL_RECEIVED", "rounds_completed": 0,
    }}), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue().replace("\n", "")
    assert "已完成 0 轮" in text
    assert "收到取消请求或停止信号（SIGNAL_RECEIVED）" in text


def test_plan_preview_and_result_keep_frozen_goal_warning_visible() -> None:
    warning = "尚未绑定 evaluator 的 frozen_goal；系统无法确认目标达成。"
    payload = {"state": "ready", "execution_policy": {"rounds": 2, "stop_on_success": True}, "warnings": [warning]}
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    assert output.getvalue().replace("\n", "").count(warning) == 1
    preview = FakeTerminal()
    interactive_cli._render_plan_preview(payload, Path("plan.json"), stdout=preview, theme=interactive_cli._Theme(False))
    assert warning in preview.getvalue().replace("\n", "")


def test_nested_outcome_keeps_invalid_evidence_reason_and_sanitizes_warnings(monkeypatch) -> None:
    from wmloop.control.terminal_ui import display_width

    monkeypatch.setenv("COLUMNS", "50")
    payload = {"campaign": {"status": "completed", "execution_result": {
        "stop_reason": "MAX_ITERATIONS_REACHED", "rounds_completed": 1,
        "warnings": ["inspect \033[2J/check\033]0;unexpected-title\a"],
        "evolution_manifest": {
            "goal_success": {"state": "not_confirmed", "reason": "GOAL_EVIDENCE_INVALID:ValueError"},
            "warnings": ["missing frozen goal"],
        },
    }}}
    output = FakeTerminal()
    interactive_cli._render_result(json.dumps(payload), stdout=output, theme=interactive_cli._Theme(False))
    text = output.getvalue()
    assert "GOAL_EVIDENCE_INVALID:ValueError" in text.replace("\n", "")
    assert "正式目标证据无效或无法读取" in text
    assert "missing frozen goal" in text
    assert "inspect /check" in text
    assert "\033" not in text and "unexpected-title" not in text
    assert all(display_width(line) < 50 for line in text.splitlines())


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
