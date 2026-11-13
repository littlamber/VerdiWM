from __future__ import annotations

from io import StringIO
import os
from pathlib import Path

import pytest

from wmloop.control.interactive_cli import COMMANDS, render_welcome
from wmloop.control.terminal_palette import draw_palette, palette_lines
from wmloop.control.terminal_ui import Dashboard, Theme, display_width, render_dashboard, wrap_lines


@pytest.mark.parametrize("columns", [24, 40, 80, 120])
def test_dashboard_preserves_cjk_resources_within_terminal(columns: int) -> None:
    output = StringIO()
    model = "models/世界模型/" + "checkpoint/" * 12
    snapshot = Dashboard(
        version="test",
        project_root=Path("workspace"),
        config={"model": model, "goal": "提高长程一致性"},
        readiness={"state": "blocked", "blockers": [{"message": "请绑定评测器", "action": "/evaluator"}]},
        llm={"state": "missing"},
        next_label="BLOCKED",
        next_command="/check",
    )
    render_dashboard(snapshot, stdout=output, theme=Theme(False), width=columns - 2)
    lines = output.getvalue().splitlines()
    assert all(display_width(line) < columns for line in lines)
    # Reassemble panel text to verify that wrapping doesn't hide path suffixes.
    text = "".join(line.replace("│", "").replace("|", "").strip() for line in lines).replace(" ", "")
    assert model in text
    assert "/check" in text
    assert "\x1b" not in text


def test_resource_values_cannot_issue_terminal_controls() -> None:
    output = StringIO()
    render_dashboard(
        Dashboard(
            version="test", project_root=Path("workspace"),
            config={"model": "before\x1b[2Jafter\x1b]0;spoofed\x07"},
            readiness=None, llm={}, next_label="SETUP", next_command="/setup",
        ),
        stdout=output, theme=Theme(False), width=80,
    )
    assert "beforeafter" in output.getvalue()
    assert "\x1b" not in output.getvalue()


@pytest.mark.parametrize("missing_database", [False, True])
def test_welcome_keeps_unreadable_campaign_state_actionable(monkeypatch, tmp_path: Path, missing_database: bool) -> None:
    state = tmp_path / "state"
    store = state / "campaigns"
    store.mkdir(parents=True)
    (store / ".sqlite-campaign-store").write_bytes(b"original store marker")
    database = store / "campaigns.sqlite3"
    if not missing_database:
        database.write_bytes(b"unreadable SQLite fixture")
    original = {path.name: path.read_bytes() for path in store.iterdir()}
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(state))
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    project = tmp_path / "new-user-project"
    project.mkdir()
    output = StringIO()

    render_welcome(stdout=output, project_root=project, color=False)

    text = output.getvalue()
    assert "STATE BLOCKED" in text
    assert "CAMPAIGN_STORE_UNAVAILABLE" in text
    assert "/status" in text
    assert "尚无记录" not in text
    assert {path.name: path.read_bytes() for path in store.iterdir()} == original


@pytest.mark.parametrize("selected", [0, 12, len(COMMANDS) - 1])
@pytest.mark.parametrize("size", [(40, 10), (80, 24)])
def test_palette_selection_stays_visible_and_fits_terminal(selected: int, size: tuple[int, int]) -> None:
    columns, rows = size
    lines = palette_lines(COMMANDS, "", selected, theme=Theme(False), columns=columns, rows=rows)
    selection = [line for line in lines if line.startswith("  > ")]
    assert len(selection) == 1
    assert "/" + COMMANDS[selected].name in selection[0]
    assert len(lines) < rows
    assert all(display_width(line) < columns for line in lines)


def test_filtered_palette_clears_rows_from_previous_longer_menu(monkeypatch) -> None:
    from wmloop.control import terminal_palette

    monkeypatch.setattr(terminal_palette, "terminal_size", lambda stream: os.terminal_size((80, 24)))
    output = StringIO()
    previous = draw_palette(COMMANDS, "", 0, stdout=output, theme=Theme(True), previous_lines=0)
    output.seek(0)
    output.truncate()
    current = draw_palette([], "no-match", 0, stdout=output, theme=Theme(True), previous_lines=previous)
    assert current < previous
    # Every old row is overwritten, even when filtering leaves zero matches.
    assert output.getvalue().count("\033[2K") == previous


def test_wrapping_retains_combining_marks_and_newlines() -> None:
    value = "项目/e\u0301/weights\n尾行"
    wrapped = wrap_lines(value, 8)
    assert "".join(wrapped) == value.replace("\n", "")
    assert all(display_width(line) <= 8 for line in wrapped)
