"""Default planning preserves evidence while allowing users to fix blockers."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmloop.control import research_plan_history
from wmloop.control.research_plan_history import write_default_research_plan
from wmloop.control.research_request import compile_research_plan, load_research_plan, write_research_plan, ResearchRequestError


def _plan(root: Path, goal: str):
    for name in ("model", "data"):
        (root / name).mkdir(exist_ok=True)
    return compile_research_plan(project_root=root, model=root / "model", data=root / "data", goal=goal)


def test_default_planning_keeps_old_versions_and_updates_latest(tmp_path):
    original = _plan(tmp_path, "first goal")
    first = write_default_research_plan(original, tmp_path)
    old_bytes = first.read_bytes()
    replacement = _plan(tmp_path, "new goal")
    second = write_default_research_plan(replacement, tmp_path)
    assert first != second
    assert first.read_bytes() == old_bytes
    assert load_research_plan(second)["goal"] == "new goal"
    assert load_research_plan(tmp_path / ".verdiwm/research-plan.json") == replacement
    assert write_default_research_plan(replacement, tmp_path) == second


def test_explicit_output_still_rejects_replacement(tmp_path):
    destination = tmp_path / "chosen.json"
    write_research_plan(_plan(tmp_path, "first"), destination)
    with pytest.raises(ResearchRequestError, match="OUTPUT_CONFLICT"):
        write_research_plan(_plan(tmp_path, "second"), destination)


def test_latest_write_failure_preserves_previous_selection(tmp_path, monkeypatch):
    first = _plan(tmp_path, "first")
    write_default_research_plan(first, tmp_path)
    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(research_plan_history, "atomic_write", fail)
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_OUTPUT_INVALID"):
        write_default_research_plan(_plan(tmp_path, "second"), tmp_path)
    assert load_research_plan(tmp_path / ".verdiwm/research-plan.json") == first
