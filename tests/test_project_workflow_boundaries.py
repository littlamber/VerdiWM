"""CPU regressions for repeated commands, project roots and competing writes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from wmloop.cli import main
from wmloop.control.commands.inputs import resolve_state_root
from wmloop.control.first_contact import FirstContactError, initialize_project
from wmloop.control.onboarding_assistant import write_onboarding_questionnaire
from wmloop.control.research_request import ResearchRequestError, compile_research_plan, load_research_plan, write_research_plan
from wmloop.control.research_plan_history import write_default_research_plan


def _project(root: Path, goal: str = "fixture") -> Path:
    root.mkdir(exist_ok=True)
    (root / "model").mkdir()
    (root / "data").mkdir()
    initialize_project(root=root, goal=goal)
    return root


def test_nested_planning_uses_project_root_and_explicit_project_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    project = _project(tmp_path / "project", "chosen goal")
    other = _project(tmp_path / "other", "wrong goal")
    nested = project / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert main(["research", "plan"]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["project_root"] == str(project)
    assert Path(first["plan_path"]).is_relative_to(project / ".verdiwm")
    assert not (nested / ".verdiwm").exists()
    monkeypatch.chdir(other)
    assert main(["research", "plan", "--project-root", str(project)]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second["goal"] == "chosen goal"
    assert second["model"] == str(project / "model")


def test_state_root_uses_nearest_project_but_respects_explicit_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    project = _project(tmp_path / "project")
    nested = project / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert resolve_state_root(None) == project / ".verdiwm/state"
    explicit = tmp_path / "global-state"
    monkeypatch.setenv("VERDIWM_STATE_ROOT", str(explicit))
    assert resolve_state_root(explicit) == explicit


def test_repeated_questionnaire_preserves_user_answers(tmp_path):
    path = tmp_path / "questions.json"
    questionnaire = {"questions": [{"id": "model", "prompt": "Model?"}]}
    write_onboarding_questionnaire(path, questionnaire)
    assert write_onboarding_questionnaire(path, questionnaire) == path
    answered = {"questions": [{"id": "model", "prompt": "Model?", "answer": "my model"}]}
    path.write_text(json.dumps(answered))
    before = path.read_bytes()
    with pytest.raises(FirstContactError, match="OUTPUT_CONFLICT"):
        write_onboarding_questionnaire(path, questionnaire)
    assert path.read_bytes() == before


def test_competing_plan_writes_keep_single_winner(tmp_path):
    root = _project(tmp_path / "project")
    plans = [compile_research_plan(project_root=root, goal=f"goal {i}") for i in range(4)]
    output = tmp_path / "chosen.json"
    barrier = Barrier(len(plans))
    def write(plan):
        barrier.wait(timeout=5)
        try:
            write_research_plan(plan, output)
            return plan["goal"]
        except ResearchRequestError as exc:
            assert str(exc) == "RESEARCH_PLAN_OUTPUT_CONFLICT"
            return None
    with ThreadPoolExecutor(max_workers=len(plans)) as executor:
        outcomes = list(executor.map(write, plans))
    winners = [value for value in outcomes if value is not None]
    assert len(winners) == 1
    assert load_research_plan(output)["goal"] == winners[0]


def test_competing_setup_without_force_does_not_overwrite(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    barrier = Barrier(4)
    def setup(index):
        barrier.wait(timeout=5)
        try:
            return initialize_project(root=tmp_path, goal=f"goal {index}")["goal"]
        except FirstContactError as exc:
            assert str(exc).startswith("PROJECT_FILE_EXISTS")
            return None
    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(setup, range(4)))
    assert len([value for value in outcomes if value is not None]) == 1


def test_competing_default_plans_keep_all_versions_and_valid_latest(tmp_path):
    root = _project(tmp_path / "project")
    plans = [compile_research_plan(project_root=root, goal=f"goal {i}") for i in range(4)]
    barrier = Barrier(len(plans))
    def write(plan):
        barrier.wait(timeout=5)
        return write_default_research_plan(plan, root)
    with ThreadPoolExecutor(max_workers=len(plans)) as executor:
        versions = list(executor.map(write, plans))
    assert len(set(versions)) == len(plans)
    for path, plan in zip(versions, plans):
        assert load_research_plan(path) == plan
    assert load_research_plan(root / ".verdiwm/research-plan.json") in plans
