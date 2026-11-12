"""CPU CLI regressions for first-use readiness and actionable input errors."""

from __future__ import annotations

import argparse
from importlib.metadata import metadata
from pathlib import Path

import pytest

from wmloop.cli import main
from wmloop.control.commands import system


def test_doctor_uses_installed_python_requirement(monkeypatch):
    monkeypatch.setattr(system.platform, "python_version", lambda: "3.12.1")
    report = system.doctor_report(argparse.Namespace(repo_root=None))
    check = next(row for row in report["checks"] if row["name"] == "python_supported")
    assert check["expected"] == metadata("verdiwm")["Requires-Python"]
    assert check["state"] == "pass"
    monkeypatch.setattr(system.platform, "python_version", lambda: "3.10.1")
    report = system.doctor_report(argparse.Namespace(repo_root=None))
    assert report["state"] == "blocked"


@pytest.mark.parametrize("command", ["guide-model", "check-model", "diagnose"])
def test_corrupt_project_is_not_silently_ignored(tmp_path: Path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    (tmp_path / "verdiwm.toml").write_text("[project\n")
    assert main([command]) == 2
    captured = capsys.readouterr()
    assert "PROJECT_CONFIG_INVALID" in captured.err
    assert "Traceback" not in captured.err


def test_missing_evaluator_source_returns_actionable_failure(tmp_path: Path, capsys):
    assert main(["evaluator", "discover", "--source-root", str(tmp_path / "missing")]) == 2
    assert "EVALUATOR_SOURCE_DIRECTORY_REQUIRED" in capsys.readouterr().err


@pytest.mark.parametrize("setting", ["mode=23", 'mode="unknown"', "target_metrics=23", "target_metrics=[1,true]", "goal=23", "adapter=true", "budget=false"])
def test_project_value_types_fail_before_planning(tmp_path, monkeypatch, capsys, setting):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "verdiwm.toml").write_text('[project]\nmodel="model"\ndata="data"\n' + setting + '\n')
    assert main(["research", "plan", "--goal", "fixture"]) == 2
    captured = capsys.readouterr()
    assert "PROJECT_CONFIG_VALUE_INVALID" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / ".verdiwm/research-plan.json").exists()


def test_plan_output_parent_file_is_actionable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    config = tmp_path / "verdiwm.toml"
    config.write_text('[project]\nmodel="model"\ndata="data"\ngoal="fixture"\n')
    before = config.read_bytes()
    assert main(["research", "plan", "--output", str(config / "plan.json")]) == 2
    captured = capsys.readouterr()
    assert "RESEARCH_PLAN_OUTPUT_INVALID" in captured.err
    assert "Traceback" not in captured.err
    assert config.read_bytes() == before


@pytest.mark.parametrize("command", [
    ["guide-model", "--output", "verdiwm.toml/questions.json"],
    ["setup", "--goal", "fixture", "--project-file", "verdiwm.toml/another.toml"],
    ["status", "--state-root", "verdiwm.toml"],
])
def test_cli_filesystem_errors_are_actionable(tmp_path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDIWM_PROJECT_CONFIG", raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    config = tmp_path / "verdiwm.toml"
    config.write_text('[project]\nmodel="model"\ndata="data"\ngoal="fixture"\n')
    before = config.read_bytes()
    assert main(command) == 2
    captured = capsys.readouterr()
    assert "LOCAL_FILESYSTEM_ERROR" in captured.err
    assert "Traceback" not in captured.err
    assert config.read_bytes() == before
