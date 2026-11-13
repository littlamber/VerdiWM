"""Progress feedback must not alter plan evidence or the CLI JSON contract."""

import json
import threading

import pytest

from wmloop.cli import main
from wmloop.control import research_request
from wmloop.control.commands.research import _ResearchProgress


def test_digest_progress_is_observational(tmp_path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    expected = research_request._path_digest(tmp_path)
    events = []
    monkeypatch.setattr(research_request, "_PROGRESS_INTERVAL_SECONDS", 0)
    assert research_request._path_digest(tmp_path, progress=events.append) == expected
    assert [event["state"] for event in events] == ["started", "running", "running", "completed"]
    assert events[-1]["files_hashed"] == 2
    assert events[-1]["input_bytes"] == 2

    def fail(_event):
        raise RuntimeError("display failure")

    assert research_request._path_digest(tmp_path, progress=fail) == expected


def test_source_bindings_use_same_exclusions_when_reverified(tmp_path):
    model, data, source = (tmp_path / name for name in ("model", "data", "source"))
    for path in (model, data, source):
        path.mkdir()
    (source / "model.py").write_text("original", encoding="utf-8")
    (source / "outputs").mkdir()
    (source / "outputs" / "result").write_text("one", encoding="utf-8")
    bindings = research_request._input_bindings(
        model=model, data=data, source=source, evaluator=None, runtime_python=None,
        adapter_profile=None, model_irg=None, asset_bindings={"--repo": str(source)},
    )
    plan = {"input_bindings": bindings, "input_digest": research_request._input_digest(bindings)}
    (source / "outputs" / "result").write_text("two", encoding="utf-8")
    assert research_request.verify_research_plan_inputs(plan)["state"] == "verified"
    (source / "model.py").write_text("changed", encoding="utf-8")
    result = research_request.verify_research_plan_inputs(plan)
    assert result["state"] == "blocked"
    assert {item["name"] for item in result["drift"]} >= {"source", "asset:--repo"}


def test_cli_progress_stays_on_stderr(tmp_path, capsys):
    (tmp_path / "model").mkdir()
    (tmp_path / "data").mkdir()
    status = main(["research", "plan", "--project-root", str(tmp_path),
                   "--model", str(tmp_path / "model"), "--data", str(tmp_path / "data"),
                   "--goal", "goal"])
    output = capsys.readouterr()
    assert status == 2
    assert json.loads(output.out)["state"] == "blocked"
    assert '"phase": "onboarding_scan"' in output.err
    assert '"phase": "input_digest"' in output.err


def test_heartbeat_stops_when_operation_raises(monkeypatch):
    heartbeat = threading.Event()
    events = []

    def record(event):
        events.append(event)
        if event["state"] == "waiting":
            heartbeat.set()

    monkeypatch.setattr(_ResearchProgress, "_write", staticmethod(record))
    progress = _ResearchProgress(interval=0.01)
    with pytest.raises(RuntimeError):
        with progress:
            progress({"phase": "onboarding_scan", "state": "started"})
            assert heartbeat.wait(1)
            raise RuntimeError("failure")
    assert not progress._thread.is_alive()
    assert events[-1]["phase"] == "onboarding_scan"
