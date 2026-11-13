from pathlib import Path
import sys

import pytest

from wmloop.control.onboarding_conformance import (
    ModelConformanceError,
    _materialize_command,
    _source_integrity_check,
)


def _report(command):
    return {
        "evaluator_contract": {"state": "ready", "command": command, "entrypoint_probe": "skip"},
        "connector": {"asset_bindings": []},
    }


def test_deferred_entrypoint_still_requires_script(tmp_path):
    with pytest.raises(ModelConformanceError, match="CONFORMANCE_ENTRYPOINT_NOT_FOUND"):
        _materialize_command(
            _report(["{python}", "scripts/missing.py"]),
            runtime=Path(sys.executable), repo=tmp_path,
        )


def test_entrypoint_resolves_against_model_repo(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text("pass\n")
    command = _materialize_command(
        _report(["{python}", "evaluate.py", "--input", "future.py"]),
        runtime=Path(sys.executable), repo=tmp_path,
    )
    assert command[1] == str(script)
    assert command[-1] == "future.py"


@pytest.mark.parametrize("options", [["-u"], ["-W", "ignore"], ["-X", "dev"], ["--"]])
def test_script_after_python_options_must_exist(tmp_path, options):
    with pytest.raises(ModelConformanceError, match="CONFORMANCE_ENTRYPOINT_NOT_FOUND"):
        _materialize_command(
            _report(["{python}", *options, "missing.py"]),
            runtime=Path(sys.executable), repo=tmp_path,
        )


@pytest.mark.parametrize("option", ["-m", "-c"])
def test_module_or_code_arguments_are_not_treated_as_script_files(tmp_path, option):
    command = _materialize_command(
        _report(["{python}", "-u", option, "not-a-file.py", "argument.py"]),
        runtime=Path(sys.executable), repo=tmp_path,
    )
    assert command[3:] == ["not-a-file.py", "argument.py"]


@pytest.mark.parametrize("before_state,after_state", [("unbound", "unbound"), ("bound", "unbound"), ("unbound", "bound")])
def test_source_integrity_requires_both_revisions_bound(before_state, after_state):
    check = _source_integrity_check({"state": before_state}, {"state": after_state})
    assert check["status"] == "fail"
    assert "CONFORMANCE_SOURCE_REVISION_UNBOUND" in str(check)


def test_source_integrity_still_rejects_changed_bound_revision():
    before = {"state": "bound", "revision": "a"}
    after = {"state": "bound", "revision": "b"}
    assert _source_integrity_check(before, after)["status"] == "fail"
    assert _source_integrity_check(before, before)["status"] == "pass"
