"""Bounded actual CPU imports and process cleanup for external-model onboarding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from wmloop.control import onboarding
from wmloop.control.onboarding_scan import runtime
from wmloop.control.onboarding_scan.types import (
    DEFAULT_RUNTIME_TIMEOUT_SECONDS,
    OnboardingError,
    OnboardingOptions,
)


@pytest.mark.parametrize("value", [0, -1, 301, float("nan"), float("inf"), True, "30", None])
def test_invalid_timeout_is_rejected_before_scan_or_output(tmp_path, value):
    with pytest.raises(OnboardingError, match="ONBOARDING_RUNTIME_TIMEOUT_INVALID"):
        onboarding.run_onboarding(OnboardingOptions(
            repo_root=tmp_path, output_root=tmp_path / "output",
            runtime_timeout_seconds=value,
        ))
    assert not (tmp_path / "output").exists()


def test_runtime_scan_records_and_applies_configured_timeout(tmp_path, monkeypatch):
    (tmp_path / "slow_fixture.py").write_text(
        "import os, time\nassert os.environ['CUDA_VISIBLE_DEVICES'] == ''\ntime.sleep(0.2)\n"
    )
    monkeypatch.setattr(runtime, "_runtime_candidates", lambda *args: [
        {"path": sys.executable, "source": "explicit", "exists": True, "version": "fixture"}
    ])
    pip_timeouts = []

    def pip_probe(path_info, *, repo, timeout_seconds):
        pip_timeouts.append(timeout_seconds)
        return {"state": "ok", "returncode": 0, "detail": "CPU fixture"}

    monkeypatch.setattr(runtime, "_probe_pip", pip_probe)
    dependencies = [{"declared_packages": ["slow_fixture"]}]
    short = runtime._discover_runtime(
        tmp_path, dependencies, runtime_python=Path(sys.executable),
        probe_imports=True, runtime_timeout_seconds=0.05,
    )
    assert short["state"] == "blocked"
    assert short["import_probes"][0]["state"] == "timeout"
    normal = runtime._discover_runtime(
        tmp_path, dependencies, runtime_python=Path(sys.executable),
        probe_imports=True, runtime_timeout_seconds=2,
    )
    assert normal["state"] == "ready"
    assert normal["import_probes"][0]["state"] == "ok"
    assert normal["probe_timeout_seconds"] == 2
    assert pip_timeouts == [0.05, 2]


def test_public_scan_passes_timeout_to_runtime_and_schema(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("json\n")
    timeouts = []
    original = runtime._run_runtime_command

    def command(argv, *, timeout_seconds, cwd, env):
        timeouts.append(timeout_seconds)
        if "pip" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return original(argv, timeout_seconds=timeout_seconds, cwd=cwd, env=env)

    monkeypatch.setattr(runtime, "_run_runtime_command", command)
    report = onboarding.scan_repository(OnboardingOptions(
        repo_root=tmp_path, runtime_python=Path(sys.executable), runtime_timeout_seconds=45,
    ))
    assert report["runtime"]["probe_timeout_seconds"] == 45
    assert report["runtime"]["import_probes"][0]["state"] == "ok"
    assert timeouts == [45, 45]
    from wmloop.contracts import validate_document

    validate_document("model_onboarding_report", report)


@pytest.mark.skipif(os.name != "posix", reason="process-group retirement is a Linux/WSL contract")
def test_timeout_retires_import_process_and_its_descendants(tmp_path):
    marker = tmp_path / "descendant-pid"
    (tmp_path / "hanging_fixture.py").write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    rows = runtime._probe_imports(
        {"path": sys.executable}, ["hanging_fixture"], repo=tmp_path, timeout_seconds=0.4,
    )
    assert rows[0]["state"] == "timeout"
    assert marker.is_file()
    pid = int(marker.read_text())
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while status.exists() and status.read_text().split()[2] != "Z" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_onboard_cli_exposes_timeout_and_preserves_stdout_json(tmp_path, monkeypatch, capsys):
    options = []
    monkeypatch.setattr(onboarding, "run_onboarding", lambda value: options.append(value) or {"state": "fixture"})
    assert onboarding.main([str(tmp_path), "--runtime-timeout-seconds", "60"]) == 0
    assert options[0].runtime_timeout_seconds == 60
    assert json.loads(capsys.readouterr().out) == {"state": "fixture"}
    assert DEFAULT_RUNTIME_TIMEOUT_SECONDS == 30
