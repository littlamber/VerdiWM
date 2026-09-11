import json
from pathlib import Path

import pytest

from experiments.ctrl_world_autonomous_transfer_v1.workflow_stages.materialization import (
    calibrate_open_method,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    compilation = tmp_path / "compilation"
    _write(
        compilation / "manifest.json",
        {"state": "ready_for_calibration", "method_id": "method-fixture", "overlay_id": "overlay-fixture"},
    )
    _write(compilation / "method-ir.json", {"method_id": "method-fixture"})
    _write(compilation / "candidate-overlay.json", {"overlay_id": "overlay-fixture"})
    work = {
        "work_id": "work-fixture",
        "context": {"open_method_compilation_root": str(compilation)},
    }
    config = {
        "paths": {"project_root": str(tmp_path)},
        "open_method_generation": {"llm_adapter": {}},
    }
    return config, work, compilation


def test_open_method_calibration_skips_optional_resource_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, work, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        "experiments.ctrl_world_autonomous_transfer_v1.workflow_stages.materialization.run_method_calibration",
        lambda **_: {"state": "passed", "method_id": "method-fixture", "overlay_id": "overlay-fixture"},
    )

    result = calibrate_open_method(config, work=work, attempt_root=tmp_path / "attempt-001")

    assert result.payload["open_method_calibration_next_state"] == "pending_screen"


def test_open_method_calibration_uses_resource_admission_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, work, _ = _fixture(tmp_path)
    config["resource_portfolio"] = {"policy_id": "fixture-policy"}
    monkeypatch.setattr(
        "experiments.ctrl_world_autonomous_transfer_v1.workflow_stages.materialization.run_method_calibration",
        lambda **_: {"state": "passed", "method_id": "method-fixture", "overlay_id": "overlay-fixture"},
    )

    result = calibrate_open_method(config, work=work, attempt_root=tmp_path / "attempt-001")

    assert result.payload["open_method_calibration_next_state"] == "pending_resource_admission"


def test_open_method_calibration_failure_is_archived_without_closed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, work, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        "experiments.ctrl_world_autonomous_transfer_v1.workflow_stages.materialization.run_method_calibration",
        lambda **_: {"state": "failed", "method_id": "method-fixture", "overlay_id": "overlay-fixture"},
    )

    result = calibrate_open_method(config, work=work, attempt_root=tmp_path / "attempt-001")

    assert result.state == "blocked"
    assert result.payload["open_method_calibration_next_state"] == "pending_knowledge"
