from __future__ import annotations

import json
import signal
from dataclasses import replace
from pathlib import Path

import pytest

from wmloop.execute import pipeline_daemon
from wmloop.execute.autonomous_pipeline import AutonomousPipelineOptions
from wmloop.execute.gpu_lease import GpuLeaseError
from wmloop.execute.pipeline_daemon import (
    PipelineDaemonError,
    PipelineDaemonOptions,
)


def _options(tmp_path: Path, **overrides: object) -> PipelineDaemonOptions:
    repo = tmp_path / "model"
    repo.mkdir(exist_ok=True)
    evaluator = tmp_path / "evaluator.json"
    if not evaluator.exists():
        evaluator.write_text('{"evaluator_id":"test"}\n', encoding="utf-8")
    pipeline = AutonomousPipelineOptions(
        repo_root=repo,
        output_root=tmp_path / "pipeline",
        evaluator_contract=evaluator,
        probe_imports=False,
    )
    values: dict[str, object] = {
        "pipeline": pipeline,
        "state_root": tmp_path / "daemon",
        "poll_seconds": 0.0,
        "max_cycles": 2,
        "max_attempts": 2,
    }
    values.update(overrides)
    return PipelineDaemonOptions(**values)  # type: ignore[arg-type]


def _settled_manifest(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "verdiwm-autonomous-pipeline-manifest",
        "state": "settled",
        "verdict": "PASS",
        "blocked_stage": None,
    }
    (output_root / "pipeline-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_gpu_deferral_is_retried_without_consuming_failure_attempts(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, max_attempts=1)
    calls = 0

    def runner(pipeline: AutonomousPipelineOptions) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GpuLeaseError("GPU_LEASE_UNAVAILABLE:4:COMPUTE_APP_PRESENT")
        return _settled_manifest(Path(pipeline.output_root))

    manifest = pipeline_daemon.run_pipeline_daemon(
        options,
        pipeline_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "completed"
    assert manifest["attempt_count"] == 2
    assert manifest["deferral_count"] == 1
    assert manifest["error_count"] == 0
    first_cycle = json.loads(
        (tmp_path / "daemon/cycles/cycle-000001.json").read_text(encoding="utf-8")
    )
    assert first_cycle["outcome"] == "deferred"


def test_non_resource_errors_are_bounded(tmp_path: Path) -> None:
    calls = 0

    def runner(_pipeline: AutonomousPipelineOptions) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic pipeline failure")

    manifest = pipeline_daemon.run_pipeline_daemon(
        _options(tmp_path, max_cycles=8, max_attempts=2),
        pipeline_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "blocked"
    assert manifest["error_count"] == 2
    assert manifest["deferral_count"] == 0
    assert calls == 2


def test_pipeline_policy_block_is_terminal_without_retry(tmp_path: Path) -> None:
    calls = 0

    def runner(_pipeline: AutonomousPipelineOptions) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "state": "blocked",
            "verdict": "BLOCKED",
            "blocked_stage": "diagnostic_probe",
        }

    manifest = pipeline_daemon.run_pipeline_daemon(
        _options(tmp_path, max_cycles=10),
        pipeline_runner=runner,
        sleeper=lambda _: None,
    )

    assert manifest["state"] == "blocked"
    assert manifest["blocked_stage"] == "diagnostic_probe"
    assert manifest["error_count"] == 0
    assert calls == 1


def test_exhausted_resource_wait_resumes_with_larger_cycle_bound(
    tmp_path: Path,
) -> None:
    first_options = _options(tmp_path, max_cycles=1)

    def busy(_pipeline: AutonomousPipelineOptions) -> dict[str, object]:
        raise GpuLeaseError("GPU_LEASE_UNAVAILABLE:0:MEMORY_BUSY")

    first = pipeline_daemon.run_pipeline_daemon(
        first_options,
        pipeline_runner=busy,
        sleeper=lambda _: None,
    )
    assert first["state"] == "exhausted"
    assert first["deferral_count"] == 1

    second = pipeline_daemon.run_pipeline_daemon(
        replace(first_options, max_cycles=2),
        pipeline_runner=lambda pipeline: _settled_manifest(
            Path(pipeline.output_root)
        ),
        sleeper=lambda _: None,
    )
    assert second["state"] == "completed"
    assert second["cycle"] == 2
    assert second["deferral_count"] == 1


def test_changed_evaluator_is_rejected_on_resume(tmp_path: Path) -> None:
    options = _options(tmp_path, max_cycles=1)
    pipeline_daemon.run_pipeline_daemon(
        options,
        pipeline_runner=lambda pipeline: _settled_manifest(
            Path(pipeline.output_root)
        ),
        sleeper=lambda _: None,
    )
    Path(options.pipeline.evaluator_contract).write_text(
        '{"evaluator_id":"changed"}\n', encoding="utf-8"
    )

    with pytest.raises(PipelineDaemonError, match="INPUT_MISMATCH"):
        pipeline_daemon.run_pipeline_daemon(
            options,
            pipeline_runner=lambda _: {},
            sleeper=lambda _: None,
        )


def test_changed_retry_policy_requires_new_state_root(tmp_path: Path) -> None:
    options = _options(tmp_path, max_cycles=1, max_attempts=2)
    pipeline_daemon.run_pipeline_daemon(
        options,
        pipeline_runner=lambda pipeline: _settled_manifest(
            Path(pipeline.output_root)
        ),
        sleeper=lambda _: None,
    )

    with pytest.raises(PipelineDaemonError, match="INPUT_MISMATCH"):
        pipeline_daemon.run_pipeline_daemon(
            replace(options, max_attempts=3),
            pipeline_runner=lambda _: {},
            sleeper=lambda _: None,
        )


def test_stop_signal_is_persisted_after_current_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: dict[str, object] = {}

    def install(handler: object) -> dict[int, object]:
        installed["handler"] = handler
        return {}

    monkeypatch.setattr(pipeline_daemon, "_install_signal_handlers", install)

    def busy(_pipeline: AutonomousPipelineOptions) -> dict[str, object]:
        raise GpuLeaseError("GPU_LEASE_UNAVAILABLE:1:COMPUTE_APP_PRESENT")

    def sleeper(_seconds: float) -> None:
        handler = installed["handler"]
        assert callable(handler)
        handler(signal.SIGTERM, None)

    manifest = pipeline_daemon.run_pipeline_daemon(
        _options(tmp_path, max_cycles=3),
        pipeline_runner=busy,
        sleeper=sleeper,
    )

    assert manifest["state"] == "stopped"
    assert manifest["cycle"] == 1
    status = json.loads(
        (tmp_path / "daemon/status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "stopped"


def test_daemon_state_cannot_overlap_pipeline_or_model(tmp_path: Path) -> None:
    options = _options(tmp_path)
    with pytest.raises(PipelineDaemonError, match="STATE_OVERLAP_INVALID"):
        pipeline_daemon.run_pipeline_daemon(
            replace(options, state_root=Path(options.pipeline.output_root) / "daemon"),
            pipeline_runner=lambda _: {},
            sleeper=lambda _: None,
        )
