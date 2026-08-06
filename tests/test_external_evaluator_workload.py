from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from wmloop.execute.external_evaluator_workload import (
    ExternalEvaluatorError,
    ExternalEvaluatorOptions,
    run_external_evaluator,
)


def test_external_evaluator_materializes_standard_result(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    command = (
        sys.executable,
        "-c",
        "from pathlib import Path; import os; "
        "root=Path(os.environ['VERDIWM_TRIAL_SCRATCH']); "
        "(root/'dynamic').mkdir(parents=True); "
        "(root/'dynamic'/'rollout-001.mp4').write_bytes(b'video')",
    )
    environment = {
        **os.environ,
        "VERDIWM_TRIAL_SCRATCH": str(scratch),
        "VERDIWM_PHYSICAL_GPU_INDEX": "4",
        "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-4",
    }

    exit_code = run_external_evaluator(
        ExternalEvaluatorOptions(
            command=command,
            scratch_root=scratch,
            working_directory=tmp_path,
            artifacts=("dynamic/*.mp4=rollout.mp4",),
        ),
        environment=environment,
        identity_provider=lambda _: {"gpu_uuid": "GPU-test-4", "name": "test"},
    )

    assert exit_code == 0
    assert (scratch / "rollout.mp4").read_bytes() == b"video"
    result = json.loads((scratch / "result.json").read_text(encoding="utf-8"))
    assert result["artifact_type"] == "verdiwm-auto-experiment-result"
    assert result["device"]["gpu_uuid"] == "GPU-test-4"
    assert result["metrics"]["runtime_ready"] == 1.0
    index = json.loads((scratch / "artifact-index.json").read_text(encoding="utf-8"))
    assert index["artifacts"][0]["target_path"] == "rollout.mp4"


def test_external_failure_writes_invalid_result(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    environment = {
        **os.environ,
        "VERDIWM_PHYSICAL_GPU_INDEX": "0",
        "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-0",
    }

    exit_code = run_external_evaluator(
        ExternalEvaluatorOptions(
            command=(sys.executable, "-c", "raise SystemExit(7)"),
            scratch_root=scratch,
            working_directory=tmp_path,
            artifacts=(),
        ),
        environment=environment,
        identity_provider=lambda _: {"gpu_uuid": "GPU-test-0", "name": "test"},
    )

    assert exit_code == 7
    result = json.loads((scratch / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "invalid"
    assert result["metrics"]["external_exit_code"] == 7.0


def test_artifact_target_cannot_escape_scratch(tmp_path: Path) -> None:
    with pytest.raises(ExternalEvaluatorError, match="PATH_ESCAPE"):
        run_external_evaluator(
            ExternalEvaluatorOptions(
                command=(sys.executable, "-c", "pass"),
                scratch_root=tmp_path / "scratch",
                working_directory=tmp_path,
                artifacts=("*.mp4=../outside.mp4",),
            ),
            environment={
                "VERDIWM_PHYSICAL_GPU_INDEX": "0",
                "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-0",
            },
            identity_provider=lambda _: {
                "gpu_uuid": "GPU-test-0",
                "name": "test",
            },
        )


def test_external_python_bytecode_is_isolated_from_source(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    source = tmp_path / "source"
    source.mkdir()
    (source / "probe_module.py").write_text("VALUE = 7\n", encoding="utf-8")
    command = (
        sys.executable,
        "-c",
        "import json, os, probe_module; "
        "from pathlib import Path; "
        "root=Path(os.environ['VERDIWM_TRIAL_SCRATCH']); "
        "(root/'environment.json').write_text(json.dumps({"
        "'dont_write': os.environ.get('PYTHONDONTWRITEBYTECODE'), "
        "'cache_prefix': os.environ.get('PYTHONPYCACHEPREFIX'), "
        "'value': probe_module.VALUE}))",
    )
    environment = {
        **os.environ,
        "VERDIWM_TRIAL_SCRATCH": str(scratch),
        "VERDIWM_PHYSICAL_GPU_INDEX": "2",
        "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-2",
    }

    exit_code = run_external_evaluator(
        ExternalEvaluatorOptions(
            command=command,
            scratch_root=scratch,
            working_directory=source,
            artifacts=("environment.json=environment-bound.json",),
        ),
        environment=environment,
        identity_provider=lambda _: {"gpu_uuid": "GPU-test-2", "name": "test"},
    )

    assert exit_code == 0
    assert not (source / "__pycache__").exists()
    observed = json.loads(
        (scratch / "environment-bound.json").read_text(encoding="utf-8")
    )
    assert observed == {
        "cache_prefix": str(scratch / "pycache"),
        "dont_write": "1",
        "value": 7,
    }


def test_external_evaluator_merges_declared_numeric_metrics(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    command = (
        sys.executable,
        "-c",
        "import json, os; from pathlib import Path; "
        "root=Path(os.environ['VERDIWM_TRIAL_SCRATCH']); "
        "(root/'diagnostic.json').write_text(json.dumps({"
        "'state':'ready','metrics':{'paired_l1_mean':3.5,'frame_count':5}}))",
    )
    environment = {
        **os.environ,
        "VERDIWM_TRIAL_SCRATCH": str(scratch),
        "VERDIWM_PHYSICAL_GPU_INDEX": "1",
        "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-1",
    }

    exit_code = run_external_evaluator(
        ExternalEvaluatorOptions(
            command=command,
            scratch_root=scratch,
            working_directory=tmp_path,
            artifacts=("diagnostic.json=diagnostic.json",),
            metrics_path="diagnostic.json",
        ),
        environment=environment,
        identity_provider=lambda _: {"gpu_uuid": "GPU-test-1", "name": "test"},
    )

    assert exit_code == 0
    result = json.loads((scratch / "result.json").read_text(encoding="utf-8"))
    assert result["metrics"]["paired_l1_mean"] == 3.5
    assert result["metrics"]["frame_count"] == 5.0


def test_external_metrics_cannot_override_runtime_evidence(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    command = (
        sys.executable,
        "-c",
        "import json, os; from pathlib import Path; "
        "root=Path(os.environ['VERDIWM_TRIAL_SCRATCH']); "
        "(root/'diagnostic.json').write_text(json.dumps({"
        "'state':'ready','metrics':{'runtime_ready':1}}))",
    )

    exit_code = run_external_evaluator(
        ExternalEvaluatorOptions(
            command=command,
            scratch_root=scratch,
            working_directory=tmp_path,
            artifacts=("diagnostic.json=diagnostic.json",),
            metrics_path="diagnostic.json",
        ),
        environment={
            **os.environ,
            "VERDIWM_TRIAL_SCRATCH": str(scratch),
            "VERDIWM_PHYSICAL_GPU_INDEX": "0",
            "VERDIWM_PHYSICAL_GPU_UUID": "GPU-test-0",
        },
        identity_provider=lambda _: {"gpu_uuid": "GPU-test-0", "name": "test"},
    )

    assert exit_code == 2
    result = json.loads((scratch / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "invalid"
    assert result["metrics"]["runtime_ready"] == 0.0
