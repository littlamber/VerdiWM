from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from wmloop.execute import autonomous_pipeline
from wmloop.execute.autonomous_pipeline import (
    AutonomousPipelineError,
    AutonomousPipelineOptions,
    run_autonomous_pipeline,
)
from wmloop.execute.experiment_scheduler import ExperimentSchedulerError


def test_pipeline_runs_and_resumes_from_bound_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _model_repo(tmp_path / "model")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json",
        template=_candidate_template(tmp_path / "candidate.json"),
    )
    calls: list[Path] = []

    def fake_run_selected_queue(**kwargs: object) -> dict[str, object]:
        calls.append(Path(str(kwargs["queue_path"])))
        return {"candidate_states": {"external-smoke": "completed"}}

    monkeypatch.setattr(
        autonomous_pipeline, "run_selected_queue", fake_run_selected_queue
    )
    options = AutonomousPipelineOptions(
        repo_root=repo,
        output_root=tmp_path / "pipeline",
        evaluator_contract=evaluator,
        runtime_python=Path(sys.executable),
        probe_imports=False,
    )

    first = run_autonomous_pipeline(options)
    second = run_autonomous_pipeline(options)

    assert first["verdict"] == "PASS"
    assert second == first
    assert len(calls) == 2
    assert (tmp_path / "pipeline" / "pipeline-input.lock.json").is_file()
    assert (tmp_path / "pipeline" / "onboarding" / "manifest.json").is_file()
    assert (tmp_path / "pipeline" / "conformance" / "manifest.json").is_file()
    assert (tmp_path / "pipeline" / "compiled" / "manifest.json").is_file()


def test_pipeline_stops_before_gpu_when_conformance_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _model_repo(tmp_path / "model")
    (repo / "broken.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json",
        template=_candidate_template(tmp_path / "candidate.json"),
        imports=["broken"],
    )

    def unexpected_execution(**_: object) -> dict[str, object]:
        raise AssertionError("GPU scheduler must not run")

    monkeypatch.setattr(autonomous_pipeline, "run_selected_queue", unexpected_execution)
    manifest = run_autonomous_pipeline(
        AutonomousPipelineOptions(
            repo_root=repo,
            output_root=tmp_path / "pipeline",
            evaluator_contract=evaluator,
            runtime_python=Path(sys.executable),
            probe_imports=False,
        )
    )

    assert manifest["state"] == "blocked"
    assert manifest["blocked_stage"] == "conformance"
    assert not (tmp_path / "pipeline" / "compiled").exists()


def test_pipeline_rejects_changed_inputs_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _model_repo(tmp_path / "model")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json",
        template=_candidate_template(tmp_path / "candidate.json"),
    )
    monkeypatch.setattr(
        autonomous_pipeline,
        "run_selected_queue",
        lambda **_: {"candidate_states": {"external-smoke": "completed"}},
    )
    base = dict(
        repo_root=repo,
        output_root=tmp_path / "pipeline",
        evaluator_contract=evaluator,
        runtime_python=Path(sys.executable),
        probe_imports=False,
    )
    run_autonomous_pipeline(AutonomousPipelineOptions(**base))

    with pytest.raises(AutonomousPipelineError, match="INPUT_MISMATCH"):
        run_autonomous_pipeline(
            AutonomousPipelineOptions(
                **base,
                conformance_timeout_seconds=31.0,
            )
        )


def test_pipeline_persists_interruption_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _model_repo(tmp_path / "model")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json",
        template=_candidate_template(tmp_path / "candidate.json"),
    )
    attempts = 0

    def flaky_execution(**_: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ExperimentSchedulerError("synthetic interruption")
        return {"candidate_states": {"external-smoke": "completed"}}

    monkeypatch.setattr(autonomous_pipeline, "run_selected_queue", flaky_execution)
    options = AutonomousPipelineOptions(
        repo_root=repo,
        output_root=tmp_path / "pipeline",
        evaluator_contract=evaluator,
        runtime_python=Path(sys.executable),
        probe_imports=False,
    )

    with pytest.raises(ExperimentSchedulerError, match="synthetic interruption"):
        run_autonomous_pipeline(options)
    interrupted = json.loads(
        (tmp_path / "pipeline" / "pipeline-manifest.json").read_text(encoding="utf-8")
    )
    assert interrupted["state"] == "interrupted"
    assert interrupted["blocked_stage"] == "execution"

    assert run_autonomous_pipeline(options)["verdict"] == "PASS"


def test_pipeline_stages_typed_methods_before_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _model_repo(tmp_path / "model")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json",
        template=_candidate_template(tmp_path / "candidate.json"),
    )

    def fake_literature_retrieval(**kwargs: object) -> dict[str, object]:
        destination = Path(str(kwargs["output_root"]))
        manifest = {
            "artifact_type": "verdiwm-literature-retrieval-manifest",
            "state": "network",
            "rows": [],
        }
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def fake_method_staging(**kwargs: object) -> dict[str, object]:
        destination = Path(str(kwargs["output_root"]))
        manifest = {
                "artifact_type": "wmloop-literature-method-staging-manifest",
                "state": "ready",
                "records": [],
                "work_order_paths": {},
        }
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(
        autonomous_pipeline,
        "run_literature_retrieval",
        fake_literature_retrieval,
    )
    monkeypatch.setattr(
        autonomous_pipeline,
        "run_literature_method_staging",
        fake_method_staging,
    )
    monkeypatch.setattr(
        autonomous_pipeline,
        "run_selected_queue",
        lambda **_: {"candidate_states": {"external-smoke": "completed"}},
    )
    result = run_autonomous_pipeline(
        AutonomousPipelineOptions(
            repo_root=repo,
            output_root=tmp_path / "pipeline",
            evaluator_contract=evaluator,
            runtime_python=Path(sys.executable),
            probe_imports=False,
            literature_query="world model drift",
        )
    )

    assert result["verdict"] == "PASS"
    assert result["literature_methods"]["state"] == "ready"
    compilation = json.loads(
        (tmp_path / "pipeline" / "compiled" / "manifest.json").read_text()
    )
    assert compilation["literature_method_manifest_path"].endswith(
        "literature-methods/manifest.json"
    )


def _model_repo(repo: Path) -> Path:
    (repo / "scripts").mkdir(parents=True)
    (repo / "requirements.txt").write_text("\n", encoding="utf-8")
    (repo / "checkpoint.pt").write_bytes(b"checkpoint")
    (repo / "scripts" / "eval_rollout.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--ckpt-path')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    return repo


def _evaluator_contract(
    path: Path,
    *,
    template: Path,
    imports: list[str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluator_id": "synthetic_eval_v1",
                "command": [
                    "{python}",
                    "scripts/eval_rollout.py",
                    "--ckpt-path",
                    "{asset:--ckpt-path}",
                ],
                "input_artifacts": ["checkpoint"],
                "output_artifacts": ["result.json"],
                "metrics": ["runtime_ready"],
                "verifier": "synthetic_receipt_v1",
                "conformance_imports": imports or ["json"],
                "scheduler_template": str(template),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _candidate_template(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-candidate-batch",
        "campaign_id": "synthetic-autonomous-v1",
        "objective": "Verify one bounded external runtime through the autonomous pipeline.",
        "selection_reason": "The declared smoke is the cheapest valid wiring check for this external model.",
        "falsification_criterion": "Any admission, runtime, GPU identity, or artifact failure blocks execution.",
        "total_budget_gpu_hours": 0.01,
        "max_selected_candidates": 1,
        "scoring": {
            "expected_gain_weight": 0.5,
            "uncertainty_weight": 0.2,
            "information_gain_weight": 0.2,
            "novelty_weight": 0.1,
            "cost_weight": 0.5,
        },
        "candidates": [
            {
                "candidate_id": "external-smoke",
                "hypothesis": "The admitted external evaluator can complete on one leased GPU.",
                "selection_reason": "A single screen validates wiring before costlier optimization.",
                "falsification_criterion": "A non-passing runtime metric voids this candidate.",
                "expected_gain": 0.1,
                "uncertainty": 0.5,
                "information_gain": 0.8,
                "novelty": 0.2,
                "stages": [
                    {
                        "stage": "screen",
                        "command": ["{verdiwm_python}", "-c", "print('screen')"],
                        "working_directory": ".",
                        "allowed_gpu_indices": [0],
                        "estimated_gpu_hours": 0.005,
                        "timeout_seconds": 30,
                        "gpu_wait_seconds": 0,
                        "sample_interval_seconds": 0.2,
                        "result_path": "result.json",
                        "artifacts": ["result.json"],
                        "metric_gates": [
                            {
                                "metric": "runtime_ready",
                                "role": "primary",
                                "operator": "gte",
                                "threshold": 1.0,
                            }
                        ],
                        "environment": {
                            "MODEL_CHECKPOINT": "{asset:--ckpt-path}",
                            "VERDIWM_RESULT_ROOT": "{scratch_dir}",
                        },
                        "cleanup_policy": "archive_then_delete",
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path
