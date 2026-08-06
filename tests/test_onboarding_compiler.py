from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from wmloop.control.onboarding import OnboardingOptions, run_onboarding
from wmloop.control.onboarding_compiler import (
    OnboardingCompilerError,
    _apply_diagnostic_routing,
    _settle_queue_admission,
    compile_and_plan,
)
from wmloop.control.onboarding_conformance import (
    ConformanceOptions,
    ModelConformanceError,
    run_conformance,
)
from wmloop.execute import experiment_scheduler


def test_passing_conformance_compiles_a_hash_bound_queue(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    template = _candidate_template(tmp_path / "template.json")
    evaluator = _evaluator_contract(tmp_path / "evaluator.json", template=template)
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )
    conformance = tmp_path / "conformance"
    assert (
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=conformance)
        )["verdict"]
        == "PASS"
    )

    output = tmp_path / "compiled"
    manifest = compile_and_plan(
        sidecar_root=sidecar,
        conformance_root=conformance,
        output_root=output,
    )

    assert manifest["optimization_launch_allowed"] is True
    batch = json.loads((output / "candidate-batch.json").read_text(encoding="utf-8"))
    assert batch["onboarding_admission"]["receipt_path"].endswith(
        "conformance-receipt.json"
    )
    assert batch["candidates"][0]["stages"][0]["environment"][
        "MODEL_CHECKPOINT"
    ].endswith("checkpoint.pt")
    assert Path(batch["candidates"][0]["stages"][0]["command"][0]).samefile(
        sys.executable
    )
    queue = json.loads((output / "queue" / "queue.json").read_text(encoding="utf-8"))
    assert queue["state"] == "ready"
    plan = json.loads(
        (output / "queue" / queue["selected"][0]["stages"][0]["plan_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert plan["onboarding_admission"] == batch["onboarding_admission"]


def test_diagnostic_routing_blocks_candidates_without_signature_overlap() -> None:
    batch = {
        "candidates": [
            {
                "candidate_id": "unrouted",
                "retrieval_keys": {"failure_signatures": ["action_binding"]},
            },
            {
                "candidate_id": "matched",
                "retrieval_keys": {"failure_signatures": ["horizon_drift"]},
            },
        ]
    }

    _apply_diagnostic_routing(
        batch, probe={"failure_signatures": ["horizon_drift"]}
    )

    assert batch["candidates"][0]["routing_admission"]["state"] == "blocked"
    assert batch["candidates"][1]["routing_admission"] == {
        "state": "eligible",
        "reason": "candidate_declares_observed_failure_signature",
        "matched_failure_signatures": ["horizon_drift"],
    }


def test_compilation_disables_launch_when_every_route_is_blocked(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "compiled"
    destination.mkdir()
    settled = _settle_queue_admission(
        destination,
        manifest={"state": "ready", "optimization_launch_allowed": True},
        queue={
            "selected": [],
            "routing_blocked": [{"candidate_id": "unrouted"}],
        },
    )

    assert settled["state"] == "blocked"
    assert settled["optimization_launch_allowed"] is False
    assert settled["blockers"] == ["NO_CANDIDATE_ROUTING_ELIGIBLE"]


def test_blocked_conformance_cannot_compile(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    (repo / "broken.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    template = _candidate_template(tmp_path / "template.json")
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json", template=template, imports=["broken"]
    )
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )
    conformance = tmp_path / "conformance"
    assert (
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=conformance)
        )["verdict"]
        == "BLOCKED"
    )

    with pytest.raises(OnboardingCompilerError, match="ADMISSION_INVALID"):
        compile_and_plan(
            sidecar_root=sidecar,
            conformance_root=conformance,
            output_root=tmp_path / "compiled",
        )


def test_external_scheduler_workspace_requires_conformance(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    template = _candidate_template(tmp_path / "template.json")

    with pytest.raises(
        experiment_scheduler.ExperimentSchedulerError,
        match="ONBOARDING_ADMISSION_INVALID",
    ):
        experiment_scheduler.plan_candidate_batch(
            batch_path=template,
            output_root=tmp_path / "queue",
            workspace_root=repo,
        )


def test_tracked_source_drift_after_pass_cannot_compile(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=VerdiWM Test",
            "-c",
            "user.email=verdiwm-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    template = _candidate_template(tmp_path / "template.json")
    evaluator = _evaluator_contract(tmp_path / "evaluator.json", template=template)
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )
    conformance = tmp_path / "conformance"
    assert (
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=conformance)
        )["verdict"]
        == "PASS"
    )

    (repo / "scripts" / "eval_rollout.py").write_text(
        "raise RuntimeError('changed after conformance')\n", encoding="utf-8"
    )

    with pytest.raises(OnboardingCompilerError, match="ADMISSION_SOURCE_TREE_DRIFT"):
        compile_and_plan(
            sidecar_root=sidecar,
            conformance_root=conformance,
            output_root=tmp_path / "compiled",
        )


def test_asset_drift_after_pass_cannot_compile_or_resume(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    template = _candidate_template(tmp_path / "template.json")
    evaluator = _evaluator_contract(tmp_path / "evaluator.json", template=template)
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )
    conformance = tmp_path / "conformance"
    assert (
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=conformance)
        )["verdict"]
        == "PASS"
    )
    (repo / "checkpoint.pt").write_bytes(b"replacement-checkpoint")

    with pytest.raises(ModelConformanceError, match="ASSET_DRIFT"):
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=conformance)
        )
    with pytest.raises(OnboardingCompilerError, match="ADMISSION_ASSET_DRIFT"):
        compile_and_plan(
            sidecar_root=sidecar,
            conformance_root=conformance,
            output_root=tmp_path / "compiled",
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
    path: Path, *, template: Path, imports: list[str] | None = None
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
                "metrics": ["success_rate"],
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
        "campaign_id": "synthetic-onboarded-v1",
        "objective": "Verify a bounded external model evaluation through the admitted scheduler.",
        "selection_reason": "The frozen evaluator template is the cheapest valid external runtime candidate.",
        "falsification_criterion": "Any source drift, admission mismatch, missing result, or metric failure blocks the candidate.",
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
                "hypothesis": "The external evaluator can produce its declared runtime result on one leased GPU.",
                "selection_reason": "A single bounded screen proves wiring before any optimization campaign.",
                "falsification_criterion": "A missing artifact or non-passing runtime metric voids this smoke candidate.",
                "expected_gain": 0.1,
                "uncertainty": 0.5,
                "information_gain": 0.8,
                "novelty": 0.2,
                "stages": [
                    {
                        "stage": "screen",
                        "command": [
                            "{verdiwm_python}",
                            "-c",
                            "print('not executed by compiler')",
                        ],
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
