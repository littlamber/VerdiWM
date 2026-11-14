"""Exercise durable evolution stops without launching model workloads."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from wmloop.execute import evolution_daemon
from wmloop.execute.backends import LocalSubprocessBackend
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.evolution_daemon import (
    EvolutionDaemonOptions,
    run_evolution_daemon,
)
from wmloop.execute.gpu_lease import GpuLeaseError
from wmloop.execute.pipeline_daemon import run_pipeline_daemon
from wmloop.storage import canonical_json_bytes, sha256_bytes


@pytest.fixture
def evolution_options(tmp_path: Path) -> EvolutionDaemonOptions:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# Isolated controller fixture.\n")
    template_path = tmp_path / "candidate-batch.json"
    example = (
        Path(__file__).resolve().parents[1]
        / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json"
    )
    template = json.loads(example.read_text())
    template["candidates"] = template["candidates"][:1]
    for stage in template["candidates"][0]["stages"]:
        stage["command"] = ["python", "-c", "pass"]
        stage["environment"] = {}
    template_path.write_text(json.dumps(template))
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(json.dumps({"scheduler_template": template_path.name}))
    return EvolutionDaemonOptions(
        repo_root=source,
        output_root=tmp_path / "output",
        state_root=tmp_path / "state",
        evaluator_contract=evaluator,
        lock_root=tmp_path / "gpu-locks",
        budget_db=tmp_path / "budget.db",
        poll_seconds=0,
        max_iterations=2,
        inner_max_cycles=1,
    )


def test_exhausted_inner_budget_stays_terminal_on_resume(evolution_options):
    options = evolution_options
    ledger = BudgetLedger(options.budget_db, BudgetPolicy(total_gpu_hours=1.0))
    ledger.admit("existing-reservation", cost_class="low", estimated_gpu_hours=0.9)
    calls = []

    def cannot_admit(_pipeline_options):
        calls.append("admission")
        ledger.admit("next-trial", cost_class="low", estimated_gpu_hours=0.2)

    def nested_runner(daemon_options):
        return run_pipeline_daemon(
            daemon_options, pipeline_runner=cannot_admit, sleeper=lambda _: None
        )

    first = run_evolution_daemon(options, pipeline_runner=nested_runner)
    assert first["state"] == "exhausted"
    assert first["stop_reason"] == "GLOBAL_BUDGET_EXHAUSTED"
    assert first["last_outcome"] == "budget_exhausted"
    assert first["failure_count"] == 0
    assert first["last_error"]["message"] == "GLOBAL_BUDGET_EXHAUSTED"
    resumed = run_evolution_daemon(options, pipeline_runner=nested_runner)
    assert resumed == first
    assert calls == ["admission"]


def test_exception_failure_limit_records_reason_and_does_not_retry(evolution_options):
    options = replace(evolution_options, max_failures=2)
    calls = []

    def fail(_daemon_options):
        calls.append("runner")
        raise RuntimeError("isolated runner failure")

    first = run_evolution_daemon(options, pipeline_runner=fail)
    assert first["state"] == "blocked"
    assert first["failure_count"] == 2
    assert first["stop_reason"] == "CONSECUTIVE_FAILURE_LIMIT"
    assert first["last_error"]["message"] == "isolated runner failure"
    resumed = run_evolution_daemon(options, pipeline_runner=fail)
    assert resumed == first
    assert calls == ["runner", "runner"]


@pytest.mark.parametrize(
    ("send_signal", "state", "reason"),
    [
        (False, "exhausted", "INNER_MAX_CYCLES_REACHED"),
        (True, "stopped", "INNER_PIPELINE_STOPPED"),
    ],
)
def test_inner_capacity_stop_is_explained_and_resumes_same_iteration(
    evolution_options, send_signal, state, reason
):
    options = replace(evolution_options, max_iterations=1)
    runs = []

    def defer(_pipeline_options):
        if send_signal:
            os.kill(os.getpid(), signal.SIGTERM)
        raise GpuLeaseError("GPU_LEASE_UNAVAILABLE:isolated-test")

    def nested_runner(daemon_options):
        runs.append(daemon_options)
        return run_pipeline_daemon(
            daemon_options, pipeline_runner=defer, sleeper=lambda _: None
        )

    first = run_evolution_daemon(options, pipeline_runner=nested_runner)
    assert first["state"] == state
    assert first["stop_reason"] == reason
    assert first["failure_count"] == 0
    assert first["iteration"] == 1
    input_lock = options.output_root / "iterations/iteration-000001/inputs/input.lock.json"
    frozen_input_lock = input_lock.read_bytes()

    def complete(daemon_options):
        runs.append(daemon_options)
        return run_pipeline_daemon(
            daemon_options,
            pipeline_runner=lambda _: {"state": "completed", "verdict": "PASS"},
            sleeper=lambda _: None,
        )

    resumed = run_evolution_daemon(options, pipeline_runner=complete)
    assert resumed["state"] == "completed"
    assert resumed["stop_reason"] == "MAX_ITERATIONS_REACHED"
    assert resumed["iteration"] == 1
    assert resumed["settled_iterations"] == 1
    assert runs[0].state_root == runs[1].state_root
    assert runs[1].max_cycles == 2
    assert input_lock.read_bytes() == frozen_input_lock


@pytest.mark.parametrize(
    ("first_base", "second_base", "first_suffix", "second_suffix"),
    [
        ("candidate" * 16, "candidate" * 16, "run-first-iteration-000001", "run-second-iteration-000001"),
        ("candidate" * 16, "candidate" * 16, "run-first-iteration-000001", "run-first-iteration-000002"),
        ("candidate" * 16 + "first", "candidate" * 16 + "second", "run-first-iteration-000001", "run-first-iteration-000001"),
    ],
)
def test_long_evolution_ids_preserve_controller_iteration_and_base_identity(
    first_base, second_base, first_suffix, second_suffix,
):
    first = evolution_daemon._new_id(first_base, first_suffix)
    second = evolution_daemon._new_id(second_base, second_suffix)
    assert first != second
    assert len(first) <= 128 and len(second) <= 128
    assert first.endswith(first_suffix) and second.endswith(second_suffix)


@pytest.mark.parametrize(
    ("max_iterations", "max_no_information", "expected_count", "reason"),
    [
        (2, 3, 2, "MAX_ITERATIONS_REACHED"),
        (0, 2, 3, "NO_NEW_INFORMATION_LIMIT"),
        (5, 0, 5, "MAX_ITERATIONS_REACHED"),
    ],
)
def test_completed_iterations_stop_at_explicit_policy(
    evolution_options, max_iterations, max_no_information, expected_count, reason
):
    options = replace(
        evolution_options,
        max_iterations=max_iterations,
        max_no_information=max_no_information,
    )
    calls = []

    def complete(daemon_options):
        calls.append(daemon_options)
        pipeline_root = daemon_options.pipeline.output_root
        pipeline_root.mkdir()
        (pipeline_root / "pipeline-manifest.json").write_text(
            json.dumps({"diagnostic_probe": {"failure_signatures": ["test-symptom"]}})
        )
        return {"state": "completed", "pipeline_verdict": "PASS"}

    result = run_evolution_daemon(options, pipeline_runner=complete)
    assert result["state"] == "completed"
    assert result["stop_reason"] == reason
    assert result["settled_iterations"] == expected_count
    assert len(calls) == expected_count
    assert len({call.state_root for call in calls}) == expected_count


@pytest.mark.parametrize("outcomes,expected_state,settled", [
    (("blocked", "blocked"), "blocked", 0),
    (("completed", "blocked"), "blocked", 1),
    (("blocked", "completed"), "completed", 1),
])
def test_round_limit_preserves_unresolved_failure_and_terminal_resume(
    evolution_options, outcomes, expected_state, settled,
):
    calls = []

    def run_iteration(daemon_options):
        outcome = outcomes[len(calls)]
        calls.append(daemon_options)
        if outcome == "completed":
            return {"state": "completed", "pipeline_verdict": "PASS"}
        return {
            "state": "blocked", "pipeline_verdict": "BLOCKED",
            "blocked_stage": "adapter-conformance",
        }

    result = run_evolution_daemon(evolution_options, pipeline_runner=run_iteration)
    assert result["state"] == expected_state
    assert result["settled_iterations"] == settled
    assert result["iteration"] == 2
    assert len(calls) == 2
    if expected_state == "blocked":
        assert result["stop_reason"] == "MAX_ITERATIONS_REACHED_WITH_FAILURES"
        assert result["last_outcome"] == "blocked"
        assert result["failure_count"] == 2 - settled
        assert result["last_error"] == {
            "type": "PIPELINE_BLOCKED", "message": "adapter-conformance",
        }
    else:
        assert result["stop_reason"] == "MAX_ITERATIONS_REACHED"
        assert result["failure_count"] == 0
        assert result["last_error"] is None
    resumed = run_evolution_daemon(
        evolution_options,
        pipeline_runner=lambda _: pytest.fail("terminal policy must not start more work"),
    )
    assert resumed == result


def test_round_limit_cannot_hide_pipeline_exceptions(evolution_options):
    calls = []

    def fail_iteration(_):
        calls.append(True)
        raise RuntimeError("isolated adapter failed")

    result = run_evolution_daemon(evolution_options, pipeline_runner=fail_iteration)
    assert len(calls) == 2
    assert result["state"] == "blocked"
    assert result["stop_reason"] == "MAX_ITERATIONS_REACHED_WITH_FAILURES"
    assert result["settled_iterations"] == 0
    assert result["last_outcome"] == "error"
    assert result["last_error"] == {
        "type": "RuntimeError", "message": "isolated adapter failed",
    }


@pytest.mark.parametrize("outcome", ["completed", "blocked"])
def test_iteration_limit_does_not_sleep_after_final_attempt(evolution_options, outcome):
    options = replace(evolution_options, poll_seconds=60, max_no_information=0)
    waits = []
    result = run_evolution_daemon(
        options,
        pipeline_runner=lambda _: {
            "state": outcome, "pipeline_verdict": "PASS" if outcome == "completed" else "BLOCKED",
        },
        sleeper=waits.append,
    )
    assert result["iteration"] == 2
    assert waits == [60]


def test_no_information_stop_does_not_wait_after_the_terminal_iteration(evolution_options):
    waits = []
    options = replace(evolution_options, max_iterations=0, max_no_information=1, poll_seconds=60)
    result = run_evolution_daemon(
        options, pipeline_runner=lambda _: {"state": "completed", "pipeline_verdict": "PASS"},
        sleeper=waits.append,
    )
    assert result["stop_reason"] == "NO_NEW_INFORMATION_LIMIT"
    assert result["iteration"] == 2
    assert waits == [60]


def test_default_stores_and_total_budget_are_shared_by_every_iteration(evolution_options):
    options = replace(evolution_options, budget_db=None, max_no_information=0)
    calls = []

    def reserve(daemon_options):
        pipeline = daemon_options.pipeline
        calls.append(pipeline)
        ledger = BudgetLedger(
            pipeline.budget_db,
            BudgetPolicy(total_gpu_hours=pipeline.budget_total_gpu_hours),
        )
        ledger.admit(
            f"trial-{len(calls)}", cost_class="low", estimated_gpu_hours=0.6
        )
        return {"state": "completed", "verdict": "PASS"}

    result = run_evolution_daemon(
        options,
        pipeline_runner=lambda daemon_options: run_pipeline_daemon(
            daemon_options, pipeline_runner=lambda _: reserve(daemon_options),
            sleeper=lambda _: None,
        ),
    )
    assert result["stop_reason"] == "GLOBAL_BUDGET_EXHAUSTED"
    assert result["settled_iterations"] == 1
    assert len(calls) == 2
    for name in ("archive_db", "cas_root", "retrieval_db", "budget_db"):
        assert getattr(calls[0], name) == getattr(calls[1], name)
        assert getattr(calls[0], name).parent == options.output_root / "shared"


def test_existing_pipeline_inputs_survive_evolution_and_are_frozen(evolution_options):
    model_irg = evolution_options.repo_root.parent / "model-irg.json"
    model_irg.write_text("{}")
    catalog = evolution_options.repo_root.parent / "catalog.json"
    catalog.write_text("{}")
    settlement = evolution_options.repo_root.parent / "settlement.json"
    settlement.write_text("{}")
    options = replace(
        evolution_options,
        adapter_asset_parameters=("weights",),
        adapter_contract_ready=True,
        runtime_timeout_seconds=17.0,
        model_irg_path=model_irg,
        irg_protected_metrics=("safety",),
        candidate_catalog=catalog,
        settlement_manifest=settlement,
    )
    calls = []

    def complete(daemon_options):
        calls.append(daemon_options.pipeline)
        return {"state": "completed", "pipeline_verdict": "PASS"}

    result = run_evolution_daemon(options, pipeline_runner=complete)
    assert result["settled_iterations"] == 2
    for pipeline in calls:
        for name in (
            "adapter_asset_parameters", "adapter_contract_ready",
            "runtime_timeout_seconds", "model_irg_path", "irg_protected_metrics",
            "candidate_catalog", "settlement_manifest",
        ):
            assert getattr(pipeline, name) == getattr(options, name)
    lock = json.loads((options.state_root / "input.lock.json").read_text())["input"]
    assert lock["adapter_asset_parameters"] == ["weights"]
    assert lock["adapter_contract_ready"] is True
    assert lock["model_irg_sha256"]
    model_irg.write_text('{"changed": true}')
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="EVOLUTION_INPUT_MISMATCH"):
        run_evolution_daemon(options, pipeline_runner=complete)
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["false", 1, None])
def test_stop_on_success_requires_boolean(evolution_options, value):
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="EVOLUTION_ARGUMENT_INVALID"):
        run_evolution_daemon(replace(evolution_options, stop_on_success=value))


def test_execution_pass_does_not_claim_goal_success(evolution_options):
    options = replace(evolution_options, stop_on_success=True, max_no_information=0)
    calls = []

    def complete(daemon_options):
        calls.append(daemon_options)
        return {"state": "completed", "pipeline_verdict": "PASS"}

    result = run_evolution_daemon(options, pipeline_runner=complete)
    assert result["stop_reason"] == "MAX_ITERATIONS_REACHED"
    assert result["goal_success"] == {
        "state": "not_confirmed", "reason": "FROZEN_GOAL_NOT_CONFIGURED",
        "evidence_refs": [],
    }
    assert len(calls) == 2


def test_confirmed_goal_stops_early_and_is_rechecked_on_resume(evolution_options, monkeypatch):
    options = replace(evolution_options, stop_on_success=True)
    calls = []
    confirmation = {
        "state": "confirmed", "reason": "FROZEN_GOAL_CONFIRMED",
        "evidence_refs": ["cas://sha256/" + "a" * 64],
    }
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs)
        return dict(confirmation)

    monkeypatch.setattr(evolution_daemon, "evaluate_iteration_goal_success", evaluate)

    def complete(daemon_options):
        calls.append(daemon_options)
        return {"state": "completed", "pipeline_verdict": "PASS"}

    result = run_evolution_daemon(options, pipeline_runner=complete)
    assert result["stop_reason"] == "GOAL_CONFIRMED"
    assert result["iteration"] == 1
    assert result["goal_success"] == confirmation
    resumed = run_evolution_daemon(options, pipeline_runner=complete)
    assert resumed == result
    assert len(calls) == 1
    assert len(evaluations) == 2
    confirmation.update(
        state="not_confirmed", reason="GOAL_EVIDENCE_INVALID:ValueError", evidence_refs=[]
    )
    invalidated = run_evolution_daemon(options, pipeline_runner=complete)
    assert invalidated["state"] == "blocked"
    assert invalidated["stop_reason"] == "GOAL_CONFIRMATION_INVALIDATED"
    assert len(calls) == 1


def test_goal_baseline_is_persisted_before_work_and_reused_on_resume(evolution_options, monkeypatch):
    options = replace(evolution_options, stop_on_success=True, max_iterations=1)
    old_receipt = "cas://sha256/" + "b" * 64
    captures = []
    evaluations = []
    monkeypatch.setattr(
        evolution_daemon, "capture_settled_receipt_refs",
        lambda path: captures.append(path) or [old_receipt],
    )
    monkeypatch.setattr(
        evolution_daemon, "evaluate_iteration_goal_success",
        lambda **kwargs: evaluations.append(kwargs) or {
            "state": "not_confirmed", "reason": "OLD_RECEIPT_IGNORED", "evidence_refs": [],
        },
    )

    def stop(daemon_options):
        assert len(captures) == 1
        lock = json.loads(
            (daemon_options.pipeline.output_root.parent / "inputs" / "input.lock.json").read_text()
        )
        assert lock["materialized"]["prior_goal_receipt_refs"] == [old_receipt]
        return {"state": "stopped"}

    first = run_evolution_daemon(options, pipeline_runner=stop)
    assert first["state"] == "stopped"
    resumed = run_evolution_daemon(
        options, pipeline_runner=lambda _: {"state": "completed", "pipeline_verdict": "PASS"}
    )
    assert resumed["stop_reason"] == "MAX_ITERATIONS_REACHED"
    assert len(captures) == 1
    assert evaluations[0]["prior_receipt_refs"] == [old_receipt]


def test_missing_goal_baseline_cannot_be_recaptured_after_partial_work(evolution_options):
    options = replace(evolution_options, stop_on_success=True, max_iterations=1)
    run_evolution_daemon(options, pipeline_runner=lambda _: {"state": "stopped"})
    lock_path = options.output_root / "iterations" / "iteration-000001" / "inputs" / "input.lock.json"
    lock = json.loads(lock_path.read_text())
    materialized = lock["materialized"]
    materialized.pop("prior_goal_receipt_refs")
    materialized.pop("input_hash")
    materialized["input_hash"] = sha256_bytes(canonical_json_bytes(materialized))
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="EVOLUTION_GOAL_BASELINE_INVALID"):
        run_evolution_daemon(
            options,
            pipeline_runner=lambda _: pytest.fail("must not execute with a missing baseline"),
        )


def test_corrupt_archive_baseline_fails_before_any_experiment(evolution_options):
    archive = evolution_options.repo_root.parent / "corrupt-archive.db"
    archive.write_bytes(b"invalid sqlite database")
    options = replace(evolution_options, stop_on_success=True, archive_db=archive)
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="GOAL_SUCCESS_BASELINE_INVALID"):
        run_evolution_daemon(
            options, pipeline_runner=lambda _: pytest.fail("baseline failure must prevent execution")
        )
    assert not (options.output_root / "iterations" / "iteration-000001").exists()
    status = json.loads((options.state_root / "status.json").read_text())
    assert status["state"] == "interrupted"
    assert status["last_error"]["message"] == "GOAL_SUCCESS_BASELINE_INVALID"


def test_legacy_per_iteration_store_lock_is_refused_without_rewriting_it(evolution_options):
    options = evolution_options
    run_evolution_daemon(options, pipeline_runner=lambda _: {"state": "stopped"})
    lock_path = options.state_root / "input.lock.json"
    lock = json.loads(lock_path.read_text())
    for name in ("archive_db", "cas_root", "retrieval_db"):
        lock["input"][name] = None
    lock["input_hash"] = sha256_bytes(canonical_json_bytes(lock["input"]))
    lock_path.write_text(json.dumps(lock))
    before = lock_path.read_bytes()
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="EVOLUTION_SHARED_STORE_INPUT_DRIFT"):
        run_evolution_daemon(
            options, pipeline_runner=lambda _: pytest.fail("legacy stores must not be moved")
        )
    assert lock_path.read_bytes() == before


def test_relative_frozen_goal_is_bound_and_cannot_drift(evolution_options):
    goal_path = evolution_options.evaluator_contract.parent / "goal.json"
    goal = {
        "schema_version": 2, "artifact_type": "wmloop-goal-spec",
        "claim_boundary": "Frozen paired target experiment acceptance policy.",
        "goal_id": "frozen-evolution-goal", "metric_family": ["quality"],
        "horizons": [16, 32], "primary_objective": "utility",
        "action_following_gate": {"tau_af": 0.5},
        "eval_protocol": {
            "dev_split": "dev", "accept_split": "accept", "accept_horizon_extra": 16,
            "seeds_accept": [101, 202, 303],
            "acceptance_policy": {
                "metric_directions": {"utility": "maximize"}, "minimum_effect": 0.5,
                "confidence_interval": "student_t_two_sided_95",
            },
        },
        "budget": {"total_gpu_hours": 1, "per_trial_max_gpu_hours": 1}, "envs": ["test"],
    }
    goal_path.write_text(json.dumps(goal))
    evaluator = json.loads(evolution_options.evaluator_contract.read_text())
    evaluator["frozen_goal"] = goal_path.name
    evolution_options.evaluator_contract.write_text(json.dumps(evaluator))
    options = replace(evolution_options, max_iterations=1, stop_on_success=True)
    result = run_evolution_daemon(
        options, pipeline_runner=lambda _: {"state": "completed", "pipeline_verdict": "PASS"}
    )
    assert result["stop_reason"] == "MAX_ITERATIONS_REACHED"
    lock = json.loads((options.state_root / "input.lock.json").read_text())["input"]
    assert lock["frozen_goal"] == str(goal_path)
    assert lock["frozen_goal_sha256"] == sha256_bytes(goal_path.read_bytes())
    generated = json.loads(
        (options.output_root / "iterations" / "iteration-000001" / "inputs" / "evaluator.json").read_text()
    )
    assert generated["frozen_goal"] == str(goal_path)
    goal["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0.1
    goal_path.write_text(json.dumps(goal))
    with pytest.raises(evolution_daemon.EvolutionDaemonError, match="EVOLUTION_INPUT_MISMATCH"):
        run_evolution_daemon(
            options, pipeline_runner=lambda _: pytest.fail("goal drift must be blocked")
        )


def test_cli_preserves_pipeline_options_and_json_stdout(evolution_options, monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(
        evolution_daemon, "run_evolution_daemon",
        lambda options: captured.append(options) or {"state": "completed"},
    )
    result = evolution_daemon.main([
        str(evolution_options.repo_root),
        "--output-root", str(evolution_options.output_root),
        "--state-root", str(evolution_options.state_root),
        "--evaluator-contract", str(evolution_options.evaluator_contract),
        "--total-budget-gpu-hours", "1", "--max-iterations", "2",
        "--max-no-information", "0", "--stop-on-success",
        "--adapter-asset-parameter", "weights", "--adapter-contract-ready",
        "--runtime-timeout-seconds", "17", "--model-irg", "model-irg.json",
        "--irg-protected-metric", "safety", "--candidate-catalog", "catalog.json",
        "--settlement-manifest", "settlement.json",
    ])
    assert result == 0
    assert json.loads(capsys.readouterr().out) == {"state": "completed"}
    options = captured[0]
    assert options.max_iterations == 2
    assert options.max_no_information == 0
    assert options.stop_on_success is True
    assert options.adapter_asset_parameters == ("weights",)
    assert options.adapter_contract_ready is True
    assert options.runtime_timeout_seconds == 17
    assert options.model_irg_path == Path("model-irg.json")
    assert options.irg_protected_metrics == ("safety",)
    assert options.candidate_catalog == Path("catalog.json")
    assert options.settlement_manifest == Path("settlement.json")


def test_budget_text_outside_structured_error_is_not_a_budget_stop(evolution_options):
    options = replace(evolution_options, max_iterations=0, max_failures=1)
    result = run_evolution_daemon(
        options,
        pipeline_runner=lambda _: {
            "state": "blocked",
            "pipeline_verdict": "BLOCKED",
            "blocked_stage": "candidate-GLOBAL_BUDGET_EXHAUSTED-name",
        },
    )
    assert result["state"] == "blocked"
    assert result["stop_reason"] == "CONSECUTIVE_FAILURE_LIMIT"


def test_interrupted_inner_manifest_resumes_current_iteration(evolution_options):
    options = replace(evolution_options, max_iterations=1)
    calls = []

    def interrupt(daemon_options):
        calls.append(daemon_options.state_root)
        return {
            "state": "interrupted",
            "last_error": {"type": "OSError", "message": "isolated inner interruption"},
        }

    first = run_evolution_daemon(options, pipeline_runner=interrupt)
    assert first["state"] == "stopped"
    assert first["stop_reason"] == "INNER_PIPELINE_INTERRUPTED"
    assert first["last_error"]["message"] == "isolated inner interruption"

    def complete(daemon_options):
        calls.append(daemon_options.state_root)
        return {"state": "completed", "pipeline_verdict": "PASS"}

    resumed = run_evolution_daemon(options, pipeline_runner=complete)
    assert resumed["state"] == "completed"
    assert resumed["settled_iterations"] == 1
    assert calls[0] == calls[1]


def test_controller_error_is_persisted_and_resumes_after_completed_iteration(
    evolution_options,
):
    options = evolution_options
    calls = []

    def complete(daemon_options):
        calls.append(daemon_options.state_root)
        return {"state": "completed", "pipeline_verdict": "PASS"}

    def interrupt_sleep(_seconds):
        raise OSError("isolated controller interruption")

    with pytest.raises(OSError, match="isolated controller interruption"):
        run_evolution_daemon(
            options, pipeline_runner=complete, sleeper=interrupt_sleep
        )
    status = json.loads((options.state_root / "status.json").read_text())
    assert status["state"] == "interrupted"
    assert status["stop_reason"] == "CONTROLLER_ERROR"
    assert status["settled_iterations"] == 1
    resumed = run_evolution_daemon(options, pipeline_runner=complete)
    assert resumed["state"] == "completed"
    assert resumed["settled_iterations"] == 2
    assert calls[0] != calls[1]


def test_controller_signal_stops_between_iterations(evolution_options):
    result = run_evolution_daemon(
        evolution_options,
        pipeline_runner=lambda _: {"state": "completed", "pipeline_verdict": "PASS"},
        sleeper=lambda _: os.kill(os.getpid(), signal.SIGTERM),
    )
    assert result["state"] == "stopped"
    assert result["stop_reason"] == "SIGNAL_RECEIVED"
    assert result["iteration"] == 1
    assert result["settled_iterations"] == 1


def test_signal_during_successful_inner_pipeline_stops_before_next_iteration(
    evolution_options, monkeypatch
):
    calls = []

    def finish_after_signal(_pipeline_options):
        calls.append("pipeline")
        os.kill(os.getpid(), signal.SIGTERM)
        return {"state": "completed", "verdict": "PASS"}

    def nested_daemon(daemon_options, **kwargs):
        return run_pipeline_daemon(
            daemon_options,
            pipeline_runner=finish_after_signal,
            sleeper=lambda _: None,
            **kwargs,
        )

    monkeypatch.setattr(evolution_daemon, "run_pipeline_daemon", nested_daemon)
    result = run_evolution_daemon(evolution_options)
    assert result["state"] == "stopped"
    assert result["stop_reason"] == "SIGNAL_RECEIVED"
    assert result["settled_iterations"] == 1
    assert calls == ["pipeline"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_during_local_backend_persists_stopped_controller(
    evolution_options, monkeypatch, signum
):
    workload_pid = evolution_options.repo_root.parent / "workload.pid"

    def run_workload(_pipeline_options):
        finished = threading.Event()

        def signal_once_running():
            deadline = time.monotonic() + 5
            while not finished.is_set() and time.monotonic() < deadline:
                if workload_pid.is_file() and workload_pid.read_text().isdigit():
                    os.kill(os.getpid(), signum)
                    return
                finished.wait(0.01)

        sender = threading.Thread(target=signal_once_running, daemon=True)
        sender.start()
        try:
            LocalSubprocessBackend().run(
                worktree=evolution_options.repo_root,
                command=[
                    sys.executable,
                    "-c",
                    "import os,pathlib,sys,time;"
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
                    "time.sleep(10)",
                    str(workload_pid),
                ],
                environment={},
                timeout_seconds=5,
            )
        finally:
            finished.set()
            sender.join(timeout=1)
        return {"state": "completed", "verdict": "PASS"}

    def nested_daemon(daemon_options, **kwargs):
        return run_pipeline_daemon(
            daemon_options,
            pipeline_runner=run_workload,
            sleeper=lambda _: None,
            **kwargs,
        )

    monkeypatch.setattr(evolution_daemon, "run_pipeline_daemon", nested_daemon)
    try:
        result = run_evolution_daemon(evolution_options)
    except (KeyboardInterrupt, SystemExit):
        pytest.fail("Signal bypassed the durable controller stop state")
    assert result["state"] == "stopped"
    assert result["stop_reason"] == "INNER_PIPELINE_STOPPED"
    assert result["settled_iterations"] == 0
    assert result["failure_count"] == 0
    inner_status = json.loads(
        (Path(result["current_iteration_path"]) / "daemon/status.json").read_text()
    )
    assert inner_status["state"] == "stopped"
    assert inner_status["cycle"] == 1
    assert not (Path("/proc") / workload_pid.read_text()).exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(23)])
def test_unrequested_process_exit_is_not_converted_to_signal_stop(
    evolution_options, monkeypatch, exception
):
    def exit_without_signal(_pipeline_options):
        raise exception

    def nested_daemon(daemon_options, **kwargs):
        return run_pipeline_daemon(
            daemon_options,
            pipeline_runner=exit_without_signal,
            sleeper=lambda _: None,
            **kwargs,
        )

    monkeypatch.setattr(evolution_daemon, "run_pipeline_daemon", nested_daemon)
    with pytest.raises(type(exception)) as raised:
        run_evolution_daemon(evolution_options)
    assert raised.value is exception
    assert not (evolution_options.state_root / "daemon.lock").exists()
