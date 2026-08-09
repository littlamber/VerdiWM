from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from wmloop.execute.evolution_daemon import (
    EvolutionDaemonOptions,
    run_evolution_daemon,
)


def _options(tmp_path: Path, *, max_iterations: int = 2) -> EvolutionDaemonOptions:
    repo = tmp_path / "model"
    repo.mkdir()
    evaluator = Path("configs/onboarding/ctrl_world_predictive_probe_evaluator_v1.json").resolve()
    probe = Path("configs/probes/ctrl_world_predictive_diagnostic_v1.json").resolve()
    return EvolutionDaemonOptions(
        repo_root=repo,
        output_root=tmp_path / "runs",
        state_root=tmp_path / "state",
        evaluator_contract=evaluator,
        probe_contract=probe,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "cas",
        budget_db=tmp_path / "budget.db",
        retrieval_db=tmp_path / "retrieval.db",
        total_budget_gpu_hours=2.0,
        poll_seconds=0.0,
        max_iterations=max_iterations,
        inner_max_cycles=1,
    )


def test_each_iteration_materializes_fresh_identities(tmp_path: Path) -> None:
    seen: list[tuple[str, str, str]] = []

    def fake_runner(options) -> dict[str, object]:
        pipeline = options.pipeline
        batch_path = Path(pipeline.evaluator_contract).parent / "candidate-batch.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        probe = json.loads(Path(pipeline.probe_contract).read_text(encoding="utf-8"))
        seen.append(
            (
                batch["campaign_id"],
                batch["candidates"][0]["candidate_id"],
                probe["probe_id"],
            )
        )
        output = Path(pipeline.output_root)
        output.mkdir(parents=True)
        (output / "pipeline-manifest.json").write_text(
            json.dumps({"diagnostic_probe": {"failure_signatures": ["horizon_drift"]}}),
            encoding="utf-8",
        )
        return {"state": "completed", "pipeline_verdict": "PASS"}

    manifest = run_evolution_daemon(
        _options(tmp_path), pipeline_runner=fake_runner, sleeper=lambda _: None
    )

    assert manifest["state"] == "completed"
    assert manifest["settled_iterations"] == 2
    assert len(seen) == 2
    assert len({row[0] for row in seen}) == 2
    assert len({row[1] for row in seen}) == 2
    assert len({row[2] for row in seen}) == 2
    first_batch = json.loads(
        (tmp_path / "runs/iterations/iteration-000001/inputs/candidate-batch.json").read_text()
    )
    second_batch = json.loads(
        (tmp_path / "runs/iterations/iteration-000002/inputs/candidate-batch.json").read_text()
    )
    assert "horizon_drift" in second_batch["candidates"][0]["retrieval_keys"]["failure_signatures"]
    assert first_batch["campaign_id"] != second_batch["campaign_id"]


def test_no_information_policy_stops_long_running_controller(tmp_path: Path) -> None:
    calls = 0

    def fake_runner(options) -> dict[str, object]:
        nonlocal calls
        calls += 1
        output = Path(options.pipeline.output_root)
        output.mkdir(parents=True)
        (output / "pipeline-manifest.json").write_text(
            json.dumps({"diagnostic_probe": {"failure_signatures": ["same_signature"]}}),
            encoding="utf-8",
        )
        return {"state": "completed", "pipeline_verdict": "PASS"}

    options = _options(tmp_path, max_iterations=0)
    options = replace(options, max_no_information=2)
    manifest = run_evolution_daemon(options, pipeline_runner=fake_runner, sleeper=lambda _: None)

    assert calls == 3
    assert manifest["state"] == "completed"
    assert manifest["stop_reason"] == "NO_NEW_INFORMATION_LIMIT"


def test_stopped_iteration_resumes_without_new_identity(tmp_path: Path) -> None:
    options = _options(tmp_path, max_iterations=1)
    seen: list[str] = []

    def stopped_runner(daemon_options) -> dict[str, object]:
        batch = json.loads(
            (
                Path(daemon_options.pipeline.evaluator_contract).parent
                / "candidate-batch.json"
            ).read_text()
        )
        seen.append(batch["campaign_id"])
        return {"state": "stopped", "pipeline_verdict": None}

    first = run_evolution_daemon(
        options, pipeline_runner=stopped_runner, sleeper=lambda _: None
    )
    assert first["state"] == "stopped"

    def completed_runner(daemon_options) -> dict[str, object]:
        batch = json.loads(
            (
                Path(daemon_options.pipeline.evaluator_contract).parent
                / "candidate-batch.json"
            ).read_text()
        )
        seen.append(batch["campaign_id"])
        output = Path(daemon_options.pipeline.output_root)
        output.mkdir(parents=True, exist_ok=True)
        (output / "pipeline-manifest.json").write_text(
            json.dumps({"diagnostic_probe": {"failure_signatures": ["new"]}})
        )
        return {"state": "completed", "pipeline_verdict": "PASS"}

    second = run_evolution_daemon(
        replace(options, inner_max_cycles=2),
        pipeline_runner=completed_runner,
        sleeper=lambda _: None,
    )
    assert second["state"] == "completed"
    assert seen[0] == seen[1]
    assert second["iteration"] == 1


def test_stale_daemon_lock_is_recovered(tmp_path: Path) -> None:
    options = _options(tmp_path, max_iterations=1)

    def completed_runner(daemon_options) -> dict[str, object]:
        output = Path(daemon_options.pipeline.output_root)
        output.mkdir(parents=True)
        (output / "pipeline-manifest.json").write_text(
            json.dumps({"diagnostic_probe": {"failure_signatures": []}})
        )
        return {"state": "completed", "pipeline_verdict": "PASS"}

    manifest = run_evolution_daemon(
        options, pipeline_runner=completed_runner, sleeper=lambda _: None
    )
    assert manifest["state"] == "completed"
    lock = tmp_path / "state/daemon.lock"
    lock.write_text(json.dumps({"pid": 999_999_999}))

    resumed = run_evolution_daemon(
        options, pipeline_runner=lambda _: {}, sleeper=lambda _: None
    )
    assert resumed["state"] == "completed"
    assert not lock.exists()
