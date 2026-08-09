from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.execute import experiment_scheduler as scheduler
from wmloop.execute.auto_experiment import _validate_plan_semantics
from wmloop.execute.experiment_scheduler import ExperimentSchedulerError


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "configs" / "smoke" / "auto_experiment_candidate_batch_cuda_v1.json"


def test_plan_is_deterministic_budget_bounded_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "queue"
    first = scheduler.plan_candidate_batch(batch_path=BATCH, output_root=output, workspace_root=ROOT)
    assert first["state"] == "ready"
    assert first["ranked_candidate_count"] == 2
    assert [row["candidate_id"] for row in first["selected"]] == ["cuda-matmul-balanced"]
    assert first["deferred"][0]["candidate_id"] == "cuda-matmul-cheap-control"
    assert first["deferred"][0]["reason"] == "MAX_SELECTED_CANDIDATES"
    assert (output / "plans" / "auto-cuda-matmul-balanced-screen.json").is_file()
    assert (output / "plans" / "auto-cuda-matmul-balanced-gate.json").is_file()
    assert (output / "plans" / "auto-cuda-matmul-balanced-confirm.json").is_file()
    for plan_path in sorted((output / "plans").glob("*.json")):
        plan = scheduler._load_json_object(plan_path, "TEST_PLAN_INVALID")
        _validate_plan_semantics(plan, workspace_root=ROOT)
        assert plan["stage"] in {"screen", "gate", "confirm"}
    second = scheduler.plan_candidate_batch(batch_path=BATCH, output_root=output, workspace_root=ROOT)
    assert second == first


def test_invalid_stage_ladder_fails_before_materialization(tmp_path: Path) -> None:
    payload = json.loads(BATCH.read_text(encoding="utf-8"))
    payload["candidates"][0]["stages"] = [payload["candidates"][0]["stages"][1]]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentSchedulerError, match="STAGE_ORDER_INVALID|STAGE_LADDER_INVALID"):
        scheduler.plan_candidate_batch(batch_path=broken, output_root=tmp_path / "queue", workspace_root=ROOT)


def test_candidate_ladder_must_fit_campaign_budget(tmp_path: Path) -> None:
    payload = json.loads(BATCH.read_text(encoding="utf-8"))
    payload["total_budget_gpu_hours"] = 0.02
    broken = tmp_path / "over-budget.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentSchedulerError, match="CANDIDATE_LADDER_EXCEEDS_BUDGET"):
        scheduler.plan_candidate_batch(batch_path=broken, output_root=tmp_path / "queue", workspace_root=ROOT)
    assert not (tmp_path / "queue").exists()


def test_screen_estimates_are_truncated_against_campaign_ceiling(tmp_path: Path) -> None:
    payload = json.loads(BATCH.read_text(encoding="utf-8"))
    payload["total_budget_gpu_hours"] = 0.007
    payload["max_selected_candidates"] = 2
    payload["candidates"][0]["stages"] = payload["candidates"][0]["stages"][:1]
    batch = tmp_path / "screen-budget.json"
    batch.write_text(json.dumps(payload), encoding="utf-8")
    queue = scheduler.plan_candidate_batch(
        batch_path=batch,
        output_root=tmp_path / "queue",
        workspace_root=ROOT,
    )
    assert [row["candidate_id"] for row in queue["selected"]] == ["cuda-matmul-balanced"]
    assert queue["deferred"][0]["reason"] == "SCREEN_BUDGET_EXCEEDED"


def test_probe_mismatched_candidate_never_enters_gpu_queue(tmp_path: Path) -> None:
    payload = json.loads(BATCH.read_text(encoding="utf-8"))
    for candidate in payload["candidates"]:
        candidate["routing_admission"] = {
            "state": "blocked",
            "reason": "candidate_failure_signature_mismatch",
            "matched_failure_signatures": [],
        }
    batch = tmp_path / "routing-blocked.json"
    batch.write_text(json.dumps(payload), encoding="utf-8")

    queue = scheduler.plan_candidate_batch(
        batch_path=batch,
        output_root=tmp_path / "queue",
        workspace_root=ROOT,
    )

    assert queue["selected"] == []
    assert queue["ranked_candidate_count"] == 0
    assert queue["routing_blocked"][0]["candidate_id"] == "cuda-matmul-balanced"


def test_run_promotes_only_after_pass_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)
    calls: list[str] = []

    def fake_run_auto_experiment(**kwargs):
        stage = Path(kwargs["output_root"]).name
        calls.append(stage)
        return {"verdict": "PASS" if stage == "screen" else "VOID", "evidence_level": "runtime_verified"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", fake_run_auto_experiment)
    first = scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
    )
    assert calls == ["screen", "gate"]
    assert first["candidate_states"]["cuda-matmul-balanced"] == "blocked"
    assert first["results"]["cuda-matmul-balanced:gate"]["verdict"] == "VOID"
    execution_path = queue_root / "execution.json"
    legacy_execution = json.loads(execution_path.read_text(encoding="utf-8"))
    legacy_execution.pop("resource_policy")
    execution_path.write_text(json.dumps(legacy_execution), encoding="utf-8")

    calls.clear()
    second = scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
    )
    assert calls == []
    assert second["candidate_states"]["cuda-matmul-balanced"] == "blocked"


def test_run_all_passes_through_confirm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)
    calls: list[str] = []

    def fake_run_auto_experiment(**kwargs):
        calls.append(Path(kwargs["output_root"]).name)
        return {"verdict": "PASS", "evidence_level": "runtime_verified"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", fake_run_auto_experiment)
    result = scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
    )
    assert calls == ["screen", "gate", "confirm"]
    assert result["candidate_states"]["cuda-matmul-balanced"] == "completed"
    assert result["promotion_decisions"]["cuda-matmul-balanced"]["state"] == "not_requested"


def test_scientific_promotion_is_explicit_and_cost_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(BATCH.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]
    candidate["promotion_policy"] = {
        "required_stage": "confirm",
        "required_quality_metrics": ["finite_fraction"],
    }
    batch = tmp_path / "promotion-batch.json"
    batch.write_text(json.dumps(payload), encoding="utf-8")
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(
        batch_path=batch, output_root=queue_root, workspace_root=ROOT
    )

    monkeypatch.setattr(
        scheduler,
        "run_auto_experiment",
        lambda **_: {"verdict": "PASS", "evidence_level": "runtime_verified"},
    )
    result = scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_total_gpu_hours=500.0,
        budget_max_trial_gpu_hours=400.0,
        budget_high_trial_limit=8,
        budget_require_high_cost_approval=False,
    )

    decision = result["promotion_decisions"]["cuda-matmul-balanced"]
    assert decision["state"] == "eligible"
    assert decision["reason"] == "QUALITY_METRICS_CONFIRMED"
    assert "cost" not in decision


def test_queue_plan_path_cannot_escape_queue_root(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)
    queue = json.loads((queue_root / "queue.json").read_text(encoding="utf-8"))
    queue["selected"][0]["stages"][0]["plan_path"] = "../outside.json"
    (queue_root / "queue.json").write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ExperimentSchedulerError, match="PLAN_PATH_INVALID"):
        scheduler.run_selected_queue(
            queue_path=queue_root / "queue.json",
            workspace_root=ROOT,
            archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
        )


def test_queue_rejects_tampered_generated_plan(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)
    plan_path = queue_root / "plans" / "auto-cuda-matmul-balanced-screen.json"
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ExperimentSchedulerError, match="PLAN_HASH_MISMATCH"):
        scheduler.run_selected_queue(
            queue_path=queue_root / "queue.json",
            workspace_root=ROOT,
            archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
        )


def test_scheduler_persists_exception_before_reraising(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)

    def fail_run(**kwargs):
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(scheduler, "run_auto_experiment", fail_run)
    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        scheduler.run_selected_queue(
            queue_path=queue_root / "queue.json",
            workspace_root=ROOT,
            archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
        )
    execution = json.loads((queue_root / "execution.json").read_text(encoding="utf-8"))
    assert execution["results"]["cuda-matmul-balanced:screen"] == {
        "state": "error",
        "error": {"type": "RuntimeError", "message": "synthetic launch failure"},
    }


def test_resume_rejects_a_different_shared_budget_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(batch_path=BATCH, output_root=queue_root, workspace_root=ROOT)

    monkeypatch.setattr(
        scheduler,
        "run_auto_experiment",
        lambda **kwargs: {"verdict": "VOID", "evidence_level": "runtime_verified"},
    )
    budget_a = tmp_path / "budget-a.db"
    scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_db=budget_a,
    )
    with pytest.raises(ExperimentSchedulerError, match="BUDGET_PATH_MISMATCH"):
        scheduler.run_selected_queue(
            queue_path=queue_root / "queue.json",
            workspace_root=ROOT,
            archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
            budget_db=tmp_path / "budget-b.db",
        )


def test_scheduler_binds_and_forwards_daemon_global_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue_root = tmp_path / "queue"
    scheduler.plan_candidate_batch(
        batch_path=BATCH,
        output_root=queue_root,
        workspace_root=ROOT,
    )
    observed: list[float] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.append(float(kwargs["budget_total_gpu_hours"]))
        return {"verdict": "VOID", "evidence_level": "runtime_verified"}

    monkeypatch.setattr(scheduler, "run_auto_experiment", fake_run)
    scheduler.run_selected_queue(
        queue_path=queue_root / "queue.json",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_db=tmp_path / "budget.db",
        budget_total_gpu_hours=0.5,
    )

    assert observed == [0.5]
    execution = json.loads((queue_root / "execution.json").read_text())
    assert execution["budget_total_gpu_hours"] == 0.5
    with pytest.raises(ExperimentSchedulerError, match="BUDGET_TOTAL_MISMATCH"):
        scheduler.run_selected_queue(
            queue_path=queue_root / "queue.json",
            workspace_root=ROOT,
            archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
            budget_db=tmp_path / "budget.db",
            budget_total_gpu_hours=0.6,
        )
