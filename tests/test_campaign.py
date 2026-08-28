from pathlib import Path

from verdi_core.campaign import CampaignPolicy, CampaignSupervisor, STAGES


def test_campaign_persists_stages_and_stops_on_positive(tmp_path: Path) -> None:
    calls = []

    def runner(idea, stage, context):
        calls.append((idea["idea_id"], stage))
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "confirmed_positive", "delta": 0.2, "protected_ok": True, "split": "heldout", "metric_direction": "maximize", "independent_replicates": 2}
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, policy=CampaignPolicy(stop_on_first_positive=True))
    supervisor.create(run_id="run-1", objective="improve", ideas=[{"idea_id": "a"}, {"idea_id": "b"}])
    result = supervisor.run_until_blocked("run-1")
    assert result["state"] == "stopped"
    assert result["stop_reason"] == "first_replicated_positive"
    assert [(idea_id, stage) for idea_id, stage in calls if idea_id == "a"] == [("a", stage) for stage in STAGES]
    assert supervisor.state.count("evidence") == 1


def test_campaign_waits_one_idea_for_human_and_continues_other(tmp_path: Path) -> None:
    def runner(idea, stage, context):
        if idea["idea_id"] == "a" and stage == "heldout_evaluate":
            return {"state": "completed", "requires_human_labels": True, "reason": "task_success_rate"}
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "null", "delta": 0.0, "protected_ok": True}
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-2", objective="improve", ideas=[{"idea_id": "a"}, {"idea_id": "b"}])
    result = supervisor.run_until_blocked("run-2")
    assert result["state"] == "waiting_human"
    assert result["ideas"]["a"]["state"] == "waiting_human"
    assert result["ideas"]["b"]["state"] == "settled"


def test_campaign_marks_runtime_failure_blocked_and_can_resume(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def runner(idea, stage, context):
        attempts["count"] += 1
        if stage == "static_check" and attempts["count"] == 1:
            return {"state": "runtime_failed", "reason": "dependency_missing"}
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "null", "delta": 0.0, "protected_ok": True}
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-3", objective="improve", ideas=[{"idea_id": "a"}])
    blocked = supervisor.run_until_blocked("run-3")
    assert blocked["state"] == "blocked"
    resumed = supervisor.resume("run-3")
    assert resumed["state"] == "settled"


def test_campaign_requires_independent_replicates_for_positive(tmp_path: Path) -> None:
    def runner(idea, stage, context):
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "positive", "delta": 0.2, "protected_ok": True, "independent_replicates": 1}
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-4", objective="improve", ideas=[{"idea_id": "a"}, {"idea_id": "b"}])
    result = supervisor.run_until_blocked("run-4")
    assert result["state"] == "settled"
    assert result["ideas"]["a"]["settlement"]["outcome"] == "abstain"
    assert result["ideas"]["a"]["settlement"]["reason"] == "requires_independent_replicates"
    assert result["ideas"]["b"]["state"] == "settled"


def test_campaign_repeats_full_train_until_runner_finishes(tmp_path: Path) -> None:
    full_train_calls = []

    def runner(idea, stage, context):
        if stage == "full_train":
            full_train_calls.append(context["attempt_count"])
            if len(full_train_calls) < 3:
                return {"state": "continue_long_train", "step": len(full_train_calls) * 1000}
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "null", "delta": 0.0, "protected_ok": True}
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-5", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-5")
    assert result["state"] == "settled"
    assert len(full_train_calls) == 3
    assert result["ideas"]["a"]["stage_index"] == STAGES.index("heldout_evaluate")


def test_campaign_validates_human_batch_before_release(tmp_path: Path) -> None:
    def runner(idea, stage, context):
        if stage == "heldout_evaluate":
            return {
                "state": "completed",
                "requires_human_labels": True,
                "reason": "task_success_rate",
                "batch": {"batch_id": "b1", "episode_ids": ["e1", "e2"], "video_paths": ["e1.mp4", "e2.mp4"], "split": "heldout"},
            }
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-6", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-6")
    assert result["state"] == "waiting_human"
    try:
        supervisor.release_human("run-6", "a", {"e1": True})
    except ValueError as error:
        assert "label_set_mismatch" in str(error)
    else:
        raise AssertionError("incomplete label batch was accepted")
    released = supervisor.release_human("run-6", "a", {"e1": True, "e2": False})
    assert released["ideas"]["a"]["state"] == "running"
    assert released["ideas"]["a"]["human_evaluation"]["success_rate"] == 0.5


def test_campaign_target_settled_mode_continues_after_positive(tmp_path: Path) -> None:
    calls = []

    def runner(idea, stage, context):
        calls.append((idea["idea_id"], stage))
        if stage == "heldout_evaluate":
            outcome = "positive" if idea["idea_id"] == "a" else "null"
            return {"state": "completed", "outcome": outcome, "delta": 0.2 if outcome == "positive" else 0.0, "protected_ok": True, "independent_replicates": 2}
        return {"state": "completed"}

    policy = CampaignPolicy(stop_on_first_positive=False, target_settled_ideas=2)
    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, policy=policy)
    supervisor.create(run_id="run-7", objective="explore", ideas=[{"idea_id": "a"}, {"idea_id": "b"}, {"idea_id": "c"}])
    result = supervisor.run_until_blocked("run-7")
    assert result["state"] == "stopped"
    assert result["stop_reason"] == "target_settled_ideas"
    assert result["ideas"]["a"]["settlement"]["outcome"] == "positive"
    assert result["ideas"]["b"]["settlement"]["outcome"] == "null"
    assert result["ideas"]["c"]["state"] in {"queued", "running"}
    assert result["ideas"]["c"]["settlement"] is None
    assert supervisor.state.count("evidence") == 2


def test_campaign_repair_runner_retries_failed_stage(tmp_path: Path) -> None:
    calls = {"stage": 0, "repair": 0}

    def runner(idea, stage, context):
        calls["stage"] += 1
        if stage == "static_check" and not context.get("repaired"):
            return {"state": "runtime_failed", "reason": "missing_hook"}
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "null", "protected_ok": True}
        return {"state": "completed"}

    def repair(idea, stage, context, failure):
        calls["repair"] += 1
        return runner(idea, stage, {**context, "repaired": True})

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, repair_runner=repair)
    supervisor.create(run_id="run-repair", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-repair")
    assert result["state"] == "settled"
    assert calls["repair"] == 1


def test_campaign_skips_supervisor_repair_after_wrapped_runner_attempt(tmp_path: Path) -> None:
    calls = {"repair": 0}

    def runner(idea, stage, context):
        return {
            "state": "requires_code_patch",
            "reason": "wrapped repair already attempted",
            "engineering": {"state": "abstain", "reason": "provider_failed"},
        }

    def repair(idea, stage, context, failure):
        calls["repair"] += 1
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, repair_runner=repair)
    supervisor.create(run_id="run-wrapped-repair", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-wrapped-repair")
    assert result["state"] == "settled"
    assert calls["repair"] == 0


def test_campaign_does_not_repair_external_resource_wait(tmp_path: Path) -> None:
    calls = {"stage": 0, "repair": 0}

    def runner(idea, stage, context):
        calls["stage"] += 1
        return {
            "state": "blocked",
            "reason": "gpu_capacity_unavailable",
            "blocker_type": "resource_unavailable",
            "retryable": True,
        }

    def repair(idea, stage, context, failure):
        calls["repair"] += 1
        return {"state": "completed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, repair_runner=repair)
    supervisor.create(run_id="run-resource", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-resource")

    assert result["state"] == "blocked"
    assert result["ideas"]["a"]["failure"]["retryable"] is True
    assert calls == {"stage": 1, "repair": 0}


def test_campaign_settles_nonretryable_abstention_as_evidence(tmp_path: Path) -> None:
    def runner(idea, stage, context):
        return {"state": "abstain", "reason": "engineering_provider_failed"}

    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner)
    supervisor.create(run_id="run-abstain", objective="improve", ideas=[{"idea_id": "a"}])
    result = supervisor.run_until_blocked("run-abstain")
    assert result["state"] == "settled"
    assert result["ideas"]["a"]["settlement"]["outcome"] == "abstain"
    assert supervisor.state.count("evidence") == 1


def test_campaign_auto_replans_after_non_positive_batch(tmp_path: Path) -> None:
    rounds = []

    def runner(idea, stage, context):
        if stage == "heldout_evaluate":
            return {"state": "completed", "outcome": "null", "protected_ok": True}
        return {"state": "completed"}

    def replan(campaign, context):
        rounds.append(context["round"])
        return [{"idea_id": "next-" + str(context["round"])}]

    policy = CampaignPolicy(max_research_rounds=1, auto_replan=True)
    supervisor = CampaignSupervisor(tmp_path, model_id="fixture", stage_runner=runner, policy=policy, replanner=replan)
    supervisor.create(run_id="run-replan", objective="improve", ideas=[{"idea_id": "first"}])
    result = supervisor.run_until_blocked("run-replan")
    assert result["state"] == "settled"
    assert rounds == [1]
    assert result["research_round"] == 1
    assert result["ideas"]["next-1"]["settlement"]["outcome"] == "null"
