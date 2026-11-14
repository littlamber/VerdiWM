"""Exercise ordinary multi-round planning without model or GPU side effects."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmloop.control import research_request
from wmloop.control.campaign_api import CampaignStore
from wmloop.control.research_execution_policy import derive_research_plan_policy
from wmloop.control.research_request import (
    ResearchRequestError,
    compile_research_plan,
    load_research_plan,
    plan_to_campaign_payload,
    run_research_plan,
    write_research_plan,
)
from wmloop.storage import canonical_bytes, sha256_bytes


@pytest.fixture
def ready_context(tmp_path, monkeypatch):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    (model / "weights.bin").write_bytes(b"original")
    runtime = tmp_path / "python"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    candidate = json.loads((Path(__file__).resolve().parents[1] / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json").read_text())
    candidate["candidates"] = candidate["candidates"][:1]
    for stage in candidate["candidates"][0]["stages"]:
        stage.update(command=["python", "-c", "pass"], environment={})
    template = tmp_path / "candidate.json"
    template.write_text(json.dumps(candidate))
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(json.dumps({"scheduler_template": template.name, "metrics": ["quality"]}))
    execution = {
        "kind": "pipeline", "repo_root": str(model), "output_root": str(tmp_path / "run"),
        "runtime_python": str(runtime), "evaluator_contract": str(evaluator),
        "asset_bindings": {"--checkpoint": str(model / "weights.bin")},
        "budget_total_gpu_hours": 1.0, "budget_db": str(tmp_path / "budget.db"),
        "adapter_contract_ready": True,
    }
    adapter = SimpleNamespace(execution=execution, profile_id="test", model_family="test", capability_level="L1", constitution_freeze="freeze")
    monkeypatch.setattr(research_request, "compile_adapter_execution", lambda **_: adapter)
    monkeypatch.setattr(research_request, "inspect_project", lambda **_: {"blockers": []})
    # Campaign creation imports the same adapter compiler in its own module.
    monkeypatch.setattr("wmloop.control.campaign_api.compile_adapter_execution", lambda **_: adapter)
    return dict(project_root=tmp_path, model=model, data=data, goal="improve", adapter_profile=profile, evaluator_contract=evaluator, runtime_python=runtime)


def test_two_round_plan_freezes_template_and_total_budget(ready_context):
    plan = compile_research_plan(**ready_context, rounds=2, budget="0.5gpu-hours", stop_on_success=True)
    assert plan["schema_version"] == 3
    assert plan["execution_policy"] == {"rounds": 2, "stop_on_success": True, "budget_scope": "all_rounds"}
    assert plan["budget"] == {"gpu_hours": 0.5}
    assert "evolution_candidate_template" in plan["input_bindings"]
    root = ready_context["project_root"]
    created = CampaignStore(root / "state/campaigns").create(plan_to_campaign_payload(plan))
    execution = created["execution"]
    assert execution["kind"] == "evolution"
    assert execution["max_iterations"] == 2
    assert execution["poll_seconds"] == 0
    assert execution["total_budget_gpu_hours"] == 0.5
    assert "budget_total_gpu_hours" not in execution
    assert execution["stop_on_success"] is True
    assert execution["open_method_policy"]["require_four_arm_study"] is True
    assert execution["research_plan_binding"]["plan_digest"] == plan["plan_digest"]


def test_missing_template_blocks_before_hashing_or_campaign_creation(ready_context, monkeypatch):
    ready_context["evaluator_contract"].write_text("{}")
    monkeypatch.setattr(research_request, "_input_bindings", lambda **_: pytest.fail("blocked plan must not hash model"))
    plan = compile_research_plan(**ready_context, rounds=2)
    assert plan["state"] == "blocked"
    assert any(str(row["code"]).startswith("EVOLUTION_") for row in plan["blockers"])
    assert plan["input_digest"] is None
    assert plan["input_bindings"] == {}
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_BLOCKED"):
        plan_to_campaign_payload(plan)


def test_policy_asset_disappearing_during_hashing_returns_an_unbound_blocked_plan(ready_context, monkeypatch):
    def disappeared(_path):
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_MISSING")

    monkeypatch.setattr(research_request, "bind_research_input", disappeared)
    plan = compile_research_plan(**ready_context, rounds=2)
    assert plan["state"] == "blocked"
    assert plan["input_bindings"] == {}
    assert plan["input_digest"] is None
    assert any(item["code"] == "RESEARCH_PLAN_INPUT_MISSING" for item in plan["blockers"])


def test_policy_change_preserves_original_plan_and_requires_review_without_confirm(ready_context):
    plan = compile_research_plan(**ready_context)
    root = ready_context["project_root"]
    source = write_research_plan(plan, root / "original.json")
    original = source.read_bytes()
    result = run_research_plan(plan, store=None, confirm=False, rounds=2)
    assert result["state"] == "awaiting_confirmation"
    assert result["policy_changed"] is True
    derived = load_research_plan(result["plan_path"])
    assert derived["parent_plan_digest"] == plan["plan_digest"]
    assert derived["plan_digest"] != plan["plan_digest"]
    assert derived["execution_policy"]["rounds"] == 2
    assert derived["budget"] == plan["budget"]
    assert source.read_bytes() == original
    assert not (root / "state/campaigns").exists()


def test_explicit_confirm_runs_new_binding_once_without_editing_old_campaign(ready_context):
    plan = compile_research_plan(**ready_context)
    root = ready_context["project_root"]
    store = CampaignStore(root / "state/campaigns")
    original = run_research_plan(plan, store=store, confirm=True, queue_only=True)
    result = run_research_plan(plan, store=store, confirm=True, queue_only=True, rounds=2)
    assert result["state"] == "queued"
    assert result["campaign"]["campaign_id"] != original["campaign"]["campaign_id"]
    assert result["campaign"]["execution"]["max_iterations"] == 2
    assert store.get(original["campaign"]["campaign_id"])["execution"]["kind"] == "pipeline"
    assert result["campaign"]["execution"]["research_plan_binding"]["plan_digest"] == result["derived_plan"]["plan_digest"]


@pytest.mark.parametrize(("terminal_status", "expected_exit"), [("blocked", 2), ("failed", 2), ("completed", 0), ("cancelled", 0)])
def test_same_terminal_plan_returns_result_without_confirming_or_rerunning(ready_context, monkeypatch, terminal_status, expected_exit):
    from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
    from wmloop.control.commands import research as command

    root = ready_context["project_root"]
    plan = compile_research_plan(**ready_context)
    plan_path = write_research_plan(plan, root / "terminal-plan.json")
    store = CampaignStore(root / "state/campaigns", project_root=root)
    first = run_research_plan(plan, store=store, confirm=True, queue_only=True)
    campaign_id = first["campaign"]["campaign_id"]
    if terminal_status == "cancelled":
        store.cancel(campaign_id)
    else:
        run_dispatcher(DispatcherOptions(store.root), runner=lambda _: {"outcome": terminal_status, "reason": "fixture"})
    before = store.get(campaign_id)
    monkeypatch.setattr(store, "confirm", lambda _: pytest.fail("terminal attempt must not be confirmed again"))
    monkeypatch.setattr("wmloop.control.campaign_dispatcher.run_dispatcher", lambda _: pytest.fail("terminal attempt must not run again"))
    result = run_research_plan(plan, store=store, confirm=True)
    assert result["state"] == terminal_status
    assert result["reused_terminal_result"] is True
    assert "/plan" in result["next_step"]
    assert result["campaign"] == before == store.get(campaign_id)
    # The real command handler keeps negative terminal results non-successful
    # and a no-confirm preview remains side-effect free.
    monkeypatch.setattr(command, "campaign_store", lambda _: store)
    monkeypatch.setattr(command, "emit_json", lambda _: None)
    arguments = SimpleNamespace(plan=plan_path, state_root=root / "state", confirm=True,
                                queue_only=False, max_parallel=1, rounds=None, stop_on_success=None)
    assert command.run(arguments) == expected_exit
    assert run_research_plan(plan, store=None, confirm=False)["state"] == "awaiting_confirmation"
    assert store.get(campaign_id) == before


def test_policy_derivation_cannot_absorb_changed_model_inputs(ready_context):
    plan = compile_research_plan(**ready_context)
    (ready_context["model"] / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INPUT_DIGEST_DRIFT"):
        derive_research_plan_policy(plan, rounds=2)


def test_template_change_and_policy_tampering_are_rejected(ready_context):
    plan = compile_research_plan(**ready_context, rounds=2)
    template = ready_context["project_root"] / "candidate.json"
    template.write_text(template.read_text() + "\n")
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INPUT_DIGEST_DRIFT"):
        plan_to_campaign_payload(plan)
    plan["execution_policy"]["rounds"] = 3
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_DIGEST_MISMATCH"):
        write_research_plan(plan, ready_context["project_root"] / "tampered.json")


def test_goal_stop_without_frozen_goal_remains_bounded_and_warns(ready_context):
    plan = compile_research_plan(**ready_context, rounds=2, stop_on_success=True)
    assert plan["state"] in {"ready", "ready_with_deferred_discovery"}
    assert plan["warnings"]
    assert "frozen_goal" not in plan["input_bindings"]
    assert plan["execution_policy"]["rounds"] == 2


def _bind_frozen_goal(ready_context):
    goal = {
        "schema_version": 2, "artifact_type": "wmloop-goal-spec",
        "claim_boundary": "A CPU fixture for immutable goal bindings.",
        "goal_id": "protected_metrics_binding", "metric_family": ["quality"],
        "horizons": [16, 32], "primary_objective": "utility",
        "action_following_gate": {"tau_af": 0.5},
        "eval_protocol": {
            "dev_split": "dev", "accept_split": "accept", "accept_horizon_extra": 16,
            "seeds_accept": [101, 202, 303],
            "acceptance_policy": {
                "metric_directions": {"utility": "maximize", "action_fidelity": "maximize"},
                "minimum_effect": 0.5, "confidence_interval": "student_t_two_sided_95",
            },
        },
        "budget": {"total_gpu_hours": 1, "per_trial_max_gpu_hours": 0.1},
        "envs": ["test"],
    }
    goal_path = ready_context["project_root"] / "goal.json"
    goal_path.write_text(json.dumps(goal))
    evaluator_path = ready_context["evaluator_contract"]
    evaluator = json.loads(evaluator_path.read_text())
    evaluator["frozen_goal"] = goal_path.name
    evaluator_path.write_text(json.dumps(evaluator))
    return goal_path


def test_protected_metrics_preview_comes_from_frozen_goal_and_stays_separate_from_irg(ready_context):
    _bind_frozen_goal(ready_context)
    plan = compile_research_plan(**ready_context, protected_metrics=("action_fidelity",))

    assert plan["state"] in {"ready", "ready_with_deferred_discovery"}
    assert plan["protected_metrics"] == ["action_fidelity"]
    assert plan["protected_metrics_requested"] == ["action_fidelity"]
    assert plan["irg_protected_metrics"] == []
    assert "frozen_goal" in plan["input_bindings"]
    campaign = plan_to_campaign_payload(plan)
    assert "protected_metrics" not in campaign
    assert "irg_protected_metrics" not in campaign


def test_protected_metrics_request_without_frozen_goal_blocks_and_is_not_an_irg_metric(ready_context):
    plan = compile_research_plan(**ready_context, protected_metrics=("action_fidelity",))

    assert plan["state"] == "blocked"
    assert plan["input_bindings"] == {}
    assert plan["protected_metrics"] == []
    assert plan["protected_metrics_requested"] == ["action_fidelity"]
    assert plan["irg_protected_metrics"] == []
    assert any(item["code"] == "PROTECTED_METRICS_REQUIRE_FROZEN_GOAL" for item in plan["blockers"])


def test_protected_metrics_mismatch_blocks_and_survives_policy_revision(ready_context):
    _bind_frozen_goal(ready_context)
    plan = compile_research_plan(**ready_context, protected_metrics=("action_fidelity", "latency"))

    assert plan["state"] == "blocked"
    assert plan["input_bindings"] == {}
    assert plan["protected_metrics"] == ["action_fidelity"]
    assert plan["irg_protected_metrics"] == []
    assert any(item["code"] == "PROTECTED_METRICS_FROZEN_GOAL_MISMATCH" for item in plan["blockers"])
    revised = derive_research_plan_policy(plan, rounds=2)
    assert revised["state"] == "blocked"
    assert revised["protected_metrics_requested"] == ["action_fidelity", "latency"]
    assert any(item["code"] == "PROTECTED_METRICS_FROZEN_GOAL_MISMATCH" for item in revised["blockers"])


def test_protected_metrics_preview_must_match_bound_goal_before_campaign_creation(ready_context):
    _bind_frozen_goal(ready_context)
    plan = compile_research_plan(**ready_context)
    plan["protected_metrics"] = ["latency"]
    plan.pop("plan_digest")
    plan["plan_digest"] = sha256_bytes(canonical_bytes(plan))

    with pytest.raises(ResearchRequestError, match="PROTECTED_METRICS_PLAN_BINDING_MISMATCH"):
        plan_to_campaign_payload(plan)


def test_frozen_goal_is_bound_before_launch_and_policy_derivation(ready_context):
    goal = {
        "schema_version": 2, "artifact_type": "wmloop-goal-spec",
        "claim_boundary": "A CPU fixture for immutable goal bindings.",
        "goal_id": "rounds_binding", "metric_family": ["quality"],
        "horizons": [16, 32], "primary_objective": "utility",
        "action_following_gate": {"tau_af": 0.5},
        "eval_protocol": {
            "dev_split": "dev", "accept_split": "accept", "accept_horizon_extra": 16,
            "seeds_accept": [101, 202, 303],
            "acceptance_policy": {"metric_directions": {"utility": "maximize"},
                                  "minimum_effect": 0.5, "confidence_interval": "student_t_two_sided_95"},
        },
        "budget": {"total_gpu_hours": 1, "per_trial_max_gpu_hours": 0.1},
        "envs": ["test"],
    }
    goal_path = ready_context["project_root"] / "goal.json"
    goal_path.write_text(json.dumps(goal))
    evaluator_path = ready_context["evaluator_contract"]
    evaluator = json.loads(evaluator_path.read_text())
    evaluator["frozen_goal"] = goal_path.name
    evaluator_path.write_text(json.dumps(evaluator))
    plan = compile_research_plan(**ready_context)
    derived = derive_research_plan_policy(plan, rounds=2, stop_on_success=True)
    assert "frozen_goal" in plan["input_bindings"]
    assert derived["input_bindings"]["frozen_goal"] == plan["input_bindings"]["frozen_goal"]
    assert derived["warnings"] == []
    goal["eval_protocol"]["acceptance_policy"]["minimum_effect"] = 0.0
    goal_path.write_text(json.dumps(goal))
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INPUT_DIGEST_DRIFT"):
        plan_to_campaign_payload(derived)
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INPUT_DIGEST_DRIFT"):
        derive_research_plan_policy(plan, rounds=3)


@pytest.mark.parametrize("version", [1, 2])
def test_historical_plan_defaults_to_one_round_and_can_be_derived(ready_context, version):
    plan = compile_research_plan(**ready_context)
    plan["schema_version"] = version
    plan.pop("execution_policy")
    plan.pop("plan_digest")
    plan["plan_digest"] = sha256_bytes(canonical_bytes(plan))
    loaded = load_research_plan(write_research_plan(plan, ready_context["project_root"] / "legacy.json"))
    assert plan_to_campaign_payload(loaded)["execution_policy"]["rounds"] == 1
    assert derive_research_plan_policy(loaded, rounds=2)["execution_policy"]["rounds"] == 2


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5, "2"])
def test_invalid_rounds_rejected_before_readiness_scan(ready_context, rounds):
    with pytest.raises(ResearchRequestError, match="RESEARCH_ROUNDS_INVALID"):
        compile_research_plan(**ready_context, rounds=rounds)
