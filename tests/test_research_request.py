from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control import research_request
from wmloop.control.research_request import (
    ResearchRequestError,
    compile_research_plan,
    load_research_plan,
    plan_to_campaign_payload,
    verify_research_plan_inputs,
    write_research_plan,
)
from wmloop.storage import canonical_bytes, sha256_bytes


def _ready_execution(root: Path) -> dict[str, object]:
    evaluator = root / "evaluator.json"
    evaluator.write_text('{"metrics": ["metric"]}', encoding="utf-8")
    runtime = root / "python"
    runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runtime.chmod(0o755)
    asset = root / "asset.bin"
    asset.write_bytes(b"asset")
    return {
        "kind": "pipeline",
        "repo_root": str(root),
        "output_root": str(root / "run"),
        "evaluator_contract": str(evaluator),
        "runtime_python": str(runtime),
        "asset_bindings": {"--checkpoint": str(asset)},
        "budget_total_gpu_hours": 1.0,
    }


def _compile_ready_plan(root: Path) -> dict[str, object]:
    model, data = root / "model", root / "data"
    model.mkdir()
    data.mkdir()
    (model / "weights.bin").write_bytes(b"v1")
    profile = root / "adapter.json"
    profile.write_text("{}", encoding="utf-8")
    adapter = SimpleNamespace(execution=_ready_execution(root), profile_id="test-profile")
    with patch.object(research_request, "compile_adapter_execution", return_value=adapter), patch.object(
        research_request, "inspect_project", return_value={"blockers": []}
    ):
        return compile_research_plan(
            project_root=root, model=model, data=data, goal="goal", adapter_profile=profile,
        )


def _resign(plan: dict[str, object]) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    plan["plan_digest"] = sha256_bytes(canonical_bytes(unsigned))


@pytest.mark.parametrize("binding,expected_match", [
    (None, True),
    ({"plan_digest": "a" * 64}, True),
    ({"plan_digest": "a" * 24 + "b" * 40}, False),
    ({}, False),
])
def test_terminal_result_uses_full_plan_binding_before_legacy_campaign_id(binding, expected_match):
    digest = "a" * 64
    campaign = {"campaign_id": "research-" + digest[:24], "status": "blocked"}
    if binding is not None:
        campaign["execution"] = {"research_plan_binding": binding}
    result = research_request.matching_terminal_research_result(
        {"schema_version": 3, "plan_digest": digest}, [campaign],
    )
    assert (result is not None) is expected_match
    if expected_match:
        assert result["campaign"] == campaign
        assert result["reused_terminal_result"] is True


def test_plan_is_read_only_and_reports_scientific_blockers() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        (model / "checkpoint.pt").write_bytes(b"checkpoint")
        (model / "evaluate.py").write_text("print('evaluation')\n", encoding="utf-8")
        plan = compile_research_plan(project_root=root, model=model, data=data, goal="improve long horizon consistency")
        assert plan["state"] == "blocked"
        assert {row["code"] for row in plan["blockers"]} >= {"EVALUATOR_CONTRACT_REQUIRED", "ADAPTER_PROFILE_NOT_FOUND"}
        assert plan["side_effects"] == {
            "model_import_executed": False,
            "gpu_execution_started": False,
            "source_modified": False,
        }
        output = write_research_plan(plan, root / "plan.json")
        assert load_research_plan(output)["plan_digest"] == plan["plan_digest"]


def test_plan_digest_binding_rejects_changed_model() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        plan = _compile_ready_plan(root)
        assert plan_to_campaign_payload(plan)["goal"] == "goal"
        (root / "model" / "weights.bin").write_bytes(b"v2")
        verification = verify_research_plan_inputs(plan)
        assert verification["state"] == "blocked"
        assert any(row["code"] == "INPUT_DIGEST_DRIFT" for row in verification["drift"])
        with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INPUT_DIGEST_DRIFT"):
            plan_to_campaign_payload(plan)


def test_blocked_plan_defers_asset_digest_and_still_round_trips(tmp_path, monkeypatch) -> None:
    for name in ("model", "data"):
        (tmp_path / name).mkdir()
    def fail_if_hashed(**kwargs):
        pytest.fail("A blocked readiness report must not traverse input assets")
    monkeypatch.setattr(research_request, "_input_bindings", fail_if_hashed)
    events = []
    plan = compile_research_plan(
        project_root=tmp_path, model=tmp_path / "model", data=tmp_path / "data", goal="goal",
        progress_callback=events.append,
    )
    assert plan["schema_version"] == 3
    assert plan["state"] == "blocked" and plan["blockers"]
    assert plan["input_bindings"] == {} and plan["input_digest"] is None
    assert {"phase": "input_digest", "state": "deferred", "reason": "readiness_blocked"} in events
    output = write_research_plan(plan, tmp_path / "blocked-plan.json")
    assert load_research_plan(output) == plan
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_BLOCKED"):
        plan_to_campaign_payload(plan)
    # Even a caller who recalculates the document digest cannot promote a
    # readiness report into execution evidence without its frozen bindings.
    plan.update(state="ready", blockers=[])
    _resign(plan)
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_INVALID"):
        plan_to_campaign_payload(plan)


def test_digest_failure_returns_persistable_blocked_plan(tmp_path, monkeypatch) -> None:
    def fail(**kwargs):
        raise ResearchRequestError("INPUT_PATH_UNREADABLE:fixture")
    monkeypatch.setattr(research_request, "_input_bindings", fail)
    plan = _compile_ready_plan(tmp_path)
    assert plan["state"] == "blocked"
    assert {row["code"] for row in plan["blockers"]} == {"INPUT_PATH_UNREADABLE"}
    assert plan["input_bindings"] == {} and plan["input_digest"] is None
    assert load_research_plan(write_research_plan(plan, tmp_path / "failed-plan.json")) == plan


@pytest.mark.parametrize("state", ["ready_with_deferred_discovery", "blocked"])
def test_legacy_fully_bound_plans_remain_readable(tmp_path, state) -> None:
    plan = _compile_ready_plan(tmp_path)
    plan.update(schema_version=1, state=state)
    plan.pop("execution_policy")
    if state == "blocked":
        plan["blockers"] = [{"code": "LEGACY_BLOCKER", "message": "fixture", "action": "resolve"}]
    _resign(plan)
    loaded = load_research_plan(write_research_plan(plan, tmp_path / "legacy-plan.json"))
    assert loaded == plan
    assert verify_research_plan_inputs(loaded)["state"] == "verified"
    if state == "blocked":
        with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_BLOCKED"):
            plan_to_campaign_payload(loaded)
    else:
        assert plan_to_campaign_payload(loaded)["plan_binding"]["plan_digest"] == plan["plan_digest"]


def test_ready_plan_keeps_all_inputs_and_rejects_in_memory_tampering(tmp_path) -> None:
    plan = _compile_ready_plan(tmp_path)
    assert set(plan["input_bindings"]) >= {
        "model", "data", "evaluator_contract", "runtime_python", "adapter_profile", "asset:--checkpoint",
    }
    assert verify_research_plan_inputs(plan)["state"] == "verified"
    plan["goal"] = "tampered goal"
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_DIGEST_MISMATCH"):
        plan_to_campaign_payload(plan)
    output = tmp_path / "tampered-plan.json"
    with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_DIGEST_MISMATCH"):
        write_research_plan(plan, output)
    assert not output.exists()


def test_ready_plan_binds_open_method_policy_and_campaign_revision() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        (model / "weights.bin").write_bytes(b"v1")
        profile = root / "adapter.json"
        profile.write_text("{}", encoding="utf-8")
        execution = _ready_execution(root)
        readiness = {
            "state": "ready_for_conformance",
            "blockers": [],
            "model": str(model),
            "source": str(model),
            "data": str(data),
            "discovered": {"entrypoints": ["evaluate.py"], "assets": [], "runtime": {}},
        }
        fake_adapter = SimpleNamespace(
            execution=execution,
            profile_id="test-profile",
            model_family="test",
            capability_level="L1",
            constitution_freeze="freeze",
        )
        with patch("wmloop.control.research_request.compile_adapter_execution", return_value=fake_adapter), patch(
            "wmloop.control.research_request.inspect_project", return_value=readiness
        ):
            plan = compile_research_plan(
                project_root=root,
                model=model,
                data=data,
                goal="goal",
                adapter_profile=profile,
            )
        assert plan["state"] == "ready_with_deferred_discovery"
        payload = plan_to_campaign_payload(plan)
        assert payload["open_method_policy"]["require_four_arm_study"] is True
        store = CampaignStore(root / "state" / "campaigns")
        campaign_payload = dict(payload)
        campaign_payload["execution"] = execution
        campaign_payload.pop("adapter_profile_path", None)
        campaign_payload.pop("runtime_python", None)
        campaign_payload.pop("assets", None)
        created = store.create(campaign_payload)
        assert created["execution"]["open_method_policy"]["authority"] == "four_arm_study_only"
        assert created["execution"]["research_plan_binding"]["plan_id"] == plan["plan_id"]


@pytest.mark.parametrize("evaluator_source", ["argument", "project_config", "profile"])
@pytest.mark.parametrize("explicit_runtime", [False, True])
def test_evaluator_and_runtime_preview_match_frozen_execution(
    evaluator_source: str, explicit_runtime: bool,
) -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        (model / "weights.bin").write_bytes(b"v1")
        execution = _ready_execution(root)
        profile_evaluator = Path(str(execution["evaluator_contract"]))
        profile_runtime = Path(str(execution["runtime_python"]))
        evaluator = root / "user-evaluator.json"
        evaluator.write_text('{"metrics": ["user_metric"]}', encoding="utf-8")
        configured_evaluator = root / "configured-evaluator.json"
        configured_evaluator.write_text('{"metrics": ["configured_metric"]}', encoding="utf-8")
        if evaluator_source != "profile":
            (root / "verdiwm.toml").write_text(
                '[project]\nevaluator_contract = "configured-evaluator.json"\n', encoding="utf-8",
            )
        runtime = root / "user-python"
        runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        runtime.chmod(0o755)
        expected_evaluator = {
            "argument": evaluator, "project_config": configured_evaluator, "profile": profile_evaluator,
        }[evaluator_source]
        expected_runtime = runtime if explicit_runtime else profile_runtime
        profile = root / "adapter.json"
        profile.write_text("{}", encoding="utf-8")
        readiness = {
            "state": "ready_for_conformance",
            "blockers": [],
            "model": str(model),
            "source": str(model),
            "data": str(data),
            "discovered": {"entrypoints": ["evaluate.py"], "assets": [], "runtime": {}},
        }
        fake_adapter = SimpleNamespace(
            execution={**execution, "runtime_python": str(expected_runtime)},
            profile_id="test-profile", model_family="test", capability_level="L1", constitution_freeze="freeze",
        )
        with patch("wmloop.control.research_request.compile_adapter_execution", return_value=fake_adapter) as compile_adapter, patch(
            "wmloop.control.research_request.inspect_project", return_value=readiness
        ) as inspect:
            plan = compile_research_plan(
                project_root=root, model=model, data=data, goal="goal", adapter_profile=profile,
                evaluator_contract=evaluator if evaluator_source == "argument" else None,
                runtime_python=runtime if explicit_runtime else None,
            )
        payload = plan_to_campaign_payload(plan)
        for field, expected in (("evaluator_contract", expected_evaluator), ("runtime_python", expected_runtime)):
            assert payload[field] == plan[field] == plan["adapter_preview"][field] == str(expected)
            assert plan["input_bindings"][field]["sha256"] == sha256_bytes(expected.read_bytes())
        assert inspect.call_args.kwargs["evaluator_contract"] == (
            str(expected_evaluator) if evaluator_source != "profile" else None
        )
        assert compile_adapter.call_args.kwargs["runtime_python"] == (runtime if explicit_runtime else None)
        # The resolved profile remains reusable; the plan owns its effective preview.
        assert fake_adapter.execution["evaluator_contract"] == str(profile_evaluator)


def test_tampered_plan_file_is_rejected() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        plan = compile_research_plan(project_root=root, model=model, data=data, goal="goal")
        path = write_research_plan(plan, root / "plan.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["goal"] = "changed"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_DIGEST_MISMATCH"):
            load_research_plan(path)
