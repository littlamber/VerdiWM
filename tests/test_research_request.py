from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.research_request import (
    ResearchRequestError,
    compile_research_plan,
    load_research_plan,
    plan_to_campaign_payload,
    verify_research_plan_inputs,
    write_research_plan,
)


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
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        (model / "weights.bin").write_bytes(b"v1")
        plan = compile_research_plan(project_root=root, model=model, data=data, goal="goal")
        (model / "weights.bin").write_bytes(b"v2")
        verification = verify_research_plan_inputs(plan)
        assert verification["state"] == "blocked"
        assert any(row["code"] == "INPUT_DIGEST_DRIFT" for row in verification["drift"])
        with pytest.raises(ResearchRequestError, match="RESEARCH_PLAN_BLOCKED"):
            plan_to_campaign_payload(plan)


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


def test_explicit_evaluator_and_runtime_are_bound_into_execution() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        model = root / "model"
        data = root / "data"
        model.mkdir()
        data.mkdir()
        (model / "weights.bin").write_bytes(b"v1")
        execution = _ready_execution(root)
        evaluator = Path(str(execution["evaluator_contract"]))
        runtime = Path(str(execution["runtime_python"]))
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
            execution={**execution, "evaluator_contract": str(root / "profile-evaluator.json")},
            profile_id="test-profile", model_family="test", capability_level="L1", constitution_freeze="freeze",
        )
        Path(str(fake_adapter.execution["evaluator_contract"])).write_text('{"metrics": ["metric"]}', encoding="utf-8")
        with patch("wmloop.control.research_request.compile_adapter_execution", return_value=fake_adapter), patch(
            "wmloop.control.research_request.inspect_project", return_value=readiness
        ):
            plan = compile_research_plan(
                project_root=root, model=model, data=data, goal="goal", adapter_profile=profile,
                evaluator_contract=evaluator, runtime_python=runtime,
            )
        payload = plan_to_campaign_payload(plan)
        assert payload["evaluator_contract"] == str(evaluator)
        assert payload["runtime_python"] == str(runtime)


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
