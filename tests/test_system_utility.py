from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wmloop.evaluate.system_utility import (
    SystemUtilityAuditError,
    run_system_utility_audit,
)
from wmloop.experiments.lobo import build_lobo_plan
from wmloop.experiments.spec import load_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = ROOT / "configs/experiments/system_utility_audit_v1.json"
CANARY_CONFIG = ROOT / "configs/experiments/ctrl_world_experience_utility_canary_v1.json"


def test_system_utility_audit_separates_operational_and_effect_states() -> None:
    with TemporaryDirectory() as temporary:
        manifest = run_system_utility_audit(
            config_path=AUDIT_CONFIG,
            repo_root=ROOT,
            output_root=Path(temporary) / "audit",
        )
        report = json.loads(
            (Path(temporary) / "audit" / "system-utility-audit.json").read_text(encoding="utf-8")
        )

    assert manifest["state"] == "partial"
    assert report["operational_state"] == "ready"
    assert report["research_effect_state"] == "not_established"
    assert report["utility_summary"]["progressive_fidelity_gpu_hour_reduction"] == pytest.approx(0.062751783)
    gates = {row["id"]: row for row in report["gates"]}
    assert gates["operational_minimal_loop"]["state"] == "pass"
    assert gates["progressive_fidelity_efficiency"]["state"] == "pass"
    assert gates["selector_quality"]["state"] == "blocked"
    assert gates["ctrl_world_local_chart"]["state"] == "pass"
    assert gates["ctrl_world_formal_chart"]["state"] == "abstain"
    assert gates["cross_backbone_confirm"]["state"] == "blocked"


def test_system_utility_audit_fails_closed_for_missing_evidence() -> None:
    with TemporaryDirectory() as temporary:
        with pytest.raises(SystemUtilityAuditError, match="SYSTEM_UTILITY_INPUT_MISSING"):
            run_system_utility_audit(
                config_path=AUDIT_CONFIG,
                repo_root=Path(temporary),
                output_root=Path(temporary) / "audit",
            )


def test_ctrl_world_experience_canary_is_launch_ready_and_bounded() -> None:
    spec = load_experiment_spec(CANARY_CONFIG)
    plan = build_lobo_plan(spec)

    assert plan["state"] == "ready"
    assert plan["launch_ready"] is True
    assert plan["planned_trial_count"] == 36
    assert plan["planned_stage_task_count"] == 108
    assert all(trial["target_backbone"] not in trial["source_backbones"] for trial in plan["trials"])
    assert all(float(stage["max_gpu_hours"]) <= 4.0 for trial in plan["trials"] for stage in trial["stages"])
