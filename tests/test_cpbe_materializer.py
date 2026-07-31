from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wmloop.experiments.cpbe_materializer import (
    CPBEMaterializerError,
    publish_cpbe_materialization,
)


def _plan(
    *,
    aggregation: str = "source_sign_margin",
    signal_source: str = "raw_action_sequence",
) -> dict[str, object]:
    program = {
        "probe_id": "cpbe_residual_fixture",
        "signal_source": signal_source,
        "hook_type": "H2",
        "spatial_mask": "all_action_embedding",
        "temporal_basis": "event_phase_tangent",
        "contrast_operator": "signed_mean_preserving_phase",
        "dose_schedule": [-0.05, 0.0, 0.05],
        "aggregation": aggregation,
        "invariants": ["same_seed"],
        "required_capabilities": ["action_embedding_hook"],
        "estimated_gpu_hours": 0.01,
        "origin": "residual",
        "parent_probe_ids": ["parent"],
        "rationale": "single-axis aggregation edit",
        "diagnostic_only": True,
        "reversible": True,
    }
    return {
        "artifact_type": "verdiwm-cpbe-plan",
        "state": "ready",
        "experiment_id": "fixture",
        "selected_work_orders": [
            {
                "probe_id": program["probe_id"],
                "environment": "push_cube",
                "signature": "mixed_source_sign",
                "priority": "P0_collision_basis_gap",
                "role": "diagnostic",
                "verdict_exposure_allowed": False,
                "program": program,
            }
        ],
    }


def test_materializes_exact_work_order_program() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
        manifest = publish_cpbe_materialization(
            plan_path=plan_path,
            output_root=root / "bundle",
        )
        descriptor = json.loads(
            (root / "bundle/configs/probes/staging/cpbe_residual_fixture.json").read_text()
        )
        assert manifest["state"] == "ready"
        assert descriptor["program"] == _plan()["selected_work_orders"][0]["program"]
        assert descriptor["implementation_parameters"]["target_label_used_for_fit"] is False
        assert (root / "bundle/wmloop/diagnose/probes/cpbe_residual_fixture.py").is_file()
        assert (root / "bundle/tests/test_cpbe_residual_fixture.py").is_file()


def test_rejects_unsupported_aggregation_instead_of_compromising() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(_plan(aggregation="horizon_weighted_goal_outcome")))
        with pytest.raises(CPBEMaterializerError, match="PROGRAM_UNSUPPORTED:aggregation"):
            publish_cpbe_materialization(plan_path=plan_path, output_root=root / "bundle")


def test_materializes_action_embedding_delta_without_changing_program_intent() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(
            aggregation="goal_outcome_vector",
            signal_source="action_embedding_delta",
        )
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        manifest = publish_cpbe_materialization(
            plan_path=plan_path,
            output_root=root / "bundle",
        )
        descriptor = json.loads(
            (root / "bundle/configs/probes/staging/cpbe_residual_fixture.json").read_text()
        )
        report = json.loads((root / "bundle/materialization-report.json").read_text())

        assert manifest["state"] == "ready"
        assert descriptor["program"] == plan["selected_work_orders"][0]["program"]
        assert descriptor["implementation_parameters"]["signal_source"] == (
            "action_embedding_delta"
        )
        assert report["probes"][0]["materialization_template"] == (
            "action_embedding_delta_goal_outcome_v1"
        )
