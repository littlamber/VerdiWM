from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.experiments.acwm_cpbe_canary import (
    ACWMCPBECanaryError,
    _aggregate_response_vectors,
    evaluate_acwm_cpbe_canary,
    prepare_acwm_cpbe_canary_bundle,
)


def test_source_sign_margin_projection_never_uses_target_label_for_fit() -> None:
    vectors = {
        "target": (0.8, 0.2),
        "positive": (1.0, 0.0),
        "negative_a": (0.0, 1.0),
        "negative_b": (0.1, 0.9),
    }
    labels = {"target": -1, "positive": 1, "negative_a": -1, "negative_b": -1}

    negative_target, audit = _aggregate_response_vectors(
        vectors,
        aggregation="source_sign_margin",
        labels=labels,
        target="target",
    )
    labels["target"] = 1
    positive_target, changed_audit = _aggregate_response_vectors(
        vectors,
        aggregation="source_sign_margin",
        labels=labels,
        target="target",
    )

    assert negative_target == positive_target
    assert audit["target_label_used_for_fit"] is False
    assert changed_audit["target_label_used_for_fit"] is False
    assert "target" not in audit["fit_environments"]


def test_prepares_frozen_campaign_and_collision_spec() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        manifest = prepare_acwm_cpbe_canary_bundle(
            **inputs,
            output_root=root / "prepared",
        )
        assert manifest["state"] == "ready"
        campaign = json.loads(
            (root / "prepared/campaigns/cpbe_residual_fixture.json").read_text()
        )
        assert campaign["probe"]["temporal_basis"] == "event_phase_curvature"
        assert campaign["probe"]["diagnostic_only"] is True
        spec = json.loads((root / "prepared/collision-spec.json").read_text())
        assert spec["target_environment"] == "push_cube"
        assert {row["environment"]: row["sign"] for row in spec["labels"]} == {
            "push_cube": -1,
            "cloth_move": -1,
            "pour_water": 1,
            "push_sand": -1,
        }
        assert spec["thresholds"]["maximum_redundancy_relative_l2"] == 0.1


def test_preparation_rejects_candidate_from_a_different_reference_parent() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        plan_path = inputs["plan_path"]
        plan = json.loads(plan_path.read_text())
        plan["selected_work_orders"][0]["program"]["parent_probe_ids"] = ["different_parent"]
        plan_path.write_text(json.dumps(plan))
        try:
            prepare_acwm_cpbe_canary_bundle(**inputs, output_root=root / "prepared")
        except ACWMCPBECanaryError as exc:
            assert "ACWM_CPBE_REFERENCE_PARENT_MISMATCH" in str(exc)
        else:
            raise AssertionError("cross-parent canary must fail before GPU execution")


def test_canary_receipt_binds_local_evidence_and_passes_real_three_part_gate() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        prepare_acwm_cpbe_canary_bundle(**inputs, output_root=root / "prepared")
        campaign = root / "prepared/campaigns/cpbe_residual_fixture.json"
        reference = {
            "push_cube": [1.0, 0.0],
            "cloth_move": [0.0, 1.0],
            "push_sand": [0.1, 1.0],
            "pour_water": [1.0, 0.1],
        }
        candidate = {
            "push_cube": [0.0, 1.0],
            "cloth_move": [0.2, 1.0],
            "push_sand": [-0.1, 1.0],
            "pour_water": [1.0, 0.0],
        }
        _campaign_evidence(root / "reference", "action_temporal_alignment_phase", reference)
        _campaign_evidence(root / "candidate", "cpbe_residual_fixture", candidate)

        manifest = evaluate_acwm_cpbe_canary(
            collision_spec_path=root / "prepared/collision-spec.json",
            candidate_campaign_path=campaign,
            reference_campaign_root=root / "reference",
            candidate_campaign_root=root / "candidate",
            output_root=root / "settled",
        )

        assert manifest["passed"] is True
        receipt = json.loads((root / "settled/cpbe-stage-receipt.json").read_text())
        assert receipt["metrics"]["collision_separation"] > 0.0
        assert receipt["metrics"]["nonredundant"] is True
        for artifact in receipt["evidence_artifacts"]:
            path = root / "settled" / artifact["path"]
            assert path.stat().st_size == artifact["size_bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_canary_fails_closed_on_cross_protocol_evidence() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        prepare_acwm_cpbe_canary_bundle(**inputs, output_root=root / "prepared")
        vectors = {
            "push_cube": [1.0, 0.0],
            "cloth_move": [0.0, 1.0],
            "push_sand": [0.1, 1.0],
            "pour_water": [1.0, 0.1],
        }
        _campaign_evidence(root / "reference", "action_temporal_alignment_phase", vectors)
        _campaign_evidence(root / "candidate", "cpbe_residual_fixture", vectors)
        manifest_path = root / "candidate/environments/push_cube/manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["protocol"] = "smoke"
        manifest_path.write_text(json.dumps(payload))

        try:
            evaluate_acwm_cpbe_canary(
                collision_spec_path=root / "prepared/collision-spec.json",
                candidate_campaign_path=root / "prepared/campaigns/cpbe_residual_fixture.json",
                reference_campaign_root=root / "reference",
                candidate_campaign_root=root / "candidate",
                output_root=root / "settled",
            )
        except ACWMCPBECanaryError as exc:
            assert "ACWM_CPBE_MEASUREMENT_CONTRACT_MISMATCH:push_cube:protocol" in str(exc)
        else:
            raise AssertionError("cross-protocol evidence must fail closed")


def _inputs(root: Path) -> dict[str, Path]:
    for directory in (root / "plan", root / "descriptors", root / "inputs"):
        directory.mkdir(parents=True)
    program = {
        "probe_id": "cpbe_residual_fixture",
        "signal_source": "raw_action_sequence",
        "hook_type": "H2",
        "spatial_mask": "all_action_embedding",
        "temporal_basis": "event_phase_curvature",
        "contrast_operator": "signed_mean_preserving_phase",
        "dose_schedule": [-0.05, -0.025, 0.0, 0.025, 0.05],
        "aggregation": "goal_outcome_vector",
        "invariants": ["same_checkpoint", "same_seed"],
        "required_capabilities": ["action_embedding_hook", "paired_seed_control"],
        "estimated_gpu_hours": 0.01,
        "origin": "residual",
        "parent_probe_ids": ["action_temporal_alignment_phase"],
        "rationale": "fixture",
        "diagnostic_only": True,
        "reversible": True,
    }
    order = {"probe_id": program["probe_id"], "program": program}
    _write(
        root / "plan/cpbe-plan.json",
        {
            "artifact_type": "verdiwm-cpbe-plan",
            "state": "ready",
            "experiment_id": "fixture",
            "selected_work_orders": [order],
            "successive_halving_thresholds": {
                "maximum_locality_residual": 0.5,
                "maximum_redundancy_cosine": 0.999,
                "maximum_redundancy_relative_l2": 0.1,
                "minimum_collision_separation": 0.0,
                "minimum_regret_reduction": 0.0,
                "minimum_coverage_gain": 0.0,
            },
        },
    )
    _write(
        root / "descriptors/cpbe_residual_fixture.json",
        {
            "probe_id": "cpbe_residual_fixture",
            "role": "diagnostic",
            "verdict_exposure_allowed": False,
            "program": program,
        },
    )
    environments = {
        name: {"dataset_name": name, "config": f"{name}.yaml", "checkpoint_dir": name}
        for name in (
            "push_cube",
            "stack_cube",
            "push_rope",
            "cloth_move",
            "push_sand",
            "pour_water",
            "robot_arm",
            "reacher",
        )
    }
    _write(
        root / "inputs/base.json",
        {
            "artifact_type": "verdiwm-acwm-fingerprint-campaign",
            "campaign_id": "base",
            "claim_scope": "fixture",
            "probe": {
                "probe_id": "action_temporal_alignment_phase",
                "doses": [-0.05, -0.025, 0.0, 0.025, 0.05],
            },
            "environments": environments,
            "seeds": [101, 202, 303],
            "protocols": {"pilot": {}},
        },
    )
    _write(
        root / "inputs/request.json",
        {
            "experiment_id": "fixture",
            "context": {
                "collision_id": "collision",
                "target_id": "push_cube",
                "primitive": "inv_dyn_reward_finetune",
            },
        },
    )
    _write(
        root / "inputs/provenance.json",
        {
            "experiment_id": "fixture",
            "collision_facts": {
                "target_positive": False,
                "source_effect_signs": {
                    "cloth_move": "negative",
                    "push_sand": "negative",
                    "pour_water": "positive",
                },
            },
        },
    )
    return {
        "plan_path": root / "plan/cpbe-plan.json",
        "base_campaign_path": root / "inputs/base.json",
        "request_path": root / "inputs/request.json",
        "provenance_path": root / "inputs/provenance.json",
        "descriptor_root": root / "descriptors",
    }


def _campaign_evidence(root: Path, probe_id: str, vectors: dict[str, list[float]]) -> None:
    for environment, vector in vectors.items():
        directory = root / "environments" / environment
        directory.mkdir(parents=True)
        _write(
            directory / "manifest.json",
            {
                "environment": environment,
                "protocol": "pilot",
                "checkpoint_sha256": f"checkpoint-{environment}",
                "config_sha256": f"config-{environment}",
                "seeds": [101, 202, 303],
                "doses": [-0.05, -0.025, 0.0, 0.025, 0.05],
                "measurement_count": 15,
            },
        )
        _write(
            directory / "response-chart.json",
            {
                "outcome_names": ["one", "two"],
                "intervention_names": [probe_id],
                "response_coordinate": vector,
                "locality_residuals": {probe_id: 0.1},
            },
        )


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
