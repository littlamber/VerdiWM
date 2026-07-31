from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wmloop.experiments.acwm_cpbe_bootstrap import (
    ACWMCPBEBootstrapError,
    build_acwm_cpbe_bootstrap,
    publish_acwm_cpbe_bootstrap,
)


def _program() -> dict[str, object]:
    return {
        "probe_id": "action_dimension_interaction",
        "signal_source": "raw_action_trajectory",
        "hook_type": "H2",
        "spatial_mask": "all_action_dimensions",
        "temporal_basis": "centered_sequence",
        "contrast_operator": "orthogonal_pair_rotation",
        "dose_schedule": [-0.05, 0.0, 0.05],
        "aggregation": "goal_outcome_vector",
        "invariants": ["same_seed", "same_trajectory", "same_evaluator"],
        "required_capabilities": ["raw_action_embedding_hook", "paired_seed_control"],
        "estimated_gpu_hours": 0.01,
        "origin": "retrieval",
        "diagnostic_only": True,
        "reversible": True,
    }


def _template() -> dict[str, object]:
    program = _program()
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-request",
        "experiment_id": "acwm_push_sand_dimension_collision_cpbe_v1",
        "evidence_class": "live",
        "context": {
            "collision_id": "push_sand_dimension_coverage_gap",
            "target_id": "push_sand",
            "backbone_family": "acwm_phys",
            "capability_class": "latent_dit_action_conditioned",
            "failure_signature": "filled_from_frozen_replay",
            "primitive": "action_dimension_balancing",
            "available_hooks": ["H2"],
            "capabilities": ["raw_action_embedding_hook", "paired_seed_control"],
            "unexplained_residual": {"spatial_mask": 1.0},
            "residual_evidence_refs": ["filled://by-adapter"],
            "max_canaries": 2,
        },
        "current_probes": [program],
        "grammar": {
            "signal_source": ["raw_action_trajectory", "action_embedding_response"],
            "hook_type": ["H2"],
            "spatial_mask": ["all_action_dimensions", "active_action_dimensions"],
            "temporal_basis": ["centered_sequence", "event_phase"],
            "contrast_operator": ["orthogonal_pair_rotation", "signed_difference"],
            "aggregation": ["goal_outcome_vector", "event_weighted_outcome"],
        },
    }


def _selector() -> dict[str, object]:
    return {
        "probe_evolution_work_orders": [
            {
                "target_environment": "push_sand",
                "primitive": "action_dimension_balancing",
                "reason": "target_required_probe_path_is_nonlocal_or_missing",
                "required_probe_paths": ["action_dimension_anisotropy"],
                "successor_probe_axis": "action_dimension_interaction",
                "action": "materialize_counterexample_driven_probe_then_replay",
            }
        ]
    }


def _redundancy() -> dict[str, object]:
    return {
        "candidate_probe_id": "action_dimension_interaction",
        "comparisons": [
            {
                "environment": "push_sand",
                "candidate_locality_pass": False,
                "candidate_locality_residual": 6.7,
                "redundant": False,
                "cosine_similarity": -0.72,
            }
        ],
    }


def _environment() -> dict[str, object]:
    return {
        "artifact_type": "verdiwm-acwm-fingerprint-environment-manifest",
        "state": "ready",
        "environment": "push_sand",
        "probe_id": "action_dimension_interaction",
        "elapsed_seconds": 36.0,
    }


def test_builds_measured_negative_history_and_heuristic_residual() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        refs = []
        for name in ("selector.json", "redundancy.json", "environment.json"):
            path = root / name
            path.write_text("{}\n", encoding="utf-8")
            refs.append(path.as_posix())
        request, trial, report = build_acwm_cpbe_bootstrap(
            request_template=_template(),
            selector_replay=_selector(),
            redundancy_report=_redundancy(),
            environment_manifest=_environment(),
            evidence_refs=refs,
        )

    assert request["context"]["failure_signature"] == "target_required_probe_path_is_nonlocal_or_missing"
    residual = request["context"]["unexplained_residual"]
    assert sum(residual.values()) == pytest.approx(1.0)
    assert residual["spatial_mask"] > residual["contrast_operator"]
    assert trial["outcomes"] == {
        "locality_pass": False,
        "nonredundant": True,
        "collision_resolved": None,
        "regret_reduction": None,
        "coverage_gain": None,
        "gpu_hours": 0.01,
    }
    assert report["residual_policy"]["policy_id"] == "acwm_counterexample_axis_attribution_v1"


def test_rejects_mismatched_probe_manifest() -> None:
    environment = _environment()
    environment["probe_id"] = "different_probe"
    with pytest.raises(ACWMCPBEBootstrapError, match="ENVIRONMENT_MANIFEST_MISMATCH"):
        build_acwm_cpbe_bootstrap(
            request_template=_template(),
            selector_replay=_selector(),
            redundancy_report=_redundancy(),
            environment_manifest=environment,
            evidence_refs=["a", "b", "c"],
        )


def test_publishes_cpbe_inputs_with_source_hashes() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = {
            "template.json": _template(),
            "selector.json": _selector(),
            "redundancy.json": _redundancy(),
            "environment.json": _environment(),
        }
        for name, payload in inputs.items():
            (root / name).write_text(json.dumps(payload), encoding="utf-8")
        manifest = publish_acwm_cpbe_bootstrap(
            template_path=root / "template.json",
            selector_replay_path=root / "selector.json",
            redundancy_report_path=root / "redundancy.json",
            environment_manifest_path=root / "environment.json",
            output_root=root / "bundle",
        )
        assert manifest["state"] == "ready"
        assert (root / "bundle/inputs/cpbe-request.json").is_file()
        history = json.loads((root / "bundle/inputs/probe-trials.jsonl").read_text())
        assert history["evidence_class"] == "live"
        report = json.loads((root / "bundle/acwm-cpbe-bootstrap.json").read_text())
        assert len(report["source_refs"]) == 3
