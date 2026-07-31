from __future__ import annotations

import pytest

from wmloop.experiments.cpbe_counterexample import (
    CPBECounterexampleError,
    _capability_filter_grammar,
    _canary_failure_credit,
    _expanded_failure_credit,
    _single_axis_candidates,
)


def test_canary_collision_failure_reallocates_credit_to_mechanism_axes() -> None:
    updated, audit = _canary_failure_credit(
        residual={
            "signal_source": 0.23,
            "hook_type": 0.02,
            "spatial_mask": 0.25,
            "temporal_basis": 0.02,
            "contrast_operator": 0.05,
            "aggregation": 0.43,
        },
        edited_axis="aggregation",
        locality_passed=True,
        nonredundant=True,
        collision_separation=-0.006,
    )

    assert sum(updated.values()) == pytest.approx(1.0)
    assert updated["signal_source"] > updated["aggregation"]
    assert updated["contrast_operator"] > 0.05
    assert audit["failure_mode"] == "local_nonredundant_collision_not_separated"
    assert audit["residual_updated"] is True
    assert audit["causal_attribution"] is False


def test_unattributable_canary_failure_does_not_change_residual_credit() -> None:
    residual = {
        "signal_source": 0.23,
        "hook_type": 0.02,
        "spatial_mask": 0.25,
        "temporal_basis": 0.02,
        "contrast_operator": 0.05,
        "aggregation": 0.43,
    }
    updated, audit = _canary_failure_credit(
        residual=residual,
        edited_axis="aggregation",
        locality_passed=False,
        nonredundant=True,
        collision_separation=-0.006,
    )

    assert updated == pytest.approx(residual)
    assert audit["residual_updated"] is False


def test_expanded_failure_reassigns_credit_away_from_failed_temporal_axis() -> None:
    updated, audit = _expanded_failure_credit(
        residual={
            "signal_source": 0.10,
            "hook_type": 0.03,
            "spatial_mask": 0.07,
            "temporal_basis": 0.32,
            "contrast_operator": 0.28,
            "aggregation": 0.20,
        },
        edited_axis="temporal_basis",
        locality_failed=True,
    )

    assert sum(updated.values()) == pytest.approx(1.0)
    assert updated["aggregation"] > updated["spatial_mask"] > updated["temporal_basis"]
    assert updated["signal_source"] > updated["contrast_operator"]
    assert audit["policy_id"] == "expanded_counterexample_axis_credit_v1"


def test_single_axis_filter_removes_entangled_llm_hypothesis() -> None:
    current = [
        {
            "probe_id": "parent",
            "signal_source": "raw",
            "hook_type": "H2",
            "spatial_mask": "all",
            "temporal_basis": "phase",
            "contrast_operator": "signed",
            "aggregation": "vector",
        }
    ]
    candidates = [
        {**current[0], "probe_id": "single", "aggregation": "source_sign_margin"},
        {
            **current[0],
            "probe_id": "entangled",
            "signal_source": "transition",
            "aggregation": "source_sign_margin",
        },
    ]

    kept, removed = _single_axis_candidates(candidates, current=current)

    assert [row["probe_id"] for row in kept] == ["single"]
    assert removed == ["entangled"]


def test_counterexample_credit_rejects_missing_axis_measurement() -> None:
    with pytest.raises(CPBECounterexampleError, match="NUMBER_INVALID:aggregation"):
        _expanded_failure_credit(
            residual={
                "signal_source": 0.10,
                "hook_type": 0.03,
                "spatial_mask": 0.07,
                "temporal_basis": 0.32,
                "contrast_operator": 0.28,
            },
            edited_axis="temporal_basis",
            locality_failed=False,
        )


def test_capability_filter_removes_unsupported_horizon_aggregation() -> None:
    grammar = {
        "signal_source": ["raw"],
        "hook_type": ["H2"],
        "spatial_mask": ["all"],
        "temporal_basis": ["phase"],
        "contrast_operator": ["signed"],
        "aggregation": [
            "goal_outcome_vector",
            "source_sign_margin",
            "horizon_weighted_goal_outcome",
        ],
    }

    filtered, removed = _capability_filter_grammar(
        grammar=grammar,
        capabilities=["action_embedding_hook"],
    )

    assert filtered["aggregation"] == ["goal_outcome_vector", "source_sign_margin"]
    assert removed == {"aggregation": ["horizon_weighted_goal_outcome"]}


def test_capability_filter_removes_unmapped_action_dimension_mask() -> None:
    grammar = {
        "signal_source": ["raw"],
        "hook_type": ["H2"],
        "spatial_mask": ["all", "active_action_dimensions"],
        "temporal_basis": ["phase"],
        "contrast_operator": ["signed"],
        "aggregation": ["vector"],
    }

    filtered, removed = _capability_filter_grammar(
        grammar=grammar,
        capabilities=["action_embedding_hook"],
    )

    assert filtered["spatial_mask"] == ["all"]
    assert removed == {"spatial_mask": ["active_action_dimensions"]}
