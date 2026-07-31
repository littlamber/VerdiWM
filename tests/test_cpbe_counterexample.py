from __future__ import annotations

import pytest

from wmloop.experiments.cpbe_counterexample import (
    CPBECounterexampleError,
    _capability_filter_grammar,
    _expanded_failure_credit,
    _single_axis_candidates,
)


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
