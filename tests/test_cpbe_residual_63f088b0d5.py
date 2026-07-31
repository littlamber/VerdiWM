from __future__ import annotations

import pytest

from wmloop.diagnose.probes.cpbe_residual_63f088b0d5 import (
    embedding_delta_event_weights,
    measure_cpbe_residual,
)


def test_embedding_delta_signal_and_goal_outcome_response_are_exact() -> None:
    weights = embedding_delta_event_weights([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    assert weights == pytest.approx([0.0, 0.5, 1.0])
    output = measure_cpbe_residual(dose_responses={-0.05: [-0.05, 0.1], -0.025: [-0.025, 0.05], 0.0: [0.0, -0.0], 0.025: [0.025, -0.05], 0.05: [0.05, -0.1]})
    assert output["probe_id"] == 'cpbe_residual_63f088b0d5'
    assert output["metrics"]["response_vector"] == pytest.approx([1.0, -2.0])
    assert output["metrics"]["target_label_used_for_fit"] is False
    assert output["verdict_exposure_allowed"] is False
