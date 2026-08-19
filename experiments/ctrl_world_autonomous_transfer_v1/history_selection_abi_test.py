from __future__ import annotations

import pytest
import torch

from models.hybrid_relevance_memory import retrieve_relevant_history


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.tensor(
        [[
            [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[2.0, 2.0], [2.0, 2.0]]],
        ]]
    )
    actions = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    return history, actions, history[:, 0], torch.tensor([[1.0, 0.0]])


def test_history_selection_abi_contract() -> None:
    history, actions, query_state, query_action = _inputs()
    result = retrieve_relevant_history(
        history,
        actions,
        query_state,
        query_action,
        max_items=2,
        token_grid_size=1,
        spatial_weight=0.6,
        action_weight=0.4,
        temporal_weight=0.0,
    )
    assert result.indices.tolist() == [[0, 1]]
    assert result.history.shape == (1, 2, 2, 2, 2)
    assert result.actions.shape == (1, 2, 2)
    assert result.memory_tokens.shape == (1, 2, 2)
    assert torch.isfinite(result.weights).all()
    torch.testing.assert_close(result.weights.sum(dim=1), torch.ones(1))


def test_history_selection_invalid_inputs_fail_closed() -> None:
    history, actions, query_state, query_action = _inputs()
    with pytest.raises(ValueError):
        retrieve_relevant_history(
            history,
            actions,
            query_state,
            query_action,
            max_items=0,
            token_grid_size=1,
            spatial_weight=0.6,
            action_weight=0.4,
            temporal_weight=0.0,
        )
    history[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        retrieve_relevant_history(
            history,
            actions,
            query_state,
            query_action,
            max_items=2,
            token_grid_size=1,
            spatial_weight=0.6,
            action_weight=0.4,
            temporal_weight=0.0,
        )


def test_history_selection_exact_ties_are_deterministic_and_chronological() -> None:
    history = torch.zeros(1, 5, 1, 2, 2)
    actions = torch.zeros(1, 5, 2)
    result = retrieve_relevant_history(
        history,
        actions,
        torch.zeros(1, 1, 2, 2),
        torch.zeros(1, 2),
        max_items=3,
        token_grid_size=1,
        spatial_weight=1.0,
        action_weight=0.0,
        temporal_weight=0.0,
    )
    assert result.indices.tolist() == [[2, 3, 4]]
