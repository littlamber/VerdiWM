"""Deterministically materialize the grounded Hybrid Memory transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODULE = '''"""Source-grounded Hybrid Memory retrieval for Ctrl-World history latents."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RetrievedHistory:
    history: torch.Tensor
    actions: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    memory_tokens: torch.Tensor


def retrieve_relevant_history(
    history: torch.Tensor,
    history_actions: torch.Tensor,
    query_state: torch.Tensor,
    query_action: torch.Tensor,
    *,
    max_items: int,
    token_grid_size: int = 4,
    spatial_weight: float = 0.55,
    action_weight: float = 0.30,
    temporal_weight: float = 0.15,
) -> RetrievedHistory:
    """Compress history into tokens and retrieve relevant states chronologically."""

    _validate_inputs(
        history,
        history_actions,
        query_state,
        query_action,
        max_items=max_items,
        token_grid_size=token_grid_size,
        weights=(spatial_weight, action_weight, temporal_weight),
    )
    batch, steps, channels, height, width = history.shape
    pooled = functional.adaptive_avg_pool2d(
        history.reshape(batch * steps, channels, height, width).float(),
        (token_grid_size, token_grid_size),
    ).reshape(batch, steps, -1)
    query_token = functional.adaptive_avg_pool2d(
        query_state.float(), (token_grid_size, token_grid_size)
    ).reshape(batch, -1)
    spatial = _bounded_cosine(pooled, query_token.unsqueeze(1))
    action = _bounded_cosine(
        history_actions.float(), query_action.float().unsqueeze(1)
    )
    recency = torch.linspace(
        0.0, 1.0, steps, device=history.device, dtype=torch.float32
    ).view(1, steps).expand(batch, -1)
    total = float(spatial_weight + action_weight + temporal_weight)
    scores = (
        spatial * float(spatial_weight)
        + action * float(action_weight)
        + recency * float(temporal_weight)
    ) / total

    # Reversing before a stable sort makes exact ties prefer recent items.
    reversed_rank = torch.argsort(
        scores.flip(1), dim=1, descending=True, stable=True
    )[:, :max_items]
    selected = (steps - 1) - reversed_rank
    selected = torch.sort(selected, dim=1).values
    history_index = selected.view(batch, max_items, 1, 1, 1).expand(
        -1, -1, channels, height, width
    )
    action_index = selected.unsqueeze(-1).expand(-1, -1, history_actions.shape[-1])
    token_index = selected.unsqueeze(-1).expand(-1, -1, pooled.shape[-1])
    selected_scores = scores.gather(1, selected)
    weights = torch.softmax(selected_scores, dim=1).to(history.dtype)
    result = RetrievedHistory(
        history=history.gather(1, history_index),
        actions=history_actions.gather(1, action_index),
        indices=selected,
        weights=weights,
        memory_tokens=pooled.gather(1, token_index),
    )
    if not torch.isfinite(result.weights).all():
        raise ValueError("HYBRID_MEMORY_WEIGHT_NONFINITE")
    return result


def _bounded_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    cosine = numerator / denominator.clamp_min(1e-12)
    return ((cosine.clamp(-1.0, 1.0) + 1.0) * 0.5).float()


def _validate_inputs(
    history: torch.Tensor,
    history_actions: torch.Tensor,
    query_state: torch.Tensor,
    query_action: torch.Tensor,
    *,
    max_items: int,
    token_grid_size: int,
    weights: tuple[float, float, float],
) -> None:
    if history.ndim != 5:
        raise ValueError("HYBRID_MEMORY_HISTORY_RANK_INVALID")
    batch, steps, channels, height, width = history.shape
    if history_actions.ndim != 3 or history_actions.shape[:2] != (batch, steps):
        raise ValueError("HYBRID_MEMORY_ACTION_SHAPE_INVALID")
    if query_state.shape != (batch, channels, height, width):
        raise ValueError("HYBRID_MEMORY_QUERY_STATE_SHAPE_INVALID")
    if query_action.shape != (batch, history_actions.shape[-1]):
        raise ValueError("HYBRID_MEMORY_QUERY_ACTION_SHAPE_INVALID")
    if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= steps:
        raise ValueError("HYBRID_MEMORY_MAX_ITEMS_INVALID")
    if (
        isinstance(token_grid_size, bool)
        or not isinstance(token_grid_size, int)
        or not 1 <= token_grid_size <= min(height, width)
    ):
        raise ValueError("HYBRID_MEMORY_TOKEN_GRID_INVALID")
    if not history.is_floating_point() or not history_actions.is_floating_point():
        raise ValueError("HYBRID_MEMORY_DTYPE_INVALID")
    if not query_state.is_floating_point() or not query_action.is_floating_point():
        raise ValueError("HYBRID_MEMORY_DTYPE_INVALID")
    if not all(
        torch.isfinite(value).all()
        for value in (history, history_actions, query_state, query_action)
    ):
        raise ValueError("HYBRID_MEMORY_INPUT_NONFINITE")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("HYBRID_MEMORY_RELEVANCE_WEIGHT_INVALID")
    if sum(weights) <= 0.0:
        raise ValueError("HYBRID_MEMORY_RELEVANCE_WEIGHT_INVALID")
'''


TEST = '''from __future__ import annotations

import pytest
import torch

from models.hybrid_relevance_memory import retrieve_relevant_history


def test_retrieval_preserves_token_compression_and_relevance() -> None:
    history = torch.tensor(
        [[
            [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[2.0, 2.0], [2.0, 2.0]]],
        ]]
    )
    actions = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    query_state = history[:, 0]
    query_action = torch.tensor([[1.0, 0.0]])

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
    assert result.memory_tokens.shape == (1, 2, 2)
    torch.testing.assert_close(result.history, history[:, :2])
    torch.testing.assert_close(result.weights.sum(dim=1), torch.ones(1))


def test_exact_ties_prefer_recent_items_then_restore_chronology() -> None:
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


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("max_items", "HYBRID_MEMORY_MAX_ITEMS_INVALID"),
        ("token_grid_size", "HYBRID_MEMORY_TOKEN_GRID_INVALID"),
        ("weight", "HYBRID_MEMORY_RELEVANCE_WEIGHT_INVALID"),
        ("nonfinite", "HYBRID_MEMORY_INPUT_NONFINITE"),
    ],
)
def test_invalid_inputs_fail_closed(field: str, code: str) -> None:
    history = torch.zeros(1, 4, 1, 2, 2)
    actions = torch.zeros(1, 4, 2)
    kwargs = {
        "max_items": 2,
        "token_grid_size": 1,
        "spatial_weight": 0.6,
        "action_weight": 0.3,
        "temporal_weight": 0.1,
    }
    if field == "max_items":
        kwargs["max_items"] = 0
    elif field == "token_grid_size":
        kwargs["token_grid_size"] = 3
    elif field == "weight":
        kwargs["spatial_weight"] = -1.0
    else:
        history[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=code):
        retrieve_relevant_history(
            history,
            actions,
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 2),
            **kwargs,
        )
'''


def materialize(
    *, workspace: Path, descriptor_path: Path, candidate_id: str, idea_id: str, prompt_path: Path
) -> None:
    prompt = prompt_path.read_text(encoding="utf-8")
    required = ("memory_token_compression", "spatiotemporal_relevance_retrieval")
    if any(component not in prompt for component in required):
        raise RuntimeError("HYBRID_MEMORY_SOURCE_COMPONENT_MISSING")
    module_path = workspace / "models" / "hybrid_relevance_memory.py"
    test_path = workspace / "tests" / "test_hybrid_relevance_memory.py"
    if module_path.exists() or test_path.exists() or descriptor_path.exists():
        raise RuntimeError("HYBRID_MEMORY_MATERIALIZATION_TARGET_EXISTS")
    module_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(MODULE, encoding="utf-8")
    test_path.write_text(TEST, encoding="utf-8")
    descriptor = {
        "schema_version": 1,
        "artifact_type": "verdiwm-materialized-method-descriptor",
        "candidate_id": candidate_id,
        "idea_id": idea_id,
        "implementation_files": [
            "models/hybrid_relevance_memory.py",
            "tests/test_hybrid_relevance_memory.py",
        ],
        "intent_to_code": [
            {
                "source_component_id": "memory_token_compression",
                "intent": "Compress each historical latent into a bounded spatial token.",
                "touchpoint": "models.hybrid_relevance_memory.retrieve_relevant_history:adaptive_avg_pool2d",
            },
            {
                "source_component_id": "spatiotemporal_relevance_retrieval",
                "intent": "Retrieve top relevant history using spatial, action, and temporal scores.",
                "touchpoint": "models.hybrid_relevance_memory.retrieve_relevant_history:scores_and_stable_topk",
            },
        ],
        "runtime_contract": (
            "An external frozen evaluator supplies history latents, aligned actions, the current "
            "latent, and the query action; the adapter returns a bounded chronological subset."
        ),
        "negative_check": (
            "A fixed recency-only window, missing token compression, non-finite input, invalid "
            "weights, or a changed evaluator must fail admission."
        ),
        "applicability_conditions": [
            "the target exposes finite latent history and aligned action history",
            "history conditioning accepts a fixed-size chronological subset",
        ],
        "failure_boundaries": [
            "does not claim HyDRA hidden-subject tracking or HM-World dataset transfer",
            "does not create missing history, actions, rewards, or return-to-go",
            "does not modify the frozen ACWM evaluator or split",
        ],
        "declared_compromises": [],
    }
    descriptor_path.write_text(
        json.dumps(descriptor, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("descriptor_path", type=Path)
    parser.add_argument("candidate_id")
    parser.add_argument("idea_id")
    parser.add_argument("prompt_path", type=Path)
    args = parser.parse_args(argv)
    materialize(
        workspace=args.workspace,
        descriptor_path=args.descriptor_path,
        candidate_id=args.candidate_id,
        idea_id=args.idea_id,
        prompt_path=args.prompt_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
