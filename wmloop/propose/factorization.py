"""Derive and queue single-axis confirmation trials after a multi-axis win."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class FactorizationError(ValueError):
    """A proposed causal-factor confirmation was malformed."""


@dataclass(frozen=True)
class FactorizationTrial:
    trial_id: str
    parent_proposal_id: str
    axis_primitive: str
    proposal: Mapping[str, Any]


def derive_single_axis_trials(proposal: Mapping[str, Any]) -> tuple[FactorizationTrial, ...]:
    """Turn an accepted multi-axis proposal into deterministic one-axis trials.

    These children preserve the failure context, goal, prediction, budget and
    frozen library version.  They contain exactly one original intervention and
    are tagged exploratory=false only after their own validator pass.
    """

    proposal_id = proposal.get("proposal_id")
    interventions = proposal.get("interventions")
    if not isinstance(proposal_id, str) or not proposal_id or not isinstance(interventions, list) or len(interventions) < 2:
        raise FactorizationError("FACTORIZATION_MULTI_AXIS_REQUIRED")
    children: list[FactorizationTrial] = []
    for index, intervention in enumerate(interventions):
        if not isinstance(intervention, Mapping) or not isinstance(intervention.get("primitive"), str):
            raise FactorizationError("FACTORIZATION_INTERVENTION_INVALID")
        child = dict(proposal)
        child["interventions"] = [dict(intervention)]
        child["factorized_from"] = proposal_id
        child["exploratory"] = False
        payload = json.dumps(child, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        child_id = f"{proposal_id}-factor-{index + 1}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]}"
        child["proposal_id"] = child_id
        children.append(
            FactorizationTrial(
                trial_id=child_id,
                parent_proposal_id=proposal_id,
                axis_primitive=str(intervention["primitive"]),
                proposal=child,
            )
        )
    return tuple(children)


class FactorizationQueue:
    """FIFO priority queue that the scheduler consumes before new exploration."""

    def __init__(self) -> None:
        self._pending: deque[FactorizationTrial] = deque()

    def enqueue_accepted_multi_axis(self, proposal: Mapping[str, Any]) -> tuple[FactorizationTrial, ...]:
        trials = derive_single_axis_trials(proposal)
        self._pending.extend(trials)
        return trials

    def pop(self) -> FactorizationTrial | None:
        return self._pending.popleft() if self._pending else None

    def pending(self) -> tuple[FactorizationTrial, ...]:
        return tuple(self._pending)
