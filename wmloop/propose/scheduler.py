"""Deterministic UCB scheduling over typed intervention cells."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class InterventionCell:
    environment: str
    layer: str
    primitive_family: str
    parameter_bucket: str


@dataclass(frozen=True)
class CellStats:
    visits: int
    mean_verified_improvement: float


class UcbScheduler:
    """Select underexplored cells, with a forced jump after local saturation."""

    def __init__(self, *, exploration_coefficient: float, saturation_rounds: int) -> None:
        if exploration_coefficient < 0 or saturation_rounds < 1:
            raise ValueError("UCB_CONFIGURATION_INVALID")
        self._coefficient = exploration_coefficient
        self._saturation_rounds = saturation_rounds
        self._stats: dict[InterventionCell, CellStats] = {}
        self._nonpositive_streak: dict[str, int] = {}
        self._last_layer: dict[str, str] = {}

    def observe(self, cell: InterventionCell, *, verified_improvement: float) -> None:
        if not math.isfinite(verified_improvement):
            raise ValueError("UCB_OBSERVATION_INVALID")
        previous = self._stats.get(cell, CellStats(visits=0, mean_verified_improvement=0.0))
        visits = previous.visits + 1
        mean = previous.mean_verified_improvement + (verified_improvement - previous.mean_verified_improvement) / visits
        self._stats[cell] = CellStats(visits=visits, mean_verified_improvement=mean)
        environment = cell.environment
        self._last_layer[environment] = cell.layer
        self._nonpositive_streak[environment] = (
            self._nonpositive_streak.get(environment, 0) + 1 if verified_improvement <= 0 else 0
        )

    def stats(self, cell: InterventionCell) -> CellStats:
        return self._stats.get(cell, CellStats(visits=0, mean_verified_improvement=0.0))

    def choose(self, candidates: Iterable[InterventionCell]) -> InterventionCell:
        cells = sorted(set(candidates))
        if not cells:
            raise ValueError("UCB_CANDIDATES_EMPTY")
        cross_layer = [
            cell
            for cell in cells
            if self._nonpositive_streak.get(cell.environment, 0) >= self._saturation_rounds
            and cell.layer != self._last_layer.get(cell.environment)
        ]
        pool = cross_layer or cells
        total_visits = sum(self.stats(cell).visits for cell in cells)
        return min(pool, key=lambda cell: (-self._score(cell, total_visits), cell))

    def _score(self, cell: InterventionCell, total_visits: int) -> float:
        stats = self.stats(cell)
        if stats.visits == 0:
            return math.inf
        return stats.mean_verified_improvement + self._coefficient * math.sqrt(
            math.log(max(total_visits, 1)) / stats.visits
        )
