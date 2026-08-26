"""Adaptive long-training gates driven by held-out progress, not train loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingPolicy:
    initial_steps: int = 100
    evaluation_interval: int = 100
    max_steps: int | None = None
    patience_evaluations: int = 4
    overfit_evaluations: int = 2
    min_delta: float = 0.0
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("training metric direction must be maximize or minimize")
        if self.evaluation_interval < 1 or self.patience_evaluations < 1 or self.overfit_evaluations < 1:
            raise ValueError("evaluation and patience values must be positive")


@dataclass
class AdaptiveTrainingController:
    policy: TrainingPolicy
    history: list[dict[str, float]] = field(default_factory=list)
    best_step: int | None = None
    best_heldout: float | None = None
    stale_evaluations: int = 0
    overfit_evaluations: int = 0

    def promote_after_probe(self, *, step: int, baseline_metric: float, candidate_metric: float, practical_threshold: float) -> dict[str, Any]:
        """Turn an early held-out signal into long training or an explicit stop."""
        improvement = candidate_metric - baseline_metric if self.policy.direction == "maximize" else baseline_metric - candidate_metric
        if improvement > abs(practical_threshold):
            return {"action": "continue_long_train", "reason": "early_heldout_improvement", "step": step, "improvement": improvement, "next_step": step + self.policy.evaluation_interval}
        return {"action": "stop", "reason": "early_signal_not_practically_positive", "step": step, "improvement": improvement}

    def observe(self, *, step: int, train_metric: float, heldout_metric: float) -> dict[str, Any]:
        if self.history and step <= self.history[-1]["step"]:
            raise ValueError("training steps must increase")
        current = {"step": float(step), "train_metric": float(train_metric), "heldout_metric": float(heldout_metric)}
        self.history.append(current)
        improved = self.best_heldout is None or self._better(heldout_metric, self.best_heldout, self.policy.min_delta)
        if improved:
            self.best_heldout = float(heldout_metric)
            self.best_step = step
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        train_improving = len(self.history) >= 2 and self._better(train_metric, self.history[-2]["train_metric"], 0.0)
        heldout_worsening = len(self.history) >= 2 and not self._better(heldout_metric, self.history[-2]["heldout_metric"], 0.0)
        self.overfit_evaluations = self.overfit_evaluations + 1 if train_improving and heldout_worsening else 0
        if self.overfit_evaluations >= self.policy.overfit_evaluations:
            return self._decision("stop", "near_overfit", improved)
        if self.stale_evaluations >= self.policy.patience_evaluations:
            return self._decision("stop", "heldout_plateau", improved)
        if self.policy.max_steps is not None and step >= self.policy.max_steps:
            return self._decision("stop", "resource_cap", improved)
        return self._decision("continue", "heldout_improving_or_not_yet_plateaued", improved)

    def _better(self, value: float, reference: float, margin: float) -> bool:
        return value > reference + margin if self.policy.direction == "maximize" else value < reference - margin

    def _decision(self, action: str, reason: str, improved: bool) -> dict[str, Any]:
        return {"action": action, "reason": reason, "improved": improved, "best_step": self.best_step, "best_heldout": self.best_heldout, "next_checkpoint": self.best_step}
