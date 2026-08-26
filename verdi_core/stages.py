"""Evidence-gated experiment stages shared by all model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import classify_paired_effect


STAGES = ("static_check", "environment_smoke", "gpu_smoke", "short_train", "replicate", "full_train", "heldout_evaluate")


@dataclass(frozen=True)
class StageDecision:
    stage: str
    state: str
    reason: str
    evidence: dict[str, Any]


class ExperimentStages:
    """Small state machine; adapters provide commands and evaluators provide evidence."""

    def __init__(self) -> None:
        self.completed: list[str] = []

    def settle(self, stage: str, *, success: bool, evidence: dict[str, Any] | None = None) -> StageDecision:
        if stage not in STAGES:
            return StageDecision(stage, "abstain", "unknown_stage", evidence or {})
        expected = STAGES[len(self.completed)] if len(self.completed) < len(STAGES) else None
        if stage != expected:
            return StageDecision(stage, "blocked", f"expected_{expected}", evidence or {})
        if not success:
            return StageDecision(stage, "runtime_failed", "stage_failed", evidence or {})
        self.completed.append(stage)
        return StageDecision(stage, "settled", "stage_complete", evidence or {})

    def classify_heldout(self, deltas: list[float], *, practical_threshold: float | None, protected_ok: bool) -> StageDecision:
        if "heldout_evaluate" not in self.completed:
            return StageDecision("heldout_evaluate", "blocked", "heldout_stage_not_complete", {})
        if practical_threshold is None:
            return StageDecision("heldout_evaluate", "abstain", "practical_threshold_not_frozen", {"replicates": len(deltas)})
        result = classify_paired_effect(deltas, practical_threshold=practical_threshold, protected_ok=protected_ok)
        return StageDecision("heldout_evaluate", result["outcome"], result.get("reason", "evidence_classified"), result)
