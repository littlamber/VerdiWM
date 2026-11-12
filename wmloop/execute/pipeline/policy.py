"""Resumable pipeline state and evidence components: policy."""

from __future__ import annotations

from wmloop.execute.pipeline.types import AutonomousPipelineOptions


def _nondefault_resource_policy(
    options: AutonomousPipelineOptions,
) -> dict[str, object]:
    if (
        options.budget_max_trial_gpu_hours == 120.0
        and options.budget_high_trial_limit == 2
        and options.budget_require_high_cost_approval
    ):
        return {}
    return {
        "resource_policy": {
            "max_trial_gpu_hours": options.budget_max_trial_gpu_hours,
            "high_trial_limit": options.budget_high_trial_limit,
            "require_high_cost_approval": (options.budget_require_high_cost_approval),
        }
    }
