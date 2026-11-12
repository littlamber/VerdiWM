"""Resumable pipeline state and evidence components: budget."""

from __future__ import annotations

import math
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.pipeline.io import _load_json
from wmloop.execute.pipeline.types import AutonomousPipelineError


def _pipeline_budget_total(
    *,
    evaluator: Path,
    probe_contract: Path | None,
    override: float | None,
) -> float:
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_BUDGET_TOTAL_INVALID")
        return float(override)
    evaluator_payload = _load_json(
        evaluator, "AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID"
    )
    template_value = evaluator_payload.get("scheduler_template")
    if not isinstance(template_value, str) or not template_value:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID")
    template_path = Path(template_value)
    template_path = (
        template_path.resolve()
        if template_path.is_absolute()
        else (evaluator.parent / template_path).resolve()
    )
    template = _load_json(template_path, "AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID")
    try:
        validate_document("auto_experiment_candidate_batch", template)
    except ContractValidationError as exc:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID"
        ) from exc
    total = float(template["total_budget_gpu_hours"])
    if probe_contract is not None:
        probe = _load_json(
            probe_contract, "AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID"
        )
        try:
            validate_document("diagnostic_probe_contract", probe)
        except ContractValidationError as exc:
            raise AutonomousPipelineError(
                "AUTONOMOUS_PIPELINE_BUDGET_CONTRACT_INVALID"
            ) from exc
        total += float(probe["estimated_gpu_hours"])
    if not math.isfinite(total) or total <= 0:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_BUDGET_TOTAL_INVALID")
    return total
