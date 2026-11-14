"""Validate the evaluator and complete candidate ladder before evolution starts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import checked_path
from wmloop.verify.goal_success import resolve_frozen_goal


@dataclass(frozen=True)
class EvolutionEvaluatorInputs:
    evaluator_path: Path
    evaluator: dict[str, object]
    template_path: Path
    template: dict[str, object]
    frozen_goal_path: Path | None


def load_evolution_evaluator(path: Path) -> EvolutionEvaluatorInputs:
    """Resolve a checked scheduler template and require screen → gate → confirm."""

    evaluator_path, evaluator = _read_document(path, "EVOLUTION_EVALUATOR_INVALID")
    template_value = evaluator.get("scheduler_template")
    if not isinstance(template_value, str) or not template_value.strip():
        raise ValueError("EVOLUTION_EVALUATOR_TEMPLATE_INVALID")
    template_path = Path(template_value)
    if not template_path.is_absolute():
        template_path = evaluator_path.parent / template_path
    template_path, template = _read_document(
        template_path, "EVOLUTION_EVALUATOR_TEMPLATE_INVALID"
    )
    _require_confirmation_ladder(template)
    try:
        validate_document("auto_experiment_candidate_batch", template)
    except ContractValidationError as exc:
        raise ValueError("EVOLUTION_CANDIDATE_TEMPLATE_INVALID") from exc
    try:
        frozen_goal_path = resolve_frozen_goal(evaluator_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("EVOLUTION_FROZEN_GOAL_INVALID") from exc
    return EvolutionEvaluatorInputs(
        evaluator_path, evaluator, template_path, template, frozen_goal_path
    )


def validate_evolution_evaluator(path: Path) -> Path:
    """Return the bound scheduler template, raising ValueError before execution."""

    return load_evolution_evaluator(path).template_path


def _read_document(path: Path, code: str) -> tuple[Path, dict[str, object]]:
    try:
        resolved = checked_path(path, code=code)
        if not resolved.is_file():
            raise ValueError(code)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return resolved, value


def _require_confirmation_ladder(batch: Mapping[str, object]) -> None:
    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("EVOLUTION_CANDIDATES_INVALID")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("EVOLUTION_CANDIDATES_INVALID")
        stages = candidate.get("stages")
        if not isinstance(stages, list):
            raise ValueError("EVOLUTION_CONFIRMATION_LADDER_REQUIRED")
        names = [stage.get("stage") for stage in stages if isinstance(stage, Mapping)]
        if names != ["screen", "gate", "confirm"]:
            raise ValueError("EVOLUTION_CONFIRMATION_LADDER_REQUIRED")
