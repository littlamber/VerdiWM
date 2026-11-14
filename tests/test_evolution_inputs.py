"""Validate unattended scheduler inputs before any daemon is scheduled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.execute.evolution_inputs import (
    load_evolution_evaluator,
    validate_evolution_evaluator,
)


@pytest.fixture
def evaluator(tmp_path: Path) -> Path:
    example = (
        Path(__file__).resolve().parents[1]
        / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json"
    )
    template = json.loads(example.read_bytes())
    template["candidates"] = template["candidates"][:1]
    (tmp_path / "candidate-batch.json").write_text(json.dumps(template))
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps({"scheduler_template": "candidate-batch.json"}))
    return path


def test_relative_template_is_checked_and_resolved(evaluator):
    assert validate_evolution_evaluator(evaluator) == evaluator.parent / "candidate-batch.json"


@pytest.mark.parametrize("value", [{}, {"scheduler_template": 3}, []])
def test_invalid_evaluator_is_rejected_before_execution(evaluator, value):
    evaluator.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="EVOLUTION_EVALUATOR"):
        validate_evolution_evaluator(evaluator)


def test_screen_only_template_is_rejected(evaluator):
    template = evaluator.parent / "candidate-batch.json"
    value = json.loads(template.read_text())
    value["candidates"][0]["stages"] = value["candidates"][0]["stages"][:1]
    template.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="EVOLUTION_CONFIRMATION_LADDER_REQUIRED"):
        validate_evolution_evaluator(evaluator)


def test_template_schema_is_checked(evaluator):
    template = evaluator.parent / "candidate-batch.json"
    value = json.loads(template.read_text())
    value["max_selected_candidates"] = "two"
    template.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="EVOLUTION_CANDIDATE_TEMPLATE_INVALID"):
        validate_evolution_evaluator(evaluator)


def test_linked_parent_is_rejected_before_resolution(evaluator):
    linked = evaluator.parent / "linked"
    linked.symlink_to(evaluator.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="EVOLUTION_EVALUATOR_INVALID"):
        validate_evolution_evaluator(linked / evaluator.name)


def test_linked_template_is_rejected(evaluator):
    linked = evaluator.parent / "linked-template.json"
    linked.symlink_to(evaluator.parent / "candidate-batch.json")
    evaluator.write_text(json.dumps({"scheduler_template": linked.name}))
    with pytest.raises(ValueError, match="EVOLUTION_EVALUATOR_TEMPLATE_INVALID"):
        validate_evolution_evaluator(evaluator)
