"""CPU-only method boundaries: uv run --offline --frozen pytest -q tests/test_method_boundary_regressions.py."""

from __future__ import annotations

from dataclasses import replace

import pytest

from wmloop.geometry.evolution import ProbeCandidate, detect_repair_collisions, rank_probe_candidates
from wmloop.geometry.memory import EffectContext, EffectMemory, EffectRecord
from wmloop.geometry.types import GeometryValidationError


def _effect() -> EffectRecord:
    return EffectRecord(
        record_id="confirmed", primitive="anchor",
        context=EffectContext("campaign", "model", "video", "goal", "outcome", "chart", "heldout", (16,)),
        status="confirmed", mean_effect=0.4, standard_error=0.05, lower_bound=0.2,
        goal_threshold=0.0, validity_gates={"protected": True}, replication_count=2,
        evidence_refs=("confirmation.json",),
    )


@pytest.mark.parametrize("gate", ["False", 1, None])
def test_effect_gates_require_actual_booleans(gate: object) -> None:
    with pytest.raises(GeometryValidationError, match="EFFECT_RECORD_GATE_TYPE_INVALID"):
        replace(_effect(), validity_gates={"protected": gate})


def test_effect_interval_cannot_exceed_mean() -> None:
    with pytest.raises(GeometryValidationError, match="EFFECT_RECORD_INTERVAL_INVALID"):
        replace(_effect(), lower_bound=0.5)


def test_memory_owns_snapshot_on_ingress_and_egress() -> None:
    effect = _effect()
    memory = EffectMemory((effect,))
    effect.validity_gates["protected"] = False
    assert memory.records()[0].validity_gates["protected"] is True
    memory.query()[0].validity_gates["protected"] = False
    assert memory.records()[0].validity_gates["protected"] is True
    assert memory.records()[0].to_dict()["validity_gates"]["protected"] is True


def test_memory_revalidates_mutated_record_before_admission() -> None:
    effect = _effect()
    effect.validity_gates["protected"] = False
    memory = EffectMemory()
    with pytest.raises(GeometryValidationError, match="CONFIRMED_EFFECT_GATE_FAILED"):
        memory.add(effect)
    assert memory.records() == ()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["distance_threshold", "minimum_effect", "fdr_alpha"])
def test_collision_thresholds_must_be_finite(field: str, value: float) -> None:
    thresholds = dict(distance_threshold=0.5, minimum_effect=0.1, fdr_alpha=0.05)
    thresholds[field] = value
    with pytest.raises(GeometryValidationError, match="ATLAS_COLLISION_THRESHOLD_INVALID"):
        detect_repair_collisions((), **thresholds)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["confidence_z", "maximum_calibration_regression"])
def test_probe_admission_thresholds_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(GeometryValidationError, match="PROBE_RANK_THRESHOLD_INVALID"):
        rank_probe_candidates((), **{field: value})


def test_probe_regression_receipt_cannot_be_truthy_string() -> None:
    with pytest.raises(GeometryValidationError, match="PROBE_CANDIDATE_REGRESSION_INVALID"):
        ProbeCandidate("history", 0.5, 0.1, 1.0, 0.0, "False")
