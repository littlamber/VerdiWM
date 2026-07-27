"""Counterexample-driven IRG evolution primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from wmloop.geometry.types import GeometryValidationError


@dataclass(frozen=True)
class EffectEstimate:
    mean: float
    lower: float
    upper: float
    sign_q_value: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in (self.mean, self.lower, self.upper, self.sign_q_value)):
            raise GeometryValidationError("ATLAS_EFFECT_INVALID")
        if self.lower > self.mean or self.mean > self.upper or not 0.0 <= self.sign_q_value <= 1.0:
            raise GeometryValidationError("ATLAS_EFFECT_INVALID")


@dataclass(frozen=True)
class AtlasPoint:
    campaign_id: str
    chart_id: str
    coordinates: tuple[float, ...]
    effects: Mapping[str, EffectEstimate]

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.chart_id or not self.coordinates:
            raise GeometryValidationError("ATLAS_POINT_INVALID")
        if any(not math.isfinite(float(value)) for value in self.coordinates):
            raise GeometryValidationError("ATLAS_POINT_INVALID")


@dataclass(frozen=True)
class RepairCollision:
    left_campaign_id: str
    right_campaign_id: str
    primitive: str
    distance: float
    left_effect: float
    right_effect: float
    q_value: float


@dataclass(frozen=True)
class ProbeCandidate:
    name: str
    nested_regret_reduction: float
    standard_error: float
    probe_cost: float
    calibration_delta: float
    regression_pass: bool

    def __post_init__(self) -> None:
        if not self.name or any(
            not math.isfinite(float(value))
            for value in (
                self.nested_regret_reduction,
                self.standard_error,
                self.probe_cost,
                self.calibration_delta,
            )
        ):
            raise GeometryValidationError("PROBE_CANDIDATE_INVALID")
        if self.standard_error < 0.0 or self.probe_cost <= 0.0:
            raise GeometryValidationError("PROBE_CANDIDATE_INVALID")


def detect_repair_collisions(
    points: Sequence[AtlasPoint],
    *,
    distance_threshold: float,
    minimum_effect: float,
    fdr_alpha: float,
) -> tuple[RepairCollision, ...]:
    """Find nearby campaigns with confidence-separated opposing effects."""

    if distance_threshold < 0.0 or minimum_effect < 0.0 or not 0.0 < fdr_alpha < 1.0:
        raise GeometryValidationError("ATLAS_COLLISION_THRESHOLD_INVALID")
    collisions: list[RepairCollision] = []
    for i, left in enumerate(points):
        for right in points[i + 1 :]:
            if left.chart_id != right.chart_id or len(left.coordinates) != len(right.coordinates):
                continue
            distance = math.dist(left.coordinates, right.coordinates)
            if distance > distance_threshold:
                continue
            for primitive in sorted(set(left.effects) & set(right.effects)):
                a, b = left.effects[primitive], right.effects[primitive]
                opposite = (a.lower > minimum_effect and b.upper < -minimum_effect) or (
                    b.lower > minimum_effect and a.upper < -minimum_effect
                )
                q_value = max(a.sign_q_value, b.sign_q_value)
                if opposite and q_value <= fdr_alpha:
                    collisions.append(
                        RepairCollision(
                            left_campaign_id=left.campaign_id,
                            right_campaign_id=right.campaign_id,
                            primitive=primitive,
                            distance=distance,
                            left_effect=a.mean,
                            right_effect=b.mean,
                            q_value=q_value,
                        )
                    )
    return tuple(sorted(collisions, key=lambda item: (item.q_value, item.distance, item.primitive)))


def rank_probe_candidates(
    candidates: Sequence[ProbeCandidate],
    *,
    confidence_z: float = 1.96,
    maximum_calibration_regression: float = 0.0,
) -> tuple[dict[str, object], ...]:
    """Rank staged directions by regret-reduction LCB per unit cost."""

    if confidence_z < 0.0 or maximum_calibration_regression < 0.0:
        raise GeometryValidationError("PROBE_RANK_THRESHOLD_INVALID")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        lower = candidate.nested_regret_reduction - confidence_z * candidate.standard_error
        calibration_pass = candidate.calibration_delta <= maximum_calibration_regression
        promoted = lower > 0.0 and calibration_pass and candidate.regression_pass
        blockers = []
        if lower <= 0.0:
            blockers.append("regret_reduction_lcb_not_positive")
        if not calibration_pass:
            blockers.append("calibration_regressed")
        if not candidate.regression_pass:
            blockers.append("frozen_regression_battery_failed")
        rows.append(
            {
                "name": candidate.name,
                "regret_reduction_lcb": lower,
                "score_per_cost": lower / candidate.probe_cost,
                "promoted": promoted,
                "blockers": blockers,
            }
        )
    return tuple(sorted(rows, key=lambda row: (-float(row["score_per_cost"]), str(row["name"]))))
