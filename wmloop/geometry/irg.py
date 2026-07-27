"""Interventional Repair Geometry estimation without heavyweight dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from wmloop.geometry.types import GeometryValidationError


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
DoseRepeats = Mapping[float, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class ResponseChart:
    """One uncertainty-bearing local chart of repairability."""

    chart_id: str
    goal_schema: str
    outcome_names: tuple[str, ...]
    intervention_names: tuple[str, ...]
    jacobian: Matrix
    repair_metric: Matrix
    response_coordinate: Vector
    covariance: Matrix
    locality_residuals: Mapping[str, float]
    repeat_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-irg-response-chart",
            "chart_id": self.chart_id,
            "goal_schema": self.goal_schema,
            "outcome_names": list(self.outcome_names),
            "intervention_names": list(self.intervention_names),
            "jacobian": [list(row) for row in self.jacobian],
            "repair_metric": [list(row) for row in self.repair_metric],
            "response_coordinate": list(self.response_coordinate),
            "covariance": [list(row) for row in self.covariance],
            "locality_residuals": dict(self.locality_residuals),
            "repeat_count": self.repeat_count,
        }


def estimate_response_chart(
    *,
    chart_id: str,
    goal_schema: str,
    outcome_names: Sequence[str],
    outcome_weights: Sequence[float],
    baseline_repeats: Sequence[Sequence[float]],
    dose_observations: Mapping[str, DoseRepeats],
    ridge: float = 1e-6,
) -> ResponseChart:
    """Estimate Eq. 2 and Eq. 3 from paired intervention repeats.

    The smallest symmetric non-zero dose is used for a central difference.  A
    path with only positive or only negative doses is represented by a secant
    against the paired zero-dose baseline.  Additional doses only assess
    locality; they never overwrite the smallest-dose derivative.
    """

    names = tuple(outcome_names)
    weights = tuple(_finite(value, "IRG_WEIGHT_INVALID") for value in outcome_weights)
    if not chart_id or not goal_schema or not names or len(set(names)) != len(names):
        raise GeometryValidationError("IRG_CHART_IDENTITY_INVALID")
    if len(weights) != len(names) or any(value <= 0.0 for value in weights):
        raise GeometryValidationError("IRG_WEIGHT_INVALID")
    if not dose_observations:
        raise GeometryValidationError("IRG_INTERVENTION_FRAME_EMPTY")
    baseline = _validate_repeats(baseline_repeats, len(names), "IRG_BASELINE_INVALID")

    intervention_names = tuple(sorted(dose_observations))
    per_path_slopes: dict[str, tuple[Vector, ...]] = {}
    locality: dict[str, float] = {}
    for name in intervention_names:
        if not name:
            raise GeometryValidationError("IRG_INTERVENTION_NAME_EMPTY")
        observations = _validate_dose_observations(dose_observations[name], len(names))
        slopes = _smallest_dose_slopes(observations, baseline)
        per_path_slopes[name] = slopes
        locality[name] = _locality_residual(observations, baseline, slopes)

    repeat_count = min(len(values) for values in per_path_slopes.values())
    if repeat_count < 1:
        raise GeometryValidationError("IRG_PAIRED_REPEATS_EMPTY")
    repeat_jacobians = tuple(
        tuple(
            tuple(per_path_slopes[path][repeat][outcome] for path in intervention_names)
            for outcome in range(len(names))
        )
        for repeat in range(repeat_count)
    )
    jacobian = _mean_matrices(repeat_jacobians)
    metric = _repair_metric(jacobian, weights, ridge=ridge)
    repeat_coordinates = tuple(_response_coordinate(matrix, weights) for matrix in repeat_jacobians)
    coordinate = _mean_vectors(repeat_coordinates)
    covariance = _sample_covariance(repeat_coordinates)
    return ResponseChart(
        chart_id=chart_id,
        goal_schema=goal_schema,
        outcome_names=names,
        intervention_names=intervention_names,
        jacobian=jacobian,
        repair_metric=metric,
        response_coordinate=coordinate,
        covariance=covariance,
        locality_residuals=locality,
        repeat_count=repeat_count,
    )


def irg_distance(
    left: ResponseChart,
    right: ResponseChart,
    *,
    capability_distance: float = 0.0,
    capability_weight: float = 1.0,
) -> float:
    """Uncertainty-normalized distance between compatible IRG charts."""

    if (
        left.goal_schema != right.goal_schema
        or left.outcome_names != right.outcome_names
        or left.intervention_names != right.intervention_names
        or len(left.response_coordinate) != len(right.response_coordinate)
    ):
        raise GeometryValidationError("IRG_CHARTS_INCOMPATIBLE")
    cap = _finite(capability_distance, "IRG_CAPABILITY_DISTANCE_INVALID")
    cap_weight = _finite(capability_weight, "IRG_CAPABILITY_WEIGHT_INVALID")
    if cap < 0.0 or cap_weight < 0.0:
        raise GeometryValidationError("IRG_CAPABILITY_DISTANCE_INVALID")
    total = 0.0
    for index, (a, b) in enumerate(zip(left.response_coordinate, right.response_coordinate, strict=True)):
        variance = left.covariance[index][index] + right.covariance[index][index]
        total += ((a - b) ** 2) / (1.0 + max(variance, 0.0))
    total += cap_weight * cap * cap
    return math.sqrt(total)


def normalized_frobenius_alignment(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Return the normalized alignment error used by the transfer certificate."""

    lhs = _matrix(left, "IRG_ALIGNMENT_MATRIX_INVALID")
    rhs = _matrix(right, "IRG_ALIGNMENT_MATRIX_INVALID")
    if len(lhs) != len(rhs) or any(len(a) != len(b) for a, b in zip(lhs, rhs, strict=True)):
        raise GeometryValidationError("IRG_ALIGNMENT_SHAPE_MISMATCH")
    difference = math.sqrt(sum((a - b) ** 2 for ra, rb in zip(lhs, rhs, strict=True) for a, b in zip(ra, rb, strict=True)))
    left_norm = math.sqrt(sum(value * value for row in lhs for value in row))
    right_norm = math.sqrt(sum(value * value for row in rhs for value in row))
    return difference / (left_norm + right_norm + 1e-12)


def _validate_dose_observations(values: DoseRepeats, width: int) -> dict[float, tuple[Vector, ...]]:
    result: dict[float, tuple[Vector, ...]] = {}
    for raw_dose, repeats in values.items():
        dose = _finite(raw_dose, "IRG_DOSE_INVALID")
        if dose == 0.0 or dose in result:
            raise GeometryValidationError("IRG_DOSE_INVALID")
        result[dose] = _validate_repeats(repeats, width, "IRG_DOSE_OBSERVATION_INVALID")
    if not result:
        raise GeometryValidationError("IRG_DOSE_OBSERVATIONS_EMPTY")
    return result


def _validate_repeats(values: Sequence[Sequence[float]], width: int, code: str) -> tuple[Vector, ...]:
    if not values:
        raise GeometryValidationError(code)
    result = tuple(tuple(_finite(value, code) for value in row) for row in values)
    if any(len(row) != width for row in result):
        raise GeometryValidationError(code)
    return result


def _smallest_dose_slopes(observations: Mapping[float, tuple[Vector, ...]], baseline: tuple[Vector, ...]) -> tuple[Vector, ...]:
    positive = sorted(dose for dose in observations if dose > 0.0)
    negative = sorted((dose for dose in observations if dose < 0.0), key=abs)
    if positive and negative and math.isclose(positive[0], abs(negative[0]), rel_tol=1e-9, abs_tol=1e-12):
        plus, minus = observations[positive[0]], observations[negative[0]]
        count = min(len(plus), len(minus))
        denominator = positive[0] - negative[0]
        return tuple(
            tuple((plus[i][j] - minus[i][j]) / denominator for j in range(len(plus[i])))
            for i in range(count)
        )
    dose = positive[0] if positive else negative[0]
    treated = observations[dose]
    count = min(len(treated), len(baseline))
    return tuple(
        tuple((treated[i][j] - baseline[i][j]) / dose for j in range(len(treated[i])))
        for i in range(count)
    )


def _locality_residual(
    observations: Mapping[float, tuple[Vector, ...]],
    baseline: tuple[Vector, ...],
    reference_slopes: tuple[Vector, ...],
) -> float:
    if len(observations) < 2:
        return 0.0
    reference = _mean_vectors(reference_slopes)
    residuals: list[float] = []
    for dose, treated in observations.items():
        count = min(len(treated), len(baseline))
        slope = _mean_vectors(
            tuple(
                tuple((treated[i][j] - baseline[i][j]) / dose for j in range(len(treated[i])))
                for i in range(count)
            )
        )
        numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(slope, reference, strict=True)))
        denominator = math.sqrt(sum(value * value for value in reference)) + 1e-12
        residuals.append(numerator / denominator)
    return max(residuals)


def _repair_metric(jacobian: Matrix, weights: Vector, *, ridge: float) -> Matrix:
    ridge = _finite(ridge, "IRG_RIDGE_INVALID")
    if ridge < 0.0:
        raise GeometryValidationError("IRG_RIDGE_INVALID")
    width = len(jacobian[0])
    return tuple(
        tuple(
            sum(weights[k] * jacobian[k][i] * jacobian[k][j] for k in range(len(jacobian)))
            + (ridge if i == j else 0.0)
            for j in range(width)
        )
        for i in range(width)
    )


def _response_coordinate(jacobian: Matrix, weights: Vector) -> Vector:
    return tuple(math.sqrt(weights[i]) * value for i, row in enumerate(jacobian) for value in row)


def _sample_covariance(vectors: Sequence[Vector]) -> Matrix:
    mean = _mean_vectors(vectors)
    width = len(mean)
    if len(vectors) == 1:
        return tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
    denominator = len(vectors) - 1
    return tuple(
        tuple(
            sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in vectors) / denominator
            for j in range(width)
        )
        for i in range(width)
    )


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise GeometryValidationError("IRG_VECTOR_SET_EMPTY")
    width = len(vectors[0])
    if width == 0 or any(len(row) != width for row in vectors):
        raise GeometryValidationError("IRG_VECTOR_SHAPE_INVALID")
    return tuple(sum(row[i] for row in vectors) / len(vectors) for i in range(width))


def _mean_matrices(matrices: Sequence[Matrix]) -> Matrix:
    if not matrices:
        raise GeometryValidationError("IRG_MATRIX_SET_EMPTY")
    rows, columns = len(matrices[0]), len(matrices[0][0])
    if any(len(matrix) != rows or any(len(row) != columns for row in matrix) for matrix in matrices):
        raise GeometryValidationError("IRG_MATRIX_SHAPE_INVALID")
    return tuple(
        tuple(sum(matrix[i][j] for matrix in matrices) / len(matrices) for j in range(columns))
        for i in range(rows)
    )


def _matrix(values: Sequence[Sequence[float]], code: str) -> Matrix:
    if not values:
        raise GeometryValidationError(code)
    result = tuple(tuple(_finite(value, code) for value in row) for row in values)
    width = len(result[0])
    if width == 0 or any(len(row) != width for row in result):
        raise GeometryValidationError(code)
    return result


def _finite(value: float, code: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(code) from exc
    if not math.isfinite(number):
        raise GeometryValidationError(code)
    return number
