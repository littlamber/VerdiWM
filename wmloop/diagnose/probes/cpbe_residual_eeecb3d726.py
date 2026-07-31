"""Offline materialization for the phase-curvature CPBE diagnostic probe."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "cpbe_residual_eeecb3d726"
ENVIRONMENT = "push_cube"
SIGNATURE = "mixed_source_sign_positive_prediction_vs_negative_target"
DOSE_SCHEDULE = (-0.05, -0.025, 0.0, 0.025, 0.05)
TEMPORAL_BASIS = "event_phase_curvature"
CONTRAST_OPERATOR = "signed_mean_preserving_phase"
AGGREGATION = "goal_outcome_vector"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class CpbeResidualPhaseCurvatureProbeError(ValueError):
    """The staged probe input or output violates its frozen DSL contract."""


def apply_phase_curvature_dose(
    *,
    action_sequence: Sequence[Sequence[float]],
    action_embeddings: Sequence[Sequence[float]],
    dose: float,
) -> list[list[float]]:
    """Apply one allowed phase-curvature dose to an action-embedding fixture."""

    actions = _matrix(action_sequence, "ACTION_SEQUENCE")
    embeddings = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    if len(actions) != len(embeddings):
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_TEMPORAL_LENGTH_MISMATCH")
    if len(actions) < 3:
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_TIMESTEPS_INSUFFICIENT")
    parsed_dose = _allowed_dose(dose)
    if parsed_dose == 0.0:
        return [row.copy() for row in embeddings]

    curvature = _event_phase_curvature(actions)
    temporal_means = [
        sum(row[column] for row in embeddings) / len(embeddings)
        for column in range(len(embeddings[0]))
    ]
    perturbation = [
        [
            curvature[index] * (value - temporal_means[column])
            for column, value in enumerate(row)
        ]
        for index, row in enumerate(embeddings)
    ]
    perturbation_means = [
        sum(row[column] for row in perturbation) / len(perturbation)
        for column in range(len(perturbation[0]))
    ]
    return [
        [
            value + parsed_dose * (perturbation[index][column] - perturbation_means[column])
            for column, value in enumerate(row)
        ]
        for index, row in enumerate(embeddings)
    ]


def measure_phase_curvature_fixture(
    *,
    action_sequence: Sequence[Sequence[float]],
    action_embeddings: Sequence[Sequence[float]],
    goal_outcome_vectors: Sequence[Sequence[float]],
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure the frozen five-dose DSL program on an offline paired fixture."""

    actions = _matrix(action_sequence, "ACTION_SEQUENCE")
    embeddings = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    outcomes = _matrix(goal_outcome_vectors, "GOAL_OUTCOME_VECTORS")
    if len(actions) != len(embeddings):
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_TEMPORAL_LENGTH_MISMATCH")
    if len(actions) < 3:
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_TIMESTEPS_INSUFFICIENT")
    if len(outcomes) != len(DOSE_SCHEDULE):
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_DOSE_GRID_MISMATCH")

    dosed_embeddings = [
        apply_phase_curvature_dose(
            action_sequence=actions,
            action_embeddings=embeddings,
            dose=dose,
        )
        for dose in DOSE_SCHEDULE
    ]
    zero_index = DOSE_SCHEDULE.index(0.0)
    zero_dose_error = _max_matrix_error(dosed_embeddings[zero_index], embeddings)
    temporal_mean_error = max(
        _max_vector_error(_temporal_mean(candidate), _temporal_mean(embeddings))
        for candidate in dosed_embeddings
    )
    antisymmetry_error = max(
        _max_signed_pair_error(
            negative=dosed_embeddings[index],
            baseline=embeddings,
            positive=dosed_embeddings[-index - 1],
        )
        for index in range(len(DOSE_SCHEDULE) // 2)
    )
    curvature = _event_phase_curvature(actions)
    output = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": PROBE_ID,
        "role": "diagnostic",
        "environment": ENVIRONMENT,
        "signature": SIGNATURE,
        "state": "measured",
        "metrics": {
            "temporal_basis": TEMPORAL_BASIS,
            "contrast_operator": CONTRAST_OPERATOR,
            "aggregation": AGGREGATION,
            "dose_schedule": list(DOSE_SCHEDULE),
            "goal_outcome_vectors": [row.copy() for row in outcomes],
            "goal_outcome_dimension": len(outcomes[0]),
            "event_phase_curvature": curvature,
            "active_curvature_step_count": sum(abs(value) > 1e-12 for value in curvature),
            "mean_absolute_embedding_delta_by_dose": [
                _mean_absolute_matrix_error(candidate, embeddings)
                for candidate in dosed_embeddings
            ],
            "zero_dose_max_abs_error": zero_dose_error,
            "max_temporal_mean_error": temporal_mean_error,
            "max_signed_dose_antisymmetry_error": antisymmetry_error,
        },
        "flags": ["offline_fixture_only"]
        + (["no_action_transition"] if not any(abs(value) > 1e-12 for value in curvature) else []),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This output is diagnostic-only and must not be routed into active-campaign verdict evidence.",
            (
                "The offline fixture validates DSL semantics but is not a runtime smoke, "
                "locality canary, or selector-gain receipt."
            ),
            (
                "Checkpoint, trajectory, seed, evaluator, and input-action pairing remain "
                "upstream runtime admission requirements."
            ),
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise CpbeResidualPhaseCurvatureProbeError(
            f"CPBE_RESIDUAL_OUTPUT_CONTRACT_INVALID:{exc}"
        ) from exc
    return output


def _event_phase_curvature(actions: list[list[float]]) -> list[float]:
    event_weight = [0.0]
    for previous, current in zip(actions, actions[1:]):
        event_weight.append(
            sum(abs(value - previous[column]) for column, value in enumerate(current))
            / len(current)
        )
    event_scale = max(event_weight)
    if event_scale <= 1e-12:
        return [0.0] * len(actions)
    event_weight = [value / event_scale for value in event_weight]
    curvature = []
    for index, value in enumerate(event_weight):
        previous = event_weight[index - 1] if index > 0 else 0.0
        following = event_weight[index + 1] if index + 1 < len(event_weight) else 0.0
        curvature.append(previous - 2.0 * value + following)
    curvature_scale = max(abs(value) for value in curvature)
    return [value / curvature_scale for value in curvature]


def _matrix(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise CpbeResidualPhaseCurvatureProbeError(f"CPBE_RESIDUAL_{name}_INVALID")
    rows: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise CpbeResidualPhaseCurvatureProbeError(f"CPBE_RESIDUAL_{name}_INVALID")
        parsed = [_finite(item, name) for item in row]
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise CpbeResidualPhaseCurvatureProbeError(f"CPBE_RESIDUAL_{name}_NOT_RECTANGULAR")
        rows.append(parsed)
    return rows


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CpbeResidualPhaseCurvatureProbeError(f"CPBE_RESIDUAL_{name}_VALUE_INVALID")
    return float(value)


def _allowed_dose(value: object) -> float:
    parsed = _finite(value, "DOSE")
    for allowed in DOSE_SCHEDULE:
        if parsed == allowed:
            return allowed
    raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_DOSE_GRID_MISMATCH")


def _temporal_mean(matrix: list[list[float]]) -> list[float]:
    return [
        sum(row[column] for row in matrix) / len(matrix)
        for column in range(len(matrix[0]))
    ]


def _max_matrix_error(first: list[list[float]], second: list[list[float]]) -> float:
    return max(abs(left - right) for row_a, row_b in zip(first, second) for left, right in zip(row_a, row_b))


def _mean_absolute_matrix_error(first: list[list[float]], second: list[list[float]]) -> float:
    errors = [abs(left - right) for row_a, row_b in zip(first, second) for left, right in zip(row_a, row_b)]
    return sum(errors) / len(errors)


def _max_vector_error(first: list[float], second: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(first, second))


def _max_signed_pair_error(
    *,
    negative: list[list[float]],
    baseline: list[list[float]],
    positive: list[list[float]],
) -> float:
    return max(
        abs((left - center) + (right - center))
        for negative_row, baseline_row, positive_row in zip(negative, baseline, positive)
        for left, center, right in zip(negative_row, baseline_row, positive_row)
    )


def _evidence_refs(value: Sequence[str]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_EVIDENCE_REFS_INVALID")
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise CpbeResidualPhaseCurvatureProbeError("CPBE_RESIDUAL_EVIDENCE_REFS_INVALID")
    return refs
