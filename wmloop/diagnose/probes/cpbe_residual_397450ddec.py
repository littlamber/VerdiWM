"""Offline materialization of the CPBE multi-scale temporal diagnostic."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "cpbe_residual_397450ddec"
ENVIRONMENT = "push_cube"
SIGNATURE = "mixed_source_sign_positive_prediction_vs_negative_target"
DOSE_SCHEDULE = (-0.05, -0.025, 0.0, 0.025, 0.05)
INVARIANTS = (
    "same_checkpoint",
    "same_input_action",
    "same_trajectory",
    "same_seed",
    "same_evaluator",
    "per_trajectory_action_embedding_temporal_mean_preserved",
    "dose_grid_matches_action_temporal_alignment_reference",
)
_TEMPORAL_SCALES = (1, 2)
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class CpbeResidual397450ddecError(ValueError):
    """The offline probe fixture or generated output violates its contract."""


def apply_multi_scale_event_phase(
    *,
    action_sequence: Sequence[Sequence[float]],
    action_embeddings: Sequence[Sequence[float]],
    dose: float,
) -> list[list[float]]:
    """Apply the staged mean-preserving multi-scale event-phase contrast.

    Event magnitude comes from the raw action sequence. Central phase
    differences at temporal scales one and two are averaged, then used to
    perturb temporally centered action embeddings. Removing the perturbation's
    temporal mean preserves each embedding dimension's trajectory mean.
    """

    parsed_dose = _dose(dose)
    actions = _matrix(action_sequence, "ACTION_SEQUENCE")
    embeddings = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    if len(actions) != len(embeddings):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_TEMPORAL_LENGTH_MISMATCH"
        )
    if len(actions) < 3:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_MULTI_SCALE_SEQUENCE_TOO_SHORT"
        )

    # Zero dose is deliberately returned before any perturbation arithmetic.
    if parsed_dose == 0.0:
        return [list(row) for row in embeddings]

    event_weight = _normalized_event_weight(actions)
    phase_basis = []
    for index in range(len(event_weight)):
        scale_phases = []
        for scale in _TEMPORAL_SCALES:
            previous = event_weight[index - scale] if index >= scale else 0.0
            following = (
                event_weight[index + scale]
                if index + scale < len(event_weight)
                else 0.0
            )
            scale_phases.append((following - previous) / (2.0 * scale))
        phase_basis.append(sum(scale_phases) / len(scale_phases))

    temporal_mean = _column_mean(embeddings)
    raw_perturbation = [
        [
            phase_basis[row_index] * (value - temporal_mean[column_index])
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(embeddings)
    ]
    perturbation_mean = _column_mean(raw_perturbation)
    return [
        [
            value
            + parsed_dose
            * (raw_perturbation[row_index][column_index] - perturbation_mean[column_index])
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(embeddings)
    ]


def measure_cpbe_residual(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate paired offline fixture outcomes on the frozen dose grid."""

    parsed = _trajectories(trajectories)
    outcome_dimension = len(parsed[0]["goal_outcomes"][0.0])
    outcome_sums = {dose: [0.0] * outcome_dimension for dose in DOSE_SCHEDULE}
    max_mean_error = 0.0
    max_zero_dose_error = 0.0

    for trajectory in parsed:
        embeddings = trajectory["action_embeddings"]
        baseline_mean = _column_mean(embeddings)
        for dose in DOSE_SCHEDULE:
            transformed = apply_multi_scale_event_phase(
                action_sequence=trajectory["action_sequence"],
                action_embeddings=embeddings,
                dose=dose,
            )
            transformed_mean = _column_mean(transformed)
            max_mean_error = max(
                max_mean_error,
                max(abs(left - right) for left, right in zip(baseline_mean, transformed_mean)),
            )
            if dose == 0.0:
                max_zero_dose_error = max(
                    max_zero_dose_error,
                    max(
                        abs(left - right)
                        for baseline_row, transformed_row in zip(embeddings, transformed)
                        for left, right in zip(baseline_row, transformed_row)
                    ),
                )
            for index, value in enumerate(trajectory["goal_outcomes"][dose]):
                outcome_sums[dose][index] += value

    trajectory_count = len(parsed)
    outcome_means = {
        dose: [value / trajectory_count for value in outcome_sums[dose]]
        for dose in DOSE_SCHEDULE
    }
    symmetric_slopes = []
    for magnitude in (0.025, 0.05):
        symmetric_slopes.append(
            [
                (positive - negative) / (2.0 * magnitude)
                for positive, negative in zip(
                    outcome_means[magnitude], outcome_means[-magnitude]
                )
            ]
        )
    phase_response = [
        sum(scale[index] for scale in symmetric_slopes) / len(symmetric_slopes)
        for index in range(outcome_dimension)
    ]

    output = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": PROBE_ID,
        "role": "diagnostic",
        "environment": ENVIRONMENT,
        "signature": SIGNATURE,
        "state": "measured",
        "metrics": {
            "trajectory_count": trajectory_count,
            "outcome_dimension": outcome_dimension,
            "dose_schedule": list(DOSE_SCHEDULE),
            "goal_outcome_vector_by_dose": [
                {"dose": dose, "vector": outcome_means[dose]}
                for dose in DOSE_SCHEDULE
            ],
            "zero_dose_goal_outcome_vector": outcome_means[0.0],
            "symmetric_phase_response_vector": phase_response,
            "temporal_scales": list(_TEMPORAL_SCALES),
            "max_temporal_mean_abs_error": max_mean_error,
            "max_zero_dose_abs_error": max_zero_dose_error,
            "invariants_checked": list(INVARIANTS),
        },
        "flags": [],
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This fixture-only diagnostic must not be routed into verdict_evidence or a frozen evaluator.",
            "Offline goal outcomes test aggregation semantics; runtime locality, nonredundancy, selector gain, and collision separation remain unmeasured.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise CpbeResidual397450ddecError(
            f"CPBE_RESIDUAL_397450DDEC_OUTPUT_CONTRACT_INVALID:{exc}"
        ) from exc
    return output


def _trajectories(
    trajectories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not isinstance(trajectories, Sequence)
        or isinstance(trajectories, (str, bytes))
        or not trajectories
    ):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_TRAJECTORIES_INVALID"
        )
    parsed = [_trajectory(value) for value in trajectories]
    trajectory_ids = [value["trajectory_id"] for value in parsed]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_TRAJECTORY_DUPLICATE"
        )
    dimensions = {
        len(vector)
        for trajectory in parsed
        for vector in trajectory["goal_outcomes"].values()
    }
    if len(dimensions) != 1:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_OUTCOME_DIMENSION_MISMATCH"
        )
    return parsed


def _trajectory(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "trajectory_id",
        "checkpoint_id",
        "input_action_id",
        "seed",
        "evaluator_id",
        "action_sequence",
        "action_embeddings",
        "goal_outcomes",
    )
    if not isinstance(value, Mapping) or any(key not in value for key in required):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_TRAJECTORY_INVALID"
        )
    for key in ("trajectory_id", "checkpoint_id", "input_action_id", "evaluator_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise CpbeResidual397450ddecError(
                f"CPBE_RESIDUAL_397450DDEC_INVARIANT_ID_INVALID:{key}"
            )
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, str)) or seed == "":
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_INVARIANT_ID_INVALID:seed"
        )
    actions = _matrix(value["action_sequence"], "ACTION_SEQUENCE")
    embeddings = _matrix(value["action_embeddings"], "ACTION_EMBEDDINGS")
    if len(actions) != len(embeddings):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_TEMPORAL_LENGTH_MISMATCH"
        )
    if len(actions) < 3:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_MULTI_SCALE_SEQUENCE_TOO_SHORT"
        )
    outcomes = _goal_outcomes(value["goal_outcomes"])
    return {
        "trajectory_id": value["trajectory_id"],
        "checkpoint_id": value["checkpoint_id"],
        "input_action_id": value["input_action_id"],
        "seed": seed,
        "evaluator_id": value["evaluator_id"],
        "action_sequence": actions,
        "action_embeddings": embeddings,
        "goal_outcomes": outcomes,
    }


def _goal_outcomes(value: object) -> dict[float, list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_GOAL_OUTCOMES_INVALID"
        )
    outcomes: dict[float, list[float]] = {}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"dose", "vector"}:
            raise CpbeResidual397450ddecError(
                "CPBE_RESIDUAL_397450DDEC_GOAL_OUTCOME_INVALID"
            )
        dose = _dose(item["dose"])
        if dose in outcomes:
            raise CpbeResidual397450ddecError(
                "CPBE_RESIDUAL_397450DDEC_DOSE_DUPLICATE"
            )
        outcomes[dose] = _vector(item["vector"], "GOAL_OUTCOME_VECTOR")
    if tuple(sorted(outcomes)) != DOSE_SCHEDULE:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_DOSE_GRID_MISMATCH"
        )
    dimensions = {len(vector) for vector in outcomes.values()}
    if len(dimensions) != 1:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_OUTCOME_DIMENSION_MISMATCH"
        )
    return outcomes


def _normalized_event_weight(actions: list[list[float]]) -> list[float]:
    event_magnitude = [0.0]
    for current, previous in zip(actions[1:], actions[:-1]):
        event_magnitude.append(
            sum(abs(left - right) for left, right in zip(current, previous))
            / len(current)
        )
    scale = max(event_magnitude)
    if scale == 0.0:
        return [0.0] * len(event_magnitude)
    return [value / scale for value in event_magnitude]


def _matrix(value: object, label: str) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise CpbeResidual397450ddecError(
            f"CPBE_RESIDUAL_397450DDEC_{label}_INVALID"
        )
    rows = [_vector(row, label) for row in value]
    if len({len(row) for row in rows}) != 1:
        raise CpbeResidual397450ddecError(
            f"CPBE_RESIDUAL_397450DDEC_{label}_RAGGED"
        )
    return rows


def _vector(value: object, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise CpbeResidual397450ddecError(
            f"CPBE_RESIDUAL_397450DDEC_{label}_INVALID"
        )
    return [_finite(item, label) for item in value]


def _dose(value: object) -> float:
    parsed = _finite(value, "DOSE")
    if parsed not in DOSE_SCHEDULE:
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_DOSE_OUTSIDE_FROZEN_GRID"
        )
    return parsed


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CpbeResidual397450ddecError(
            f"CPBE_RESIDUAL_397450DDEC_{label}_NONFINITE"
        )
    return float(value)


def _column_mean(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        sum(row[column] for row in matrix) / len(matrix)
        for column in range(len(matrix[0]))
    ]


def _evidence_refs(value: Sequence[str]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_EVIDENCE_REFS_INVALID"
        )
    refs = list(value)
    if any(
        not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None
        for ref in refs
    ):
        raise CpbeResidual397450ddecError(
            "CPBE_RESIDUAL_397450DDEC_EVIDENCE_REFS_INVALID"
        )
    return refs
