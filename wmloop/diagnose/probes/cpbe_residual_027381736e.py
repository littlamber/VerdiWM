"""Offline materialization of the cpbe_residual_027381736e diagnostic probe."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "cpbe_residual_027381736e"
ENVIRONMENT = "push_cube"
SIGNATURE = "mixed_source_sign_positive_prediction_vs_negative_target"
DOSE_SCHEDULE = (-0.05, -0.025, 0.0, 0.025, 0.05)
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")
_MEAN_TOLERANCE = 1e-12


class CPBEResidual027381736eError(ValueError):
    """The offline probe fixture or its derived output violates the contract."""


def program_contract() -> dict[str, object]:
    """Return the exact Probe DSL program materialized by this module."""

    return {
        "aggregation": "goal_outcome_vector",
        "contrast_operator": "signed_mean_preserving_scale",
        "diagnostic_only": True,
        "dose_schedule": list(DOSE_SCHEDULE),
        "estimated_gpu_hours": 0.011611821701388888,
        "hook_type": "H2",
        "invariants": [
            "same_checkpoint",
            "same_input_action",
            "same_trajectory",
            "same_seed",
            "same_evaluator",
            "per_trajectory_action_embedding_temporal_mean_preserved",
            "dose_grid_matches_action_temporal_alignment_reference",
        ],
        "origin": "residual",
        "parent_probe_ids": ["action_temporal_alignment_phase"],
        "probe_id": PROBE_ID,
        "required_capabilities": [
            "action_embedding_hook",
            "action_sequence_hook",
            "paired_seed_control",
        ],
        "rationale": (
            "Change contrast_operator from signed_mean_preserving_phase to "
            "signed_mean_preserving_scale. Residual weight=0.280000."
        ),
        "reversible": True,
        "signal_source": "raw_action_sequence",
        "spatial_mask": "all_action_embedding",
        "temporal_basis": "event_phase_tangent",
    }


def apply_contrast_dose(
    *,
    action_sequence: Sequence[Sequence[float]],
    action_embeddings: Sequence[Sequence[float]],
    dose: float,
) -> list[list[float]]:
    """Scale event-phase embedding contrast without changing its temporal mean."""

    actions = _matrix(action_sequence, "ACTION_SEQUENCE")
    embeddings = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    if len(actions) < 2 or len(embeddings) < 2:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_SEQUENCE_TOO_SHORT")

    parsed_dose = _finite(dose, "dose")
    if parsed_dose not in DOSE_SCHEDULE:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_DOSE_OUTSIDE_FROZEN_SCHEDULE")
    if parsed_dose == 0.0:
        return [row.copy() for row in embeddings]

    phase_tangent = _event_phase_tangent(actions, target_length=len(embeddings))
    temporal_mean = _column_means(embeddings)
    centered = [
        [value - temporal_mean[column] for column, value in enumerate(row)]
        for row in embeddings
    ]
    perturbation = [
        [phase_tangent[index] * value for value in row]
        for index, row in enumerate(centered)
    ]
    perturbation_mean = _column_means(perturbation)
    transformed = [
        [
            value + parsed_dose * (perturbation[index][column] - perturbation_mean[column])
            for column, value in enumerate(row)
        ]
        for index, row in enumerate(embeddings)
    ]
    if _temporal_mean_max_abs_error(embeddings, transformed) > _MEAN_TOLERANCE:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_TEMPORAL_MEAN_INVARIANT_FAILED")
    return transformed


def measure_probe(
    *,
    fixture: Mapping[str, Any],
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure an offline paired-dose fixture and emit diagnostic-only evidence."""

    if not isinstance(fixture, Mapping):
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_FIXTURE_INVALID")
    required = ("action_sequence", "action_embeddings", "outcome_names", "dose_observations")
    if any(key not in fixture for key in required):
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_FIXTURE_INVALID")

    actions = _matrix(fixture["action_sequence"], "ACTION_SEQUENCE")
    embeddings = _matrix(fixture["action_embeddings"], "ACTION_EMBEDDINGS")
    if len(actions) < 2 or len(embeddings) < 2:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_SEQUENCE_TOO_SHORT")
    outcome_names = _outcome_names(fixture["outcome_names"])
    observations = _dose_observations(fixture["dose_observations"], len(outcome_names))

    observed_schedule = tuple(observation["dose"] for observation in observations)
    if observed_schedule != DOSE_SCHEDULE:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_DOSE_SCHEDULE_MISMATCH")
    contexts = {
        (
            observation["checkpoint_id"],
            observation["trajectory_id"],
            observation["seed"],
            observation["evaluator_id"],
        )
        for observation in observations
    }
    if len(contexts) != 1:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_PAIRED_CONTEXT_MISMATCH")

    transformed_by_dose = {
        observation["dose"]: apply_contrast_dose(
            action_sequence=actions,
            action_embeddings=embeddings,
            dose=observation["dose"],
        )
        for observation in observations
    }
    zero_dose_identity = transformed_by_dose[0.0] == embeddings
    if not zero_dose_identity:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_ZERO_DOSE_IDENTITY_FAILED")
    mean_errors = {
        _dose_key(dose): _temporal_mean_max_abs_error(embeddings, transformed)
        for dose, transformed in transformed_by_dose.items()
    }
    maximum_mean_error = max(mean_errors.values())
    if maximum_mean_error > _MEAN_TOLERANCE:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_TEMPORAL_MEAN_INVARIANT_FAILED")

    outcomes_by_dose = {
        observation["dose"]: observation["goal_outcome_vector"]
        for observation in observations
    }
    zero_outcome = outcomes_by_dose[0.0]
    denominator = sum(dose * dose for dose in DOSE_SCHEDULE)
    signed_response = [
        sum(
            dose * (outcomes_by_dose[dose][column] - zero_outcome[column])
            for dose in DOSE_SCHEDULE
        )
        / denominator
        for column in range(len(outcome_names))
    ]
    pair_curvature = max(
        abs(
            (outcomes_by_dose[dose][column] + outcomes_by_dose[-dose][column]) / 2.0
            - zero_outcome[column]
        )
        for dose in DOSE_SCHEDULE
        if dose > 0
        for column in range(len(outcome_names))
    )
    phase_tangent = _event_phase_tangent(actions, target_length=len(embeddings))
    perturbation_energy = max(
        _matrix_l2_delta(embeddings, transformed) / abs(dose)
        for dose, transformed in transformed_by_dose.items()
        if dose != 0.0
    )

    output = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": PROBE_ID,
        "role": "diagnostic",
        "environment": ENVIRONMENT,
        "signature": SIGNATURE,
        "state": "measured",
        "metrics": {
            "contrast_operator": "signed_mean_preserving_scale",
            "temporal_basis": "event_phase_tangent",
            "aggregation": "goal_outcome_vector",
            "dose_schedule": list(DOSE_SCHEDULE),
            "outcome_names": outcome_names,
            "zero_dose_goal_outcome_vector": zero_outcome,
            "signed_goal_outcome_response_vector": signed_response,
            "maximum_paired_curvature": pair_curvature,
            "event_phase_tangent": phase_tangent,
            "contrast_perturbation_l2_per_unit_dose": perturbation_energy,
            "temporal_mean_max_abs_error_by_dose": mean_errors,
            "maximum_temporal_mean_abs_error": maximum_mean_error,
            "zero_dose_exact_identity": zero_dose_identity,
            "paired_context_count": len(contexts),
        },
        "flags": ["zero_event_phase_contrast"] if perturbation_energy <= _MEAN_TOLERANCE else [],
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "Offline fixture evidence only; GPU runtime, locality, nonredundancy, and selector gain remain unmeasured.",
            "This diagnostic output must not be routed into verdict_evidence or a frozen evaluator.",
            "The fixture supplies goal-oriented outcomes; this module does not invoke an evaluator.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise CPBEResidual027381736eError(f"CPBE_RESIDUAL_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _matrix(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise CPBEResidual027381736eError(f"CPBE_RESIDUAL_{name}_INVALID")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise CPBEResidual027381736eError(f"CPBE_RESIDUAL_{name}_INVALID")
        rows.append([_finite(item, name.lower()) for item in row])
    if len({len(row) for row in rows}) != 1:
        raise CPBEResidual027381736eError(f"CPBE_RESIDUAL_{name}_INVALID")
    return rows


def _outcome_names(value: object) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(name, str) or not name for name in value)
        or len(set(value)) != len(value)
    ):
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_OUTCOME_NAMES_INVALID")
    return list(value)


def _dose_observations(value: object, outcome_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_DOSE_OBSERVATIONS_INVALID")
    observations = []
    required = ("dose", "goal_outcome_vector", "checkpoint_id", "trajectory_id", "seed", "evaluator_id")
    for item in value:
        if not isinstance(item, Mapping) or any(key not in item for key in required):
            raise CPBEResidual027381736eError("CPBE_RESIDUAL_DOSE_OBSERVATION_INVALID")
        vector = item["goal_outcome_vector"]
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != outcome_count:
            raise CPBEResidual027381736eError("CPBE_RESIDUAL_GOAL_OUTCOME_VECTOR_INVALID")
        context_values = (item["checkpoint_id"], item["trajectory_id"], item["evaluator_id"])
        if any(not isinstance(context, str) or not context for context in context_values):
            raise CPBEResidual027381736eError("CPBE_RESIDUAL_PAIRED_CONTEXT_INVALID")
        if isinstance(item["seed"], bool) or not isinstance(item["seed"], int):
            raise CPBEResidual027381736eError("CPBE_RESIDUAL_PAIRED_CONTEXT_INVALID")
        observations.append(
            {
                "dose": _finite(item["dose"], "dose"),
                "goal_outcome_vector": [_finite(element, "goal_outcome_vector") for element in vector],
                "checkpoint_id": item["checkpoint_id"],
                "trajectory_id": item["trajectory_id"],
                "seed": item["seed"],
                "evaluator_id": item["evaluator_id"],
            }
        )
    return observations


def _event_phase_tangent(
    actions: Sequence[Sequence[float]], *, target_length: int
) -> list[float]:
    transition = [0.0]
    for previous, current in zip(actions, actions[1:]):
        transition.append(sum(abs(right - left) for left, right in zip(previous, current)) / len(current))
    scale = max(transition)
    event_weight = [value / scale for value in transition] if scale > 1e-12 else [0.0] * len(transition)
    if len(event_weight) != target_length:
        event_weight = _linear_resample(event_weight, target_length)
    previous_weight = [0.0, *event_weight[:-1]]
    next_weight = [*event_weight[1:], 0.0]
    return [0.5 * (right - left) for left, right in zip(previous_weight, next_weight)]


def _linear_resample(values: Sequence[float], target_length: int) -> list[float]:
    if target_length < 2 or len(values) < 2:
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_SEQUENCE_TOO_SHORT")
    source_last = len(values) - 1
    target_last = target_length - 1
    output = []
    for index in range(target_length):
        position = index * source_last / target_last
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        output.append(values[lower] * (1.0 - fraction) + values[upper] * fraction)
    return output


def _column_means(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [sum(row[column] for row in matrix) / len(matrix) for column in range(len(matrix[0]))]


def _temporal_mean_max_abs_error(
    original: Sequence[Sequence[float]], transformed: Sequence[Sequence[float]]
) -> float:
    return max(
        abs(left - right)
        for left, right in zip(_column_means(original), _column_means(transformed))
    )


def _matrix_l2_delta(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_row, right_row in zip(left, right)
            for left_value, right_value in zip(left_row, right_row)
        )
    )


def _dose_key(dose: float) -> str:
    return "0" if dose == 0.0 else str(dose)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CPBEResidual027381736eError(f"CPBE_RESIDUAL_VALUE_INVALID:{field}")
    return float(value)


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise CPBEResidual027381736eError("CPBE_RESIDUAL_EVIDENCE_REFS_INVALID")
    return refs
