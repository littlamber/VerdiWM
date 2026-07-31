"""Generated CPBE action-embedding-delta diagnostic for cpbe_residual_63f088b0d5."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document

PROBE_ID = 'cpbe_residual_63f088b0d5'
ENVIRONMENT = 'push_cube'
SIGNATURE = 'mixed_source_sign_positive_prediction_vs_negative_target'
DOSE_SCHEDULE = (-0.05, -0.025, 0.0, 0.025, 0.05)


class CPBEEmbeddingDeltaError(ValueError):
    """The embedding-delta fixture or diagnostic output is invalid."""


def embedding_delta_event_weights(
    action_embeddings: Sequence[Sequence[float]],
) -> list[float]:
    rows = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    if len(rows) < 2:
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_SEQUENCE_TOO_SHORT")
    weights = [0.0]
    for previous, current in zip(rows, rows[1:]):
        weights.append(sum(abs(right - left) for left, right in zip(previous, current)) / len(current))
    scale = max(weights)
    return [value / scale if scale > 1e-12 else 0.0 for value in weights]


def measure_cpbe_residual(
    *,
    dose_responses: Mapping[float, Sequence[float]],
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    parsed = {float(dose): _vector(vector, "GOAL_OUTCOME_VECTOR") for dose, vector in dose_responses.items()}
    if tuple(sorted(parsed)) != tuple(sorted(DOSE_SCHEDULE)) or 0.0 not in parsed:
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_DOSE_SCHEDULE_MISMATCH")
    width = len(parsed[0.0])
    if any(len(vector) != width for vector in parsed.values()):
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_OUTCOME_WIDTH_MISMATCH")
    denominator = sum(dose * dose for dose in DOSE_SCHEDULE)
    zero = parsed[0.0]
    response = [
        sum(dose * (parsed[dose][column] - zero[column]) for dose in DOSE_SCHEDULE) / denominator
        for column in range(width)
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
            "signal_source": "action_embedding_delta",
            "aggregation": "goal_outcome_vector",
            "response_vector": response,
            "target_label_used_for_fit": False,
        },
        "flags": ["diagnostic_only", "target_label_free"],
        "evidence_refs": list(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "Offline diagnostic fixture only; runtime locality and collision separation remain unsettled."
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output, root=Path(__file__).resolve().parents[3])
    except ContractValidationError as exc:
        raise CPBEEmbeddingDeltaError(f"CPBE_EMBEDDING_DELTA_OUTPUT_INVALID:{exc}") from exc
    return output


def _matrix(value: Sequence[Sequence[float]], code: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CPBEEmbeddingDeltaError(code)
    rows = [_vector(row, code) for row in value]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise CPBEEmbeddingDeltaError(code)
    return rows


def _vector(value: Sequence[float], code: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise CPBEEmbeddingDeltaError(code)
    parsed = [float(item) for item in value]
    if any(not math.isfinite(item) for item in parsed):
        raise CPBEEmbeddingDeltaError(code)
    return parsed
