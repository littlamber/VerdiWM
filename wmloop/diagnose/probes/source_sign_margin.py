"""Leakage-safe source-sign discriminant projection for CPBE probes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document


class SourceSignMarginError(ValueError):
    """Source-sign projection inputs cannot define a valid discriminant."""


def measure_source_sign_margin(
    *,
    probe_id: str,
    environment: str,
    signature: str,
    response_vectors: Mapping[str, Sequence[float]],
    source_signs: Mapping[str, int],
    target: str,
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    projected, audit = project_source_sign_margin(
        response_vectors,
        source_signs=source_signs,
        target=target,
    )
    output = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": probe_id,
        "role": "diagnostic",
        "environment": environment,
        "signature": signature,
        "state": "measured",
        "metrics": {
            "aggregation": "source_sign_margin",
            "projected_response_vectors": {
                name: list(vector) for name, vector in projected.items()
            },
            "fit_audit": audit,
        },
        "flags": [],
        "evidence_refs": list(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "Source effect signs fit the diagnostic coordinate; the target effect sign is never an input.",
            "Offline projection validity does not establish runtime locality, selector gain, repair quality, or transfer.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise SourceSignMarginError(f"SOURCE_SIGN_MARGIN_OUTPUT_INVALID:{exc}") from exc
    return output


def project_source_sign_margin(
    vectors: Mapping[str, Sequence[float]],
    *,
    source_signs: Mapping[str, int],
    target: str,
) -> tuple[dict[str, tuple[float, ...]], dict[str, object]]:
    """Project responses onto a direction fitted only from signed sources.

    The target label is deliberately absent from this API. The target response
    is projected after fitting, so downstream evaluation can score it without
    leaking the held-out target effect sign into the diagnostic coordinate.
    """

    parsed = {name: _unit(vector, name) for name, vector in vectors.items()}
    if target not in parsed or target in source_signs:
        raise SourceSignMarginError("SOURCE_SIGN_MARGIN_TARGET_SCOPE_INVALID")
    if set(source_signs) - set(parsed):
        raise SourceSignMarginError("SOURCE_SIGN_MARGIN_SOURCE_MISSING")
    positive = [parsed[name] for name, sign in source_signs.items() if sign > 0]
    negative = [parsed[name] for name, sign in source_signs.items() if sign < 0]
    if not positive or not negative or any(sign not in {-1, 1} for sign in source_signs.values()):
        raise SourceSignMarginError("SOURCE_SIGN_MARGIN_CLASS_INVALID")
    dimension = len(positive[0])
    if any(len(row) != dimension for row in (*positive, *negative, *parsed.values())):
        raise SourceSignMarginError("SOURCE_SIGN_MARGIN_DIMENSION_MISMATCH")

    positive_centroid = _centroid(positive)
    negative_centroid = _centroid(negative)
    direction = _unit(
        tuple(left - right for left, right in zip(positive_centroid, negative_centroid, strict=True)),
        "source_direction",
    )
    midpoint = tuple(
        0.5 * (left + right)
        for left, right in zip(positive_centroid, negative_centroid, strict=True)
    )
    projected: dict[str, tuple[float, ...]] = {}
    scalars: dict[str, float] = {}
    for name, vector in parsed.items():
        scalar = sum(
            (value - center) * axis
            for value, center, axis in zip(vector, midpoint, direction, strict=True)
        )
        if abs(scalar) <= 1e-12:
            raise SourceSignMarginError(f"SOURCE_SIGN_MARGIN_ZERO:{name}")
        projected[name] = tuple(scalar * axis for axis in direction)
        scalars[name] = scalar
    return projected, {
        "aggregation": "source_sign_margin",
        "fit_environments": sorted(source_signs),
        "target_label_used_for_fit": False,
        "source_positive_count": len(positive),
        "source_negative_count": len(negative),
        "source_direction": list(direction),
        "projected_margin_by_environment": scalars,
    }


def _centroid(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0])))


def _unit(value: Sequence[float], name: str) -> tuple[float, ...]:
    if not value:
        raise SourceSignMarginError(f"SOURCE_SIGN_MARGIN_VECTOR_INVALID:{name}")
    parsed = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in parsed):
        raise SourceSignMarginError(f"SOURCE_SIGN_MARGIN_VECTOR_INVALID:{name}")
    norm = math.sqrt(sum(item * item for item in parsed))
    if norm <= 1e-12:
        raise SourceSignMarginError(f"SOURCE_SIGN_MARGIN_VECTOR_ZERO:{name}")
    return tuple(item / norm for item in parsed)
