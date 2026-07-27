"""Diagnostic-only endpoint-path probe for ACWM-Phys push_rope."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_rope_endpoint_path_diagnostic_v1"
ENVIRONMENT = "push_rope"
SIGNATURE = "endpoint_path"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushRopeEndpointPathProbeError(ValueError):
    """Push-rope endpoint-path diagnostic input or output is invalid."""


@dataclass(frozen=True)
class EndpointPathThresholds:
    max_mean_path_error: float = 0.08
    max_final_path_error: float = 0.06
    max_path_error_regression: float = 0.02

    def __post_init__(self) -> None:
        values = (self.max_mean_path_error, self.max_final_path_error, self.max_path_error_regression)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("ROPE_ENDPOINT_PATH_THRESHOLDS_INVALID")


def measure_endpoint_path(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: EndpointPathThresholds = EndpointPathThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured rope endpoint tracking against a target path."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_FRAME_COUNT_INSUFFICIENT")
    errors = [_path_error(frame) for frame in parsed]
    mean_path_error = sum(errors) / len(errors)
    worst_path_error_regression = max(0.0, max(errors[index + 1] - errors[index] for index in range(len(errors) - 1)))
    endpoint_path_score = _clip01(
        (
            (1.0 - _clip01(mean_path_error / max(thresholds.max_mean_path_error, 1e-12)))
            + (1.0 - _clip01(errors[-1] / max(thresholds.max_final_path_error, 1e-12)))
            + (1.0 - _clip01(worst_path_error_regression / max(thresholds.max_path_error_regression, 1e-12)))
        )
        / 3.0
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
            "frame_count": len(parsed),
            "initial_path_error": errors[0],
            "final_path_error": errors[-1],
            "mean_path_error": mean_path_error,
            "max_path_error": max(errors),
            "worst_path_error_regression": worst_path_error_regression,
            "endpoint_path_score": endpoint_path_score,
        },
        "flags": _flags(
            mean_path_error=mean_path_error,
            final_path_error=errors[-1],
            worst_path_error_regression=worst_path_error_regression,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream rope endpoint and target-path measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushRopeEndpointPathProbeError(f"ROPE_ENDPOINT_PATH_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "endpoint_x", "endpoint_y", "target_path_x", "target_path_y")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushRopeEndpointPathProbeError(f"ROPE_ENDPOINT_PATH_VALUE_INVALID:{key}")
    return float(value)


def _path_error(frame: Mapping[str, float]) -> float:
    return math.hypot(frame["endpoint_x"] - frame["target_path_x"], frame["endpoint_y"] - frame["target_path_y"])


def _flags(
    *,
    mean_path_error: float,
    final_path_error: float,
    worst_path_error_regression: float,
    thresholds: EndpointPathThresholds,
) -> list[str]:
    flags = []
    if mean_path_error > thresholds.max_mean_path_error:
        flags.append("endpoint_path_error_high")
    if final_path_error > thresholds.max_final_path_error:
        flags.append("final_endpoint_path_miss")
    if worst_path_error_regression > thresholds.max_path_error_regression:
        flags.append("endpoint_path_regression")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushRopeEndpointPathProbeError("ROPE_ENDPOINT_PATH_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
