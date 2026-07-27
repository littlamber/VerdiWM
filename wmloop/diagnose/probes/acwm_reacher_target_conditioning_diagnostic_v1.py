"""Diagnostic-only target-conditioning probe for ACWM-Phys reacher."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_reacher_target_conditioning_diagnostic_v1"
ENVIRONMENT = "reacher"
SIGNATURE = "target_conditioning"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ReacherTargetConditioningProbeError(ValueError):
    """Reacher target-conditioning diagnostic input or output is invalid."""


@dataclass(frozen=True)
class TargetConditioningThresholds:
    min_distance_reduction: float = 0.03
    max_final_target_distance: float = 0.08
    max_distance_regression: float = 0.02

    def __post_init__(self) -> None:
        values = (self.min_distance_reduction, self.max_final_target_distance, self.max_distance_regression)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("TARGET_CONDITIONING_THRESHOLDS_INVALID")


def measure_target_conditioning(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: TargetConditioningThresholds = TargetConditioningThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured endpoint distance to the conditioned target."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_FRAME_COUNT_INSUFFICIENT")
    distances = [_target_distance(frame) for frame in parsed]
    distance_reduction = distances[0] - distances[-1]
    worst_distance_regression = max(0.0, max(distances[index + 1] - distances[index] for index in range(len(distances) - 1)))
    target_conditioning_score = _clip01(
        (
            _clip01(distance_reduction / max(thresholds.min_distance_reduction, 1e-12))
            + (1.0 - _clip01(distances[-1] / max(thresholds.max_final_target_distance, 1e-12)))
            + (1.0 - _clip01(worst_distance_regression / max(thresholds.max_distance_regression, 1e-12)))
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
            "initial_target_distance": distances[0],
            "final_target_distance": distances[-1],
            "min_target_distance": min(distances),
            "distance_reduction": distance_reduction,
            "worst_distance_regression": worst_distance_regression,
            "target_conditioning_score": target_conditioning_score,
        },
        "flags": _flags(
            distance_reduction=distance_reduction,
            final_target_distance=distances[-1],
            worst_distance_regression=worst_distance_regression,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream target and endpoint measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ReacherTargetConditioningProbeError(f"TARGET_CONDITIONING_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "target_x", "target_y", "endpoint_x", "endpoint_y")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReacherTargetConditioningProbeError(f"TARGET_CONDITIONING_VALUE_INVALID:{key}")
    return float(value)


def _target_distance(frame: Mapping[str, float]) -> float:
    return math.hypot(frame["endpoint_x"] - frame["target_x"], frame["endpoint_y"] - frame["target_y"])


def _flags(
    *,
    distance_reduction: float,
    final_target_distance: float,
    worst_distance_regression: float,
    thresholds: TargetConditioningThresholds,
) -> list[str]:
    flags = []
    if distance_reduction < thresholds.min_distance_reduction:
        flags.append("weak_target_progress")
    if final_target_distance > thresholds.max_final_target_distance:
        flags.append("final_target_miss")
    if worst_distance_regression > thresholds.max_distance_regression:
        flags.append("target_distance_regression")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ReacherTargetConditioningProbeError("TARGET_CONDITIONING_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
