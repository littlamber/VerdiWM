"""Diagnostic-only inverse-dynamics confidence probe for ACWM-Phys reacher."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_reacher_inverse_dynamics_confidence_diagnostic_v1"
ENVIRONMENT = "reacher"
SIGNATURE = "inverse_dynamics_confidence"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ReacherInverseDynamicsConfidenceProbeError(ValueError):
    """Reacher inverse-dynamics confidence diagnostic input or output is invalid."""


@dataclass(frozen=True)
class InverseDynamicsConfidenceThresholds:
    min_action_magnitude: float = 0.005
    max_mean_action_error: float = 0.05
    min_mean_confidence: float = 0.45
    min_alignment_cosine: float = 0.50
    min_active_step_count: int = 1

    def __post_init__(self) -> None:
        values = (
            self.min_action_magnitude,
            self.max_mean_action_error,
            self.min_mean_confidence,
            self.min_alignment_cosine,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("INV_DYN_CONFIDENCE_THRESHOLDS_INVALID")
        if self.min_action_magnitude <= 0 or self.max_mean_action_error < 0:
            raise ValueError("INV_DYN_CONFIDENCE_THRESHOLDS_INVALID")
        if not 0 <= self.min_mean_confidence <= 1 or not -1 <= self.min_alignment_cosine <= 1:
            raise ValueError("INV_DYN_CONFIDENCE_THRESHOLDS_INVALID")
        if self.min_active_step_count < 1:
            raise ValueError("INV_DYN_CONFIDENCE_THRESHOLDS_INVALID")


def measure_inverse_dynamics_confidence(
    *,
    steps: Sequence[Mapping[str, Any]],
    thresholds: InverseDynamicsConfidenceThresholds = InverseDynamicsConfidenceThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate inverse-dynamics action error, confidence, and alignment."""

    parsed = _parse_steps(steps)
    active_steps = [
        step for step in parsed if _norm(step["target_action_x"], step["target_action_y"]) >= thresholds.min_action_magnitude
    ]
    if not active_steps:
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_NO_ACTIVE_STEPS")
    action_errors = [
        _norm(step["predicted_action_x"] - step["target_action_x"], step["predicted_action_y"] - step["target_action_y"])
        for step in active_steps
    ]
    confidence_values = [step["confidence"] for step in active_steps]
    alignment_cosines = [
        _cosine(
            step["target_action_x"],
            step["target_action_y"],
            step["predicted_action_x"],
            step["predicted_action_y"],
        )
        for step in active_steps
    ]
    mean_action_error = sum(action_errors) / len(action_errors)
    mean_confidence = sum(confidence_values) / len(confidence_values)
    mean_alignment_cosine = sum(alignment_cosines) / len(alignment_cosines)
    confidence_score = _clip01(
        (
            (1.0 - _clip01(mean_action_error / max(thresholds.max_mean_action_error, 1e-12)))
            + _clip01(mean_confidence / max(thresholds.min_mean_confidence, 1e-12))
            + _clip01((mean_alignment_cosine + 1.0) / 2.0)
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
            "step_count": len(parsed),
            "active_step_count": len(active_steps),
            "mean_action_error": mean_action_error,
            "max_action_error": max(action_errors),
            "mean_confidence": mean_confidence,
            "min_confidence": min(confidence_values),
            "mean_alignment_cosine": mean_alignment_cosine,
            "min_alignment_cosine": min(alignment_cosines),
            "inverse_dynamics_confidence_score": confidence_score,
        },
        "flags": _flags(
            active_step_count=len(active_steps),
            mean_action_error=mean_action_error,
            mean_confidence=mean_confidence,
            mean_alignment_cosine=mean_alignment_cosine,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream inverse-dynamics measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ReacherInverseDynamicsConfidenceProbeError(f"INV_DYN_CONFIDENCE_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_STEPS_INVALID")
    parsed = [_parse_step(step) for step in steps]
    if len({int(step["frame"]) for step in parsed}) != len(parsed):
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda step: step["frame"])


def _parse_step(step: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "target_action_x", "target_action_y", "predicted_action_x", "predicted_action_y", "confidence")
    if not isinstance(step, Mapping) or any(key not in step for key in required):
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_STEP_INVALID")
    parsed = {key: _finite(step[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_STEP_INVALID")
    if parsed["confidence"] < 0 or parsed["confidence"] > 1:
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_STEP_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReacherInverseDynamicsConfidenceProbeError(f"INV_DYN_CONFIDENCE_VALUE_INVALID:{key}")
    return float(value)


def _norm(x_value: float, y_value: float) -> float:
    return math.hypot(x_value, y_value)


def _cosine(ax: float, ay: float, bx: float, by: float) -> float:
    first = _norm(ax, ay)
    second = _norm(bx, by)
    if first == 0 or second == 0:
        return 0.0
    return (ax * bx + ay * by) / (first * second)


def _flags(
    *,
    active_step_count: int,
    mean_action_error: float,
    mean_confidence: float,
    mean_alignment_cosine: float,
    thresholds: InverseDynamicsConfidenceThresholds,
) -> list[str]:
    flags = []
    if active_step_count < thresholds.min_active_step_count:
        flags.append("insufficient_active_actions")
    if mean_action_error > thresholds.max_mean_action_error:
        flags.append("high_inverse_dynamics_error")
    if mean_confidence < thresholds.min_mean_confidence:
        flags.append("low_inverse_dynamics_confidence")
    if mean_alignment_cosine < thresholds.min_alignment_cosine:
        flags.append("inverse_dynamics_misaligned")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ReacherInverseDynamicsConfidenceProbeError("INV_DYN_CONFIDENCE_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
