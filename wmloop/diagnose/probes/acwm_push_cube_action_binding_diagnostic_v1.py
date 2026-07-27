"""Diagnostic-only action-binding probe for ACWM-Phys push_cube."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_cube_action_binding_diagnostic_v1"
ENVIRONMENT = "push_cube"
SIGNATURE = "action_binding"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushCubeActionBindingProbeError(ValueError):
    """Push-cube action-binding diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ActionBindingThresholds:
    min_action_magnitude: float = 0.01
    min_response_ratio: float = 0.10
    min_alignment_cosine: float = 0.50
    min_action_step_count: int = 1

    def __post_init__(self) -> None:
        values = (self.min_action_magnitude, self.min_response_ratio, self.min_alignment_cosine)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("PUSH_CUBE_ACTION_BINDING_THRESHOLDS_INVALID")
        if self.min_action_magnitude <= 0 or self.min_response_ratio < 0 or not -1 <= self.min_alignment_cosine <= 1:
            raise ValueError("PUSH_CUBE_ACTION_BINDING_THRESHOLDS_INVALID")
        if self.min_action_step_count < 1:
            raise ValueError("PUSH_CUBE_ACTION_BINDING_THRESHOLDS_INVALID")


def measure_action_binding(
    *,
    steps: Sequence[Mapping[str, Any]],
    thresholds: ActionBindingThresholds = ActionBindingThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure whether predicted cube displacement follows commanded action."""

    parsed = _parse_steps(steps)
    action_steps = [step for step in parsed if _norm(step["action_dx"], step["action_dy"]) >= thresholds.min_action_magnitude]
    if not action_steps:
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_NO_ACTION_STEPS")
    response_ratios = []
    alignment_cosines = []
    aligned_count = 0
    for step in action_steps:
        action_norm = _norm(step["action_dx"], step["action_dy"])
        response_norm = _norm(step["cube_delta_x"], step["cube_delta_y"])
        response_ratios.append(response_norm / action_norm)
        cosine = _cosine(step["action_dx"], step["action_dy"], step["cube_delta_x"], step["cube_delta_y"])
        alignment_cosines.append(cosine)
        if cosine >= thresholds.min_alignment_cosine:
            aligned_count += 1
    mean_response_ratio = sum(response_ratios) / len(response_ratios)
    mean_alignment_cosine = sum(alignment_cosines) / len(alignment_cosines)
    action_binding_score = _clip01(
        (_clip01(mean_response_ratio / thresholds.min_response_ratio) + _clip01((mean_alignment_cosine + 1.0) / 2.0)) / 2.0
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
            "action_step_count": len(action_steps),
            "aligned_action_step_count": aligned_count,
            "aligned_action_step_fraction": aligned_count / len(action_steps),
            "mean_response_ratio": mean_response_ratio,
            "min_response_ratio": min(response_ratios),
            "mean_alignment_cosine": mean_alignment_cosine,
            "min_alignment_cosine": min(alignment_cosines),
            "action_binding_score": action_binding_score,
        },
        "flags": _flags(
            action_step_count=len(action_steps),
            mean_response_ratio=mean_response_ratio,
            mean_alignment_cosine=mean_alignment_cosine,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream action and cube displacement measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushCubeActionBindingProbeError(f"PUSH_CUBE_ACTION_BINDING_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_STEPS_INVALID")
    parsed = [_parse_step(step) for step in steps]
    if len({int(step["frame"]) for step in parsed}) != len(parsed):
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda step: step["frame"])


def _parse_step(step: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "action_dx", "action_dy", "cube_delta_x", "cube_delta_y")
    if not isinstance(step, Mapping) or any(key not in step for key in required):
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_STEP_INVALID")
    parsed = {key: _finite(step[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_STEP_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushCubeActionBindingProbeError(f"PUSH_CUBE_ACTION_BINDING_VALUE_INVALID:{key}")
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
    action_step_count: int,
    mean_response_ratio: float,
    mean_alignment_cosine: float,
    thresholds: ActionBindingThresholds,
) -> list[str]:
    flags = []
    if action_step_count < thresholds.min_action_step_count:
        flags.append("insufficient_action_coverage")
    if mean_response_ratio < thresholds.min_response_ratio:
        flags.append("action_ignored")
    if mean_alignment_cosine < thresholds.min_alignment_cosine:
        flags.append("action_response_misaligned")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushCubeActionBindingProbeError("PUSH_CUBE_ACTION_BINDING_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
