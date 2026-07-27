"""Diagnostic-only object-identity probe for ACWM-Phys stack_cube."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_stack_cube_object_identity_diagnostic_v1"
ENVIRONMENT = "stack_cube"
SIGNATURE = "object_identity"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class StackCubeObjectIdentityProbeError(ValueError):
    """Stack-cube object-identity diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ObjectIdentityThresholds:
    min_identity_confidence: float = 0.70
    max_id_swap_score: float = 0.20
    max_identity_drop_count: int = 0

    def __post_init__(self) -> None:
        values = (self.min_identity_confidence, self.max_id_swap_score)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("STACK_CUBE_IDENTITY_THRESHOLDS_INVALID")
        if self.max_identity_drop_count < 0:
            raise ValueError("STACK_CUBE_IDENTITY_THRESHOLDS_INVALID")


def measure_object_identity(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ObjectIdentityThresholds = ObjectIdentityThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure top/bottom cube identity confidence and swap evidence."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAME_COUNT_INSUFFICIENT")
    min_confidences = [
        min(frame["top_identity_confidence"], frame["bottom_identity_confidence"]) for frame in parsed
    ]
    swap_scores = [frame["id_swap_score"] for frame in parsed]
    identity_drop_count = sum(1 for value in min_confidences if value < thresholds.min_identity_confidence)
    mean_identity_confidence = sum(min_confidences) / len(min_confidences)
    identity_score = _clip01((mean_identity_confidence + (1.0 - max(swap_scores))) / 2.0)
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
            "identity_drop_count": identity_drop_count,
            "min_identity_confidence": min(min_confidences),
            "mean_identity_confidence": mean_identity_confidence,
            "final_identity_confidence": min_confidences[-1],
            "max_id_swap_score": max(swap_scores),
            "mean_id_swap_score": sum(swap_scores) / len(swap_scores),
            "identity_score": identity_score,
        },
        "flags": _flags(
            identity_drop_count=identity_drop_count,
            min_identity_confidence=min(min_confidences),
            max_id_swap_score=max(swap_scores),
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream top/bottom object identity measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise StackCubeObjectIdentityProbeError(f"STACK_CUBE_IDENTITY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "top_identity_confidence", "bottom_identity_confidence", "id_swap_score")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAME_INVALID")
    for key in ("top_identity_confidence", "bottom_identity_confidence", "id_swap_score"):
        if parsed[key] < 0 or parsed[key] > 1:
            raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StackCubeObjectIdentityProbeError(f"STACK_CUBE_IDENTITY_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    identity_drop_count: int,
    min_identity_confidence: float,
    max_id_swap_score: float,
    thresholds: ObjectIdentityThresholds,
) -> list[str]:
    flags = []
    if min_identity_confidence < thresholds.min_identity_confidence:
        flags.append("low_identity_confidence")
    if max_id_swap_score > thresholds.max_id_swap_score:
        flags.append("object_identity_swap")
    if identity_drop_count > thresholds.max_identity_drop_count:
        flags.append("identity_unstable")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise StackCubeObjectIdentityProbeError("STACK_CUBE_IDENTITY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
