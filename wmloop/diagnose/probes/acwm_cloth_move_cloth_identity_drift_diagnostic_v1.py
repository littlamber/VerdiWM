"""Diagnostic-only cloth-identity-drift probe for ACWM-Phys cloth_move."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_cloth_move_cloth_identity_drift_diagnostic_v1"
ENVIRONMENT = "cloth_move"
SIGNATURE = "cloth_identity_drift"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ClothMoveIdentityDriftProbeError(ValueError):
    """Cloth-move identity-drift diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ClothIdentityDriftThresholds:
    min_identity_confidence: float = 0.70
    max_texture_drift: float = 0.25
    max_mask_iou_drop: float = 0.25

    def __post_init__(self) -> None:
        values = (self.min_identity_confidence, self.max_texture_drift, self.max_mask_iou_drop)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("CLOTH_IDENTITY_THRESHOLDS_INVALID")


def measure_cloth_identity_drift(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ClothIdentityDriftThresholds = ClothIdentityDriftThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured cloth identity confidence, texture drift, and mask IoU."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAME_COUNT_INSUFFICIENT")
    confidence_values = [frame["identity_confidence"] for frame in parsed]
    texture_drift_values = [frame["texture_drift"] for frame in parsed]
    mask_iou_values = [frame["mask_iou_to_initial"] for frame in parsed]
    max_mask_iou_drop = max(0.0, mask_iou_values[0] - min(mask_iou_values))
    identity_score = _clip01(
        (
            _clip01(min(confidence_values) / max(thresholds.min_identity_confidence, 1e-12))
            + (1.0 - _clip01(max(texture_drift_values) / max(thresholds.max_texture_drift, 1e-12)))
            + (1.0 - _clip01(max_mask_iou_drop / max(thresholds.max_mask_iou_drop, 1e-12)))
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
            "min_identity_confidence": min(confidence_values),
            "mean_identity_confidence": sum(confidence_values) / len(confidence_values),
            "final_identity_confidence": confidence_values[-1],
            "max_texture_drift": max(texture_drift_values),
            "mean_texture_drift": sum(texture_drift_values) / len(texture_drift_values),
            "initial_mask_iou_to_initial": mask_iou_values[0],
            "final_mask_iou_to_initial": mask_iou_values[-1],
            "max_mask_iou_drop": max_mask_iou_drop,
            "cloth_identity_score": identity_score,
        },
        "flags": _flags(
            min_identity_confidence=min(confidence_values),
            max_texture_drift=max(texture_drift_values),
            max_mask_iou_drop=max_mask_iou_drop,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream cloth identity, texture, and mask measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ClothMoveIdentityDriftProbeError(f"CLOTH_IDENTITY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "identity_confidence", "texture_drift", "mask_iou_to_initial")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAME_INVALID")
    for key in ("identity_confidence", "texture_drift", "mask_iou_to_initial"):
        if parsed[key] < 0 or parsed[key] > 1:
            raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ClothMoveIdentityDriftProbeError(f"CLOTH_IDENTITY_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    min_identity_confidence: float,
    max_texture_drift: float,
    max_mask_iou_drop: float,
    thresholds: ClothIdentityDriftThresholds,
) -> list[str]:
    flags = []
    if min_identity_confidence < thresholds.min_identity_confidence:
        flags.append("low_cloth_identity_confidence")
    if max_texture_drift > thresholds.max_texture_drift:
        flags.append("cloth_texture_drift")
    if max_mask_iou_drop > thresholds.max_mask_iou_drop:
        flags.append("cloth_mask_identity_drift")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ClothMoveIdentityDriftProbeError("CLOTH_IDENTITY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
