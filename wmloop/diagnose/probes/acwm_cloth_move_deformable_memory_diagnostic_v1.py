"""Diagnostic-only deformable-memory probe for ACWM-Phys cloth_move."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_cloth_move_deformable_memory_diagnostic_v1"
ENVIRONMENT = "cloth_move"
SIGNATURE = "deformable_memory"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ClothMoveDeformableMemoryProbeError(ValueError):
    """Cloth-move deformable-memory diagnostic input or output is invalid."""


@dataclass(frozen=True)
class DeformableMemoryThresholds:
    max_mean_landmark_error: float = 0.08
    max_final_landmark_error: float = 0.10
    max_shape_memory_loss: float = 0.25
    min_recovery_fraction: float = 0.50

    def __post_init__(self) -> None:
        values = (
            self.max_mean_landmark_error,
            self.max_final_landmark_error,
            self.max_shape_memory_loss,
            self.min_recovery_fraction,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("CLOTH_DEFORMABLE_MEMORY_THRESHOLDS_INVALID")
        if self.max_shape_memory_loss > 1 or self.min_recovery_fraction > 1:
            raise ValueError("CLOTH_DEFORMABLE_MEMORY_THRESHOLDS_INVALID")


def measure_deformable_memory(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: DeformableMemoryThresholds = DeformableMemoryThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate cloth landmark error, shape memory loss, and recovery."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_COUNT_INSUFFICIENT")
    landmark_errors = [frame["landmark_error"] for frame in parsed]
    shape_memory_losses = [frame["shape_memory_loss"] for frame in parsed]
    recovery_values = [frame["recovery_fraction"] for frame in parsed]
    mean_landmark_error = sum(landmark_errors) / len(landmark_errors)
    deformable_memory_score = _clip01(
        (
            (1.0 - _clip01(mean_landmark_error / max(thresholds.max_mean_landmark_error, 1e-12)))
            + (1.0 - _clip01(landmark_errors[-1] / max(thresholds.max_final_landmark_error, 1e-12)))
            + (1.0 - _clip01(max(shape_memory_losses) / max(thresholds.max_shape_memory_loss, 1e-12)))
            + _clip01(recovery_values[-1] / max(thresholds.min_recovery_fraction, 1e-12))
        )
        / 4.0
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
            "initial_landmark_error": landmark_errors[0],
            "final_landmark_error": landmark_errors[-1],
            "mean_landmark_error": mean_landmark_error,
            "max_landmark_error": max(landmark_errors),
            "max_shape_memory_loss": max(shape_memory_losses),
            "mean_shape_memory_loss": sum(shape_memory_losses) / len(shape_memory_losses),
            "final_recovery_fraction": recovery_values[-1],
            "min_recovery_fraction": min(recovery_values),
            "deformable_memory_score": deformable_memory_score,
        },
        "flags": _flags(
            mean_landmark_error=mean_landmark_error,
            final_landmark_error=landmark_errors[-1],
            max_shape_memory_loss=max(shape_memory_losses),
            final_recovery_fraction=recovery_values[-1],
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream cloth landmark and shape-memory measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ClothMoveDeformableMemoryProbeError(f"CLOTH_DEFORMABLE_MEMORY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "landmark_error", "shape_memory_loss", "recovery_fraction")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_INVALID")
    if parsed["landmark_error"] < 0:
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_INVALID")
    for key in ("shape_memory_loss", "recovery_fraction"):
        if parsed[key] < 0 or parsed[key] > 1:
            raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ClothMoveDeformableMemoryProbeError(f"CLOTH_DEFORMABLE_MEMORY_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    mean_landmark_error: float,
    final_landmark_error: float,
    max_shape_memory_loss: float,
    final_recovery_fraction: float,
    thresholds: DeformableMemoryThresholds,
) -> list[str]:
    flags = []
    if mean_landmark_error > thresholds.max_mean_landmark_error:
        flags.append("high_mean_deformable_landmark_error")
    if final_landmark_error > thresholds.max_final_landmark_error:
        flags.append("final_deformable_landmark_miss")
    if max_shape_memory_loss > thresholds.max_shape_memory_loss:
        flags.append("shape_memory_loss")
    if final_recovery_fraction < thresholds.min_recovery_fraction:
        flags.append("weak_deformable_recovery")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ClothMoveDeformableMemoryProbeError("CLOTH_DEFORMABLE_MEMORY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
