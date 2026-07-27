"""Diagnostic-only surface-fold probe for ACWM-Phys cloth_move."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_cloth_move_surface_fold_diagnostic_v1"
ENVIRONMENT = "cloth_move"
SIGNATURE = "surface_fold"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ClothMoveSurfaceFoldProbeError(ValueError):
    """Cloth-move surface-fold diagnostic input or output is invalid."""


@dataclass(frozen=True)
class SurfaceFoldThresholds:
    max_fold_area_fraction: float = 0.35
    max_fold_sharpness: float = 0.45
    max_fold_area_growth: float = 0.20

    def __post_init__(self) -> None:
        values = (self.max_fold_area_fraction, self.max_fold_sharpness, self.max_fold_area_growth)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("CLOTH_SURFACE_FOLD_THRESHOLDS_INVALID")


def measure_surface_fold(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: SurfaceFoldThresholds = SurfaceFoldThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured cloth fold area and fold sharpness."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_COUNT_INSUFFICIENT")
    fold_area_values = [frame["fold_area_fraction"] for frame in parsed]
    sharpness_values = [frame["fold_sharpness"] for frame in parsed]
    max_fold_area_fraction = max(fold_area_values)
    max_fold_sharpness = max(sharpness_values)
    fold_area_growth = max_fold_area_fraction - fold_area_values[0]
    fold_stability_score = _clip01(
        (
            (1.0 - _clip01(max_fold_area_fraction / max(thresholds.max_fold_area_fraction, 1e-12)))
            + (1.0 - _clip01(max_fold_sharpness / max(thresholds.max_fold_sharpness, 1e-12)))
            + (1.0 - _clip01(max(0.0, fold_area_growth) / max(thresholds.max_fold_area_growth, 1e-12)))
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
            "initial_fold_area_fraction": fold_area_values[0],
            "final_fold_area_fraction": fold_area_values[-1],
            "max_fold_area_fraction": max_fold_area_fraction,
            "fold_area_growth": fold_area_growth,
            "max_fold_sharpness": max_fold_sharpness,
            "mean_fold_sharpness": sum(sharpness_values) / len(sharpness_values),
            "fold_stability_score": fold_stability_score,
        },
        "flags": _flags(
            max_fold_area_fraction=max_fold_area_fraction,
            max_fold_sharpness=max_fold_sharpness,
            fold_area_growth=fold_area_growth,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream cloth fold measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ClothMoveSurfaceFoldProbeError(f"CLOTH_SURFACE_FOLD_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "fold_area_fraction", "fold_sharpness")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_INVALID")
    if parsed["fold_area_fraction"] < 0 or parsed["fold_area_fraction"] > 1:
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_INVALID")
    if parsed["fold_sharpness"] < 0 or parsed["fold_sharpness"] > 1:
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ClothMoveSurfaceFoldProbeError(f"CLOTH_SURFACE_FOLD_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    max_fold_area_fraction: float,
    max_fold_sharpness: float,
    fold_area_growth: float,
    thresholds: SurfaceFoldThresholds,
) -> list[str]:
    flags = []
    if max_fold_area_fraction > thresholds.max_fold_area_fraction:
        flags.append("large_surface_fold")
    if max_fold_sharpness > thresholds.max_fold_sharpness:
        flags.append("sharp_surface_fold")
    if fold_area_growth > thresholds.max_fold_area_growth:
        flags.append("progressive_fold_growth")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ClothMoveSurfaceFoldProbeError("CLOTH_SURFACE_FOLD_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
