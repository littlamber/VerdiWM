"""Diagnostic-only free-surface probe for ACWM-Phys pour_water."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_pour_water_free_surface_diagnostic_v1"
ENVIRONMENT = "pour_water"
SIGNATURE = "free_surface"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PourWaterFreeSurfaceProbeError(ValueError):
    """Pour-water free-surface diagnostic input or output is invalid."""


@dataclass(frozen=True)
class FreeSurfaceThresholds:
    max_surface_roughness: float = 0.25
    max_area_variation_fraction: float = 0.25
    max_centroid_y_range: float = 0.20

    def __post_init__(self) -> None:
        values = (self.max_surface_roughness, self.max_area_variation_fraction, self.max_centroid_y_range)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("FREE_SURFACE_THRESHOLDS_INVALID")


def measure_free_surface(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: FreeSurfaceThresholds = FreeSurfaceThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured free-surface area, centroid, and roughness."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAME_COUNT_INSUFFICIENT")
    initial_area = parsed[0]["surface_area"]
    if initial_area <= 0:
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_INITIAL_AREA_INVALID")
    areas = [frame["surface_area"] for frame in parsed]
    roughness_values = [frame["surface_roughness"] for frame in parsed]
    centroid_y_values = [frame["surface_centroid_y"] for frame in parsed]
    area_variation_fraction = (max(areas) - min(areas)) / initial_area
    centroid_y_range = max(centroid_y_values) - min(centroid_y_values)
    surface_score = 1.0 - _clip01(
        (
            max(roughness_values) / max(thresholds.max_surface_roughness, 1e-12)
            + area_variation_fraction / max(thresholds.max_area_variation_fraction, 1e-12)
            + centroid_y_range / max(thresholds.max_centroid_y_range, 1e-12)
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
            "initial_surface_area": initial_area,
            "final_surface_area": areas[-1],
            "area_variation_fraction": area_variation_fraction,
            "max_surface_roughness": max(roughness_values),
            "mean_surface_roughness": sum(roughness_values) / len(roughness_values),
            "centroid_y_range": centroid_y_range,
            "surface_score": surface_score,
        },
        "flags": _flags(
            max_surface_roughness=max(roughness_values),
            area_variation_fraction=area_variation_fraction,
            centroid_y_range=centroid_y_range,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream water free-surface measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PourWaterFreeSurfaceProbeError(f"FREE_SURFACE_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "surface_area", "surface_centroid_y", "surface_roughness")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAME_INVALID")
    if parsed["surface_area"] < 0 or parsed["surface_roughness"] < 0:
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PourWaterFreeSurfaceProbeError(f"FREE_SURFACE_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    max_surface_roughness: float,
    area_variation_fraction: float,
    centroid_y_range: float,
    thresholds: FreeSurfaceThresholds,
) -> list[str]:
    flags = []
    if max_surface_roughness > thresholds.max_surface_roughness:
        flags.append("surface_unstable")
    if area_variation_fraction > thresholds.max_area_variation_fraction:
        flags.append("surface_area_flicker")
    if centroid_y_range > thresholds.max_centroid_y_range:
        flags.append("surface_centroid_oscillation")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PourWaterFreeSurfaceProbeError("FREE_SURFACE_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
