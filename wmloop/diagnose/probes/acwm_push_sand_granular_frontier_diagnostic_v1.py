"""Diagnostic-only granular-frontier probe for ACWM-Phys push_sand."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_sand_granular_frontier_diagnostic_v1"
ENVIRONMENT = "push_sand"
SIGNATURE = "granular_frontier"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushSandGranularFrontierProbeError(ValueError):
    """Push-sand granular-frontier diagnostic input or output is invalid."""


@dataclass(frozen=True)
class GranularFrontierThresholds:
    min_frontier_progress: float = 0.04
    min_active_area_retention: float = 0.55
    max_frontier_roughness: float = 0.35
    min_mean_frontier_speed: float = 0.005

    def __post_init__(self) -> None:
        values = (
            self.min_frontier_progress,
            self.min_active_area_retention,
            self.max_frontier_roughness,
            self.min_mean_frontier_speed,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("GRANULAR_FRONTIER_THRESHOLDS_INVALID")
        if self.min_active_area_retention > 1:
            raise ValueError("GRANULAR_FRONTIER_THRESHOLDS_INVALID")


def measure_granular_frontier(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: GranularFrontierThresholds = GranularFrontierThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured sand-mask frontier propagation statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAME_COUNT_INSUFFICIENT")
    initial_area = parsed[0]["active_sand_area"]
    if initial_area <= 0:
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_INITIAL_AREA_INVALID")
    frontier_progress = parsed[-1]["frontier_position"] - parsed[0]["frontier_position"]
    active_area_retention = parsed[-1]["active_sand_area"] / initial_area
    roughness_values = [frame["frontier_roughness"] for frame in parsed]
    speed_values = [frame["frontier_speed"] for frame in parsed]
    max_frontier_roughness = max(roughness_values)
    mean_frontier_speed = sum(speed_values) / len(speed_values)
    frontier_score = _clip01(
        (
            _clip01(frontier_progress / max(thresholds.min_frontier_progress, 1e-12))
            + _clip01(active_area_retention / max(thresholds.min_active_area_retention, 1e-12))
            + (1.0 - _clip01(max_frontier_roughness / max(thresholds.max_frontier_roughness, 1e-12)))
            + _clip01(mean_frontier_speed / max(thresholds.min_mean_frontier_speed, 1e-12))
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
            "initial_frontier_position": parsed[0]["frontier_position"],
            "final_frontier_position": parsed[-1]["frontier_position"],
            "frontier_progress": frontier_progress,
            "initial_active_sand_area": initial_area,
            "final_active_sand_area": parsed[-1]["active_sand_area"],
            "active_area_retention": active_area_retention,
            "max_frontier_roughness": max_frontier_roughness,
            "mean_frontier_roughness": sum(roughness_values) / len(roughness_values),
            "mean_frontier_speed": mean_frontier_speed,
            "granular_frontier_score": frontier_score,
        },
        "flags": _flags(
            frontier_progress=frontier_progress,
            active_area_retention=active_area_retention,
            max_frontier_roughness=max_frontier_roughness,
            mean_frontier_speed=mean_frontier_speed,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream sand-mask frontier measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushSandGranularFrontierProbeError(f"GRANULAR_FRONTIER_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "frontier_position", "active_sand_area", "frontier_roughness", "frontier_speed")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAME_INVALID")
    if parsed["active_sand_area"] < 0 or parsed["frontier_roughness"] < 0 or parsed["frontier_speed"] < 0:
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushSandGranularFrontierProbeError(f"GRANULAR_FRONTIER_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    frontier_progress: float,
    active_area_retention: float,
    max_frontier_roughness: float,
    mean_frontier_speed: float,
    thresholds: GranularFrontierThresholds,
) -> list[str]:
    flags = []
    if frontier_progress < thresholds.min_frontier_progress:
        flags.append("weak_granular_frontier_motion")
    if active_area_retention < thresholds.min_active_area_retention:
        flags.append("active_area_collapse")
    if max_frontier_roughness > thresholds.max_frontier_roughness:
        flags.append("frontier_fragmentation")
    if mean_frontier_speed < thresholds.min_mean_frontier_speed:
        flags.append("stalled_frontier_velocity")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushSandGranularFrontierProbeError("GRANULAR_FRONTIER_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
