"""Diagnostic-only mass-redistribution probe for ACWM-Phys push_sand."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_sand_mass_redistribution_diagnostic_v1"
ENVIRONMENT = "push_sand"
SIGNATURE = "mass_redistribution"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushSandMassRedistributionProbeError(ValueError):
    """Push-sand mass-redistribution diagnostic input or output is invalid."""


@dataclass(frozen=True)
class MassRedistributionThresholds:
    max_total_mass_drift_fraction: float = 0.05
    min_centroid_displacement: float = 0.03
    min_redistributed_mass_fraction: float = 0.08
    max_orthogonal_drift: float = 0.06

    def __post_init__(self) -> None:
        values = (
            self.max_total_mass_drift_fraction,
            self.min_centroid_displacement,
            self.min_redistributed_mass_fraction,
            self.max_orthogonal_drift,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("MASS_REDISTRIBUTION_THRESHOLDS_INVALID")


def measure_mass_redistribution(
    *,
    frames: Sequence[Mapping[str, Any]],
    expected_motion: tuple[float, float] = (1.0, 0.0),
    thresholds: MassRedistributionThresholds = MassRedistributionThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured sand mass conservation and transport statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAME_COUNT_INSUFFICIENT")
    direction = _normalised_direction(expected_motion)
    initial_total_mass = parsed[0]["total_mass"]
    if initial_total_mass <= 0:
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_INITIAL_MASS_INVALID")
    total_mass_values = [frame["total_mass"] for frame in parsed]
    total_mass_drift_fraction = max(abs(value - initial_total_mass) / initial_total_mass for value in total_mass_values)
    centroid_dx = parsed[-1]["mass_centroid_x"] - parsed[0]["mass_centroid_x"]
    centroid_dy = parsed[-1]["mass_centroid_y"] - parsed[0]["mass_centroid_y"]
    axial_centroid_displacement = centroid_dx * direction[0] + centroid_dy * direction[1]
    orthogonal_drift = abs(centroid_dx * direction[1] - centroid_dy * direction[0])
    redistributed_mass_values = [frame["redistributed_mass_fraction"] for frame in parsed]
    max_redistributed_mass_fraction = max(redistributed_mass_values)
    redistribution_score = _clip01(
        (
            (1.0 - _clip01(total_mass_drift_fraction / max(thresholds.max_total_mass_drift_fraction, 1e-12)))
            + _clip01(axial_centroid_displacement / max(thresholds.min_centroid_displacement, 1e-12))
            + _clip01(
                max_redistributed_mass_fraction / max(thresholds.min_redistributed_mass_fraction, 1e-12)
            )
            + (1.0 - _clip01(orthogonal_drift / max(thresholds.max_orthogonal_drift, 1e-12)))
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
            "initial_total_mass": initial_total_mass,
            "final_total_mass": total_mass_values[-1],
            "total_mass_drift_fraction": total_mass_drift_fraction,
            "centroid_dx": centroid_dx,
            "centroid_dy": centroid_dy,
            "axial_centroid_displacement": axial_centroid_displacement,
            "orthogonal_drift": orthogonal_drift,
            "max_redistributed_mass_fraction": max_redistributed_mass_fraction,
            "mean_redistributed_mass_fraction": sum(redistributed_mass_values) / len(redistributed_mass_values),
            "redistribution_score": redistribution_score,
        },
        "flags": _flags(
            total_mass_drift_fraction=total_mass_drift_fraction,
            axial_centroid_displacement=axial_centroid_displacement,
            max_redistributed_mass_fraction=max_redistributed_mass_fraction,
            orthogonal_drift=orthogonal_drift,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream sand mass and centroid measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushSandMassRedistributionProbeError(f"MASS_REDISTRIBUTION_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "total_mass", "mass_centroid_x", "mass_centroid_y", "redistributed_mass_fraction")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAME_INVALID")
    if parsed["total_mass"] < 0 or parsed["redistributed_mass_fraction"] < 0 or parsed["redistributed_mass_fraction"] > 1:
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushSandMassRedistributionProbeError(f"MASS_REDISTRIBUTION_VALUE_INVALID:{key}")
    return float(value)


def _normalised_direction(value: tuple[float, float]) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_EXPECTED_MOTION_INVALID")
    dx = float(value[0])
    dy = float(value[1])
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm <= 0:
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_EXPECTED_MOTION_INVALID")
    return dx / norm, dy / norm


def _flags(
    *,
    total_mass_drift_fraction: float,
    axial_centroid_displacement: float,
    max_redistributed_mass_fraction: float,
    orthogonal_drift: float,
    thresholds: MassRedistributionThresholds,
) -> list[str]:
    flags = []
    if total_mass_drift_fraction > thresholds.max_total_mass_drift_fraction:
        flags.append("mass_not_conserved")
    if axial_centroid_displacement < thresholds.min_centroid_displacement:
        flags.append("weak_centroid_transport")
    if max_redistributed_mass_fraction < thresholds.min_redistributed_mass_fraction:
        flags.append("low_redistributed_mass")
    if orthogonal_drift > thresholds.max_orthogonal_drift:
        flags.append("off_axis_mass_drift")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushSandMassRedistributionProbeError("MASS_REDISTRIBUTION_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
