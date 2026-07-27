"""Diagnostic-only particle-boundary probe for ACWM-Phys push_sand."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_sand_particle_boundary_diagnostic_v1"
ENVIRONMENT = "push_sand"
SIGNATURE = "particle_boundary"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushSandParticleBoundaryProbeError(ValueError):
    """Push-sand particle-boundary diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ParticleBoundaryThresholds:
    max_escape_fraction: float = 0.08
    max_escape_growth: float = 0.05
    min_final_inside_ratio: float = 0.88
    min_mean_boundary_contact_score: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.max_escape_fraction,
            self.max_escape_growth,
            self.min_final_inside_ratio,
            self.min_mean_boundary_contact_score,
        )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("PARTICLE_BOUNDARY_THRESHOLDS_INVALID")


def measure_particle_boundary(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ParticleBoundaryThresholds = ParticleBoundaryThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured inside/outside particle-boundary statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_COUNT_INSUFFICIENT")
    escape_fractions = [_escape_fraction(frame) for frame in parsed]
    inside_ratios = [1.0 - fraction for fraction in escape_fractions]
    max_escape_fraction = max(escape_fractions)
    escape_growth = max_escape_fraction - escape_fractions[0]
    boundary_contact_scores = [frame["boundary_contact_score"] for frame in parsed]
    mean_boundary_contact_score = sum(boundary_contact_scores) / len(boundary_contact_scores)
    boundary_integrity_score = _clip01(
        (
            (1.0 - _clip01(max_escape_fraction / max(thresholds.max_escape_fraction, 1e-12)))
            + (1.0 - _clip01(max(0.0, escape_growth) / max(thresholds.max_escape_growth, 1e-12)))
            + _clip01(inside_ratios[-1] / max(thresholds.min_final_inside_ratio, 1e-12))
            + _clip01(
                mean_boundary_contact_score / max(thresholds.min_mean_boundary_contact_score, 1e-12)
            )
        )
        / 4.0
    )
    total_particle_mass_values = [frame["inside_boundary_mass"] + frame["outside_boundary_mass"] for frame in parsed]
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
            "initial_escape_fraction": escape_fractions[0],
            "final_escape_fraction": escape_fractions[-1],
            "max_escape_fraction": max_escape_fraction,
            "escape_growth": escape_growth,
            "final_inside_ratio": inside_ratios[-1],
            "min_inside_ratio": min(inside_ratios),
            "mean_boundary_contact_score": mean_boundary_contact_score,
            "initial_total_particle_mass": total_particle_mass_values[0],
            "final_total_particle_mass": total_particle_mass_values[-1],
            "boundary_integrity_score": boundary_integrity_score,
        },
        "flags": _flags(
            max_escape_fraction=max_escape_fraction,
            escape_growth=escape_growth,
            final_inside_ratio=inside_ratios[-1],
            mean_boundary_contact_score=mean_boundary_contact_score,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream in-boundary and out-of-boundary particle measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushSandParticleBoundaryProbeError(f"PARTICLE_BOUNDARY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "inside_boundary_mass", "outside_boundary_mass", "boundary_contact_score")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_INVALID")
    if parsed["inside_boundary_mass"] < 0 or parsed["outside_boundary_mass"] < 0:
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_INVALID")
    if parsed["boundary_contact_score"] < 0 or parsed["boundary_contact_score"] > 1:
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_FRAME_INVALID")
    if parsed["inside_boundary_mass"] + parsed["outside_boundary_mass"] <= 0:
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_TOTAL_MASS_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushSandParticleBoundaryProbeError(f"PARTICLE_BOUNDARY_VALUE_INVALID:{key}")
    return float(value)


def _escape_fraction(frame: Mapping[str, float]) -> float:
    return frame["outside_boundary_mass"] / (frame["inside_boundary_mass"] + frame["outside_boundary_mass"])


def _flags(
    *,
    max_escape_fraction: float,
    escape_growth: float,
    final_inside_ratio: float,
    mean_boundary_contact_score: float,
    thresholds: ParticleBoundaryThresholds,
) -> list[str]:
    flags = []
    if max_escape_fraction > thresholds.max_escape_fraction:
        flags.append("particle_boundary_escape")
    if escape_growth > thresholds.max_escape_growth:
        flags.append("progressive_boundary_escape")
    if final_inside_ratio < thresholds.min_final_inside_ratio:
        flags.append("low_boundary_retention")
    if mean_boundary_contact_score < thresholds.min_mean_boundary_contact_score:
        flags.append("weak_boundary_contact")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushSandParticleBoundaryProbeError("PARTICLE_BOUNDARY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
