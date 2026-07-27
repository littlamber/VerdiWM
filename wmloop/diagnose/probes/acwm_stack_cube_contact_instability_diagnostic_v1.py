"""Diagnostic-only contact-instability probe for ACWM-Phys stack_cube."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_stack_cube_contact_instability_diagnostic_v1"
ENVIRONMENT = "stack_cube"
SIGNATURE = "contact_instability"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class StackCubeContactInstabilityProbeError(ValueError):
    """Stack-cube contact-instability diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ContactInstabilityThresholds:
    min_contact_score: float = 0.50
    min_contact_active_fraction: float = 0.50
    max_contact_transition_count: int = 2
    max_contact_pose_jitter: float = 0.03

    def __post_init__(self) -> None:
        floats = (self.min_contact_score, self.min_contact_active_fraction, self.max_contact_pose_jitter)
        if any(not math.isfinite(value) or value < 0 for value in floats):
            raise ValueError("STACK_CUBE_CONTACT_INSTABILITY_THRESHOLDS_INVALID")
        if self.min_contact_score > 1 or self.min_contact_active_fraction > 1 or self.max_contact_transition_count < 0:
            raise ValueError("STACK_CUBE_CONTACT_INSTABILITY_THRESHOLDS_INVALID")


def measure_contact_instability(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ContactInstabilityThresholds = ContactInstabilityThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure contact flicker and top-cube jitter during stack contact."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_FRAME_COUNT_INSUFFICIENT")
    contact_states = [frame["contact_score"] >= thresholds.min_contact_score for frame in parsed]
    contact_transition_count = sum(1 for previous, current in zip(contact_states, contact_states[1:]) if previous != current)
    contact_active_fraction = sum(1 for state in contact_states if state) / len(contact_states)
    contact_jitters = [
        _top_translation(previous, current)
        for previous, current, previous_contact, current_contact in zip(
            parsed, parsed[1:], contact_states, contact_states[1:]
        )
        if previous_contact and current_contact
    ]
    max_contact_pose_jitter = max(contact_jitters, default=0.0)
    mean_contact_score = sum(frame["contact_score"] for frame in parsed) / len(parsed)
    instability_score = _clip01(
        (
            _clip01(contact_transition_count / max(thresholds.max_contact_transition_count, 1))
            + _clip01(max_contact_pose_jitter / max(thresholds.max_contact_pose_jitter, 1e-12))
            + (1.0 - contact_active_fraction)
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
            "contact_active_fraction": contact_active_fraction,
            "contact_transition_count": contact_transition_count,
            "max_contact_pose_jitter": max_contact_pose_jitter,
            "mean_contact_pose_jitter": sum(contact_jitters) / len(contact_jitters) if contact_jitters else 0.0,
            "mean_contact_score": mean_contact_score,
            "instability_score": instability_score,
        },
        "flags": _flags(
            contact_active_fraction=contact_active_fraction,
            contact_transition_count=contact_transition_count,
            max_contact_pose_jitter=max_contact_pose_jitter,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream stack contact and top-cube pose measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise StackCubeContactInstabilityProbeError(f"STACK_CUBE_CONTACT_INSTABILITY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "top_cube_centroid_x", "top_cube_centroid_y", "contact_score")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]) or not 0 <= parsed["contact_score"] <= 1:
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StackCubeContactInstabilityProbeError(f"STACK_CUBE_CONTACT_INSTABILITY_VALUE_INVALID:{key}")
    return float(value)


def _top_translation(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    return math.hypot(
        current["top_cube_centroid_x"] - previous["top_cube_centroid_x"],
        current["top_cube_centroid_y"] - previous["top_cube_centroid_y"],
    )


def _flags(
    *,
    contact_active_fraction: float,
    contact_transition_count: int,
    max_contact_pose_jitter: float,
    thresholds: ContactInstabilityThresholds,
) -> list[str]:
    flags = []
    if contact_active_fraction < thresholds.min_contact_active_fraction:
        flags.append("weak_stack_contact")
    if contact_transition_count > thresholds.max_contact_transition_count:
        flags.append("contact_flicker")
    if max_contact_pose_jitter > thresholds.max_contact_pose_jitter:
        flags.append("high_contact_pose_jitter")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise StackCubeContactInstabilityProbeError("STACK_CUBE_CONTACT_INSTABILITY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
