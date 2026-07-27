"""Diagnostic-only deformable-contact probe for ACWM-Phys push_rope."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_rope_deformable_contact_diagnostic_v1"
ENVIRONMENT = "push_rope"
SIGNATURE = "deformable_contact"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushRopeDeformableContactProbeError(ValueError):
    """Push-rope deformable-contact diagnostic input or output is invalid."""


@dataclass(frozen=True)
class DeformableContactThresholds:
    min_contact_score: float = 0.45
    min_contact_frame_count: int = 1
    min_post_contact_displacement: float = 0.03
    max_pre_contact_drift: float = 0.025

    def __post_init__(self) -> None:
        values = (self.min_contact_score, self.min_post_contact_displacement, self.max_pre_contact_drift)
        if any(not math.isfinite(value) or value < 0 for value in values) or self.min_contact_score > 1:
            raise ValueError("ROPE_DEFORMABLE_CONTACT_THRESHOLDS_INVALID")
        if self.min_contact_frame_count < 1:
            raise ValueError("ROPE_DEFORMABLE_CONTACT_THRESHOLDS_INVALID")


def measure_deformable_contact(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: DeformableContactThresholds = DeformableContactThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured pusher/rope contact and response statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAME_COUNT_INSUFFICIENT")
    contact_indices = [
        index for index, frame in enumerate(parsed) if frame["contact_score"] >= thresholds.min_contact_score
    ]
    first_contact_index = contact_indices[0] if contact_indices else None
    first_contact_frame = int(parsed[first_contact_index]["frame"]) if first_contact_index is not None else None
    max_contact_score = max(frame["contact_score"] for frame in parsed)
    if first_contact_index is None:
        pre_contact_drift = _rope_distance(parsed[0], parsed[-1])
        post_contact_displacement = 0.0
    else:
        pre_anchor_index = max(0, first_contact_index - 1)
        pre_contact_drift = _rope_distance(parsed[0], parsed[pre_anchor_index])
        post_contact_displacement = _rope_distance(parsed[first_contact_index], parsed[-1])
    pusher_displacement = _pusher_distance(parsed[0], parsed[-1])
    contact_response_ratio = 0.0 if pusher_displacement == 0 else post_contact_displacement / pusher_displacement
    deformable_contact_score = _clip01(
        (
            _clip01(len(contact_indices) / thresholds.min_contact_frame_count)
            + _clip01(post_contact_displacement / thresholds.min_post_contact_displacement)
            + (1.0 - _clip01(pre_contact_drift / max(thresholds.max_pre_contact_drift, 1e-12)))
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
            "contact_frame_count": len(contact_indices),
            "first_contact_frame": first_contact_frame,
            "max_contact_score": max_contact_score,
            "mean_contact_score": sum(frame["contact_score"] for frame in parsed) / len(parsed),
            "pre_contact_rope_drift": pre_contact_drift,
            "post_contact_rope_displacement": post_contact_displacement,
            "pusher_displacement": pusher_displacement,
            "contact_response_ratio": contact_response_ratio,
            "deformable_contact_score": deformable_contact_score,
        },
        "flags": _flags(
            contact_frame_count=len(contact_indices),
            post_contact_displacement=post_contact_displacement,
            pre_contact_drift=pre_contact_drift,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream rope, pusher, and contact measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushRopeDeformableContactProbeError(f"ROPE_DEFORMABLE_CONTACT_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = (
        "frame",
        "rope_centroid_x",
        "rope_centroid_y",
        "pusher_centroid_x",
        "pusher_centroid_y",
        "contact_score",
    )
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAME_INVALID")
    if parsed["contact_score"] < 0 or parsed["contact_score"] > 1:
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushRopeDeformableContactProbeError(f"ROPE_DEFORMABLE_CONTACT_VALUE_INVALID:{key}")
    return float(value)


def _rope_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return math.hypot(second["rope_centroid_x"] - first["rope_centroid_x"], second["rope_centroid_y"] - first["rope_centroid_y"])


def _pusher_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return math.hypot(
        second["pusher_centroid_x"] - first["pusher_centroid_x"],
        second["pusher_centroid_y"] - first["pusher_centroid_y"],
    )


def _flags(
    *,
    contact_frame_count: int,
    post_contact_displacement: float,
    pre_contact_drift: float,
    thresholds: DeformableContactThresholds,
) -> list[str]:
    flags = []
    if contact_frame_count < thresholds.min_contact_frame_count:
        flags.append("missing_deformable_contact")
    if post_contact_displacement < thresholds.min_post_contact_displacement:
        flags.append("weak_rope_contact_response")
    if pre_contact_drift > thresholds.max_pre_contact_drift:
        flags.append("pre_contact_rope_drift")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushRopeDeformableContactProbeError("ROPE_DEFORMABLE_CONTACT_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
