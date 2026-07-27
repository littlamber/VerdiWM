"""Diagnostic-only rigid-pose slip probe for ACWM-Phys push_cube."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_cube_rigid_pose_slip_diagnostic_v1"
ENVIRONMENT = "push_cube"
SIGNATURE = "rigid_pose_slip"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushCubeRigidPoseSlipProbeError(ValueError):
    """Push-cube rigid-pose slip diagnostic input or output is invalid."""


@dataclass(frozen=True)
class RigidPoseSlipThresholds:
    max_no_contact_translation: float = 0.02
    max_no_contact_rotation: float = 0.05
    max_no_contact_score: float = 0.20

    def __post_init__(self) -> None:
        values = (self.max_no_contact_translation, self.max_no_contact_rotation, self.max_no_contact_score)
        if any(not math.isfinite(value) or value < 0 for value in values) or self.max_no_contact_score > 1:
            raise ValueError("PUSH_CUBE_RIGID_SLIP_THRESHOLDS_INVALID")


def measure_rigid_pose_slip(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: RigidPoseSlipThresholds = RigidPoseSlipThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Measure predicted rigid pose drift when the cube is not in contact."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_FRAME_COUNT_INSUFFICIENT")
    no_contact_deltas = []
    contact_deltas = []
    for previous, current in zip(parsed, parsed[1:]):
        delta = {
            "translation": _translation(previous, current),
            "rotation": abs(current["cube_angle"] - previous["cube_angle"]),
        }
        if previous["contact_score"] <= thresholds.max_no_contact_score and current["contact_score"] <= thresholds.max_no_contact_score:
            no_contact_deltas.append(delta)
        else:
            contact_deltas.append(delta)
    max_no_contact_translation = max((delta["translation"] for delta in no_contact_deltas), default=0.0)
    max_no_contact_rotation = max((delta["rotation"] for delta in no_contact_deltas), default=0.0)
    mean_no_contact_translation = (
        sum(delta["translation"] for delta in no_contact_deltas) / len(no_contact_deltas) if no_contact_deltas else 0.0
    )
    mean_contact_translation = (
        sum(delta["translation"] for delta in contact_deltas) / len(contact_deltas) if contact_deltas else 0.0
    )
    stability_score = 1.0 - _clip01(
        (
            max_no_contact_translation / max(thresholds.max_no_contact_translation, 1e-12)
            + max_no_contact_rotation / max(thresholds.max_no_contact_rotation, 1e-12)
        )
        / 2.0
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
            "no_contact_interval_count": len(no_contact_deltas),
            "contact_interval_count": len(contact_deltas),
            "max_no_contact_translation": max_no_contact_translation,
            "mean_no_contact_translation": mean_no_contact_translation,
            "max_no_contact_rotation": max_no_contact_rotation,
            "mean_contact_translation": mean_contact_translation,
            "pose_stability_score": stability_score,
        },
        "flags": _flags(
            max_no_contact_translation=max_no_contact_translation,
            max_no_contact_rotation=max_no_contact_rotation,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream cube pose and contact measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushCubeRigidPoseSlipProbeError(f"PUSH_CUBE_RIGID_SLIP_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "cube_centroid_x", "cube_centroid_y", "cube_angle", "contact_score")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]) or not 0 <= parsed["contact_score"] <= 1:
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushCubeRigidPoseSlipProbeError(f"PUSH_CUBE_RIGID_SLIP_VALUE_INVALID:{key}")
    return float(value)


def _translation(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    return math.hypot(
        current["cube_centroid_x"] - previous["cube_centroid_x"],
        current["cube_centroid_y"] - previous["cube_centroid_y"],
    )


def _flags(
    *,
    max_no_contact_translation: float,
    max_no_contact_rotation: float,
    thresholds: RigidPoseSlipThresholds,
) -> list[str]:
    flags = []
    if max_no_contact_translation > thresholds.max_no_contact_translation:
        flags.append("no_contact_translation_slip")
    if max_no_contact_rotation > thresholds.max_no_contact_rotation:
        flags.append("no_contact_rotation_slip")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushCubeRigidPoseSlipProbeError("PUSH_CUBE_RIGID_SLIP_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
