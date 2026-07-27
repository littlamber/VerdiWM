"""Diagnostic-only contact-event probe for ACWM-Phys push_cube."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_cube_contact_event_diagnostic_v1"
ENVIRONMENT = "push_cube"
SIGNATURE = "contact_event"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushCubeContactEventProbeError(ValueError):
    """Push-cube contact-event diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ContactEventThresholds:
    min_contact_score: float = 0.50
    min_contact_frame_count: int = 1
    min_post_contact_displacement: float = 0.03
    max_pre_contact_drift: float = 0.02

    def __post_init__(self) -> None:
        floats = (self.min_contact_score, self.min_post_contact_displacement, self.max_pre_contact_drift)
        if any(not math.isfinite(value) or value < 0 for value in floats) or self.min_contact_score > 1:
            raise ValueError("PUSH_CUBE_CONTACT_THRESHOLDS_INVALID")
        if self.min_contact_frame_count < 1:
            raise ValueError("PUSH_CUBE_CONTACT_THRESHOLDS_INVALID")


def measure_contact_event(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ContactEventThresholds = ContactEventThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured cube/pusher contact statistics into diagnostic evidence.

    This probe intentionally consumes upstream geometric measurements rather
    than performing object detection.  Its output is diagnostic-only and can be
    used for proposal routing after admission, not for frozen verdict evidence.
    """

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_FRAME_COUNT_INSUFFICIENT")
    contact_indices = [
        index for index, frame in enumerate(parsed) if frame["contact_score"] >= thresholds.min_contact_score
    ]
    first_contact_index = contact_indices[0] if contact_indices else None
    first_contact_frame = int(parsed[first_contact_index]["frame"]) if first_contact_index is not None else None
    max_contact_score = max(frame["contact_score"] for frame in parsed)
    mean_contact_score = sum(frame["contact_score"] for frame in parsed) / len(parsed)
    if first_contact_index is None:
        pre_contact_drift = _cube_distance(parsed[0], parsed[-1])
        post_contact_displacement = 0.0
    else:
        pre_anchor_index = max(0, first_contact_index - 1)
        pre_contact_drift = _cube_distance(parsed[0], parsed[pre_anchor_index])
        post_contact_displacement = _cube_distance(parsed[first_contact_index], parsed[-1])
    total_cube_displacement = _cube_distance(parsed[0], parsed[-1])
    pusher_displacement = _pusher_distance(parsed[0], parsed[-1])
    contact_response_ratio = 0.0 if pusher_displacement == 0 else post_contact_displacement / pusher_displacement
    event_score = _clip01(
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
            "mean_contact_score": mean_contact_score,
            "pre_contact_cube_drift": pre_contact_drift,
            "post_contact_cube_displacement": post_contact_displacement,
            "total_cube_displacement": total_cube_displacement,
            "pusher_displacement": pusher_displacement,
            "contact_response_ratio": contact_response_ratio,
            "event_score": event_score,
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
            "The probe assumes upstream cube, pusher, and contact measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushCubeContactEventProbeError(f"PUSH_CUBE_CONTACT_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = (
        "frame",
        "cube_centroid_x",
        "cube_centroid_y",
        "pusher_centroid_x",
        "pusher_centroid_y",
        "contact_score",
    )
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if (
        parsed["frame"] < 0
        or parsed["frame"] != int(parsed["frame"])
        or parsed["contact_score"] < 0
        or parsed["contact_score"] > 1
    ):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushCubeContactEventProbeError(f"PUSH_CUBE_CONTACT_VALUE_INVALID:{key}")
    return float(value)


def _cube_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return math.hypot(second["cube_centroid_x"] - first["cube_centroid_x"], second["cube_centroid_y"] - first["cube_centroid_y"])


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
    thresholds: ContactEventThresholds,
) -> list[str]:
    flags = []
    if contact_frame_count < thresholds.min_contact_frame_count:
        flags.append("missing_contact_event")
    if post_contact_displacement < thresholds.min_post_contact_displacement:
        flags.append("weak_contact_response")
    if pre_contact_drift > thresholds.max_pre_contact_drift:
        flags.append("pre_contact_cube_drift")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_measurements(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_MEASUREMENTS_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-push-cube-contact-event-measurements"
        or payload.get("environment") != ENVIRONMENT
        or not isinstance(payload.get("frames"), list)
    ):
        raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_MEASUREMENTS_INVALID")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    measurements = _load_measurements(args.measurements)
    output = measure_contact_event(
        frames=measurements["frames"],
        evidence_refs=measurements.get("evidence_refs", []),
    )
    payload = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise PushCubeContactEventProbeError("PUSH_CUBE_CONTACT_OUTPUT_EXISTS")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
