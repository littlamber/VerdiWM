"""Diagnostic-only fluid-volume transport probe for ACWM-Phys pour_water."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_pour_water_fluid_volume_transport_diagnostic_v1"
ENVIRONMENT = "pour_water"
SIGNATURE = "fluid_volume_transport"


class FluidVolumeTransportProbeError(ValueError):
    """Fluid-volume transport diagnostic input or output is invalid."""


@dataclass(frozen=True)
class FluidTransportThresholds:
    min_retained_area_ratio: float = 0.70
    min_target_progress: float = 0.25
    max_spill_fraction: float = 0.15

    def __post_init__(self) -> None:
        values = (self.min_retained_area_ratio, self.min_target_progress, self.max_spill_fraction)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("FLUID_TRANSPORT_THRESHOLDS_INVALID")


def measure_fluid_volume_transport(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: FluidTransportThresholds = FluidTransportThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured water-mask statistics into diagnostic-only evidence.

    The probe intentionally consumes explicit measurements instead of doing
    segmentation itself.  A later adapter can produce these fields from video,
    but this function keeps the diagnostic contract pure and testable.
    """

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_FRAME_COUNT_INSUFFICIENT")
    first = parsed[0]
    last = parsed[-1]
    initial_area = first["water_area"]
    if initial_area <= 0:
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_INITIAL_AREA_INVALID")
    min_area = min(frame["water_area"] for frame in parsed)
    max_spill_area = max(frame["spill_area"] for frame in parsed)
    retained_area_ratio = last["water_area"] / initial_area
    max_area_loss_fraction = max(0.0, (initial_area - min_area) / initial_area)
    spill_fraction = max_spill_area / initial_area
    start_distance = _distance(first)
    end_distance = _distance(last)
    target_progress = 1.0 if start_distance == 0 and end_distance == 0 else 0.0 if start_distance == 0 else (start_distance - end_distance) / start_distance
    clipped_retention = _clip01(retained_area_ratio)
    clipped_progress = _clip01(target_progress)
    transport_score = _clip01((clipped_retention + clipped_progress + (1.0 - _clip01(spill_fraction))) / 3.0)
    flags = _flags(
        retained_area_ratio=retained_area_ratio,
        target_progress=target_progress,
        spill_fraction=spill_fraction,
        thresholds=thresholds,
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
            "initial_water_area": initial_area,
            "final_water_area": last["water_area"],
            "min_water_area": min_area,
            "retained_area_ratio": retained_area_ratio,
            "max_area_loss_fraction": max_area_loss_fraction,
            "max_spill_area": max_spill_area,
            "spill_fraction": spill_fraction,
            "start_target_distance": start_distance,
            "end_target_distance": end_distance,
            "target_progress": target_progress,
            "transport_score": transport_score,
        },
        "flags": flags,
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream water-mask and target measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise FluidVolumeTransportProbeError(f"FLUID_TRANSPORT_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = (
        "frame",
        "water_area",
        "water_centroid_x",
        "water_centroid_y",
        "target_centroid_x",
        "target_centroid_y",
    )
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    parsed["spill_area"] = _finite(frame.get("spill_area", 0.0), "spill_area")
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]) or parsed["water_area"] < 0 or parsed["spill_area"] < 0:
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FluidVolumeTransportProbeError(f"FLUID_TRANSPORT_VALUE_INVALID:{key}")
    return float(value)


def _distance(frame: Mapping[str, float]) -> float:
    return math.hypot(
        frame["water_centroid_x"] - frame["target_centroid_x"],
        frame["water_centroid_y"] - frame["target_centroid_y"],
    )


def _flags(
    *,
    retained_area_ratio: float,
    target_progress: float,
    spill_fraction: float,
    thresholds: FluidTransportThresholds,
) -> list[str]:
    flags = []
    if retained_area_ratio < thresholds.min_retained_area_ratio:
        flags.append("low_volume_retention")
    if target_progress < thresholds.min_target_progress:
        flags.append("poor_transport_progress")
    if spill_fraction > thresholds.max_spill_fraction:
        flags.append("container_boundary_leak")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or not ref.startswith("cas://sha256/") for ref in refs):
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_measurements(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_MEASUREMENTS_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-fluid-transport-measurements"
        or payload.get("environment") != ENVIRONMENT
        or not isinstance(payload.get("frames"), list)
    ):
        raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_MEASUREMENTS_INVALID")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    measurements = _load_measurements(args.measurements)
    output = measure_fluid_volume_transport(
        frames=measurements["frames"],
        evidence_refs=measurements.get("evidence_refs", []),
    )
    payload = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise FluidVolumeTransportProbeError("FLUID_TRANSPORT_OUTPUT_EXISTS")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
