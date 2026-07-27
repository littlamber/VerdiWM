"""Diagnostic-only container-boundary leak probe for ACWM-Phys pour_water."""

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


PROBE_ID = "acwm_pour_water_container_boundary_leak_diagnostic_v1"
ENVIRONMENT = "pour_water"
SIGNATURE = "container_boundary_leak"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class ContainerBoundaryLeakProbeError(ValueError):
    """Container-boundary leak diagnostic input or output is invalid."""


@dataclass(frozen=True)
class ContainerLeakThresholds:
    max_leak_fraction: float = 0.12
    max_leak_growth: float = 0.08
    min_final_containment_ratio: float = 0.80

    def __post_init__(self) -> None:
        values = (self.max_leak_fraction, self.max_leak_growth, self.min_final_containment_ratio)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("CONTAINER_LEAK_THRESHOLDS_INVALID")


def measure_container_boundary_leak(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: ContainerLeakThresholds = ContainerLeakThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured inside/outside water-mask areas."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAME_COUNT_INSUFFICIENT")
    leak_fractions = [_leak_fraction(frame) for frame in parsed]
    containment_ratios = [1.0 - fraction for fraction in leak_fractions]
    max_leak_fraction = max(leak_fractions)
    final_leak_fraction = leak_fractions[-1]
    leak_growth = max_leak_fraction - leak_fractions[0]
    total_area_values = [frame["in_container_area"] + frame["outside_container_area"] for frame in parsed]
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
            "initial_leak_fraction": leak_fractions[0],
            "final_leak_fraction": final_leak_fraction,
            "max_leak_fraction": max_leak_fraction,
            "leak_growth": leak_growth,
            "final_containment_ratio": containment_ratios[-1],
            "min_containment_ratio": min(containment_ratios),
            "mean_containment_ratio": sum(containment_ratios) / len(containment_ratios),
            "initial_total_water_area": total_area_values[0],
            "final_total_water_area": total_area_values[-1],
            "total_area_retention_ratio": total_area_values[-1] / total_area_values[0],
            "leak_score": 1.0 - _clip01((max_leak_fraction + max(0.0, leak_growth)) / 2.0),
        },
        "flags": _flags(
            max_leak_fraction=max_leak_fraction,
            leak_growth=leak_growth,
            final_containment_ratio=containment_ratios[-1],
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream in-container and outside-container water-mask measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise ContainerBoundaryLeakProbeError(f"CONTAINER_LEAK_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "in_container_area", "outside_container_area")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAME_INVALID")
    if parsed["in_container_area"] < 0 or parsed["outside_container_area"] < 0:
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_FRAME_INVALID")
    if parsed["in_container_area"] + parsed["outside_container_area"] <= 0:
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_TOTAL_AREA_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContainerBoundaryLeakProbeError(f"CONTAINER_LEAK_VALUE_INVALID:{key}")
    return float(value)


def _leak_fraction(frame: Mapping[str, float]) -> float:
    return frame["outside_container_area"] / (frame["in_container_area"] + frame["outside_container_area"])


def _flags(
    *,
    max_leak_fraction: float,
    leak_growth: float,
    final_containment_ratio: float,
    thresholds: ContainerLeakThresholds,
) -> list[str]:
    flags = []
    if max_leak_fraction > thresholds.max_leak_fraction:
        flags.append("boundary_leak_detected")
    if leak_growth > thresholds.max_leak_growth:
        flags.append("progressive_leak")
    if final_containment_ratio < thresholds.min_final_containment_ratio:
        flags.append("low_final_containment")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_measurements(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_MEASUREMENTS_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-container-boundary-leak-measurements"
        or payload.get("environment") != ENVIRONMENT
        or not isinstance(payload.get("frames"), list)
    ):
        raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_MEASUREMENTS_INVALID")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    measurements = _load_measurements(args.measurements)
    output = measure_container_boundary_leak(
        frames=measurements["frames"],
        evidence_refs=measurements.get("evidence_refs", []),
    )
    payload = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise ContainerBoundaryLeakProbeError("CONTAINER_LEAK_OUTPUT_EXISTS")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
