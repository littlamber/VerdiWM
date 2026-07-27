"""Diagnostic-only support-relation probe for ACWM-Phys stack_cube."""

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


PROBE_ID = "acwm_stack_cube_support_relation_diagnostic_v1"
ENVIRONMENT = "stack_cube"
SIGNATURE = "support_relation"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class StackCubeSupportRelationProbeError(ValueError):
    """Stack-cube support-relation diagnostic input or output is invalid."""


@dataclass(frozen=True)
class SupportRelationThresholds:
    min_support_overlap_ratio: float = 0.55
    max_vertical_gap: float = 0.08
    min_stable_frame_fraction: float = 0.60

    def __post_init__(self) -> None:
        values = (self.min_support_overlap_ratio, self.max_vertical_gap, self.min_stable_frame_fraction)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("STACK_CUBE_SUPPORT_THRESHOLDS_INVALID")
        if self.min_support_overlap_ratio > 1 or self.min_stable_frame_fraction > 1:
            raise ValueError("STACK_CUBE_SUPPORT_THRESHOLDS_INVALID")


def measure_support_relation(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: SupportRelationThresholds = SupportRelationThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured top/bottom cube support statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAME_COUNT_INSUFFICIENT")
    stable = [
        frame
        for frame in parsed
        if frame["support_overlap_ratio"] >= thresholds.min_support_overlap_ratio
        and abs(frame["vertical_gap"]) <= thresholds.max_vertical_gap
    ]
    stable_frame_fraction = len(stable) / len(parsed)
    support_loss_count = _support_loss_count(parsed, thresholds)
    overlap_values = [frame["support_overlap_ratio"] for frame in parsed]
    gap_values = [abs(frame["vertical_gap"]) for frame in parsed]
    support_scores = [
        _clip01(frame["support_overlap_ratio"])
        * (1.0 - _clip01(abs(frame["vertical_gap"]) / max(thresholds.max_vertical_gap, 1e-12)))
        for frame in parsed
    ]
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
            "stable_frame_count": len(stable),
            "stable_frame_fraction": stable_frame_fraction,
            "support_loss_count": support_loss_count,
            "min_support_overlap_ratio": min(overlap_values),
            "mean_support_overlap_ratio": sum(overlap_values) / len(overlap_values),
            "final_support_overlap_ratio": parsed[-1]["support_overlap_ratio"],
            "max_abs_vertical_gap": max(gap_values),
            "mean_abs_vertical_gap": sum(gap_values) / len(gap_values),
            "final_abs_vertical_gap": abs(parsed[-1]["vertical_gap"]),
            "support_score": sum(support_scores) / len(support_scores),
        },
        "flags": _flags(
            stable_frame_fraction=stable_frame_fraction,
            final_overlap=parsed[-1]["support_overlap_ratio"],
            max_abs_vertical_gap=max(gap_values),
            support_loss_count=support_loss_count,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream top-cube, bottom-cube, overlap, and vertical-gap measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise StackCubeSupportRelationProbeError(f"STACK_CUBE_SUPPORT_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "support_overlap_ratio", "vertical_gap")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAME_INVALID")
    if parsed["support_overlap_ratio"] < 0 or parsed["support_overlap_ratio"] > 1:
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StackCubeSupportRelationProbeError(f"STACK_CUBE_SUPPORT_VALUE_INVALID:{key}")
    return float(value)


def _support_loss_count(frames: Sequence[Mapping[str, float]], thresholds: SupportRelationThresholds) -> int:
    states = [
        frame["support_overlap_ratio"] >= thresholds.min_support_overlap_ratio
        and abs(frame["vertical_gap"]) <= thresholds.max_vertical_gap
        for frame in frames
    ]
    return sum(1 for previous, current in zip(states, states[1:]) if previous and not current)


def _flags(
    *,
    stable_frame_fraction: float,
    final_overlap: float,
    max_abs_vertical_gap: float,
    support_loss_count: int,
    thresholds: SupportRelationThresholds,
) -> list[str]:
    flags = []
    if stable_frame_fraction < thresholds.min_stable_frame_fraction:
        flags.append("support_relation_unstable")
    if final_overlap < thresholds.min_support_overlap_ratio:
        flags.append("low_final_support_overlap")
    if max_abs_vertical_gap > thresholds.max_vertical_gap:
        flags.append("vertical_gap_unstable")
    if support_loss_count > 0:
        flags.append("support_relation_lost")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_measurements(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_MEASUREMENTS_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-stack-cube-support-relation-measurements"
        or payload.get("environment") != ENVIRONMENT
        or not isinstance(payload.get("frames"), list)
    ):
        raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_MEASUREMENTS_INVALID")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    measurements = _load_measurements(args.measurements)
    output = measure_support_relation(
        frames=measurements["frames"],
        evidence_refs=measurements.get("evidence_refs", []),
    )
    payload = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise StackCubeSupportRelationProbeError("STACK_CUBE_SUPPORT_OUTPUT_EXISTS")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
