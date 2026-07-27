"""Diagnostic-only topology-change probe for ACWM-Phys push_rope."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


PROBE_ID = "acwm_push_rope_topology_change_diagnostic_v1"
ENVIRONMENT = "push_rope"
SIGNATURE = "topology_change"
_CAS_REF_PATTERN = re.compile(r"^cas://sha256/[0-9a-f]{64}$")


class PushRopeTopologyChangeProbeError(ValueError):
    """Push-rope topology-change diagnostic input or output is invalid."""


@dataclass(frozen=True)
class TopologyChangeThresholds:
    max_component_count_delta: int = 0
    max_crossing_count_delta: int = 1
    max_bend_energy_growth: float = 0.40

    def __post_init__(self) -> None:
        if self.max_component_count_delta < 0 or self.max_crossing_count_delta < 0:
            raise ValueError("ROPE_TOPOLOGY_THRESHOLDS_INVALID")
        if not math.isfinite(self.max_bend_energy_growth) or self.max_bend_energy_growth < 0:
            raise ValueError("ROPE_TOPOLOGY_THRESHOLDS_INVALID")


def measure_topology_change(
    *,
    frames: Sequence[Mapping[str, Any]],
    thresholds: TopologyChangeThresholds = TopologyChangeThresholds(),
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Aggregate measured rope component, crossing, and bend-energy statistics."""

    parsed = _parse_frames(frames)
    if len(parsed) < 2:
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_COUNT_INSUFFICIENT")
    initial_bend_energy = parsed[0]["bend_energy"]
    if initial_bend_energy <= 0:
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_INITIAL_BEND_ENERGY_INVALID")
    component_count_delta = max(abs(int(frame["component_count"]) - int(parsed[0]["component_count"])) for frame in parsed)
    crossing_count_delta = max(abs(int(frame["crossing_count"]) - int(parsed[0]["crossing_count"])) for frame in parsed)
    bend_energy_values = [frame["bend_energy"] for frame in parsed]
    bend_energy_growth = (max(bend_energy_values) - initial_bend_energy) / initial_bend_energy
    topology_stability_score = _clip01(
        (
            (1.0 - _clip01(component_count_delta / max(thresholds.max_component_count_delta, 1e-12)))
            + (1.0 - _clip01(crossing_count_delta / max(thresholds.max_crossing_count_delta, 1e-12)))
            + (1.0 - _clip01(bend_energy_growth / max(thresholds.max_bend_energy_growth, 1e-12)))
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
            "initial_component_count": int(parsed[0]["component_count"]),
            "final_component_count": int(parsed[-1]["component_count"]),
            "component_count_delta": component_count_delta,
            "initial_crossing_count": int(parsed[0]["crossing_count"]),
            "final_crossing_count": int(parsed[-1]["crossing_count"]),
            "crossing_count_delta": crossing_count_delta,
            "initial_bend_energy": initial_bend_energy,
            "max_bend_energy": max(bend_energy_values),
            "bend_energy_growth": bend_energy_growth,
            "topology_stability_score": topology_stability_score,
        },
        "flags": _flags(
            component_count_delta=component_count_delta,
            crossing_count_delta=crossing_count_delta,
            bend_energy_growth=bend_energy_growth,
            thresholds=thresholds,
        ),
        "evidence_refs": _evidence_refs(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "This diagnostic output must not be routed into verdict_evidence during the active campaign.",
            "The probe assumes upstream rope skeleton topology measurements are already produced by a separate adapter.",
        ],
    }
    try:
        validate_document("diagnostic_probe_output", output)
    except ContractValidationError as exc:
        raise PushRopeTopologyChangeProbeError(f"ROPE_TOPOLOGY_OUTPUT_CONTRACT_INVALID:{exc}") from exc
    return output


def _parse_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAMES_INVALID")
    parsed = [_parse_frame(frame) for frame in frames]
    if len({int(frame["frame"]) for frame in parsed}) != len(parsed):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_DUPLICATE")
    return sorted(parsed, key=lambda frame: frame["frame"])


def _parse_frame(frame: Mapping[str, Any]) -> dict[str, float]:
    required = ("frame", "component_count", "crossing_count", "bend_energy")
    if not isinstance(frame, Mapping) or any(key not in frame for key in required):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_INVALID")
    parsed = {key: _finite(frame[key], key) for key in required}
    if parsed["frame"] < 0 or parsed["frame"] != int(parsed["frame"]):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_INVALID")
    if parsed["component_count"] < 1 or parsed["component_count"] != int(parsed["component_count"]):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_INVALID")
    if parsed["crossing_count"] < 0 or parsed["crossing_count"] != int(parsed["crossing_count"]):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_INVALID")
    if parsed["bend_energy"] < 0:
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_FRAME_INVALID")
    return parsed


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PushRopeTopologyChangeProbeError(f"ROPE_TOPOLOGY_VALUE_INVALID:{key}")
    return float(value)


def _flags(
    *,
    component_count_delta: int,
    crossing_count_delta: int,
    bend_energy_growth: float,
    thresholds: TopologyChangeThresholds,
) -> list[str]:
    flags = []
    if component_count_delta > thresholds.max_component_count_delta:
        flags.append("rope_component_count_changed")
    if crossing_count_delta > thresholds.max_crossing_count_delta:
        flags.append("rope_crossing_instability")
    if bend_energy_growth > thresholds.max_bend_energy_growth:
        flags.append("rope_bend_energy_spike")
    return flags


def _evidence_refs(value: Sequence[str]) -> list[str]:
    refs = list(value)
    if any(not isinstance(ref, str) or _CAS_REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise PushRopeTopologyChangeProbeError("ROPE_TOPOLOGY_EVIDENCE_REFS_INVALID")
    return refs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
