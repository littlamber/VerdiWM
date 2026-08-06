"""Frozen, deterministic verification for generic auto-experiment results."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


class AutoExperimentVerificationError(ValueError):
    """An auto-experiment result or verification contract is malformed."""


_OPERATORS = {
    "gte": lambda value, threshold: value >= threshold,
    "lte": lambda value, threshold: value <= threshold,
    "gt": lambda value, threshold: value > threshold,
    "lt": lambda value, threshold: value < threshold,
}


def verify_auto_experiment_result(
    *,
    result: Mapping[str, object],
    metric_gates: Sequence[Mapping[str, object]],
    expected_gpu_uuid: str,
    gpu_sampling: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    """Verify output metrics and prove that the leased GPU performed work."""

    blockers: list[dict[str, object]] = []
    if result.get("schema_version") != 1 or result.get("artifact_type") != "verdiwm-auto-experiment-result":
        blockers.append({"code": "RESULT_CONTRACT_INVALID"})
    if result.get("state") != "ready":
        blockers.append({"code": "RESULT_NOT_READY", "observed": result.get("state")})
    device = result.get("device")
    if not isinstance(device, Mapping) or device.get("type") != "cuda":
        blockers.append({"code": "RESULT_CUDA_DEVICE_REQUIRED"})
    else:
        observed_uuid = device.get("gpu_uuid")
        if observed_uuid != expected_gpu_uuid:
            blockers.append(
                {
                    "code": "RESULT_GPU_UUID_MISMATCH",
                    "expected": expected_gpu_uuid,
                    "observed": observed_uuid,
                }
            )

    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        blockers.append({"code": "RESULT_METRICS_INVALID"})
        metrics = {}
    gate_results = _evaluate_metric_gates(metrics=metrics, gates=metric_gates)
    blockers.extend(
        {"code": "METRIC_GATE_FAILED", **gate}
        for gate in gate_results
        if gate["pass"] is not True
    )
    gpu_activity = _gpu_activity_summary(gpu_sampling, expected_gpu_uuid=expected_gpu_uuid)
    if gpu_activity["verified"] is not True:
        blockers.append({"code": "PHYSICAL_GPU_ACTIVITY_UNVERIFIED", **gpu_activity})

    passed = not blockers
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-verdict",
        "state": "ready",
        "verdict": "PASS" if passed else "VOID",
        "stage": stage,
        "evidence_level": _evidence_level(stage, passed=passed),
        "expected_gpu_uuid": expected_gpu_uuid,
        "gpu_activity": gpu_activity,
        "metric_gates": gate_results,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "claim_boundary": (
            "A passing generic runtime verdict is execution evidence. It remains exploratory until a "
            "backbone-specific frozen evaluator promotes it into effect memory."
        ),
    }


def _evaluate_metric_gates(
    *,
    metrics: Mapping[str, object],
    gates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not gates:
        raise AutoExperimentVerificationError("AUTO_EXPERIMENT_METRIC_GATES_EMPTY")
    results = []
    for gate in gates:
        metric = gate.get("metric")
        operator = gate.get("operator")
        threshold = gate.get("threshold")
        if not isinstance(metric, str) or not metric or operator not in _OPERATORS or not _finite_number(threshold):
            raise AutoExperimentVerificationError("AUTO_EXPERIMENT_METRIC_GATE_INVALID")
        value = metrics.get(metric)
        passed = _finite_number(value) and _OPERATORS[str(operator)](float(value), float(threshold))
        results.append(
            {
                "metric": metric,
                "role": str(gate.get("role") or "primary"),
                "operator": operator,
                "threshold": float(threshold),
                "observed": float(value) if _finite_number(value) else None,
                "pass": bool(passed),
            }
        )
    return results


def _gpu_activity_summary(
    sampling: Mapping[str, object], *, expected_gpu_uuid: str
) -> dict[str, object]:
    raw_samples = sampling.get("samples")
    samples = [item for item in raw_samples if isinstance(item, Mapping)] if isinstance(raw_samples, list) else []
    matching = [
        item
        for item in samples
        if item.get("status") == "ready" and item.get("gpu_uuid") == expected_gpu_uuid
    ]
    memories = [
        float(item["memory_used_mib"])
        for item in matching
        if _finite_number(item.get("memory_used_mib"))
    ]
    utilizations = [
        float(item["utilization_gpu_percent"])
        for item in matching
        if _finite_number(item.get("utilization_gpu_percent"))
    ]
    baseline_memory = min(memories) if memories else None
    peak_memory = max(memories) if memories else None
    peak_utilization = max(utilizations) if utilizations else None
    memory_delta = (
        peak_memory - baseline_memory
        if baseline_memory is not None and peak_memory is not None
        else None
    )
    during_samples = [item for item in matching if item.get("phase") == "during"]
    active_during = [
        item
        for item in during_samples
        if (_number(item.get("utilization_gpu_percent")) or 0.0) > 0.0
        or (
            baseline_memory is not None
            and (_number(item.get("memory_used_mib")) or 0.0) >= baseline_memory + 32.0
        )
    ]
    return {
        "verified": bool(matching and active_during),
        "sample_count": len(samples),
        "matching_gpu_sample_count": len(matching),
        "active_during_sample_count": len(active_during),
        "baseline_memory_used_mib": baseline_memory,
        "peak_memory_used_mib": peak_memory,
        "memory_delta_mib": memory_delta,
        "peak_utilization_gpu_percent": peak_utilization,
    }


def _evidence_level(stage: str, *, passed: bool) -> str:
    if not passed:
        return "void"
    return {
        "smoke": "runtime_verified",
        "screen": "screened",
        "gate": "gate_passed",
        "confirm": "confirmed_result",
    }.get(stage, "exploratory")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: object) -> float | None:
    return float(value) if _finite_number(value) else None
