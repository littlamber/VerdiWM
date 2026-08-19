"""Deterministic admission planning for training versus runtime methods.

The planner is deliberately advisory: it computes a reproducible resource
proposal from the method contract, while the resource portfolio and GPU lease
remain the only execution authorities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class TrainingResourcePlanningError(ValueError):
    """The method contract is insufficient for a safe resource proposal."""


METHOD_CLASSES = {
    "inference_runtime": False,
    "diagnostic": False,
    "adapter_training": True,
    "backbone_training": True,
}


def plan_training_resources(
    *,
    method_class: str,
    trainable_parameters: int = 0,
    train_examples: int = 0,
    sequence_length: int = 1,
    batch_size: int = 1,
    planned_steps: int = 0,
    available_gpus: Sequence[int] = tuple(range(6)),
    competing_candidates: int = 1,
) -> dict[str, object]:
    """Return a scale proposal without launching or reserving any GPU.

    Small adapters favor candidate parallelism.  Large/backbone jobs favor a
    power-of-two world size, capped by the six Ctrl-World GPUs.  The thresholds
    are policy inputs, not claims about model quality.
    """
    if method_class not in METHOD_CLASSES:
        raise TrainingResourcePlanningError("TRAINING_RESOURCE_METHOD_CLASS_INVALID")
    integers = {
        "trainable_parameters": trainable_parameters,
        "train_examples": train_examples,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "planned_steps": planned_steps,
        "competing_candidates": competing_candidates,
    }
    for name, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingResourcePlanningError(f"TRAINING_RESOURCE_{name.upper()}_INVALID")
    gpus = sorted({int(value) for value in available_gpus})
    if not gpus:
        raise TrainingResourcePlanningError("TRAINING_RESOURCE_GPU_SET_EMPTY")
    requires_training = METHOD_CLASSES[method_class]
    if not requires_training:
        return {
            "schema_version": 1,
            "state": "runtime_only",
            "method_class": method_class,
            "requires_training": False,
            "requested_gpu_count": 1,
            "world_size": 1,
            "parallelism_decision": "runtime_or_parallel_screen",
            "gpu_hours_estimate": 0.0,
            "rationale": "The method changes execution or diagnosis, not trainable weights.",
        }
    if trainable_parameters < 1 or train_examples < 1 or planned_steps < 1:
        raise TrainingResourcePlanningError("TRAINING_RESOURCE_TRAINING_METADATA_REQUIRED")
    # A rough activation-plus-optimizer pressure score. It is only used to
    # choose a proposal; actual admission still validates the scale receipt.
    pressure = (
        trainable_parameters / 1e8
        + sequence_length / 512.0
        + batch_size / 8.0
    )
    if method_class == "adapter_training" and pressure < 1.5 and trainable_parameters < 50_000_000:
        world_size = 1
        decision = "single_gpu_parallel_candidates"
    else:
        desired = 1 if pressure < 2.0 else 2 if pressure < 5.0 else 4
        if method_class == "backbone_training" and pressure >= 8.0:
            desired = 6
        world_size = min(len(gpus), desired)
        decision = "distributed_training" if world_size > 1 else "single_gpu_training"
    # More candidate slots make a one-GPU proposal more valuable unless the
    # method is genuinely too large to fit efficiently on one device.
    if world_size == 1 and competing_candidates > 1:
        decision = "single_gpu_parallel_candidates"
    gpu_hours = planned_steps * max(1, batch_size) * max(1, sequence_length) / 1_000_000
    return {
        "schema_version": 1,
        "state": "ready",
        "method_class": method_class,
        "requires_training": True,
        "requested_gpu_count": world_size,
        "world_size": world_size,
        "available_gpu_indices": gpus,
        "parallelism_decision": decision,
        "estimated_pressure": round(pressure, 6),
        "gpu_hours_estimate": round(float(gpu_hours) / max(1, world_size), 6),
        "rationale": (
            "Scale is derived from trainable parameter count, sequence length, batch size, "
            "planned steps, and opportunity cost; it is not an LLM-controlled lease."
        ),
    }


def method_class_from_candidate(candidate: Mapping[str, object]) -> str:
    """Map a materialized candidate kind to the planner's coarse contract."""
    kind = str(candidate.get("candidate_kind") or "")
    if "masked_intermediate_action_adapter" in kind:
        return "adapter_training"
    if any(token in kind for token in ("memory", "guidance", "schedule", "diagnostic")):
        return "inference_runtime"
    if "training" in kind or "backbone" in kind:
        return "backbone_training"
    return "diagnostic"
