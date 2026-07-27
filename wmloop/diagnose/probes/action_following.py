"""Per-frame inverse-dynamics and no-action-control measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence


def per_frame_inverse_dynamics_accuracy(
    *, predicted_actions: Sequence[Sequence[float]], target_actions: Sequence[Sequence[float]], tolerance: float = 1e-3
) -> float:
    if len(predicted_actions) != len(target_actions) or not predicted_actions or tolerance < 0:
        raise ValueError("ACTION_PROBE_INPUT_INVALID")
    correct = 0
    for predicted, target in zip(predicted_actions, target_actions):
        if len(predicted) != len(target) or not predicted:
            raise ValueError("ACTION_PROBE_INPUT_INVALID")
        if any(not math.isfinite(value) for value in (*predicted, *target)):
            raise ValueError("ACTION_PROBE_INPUT_INVALID")
        correct += int(all(abs(left - right) <= tolerance for left, right in zip(predicted, target)))
    return correct / len(target_actions)


def no_action_delta_psnr(*, action_conditioned_psnr: float, no_action_psnr: float) -> float:
    if not math.isfinite(action_conditioned_psnr) or not math.isfinite(no_action_psnr):
        raise ValueError("ACTION_CONTROL_METRIC_INVALID")
    return action_conditioned_psnr - no_action_psnr
