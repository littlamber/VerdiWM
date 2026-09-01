"""WAN2.2-DROID conditioning adapter contract.

This file contains the model-facing shape contract only.  The actual WAN2.2
DiT hook is injected by the selected runtime runner; keeping this projection
separate prevents a Wan2.1 adapter from being loaded by name or by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Wan22DroidConditioning:
    """A validated batch of aligned robot conditions."""

    actions: Sequence[Sequence[float]]
    proprio: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        if not self.actions or not self.proprio or len(self.actions) != len(self.proprio):
            raise ValueError("WAN22_DROID_CONDITIONING_LENGTH_INVALID")
        if any(len(row) != 7 for row in self.actions):
            raise ValueError("WAN22_DROID_ACTION_DIM_INVALID")
        if any(len(row) != 14 for row in self.proprio):
            raise ValueError("WAN22_DROID_PROPRIO_DIM_INVALID")


def validate_window(actions: Sequence[Sequence[float]], proprio: Sequence[Sequence[float]], *, horizon_frames: int = 150) -> Wan22DroidConditioning:
    """Validate one first-frame-conditioned future-prediction window."""

    if len(actions) != horizon_frames or len(proprio) != horizon_frames:
        raise ValueError("WAN22_DROID_HORIZON_INVALID")
    return Wan22DroidConditioning(actions=actions, proprio=proprio)
