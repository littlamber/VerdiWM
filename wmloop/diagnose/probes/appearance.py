"""Low-motion appearance retention probe computed from measured frame pairs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def low_motion_ssim(*, motion_magnitudes: Sequence[float], ssim_scores: Sequence[float], fraction: float = 0.25) -> float:
    if (
        len(motion_magnitudes) != len(ssim_scores)
        or not motion_magnitudes
        or not 0 < fraction <= 1
        or any(not math.isfinite(value) or value < 0 for value in motion_magnitudes)
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in ssim_scores)
    ):
        raise ValueError("APPEARANCE_PROBE_INPUT_INVALID")
    count = max(1, math.ceil(len(motion_magnitudes) * fraction))
    selected = sorted(zip(motion_magnitudes, ssim_scores), key=lambda item: item[0])[:count]
    return sum(score for _, score in selected) / len(selected)
