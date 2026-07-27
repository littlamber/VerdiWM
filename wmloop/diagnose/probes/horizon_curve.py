"""Long-horizon roll-out aggregation without hiding local collapses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wmloop.diagnose.diagnoser import summarize_horizon_curve, summarize_segment_drift


def analyze_horizon_curve(
    *, metric: str, by_horizon: Mapping[int, float], per_frame: Mapping[int, float], segment_window: int = 12
) -> dict[str, Any]:
    summary = summarize_horizon_curve(metric=metric, observations=by_horizon)
    summary["segment_drift"] = summarize_segment_drift(observations=per_frame, window=segment_window)
    return summary
