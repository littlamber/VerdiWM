"""Stable ``wan-future-rgb-v1`` numerical recipe for diagnostic video pairs.

This file is frozen as algorithm evidence. Changing its numerical semantics
requires a new recipe version; CLI and process-management changes live in
``future_video_metrics`` and do not change this recipe's identity.
"""

from __future__ import annotations

import math

import numpy as np


RECIPE_ID = "wan-future-rgb-v1"
METRIC_DIRECTIONS = {"psnr_db": "higher", "global_ssim_rgb": "higher", "aggregate_mse": "lower"}


class FutureVideoMetricError(ValueError):
    """The frozen numeric recipe could not be reproduced."""


def global_rgb_ssim(x: np.ndarray, y: np.ndarray, *, data_range: float = 255.0) -> float:
    """Population-moment SSIM across all RGB pixels, matching the freeze."""
    x, y = x.reshape(-1), y.reshape(-1)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    variance_x, variance_y = float(np.var(x)), float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
    )
    return float(max(-1.0, min(1.0, value)))


def future_frame_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Use future frames 1..12; PSNR is computed from aggregate RGB MSE."""
    if (prediction.shape != (13, 512, 768, 3) or target.shape != prediction.shape
        or prediction.dtype != np.uint8 or target.dtype != np.uint8):
        raise FutureVideoMetricError("FUTURE_VIDEO_FRAME_CONTRACT_INVALID")
    rows = []
    for index in range(1, 13):
        x, y = prediction[index].astype(np.float64), target[index].astype(np.float64)
        mse = float(np.mean(np.square(x - y)))
        ssim = global_rgb_ssim(x, y)
        if not math.isfinite(mse) or not math.isfinite(ssim):
            raise FutureVideoMetricError("FUTURE_VIDEO_METRIC_NOT_FINITE")
        rows.append({"video_frame_index": index, "mse": mse, "global_ssim_rgb": ssim})
    mse = float(np.mean([row["mse"] for row in rows]))
    return {
        "metrics": {
            "future_frame_count": 12, "aggregate_mse": mse,
            "psnr_db": None if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse),
            "psnr_is_infinite": mse == 0,
            "global_ssim_rgb": float(np.mean([row["global_ssim_rgb"] for row in rows])),
        },
        "per_frame": rows,
    }
