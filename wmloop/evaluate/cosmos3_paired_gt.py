"""Fail-closed paired-GT metrics for Cosmos3 forward dynamics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


class Cosmos3PairedGroundTruthError(ValueError):
    """The rollout cannot be aligned to the frozen Cosmos3 target window."""


def compute_cosmos3_paired_metrics(
    *,
    ground_truth: np.ndarray,
    rollout: np.ndarray,
    horizon_frames: int = 16,
    max_conditioning_frame_mae: float = 8.0 / 255.0,
) -> tuple[dict[str, float], dict[str, object]]:
    """Measure future frames after validating the shared conditioning frame.

    The official action pipeline keeps content at the top-left when removing
    reflection padding.  Rollouts can therefore be shorter or narrower than
    the decoded DROID tensor, but arbitrary resize or center-crop alignment is
    forbidden.
    """

    gt = _rgb_video(ground_truth, "COSMOS3_GT")
    pred = _rgb_video(rollout, "COSMOS3_ROLLOUT")
    expected_frames = horizon_frames + 1
    if gt.shape[0] != expected_frames or pred.shape[0] != expected_frames:
        raise Cosmos3PairedGroundTruthError(
            f"COSMOS3_FRAME_COUNT_MISMATCH:expected={expected_frames}:gt={gt.shape[0]}:rollout={pred.shape[0]}"
        )
    if pred.shape[1] > gt.shape[1] or pred.shape[2] > gt.shape[2]:
        raise Cosmos3PairedGroundTruthError(
            f"COSMOS3_SPATIAL_SHAPE_INVALID:gt={list(gt.shape)}:rollout={list(pred.shape)}"
        )

    aligned_gt = gt[:, : pred.shape[1], : pred.shape[2], :]
    gt_unit = aligned_gt.astype(np.float64) / 255.0
    pred_unit = pred.astype(np.float64) / 255.0
    conditioning_mae = float(np.mean(np.abs(gt_unit[0] - pred_unit[0])))
    if conditioning_mae > max_conditioning_frame_mae:
        raise Cosmos3PairedGroundTruthError(
            "COSMOS3_CONDITIONING_FRAME_MISALIGNED:"
            f"mae={conditioning_mae:.8f}:limit={max_conditioning_frame_mae:.8f}"
        )

    gt_future = gt_unit[1:]
    pred_future = pred_unit[1:]
    error = pred_future - gt_future
    mse = float(np.mean(np.square(error)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    metrics = {
        "rollout_video_psnr": psnr,
        "rollout_video_l1": float(np.mean(np.abs(error))),
        "final_frame_mae": float(np.mean(np.abs(error[-1]))),
        "temporal_difference_mae": float(
            np.mean(np.abs(np.diff(pred_unit, axis=0) - np.diff(gt_unit, axis=0)))
        ),
    }
    alignment: dict[str, object] = {
        "ground_truth_shape": list(gt.shape),
        "rollout_shape": list(pred.shape),
        "condition_frame_index": 0,
        "future_start_index": 1,
        "future_frame_count": horizon_frames,
        "spatial_policy": "top_left_content_crop_to_rollout",
        "conditioning_frame_mae": conditioning_mae,
        "max_conditioning_frame_mae": max_conditioning_frame_mae,
    }
    return metrics, alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_video(value: np.ndarray, code: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise Cosmos3PairedGroundTruthError(f"{code}_SHAPE_INVALID:{list(array.shape)}")
    if array.dtype != np.uint8:
        raise Cosmos3PairedGroundTruthError(f"{code}_DTYPE_INVALID:{array.dtype}")
    if any(size <= 0 for size in array.shape):
        raise Cosmos3PairedGroundTruthError(f"{code}_EMPTY")
    return array
