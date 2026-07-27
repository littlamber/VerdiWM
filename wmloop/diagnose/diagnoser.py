"""Pure aggregation and attribution logic for the four M1 probes.

GPU/video execution belongs to provider-specific probe adapters.  This module
consumes their measured outputs and makes the dominant-failure classification
deterministic, configured and contract-checked before an LLM sees the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document


@dataclass(frozen=True)
class DiagnosisThresholds:
    appearance_low_motion_ssim_min: float = 0.65
    action_following_accuracy_min: float = 0.60
    no_action_delta_psnr_min: float = 0.50
    ood_auc_gap_min: float = 100.0
    negative_drift_slope_min: float = 0.20
    mixed_relative_margin: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"DIAGNOSIS_THRESHOLD_INVALID:{name}")


def summarize_horizon_curve(*, metric: str, observations: Mapping[int, float]) -> dict[str, object]:
    """Compute a trapezoid AUC and least-squares error drift over horizons."""

    if not metric or len(observations) < 2:
        raise ValueError("HORIZON_CURVE_INSUFFICIENT")
    points = sorted((int(horizon), float(value)) for horizon, value in observations.items())
    if any(horizon <= 0 for horizon, _ in points) or len({horizon for horizon, _ in points}) != len(points):
        raise ValueError("HORIZON_CURVE_HORIZON_INVALID")
    if any(not math.isfinite(value) for _, value in points):
        raise ValueError("HORIZON_CURVE_VALUE_INVALID")
    auc = sum((right_horizon - left_horizon) * (left_value + right_value) / 2.0 for (left_horizon, left_value), (right_horizon, right_value) in zip(points, points[1:]))
    mean_horizon = sum(horizon for horizon, _ in points) / len(points)
    mean_value = sum(value for _, value in points) / len(points)
    denominator = sum((horizon - mean_horizon) ** 2 for horizon, _ in points)
    slope = sum((horizon - mean_horizon) * (value - mean_value) for horizon, value in points) / denominator
    first_horizon, last_horizon = points[0][0], points[-1][0]
    return {
        metric: {str(horizon): value for horizon, value in points},
        f"auc_{metric}_{first_horizon}_{last_horizon}": auc,
        f"auc_{metric}_envmax": auc,
        "drift_slope": slope,
    }


def summarize_segment_drift(*, observations: Mapping[int, float], window: int = 12) -> dict[str, object]:
    """Report the worst measured within-window decline without smoothing it away.

    Long-horizon averages can hide a middle-of-rollout collapse.  The input is
    intentionally per-frame probe output, not interpolated horizon values: an
    unavailable per-frame measurement must block the report rather than being
    fabricated from a four-point curve.
    """

    if window < 2 or len(observations) < window:
        raise ValueError("SEGMENT_DRIFT_INSUFFICIENT")
    points = sorted((int(frame), float(value)) for frame, value in observations.items())
    if any(frame < 0 for frame, _ in points) or len({frame for frame, _ in points}) != len(points):
        raise ValueError("SEGMENT_DRIFT_FRAME_INVALID")
    if any(not math.isfinite(value) for _, value in points):
        raise ValueError("SEGMENT_DRIFT_VALUE_INVALID")
    windows = [
        (points[index][0], points[index][1] - points[index + window - 1][1])
        for index in range(len(points) - window + 1)
    ]
    onset_frame, worst_drop = max(windows, key=lambda item: (item[1], -item[0]))
    return {"window": window, "worst_drop": worst_drop, "onset_frame": onset_frame}


def build_failure_report(
    *,
    env: str,
    model_ref: str,
    round_number: int,
    goal_id: str,
    psnr_by_horizon: Mapping[int, float],
    per_frame_psnr: Mapping[int, float],
    appearance_low_motion_ssim: float,
    action_following_accuracy: float,
    no_action_delta_psnr: float,
    ind_auc: float,
    ood_auc: float,
    worst_ood_condition: str,
    evidence_frames: Sequence[str],
    thresholds: DiagnosisThresholds,
    inverse_dynamics_r2: float | None = None,
) -> dict[str, object]:
    """Build a schema-valid report from measured probe outputs only."""

    values = (
        appearance_low_motion_ssim,
        action_following_accuracy,
        no_action_delta_psnr,
        ind_auc,
        ood_auc,
    )
    if (
        not env
        or not goal_id
        or not worst_ood_condition
        or round_number < 0
        or any(not math.isfinite(value) for value in values)
        or (inverse_dynamics_r2 is not None and not math.isfinite(inverse_dynamics_r2))
        or any(not isinstance(uri, str) or not uri.startswith("cas://sha256/") for uri in evidence_frames)
    ):
        raise ValueError("FAILURE_REPORT_INPUT_INVALID")
    horizon_curve = summarize_horizon_curve(metric="psnr", observations=psnr_by_horizon)
    horizon_curve["segment_drift"] = summarize_segment_drift(observations=per_frame_psnr)
    ind_minus_ood = ind_auc - ood_auc
    candidates = _rank_failures(
        drift_slope=float(horizon_curve["drift_slope"]),
        appearance_low_motion_ssim=appearance_low_motion_ssim,
        action_following_accuracy=action_following_accuracy,
        no_action_delta_psnr=no_action_delta_psnr,
        ood_gap=ind_minus_ood,
        thresholds=thresholds,
    )
    dominant = _dominant_failure(candidates, thresholds)
    report: dict[str, object] = {
        "env": env,
        "model_ref": model_ref,
        "round": round_number,
        "goal_id": goal_id,
        "horizon_curve": horizon_curve,
        "appearance_drift": {"low_motion_ssim_64": appearance_low_motion_ssim},
        "action_following": {
            "inv_dyn_acc_perframe": action_following_accuracy,
            "no_action_delta_psnr": no_action_delta_psnr,
            "inverse_dynamics_r2": inverse_dynamics_r2,
            "low_confidence": inverse_dynamics_r2 is not None and inverse_dynamics_r2 < 0.5,
        },
        "ood_profile": {
            "ind_auc": ind_auc,
            "ood_auc": ood_auc,
            "gap": ind_minus_ood,
            "worst_ood_condition": worst_ood_condition,
        },
        "dominant_failure": dominant,
        "dominant_failure_candidates": [name for name, _ in candidates] or ["mixed"],
        "evidence_frames": list(evidence_frames),
    }
    try:
        validate_document("failure_report", report)
    except ContractValidationError as exc:
        raise ValueError(f"FAILURE_REPORT_CONTRACT_INVALID:{exc}") from exc
    return report


def _rank_failures(
    *,
    drift_slope: float,
    appearance_low_motion_ssim: float,
    action_following_accuracy: float,
    no_action_delta_psnr: float,
    ood_gap: float,
    thresholds: DiagnosisThresholds,
) -> list[tuple[str, float]]:
    scores = {
        "ood_physics": max(0.0, (ood_gap - thresholds.ood_auc_gap_min) / thresholds.ood_auc_gap_min),
        "action_binding": max(
            0.0,
            (thresholds.action_following_accuracy_min - action_following_accuracy) / thresholds.action_following_accuracy_min,
            (thresholds.no_action_delta_psnr_min - no_action_delta_psnr) / thresholds.no_action_delta_psnr_min,
        ),
        "appearance_drift": max(
            0.0,
            (thresholds.appearance_low_motion_ssim_min - appearance_low_motion_ssim)
            / thresholds.appearance_low_motion_ssim_min,
        ),
        "train_infer_mismatch": max(
            0.0,
            (-drift_slope - thresholds.negative_drift_slope_min) / thresholds.negative_drift_slope_min,
        ),
    }
    precedence = {"ood_physics": 0, "action_binding": 1, "appearance_drift": 2, "train_infer_mismatch": 3}
    return [(name, score) for name, score in sorted(scores.items(), key=lambda item: (-item[1], precedence[item[0]])) if score > 0]


def _dominant_failure(candidates: list[tuple[str, float]], thresholds: DiagnosisThresholds) -> str:
    if not candidates:
        return "mixed"
    if len(candidates) == 1:
        return candidates[0][0]
    top_name, top_score = candidates[0]
    _, second_score = candidates[1]
    if top_score > 0 and (top_score - second_score) / top_score <= thresholds.mixed_relative_margin:
        return "mixed"
    return top_name
