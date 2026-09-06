from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wmloop.verify.wan22_paired_effect import (
    compute_paired_video_metrics,
    verify_paired_effect,
)


METRIC_IDS = (
    "action_following",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "photometric_smoothness",
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _arm(*, baseline: bool) -> dict[str, object]:
    return {
        "conditioning_mode": "visual_anchor_only" if baseline else "action_proprio_ema",
        "history_decay": 0.8,
        "anchor_policy": "previous_generated" if baseline else "initial_reference_blend",
        "anchor_refresh_strength": 0.0 if baseline else 0.25,
        "branch_count": 2,
        "branch_selection": "first" if baseline else "terminal_reference_consistency",
    }


def _receipt(root: Path, *, baseline: bool) -> Path:
    run_root = root / "seed-7"
    panel_root = run_root / "validation-panel" / "sample-0"
    panel_root.mkdir(parents=True)
    (panel_root / "generated_150f.mp4").write_bytes(b"generated")
    (panel_root / "ground_truth_150f.mp4").write_bytes(b"same-ground-truth")
    _write(
        run_root / "validation_panel.json",
        {
            "rows": [
                {
                    "sample_index": 0,
                    "sample_id": "episode-a:0",
                    "episode_id": "episode-a",
                    "run_root": str(panel_root),
                }
            ]
        },
    )
    lint = _write(root / "artifact_lint.json", {"state": "pass", "error_count": 0})
    metrics = {
        metric_id: {"aggregate_reported": 0.5 if baseline else 0.51}
        for metric_id in METRIC_IDS
    }
    return _write(
        root / "closed_loop_receipt.json",
        {
            "artifact_type": "verdiwm-wan22-droid-closed-loop-receipt",
            "state": "completed",
            "started_unix_seconds": 1_800_000_000.0,
            "candidate": _arm(baseline=baseline),
            "artifact_lint": str(lint),
            "training_mode": "long",
            "training_sampler": "episode_balanced",
            "train_record_limit": 32,
            "steps": 512,
            "seeds": [7],
            "validation_sample_indices": [0],
            "worldarena_dimensions": list(METRIC_IDS),
            "worldarena_metrics_omitted": ["trajectory_accuracy"],
            "runs": [
                {
                    "seed": 7,
                    "state": "evaluated",
                    "run_root": str(run_root),
                    "validation_panel_metrics": [
                        {"sample_index": 0, "metrics": metrics}
                    ],
                }
            ],
        },
    )


def _policy(root: Path, *, minimum_delta: float = 0.1) -> Path:
    return _write(
        root / "effect-policy.json",
        {
            "artifact_type": "verdiwm-wan22-droid-paired-effect-policy",
            "policy_id": "test-policy",
            "created_at": "2026-01-01T00:00:00Z",
            "frozen_before_results": True,
            "baseline_arm": _arm(baseline=True),
            "candidate_arm": _arm(baseline=False),
            "paired_contract": {
                "seeds": [7],
                "validation_sample_indices": [0],
                "required_pair_count": 1,
                "horizon_frames": 4,
                "tail_frames": 2,
                "primary_metric": {
                    "id": "tail_2_frame_psnr",
                    "direction": "higher",
                    "minimum_mean_delta": minimum_delta,
                    "minimum_pair_wins": 1,
                },
                "protected_metrics": [
                    {
                        "id": "full_future_psnr",
                        "direction": "higher",
                        "maximum_mean_regression": 0.0,
                    },
                    {
                        "id": "final_frame_mae",
                        "direction": "lower",
                        "maximum_mean_regression": 0.0,
                    },
                    {
                        "id": "temporal_difference_mae",
                        "direction": "lower",
                        "maximum_mean_regression": 0.0,
                    },
                    *[
                        {
                            "id": metric_id,
                            "direction": "higher",
                            "maximum_mean_regression": 0.0,
                        }
                        for metric_id in METRIC_IDS
                    ],
                ],
            },
        },
    )


def _video_loader(path: Path) -> np.ndarray:
    gt = np.full((4, 2, 2, 3), 128, dtype=np.uint8)
    if path.name == "ground_truth_150f.mp4":
        return gt
    value = 64 if "baseline" in path.parts else 120
    rollout = np.full_like(gt, value)
    rollout[0] = gt[0]
    return rollout


def test_compute_paired_video_metrics_excludes_conditioning_frame() -> None:
    gt = np.full((4, 2, 2, 3), 128, dtype=np.uint8)
    rollout = gt.copy()
    rollout[1:] = 120
    metrics = compute_paired_video_metrics(
        ground_truth=gt, rollout=rollout, horizon_frames=4, tail_frames=2
    )
    assert metrics["conditioning_frame_mae"] == 0.0
    assert metrics["tail_2_frame_psnr"] == pytest.approx(
        metrics["full_future_psnr"]
    )
    assert metrics["final_frame_mae"] == pytest.approx(8.0 / 255.0)


def test_compute_paired_video_metrics_rejects_misaligned_conditioning() -> None:
    gt = np.full((4, 2, 2, 3), 128, dtype=np.uint8)
    rollout = gt.copy()
    rollout[0] = 0
    with pytest.raises(ValueError, match="CONDITIONING_FRAME_MISALIGNED"):
        compute_paired_video_metrics(
            ground_truth=gt, rollout=rollout, horizon_frames=4, tail_frames=2
        )


def test_frozen_paired_effect_passes_only_real_paired_improvement(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    baseline = _receipt(tmp_path / "baseline", baseline=True)
    candidate = _receipt(tmp_path / "candidate", baseline=False)
    output = tmp_path / "verdict.json"

    verdict = verify_paired_effect(
        policy_path=policy,
        baseline_receipt_path=baseline,
        candidate_receipt_path=candidate,
        output_path=output,
        video_loader=_video_loader,
    )

    assert verdict["state"] == "pass"
    assert verdict["effect_established"] is True
    assert verdict["pair_count"] == 1
    assert output.is_file()


def test_frozen_paired_effect_records_failed_threshold_without_claim(tmp_path: Path) -> None:
    policy = _policy(tmp_path, minimum_delta=100.0)
    baseline = _receipt(tmp_path / "baseline", baseline=True)
    candidate = _receipt(tmp_path / "candidate", baseline=False)

    verdict = verify_paired_effect(
        policy_path=policy,
        baseline_receipt_path=baseline,
        candidate_receipt_path=candidate,
        video_loader=_video_loader,
    )

    assert verdict["state"] == "fail"
    assert verdict["effect_established"] is False
    assert any("PRIMARY_MEAN_DELTA" in item for item in verdict["blockers"])
