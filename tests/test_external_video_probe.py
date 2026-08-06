from __future__ import annotations

import numpy as np

from wmloop.diagnose.external_video_probe import _diagnose_frames


def test_video_probe_emits_paired_error_and_short_horizon_signatures() -> None:
    frames = np.zeros((3, 8, 4, 3), dtype=np.uint8)
    frames[:, :4] = 0
    frames[:, 4:] = 255
    metrics, signatures = _diagnose_frames(
        frames,
        layout="vertical",
        min_frames=5,
        max_pair_l1=10.0,
        min_pred_temporal_change=1.0,
        max_horizon_drift_slope=100.0,
        fallback_signature="no_failure",
    )

    assert metrics["paired_l1_mean"] == 255.0
    assert "paired_rollout_error_high" in signatures
    assert "short_horizon_observed" in signatures
    assert "predicted_temporal_collapse" in signatures


def test_video_probe_preserves_horizontal_pairing_and_fallback() -> None:
    frames = np.zeros((4, 4, 8, 3), dtype=np.uint8)
    frames[:, :, :4] = 32
    frames[:, :, 4:] = 32
    metrics, signatures = _diagnose_frames(
        frames,
        layout="horizontal",
        min_frames=2,
        max_pair_l1=1.0,
        min_pred_temporal_change=0.0,
        max_horizon_drift_slope=1.0,
        fallback_signature="paired_video_consistent",
    )

    assert metrics["paired_l1_mean"] == 0.0
    assert signatures == ["paired_video_consistent"]
