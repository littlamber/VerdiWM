from pathlib import Path

import pytest

from verdi_core.evidence import MetricSpec, classify_paired_effect, compare_metrics, verify_artifacts
from verdi_core.training import AdaptiveTrainingController, TrainingPolicy


def test_metric_direction_normalizes_minimize() -> None:
    result = compare_metrics({"mae": 0.5}, {"mae": 0.4}, [MetricSpec("mae", "minimize")])
    assert result["deltas"]["mae"] == pytest.approx(0.1)


def test_paired_effect_marks_degradation_harmful() -> None:
    result = classify_paired_effect([-0.2, -0.25, -0.21], practical_threshold=0.05, protected_ok=True, bootstrap_samples=1000)
    assert result["outcome"] == "harmful"


def test_artifact_integrity(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    from verdi_core.evidence import sha256_file
    assert verify_artifacts([{"path": str(path), "sha256": sha256_file(path)}])["state"] == "complete"
    assert verify_artifacts([{"path": str(path), "sha256": "sha256:bad"}])["state"] == "invalid"


def test_adaptive_training_stops_near_overfit() -> None:
    controller = AdaptiveTrainingController(TrainingPolicy(direction="maximize", overfit_evaluations=2, patience_evaluations=5))
    assert controller.observe(step=100, train_metric=1.0, heldout_metric=0.5)["action"] == "continue"
    assert controller.observe(step=200, train_metric=1.1, heldout_metric=0.49)["action"] == "continue"
    decision = controller.observe(step=300, train_metric=1.2, heldout_metric=0.48)
    assert decision["action"] == "stop"
    assert decision["reason"] == "near_overfit"


def test_early_positive_probe_promotes_long_training() -> None:
    controller = AdaptiveTrainingController(TrainingPolicy(direction="minimize", evaluation_interval=300))
    decision = controller.promote_after_probe(step=3000, baseline_metric=0.5, candidate_metric=0.4, practical_threshold=0.05)
    assert decision["action"] == "continue_long_train"
    assert decision["next_step"] == 3300
