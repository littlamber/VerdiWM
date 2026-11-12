"""Training and frozen evaluation smoke components: metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID"
        )
    return value


def _environment_spec(environment: str) -> Any:
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise OrchestratorTrainingEvalSmokeError(
        f"M3_TRAINING_EVAL_SMOKE_ENVIRONMENT_UNKNOWN:{environment}"
    )


def _goal_action_following_threshold(goal: Mapping[str, object]) -> float | None:
    raw = goal.get("action_following_gate")
    if not isinstance(raw, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_GOAL_AF_GATE_INVALID"
        )
    tau = _finite_float(
        raw.get("tau_af"), "M3_TRAINING_EVAL_SMOKE_GOAL_AF_GATE_INVALID"
    )
    return None if tau < 0.0 else tau


def _goal_id(goal: Mapping[str, object]) -> str:
    value = goal.get("goal_id")
    if not isinstance(value, str) or not value:
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GOAL_INVALID")
    return value


def _verify_goal_environment(goal: Mapping[str, object], environment: str) -> None:
    raw = goal.get("envs")
    if raw is None:
        return
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_GOAL_INVALID")
    if environment not in raw:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FAILURE_SCOPE_INVALID"
        )


def _goal_metric_name(*, goal: Mapping[str, object], horizons: Sequence[int]) -> str:
    primary = goal.get("primary_objective")
    if isinstance(primary, str) and primary:
        return primary
    return _auc_metric_name(horizons)


def _auc_metric_name(horizons: Sequence[int]) -> str:
    points = sorted(int(value) for value in horizons)
    if len(points) < 2:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    return f"auc_psnr_{points[0]}_{points[-1]}"


def _max_observed_horizon(failure: Mapping[str, object]) -> int:
    curve = failure.get("horizon_curve")
    if not isinstance(curve, Mapping):
        return 64
    psnr = curve.get("psnr")
    if not isinstance(psnr, Mapping):
        return 64
    horizons = []
    for key in psnr:
        try:
            horizons.append(int(key))
        except (TypeError, ValueError):
            continue
    return max(horizons) if horizons else 64


def _vendor_environment_name(environment: str) -> str:
    return "clothmove" if environment == "cloth_move" else environment


def _auc_value(
    manifest: Mapping[str, object], *, horizons: Sequence[int] | None = None
) -> float:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_PROBE_MANIFEST_INVALID"
        )
    curve = aggregate.get("horizon_curve")
    if not isinstance(curve, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_PROBE_MANIFEST_INVALID"
        )
    candidate_keys: list[str] = []
    if horizons is not None:
        candidate_keys.append(_auc_metric_name(horizons))
    candidate_keys.extend(["auc_psnr_16_64", "auc_psnr_envmax"])
    value = None
    for key in candidate_keys:
        if key in curve:
            value = curve.get(key)
            break
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise OrchestratorTrainingEvalSmokeError("M3_TRAINING_EVAL_SMOKE_AUC_MISSING")
    return float(value)


def _replication_value(
    replication: Mapping[str, object], *, prefix: str, metric_name: str
) -> float:
    for key in (
        f"{prefix}_{metric_name}",
        f"{prefix}_primary_metric",
        f"{prefix}_auc_psnr_16_64",
    ):
        value = replication.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError(
        "M3_TRAINING_EVAL_SMOKE_REPLICATION_METRIC_MISSING"
    )


def _replication_delta(replication: Mapping[str, object], *, metric_name: str) -> float:
    for key in (f"delta_{metric_name}", "delta_primary_metric", "delta_auc_psnr_16_64"):
        value = replication.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError(
        "M3_TRAINING_EVAL_SMOKE_REPLICATION_METRIC_MISSING"
    )


def _finite_float(value: object, error_code: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise OrchestratorTrainingEvalSmokeError(error_code)
    return float(value)


def _has_required_horizons(
    manifest: Mapping[str, object], required_horizons: Sequence[int]
) -> bool:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return False
    metrics = aggregate.get("horizon_metrics")
    if not isinstance(metrics, Mapping):
        return False
    return all(str(horizon) in metrics for horizon in required_horizons)


def _receipt_gpu_hours(candidate: Mapping[str, object]) -> float:
    receipts = candidate.get("receipts")
    if not isinstance(receipts, list):
        return 0.0
    seconds = 0.0
    for receipt in receipts:
        if isinstance(receipt, Mapping):
            duration = receipt.get("duration_seconds")
            if (
                isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and math.isfinite(float(duration))
            ):
                seconds += float(duration)
    return seconds / 3600.0
