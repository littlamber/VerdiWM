"""Frozen paired-effect verification for WAN2.2-DROID closed-loop runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import imageio.v3 as iio
import numpy as np


class Wan22PairedEffectError(ValueError):
    """The paired evidence cannot support a trustworthy effect decision."""


VideoLoader = Callable[[Path], np.ndarray]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_paired_video_metrics(
    *,
    ground_truth: np.ndarray,
    rollout: np.ndarray,
    horizon_frames: int,
    tail_frames: int,
) -> dict[str, float]:
    """Compute aligned full-horizon and tail metrics in RGB unit space."""

    gt = _rgb_video(ground_truth, "WAN22_GT")
    pred = _rgb_video(rollout, "WAN22_ROLLOUT")
    expected_shape = (horizon_frames, *gt.shape[1:])
    if gt.shape != expected_shape or pred.shape != expected_shape:
        raise Wan22PairedEffectError(
            "WAN22_PAIRED_VIDEO_SHAPE_MISMATCH:"
            f"expected={list(expected_shape)}:gt={list(gt.shape)}:rollout={list(pred.shape)}"
        )
    if tail_frames < 1 or tail_frames > horizon_frames:
        raise Wan22PairedEffectError("WAN22_PAIRED_TAIL_FRAMES_INVALID")

    gt_unit = gt.astype(np.float64) / 255.0
    pred_unit = pred.astype(np.float64) / 255.0
    # Frame zero is the shared visual condition. It is checked for alignment
    # but excluded from the predictive full-horizon metric.
    conditioning_mae = float(np.mean(np.abs(gt_unit[0] - pred_unit[0])))
    if conditioning_mae > 8.0 / 255.0:
        raise Wan22PairedEffectError(
            "WAN22_CONDITIONING_FRAME_MISALIGNED:"
            f"mae={conditioning_mae}:limit={8.0 / 255.0}"
        )
    future_gt = gt_unit[1:]
    future_pred = pred_unit[1:]
    full_mse = float(np.mean(np.square(future_pred - future_gt)))
    tail_mse = float(
        np.mean(np.square(pred_unit[-tail_frames:] - gt_unit[-tail_frames:]))
    )
    return {
        "conditioning_frame_mae": conditioning_mae,
        "full_future_psnr": _psnr(full_mse),
        f"tail_{tail_frames}_frame_psnr": _psnr(tail_mse),
        "final_frame_mae": float(np.mean(np.abs(pred_unit[-1] - gt_unit[-1]))),
        "temporal_difference_mae": float(
            np.mean(np.abs(np.diff(pred_unit, axis=0) - np.diff(gt_unit, axis=0)))
        ),
    }


def verify_paired_effect(
    *,
    policy_path: Path,
    baseline_receipt_path: Path,
    candidate_receipt_path: Path,
    output_path: Path | None = None,
    video_loader: VideoLoader | None = None,
) -> dict[str, object]:
    """Evaluate two terminal closed-loop receipts against a pre-frozen policy."""

    policy_path = _regular_file(policy_path, "WAN22_EFFECT_POLICY_INVALID")
    baseline_receipt_path = _regular_file(
        baseline_receipt_path, "WAN22_BASELINE_RECEIPT_INVALID"
    )
    candidate_receipt_path = _regular_file(
        candidate_receipt_path, "WAN22_CANDIDATE_RECEIPT_INVALID"
    )
    policy = _load_mapping(policy_path, "WAN22_EFFECT_POLICY_INVALID")
    baseline = _load_mapping(
        baseline_receipt_path, "WAN22_BASELINE_RECEIPT_INVALID"
    )
    candidate = _load_mapping(
        candidate_receipt_path, "WAN22_CANDIDATE_RECEIPT_INVALID"
    )
    contract = _validate_policy(policy)
    _validate_closed_loop_receipt(
        baseline, expected_arm=_mapping(policy, "baseline_arm"), label="BASELINE"
    )
    _validate_closed_loop_receipt(
        candidate, expected_arm=_mapping(policy, "candidate_arm"), label="CANDIDATE"
    )
    _validate_common_contract(baseline, candidate, contract)
    _validate_policy_precedes_runs(policy, baseline, candidate)

    loader = video_loader or (lambda path: iio.imread(path))
    baseline_pairs = _load_pairs(baseline, contract, loader)
    candidate_pairs = _load_pairs(candidate, contract, loader)
    if set(baseline_pairs) != set(candidate_pairs):
        raise Wan22PairedEffectError("WAN22_EFFECT_PAIR_IDENTITY_MISMATCH")

    per_pair: list[dict[str, object]] = []
    metric_ids = _metric_ids(contract)
    for identity in sorted(baseline_pairs):
        baseline_row = baseline_pairs[identity]
        candidate_row = candidate_pairs[identity]
        if baseline_row["ground_truth_sha256"] != candidate_row["ground_truth_sha256"]:
            raise Wan22PairedEffectError(
                f"WAN22_EFFECT_GROUND_TRUTH_MISMATCH:{identity}"
            )
        if (
            baseline_row["sample_id"] != candidate_row["sample_id"]
            or baseline_row["episode_id"] != candidate_row["episode_id"]
        ):
            raise Wan22PairedEffectError(
                f"WAN22_EFFECT_SAMPLE_IDENTITY_MISMATCH:{identity}"
            )
        baseline_metrics = _mapping(baseline_row, "metrics")
        candidate_metrics = _mapping(candidate_row, "metrics")
        deltas = {
            metric_id: _signed_improvement(
                candidate=float(candidate_metrics[metric_id]),
                baseline=float(baseline_metrics[metric_id]),
                direction=_metric_direction(contract, metric_id),
            )
            for metric_id in metric_ids
        }
        per_pair.append(
            {
                "seed": identity[0],
                "sample_index": identity[1],
                "sample_id": baseline_row["sample_id"],
                "episode_id": baseline_row["episode_id"],
                "ground_truth_sha256": baseline_row["ground_truth_sha256"],
                "baseline": dict(baseline_metrics),
                "candidate": dict(candidate_metrics),
                "signed_improvement": deltas,
            }
        )

    required_pair_count = _positive_int(contract, "required_pair_count")
    blockers: list[str] = []
    if len(per_pair) != required_pair_count:
        blockers.append(
            "WAN22_EFFECT_PAIR_COUNT_INVALID:"
            f"expected={required_pair_count}:observed={len(per_pair)}"
        )

    aggregates = _aggregate_metrics(per_pair, metric_ids)
    primary = _mapping(contract, "primary_metric")
    primary_id = _string(primary, "id")
    primary_mean_delta = float(aggregates[primary_id]["mean_signed_improvement"])
    primary_wins = int(aggregates[primary_id]["pair_wins"])
    minimum_mean_delta = _finite_float(primary, "minimum_mean_delta")
    minimum_pair_wins = _positive_int(primary, "minimum_pair_wins")
    if primary_mean_delta < minimum_mean_delta:
        blockers.append(
            "WAN22_EFFECT_PRIMARY_MEAN_DELTA_BELOW_THRESHOLD:"
            f"metric={primary_id}:required={minimum_mean_delta}:observed={primary_mean_delta}"
        )
    if primary_wins < minimum_pair_wins:
        blockers.append(
            "WAN22_EFFECT_PRIMARY_PAIR_WINS_BELOW_THRESHOLD:"
            f"metric={primary_id}:required={minimum_pair_wins}:observed={primary_wins}"
        )

    protected_gates: list[dict[str, object]] = []
    for item in _mapping_sequence(contract, "protected_metrics"):
        metric_id = _string(item, "id")
        observed_regression = max(
            0.0, -float(aggregates[metric_id]["mean_signed_improvement"])
        )
        if "maximum_mean_regression" in item:
            allowed = _nonnegative_float(item, "maximum_mean_regression")
            regression_kind = "absolute"
        else:
            fraction = _nonnegative_float(item, "maximum_relative_mean_regression")
            baseline_mean = abs(float(aggregates[metric_id]["baseline_mean"]))
            allowed = fraction * baseline_mean
            regression_kind = "relative_to_baseline_mean"
        passed = observed_regression <= allowed
        if not passed:
            blockers.append(
                "WAN22_EFFECT_PROTECTED_METRIC_REGRESSION:"
                f"metric={metric_id}:allowed={allowed}:observed={observed_regression}"
            )
        protected_gates.append(
            {
                "metric_id": metric_id,
                "state": "pass" if passed else "fail",
                "regression_kind": regression_kind,
                "allowed_mean_regression": allowed,
                "observed_mean_regression": observed_regression,
            }
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-paired-effect-verdict",
        "state": "pass" if not blockers else "fail",
        "effect_established": not blockers,
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256_file(policy_path),
        "baseline_receipt_sha256": sha256_file(baseline_receipt_path),
        "candidate_receipt_sha256": sha256_file(candidate_receipt_path),
        "verifier_implementation_sha256": sha256_file(Path(__file__)),
        "pair_count": len(per_pair),
        "primary_gate": {
            "metric_id": primary_id,
            "minimum_mean_delta": minimum_mean_delta,
            "observed_mean_delta": primary_mean_delta,
            "minimum_pair_wins": minimum_pair_wins,
            "observed_pair_wins": primary_wins,
        },
        "protected_gates": protected_gates,
        "aggregates": aggregates,
        "pairs": per_pair,
        "blockers": sorted(blockers),
        "claim_boundary": (
            "A pass establishes only the pre-frozen paired effect claim for the "
            "declared WAN2.2-DROID horizon, seeds, episodes, metrics, and arms."
        ),
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _load_pairs(
    receipt: Mapping[str, object],
    contract: Mapping[str, object],
    loader: VideoLoader,
) -> dict[tuple[int, int], dict[str, object]]:
    horizon_frames = _positive_int(contract, "horizon_frames")
    tail_frames = _positive_int(contract, "tail_frames")
    pairs: dict[tuple[int, int], dict[str, object]] = {}
    for run in _mapping_sequence(receipt, "runs"):
        seed = _int(run, "seed")
        run_root = _directory(
            Path(_string(run, "run_root")), "WAN22_RUN_ROOT_INVALID"
        )
        panel = _load_mapping(
            run_root / "validation_panel.json", "WAN22_VALIDATION_PANEL_INVALID"
        )
        metrics_by_index = {
            _int(row, "sample_index"): row
            for row in _mapping_sequence(run, "validation_panel_metrics")
        }
        for panel_row in _mapping_sequence(panel, "rows"):
            sample_index = _int(panel_row, "sample_index")
            identity = (seed, sample_index)
            if identity in pairs:
                raise Wan22PairedEffectError(
                    f"WAN22_EFFECT_DUPLICATE_PAIR:{seed}:{sample_index}"
                )
            panel_root = _directory(
                Path(_string(panel_row, "run_root")), "WAN22_PANEL_ROOT_INVALID"
            )
            generated = _regular_file(
                panel_root / "generated_150f.mp4", "WAN22_GENERATED_VIDEO_INVALID"
            )
            ground_truth = _regular_file(
                panel_root / "ground_truth_150f.mp4", "WAN22_GROUND_TRUTH_VIDEO_INVALID"
            )
            video_metrics = compute_paired_video_metrics(
                ground_truth=loader(ground_truth),
                rollout=loader(generated),
                horizon_frames=horizon_frames,
                tail_frames=tail_frames,
            )
            worldarena_row = metrics_by_index.get(sample_index)
            if worldarena_row is None:
                raise Wan22PairedEffectError(
                    f"WAN22_WORLDARENA_PANEL_METRICS_MISSING:{seed}:{sample_index}"
                )
            worldarena_metrics = _mapping(worldarena_row, "metrics")
            for metric_id in (
                "action_following",
                "subject_consistency",
                "background_consistency",
                "motion_smoothness",
                "photometric_smoothness",
            ):
                metric = _mapping(worldarena_metrics, metric_id)
                video_metrics[metric_id] = _finite_float(metric, "aggregate_reported")
            pairs[identity] = {
                "sample_id": str(panel_row.get("sample_id") or ""),
                "episode_id": str(panel_row.get("episode_id") or ""),
                "ground_truth_sha256": sha256_file(ground_truth),
                "metrics": video_metrics,
            }
    return pairs


def _validate_policy(policy: Mapping[str, object]) -> Mapping[str, object]:
    if policy.get("artifact_type") != "verdiwm-wan22-droid-paired-effect-policy":
        raise Wan22PairedEffectError("WAN22_EFFECT_POLICY_TYPE_INVALID")
    if policy.get("frozen_before_results") is not True:
        raise Wan22PairedEffectError("WAN22_EFFECT_POLICY_NOT_FROZEN")
    _string(policy, "policy_id")
    contract = _mapping(policy, "paired_contract")
    primary = _mapping(contract, "primary_metric")
    _string(primary, "id")
    _direction(primary)
    _finite_float(primary, "minimum_mean_delta")
    _positive_int(primary, "minimum_pair_wins")
    protected = _mapping_sequence(contract, "protected_metrics")
    if not protected:
        raise Wan22PairedEffectError("WAN22_EFFECT_PROTECTED_METRICS_EMPTY")
    for item in protected:
        _string(item, "id")
        _direction(item)
        has_absolute = "maximum_mean_regression" in item
        has_relative = "maximum_relative_mean_regression" in item
        if has_absolute == has_relative:
            raise Wan22PairedEffectError("WAN22_EFFECT_REGRESSION_LIMIT_INVALID")
    return contract


def _validate_closed_loop_receipt(
    receipt: Mapping[str, object], *, expected_arm: Mapping[str, object], label: str
) -> None:
    if receipt.get("artifact_type") != "verdiwm-wan22-droid-closed-loop-receipt":
        raise Wan22PairedEffectError(f"WAN22_{label}_RECEIPT_TYPE_INVALID")
    if receipt.get("state") != "completed":
        raise Wan22PairedEffectError(f"WAN22_{label}_RUN_NOT_COMPLETED")
    observed_arm = _mapping(receipt, "candidate")
    # Receipts may contain additional reproducibility metadata (for example
    # branch-reference weight or rollout horizon) added after the policy was
    # frozen.  The frozen policy remains authoritative for the fields it
    # declares; unknown receipt keys must not invalidate an otherwise paired
    # run.
    if any(observed_arm.get(key) != value for key, value in expected_arm.items()):
        raise Wan22PairedEffectError(f"WAN22_{label}_ARM_MISMATCH")
    lint_path = _regular_file(
        Path(_string(receipt, "artifact_lint")), f"WAN22_{label}_ARTIFACT_LINT_INVALID"
    )
    lint = _load_mapping(lint_path, f"WAN22_{label}_ARTIFACT_LINT_INVALID")
    if lint.get("state") != "pass" or int(lint.get("error_count", -1)) != 0:
        raise Wan22PairedEffectError(f"WAN22_{label}_ARTIFACT_LINT_FAILED")


def _validate_common_contract(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    keys = (
        "training_mode",
        "training_sampler",
        "train_record_limit",
        "steps",
        "seeds",
        "validation_sample_indices",
        "worldarena_dimensions",
        "worldarena_metrics_omitted",
    )
    if any(baseline.get(key) != candidate.get(key) for key in keys):
        raise Wan22PairedEffectError("WAN22_EFFECT_COMMON_CONTRACT_MISMATCH")
    if baseline.get("seeds") != contract.get("seeds"):
        raise Wan22PairedEffectError("WAN22_EFFECT_POLICY_SEEDS_MISMATCH")
    if baseline.get("validation_sample_indices") != contract.get(
        "validation_sample_indices"
    ):
        raise Wan22PairedEffectError("WAN22_EFFECT_POLICY_PANEL_MISMATCH")


def _validate_policy_precedes_runs(
    policy: Mapping[str, object],
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    created_at = _parse_utc(_string(policy, "created_at"))
    starts = [
        dt.datetime.fromtimestamp(
            _finite_float(receipt, "started_unix_seconds"), tz=dt.timezone.utc
        )
        for receipt in (baseline, candidate)
    ]
    if created_at > min(starts):
        raise Wan22PairedEffectError("WAN22_EFFECT_POLICY_FROZEN_AFTER_RUN_START")


def _aggregate_metrics(
    rows: Sequence[Mapping[str, object]], metric_ids: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    if not rows:
        raise Wan22PairedEffectError("WAN22_EFFECT_PAIRS_EMPTY")
    result: dict[str, dict[str, float | int]] = {}
    for metric_id in metric_ids:
        baseline_values = [float(_mapping(row, "baseline")[metric_id]) for row in rows]
        candidate_values = [float(_mapping(row, "candidate")[metric_id]) for row in rows]
        deltas = [
            float(_mapping(row, "signed_improvement")[metric_id]) for row in rows
        ]
        result[metric_id] = {
            "baseline_mean": mean(baseline_values),
            "candidate_mean": mean(candidate_values),
            "mean_signed_improvement": mean(deltas),
            "minimum_signed_improvement": min(deltas),
            "maximum_signed_improvement": max(deltas),
            "pair_wins": sum(delta > 0.0 for delta in deltas),
        }
    return result


def _metric_ids(contract: Mapping[str, object]) -> tuple[str, ...]:
    primary_id = _string(_mapping(contract, "primary_metric"), "id")
    protected = tuple(
        _string(item, "id") for item in _mapping_sequence(contract, "protected_metrics")
    )
    if primary_id in protected or len(set(protected)) != len(protected):
        raise Wan22PairedEffectError("WAN22_EFFECT_METRIC_IDS_DUPLICATED")
    return (primary_id, *protected)


def _metric_direction(contract: Mapping[str, object], metric_id: str) -> str:
    primary = _mapping(contract, "primary_metric")
    if primary.get("id") == metric_id:
        return _direction(primary)
    for item in _mapping_sequence(contract, "protected_metrics"):
        if item.get("id") == metric_id:
            return _direction(item)
    raise Wan22PairedEffectError(f"WAN22_EFFECT_METRIC_UNDECLARED:{metric_id}")


def _signed_improvement(*, candidate: float, baseline: float, direction: str) -> float:
    return candidate - baseline if direction == "higher" else baseline - candidate


def _psnr(mse: float) -> float:
    if mse < 0.0 or not math.isfinite(mse):
        raise Wan22PairedEffectError("WAN22_EFFECT_MSE_INVALID")
    return 120.0 if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def _rgb_video(value: np.ndarray, code: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise Wan22PairedEffectError(
            f"{code}_INVALID:shape={list(array.shape)}:dtype={array.dtype}"
        )
    return array


def _load_mapping(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Wan22PairedEffectError(f"{code}:{path}") from exc
    if not isinstance(value, dict):
        raise Wan22PairedEffectError(f"{code}:{path}")
    return value


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Wan22PairedEffectError(f"{code}:{candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Wan22PairedEffectError(f"{code}:{candidate}") from exc
    if not resolved.is_file():
        raise Wan22PairedEffectError(f"{code}:{resolved}")
    return resolved


def _directory(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Wan22PairedEffectError(f"{code}:{candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Wan22PairedEffectError(f"{code}:{candidate}") from exc
    if not resolved.is_dir():
        raise Wan22PairedEffectError(f"{code}:{resolved}")
    return resolved


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise Wan22PairedEffectError(f"WAN22_EFFECT_MAPPING_REQUIRED:{key}")
    return item


def _mapping_sequence(
    value: Mapping[str, object], key: str
) -> tuple[Mapping[str, object], ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, Mapping) for row in item):
        raise Wan22PairedEffectError(f"WAN22_EFFECT_MAPPING_LIST_REQUIRED:{key}")
    return tuple(item)


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise Wan22PairedEffectError(f"WAN22_EFFECT_STRING_REQUIRED:{key}")
    return item


def _int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise Wan22PairedEffectError(f"WAN22_EFFECT_INTEGER_REQUIRED:{key}")
    return item


def _positive_int(value: Mapping[str, object], key: str) -> int:
    item = _int(value, key)
    if item < 1:
        raise Wan22PairedEffectError(f"WAN22_EFFECT_POSITIVE_INTEGER_REQUIRED:{key}")
    return item


def _finite_float(value: Mapping[str, object], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise Wan22PairedEffectError(f"WAN22_EFFECT_FLOAT_REQUIRED:{key}")
    number = float(item)
    if not math.isfinite(number):
        raise Wan22PairedEffectError(f"WAN22_EFFECT_FINITE_FLOAT_REQUIRED:{key}")
    return number


def _nonnegative_float(value: Mapping[str, object], key: str) -> float:
    item = _finite_float(value, key)
    if item < 0.0:
        raise Wan22PairedEffectError(f"WAN22_EFFECT_NONNEGATIVE_FLOAT_REQUIRED:{key}")
    return item


def _direction(value: Mapping[str, object]) -> str:
    direction = _string(value, "direction")
    if direction not in {"higher", "lower"}:
        raise Wan22PairedEffectError("WAN22_EFFECT_DIRECTION_INVALID")
    return direction


def _parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Wan22PairedEffectError("WAN22_EFFECT_CREATED_AT_INVALID") from exc
    if parsed.tzinfo is None:
        raise Wan22PairedEffectError("WAN22_EFFECT_CREATED_AT_TIMEZONE_REQUIRED")
    return parsed.astimezone(dt.timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_paired_effect(
            policy_path=args.policy,
            baseline_receipt_path=args.baseline_receipt,
            candidate_receipt_path=args.candidate_receipt,
            output_path=args.output,
        )
    except Wan22PairedEffectError as exc:
        result = {
            "schema_version": 1,
            "artifact_type": "verdiwm-wan22-droid-paired-effect-verdict",
            "state": "blocked",
            "effect_established": False,
            "blockers": [str(exc)],
            "claim_boundary": "Invalid or incomplete evidence cannot establish an effect.",
        }
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
