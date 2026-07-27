#!/usr/bin/env python3
"""Export paired horizon-response evidence for one ACWM primitive.

This exporter is deliberately claim-conservative. It pairs the same baseline
and candidate trajectories, reports aggregate and per-trajectory effects, and
keeps hard-case improvements separate from causal-map admission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
_METRICS = ("psnr", "ssim", "mse", "masked_mse")
_METRIC_POSITIVE_SCOPES = {
    "aggregate_long_horizon_positive",
    "short_horizon_only_positive",
    "hard_case_only_positive",
}


class AcwmHorizonEffectProfileError(RuntimeError):
    """Paired horizon evidence was incomplete or incompatible."""


def build_horizon_effect_profile(
    *,
    baseline_manifest: Path,
    candidate_manifest: Path,
    primitive: str,
    mechanism_primitive: str | None = None,
    output_root: Path,
    failure_report: Path | None = None,
    mechanism_cards: Path | None = None,
    event_gate: Path | None = None,
    checkpoint_ladder_manifest: Path | None = None,
    fps: float = 10.0,
) -> dict[str, object]:
    """Build an auditable primitive effect profile from paired rollouts."""

    if not primitive.strip():
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_PRIMITIVE_EMPTY")
    resolved_mechanism_primitive = (mechanism_primitive or primitive).strip()
    if not resolved_mechanism_primitive:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MECHANISM_PRIMITIVE_EMPTY")
    if not math.isfinite(fps) or fps <= 0:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_FPS_INVALID")

    baseline_path = Path(baseline_manifest).resolve(strict=True)
    candidate_path = Path(candidate_manifest).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_OUTPUT_EXISTS")

    baseline = _load_manifest(baseline_path)
    candidate = _load_manifest(candidate_path)
    environment, split, horizons = _validate_pair(baseline, candidate)
    baseline_trajectories = _trajectory_map(baseline)
    candidate_trajectories = _trajectory_map(candidate)
    if set(baseline_trajectories) != set(candidate_trajectories):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_TRAJECTORY_SET_MISMATCH")

    horizon_rows = [
        _horizon_row(
            horizon=horizon,
            baseline=_horizon_metrics(baseline, horizon),
            candidate=_horizon_metrics(candidate, horizon),
            fps=fps,
        )
        for horizon in horizons
    ]
    trajectory_rows = [
        _trajectory_row(
            trajectory_index=index,
            baseline=baseline_trajectories[index],
            candidate=candidate_trajectories[index],
            max_horizon=max(horizons),
        )
        for index in sorted(baseline_trajectories)
    ]
    per_frame = _per_frame_profile(baseline, candidate, fps=fps)
    classification = _classify_effect(horizon_rows, trajectory_rows, per_frame)
    event_semantics = _event_semantics(event_gate, environment)
    if event_semantics is not None:
        metric_scope = str(classification["effect_scope"])
        classification["metric_effect_scope"] = metric_scope
        classification["event_semantic_classification"] = event_semantics["classification"]
        classification["candidate_event_pass"] = event_semantics["candidate_event_pass"]
        if metric_scope in _METRIC_POSITIVE_SCOPES and event_semantics["candidate_event_pass"] is not True:
            classification["effect_scope"] = "metric_positive_event_failure"
    failure_signatures = _failure_signatures(failure_report, environment)
    mechanism = _mechanism_card(mechanism_cards, resolved_mechanism_primitive)
    checkpoint_ladder = _checkpoint_ladder_evidence(
        checkpoint_ladder_manifest,
        environment=environment,
        primitive=primitive,
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-horizon-effect-profile",
        "state": "ready",
        "environment": environment,
        "split": split,
        "primitive": primitive,
        "mechanism_primitive": resolved_mechanism_primitive,
        "fps": fps,
        "horizons": horizons,
        "horizon_seconds": {str(horizon): horizon / fps for horizon in horizons},
        "paired_trajectory_count": len(trajectory_rows),
        "source": {
            "baseline_manifest": str(baseline_path),
            "candidate_manifest": str(candidate_path),
            "baseline_checkpoint": baseline.get("checkpoint_path"),
            "candidate_checkpoint": candidate.get("checkpoint_path"),
            "candidate_checkpoint_step": candidate.get("checkpoint_step"),
            "failure_report": str(Path(failure_report).resolve()) if failure_report is not None else None,
            "mechanism_cards": str(Path(mechanism_cards).resolve()) if mechanism_cards is not None else None,
            "mechanism_card_primitive": resolved_mechanism_primitive,
            "event_gate": str(Path(event_gate).resolve()) if event_gate is not None else None,
            "checkpoint_ladder_manifest": (
                str(Path(checkpoint_ladder_manifest).resolve())
                if checkpoint_ladder_manifest is not None
                else None
            ),
        },
        "horizon_effects": horizon_rows,
        "trajectory_effects_at_max_horizon": trajectory_rows,
        "per_frame_effect": per_frame,
        "effect_classification": classification,
        "transfer_prior": {
            "failure_signatures": failure_signatures,
            "primitive": primitive,
            "mechanism_primitive": resolved_mechanism_primitive,
            "intervention_chain": list(
                dict.fromkeys((resolved_mechanism_primitive, primitive))
            ),
            "mechanism_family": mechanism.get("mechanism_family"),
            "layer": mechanism.get("layer"),
            "effect_scope": classification["effect_scope"],
            "effective_horizons": classification["aggregate_passing_horizons"],
            "first_aggregate_failure_horizon": classification["first_aggregate_failure_horizon"],
            "transfer_preconditions": mechanism.get("transfer_preconditions"),
            "anti_conditions": mechanism.get("known_anti_conditions"),
            "event_semantics": event_semantics,
            "checkpoint_ladder": checkpoint_ladder,
            "effective_training_window": (
                checkpoint_ladder.get("effective_training_window")
                if checkpoint_ladder is not None
                else None
            ),
            "evidence_level": "paired_horizon_observation",
            "causal_credit_eligible": False,
            "causal_credit_blocker": "requires_factorized_heldout_replication_and_settled_verdict",
        },
        "selection_policy": {
            "trajectory_pool": "exact_baseline_candidate_intersection",
            "candidate_gain_used_for_sample_selection": False,
            "hard_case_selection_allowed_for_qualitative_showcase_only": True,
        },
        "claim_boundary": (
            "This artifact estimates a paired horizon response. It can update routing priors and anti-conditions, "
            "but cannot enter the failure-to-fix causal map until a factorized held-out replication receives a "
            "settled verifier verdict. A hard-case-only improvement is qualitative evidence, not aggregate uplift. "
            "When an event gate is present, pixel-metric gains cannot override a failed physical event."
        ),
    }
    return _write_bundle(destination, report)


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("state") != "ready":
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MANIFEST_NOT_READY")
    if payload.get("artifact_type") != "wmloop-horizon-probe-run":
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MANIFEST_TYPE_INVALID")
    return payload


def _validate_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[str, str, list[int]]:
    environment = str(baseline.get("environment") or "")
    split = str(baseline.get("split") or "")
    if not environment or not split:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_IDENTITY_MISSING")
    for key in ("environment", "split", "metadata_sha256", "mode", "num_inference_steps"):
        if baseline.get(key) != candidate.get(key):
            raise AcwmHorizonEffectProfileError(f"HORIZON_EFFECT_PAIR_MISMATCH:{key}")
    raw_horizons = baseline.get("horizons")
    if raw_horizons != candidate.get("horizons") or not isinstance(raw_horizons, list):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_PAIR_MISMATCH:horizons")
    try:
        horizons = sorted({int(value) for value in raw_horizons})
    except (TypeError, ValueError) as exc:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_HORIZONS_INVALID") from exc
    if not horizons or horizons[0] <= 0:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_HORIZONS_INVALID")
    return environment, split, horizons


def _trajectory_map(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = manifest.get("trajectory_results")
    if not isinstance(rows, list) or not rows:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_TRAJECTORIES_MISSING")
    indexed: dict[int, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_TRAJECTORY_INVALID")
        try:
            index = int(raw["trajectory_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_TRAJECTORY_INVALID") from exc
        if index in indexed:
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_TRAJECTORY_DUPLICATE")
        indexed[index] = raw
    return indexed


def _horizon_metrics(manifest: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_AGGREGATE_MISSING")
    metrics = aggregate.get("horizon_metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(str(horizon)), Mapping):
        raise AcwmHorizonEffectProfileError(f"HORIZON_EFFECT_METRIC_MISSING:{horizon}")
    return metrics[str(horizon)]


def _horizon_row(
    *, horizon: int, baseline: Mapping[str, Any], candidate: Mapping[str, Any], fps: float
) -> dict[str, object]:
    baseline_values = _metric_values(baseline)
    candidate_values = _metric_values(candidate)
    delta = {name: candidate_values[name] - baseline_values[name] for name in _METRICS}
    checks = _quality_checks(delta)
    return {
        "horizon": horizon,
        "seconds": horizon / fps,
        "sample_count": _matching_sample_count(baseline, candidate),
        "baseline": baseline_values,
        "candidate": candidate_values,
        "delta_candidate_minus_baseline": delta,
        "aligned_gain": {
            "psnr": delta["psnr"],
            "ssim": delta["ssim"],
            "mse": -delta["mse"],
            "masked_mse": -delta["masked_mse"],
        },
        "checks": checks,
        "strict_quality_pass": all(checks.values()),
    }


def _trajectory_row(
    *,
    trajectory_index: int,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    max_horizon: int,
) -> dict[str, object]:
    baseline_values = _trajectory_horizon_metrics(baseline, max_horizon)
    candidate_values = _trajectory_horizon_metrics(candidate, max_horizon)
    delta = {name: candidate_values[name] - baseline_values[name] for name in _METRICS}
    checks = _quality_checks(delta)
    return {
        "trajectory_index": trajectory_index,
        "horizon": max_horizon,
        "baseline": baseline_values,
        "candidate": candidate_values,
        "delta_candidate_minus_baseline": delta,
        "strict_quality_pass": all(checks.values()),
    }


def _trajectory_horizon_metrics(row: Mapping[str, Any], horizon: int) -> dict[str, float]:
    metrics = row.get("metrics_by_horizon")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(str(horizon)), Mapping):
        raise AcwmHorizonEffectProfileError(f"HORIZON_EFFECT_TRAJECTORY_HORIZON_MISSING:{horizon}")
    return _metric_values(metrics[str(horizon)])


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in _METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcwmHorizonEffectProfileError(f"HORIZON_EFFECT_METRIC_INVALID:{name}") from exc
        if not math.isfinite(value):
            raise AcwmHorizonEffectProfileError(f"HORIZON_EFFECT_METRIC_INVALID:{name}")
        values[name] = value
    return values


def _matching_sample_count(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    try:
        left = int(baseline["sample_count"])
        right = int(candidate["sample_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_SAMPLE_COUNT_INVALID") from exc
    if left <= 0 or left != right:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_SAMPLE_COUNT_MISMATCH")
    return left


def _quality_checks(delta: Mapping[str, float]) -> dict[str, bool]:
    return {
        "psnr_strictly_improves": delta["psnr"] > 0.0,
        "ssim_does_not_regress": delta["ssim"] >= 0.0,
        "mse_does_not_regress": delta["mse"] <= 0.0,
        "masked_mse_does_not_regress": delta["masked_mse"] <= 0.0,
    }


def _per_frame_profile(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, fps: float
) -> dict[str, object]:
    baseline_curve = _per_frame_curve(baseline)
    candidate_curve = _per_frame_curve(candidate)
    if set(baseline_curve) != set(candidate_curve):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_PER_FRAME_SET_MISMATCH")
    frames = sorted(baseline_curve)
    delta = {frame: candidate_curve[frame] - baseline_curve[frame] for frame in frames}
    split_index = max(1, len(frames) // 2)
    early = frames[:split_index]
    late = frames[split_index:]
    return {
        "frame_count": len(frames),
        "duration_seconds": len(frames) / fps,
        "delta_psnr_by_frame": {str(frame): delta[frame] for frame in frames},
        "early_half_mean_delta_psnr": _mean(delta[frame] for frame in early),
        "late_half_mean_delta_psnr": _mean(delta[frame] for frame in late),
        "tail_frame_delta_psnr": delta[frames[-1]],
    }


def _per_frame_curve(manifest: Mapping[str, Any]) -> dict[int, float]:
    aggregate = manifest.get("aggregate")
    raw = aggregate.get("per_frame_psnr") if isinstance(aggregate, Mapping) else None
    if not isinstance(raw, Mapping) or not raw:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_PER_FRAME_MISSING")
    curve: dict[int, float] = {}
    for frame, value in raw.items():
        point = float(value)
        if not math.isfinite(point):
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_PER_FRAME_INVALID")
        curve[int(frame)] = point
    return curve


def _classify_effect(
    horizons: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    per_frame: Mapping[str, Any],
) -> dict[str, object]:
    passing = [int(row["horizon"]) for row in horizons if row["strict_quality_pass"] is True]
    max_horizon = max(int(row["horizon"]) for row in horizons)
    max_pass = max_horizon in passing
    positive_trajectories = [
        int(row["trajectory_index"]) for row in trajectories if row["strict_quality_pass"] is True
    ]
    if max_pass:
        scope = "aggregate_long_horizon_positive"
    elif passing:
        scope = "short_horizon_only_positive"
    elif positive_trajectories:
        scope = "hard_case_only_positive"
    else:
        scope = "aggregate_negative_or_mixed"
    first_failure = next(
        (int(row["horizon"]) for row in horizons if row["strict_quality_pass"] is not True),
        None,
    )
    return {
        "effect_scope": scope,
        "aggregate_passing_horizons": passing,
        "aggregate_max_horizon_pass": max_pass,
        "first_aggregate_failure_horizon": first_failure,
        "positive_trajectory_indices_at_max_horizon": positive_trajectories,
        "positive_trajectory_rate_at_max_horizon": len(positive_trajectories) / len(trajectories),
        "late_half_mean_delta_psnr": per_frame["late_half_mean_delta_psnr"],
        "claim_license": "routing_prior_only_pending_factorized_heldout_replication",
    }


def _failure_signatures(path: Path | None, environment: str) -> list[str]:
    if path is None:
        return []
    payload = _load_json_object(Path(path).resolve(strict=True), "HORIZON_EFFECT_FAILURE_REPORT_INVALID")
    if str(payload.get("env") or "") != environment:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_FAILURE_REPORT_ENV_MISMATCH")
    values = payload.get("dominant_failure_candidates")
    if isinstance(values, list) and all(isinstance(value, str) and value for value in values):
        return list(dict.fromkeys(values))
    dominant = payload.get("dominant_failure")
    return [str(dominant)] if isinstance(dominant, str) and dominant else []


def _mechanism_card(path: Path | None, primitive: str) -> dict[str, str | None]:
    empty = {
        "mechanism_family": None,
        "layer": None,
        "transfer_preconditions": None,
        "known_anti_conditions": None,
    }
    if path is None:
        return empty
    try:
        with Path(path).resolve(strict=True).open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MECHANISM_CARDS_INVALID") from exc
    row = next((item for item in rows if item.get("primitive") == primitive), None)
    if row is None:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MECHANISM_CARD_MISSING")
    return {key: row.get(key) or None for key in empty}


def _event_semantics(path: Path | None, environment: str) -> dict[str, object] | None:
    if path is None:
        return None
    payload = _load_json_object(Path(path).resolve(strict=True), "HORIZON_EFFECT_EVENT_GATE_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-acwm-pour-water-event-gate"
        or payload.get("state") != "ready"
        or payload.get("environment") != environment
        or not isinstance(payload.get("classification"), str)
        or not isinstance(payload.get("candidate_event_pass"), bool)
    ):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_EVENT_GATE_INVALID")
    return {
        "classification": payload["classification"],
        "candidate_event_pass": payload["candidate_event_pass"],
        "baseline_event_pass": payload.get("baseline_event_pass") is True,
        "baseline_mean_completion_ratio": payload.get("baseline_mean_completion_ratio"),
        "candidate_mean_completion_ratio": payload.get("candidate_mean_completion_ratio"),
        "mean_completion_uplift": payload.get("mean_completion_uplift"),
        "report_path": str(Path(path).resolve()),
    }


def _checkpoint_ladder_evidence(
    path: Path | None,
    *,
    environment: str,
    primitive: str,
) -> dict[str, object] | None:
    if path is None:
        return None
    manifest_path = Path(path).resolve(strict=True)
    manifest = _load_json_object(manifest_path, "HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID")
    if (
        manifest.get("artifact_type") != "wmloop-acwm-checkpoint-ladder-finalization-manifest"
        or manifest.get("state") != "ready"
        or manifest.get("environment") != environment
        or manifest.get("primitive") != primitive
        or manifest.get("confirmation_passed") is not True
    ):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID")
    report_path = Path(str(manifest.get("report_path") or "")).resolve(strict=True)
    report = _load_json_object(report_path, "HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID")
    selection = report.get("selection")
    if (
        report.get("artifact_type") != "wmloop-acwm-checkpoint-ladder-finalization"
        or report.get("state") != "ready"
        or report.get("environment") != environment
        or report.get("primitive") != primitive
        or not isinstance(selection, Mapping)
        or selection.get("state") != "ready"
        or not isinstance(selection.get("records"), list)
    ):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID")
    try:
        best_step = int(report["best_checkpoint_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID") from exc
    records: list[dict[str, object]] = []
    regression_steps: list[int] = []
    for raw in selection["records"]:
        if not isinstance(raw, Mapping):
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID")
        quality = raw.get("official_quality_gate")
        delta = quality.get("delta_candidate_minus_baseline") if isinstance(quality, Mapping) else None
        try:
            step = int(raw["checkpoint_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_CHECKPOINT_LADDER_INVALID") from exc
        regressed = raw.get("regressed_from_running_best") is True
        if regressed:
            regression_steps.append(step)
        records.append(
            {
                "checkpoint_step": step,
                "official_gate_passed": raw.get("official_gate_passed") is True,
                "regressed_from_running_best": regressed,
                "delta_candidate_minus_baseline": dict(delta) if isinstance(delta, Mapping) else None,
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "best_checkpoint_step": best_step,
        "evaluated_steps": [int(value) for value in report.get("evaluated_steps", [])],
        "records": records,
        "regression_steps": regression_steps,
        "effective_training_window": {
            "selected_step": best_step,
            "later_regression_observed": any(step > best_step for step in regression_steps),
            "first_later_regression_step": min(
                (step for step in regression_steps if step > best_step),
                default=None,
            ),
            "extension_allowed": selection.get("extension_allowed") is True,
            "stop_requested": selection.get("stop_requested") is True,
        },
    }


def _load_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmHorizonEffectProfileError(code) from exc
    if not isinstance(payload, Mapping):
        raise AcwmHorizonEffectProfileError(code)
    return payload


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    if not materialized:
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_MEAN_EMPTY")
    return sum(materialized) / len(materialized)


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json(temporary / "horizon-effect-profile.json", report)
        _write_horizon_csv(temporary / "horizon-effects.csv", report["horizon_effects"])
        (temporary / "horizon-effect-profile.md").write_text(_markdown(report), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-horizon-effect-profile-manifest",
        "state": "ready",
        "environment": report["environment"],
        "primitive": report["primitive"],
        "effect_scope": report["effect_classification"]["effect_scope"],
        "report_path": str(destination / "horizon-effect-profile.json"),
        "markdown_path": str(destination / "horizon-effect-profile.md"),
        "csv_path": str(destination / "horizon-effects.csv"),
        "causal_credit_eligible": False,
    }
    _write_json(destination / "manifest.json", manifest)
    return {**dict(report), "manifest": manifest}


def _write_horizon_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise AcwmHorizonEffectProfileError("HORIZON_EFFECT_ROWS_INVALID")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "horizon",
                "seconds",
                "sample_count",
                "baseline_psnr",
                "candidate_psnr",
                "delta_psnr",
                "delta_ssim",
                "delta_mse",
                "delta_masked_mse",
                "strict_quality_pass",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "horizon": row["horizon"],
                    "seconds": row["seconds"],
                    "sample_count": row["sample_count"],
                    "baseline_psnr": row["baseline"]["psnr"],
                    "candidate_psnr": row["candidate"]["psnr"],
                    "delta_psnr": row["delta_candidate_minus_baseline"]["psnr"],
                    "delta_ssim": row["delta_candidate_minus_baseline"]["ssim"],
                    "delta_mse": row["delta_candidate_minus_baseline"]["mse"],
                    "delta_masked_mse": row["delta_candidate_minus_baseline"]["masked_mse"],
                    "strict_quality_pass": row["strict_quality_pass"],
                }
            )


def _markdown(report: Mapping[str, object]) -> str:
    effect = report["effect_classification"]
    lines = [
        "# ACWM Horizon Effect Profile",
        "",
        f"- Environment: `{report['environment']}`",
        f"- Primitive: `{report['primitive']}`",
        f"- Mechanism primitive: `{report['mechanism_primitive']}`",
        f"- Effect scope: `{effect['effect_scope']}`",
        *(
            [
                f"- Metric-only scope: `{effect['metric_effect_scope']}`",
                f"- Event classification: `{effect['event_semantic_classification']}`",
                f"- Candidate event pass: `{effect['candidate_event_pass']}`",
            ]
            if "event_semantic_classification" in effect
            else []
        ),
        f"- Paired trajectories: `{report['paired_trajectory_count']}`",
        f"- Causal credit eligible: `False`",
        "",
        "| Horizon | Seconds | Delta PSNR | Delta SSIM | Delta MSE | Delta masked-MSE | Strict pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["horizon_effects"]:
        delta = row["delta_candidate_minus_baseline"]
        lines.append(
            f"| {row['horizon']} | {row['seconds']:.2f} | {delta['psnr']:+.4f} | "
            f"{delta['ssim']:+.6f} | {delta['mse']:+.6f} | {delta['masked_mse']:+.6f} | "
            f"{row['strict_quality_pass']} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument(
        "--mechanism-primitive",
        help="Registry primitive that supplies the mechanism card for a composite intervention.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--mechanism-cards", type=Path)
    parser.add_argument("--event-gate", type=Path)
    parser.add_argument("--checkpoint-ladder-manifest", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_horizon_effect_profile(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        primitive=args.primitive,
        mechanism_primitive=args.mechanism_primitive,
        output_root=args.output_root,
        failure_report=args.failure_report,
        mechanism_cards=args.mechanism_cards,
        event_gate=args.event_gate,
        checkpoint_ladder_manifest=args.checkpoint_ladder_manifest,
        fps=args.fps,
    )
    print(json.dumps(report["manifest"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
