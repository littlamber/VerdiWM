"""Generate the T4.4 prior-vs-cold-start convergence report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import load_yaml_document, validate_document


REQUIRED_ARMS = ("prior", "cold_start", "shuffled_prior")
DEFAULT_PROGRESS_THRESHOLD = 0.8
DEFAULT_MIN_SEED_COUNT = 3
DEFAULT_PLATEAU_TOLERANCE = 0.05
DEFAULT_SPEEDUP_TARGET = 1.5


class T44PriorConvergenceError(RuntimeError):
    """T4.4 prior convergence reporting failed closed."""


def run_t44_prior_convergence(
    *,
    archive_db: Path,
    output_root: Path,
    arms: Mapping[str, Sequence[Path]] | None = None,
    cas_root: Path | None = None,
    primitive_materialization_gate_manifest: Path | None = None,
    goal_config: Path | None = None,
    target_progress_delta: float | None = None,
    progress_threshold: float | None = None,
    min_seed_count: int | None = None,
    plateau_tolerance: float | None = None,
    speedup_target: float | None = None,
) -> dict[str, object]:
    """Write a read-only three-arm convergence report for T4.4."""

    archive = ArchiveStore(archive_db)
    cas_storage_root = Path(cas_root) if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(cas_storage_root)
    goal_protocol = _load_goal_t44_protocol(goal_config, cas=cas, archive=archive) if goal_config is not None else None
    target_progress_delta, target_source = _resolve_protocol_float(
        name="target_progress_delta",
        cli_value=target_progress_delta,
        protocol=goal_protocol,
        default=None,
    )
    progress_threshold, progress_source = _resolve_protocol_float(
        name="progress_threshold",
        cli_value=progress_threshold,
        protocol=goal_protocol,
        default=DEFAULT_PROGRESS_THRESHOLD,
    )
    min_seed_count, min_seed_source = _resolve_protocol_int(
        name="min_seed_count",
        cli_value=min_seed_count,
        protocol=goal_protocol,
        default=DEFAULT_MIN_SEED_COUNT,
    )
    plateau_tolerance, plateau_source = _resolve_protocol_float(
        name="plateau_tolerance",
        cli_value=plateau_tolerance,
        protocol=goal_protocol,
        default=DEFAULT_PLATEAU_TOLERANCE,
    )
    speedup_target, speedup_source = _resolve_protocol_float(
        name="speedup_target",
        cli_value=speedup_target,
        protocol=goal_protocol,
        default=DEFAULT_SPEEDUP_TARGET,
    )
    if progress_threshold <= 0.0 or progress_threshold > 1.0:
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_PROGRESS_THRESHOLD_INVALID")
    if min_seed_count < 1:
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MIN_SEED_INVALID")
    if plateau_tolerance < 0.0 or not math.isfinite(plateau_tolerance):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_PLATEAU_TOLERANCE_INVALID")
    if speedup_target <= 0.0 or not math.isfinite(speedup_target):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_SPEEDUP_TARGET_INVALID")
    if target_progress_delta is not None and (
        not math.isfinite(float(target_progress_delta)) or float(target_progress_delta) <= 0.0
    ):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_TARGET_DELTA_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_OUTPUT_EXISTS")
    materialization_gate = (
        _load_materialization_gate(primitive_materialization_gate_manifest, cas=cas, archive=archive)
        if primitive_materialization_gate_manifest is not None
        else None
    )
    sources: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for arm, paths in sorted((arms or {}).items()):
        if arm not in REQUIRED_ARMS:
            raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_ARM_INVALID:{arm}")
        for path in paths:
            source, loaded = _load_trial_or_campaign(path, arm=arm, cas=cas, archive=archive)
            sources.append(source)
            records.extend(loaded)
    report = _report(
        records=records,
        sources=sources,
        materialization_gate=materialization_gate,
        target_progress_delta=target_progress_delta,
        progress_threshold=progress_threshold,
        min_seed_count=min_seed_count,
        plateau_tolerance=plateau_tolerance,
        speedup_target=speedup_target,
        protocol_config=goal_protocol,
        protocol_parameter_sources={
            "target_progress_delta": target_source,
            "progress_threshold": progress_source,
            "min_seed_count": min_seed_source,
            "plateau_tolerance": plateau_source,
            "speedup_target": speedup_source,
        },
    )
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _load_goal_t44_protocol(
    goal_config: Path,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    source = Path(goal_config).resolve(strict=True)
    payload_bytes = source.read_bytes()
    try:
        goal = load_yaml_document(source)
        validate_document("goal_spec", goal)
    except Exception as exc:
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_GOAL_CONFIG_INVALID:{source}") from exc
    eval_protocol = goal.get("eval_protocol")
    if not isinstance(eval_protocol, Mapping):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_GOAL_EVAL_PROTOCOL_INVALID")
    protocol = eval_protocol.get("t4_4_prior_convergence")
    if not isinstance(protocol, Mapping):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_GOAL_T44_PROTOCOL_MISSING")
    ref = cas.put_bytes(payload_bytes, media_type="application/yaml").uri
    archive.record_artifact_reference(ref)
    return {
        "summary": {
            "path": str(source),
            "artifact_type": "goal_spec",
            "goal_id": goal.get("goal_id"),
            "cas_ref": ref,
        },
        "protocol": dict(protocol),
    }


def _resolve_protocol_float(
    *,
    name: str,
    cli_value: float | None,
    protocol: Mapping[str, object] | None,
    default: float | None,
) -> tuple[float | None, str]:
    protocol_value = _protocol_value(protocol, name)
    if cli_value is not None and protocol_value is not None and not math.isclose(float(cli_value), float(protocol_value), rel_tol=0.0, abs_tol=1e-12):
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_PROTOCOL_OVERRIDE_CONFLICT:{name}")
    if protocol_value is not None:
        return float(protocol_value), "goal_config"
    if cli_value is not None:
        return float(cli_value), "cli"
    if default is None:
        return None, "missing"
    return float(default), "default"


def _resolve_protocol_int(
    *,
    name: str,
    cli_value: int | None,
    protocol: Mapping[str, object] | None,
    default: int,
) -> tuple[int, str]:
    protocol_value = _protocol_value(protocol, name)
    if cli_value is not None and protocol_value is not None and int(cli_value) != int(protocol_value):
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_PROTOCOL_OVERRIDE_CONFLICT:{name}")
    if protocol_value is not None:
        return int(protocol_value), "goal_config"
    if cli_value is not None:
        return int(cli_value), "cli"
    return int(default), "default"


def _protocol_value(protocol: Mapping[str, object] | None, name: str) -> object | None:
    if protocol is None:
        return None
    payload = protocol.get("protocol")
    if not isinstance(payload, Mapping):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_GOAL_T44_PROTOCOL_INVALID")
    return payload.get(name)


def _load_trial_or_campaign(
    path: Path,
    *,
    arm: str,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    artifact_type = payload.get("artifact_type")
    if artifact_type == "wmloop-training-eval-limited-campaign-manifest":
        trial_sources = []
        trials: list[dict[str, object]] = []
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_CAMPAIGN_RECORDS_INVALID:{path}")
        for item in raw_records:
            if not isinstance(item, Mapping) or item.get("state") != "ready":
                continue
            manifest_path = item.get("manifest_path")
            if not isinstance(manifest_path, str) or not manifest_path:
                raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_CAMPAIGN_TRIAL_MANIFEST_MISSING")
            child_source, child_payload = _load_trial_manifest(Path(manifest_path), cas=cas, archive=archive)
            trial_sources.append(child_source["summary"])
            trials.append(_trial_record(child_payload, arm=arm, cas=cas, archive=archive))
        summary = source["summary"]
        if isinstance(summary, dict):
            summary["trial_sources"] = trial_sources
        return source, trials
    if artifact_type == "wmloop-m3-training-eval-smoke-manifest":
        return source, [_trial_record(payload, arm=arm, cas=cas, archive=archive)]
    raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_SOURCE_INVALID:{path}")


def _load_trial_manifest(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> tuple[dict[str, object], Mapping[str, Any]]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    if payload.get("artifact_type") != "wmloop-m3-training-eval-smoke-manifest":
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_TRIAL_INVALID:{path}")
    return source, payload


def _trial_record(
    manifest: Mapping[str, Any],
    *,
    arm: str,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    report = _load_trial_report(manifest, cas=cas, archive=archive)
    evaluation = _evaluation(report)
    baseline = _finite_first(
        [
            evaluation.get("baseline_primary_metric"),
            evaluation.get("baseline_auc_psnr_16_64"),
        ],
        "T44_PRIOR_CONVERGENCE_BASELINE_METRIC_MISSING",
    )
    candidate = _finite_first(
        [
            evaluation.get("candidate_primary_metric"),
            evaluation.get("candidate_auc_psnr_16_64"),
        ],
        "T44_PRIOR_CONVERGENCE_CANDIDATE_METRIC_MISSING",
    )
    delta = _finite_first(
        [
            _delta_for_primary(manifest),
            evaluation.get("delta_primary_metric"),
            evaluation.get("delta_auc_psnr_16_64"),
        ],
        "T44_PRIOR_CONVERGENCE_DELTA_MISSING",
    )
    gpu_hours = _finite_first(
        [
            report.get("actual_gpu_hours"),
            _nested(report, ("receipt", "actual_gpu_hours")),
        ],
        "T44_PRIOR_CONVERGENCE_GPU_HOURS_MISSING",
    )
    if gpu_hours < 0:
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_GPU_HOURS_INVALID")
    return {
        "arm": arm,
        "seed": _seed(manifest, report),
        "environment": manifest.get("environment"),
        "goal_id": manifest.get("goal_id"),
        "primary_metric": manifest.get("primary_metric") or evaluation.get("primary_metric"),
        "proposal_id": manifest.get("proposal_id"),
        "verdict": manifest.get("verdict"),
        "baseline_primary_metric": baseline,
        "candidate_primary_metric": candidate,
        "delta_primary_metric": delta,
        "gpu_hours": gpu_hours,
        "rendered_primitives": _rendered_primitives(report),
        "report_path": manifest.get("report_path"),
    }


def _report(
    *,
    records: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
    materialization_gate: Mapping[str, object] | None,
    target_progress_delta: float | None,
    progress_threshold: float,
    min_seed_count: int,
    plateau_tolerance: float,
    speedup_target: float,
    protocol_config: Mapping[str, object] | None,
    protocol_parameter_sources: Mapping[str, object],
) -> dict[str, object]:
    normalized = _normalized_records(records, target_progress_delta=target_progress_delta)
    curves = _curves(normalized)
    arm_names = sorted({str(record["arm"]) for record in normalized})
    seed_counts = _seed_counts(normalized)
    threshold_budgets = _threshold_budgets(curves, threshold=progress_threshold)
    budget_ratio = _budget_ratio(threshold_budgets, speedup_target=speedup_target)
    plateau = _plateau(curves, tolerance=plateau_tolerance)
    blockers = []
    missing_arms = [arm for arm in REQUIRED_ARMS if arm not in arm_names]
    if missing_arms:
        blockers.append({"code": "required_arms_missing", "arms": missing_arms})
    if target_progress_delta is None:
        blockers.append({"code": "target_progress_delta_missing"})
    for arm in REQUIRED_ARMS:
        if seed_counts.get(arm, 0) < min_seed_count:
            blockers.append(
                {
                    "code": "arm_seed_count_below_minimum",
                    "arm": arm,
                    "observed": seed_counts.get(arm, 0),
                    "minimum": min_seed_count,
                }
            )
    if not all(curves.get(arm) for arm in REQUIRED_ARMS):
        blockers.append({"code": "convergence_curve_missing"})
    if not plateau["ready"]:
        blockers.append({"code": "plateau_consistency_unavailable", "observed": plateau})
    blockers.extend(_materialization_blockers(normalized, materialization_gate=materialization_gate))
    result_settled = not blockers
    outcome = _t44_outcome(result_settled=result_settled, budget_ratio=budget_ratio, plateau=plateau)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-t4-4-prior-convergence",
        "state": "ready" if result_settled else "blocked",
        "t4_4_prior_convergence_ready": result_settled,
        "t4_4_result_settled": result_settled,
        "t4_4_outcome": outcome["outcome"],
        "positive_result_ready": outcome["positive_result_ready"],
        "negative_result_ready": outcome["negative_result_ready"],
        "outcome_reasons": outcome["reasons"],
        "arms": arm_names,
        "required_arms": list(REQUIRED_ARMS),
        "target_progress_delta": target_progress_delta,
        "target_progress_delta_basis": _target_progress_delta_basis(protocol_config),
        "protocol_config": None if protocol_config is None else protocol_config["summary"],
        "protocol_parameter_sources": dict(protocol_parameter_sources),
        "progress_threshold": progress_threshold,
        "min_seed_count": min_seed_count,
        "seed_counts": seed_counts,
        "convergence_curves_ready": all(curves.get(arm) for arm in REQUIRED_ARMS),
        "budget_ratio_ready": bool(budget_ratio["ready"]),
        "plateau_consistency_ready": bool(plateau["ready"]),
        "convergence_curves": curves,
        "threshold_budgets": threshold_budgets,
        "budget_ratio": budget_ratio,
        "plateau_consistency": plateau,
        "records": [dict(record) for record in normalized],
        "blockers": blockers,
        "sources": [source["summary"] for source in sources],
        "primitive_materialization_gate": materialization_gate,
        "limitations": [
            "T4.4 readiness means the three-arm hypothesis test is settled; the outcome may be positive or negative.",
            "Normalized progress uses delta_primary_metric / target_progress_delta; target_progress_delta must be declared before a ready report.",
            "A missing or failed budget-ratio crossing is recorded as negative evidence when the required arms, seeds, curves, plateau, and primitive materialization checks are complete.",
            "If a primitive materialization gate is supplied, every T4.4 trial primitive must be closed-loop eligible.",
            "This reporter only aggregates closed-loop evidence; it does not run new training or alter the archive.",
        ],
    }


def _t44_outcome(
    *,
    result_settled: bool,
    budget_ratio: Mapping[str, object],
    plateau: Mapping[str, object],
) -> dict[str, object]:
    if not result_settled:
        return {
            "outcome": "incomplete",
            "positive_result_ready": False,
            "negative_result_ready": False,
            "reasons": [{"code": "hard_blockers_present"}],
        }
    positive = (
        budget_ratio.get("ready") is True
        and budget_ratio.get("passes_speedup_target") is True
        and budget_ratio.get("shuffled_not_better_than_cold_start") is True
        and plateau.get("prior_not_lower_than_cold_start") is True
    )
    if positive:
        return {
            "outcome": "positive",
            "positive_result_ready": True,
            "negative_result_ready": False,
            "reasons": [{"code": "prior_speedup_supported"}],
        }
    reasons: list[dict[str, object]] = []
    if budget_ratio.get("ready") is not True:
        reasons.append({"code": "budget_ratio_unavailable", "observed": budget_ratio})
    elif budget_ratio.get("passes_speedup_target") is not True:
        reasons.append(
            {
                "code": "speedup_target_not_met",
                "observed": {
                    "prior_vs_cold_start_speedup": budget_ratio.get("prior_vs_cold_start_speedup"),
                    "speedup_target": budget_ratio.get("speedup_target"),
                },
            }
        )
    if budget_ratio.get("shuffled_not_better_than_cold_start") is False:
        reasons.append(
            {
                "code": "shuffled_prior_better_than_cold_start",
                "observed": {
                    "shuffled_prior_vs_cold_start_speedup": budget_ratio.get("shuffled_prior_vs_cold_start_speedup"),
                },
            }
        )
    if plateau.get("prior_not_lower_than_cold_start") is False:
        reasons.append(
            {
                "code": "prior_plateau_lower_than_cold_start",
                "observed": {
                    "prior_minus_cold_start": plateau.get("prior_minus_cold_start"),
                    "plateau_tolerance": plateau.get("plateau_tolerance"),
                },
            }
        )
    if not reasons:
        reasons.append({"code": "positive_criteria_not_met"})
    return {
        "outcome": "negative",
        "positive_result_ready": False,
        "negative_result_ready": True,
        "reasons": reasons,
    }


def _normalized_records(
    records: Sequence[Mapping[str, object]],
    *,
    target_progress_delta: float | None,
) -> list[dict[str, object]]:
    output = []
    for index, record in enumerate(records):
        delta = _finite_float(record.get("delta_primary_metric"), "T44_PRIOR_CONVERGENCE_DELTA_INVALID")
        progress = None if target_progress_delta is None else delta / float(target_progress_delta)
        output.append({**dict(record), "input_order": index, "normalized_progress": progress})
    return output


def _target_progress_delta_basis(protocol_config: Mapping[str, object] | None) -> str | None:
    if protocol_config is None:
        return None
    protocol = protocol_config.get("protocol")
    if not isinstance(protocol, Mapping):
        return None
    basis = protocol.get("target_progress_delta_basis")
    return str(basis) if isinstance(basis, str) and basis else None


def _curves(records: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_arm_seed: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_arm_seed[(str(record["arm"]), str(record["seed"]))].append(record)
    curves: dict[str, list[dict[str, object]]] = {}
    for (arm, seed), seed_records in sorted(by_arm_seed.items()):
        cumulative = 0.0
        for step, record in enumerate(sorted(seed_records, key=lambda item: int(item.get("input_order", 0))), start=1):
            gpu_hours = _finite_float(record.get("gpu_hours"), "T44_PRIOR_CONVERGENCE_GPU_HOURS_INVALID")
            cumulative += gpu_hours
            curves.setdefault(arm, []).append(
                {
                    "seed": seed,
                    "step": step,
                    "cumulative_gpu_hours": cumulative,
                    "normalized_progress": record.get("normalized_progress"),
                    "delta_primary_metric": record.get("delta_primary_metric"),
                    "verdict": record.get("verdict"),
                    "proposal_id": record.get("proposal_id"),
                    "environment": record.get("environment"),
                }
            )
    for values in curves.values():
        values.sort(key=lambda item: (str(item["seed"]), int(item["step"])))
    return curves


def _seed_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    seeds: dict[str, set[str]] = defaultdict(set)
    for record in records:
        seeds[str(record["arm"])].add(str(record["seed"]))
    return {arm: len(values) for arm, values in sorted(seeds.items())}


def _threshold_budgets(curves: Mapping[str, Sequence[Mapping[str, object]]], *, threshold: float) -> dict[str, object]:
    budgets: dict[str, object] = {}
    for arm, values in sorted(curves.items()):
        by_seed: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for item in values:
            by_seed[str(item["seed"])].append(item)
        seed_results = []
        reached_values = []
        for seed, seed_values in sorted(by_seed.items()):
            crossing = None
            for item in sorted(seed_values, key=lambda entry: int(entry["step"])):
                progress = item.get("normalized_progress")
                if isinstance(progress, (int, float)) and not isinstance(progress, bool) and math.isfinite(float(progress)):
                    if float(progress) >= threshold:
                        crossing = item
                        break
            result = {
                "seed": seed,
                "reached": crossing is not None,
                "gpu_hours": None if crossing is None else crossing["cumulative_gpu_hours"],
                "step": None if crossing is None else crossing["step"],
            }
            if crossing is not None:
                reached_values.append(float(crossing["cumulative_gpu_hours"]))
            seed_results.append(result)
        budgets[arm] = {
            "seed_results": seed_results,
            "mean_gpu_hours_to_threshold": None if not reached_values else sum(reached_values) / len(reached_values),
            "reached_seed_count": len(reached_values),
            "seed_count": len(seed_results),
        }
    return budgets


def _budget_ratio(threshold_budgets: Mapping[str, object], *, speedup_target: float) -> dict[str, object]:
    prior = _mean_budget(threshold_budgets, "prior")
    cold = _mean_budget(threshold_budgets, "cold_start")
    shuffled = _mean_budget(threshold_budgets, "shuffled_prior")
    ready = prior is not None and cold is not None and shuffled is not None
    speedup = None if prior is None or cold is None or prior <= 0 else cold / prior
    shuffled_vs_cold = None if shuffled is None or cold is None or shuffled <= 0 else cold / shuffled
    return {
        "ready": ready,
        "prior_gpu_hours_to_80": prior,
        "cold_start_gpu_hours_to_80": cold,
        "shuffled_prior_gpu_hours_to_80": shuffled,
        "prior_vs_cold_start_speedup": speedup,
        "shuffled_prior_vs_cold_start_speedup": shuffled_vs_cold,
        "speedup_target": speedup_target,
        "passes_speedup_target": None if speedup is None else speedup >= speedup_target,
        "shuffled_not_better_than_cold_start": None if shuffled is None or cold is None else shuffled >= cold,
    }


def _mean_budget(threshold_budgets: Mapping[str, object], arm: str) -> float | None:
    payload = threshold_budgets.get(arm)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("mean_gpu_hours_to_threshold")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _plateau(curves: Mapping[str, Sequence[Mapping[str, object]]], *, tolerance: float) -> dict[str, object]:
    means: dict[str, float] = {}
    for arm in REQUIRED_ARMS:
        values = curves.get(arm)
        if not values:
            continue
        final_by_seed: dict[str, Mapping[str, object]] = {}
        for item in values:
            final_by_seed[str(item["seed"])] = item
        final_progress = []
        for item in final_by_seed.values():
            progress = item.get("normalized_progress")
            if isinstance(progress, (int, float)) and not isinstance(progress, bool) and math.isfinite(float(progress)):
                final_progress.append(float(progress))
        if final_progress:
            means[arm] = sum(final_progress) / len(final_progress)
    ready = all(arm in means for arm in REQUIRED_ARMS)
    prior_minus_cold = None if not ready else means["prior"] - means["cold_start"]
    return {
        "ready": ready,
        "mean_final_progress_by_arm": means,
        "plateau_tolerance": tolerance,
        "prior_minus_cold_start": prior_minus_cold,
        "prior_not_lower_than_cold_start": None if prior_minus_cold is None else prior_minus_cold >= -tolerance,
    }


def _load_trial_report(manifest: Mapping[str, Any], *, cas: ContentAddressedStore, archive: ArchiveStore) -> Mapping[str, Any]:
    path = manifest.get("report_path")
    if not isinstance(path, str) or not path:
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_REPORT_PATH_MISSING")
    source = _load_source(Path(path), cas=cas, archive=archive)
    payload = source["payload"]
    if payload.get("artifact_type") != "wmloop-m3-training-eval-smoke-report":
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_REPORT_INVALID")
    return payload


def _evaluation(report: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = report.get("receipt")
    if isinstance(receipt, Mapping):
        evaluation = receipt.get("evaluation")
        if isinstance(evaluation, Mapping):
            return evaluation
    raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_EVALUATION_MISSING")


def _delta_for_primary(manifest: Mapping[str, Any]) -> float | None:
    deltas = manifest.get("delta_m_ver")
    if not isinstance(deltas, Mapping) or not deltas:
        return None
    primary = manifest.get("primary_metric")
    if isinstance(primary, str):
        value = deltas.get(primary)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    for value in deltas.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _seed(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    for payload in (manifest, report):
        value = payload.get("seed")
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    evaluation = _nested(report, ("receipt", "evaluation"))
    if isinstance(evaluation, Mapping):
        replications = evaluation.get("replications")
        if isinstance(replications, list) and len(replications) == 1 and isinstance(replications[0], Mapping):
            value = replications[0].get("seed")
            if isinstance(value, (str, int)) and str(value):
                return str(value)
    proposal_id = manifest.get("proposal_id")
    return str(proposal_id) if isinstance(proposal_id, str) and proposal_id else "unknown"


def _rendered_primitives(report: Mapping[str, Any]) -> list[str]:
    receipt = report.get("receipt")
    if not isinstance(receipt, Mapping):
        return []
    rendered = receipt.get("rendered_primitives")
    if not isinstance(rendered, list):
        return []
    names: list[str] = []
    for item in rendered:
        if isinstance(item, Mapping) and isinstance(item.get("name"), str) and item["name"]:
            names.append(str(item["name"]))
    return names


def _load_materialization_gate(
    manifest_path: Path,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    source = _load_source(manifest_path, cas=cas, archive=archive)
    payload = source["payload"]
    if payload.get("artifact_type") != "wmloop-primitive-materialization-gate-manifest":
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MATERIALIZATION_GATE_INVALID")
    report_path = payload.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MATERIALIZATION_GATE_REPORT_MISSING")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = report_source["payload"]
    if report.get("artifact_type") != "wmloop-primitive-materialization-gate":
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MATERIALIZATION_GATE_REPORT_INVALID")
    ready = report.get("closed_loop_ready_primitives")
    if not isinstance(ready, list) or not all(isinstance(item, str) and item for item in ready):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MATERIALIZATION_GATE_REPORT_INVALID")
    return {
        "manifest_path": str(Path(manifest_path).resolve(strict=True)),
        "report_path": str(Path(report_path).resolve(strict=True)),
        "state": report.get("state"),
        "closed_loop_ready_count": report.get("closed_loop_ready_count"),
        "closed_loop_ready_primitives": sorted(set(ready)),
        "manifest_ref": source["summary"]["cas_ref"],
        "report_ref": report_source["summary"]["cas_ref"],
    }


def _materialization_blockers(
    records: Sequence[Mapping[str, object]],
    *,
    materialization_gate: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if materialization_gate is None:
        return []
    ready = materialization_gate.get("closed_loop_ready_primitives")
    if not isinstance(ready, list):
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_MATERIALIZATION_GATE_REPORT_INVALID")
    eligible = {str(item) for item in ready}
    blockers: list[dict[str, object]] = []
    for record in records:
        primitives = record.get("rendered_primitives")
        if not isinstance(primitives, list) or not primitives:
            blockers.append(
                {
                    "code": "trial_rendered_primitives_missing",
                    "arm": record.get("arm"),
                    "seed": record.get("seed"),
                    "proposal_id": record.get("proposal_id"),
                }
            )
            continue
        ineligible = sorted({str(item) for item in primitives if str(item) not in eligible})
        if ineligible:
            blockers.append(
                {
                    "code": "trial_primitives_not_closed_loop_eligible",
                    "arm": record.get("arm"),
                    "seed": record.get("seed"),
                    "proposal_id": record.get("proposal_id"),
                    "primitives": ineligible,
                }
            )
    return blockers


def _nested(payload: Mapping[str, Any], keys: Sequence[str]) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_first(values: Sequence[object], code: str) -> float:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    raise T44PriorConvergenceError(code)


def _finite_float(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise T44PriorConvergenceError(code)
    return float(value)


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise T44PriorConvergenceError(f"T44_PRIOR_CONVERGENCE_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "cas_ref": ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        csv_bytes = _render_csv(report).encode("utf-8")
        _write_bytes_atomic(temporary / "prior-convergence.json", report_bytes)
        _write_bytes_atomic(temporary / "prior-convergence.md", markdown_bytes)
        _write_bytes_atomic(temporary / "prior-convergence.csv", csv_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        csv_ref = cas.put_bytes(csv_bytes, media_type="text/csv").uri
        for ref in (report_ref, markdown_ref, csv_ref):
            archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-t4-4-prior-convergence-manifest",
            "state": report["state"],
            "t4_4_prior_convergence_ready": report["t4_4_prior_convergence_ready"],
            "t4_4_result_settled": report["t4_4_result_settled"],
            "t4_4_outcome": report["t4_4_outcome"],
            "positive_result_ready": report["positive_result_ready"],
            "negative_result_ready": report["negative_result_ready"],
            "outcome_reasons": report["outcome_reasons"],
            "arms": report["arms"],
            "convergence_curves_ready": report["convergence_curves_ready"],
            "budget_ratio_ready": report["budget_ratio_ready"],
            "plateau_consistency_ready": report["plateau_consistency_ready"],
            "seed_counts": report["seed_counts"],
            "target_progress_delta": report["target_progress_delta"],
            "blockers": report["blockers"],
            "primitive_materialization_gate": report.get("primitive_materialization_gate"),
            "report_path": str(destination / "prior-convergence.json"),
            "markdown_path": str(destination / "prior-convergence.md"),
            "csv_path": str(destination / "prior-convergence.csv"),
            "cas_refs": {
                "prior_convergence_json": report_ref,
                "prior_convergence_markdown": markdown_ref,
                "prior_convergence_csv": csv_ref,
            },
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# T4.4 Prior vs Cold-Start Convergence",
        "",
        f"State: `{report['state']}`",
        f"Arms: `{report['arms']}`",
        f"Seed counts: `{report['seed_counts']}`",
        f"Outcome: `{report['t4_4_outcome']}`",
        f"Result settled: `{report['t4_4_result_settled']}`",
        f"Positive result ready: `{report['positive_result_ready']}`",
        f"Negative result ready: `{report['negative_result_ready']}`",
        f"Convergence curves ready: `{report['convergence_curves_ready']}`",
        f"Budget ratio ready: `{report['budget_ratio_ready']}`",
        f"Plateau consistency ready: `{report['plateau_consistency_ready']}`",
        "",
        "## Budget Ratio",
        "",
    ]
    for key, value in report["budget_ratio"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outcome Reasons", ""])
    for reason in report.get("outcome_reasons", []):
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _render_csv(report: Mapping[str, Any]) -> str:
    import io

    handle = io.StringIO()
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "arm",
            "seed",
            "environment",
            "proposal_id",
            "verdict",
            "gpu_hours",
            "baseline_primary_metric",
            "candidate_primary_metric",
            "delta_primary_metric",
            "normalized_progress",
            "report_path",
        ],
    )
    writer.writeheader()
    for record in report.get("records", []):
        if not isinstance(record, Mapping):
            continue
        writer.writerow({key: record.get(key, "") for key in writer.fieldnames or []})
    return handle.getvalue()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARM=PATH")
    arm, path = value.split("=", 1)
    if arm not in REQUIRED_ARMS or not path:
        raise argparse.ArgumentTypeError("expected one of prior,cold_start,shuffled_prior=PATH")
    return arm, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate T4.4 prior convergence report")
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--arm", action="append", type=_parse_arm, default=[])
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--primitive-materialization-gate-manifest", type=Path)
    run.add_argument("--goal-config", type=Path)
    run.add_argument("--target-progress-delta", type=float)
    run.add_argument("--progress-threshold", type=float)
    run.add_argument("--min-seed-count", type=int)
    run.add_argument("--plateau-tolerance", type=float)
    run.add_argument("--speedup-target", type=float)
    args = parser.parse_args(argv)
    if args.command == "run":
        arms: dict[str, list[Path]] = defaultdict(list)
        for arm, path in args.arm:
            arms[arm].append(path)
        manifest = run_t44_prior_convergence(
            archive_db=args.archive_db,
            output_root=args.output_root,
            arms=arms,
            cas_root=args.cas_root,
            primitive_materialization_gate_manifest=args.primitive_materialization_gate_manifest,
            goal_config=args.goal_config,
            target_progress_delta=args.target_progress_delta,
            progress_threshold=args.progress_threshold,
            min_seed_count=args.min_seed_count,
            plateau_tolerance=args.plateau_tolerance,
            speedup_target=args.speedup_target,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise T44PriorConvergenceError("T44_PRIOR_CONVERGENCE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
