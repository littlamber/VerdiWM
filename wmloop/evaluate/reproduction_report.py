"""Generate a fail-closed M0 baseline reproduction report.

The M0 baseline summary proves that the frozen upstream evaluator completed.
The stricter T0.3 DoD also asks for a PSNR +/-0.5dB comparison against an
official reference.  This module records that comparison when a reference file
is supplied, and records a non-passing "reference unavailable" state otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.evaluate.launch import BaselineLaunchPlan, load_baseline_launch_plan


class BaselineReproductionReportError(RuntimeError):
    """A baseline reproduction report could not be produced safely."""


_METRICS = ("mse", "masked_mse", "psnr", "ssim")
_PROTOCOL_SPLITS = ("ind", "ood")


def generate_baseline_reproduction_report(
    *,
    launch_plan_path: Path,
    summary_path: Path,
    output_root: Path,
    official_reference_path: Path | None = None,
    checkpoint_step_audit_path: Path | None = None,
    psnr_tolerance: float = 0.5,
    expected_environment_count: int = 8,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Write a durable M0 T0.3 report without inventing official metrics."""

    if not math.isfinite(psnr_tolerance) or psnr_tolerance <= 0:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TOLERANCE_INVALID")
    if expected_environment_count < 1:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_ENVIRONMENT_COUNT_INVALID")
    plan = load_baseline_launch_plan(launch_plan_path, repo_root=repo_root)
    summary = _load_json_mapping(summary_path, "BASELINE_REPRODUCTION_SUMMARY_INVALID")
    _verify_summary_matches_plan(summary, plan)
    measured_by_environment = _measured_environment_metrics(summary, plan)
    measured_by_protocol_split = _measured_protocol_split_metrics(summary, plan)
    reference = _load_official_reference(official_reference_path) if official_reference_path is not None else None
    checkpoint_audit = _load_checkpoint_step_audit(checkpoint_step_audit_path) if checkpoint_step_audit_path is not None else None
    comparisons, reference_status, delta_evaluable, delta_pass, comparison_basis = _compare_reference(
        measured_by_environment,
        measured_by_protocol_split,
        reference=reference,
        psnr_tolerance=psnr_tolerance,
    )
    archive_status = _archive_status(plan, archive_db)
    environment_count_pass = len(measured_by_environment) == expected_environment_count
    checkpoint_audit_pass = checkpoint_audit is None or checkpoint_audit.get("state") == "ready"
    warnings = _warnings(
        reference_status=reference_status,
        checkpoint_audit=checkpoint_audit,
        environment_count_pass=environment_count_pass,
        expected_environment_count=expected_environment_count,
        actual_environment_count=len(measured_by_environment),
        archive_status=archive_status,
    )
    strict_t03_pass = bool(
        environment_count_pass
        and archive_status["generation_zero_archive_pass"]
        and checkpoint_audit_pass
        and delta_evaluable
        and delta_pass
    )
    state = _state(
        reference_status=reference_status,
        delta_evaluable=delta_evaluable,
        delta_pass=delta_pass,
        environment_count_pass=environment_count_pass,
        archive_pass=bool(archive_status["generation_zero_archive_pass"]),
        checkpoint_audit_pass=checkpoint_audit_pass,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-reproduction-report",
        "state": state,
        "strict_m0_t03_pass": strict_t03_pass,
        "source_summary_path": str(Path(summary_path).resolve()),
        "launch_plan_path": str(Path(launch_plan_path).resolve()),
        "run_root": str(plan.run_root),
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
        "source_revision": plan.source_revision,
        "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
        "expected_environment_count": expected_environment_count,
        "environment_count": len(measured_by_environment),
        "environment_count_pass": environment_count_pass,
        "task_count": len(plan.tasks),
        "metrics_ready_tasks": int(summary["metrics_ready_tasks"]),
        "official_reference": _reference_document(reference, official_reference_path),
        "checkpoint_step_audit": _checkpoint_audit_document(checkpoint_audit, checkpoint_step_audit_path),
        "psnr_tolerance": psnr_tolerance,
        "official_psnr_delta_evaluable": delta_evaluable,
        "official_psnr_delta_pass": delta_pass,
        "archive": archive_status,
        "overall_unweighted": _metric_mapping(summary["aggregate_metrics"]["overall_unweighted"], "BASELINE_REPRODUCTION_AGGREGATE_INVALID"),  # type: ignore[index]
        "by_cohort_unweighted": _cohort_metrics(summary),
        "by_environment_unweighted": measured_by_environment,
        "by_protocol_split_weighted": measured_by_protocol_split,
        "comparison_basis": comparison_basis,
        "comparisons": comparisons,
        "warnings": warnings,
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(
        report=report,
        markdown=markdown,
        output_root=output_root,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_json_mapping(path: Path, code: str) -> Mapping[str, Any]:
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            raise BaselineReproductionReportError(code)
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineReproductionReportError(code) from exc
    if not isinstance(payload, Mapping):
        raise BaselineReproductionReportError(code)
    return payload


def _verify_summary_matches_plan(summary: Mapping[str, Any], plan: BaselineLaunchPlan) -> None:
    if summary.get("schema_version") != 1 or summary.get("artifact_type") != "acwm-m0-baseline-summary":
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_SUMMARY_INVALID")
    if summary.get("state") != "ready" or summary.get("ready_for_archive") is not True:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_SUMMARY_NOT_READY")
    expected = {
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
        "source_revision": plan.source_revision,
        "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
        "total_tasks": len(plan.tasks),
        "metrics_ready_tasks": len(plan.tasks),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise BaselineReproductionReportError(f"BASELINE_REPRODUCTION_SUMMARY_PLAN_MISMATCH:{key}")
    if summary.get("state_counts") != {"completed": len(plan.tasks)}:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_STATE_COUNTS_INVALID")


def _measured_environment_metrics(summary: Mapping[str, Any], plan: BaselineLaunchPlan) -> dict[str, dict[str, float]]:
    aggregate = summary.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_AGGREGATE_INVALID")
    raw_by_environment = aggregate.get("by_environment_unweighted")
    if not isinstance(raw_by_environment, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_AGGREGATE_INVALID")
    expected_environments = {task.environment for task in plan.tasks}
    if set(raw_by_environment) != expected_environments:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_ENVIRONMENT_METRICS_MISMATCH")
    return {
        environment: _metric_mapping(raw_by_environment[environment], "BASELINE_REPRODUCTION_AGGREGATE_INVALID")
        for environment in sorted(expected_environments)
    }


def _cohort_metrics(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    aggregate = summary.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_AGGREGATE_INVALID")
    raw = aggregate.get("by_cohort_unweighted")
    if not isinstance(raw, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_AGGREGATE_INVALID")
    return {str(cohort): _metric_mapping(metrics, "BASELINE_REPRODUCTION_AGGREGATE_INVALID") for cohort, metrics in sorted(raw.items())}


def _measured_protocol_split_metrics(summary: Mapping[str, Any], plan: BaselineLaunchPlan) -> dict[str, dict[str, dict[str, object]]]:
    raw = summary.get("task_metrics")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
    expected_environments = {task.environment for task in plan.tasks}
    buckets: dict[str, dict[str, list[dict[str, object]]]] = {
        environment: {split: [] for split in _PROTOCOL_SPLITS}
        for environment in expected_environments
    }
    for item in raw:
        if not isinstance(item, Mapping):
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        environment = item.get("environment")
        if not isinstance(environment, str) or environment not in expected_environments:
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        split = _protocol_split_key(item.get("split"), item.get("cohort"))
        if split is None:
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        metrics = _metric_mapping(item.get("metrics"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        window_count = _positive_count(item.get("window_count"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        trajectory_count = _positive_count(item.get("trajectory_count"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        cohort = item.get("cohort")
        source_split = item.get("split")
        if not isinstance(cohort, str) or not cohort or not isinstance(source_split, str) or not source_split:
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
        buckets[environment][split].append(
            {
                "cohort": cohort,
                "source_split": source_split,
                "window_count": window_count,
                "trajectory_count": trajectory_count,
                "metrics": metrics,
            }
        )
    result: dict[str, dict[str, dict[str, object]]] = {}
    for environment in sorted(expected_environments):
        result[environment] = {}
        for split in _PROTOCOL_SPLITS:
            rows = buckets[environment][split]
            if not rows:
                raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INCOMPLETE")
            result[environment][split] = _weighted_protocol_split(rows)
    return result


def _protocol_split_key(split: object, cohort: object) -> str | None:
    if split == "ind_test":
        return "ind"
    if split == "ood_test":
        return "ood"
    if cohort == "ind_dev" or cohort == "ind_accept":
        return "ind"
    if cohort == "ood_accept":
        return "ood"
    return None


def _positive_count(value: object, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BaselineReproductionReportError(code)
    return value


def _weighted_protocol_split(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_windows = sum(_positive_count(row.get("window_count"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID") for row in rows)
    total_trajectories = sum(_positive_count(row.get("trajectory_count"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID") for row in rows)
    weighted_metrics: dict[str, float] = {}
    for metric in _METRICS:
        weighted_metrics[metric] = sum(
            _metric_mapping(row.get("metrics"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID")[metric]
            * _positive_count(row.get("window_count"), "BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
            for row in rows
        ) / total_windows
    return {
        "aggregation": "window_count_weighted_task_metric_average",
        "cohorts": [str(row["cohort"]) for row in rows],
        "source_splits": sorted({str(row["source_split"]) for row in rows}),
        "trajectory_count": total_trajectories,
        "window_count": total_windows,
        "metrics": weighted_metrics,
    }


def _metric_mapping(payload: object, code: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise BaselineReproductionReportError(code)
    result: dict[str, float] = {}
    for metric in _METRICS:
        value = payload.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise BaselineReproductionReportError(code)
        result[metric] = float(value)
    return result


def _load_official_reference(path: Path) -> dict[str, object]:
    payload = _load_json_mapping(path, "BASELINE_REPRODUCTION_REFERENCE_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") not in {
        "acwm-m0-baseline-official-reference",
        "acwm-m0-official-reference",
    }:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    raw_environments = payload.get("environments")
    if not isinstance(raw_environments, Mapping) or not raw_environments:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    environments: dict[str, dict[str, object]] = {}
    reference_type: str | None = None
    for environment, item in raw_environments.items():
        if not isinstance(environment, str) or not environment:
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
        normalized = _normalize_reference_environment(item)
        item_type = "protocol_split" if "splits" in normalized else "environment_unweighted"
        if reference_type is None:
            reference_type = item_type
        elif reference_type != item_type:
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
        environments[environment] = normalized
    source = payload.get("source")
    if source is not None and (not isinstance(source, str) or not source):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    return {
        "path": str(Path(path).resolve()),
        "source": source if source is not None else str(Path(path).resolve()),
        "reference_type": reference_type,
        "environments": environments,
    }


def _load_checkpoint_step_audit(path: Path) -> dict[str, object]:
    payload = _load_json_mapping(path, "BASELINE_REPRODUCTION_CHECKPOINT_AUDIT_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "acwm-m0-checkpoint-step-audit":
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_CHECKPOINT_AUDIT_INVALID")
    state = payload.get("state")
    mismatch_count = payload.get("mismatch_count")
    if not isinstance(state, str) or not isinstance(mismatch_count, int) or isinstance(mismatch_count, bool) or mismatch_count < 0:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_CHECKPOINT_AUDIT_INVALID")
    return {
        "path": str(Path(path).resolve()),
        "state": state,
        "expected_step": payload.get("expected_step"),
        "mismatch_count": mismatch_count,
        "records": payload.get("records", []),
    }


def _normalize_reference_environment(item: object) -> dict[str, object]:
    if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
        return {"psnr": float(item)}
    if not isinstance(item, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    value = item.get("psnr")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return {"psnr": float(value)}
    splits: dict[str, dict[str, float]] = {}
    for raw_split, split in (("ind", "ind"), ("ood", "ood"), ("ind_test", "ind"), ("ood_test", "ood")):
        if raw_split not in item:
            continue
        split_item = item[raw_split]
        split_value = split_item.get("psnr") if isinstance(split_item, Mapping) else split_item
        if not isinstance(split_value, (int, float)) or isinstance(split_value, bool) or not math.isfinite(float(split_value)):
            raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
        splits[split] = {"psnr": float(split_value)}
    if not splits:
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    return {"splits": splits}


def _compare_reference(
    measured_by_environment: Mapping[str, Mapping[str, float]],
    measured_by_protocol_split: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    reference: Mapping[str, object] | None,
    psnr_tolerance: float,
) -> tuple[list[dict[str, object]], str, bool, bool, str]:
    comparisons: list[dict[str, object]] = []
    if reference is None:
        for environment, measured in measured_by_environment.items():
            comparisons.append(
                {
                    "environment": environment,
                    "split": None,
                    "measured_psnr": measured["psnr"],
                    "official_psnr": None,
                    "delta_psnr": None,
                    "within_tolerance": False,
                    "status": "reference_unavailable",
                }
            )
        return comparisons, "unavailable", False, False, "environment_unweighted"
    raw_reference = reference.get("environments")
    if not isinstance(raw_reference, Mapping):
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    reference_type = reference.get("reference_type")
    if reference_type == "protocol_split":
        return _compare_protocol_split_reference(
            measured_by_protocol_split,
            raw_reference=raw_reference,
            psnr_tolerance=psnr_tolerance,
        )
    if reference_type != "environment_unweighted":
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_REFERENCE_INVALID")
    missing = sorted(set(measured_by_environment) - set(raw_reference))
    extra = sorted(set(raw_reference) - set(measured_by_environment))
    complete = not missing and not extra
    all_pass = complete
    for environment, measured in measured_by_environment.items():
        ref_item = raw_reference.get(environment)
        official_psnr = ref_item.get("psnr") if isinstance(ref_item, Mapping) else None
        if official_psnr is None:
            comparisons.append(
                {
                    "environment": environment,
                    "split": None,
                    "measured_psnr": measured["psnr"],
                    "official_psnr": None,
                    "delta_psnr": None,
                    "within_tolerance": False,
                    "status": "reference_missing",
                }
            )
            all_pass = False
            continue
        delta = measured["psnr"] - float(official_psnr)
        within_tolerance = abs(delta) <= psnr_tolerance
        all_pass = all_pass and within_tolerance
        comparisons.append(
            {
                "environment": environment,
                "split": None,
                "measured_psnr": measured["psnr"],
                "official_psnr": float(official_psnr),
                "delta_psnr": delta,
                "within_tolerance": within_tolerance,
                "status": "pass" if within_tolerance else "fail",
            }
        )
    if extra:
        for environment in extra:
            comparisons.append(
                {
                    "environment": environment,
                    "split": None,
                    "measured_psnr": None,
                    "official_psnr": raw_reference[environment]["psnr"],  # type: ignore[index]
                    "delta_psnr": None,
                    "within_tolerance": False,
                    "status": "unexpected_reference_environment",
                }
            )
    if not complete:
        return comparisons, "incomplete", False, False, "environment_unweighted"
    return comparisons, "available", True, bool(all_pass), "environment_unweighted"


def _compare_protocol_split_reference(
    measured_by_protocol_split: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    raw_reference: Mapping[str, object],
    psnr_tolerance: float,
) -> tuple[list[dict[str, object]], str, bool, bool, str]:
    comparisons: list[dict[str, object]] = []
    missing = sorted(set(measured_by_protocol_split) - set(raw_reference))
    extra = sorted(set(raw_reference) - set(measured_by_protocol_split))
    complete = not missing and not extra
    all_pass = complete
    for environment, measured_splits in measured_by_protocol_split.items():
        ref_item = raw_reference.get(environment)
        ref_splits = ref_item.get("splits") if isinstance(ref_item, Mapping) else None
        if not isinstance(ref_splits, Mapping):
            ref_splits = {}
            complete = False
            all_pass = False
        for split in _PROTOCOL_SPLITS:
            measured = measured_splits.get(split)
            metrics = measured.get("metrics") if isinstance(measured, Mapping) else None
            measured_psnr = metrics.get("psnr") if isinstance(metrics, Mapping) else None
            ref_split = ref_splits.get(split)
            official_psnr = ref_split.get("psnr") if isinstance(ref_split, Mapping) else None
            if not isinstance(measured_psnr, (int, float)) or isinstance(measured_psnr, bool):
                raise BaselineReproductionReportError("BASELINE_REPRODUCTION_TASK_METRICS_INVALID")
            if official_psnr is None:
                comparisons.append(
                    {
                        "environment": environment,
                        "split": split,
                        "measured_psnr": float(measured_psnr),
                        "official_psnr": None,
                        "delta_psnr": None,
                        "within_tolerance": False,
                        "status": "reference_split_missing",
                    }
                )
                complete = False
                all_pass = False
                continue
            delta = float(measured_psnr) - float(official_psnr)
            within_tolerance = abs(delta) <= psnr_tolerance
            all_pass = all_pass and within_tolerance
            comparisons.append(
                {
                    "environment": environment,
                    "split": split,
                    "measured_psnr": float(measured_psnr),
                    "official_psnr": float(official_psnr),
                    "delta_psnr": delta,
                    "within_tolerance": within_tolerance,
                    "status": "pass" if within_tolerance else "fail",
                }
            )
    for environment in extra:
        comparisons.append(
            {
                "environment": environment,
                "split": None,
                "measured_psnr": None,
                "official_psnr": None,
                "delta_psnr": None,
                "within_tolerance": False,
                "status": "unexpected_reference_environment",
            }
        )
    if not complete:
        return comparisons, "incomplete", False, False, "protocol_split_weighted"
    return comparisons, "available", True, bool(all_pass), "protocol_split_weighted"


def _archive_status(plan: BaselineLaunchPlan, archive_db: Path | None) -> dict[str, object]:
    if archive_db is None:
        return {
            "archive_db": None,
            "status": "not_checked",
            "baseline_count": 0,
            "expected_baseline_count": len({task.environment for task in plan.tasks}),
            "generation_zero_archive_pass": False,
            "missing_environments": sorted({task.environment for task in plan.tasks}),
            "extra_environments": [],
        }
    archive = ArchiveStore(archive_db)
    expected = {task.environment for task in plan.tasks}
    actual = {record.environment for record in archive.list_baselines()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "archive_db": str(Path(archive_db).resolve()),
        "status": "ready" if not missing and not extra else "mismatch",
        "baseline_count": len(actual),
        "expected_baseline_count": len(expected),
        "generation_zero_archive_pass": not missing and not extra,
        "missing_environments": missing,
        "extra_environments": extra,
    }


def _reference_document(reference: Mapping[str, object] | None, path: Path | None) -> dict[str, object]:
    if reference is None:
        return {"status": "unavailable", "path": str(Path(path).resolve()) if path is not None else None, "source": None}
    return {
        "status": "available",
        "path": reference["path"],
        "source": reference["source"],
        "reference_type": reference["reference_type"],
        "environment_count": len(reference["environments"]),  # type: ignore[arg-type]
    }


def _checkpoint_audit_document(checkpoint_audit: Mapping[str, object] | None, path: Path | None) -> dict[str, object]:
    if checkpoint_audit is None:
        return {"status": "not_checked", "path": str(Path(path).resolve()) if path is not None else None}
    return {
        "status": "ready" if checkpoint_audit["state"] == "ready" else "failed",
        "path": checkpoint_audit["path"],
        "state": checkpoint_audit["state"],
        "expected_step": checkpoint_audit["expected_step"],
        "mismatch_count": checkpoint_audit["mismatch_count"],
    }


def _warnings(
    *,
    reference_status: str,
    checkpoint_audit: Mapping[str, object] | None,
    environment_count_pass: bool,
    expected_environment_count: int,
    actual_environment_count: int,
    archive_status: Mapping[str, object],
) -> list[str]:
    warnings: list[str] = []
    if reference_status == "unavailable":
        warnings.append(
            "No official per-environment PSNR reference was supplied; official-code reproduction completed, PSNR +/-0.5dB comparison is not evaluable."
        )
    elif reference_status == "incomplete":
        warnings.append("Official PSNR reference does not cover exactly the measured environments; strict PSNR delta comparison is not evaluable.")
    if not environment_count_pass:
        warnings.append(f"Measured environment count is {actual_environment_count}, expected {expected_environment_count} for strict M0 T0.3.")
    if archive_status["generation_zero_archive_pass"] is not True:
        warnings.append("Generation-zero archive baselines do not exactly match the launch-plan environments.")
    if checkpoint_audit is not None and checkpoint_audit.get("state") != "ready":
        warnings.append(
            f"Checkpoint step audit is not ready: state={checkpoint_audit.get('state')}, mismatch_count={checkpoint_audit.get('mismatch_count')}."
        )
    return warnings


def _state(
    *,
    reference_status: str,
    delta_evaluable: bool,
    delta_pass: bool,
    environment_count_pass: bool,
    archive_pass: bool,
    checkpoint_audit_pass: bool,
) -> str:
    if not environment_count_pass:
        return "environment_count_mismatch"
    if not archive_pass:
        return "archive_mismatch"
    if not checkpoint_audit_pass:
        return "checkpoint_step_mismatch"
    if reference_status == "unavailable":
        return "reference_unavailable"
    if reference_status == "incomplete" or not delta_evaluable:
        return "reference_incomplete"
    return "ready" if delta_pass else "delta_out_of_tolerance"


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Baseline Reproduction Report",
        "",
        f"State: `{report['state']}`",
        f"Strict T0.3 pass: `{str(report['strict_m0_t03_pass']).lower()}`",
        f"Official PSNR delta evaluable: `{str(report['official_psnr_delta_evaluable']).lower()}`",
        f"PSNR tolerance: `{report['psnr_tolerance']}` dB",
        "",
        "## Provenance",
        "",
        f"- Source summary: `{report['source_summary_path']}`",
        f"- Launch plan: `{report['launch_plan_path']}`",
        f"- Vendor/source revision: `{report['source_revision']}`",
        f"- Evaluator freeze: `{report['evaluator_freeze_sha256']}`",
        f"- Dataset freeze: `{report['dataset_freeze_sha256']}`",
        f"- Held-out protocol: `{report['heldout_protocol_sha256']}`",
    ]
    official_reference = report["official_reference"]
    if official_reference["status"] == "available":
        lines.extend(
            [
                f"- Official reference source: `{official_reference['source']}`",
                f"- Official reference type: `{official_reference['reference_type']}`",
            ]
        )
    checkpoint_step_audit = report["checkpoint_step_audit"]
    if checkpoint_step_audit["status"] != "not_checked":
        lines.append(
            f"- Checkpoint step audit: `{checkpoint_step_audit['state']}` "
            f"(mismatch_count `{checkpoint_step_audit['mismatch_count']}`)"
        )
    lines.extend(["", "## PSNR Delta", ""])
    if report.get("comparison_basis") == "protocol_split_weighted":
        lines.extend(
            [
                "Comparison basis: `protocol_split_weighted` (`ind` = window-count weighted `ind_dev` + `ind_accept`; `ood` = `ood_accept`).",
                "",
                "| Environment | Split | Measured PSNR | Official PSNR | Delta | Status |",
                "|:--|:--|--:|--:|--:|:--|",
            ]
        )
    else:
        lines.extend(
            [
                "Comparison basis: `environment_unweighted`.",
                "",
                "| Environment | Measured PSNR | Official PSNR | Delta | Status |",
                "|:--|--:|--:|--:|:--|",
            ]
        )
    for item in report["comparisons"]:
        official = _format_optional_float(item["official_psnr"])
        delta = _format_optional_float(item["delta_psnr"])
        measured = _format_optional_float(item["measured_psnr"])
        if report.get("comparison_basis") == "protocol_split_weighted":
            lines.append(f"| {item['environment']} | {item['split']} | {measured} | {official} | {delta} | {item['status']} |")
        else:
            lines.append(f"| {item['environment']} | {measured} | {official} | {delta} | {item['status']} |")
    lines.extend(["", "## Aggregate Metrics", ""])
    overall = report["overall_unweighted"]
    lines.append(
        f"Overall unweighted: MSE `{overall['mse']:.6f}`, masked-MSE `{overall['masked_mse']:.6f}`, PSNR `{overall['psnr']:.4f}`, SSIM `{overall['ssim']:.4f}`."
    )
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _format_optional_float(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    markdown: str,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "baseline-reproduction.json", report_bytes)
        _write_bytes_atomic(temporary / "baseline-reproduction.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("baseline_reproduction_json", report_bytes, "application/json"),
                ("baseline_reproduction_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-baseline-reproduction-manifest",
            "state": report["state"],
            "strict_m0_t03_pass": report["strict_m0_t03_pass"],
            "report_path": str(destination / "baseline-reproduction.json"),
            "markdown_path": str(destination / "baseline-reproduction.md"),
            "cas_refs": cas_refs,
            "warnings": report["warnings"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BaselineReproductionReportError("BASELINE_REPRODUCTION_OUTPUT_EXISTS")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate an M0 T0.3 reproduction report")
    generate.add_argument("--launch-plan", type=Path, required=True)
    generate.add_argument("--summary", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--official-reference", type=Path)
    generate.add_argument("--checkpoint-step-audit", type=Path)
    generate.add_argument("--psnr-tolerance", type=float, default=0.5)
    generate.add_argument("--expected-environment-count", type=int, default=8)
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    generate.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_baseline_reproduction_report(
            launch_plan_path=args.launch_plan,
            summary_path=args.summary,
            output_root=args.output_root,
            official_reference_path=args.official_reference,
            checkpoint_step_audit_path=args.checkpoint_step_audit,
            psnr_tolerance=args.psnr_tolerance,
            expected_environment_count=args.expected_environment_count,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            repo_root=args.repo_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
