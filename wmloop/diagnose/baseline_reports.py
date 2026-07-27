"""Generate generation-zero M1 failure reports from archived M0 baselines.

This module is intentionally explicit about its evidence boundary.  The M0
baseline summary contains official cohort metrics, not raw long-horizon probe
frames or an inverse-dynamics cache.  The generated reports are therefore a
metric projection suitable for exercising the M1 contract and downstream
archive plumbing, with the projection mode recorded in the batch manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, BaselineRecord, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.diagnose.diagnoser import DiagnosisThresholds, build_failure_report
from wmloop.diagnose.probe_registry import build_verdict_evidence, load_probe_registry
from wmloop.evaluate.launch import BaselineLaunchPlan, load_baseline_launch_plan


class BaselineFailureReportError(RuntimeError):
    """A generation-zero failure-report batch could not be produced safely."""


_REQUIRED_COHORTS = ("ind_dev", "ind_accept", "ood_accept")


def generate_baseline_failure_reports(
    *,
    launch_plan_path: Path,
    summary_path: Path,
    archive_db: Path,
    output_root: Path,
    goal_config: Path,
    diagnosis_config: Path,
    probe_registry_path: Path,
    repo_root: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write one schema-valid failure report and verdict evidence per env."""

    plan = load_baseline_launch_plan(launch_plan_path, repo_root=repo_root)
    summary = _load_summary(summary_path)
    _verify_summary_matches_plan(summary, plan)
    goal = _load_goal(goal_config)
    thresholds, diagnosis_metadata = _load_diagnosis_config(diagnosis_config)
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    registry = load_probe_registry(probe_registry_path, root=Path(repo_root).resolve() if repo_root is not None else None)
    baseline_by_env = _baseline_records_by_environment(archive.list_baselines())
    metrics_by_env = _task_metrics_by_environment(summary)
    goal_envs = tuple(str(item) for item in goal["envs"])
    if set(goal_envs) - set(metrics_by_env):
        missing = ",".join(sorted(set(goal_envs) - set(metrics_by_env)))
        raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_ENVIRONMENT_MISSING:{missing}")
    if set(goal_envs) - set(baseline_by_env):
        missing = ",".join(sorted(set(goal_envs) - set(baseline_by_env)))
        raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_ARCHIVE_MISSING:{missing}")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    reports: list[dict[str, object]] = []
    warnings = [
        "m0_baseline_metric_projection: horizon/action probes are derived from aggregate M0 metrics, not raw probe execution",
        "inverse_dynamics_head_untrained: action_following fields are neutral placeholders with low_confidence=true",
    ]
    try:
        (temporary / "failure_reports").mkdir(mode=0o700, parents=True)
        (temporary / "verdict_evidence").mkdir(mode=0o700)
        for environment in goal_envs:
            report = _build_environment_report(
                environment=environment,
                model_ref=baseline_by_env[environment].model_ref,
                goal_id=str(goal["goal_id"]),
                horizons=tuple(int(item) for item in goal["horizons"]),
                cohort_metrics=metrics_by_env[environment],
                thresholds=thresholds,
                inverse_dynamics_r2=float(diagnosis_metadata["projection"]["inverse_dynamics_r2"]),
            )
            evidence = build_verdict_evidence(report, registry)
            report_bytes = _canonical_json_bytes(report)
            evidence_bytes = _canonical_json_bytes(evidence)
            report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
            evidence_ref = cas.put_bytes(evidence_bytes, media_type="application/json").uri
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(evidence_ref)
            report_path = temporary / "failure_reports" / f"{environment}.json"
            evidence_path = temporary / "verdict_evidence" / f"{environment}.json"
            _write_bytes_atomic(report_path, report_bytes)
            _write_bytes_atomic(evidence_path, evidence_bytes)
            reports.append(
                {
                    "environment": environment,
                    "failure_report_path": str(destination / "failure_reports" / f"{environment}.json"),
                    "failure_report_ref": report_ref,
                    "verdict_evidence_path": str(destination / "verdict_evidence" / f"{environment}.json"),
                    "verdict_evidence_ref": evidence_ref,
                    "dominant_failure": report["dominant_failure"],
                    "dominant_failure_candidates": report["dominant_failure_candidates"],
                    "source_task_ids": [metrics_by_env[environment][cohort]["task_id"] for cohort in _REQUIRED_COHORTS],
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m1-baseline-failure-report-batch",
            "state": "ready",
            "diagnostic_mode": diagnosis_metadata["mode"],
            "source_summary_path": str(Path(summary_path).resolve()),
            "launch_plan_path": str(Path(launch_plan_path).resolve()),
            "archive_db": str(Path(archive_db).resolve()),
            "cas_root": str((cas_root if cas_root is not None else Path(archive_db).resolve().parent).resolve()),
            "goal_id": goal["goal_id"],
            "horizons": list(goal["horizons"]),
            "thresholds": asdict(thresholds),
            "projection": diagnosis_metadata["projection"],
            "report_count": len(reports),
            "schema_valid_reports": len(reports),
            "reports": reports,
            "warnings": warnings,
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


def _build_environment_report(
    *,
    environment: str,
    model_ref: str,
    goal_id: str,
    horizons: Sequence[int],
    cohort_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: DiagnosisThresholds,
    inverse_dynamics_r2: float,
) -> dict[str, object]:
    if len(horizons) < 2:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_HORIZONS_INVALID")
    ordered_horizons = tuple(sorted(int(item) for item in horizons))
    if len(set(ordered_horizons)) != len(ordered_horizons) or ordered_horizons[0] <= 0:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_HORIZONS_INVALID")
    ind_psnr = _mean_metric(cohort_metrics, ("ind_dev", "ind_accept"), "psnr")
    ood_psnr = _metric(cohort_metrics, "ood_accept", "psnr")
    gap = max(0.0, ind_psnr - ood_psnr)
    psnr_by_horizon = {
        horizon: ind_psnr - gap * (index / (len(ordered_horizons) - 1))
        for index, horizon in enumerate(ordered_horizons)
    }
    frame_count = max(16, 12)
    per_frame_psnr = {
        frame: ind_psnr - gap * (frame / (frame_count - 1))
        for frame in range(frame_count)
    }
    horizon_span = ordered_horizons[-1] - ordered_horizons[0]
    ind_auc = ind_psnr * horizon_span
    ood_auc = ood_psnr * horizon_span
    report = build_failure_report(
        env=environment,
        model_ref=model_ref,
        round_number=0,
        goal_id=goal_id,
        psnr_by_horizon=psnr_by_horizon,
        per_frame_psnr=per_frame_psnr,
        appearance_low_motion_ssim=_metric(cohort_metrics, "ood_accept", "ssim"),
        action_following_accuracy=1.0,
        no_action_delta_psnr=1.0,
        ind_auc=ind_auc,
        ood_auc=ood_auc,
        worst_ood_condition="ood_accept",
        evidence_frames=(),
        thresholds=thresholds,
        inverse_dynamics_r2=inverse_dynamics_r2,
    )
    validate_document("failure_report", report)
    return report


def _load_summary(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_SUMMARY_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_SUMMARY_INVALID")
    return payload


def _verify_summary_matches_plan(summary: Mapping[str, Any], plan: BaselineLaunchPlan) -> None:
    if summary.get("schema_version") != 1 or summary.get("artifact_type") != "acwm-m0-baseline-summary":
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_SUMMARY_INVALID")
    if summary.get("state") != "ready" or summary.get("ready_for_archive") is not True:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_SUMMARY_NOT_READY")
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
            raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_SUMMARY_PLAN_MISMATCH:{key}")
    if summary.get("state_counts") != {"completed": len(plan.tasks)}:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_SUMMARY_STATE_COUNTS_INVALID")


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_GOAL_INVALID") from exc
    if not isinstance(payload.get("horizons"), list) or len(payload["horizons"]) < 2:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_GOAL_INVALID")
    return payload


def _load_diagnosis_config(path: Path) -> tuple[DiagnosisThresholds, Mapping[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-diagnosis-thresholds"
        or not isinstance(payload.get("mode"), str)
        or not isinstance(payload.get("thresholds"), Mapping)
        or not isinstance(payload.get("projection"), Mapping)
    ):
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID")
    thresholds = DiagnosisThresholds(**{key: float(value) for key, value in payload["thresholds"].items()})
    projection = payload["projection"]
    if not isinstance(projection.get("inverse_dynamics_r2"), (int, float)) or isinstance(projection.get("inverse_dynamics_r2"), bool):
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID")
    return thresholds, payload


def _baseline_records_by_environment(records: Sequence[BaselineRecord]) -> dict[str, BaselineRecord]:
    grouped: dict[str, BaselineRecord] = {}
    for record in records:
        if record.environment in grouped:
            raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_ARCHIVE_DUPLICATE:{record.environment}")
        grouped[record.environment] = record
    return grouped


def _task_metrics_by_environment(summary: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    raw = summary.get("task_metrics")
    if not isinstance(raw, list):
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_TASK_METRICS_INVALID")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("environment"), str) or not isinstance(item.get("cohort"), str):
            raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_TASK_METRICS_INVALID")
        environment = str(item["environment"])
        cohort = str(item["cohort"])
        if cohort not in _REQUIRED_COHORTS:
            continue
        by_cohort = grouped.setdefault(environment, {})
        if cohort in by_cohort:
            raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_DUPLICATE_COHORT:{environment}:{cohort}")
        _validate_metric_payload(item)
        by_cohort[cohort] = item
    for environment, cohorts in grouped.items():
        missing = [cohort for cohort in _REQUIRED_COHORTS if cohort not in cohorts]
        if missing:
            raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_COHORT_MISSING:{environment}:{','.join(missing)}")
    return grouped


def _validate_metric_payload(item: Mapping[str, Any]) -> None:
    metrics = item.get("metrics")
    if not isinstance(metrics, Mapping):
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_TASK_METRICS_INVALID")
    for key in ("mse", "masked_mse", "psnr", "ssim"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_TASK_METRICS_INVALID")
    task_id = item.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_TASK_METRICS_INVALID")


def _metric(cohort_metrics: Mapping[str, Mapping[str, Any]], cohort: str, metric: str) -> float:
    try:
        value = cohort_metrics[cohort]["metrics"][metric]  # type: ignore[index]
    except KeyError as exc:
        raise BaselineFailureReportError(f"BASELINE_FAILURE_REPORT_METRIC_MISSING:{cohort}:{metric}") from exc
    return float(value)


def _mean_metric(cohort_metrics: Mapping[str, Mapping[str, Any]], cohorts: Sequence[str], metric: str) -> float:
    values = [_metric(cohort_metrics, cohort, metric) for cohort in cohorts]
    return sum(values) / len(values)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BaselineFailureReportError("BASELINE_FAILURE_REPORT_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate generation-zero M1 reports from a ready M0 baseline")
    generate.add_argument("--launch-plan", type=Path, required=True)
    generate.add_argument("--summary", type=Path, required=True)
    generate.add_argument("--archive-db", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--diagnosis-config", type=Path, default=Path("configs/diagnose/acwm_m1_baseline_projection.json"))
    generate.add_argument("--probe-registry", type=Path, default=Path("configs/probes/acwm_v1.json"))
    generate.add_argument("--repo-root", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_baseline_failure_reports(
            launch_plan_path=args.launch_plan,
            summary_path=args.summary,
            archive_db=args.archive_db,
            output_root=args.output_root,
            goal_config=args.goal_config,
            diagnosis_config=args.diagnosis_config,
            probe_registry_path=args.probe_registry,
            repo_root=args.repo_root,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
