"""Emit schema-valid M1 failure reports from measured raw-probe coverage."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.diagnose.diagnoser import DiagnosisThresholds, _dominant_failure, _rank_failures, summarize_horizon_curve
from wmloop.diagnose.probe_registry import build_verdict_evidence, load_probe_registry


class RawFailureReportError(RuntimeError):
    """Measured raw failure-report emission failed closed."""


def generate_raw_failure_reports(
    *,
    coverage_report_path: Path,
    archive_db: Path,
    output_root: Path,
    goal_config: Path = Path("configs/goal/long_horizon_v1.yaml"),
    probe_registry_path: Path = Path("configs/probes/acwm_v1.json"),
    diagnosis_config_path: Path = Path("configs/diagnose/acwm_m1_measured_v1.json"),
    repo_root: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write failure reports only for coverage records that are contract-ready."""

    coverage = _load_coverage(coverage_report_path)
    goal = _load_goal(goal_config)
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    registry = load_probe_registry(probe_registry_path, root=Path(repo_root).resolve() if repo_root is not None else None)
    thresholds, diagnosis_mode = _load_diagnosis_thresholds(diagnosis_config_path)
    baseline_by_env = {record.environment: record for record in archive.list_baselines()}
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise RawFailureReportError("RAW_FAILURE_REPORT_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    reports: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    try:
        (temporary / "failure_reports").mkdir(mode=0o700, parents=True)
        (temporary / "verdict_evidence").mkdir(mode=0o700)
        for record in coverage["records"]:
            if not isinstance(record, Mapping) or not isinstance(record.get("environment"), str):
                raise RawFailureReportError("RAW_FAILURE_REPORT_COVERAGE_INVALID")
            environment = str(record["environment"])
            if record.get("failure_report_contract_state") != "ready":
                blocked.append(
                    {
                        "environment": environment,
                        "blockers": record.get("failure_report_blockers", []),
                        "confidence_warnings": record.get("probe_confidence_warnings", []),
                    }
                )
                continue
            baseline = baseline_by_env.get(environment)
            if baseline is None:
                raise RawFailureReportError(f"RAW_FAILURE_REPORT_BASELINE_MISSING:{environment}")
            report = _build_report(
                environment=environment,
                model_ref=baseline.model_ref,
                goal_id=str(goal["goal_id"]),
                coverage_record=record,
                thresholds=thresholds,
            )
            evidence = build_verdict_evidence(report, registry)
            report_bytes = _canonical_json_bytes(report)
            evidence_bytes = _canonical_json_bytes(evidence)
            report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
            evidence_ref = cas.put_bytes(evidence_bytes, media_type="application/json").uri
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(evidence_ref)
            _write_bytes_atomic(temporary / "failure_reports" / f"{environment}.json", report_bytes)
            _write_bytes_atomic(temporary / "verdict_evidence" / f"{environment}.json", evidence_bytes)
            reports.append(
                {
                    "environment": environment,
                    "failure_report_path": str(destination / "failure_reports" / f"{environment}.json"),
                    "failure_report_ref": report_ref,
                    "verdict_evidence_path": str(destination / "verdict_evidence" / f"{environment}.json"),
                    "verdict_evidence_ref": evidence_ref,
                    "dominant_failure": report["dominant_failure"],
                    "dominant_failure_candidates": report["dominant_failure_candidates"],
                    "confidence_warnings": record.get("probe_confidence_warnings", []),
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-raw-failure-report-batch",
            "state": "ready" if len(reports) == int(coverage["environment_count"]) else "partial",
            "diagnostic_mode": diagnosis_mode,
            "diagnosis_config_path": str(Path(diagnosis_config_path).resolve()),
            "coverage_report_path": str(Path(coverage_report_path).resolve()),
            "archive_db": str(Path(archive_db).resolve()),
            "cas_root": str((cas_root if cas_root is not None else Path(archive_db).resolve().parent).resolve()),
            "goal_id": goal["goal_id"],
            "thresholds": asdict(thresholds),
            "environment_count": coverage["environment_count"],
            "report_count": len(reports),
            "schema_valid_reports": len(reports),
            "blocked_count": len(blocked),
            "reports": reports,
            "blocked_records": blocked,
            "warnings": [
                "partial_raw_failure_report_batch: blocked environments were not fabricated",
                "low-confidence inverse dynamics remains encoded in report.action_following.low_confidence",
            ],
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


def _build_report(
    *,
    environment: str,
    model_ref: str,
    goal_id: str,
    coverage_record: Mapping[str, Any],
    thresholds: DiagnosisThresholds,
) -> dict[str, object]:
    coverage = coverage_record.get("probe_coverage")
    if not isinstance(coverage, Mapping):
        raise RawFailureReportError("RAW_FAILURE_REPORT_COVERAGE_INVALID")
    horizon = _mapping(coverage.get("horizon_curve"), "RAW_FAILURE_REPORT_HORIZON_INVALID")
    appearance = _mapping(coverage.get("appearance_drift"), "RAW_FAILURE_REPORT_APPEARANCE_INVALID")
    action = _mapping(coverage.get("action_following"), "RAW_FAILURE_REPORT_ACTION_INVALID")
    ood = _mapping(coverage.get("ood_profile"), "RAW_FAILURE_REPORT_OOD_INVALID")
    psnr_by_horizon = {int(key): float(value) for key, value in _mapping(horizon.get("psnr"), "RAW_FAILURE_REPORT_HORIZON_INVALID").items()}
    horizon_curve = summarize_horizon_curve(metric="psnr", observations=psnr_by_horizon)
    horizon_curve["segment_drift"] = _mapping(horizon.get("segment_drift"), "RAW_FAILURE_REPORT_SEGMENT_INVALID")
    appearance_ssim = _float(appearance.get("low_motion_ssim_64"), "RAW_FAILURE_REPORT_APPEARANCE_INVALID")
    action_accuracy = _float(action.get("inv_dyn_acc_perframe"), "RAW_FAILURE_REPORT_ACTION_INVALID")
    no_action_delta = _float(action.get("no_action_delta_psnr"), "RAW_FAILURE_REPORT_ACTION_INVALID")
    inverse_r2 = action.get("inverse_dynamics_r2")
    if inverse_r2 is not None:
        inverse_r2 = _float(inverse_r2, "RAW_FAILURE_REPORT_ACTION_INVALID")
    ind_auc = _float(ood.get("ind_auc"), "RAW_FAILURE_REPORT_OOD_INVALID")
    ood_auc = _float(ood.get("ood_auc"), "RAW_FAILURE_REPORT_OOD_INVALID")
    ood_gap = _float(ood.get("gap"), "RAW_FAILURE_REPORT_OOD_INVALID")
    condition = ood.get("worst_ood_condition")
    if not isinstance(condition, str) or not condition:
        raise RawFailureReportError("RAW_FAILURE_REPORT_OOD_INVALID")
    candidates = _rank_failures(
        drift_slope=float(horizon_curve["drift_slope"]),
        appearance_low_motion_ssim=appearance_ssim,
        action_following_accuracy=action_accuracy,
        no_action_delta_psnr=no_action_delta,
        ood_gap=ood_gap,
        thresholds=thresholds,
    )
    report = {
        "env": environment,
        "model_ref": model_ref,
        "round": 0,
        "goal_id": goal_id,
        "horizon_curve": horizon_curve,
        "appearance_drift": {"low_motion_ssim_64": appearance_ssim},
        "action_following": {
            "inv_dyn_acc_perframe": action_accuracy,
            "no_action_delta_psnr": no_action_delta,
            "inverse_dynamics_r2": inverse_r2,
            "low_confidence": bool(action.get("low_confidence")),
        },
        "ood_profile": {
            "ind_auc": ind_auc,
            "ood_auc": ood_auc,
            "gap": ood_gap,
            "worst_ood_condition": condition,
        },
        "dominant_failure": _dominant_failure(candidates, thresholds),
        "dominant_failure_candidates": [name for name, _ in candidates] or ["mixed"],
        "evidence_frames": _evidence_frames(horizon),
    }
    try:
        validate_document("failure_report", report)
    except ContractValidationError as exc:
        raise RawFailureReportError(f"RAW_FAILURE_REPORT_CONTRACT_INVALID:{exc}") from exc
    return report


def _load_coverage(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawFailureReportError("RAW_FAILURE_REPORT_COVERAGE_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-m1-raw-probe-coverage-report"
        or not isinstance(payload.get("records"), list)
        or not isinstance(payload.get("environment_count"), int)
    ):
        raise RawFailureReportError("RAW_FAILURE_REPORT_COVERAGE_INVALID")
    return payload


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise RawFailureReportError("RAW_FAILURE_REPORT_GOAL_INVALID") from exc
    return payload


def _load_diagnosis_thresholds(path: Path) -> tuple[DiagnosisThresholds, str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawFailureReportError("RAW_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID") from exc
    thresholds = payload.get("thresholds") if isinstance(payload, Mapping) else None
    mode = payload.get("mode") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-diagnosis-thresholds"
        or not isinstance(mode, str)
        or not mode
        or not isinstance(thresholds, Mapping)
    ):
        raise RawFailureReportError("RAW_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID")
    try:
        return DiagnosisThresholds(**{str(key): float(value) for key, value in thresholds.items()}), mode
    except (TypeError, ValueError) as exc:
        raise RawFailureReportError("RAW_FAILURE_REPORT_DIAGNOSIS_CONFIG_INVALID") from exc


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawFailureReportError(code)
    return value


def _float(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise RawFailureReportError(code)
    return float(value)


def _evidence_frames(horizon: Mapping[str, Any]) -> list[str]:
    raw = horizon.get("evidence_refs", [])
    if not isinstance(raw, list) or any(not isinstance(item, str) or not item.startswith("cas://sha256/") for item in raw):
        raise RawFailureReportError("RAW_FAILURE_REPORT_EVIDENCE_INVALID")
    return list(raw)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RawFailureReportError("RAW_FAILURE_REPORT_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate raw failure reports from coverage")
    generate.add_argument("--coverage-report", type=Path, required=True)
    generate.add_argument("--archive-db", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--probe-registry", type=Path, default=Path("configs/probes/acwm_v1.json"))
    generate.add_argument("--diagnosis-config", type=Path, default=Path("configs/diagnose/acwm_m1_measured_v1.json"))
    generate.add_argument("--repo-root", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_raw_failure_reports(
            coverage_report_path=args.coverage_report,
            archive_db=args.archive_db,
            output_root=args.output_root,
            goal_config=args.goal_config,
            probe_registry_path=args.probe_registry,
            diagnosis_config_path=args.diagnosis_config,
            repo_root=args.repo_root,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
