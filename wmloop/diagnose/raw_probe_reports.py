"""Generate M1 raw-probe coverage reports without fabricating probe fields."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class RawProbeReportError(RuntimeError):
    """A raw-probe coverage report could not be produced safely."""


def generate_raw_probe_coverage_report(
    *,
    horizon_summary_path: Path,
    inverse_summary_path: Path,
    output_root: Path,
    goal_config: Path = Path("configs/goal/long_horizon_v1.yaml"),
    horizon_availability_path: Path | None = None,
    horizon_protocol_decision_path: Path | None = None,
    raw_probe_evidence_path: Path | Sequence[Path] | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a status bundle for measured M1 raw probes.

    The report is deliberately separate from ``failure_report``.  A formal
    failure report requires all contract fields; this coverage layer records
    partial raw evidence and the blockers that prevent contract emission.
    """

    goal = _load_goal(goal_config)
    expected_envs = tuple(str(item) for item in goal["envs"])
    horizon_protocol = _goal_horizon_protocol(goal=goal, goal_config=Path(goal_config))
    required_horizons_by_env = horizon_protocol["required_horizons_by_environment"]
    horizon_summary = _load_summary(horizon_summary_path, "wmloop-horizon-probe-summary", "RAW_PROBE_HORIZON_SUMMARY_INVALID")
    inverse_summary = _load_summary(inverse_summary_path, "wmloop-inverse-dynamics-cache-summary", "RAW_PROBE_INVERSE_SUMMARY_INVALID")
    availability_by_env_split = _load_availability_records(horizon_availability_path) if horizon_availability_path is not None else {}
    protocol_by_env_split = _load_protocol_records(horizon_protocol_decision_path) if horizon_protocol_decision_path is not None else {}
    evidence_paths = _evidence_paths(raw_probe_evidence_path)
    evidence_by_env_split = _load_evidence_records(evidence_paths) if evidence_paths else {}
    horizon_by_env = _records_by_environment(horizon_summary, "RAW_PROBE_HORIZON_RECORD_INVALID")
    inverse_by_env = _records_by_environment(inverse_summary, "RAW_PROBE_INVERSE_RECORD_INVALID")
    records: list[dict[str, object]] = []
    for environment in expected_envs:
        records.append(
            _environment_record(
                environment=environment,
                required_horizons=required_horizons_by_env[environment],
                horizon_record=horizon_by_env.get(environment),
                inverse_record=inverse_by_env.get(environment),
                availability_by_env_split=availability_by_env_split,
                protocol_by_env_split=protocol_by_env_split,
                evidence_by_env_split=evidence_by_env_split,
            )
        )
    ready_horizon_count = sum(1 for record in records if record["probe_coverage"]["horizon_curve"]["state"] == "ready")  # type: ignore[index]
    formal_ready_count = sum(1 for record in records if record["failure_report_contract_state"] == "ready")
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-raw-probe-coverage-report",
        "state": "ready" if formal_ready_count == len(expected_envs) else "incomplete",
        "goal_id": goal["goal_id"],
        "required_horizons": list(int(item) for item in goal["horizons"]),
        "required_horizons_by_environment": required_horizons_by_env,
        "horizon_protocol": horizon_protocol["protocol"],
        "horizon_summary_path": str(Path(horizon_summary_path).resolve()),
        "inverse_summary_path": str(Path(inverse_summary_path).resolve()),
        "horizon_availability_path": str(Path(horizon_availability_path).resolve()) if horizon_availability_path is not None else None,
        "horizon_protocol_decision_path": str(Path(horizon_protocol_decision_path).resolve()) if horizon_protocol_decision_path is not None else None,
        "raw_probe_evidence_paths": [str(path.resolve()) for path in evidence_paths],
        "environment_count": len(expected_envs),
        "ready_horizon_count": ready_horizon_count,
        "formal_failure_report_ready_count": formal_ready_count,
        "records": records,
        "warnings": _report_warnings(records),
        "next_actions": _next_actions(records),
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise RawProbeReportError("RAW_PROBE_GOAL_INVALID") from exc
    if not isinstance(payload.get("envs"), list) or not isinstance(payload.get("horizons"), list):
        raise RawProbeReportError("RAW_PROBE_GOAL_INVALID")
    return payload


def _goal_horizon_protocol(*, goal: Mapping[str, Any], goal_config: Path) -> dict[str, Any]:
    envs = tuple(str(item) for item in goal["envs"])
    base_horizons = tuple(int(item) for item in goal["horizons"])
    protocol = goal.get("eval_protocol")
    if not isinstance(protocol, Mapping):
        raise RawProbeReportError("RAW_PROBE_GOAL_INVALID")
    if protocol.get("mode") != "per_environment_horizon_ladder":
        return {
            "required_horizons_by_environment": {environment: list(base_horizons) for environment in envs},
            "protocol": {
                "mode": protocol.get("mode", "uniform_horizons"),
                "horizon_ladder_path": None,
                "cross_environment_comparison_horizons": list(base_horizons),
            },
        }
    ladder_path = protocol.get("horizon_ladder_path")
    if not isinstance(ladder_path, str) or not ladder_path:
        raise RawProbeReportError("RAW_PROBE_HORIZON_LADDER_INVALID")
    ladder = _load_horizon_ladder(_resolve_ladder_path(ladder_path, goal_config=goal_config))
    if ladder.get("goal_id") != goal.get("goal_id"):
        raise RawProbeReportError("RAW_PROBE_HORIZON_LADDER_GOAL_MISMATCH")
    horizons_by_environment = ladder.get("horizons_by_environment")
    if not isinstance(horizons_by_environment, Mapping):
        raise RawProbeReportError("RAW_PROBE_HORIZON_LADDER_INVALID")
    required_by_env: dict[str, list[int]] = {}
    for environment in envs:
        raw = horizons_by_environment.get(environment)
        if not isinstance(raw, list) or not raw:
            raise RawProbeReportError(f"RAW_PROBE_HORIZON_LADDER_ENV_MISSING:{environment}")
        required_by_env[environment] = [int(item) for item in raw]
    return {
        "required_horizons_by_environment": required_by_env,
        "protocol": {
            "mode": "per_environment_horizon_ladder",
            "horizon_ladder_path": str(_resolve_ladder_path(ladder_path, goal_config=goal_config)),
            "cross_environment_comparison_horizons": ladder.get("cross_environment_comparison_horizons", []),
            "long_horizon_64_envs": ladder.get("long_horizon_64_envs", []),
        },
    }


def _load_horizon_ladder(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(path)
        validate_document("horizon_ladder", payload)
    except (OSError, ContractValidationError) as exc:
        raise RawProbeReportError("RAW_PROBE_HORIZON_LADDER_INVALID") from exc
    return payload


def _resolve_ladder_path(raw_path: str, *, goal_config: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve(strict=True)
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    goal_candidate = (Path(goal_config).resolve().parent / path).resolve()
    if goal_candidate.is_file():
        return goal_candidate
    raise RawProbeReportError(f"RAW_PROBE_HORIZON_LADDER_MISSING:{raw_path}")


def _load_summary(path: Path, artifact_type: str, code: str) -> Mapping[str, Any]:
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            raise RawProbeReportError(code)
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawProbeReportError(code) from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or payload.get("artifact_type") != artifact_type:
        raise RawProbeReportError(code)
    records = payload.get("records")
    if not isinstance(records, list):
        raise RawProbeReportError(code)
    return payload


def _records_by_environment(summary: Mapping[str, Any], code: str) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, Mapping[str, Any]] = {}
    for item in summary["records"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("environment"), str) or not item["environment"]:
            raise RawProbeReportError(code)
        environment = str(item["environment"])
        if environment in grouped:
            raise RawProbeReportError(f"{code}:DUPLICATE:{environment}")
        grouped[environment] = item
    return grouped


def _load_availability_records(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    summary = _load_summary(path, "wmloop-horizon-availability-report", "RAW_PROBE_HORIZON_AVAILABILITY_INVALID")
    grouped: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in summary["records"]:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("environment"), str)
            or not isinstance(item.get("split"), str)
        ):
            raise RawProbeReportError("RAW_PROBE_HORIZON_AVAILABILITY_INVALID")
        key = (str(item["environment"]), str(item["split"]))
        if key in grouped:
            raise RawProbeReportError(f"RAW_PROBE_HORIZON_AVAILABILITY_INVALID:DUPLICATE:{key[0]}:{key[1]}")
        grouped[key] = item
    return grouped


def _load_protocol_records(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    summary = _load_summary(path, "wmloop-m1-horizon-protocol-decision", "RAW_PROBE_HORIZON_PROTOCOL_INVALID")
    grouped: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in summary["records"]:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("environment"), str)
            or not isinstance(item.get("split"), str)
            or not isinstance(item.get("unavailable_horizons"), list)
        ):
            raise RawProbeReportError("RAW_PROBE_HORIZON_PROTOCOL_INVALID")
        key = (str(item["environment"]), str(item["split"]))
        if key in grouped:
            raise RawProbeReportError(f"RAW_PROBE_HORIZON_PROTOCOL_INVALID:DUPLICATE:{key[0]}:{key[1]}")
        grouped[key] = item
    return grouped


def _evidence_paths(value: Path | Sequence[Path] | None) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, Path):
        return (value,)
    return tuple(Path(item) for item in value)


def _load_evidence_records(paths: Sequence[Path]) -> dict[tuple[str, str], Mapping[str, Any]]:
    grouped: dict[tuple[str, str], Mapping[str, Any]] = {}
    for path in paths:
        summary = _load_summary(path, "wmloop-m1-raw-probe-evidence-report", "RAW_PROBE_EVIDENCE_INVALID")
        if summary.get("state") != "ready":
            raise RawProbeReportError("RAW_PROBE_EVIDENCE_NOT_READY")
        for item in summary["records"]:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("environment"), str)
                or not isinstance(item.get("split"), str)
            ):
                raise RawProbeReportError("RAW_PROBE_EVIDENCE_INVALID")
            key = (str(item["environment"]), str(item["split"]))
            if key in grouped:
                raise RawProbeReportError(f"RAW_PROBE_EVIDENCE_INVALID:DUPLICATE:{key[0]}:{key[1]}")
            grouped[key] = item
    return grouped


def _environment_record(
    *,
    environment: str,
    required_horizons: Sequence[int],
    horizon_record: Mapping[str, Any] | None,
    inverse_record: Mapping[str, Any] | None,
    availability_by_env_split: Mapping[tuple[str, str], Mapping[str, Any]],
    protocol_by_env_split: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence_by_env_split: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, object]:
    split = str(horizon_record.get("split")) if horizon_record is not None and isinstance(horizon_record.get("split"), str) else "ind_test"
    evidence_record = evidence_by_env_split.get((environment, split))
    horizon = _horizon_coverage(
        required_horizons=required_horizons,
        record=horizon_record,
        availability_record=availability_by_env_split.get((environment, split)),
        protocol_record=protocol_by_env_split.get((environment, split)),
    )
    appearance = _appearance_coverage(evidence_record)
    action = _action_coverage(inverse_record, evidence_record)
    ood = _ood_coverage(evidence_record)
    blockers: list[str] = []
    if horizon["state"] != "ready":
        unavailable_by_protocol = set(horizon.get("unavailable_by_protocol", []))
        unsupported = set(horizon.get("unsupported_by_availability", []))
        missing = set(horizon.get("missing_horizons", []))
        if missing and missing <= unavailable_by_protocol:
            blockers.append("horizon_unavailable_by_protocol")
        else:
            blockers.append("horizon_unsupported_by_dataset_length" if missing and missing <= unsupported else "horizon_curve_incomplete")
    if appearance["state"] != "measured":
        blockers.append("appearance_drift_raw_probe_not_measured")
    if action["no_action_delta_state"] != "measured":
        blockers.append("action_following_no_action_delta_not_measured")
    if ood["state"] != "measured":
        blockers.append("ood_profile_raw_probe_not_measured")
    confidence_warnings: list[str] = []
    if action["inverse_cache_state"] == "missing":
        blockers.append("inverse_dynamics_cache_missing")
    elif action["low_confidence"]:
        confidence_warnings.append("inverse_dynamics_low_confidence")
    return {
        "environment": environment,
        "failure_report_contract_state": "ready" if not blockers else "blocked",
        "failure_report_blockers": blockers,
        "probe_confidence_warnings": confidence_warnings,
        "probe_coverage": {
            "horizon_curve": horizon,
            "appearance_drift": appearance,
            "action_following": action,
            "ood_profile": ood,
        },
    }


def _horizon_coverage(
    *,
    required_horizons: Sequence[int],
    record: Mapping[str, Any] | None,
    availability_record: Mapping[str, Any] | None = None,
    protocol_record: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    availability = _availability_context(availability_record)
    protocol = _protocol_context(protocol_record)
    limit_context = _horizon_limit_context(required_horizons=required_horizons, availability=availability, protocol=protocol)
    if record is None:
        return {
            "state": "missing",
            "required_horizons": [str(item) for item in required_horizons],
            "available_horizons": [],
            "missing_horizons": [str(item) for item in required_horizons],
            "has_required_auc": False,
            "has_segment_drift": False,
            **availability,
            **protocol,
            **limit_context,
        }
    metrics = record.get("horizon_metrics")
    curve = record.get("horizon_curve")
    segment = record.get("segment_drift")
    if not isinstance(metrics, Mapping) or not isinstance(curve, Mapping):
        raise RawProbeReportError("RAW_PROBE_HORIZON_RECORD_INVALID")
    available = sorted(str(item) for item in metrics)
    missing = [str(horizon) for horizon in required_horizons if str(horizon) not in metrics]
    auc_key = f"auc_psnr_{required_horizons[0]}_{required_horizons[-1]}"
    has_required_auc = _finite_number(curve.get(auc_key))
    has_segment_drift = isinstance(segment, Mapping) and _finite_number(segment.get("worst_drop"))
    psnr = curve.get("psnr")
    if not isinstance(psnr, Mapping):
        raise RawProbeReportError("RAW_PROBE_HORIZON_RECORD_INVALID")
    return {
        "state": "ready" if not missing and has_required_auc and has_segment_drift else "incomplete",
        "required_horizons": [str(item) for item in required_horizons],
        "available_horizons": available,
        "missing_horizons": missing,
        "has_required_auc": has_required_auc,
        "available_auc_keys": sorted(key for key in curve if isinstance(key, str) and key.startswith("auc_psnr_")),
        "has_segment_drift": has_segment_drift,
        "psnr": {str(key): float(value) for key, value in sorted(psnr.items()) if _finite_number(value)},
        "segment_drift": dict(segment) if isinstance(segment, Mapping) else None,
        "manifest_path": record.get("manifest_path"),
        "manifest_ref": record.get("manifest_ref"),
        "metrics_ref": record.get("metrics_ref"),
        "evidence_refs": record.get("evidence_refs", []),
        "used_trajectory_count": record.get("used_trajectory_count"),
        "attempted_trajectory_count": record.get("attempted_trajectory_count"),
        **availability,
        **protocol,
        **limit_context,
    }


def _horizon_limit_context(
    *,
    required_horizons: Sequence[int],
    availability: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    required = {str(item) for item in required_horizons}
    unsupported = set(str(item) for item in availability.get("unsupported_by_availability", []))
    unavailable = set(str(item) for item in protocol.get("unavailable_by_protocol", []))
    return {
        "unsupported_required_by_availability": sorted(unsupported.intersection(required), key=int),
        "unavailable_required_by_protocol": sorted(unavailable.intersection(required), key=int),
    }


def _availability_context(record: Mapping[str, Any] | None) -> dict[str, object]:
    if record is None:
        return {
            "availability_state": "not_checked",
            "unsupported_by_availability": [],
            "available_steps_max": None,
        }
    unsupported = record.get("unsupported_horizons")
    if not isinstance(unsupported, list) or any(not isinstance(item, str) for item in unsupported):
        raise RawProbeReportError("RAW_PROBE_HORIZON_AVAILABILITY_INVALID")
    available_steps_max = record.get("available_steps_max")
    if not _finite_number(available_steps_max):
        raise RawProbeReportError("RAW_PROBE_HORIZON_AVAILABILITY_INVALID")
    return {
        "availability_state": record.get("state", "unknown"),
        "unsupported_by_availability": list(unsupported),
        "available_steps_max": int(available_steps_max),
    }


def _protocol_context(record: Mapping[str, Any] | None) -> dict[str, object]:
    if record is None:
        return {
            "protocol_state": "not_checked",
            "unavailable_by_protocol": [],
            "protocol_contract_effect": None,
        }
    unavailable = record.get("unavailable_horizons")
    if not isinstance(unavailable, list) or any(not isinstance(item, str) for item in unavailable):
        raise RawProbeReportError("RAW_PROBE_HORIZON_PROTOCOL_INVALID")
    effect = record.get("failure_report_contract_effect")
    if effect is not None and not isinstance(effect, str):
        raise RawProbeReportError("RAW_PROBE_HORIZON_PROTOCOL_INVALID")
    return {
        "protocol_state": "ready",
        "unavailable_by_protocol": list(unavailable),
        "protocol_contract_effect": effect,
    }


def _inverse_coverage(record: Mapping[str, Any] | None) -> dict[str, object]:
    if record is None:
        return {
            "state": "missing",
            "inverse_dynamics_r2": None,
            "low_confidence": True,
            "reason": "inverse dynamics cache record missing",
        }
    r2 = record.get("gt_r2")
    low_confidence = record.get("low_confidence")
    if not _finite_number(r2) or not isinstance(low_confidence, bool):
        raise RawProbeReportError("RAW_PROBE_INVERSE_RECORD_INVALID")
    return {
        "state": "low_confidence" if low_confidence else "cache_ready",
        "inverse_dynamics_r2": float(r2),
        "low_confidence": low_confidence,
        "checkpoint_ref": record.get("checkpoint_ref"),
        "cache_record_ref": record.get("cache_record_ref"),
        "sample_count": record.get("sample_count"),
        "used_trajectory_count": record.get("used_trajectory_count"),
        "note": "no-action rollout delta is not measured by the inverse-dynamics cache",
    }


def _appearance_coverage(evidence_record: Mapping[str, Any] | None) -> dict[str, object]:
    if evidence_record is None:
        return {
            "state": "not_measured",
            "reason": "no low-motion raw appearance probe artifact was supplied",
        }
    raw = evidence_record.get("appearance_drift")
    if not isinstance(raw, Mapping) or raw.get("state") != "measured" or not _finite_number(raw.get("low_motion_ssim_64")):
        return {"state": "not_measured", "reason": "appearance_drift evidence missing or incomplete"}
    return {
        "state": "measured",
        "low_motion_ssim_64": float(raw["low_motion_ssim_64"]),
        "sample_count": raw.get("sample_count"),
        "evidence_refs": raw.get("evidence_refs", []),
    }


def _action_coverage(inverse_record: Mapping[str, Any] | None, evidence_record: Mapping[str, Any] | None) -> dict[str, object]:
    inverse = _inverse_coverage(inverse_record)
    coverage = {
        **inverse,
        "inverse_cache_state": inverse["state"],
        "no_action_delta_state": "not_measured",
        "inv_dyn_acc_perframe": None,
        "no_action_delta_psnr": None,
        "evidence_refs": [],
    }
    if evidence_record is None:
        return coverage
    raw = evidence_record.get("action_following")
    if not isinstance(raw, Mapping) or raw.get("state") != "measured":
        return coverage
    if not _finite_number(raw.get("inv_dyn_acc_perframe")) or not _finite_number(raw.get("no_action_delta_psnr")):
        raise RawProbeReportError("RAW_PROBE_EVIDENCE_INVALID")
    coverage.update(
        {
            "state": "measured_low_confidence" if coverage["low_confidence"] else "measured",
            "no_action_delta_state": "measured",
            "inv_dyn_acc_perframe": float(raw["inv_dyn_acc_perframe"]),
            "no_action_delta_psnr": float(raw["no_action_delta_psnr"]),
            "evidence_refs": raw.get("evidence_refs", []),
        }
    )
    return coverage


def _ood_coverage(evidence_record: Mapping[str, Any] | None) -> dict[str, object]:
    if evidence_record is None:
        return {
            "state": "not_measured",
            "reason": "no raw InD/OoD horizon profile artifact was supplied",
        }
    raw = evidence_record.get("ood_profile")
    required = ("ind_auc", "ood_auc", "gap", "worst_ood_condition")
    if not isinstance(raw, Mapping) or raw.get("state") != "measured" or any(key not in raw for key in required):
        return {"state": "not_measured", "reason": "ood_profile evidence missing or incomplete"}
    if not all(_finite_number(raw.get(key)) for key in ("ind_auc", "ood_auc", "gap")) or not isinstance(raw.get("worst_ood_condition"), str):
        raise RawProbeReportError("RAW_PROBE_EVIDENCE_INVALID")
    return {
        "state": "measured",
        "ind_auc": float(raw["ind_auc"]),
        "ood_auc": float(raw["ood_auc"]),
        "gap": float(raw["gap"]),
        "worst_ood_condition": raw["worst_ood_condition"],
        "evidence_refs": raw.get("evidence_refs", []),
    }


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _report_warnings(records: Sequence[Mapping[str, object]]) -> list[str]:
    warnings = []
    incomplete = [record["environment"] for record in records if record["failure_report_contract_state"] != "ready"]
    if incomplete:
        warnings.append("Formal failure_report emission is blocked for at least one environment; missing probe fields were not fabricated.")
    missing_64 = [
        record["environment"]
        for record in records
        if "64" in record["probe_coverage"]["horizon_curve"].get("missing_horizons", [])  # type: ignore[index,union-attr]
    ]
    if missing_64:
        warnings.append(f"Missing 64-frame horizon coverage: {','.join(str(item) for item in missing_64)}")
    length_limited = [
        record["environment"]
        for record in records
        if record["probe_coverage"]["horizon_curve"].get("unsupported_required_by_availability")  # type: ignore[index,union-attr]
    ]
    if length_limited:
        warnings.append(f"Dataset/action length audit marks horizons unsupported for: {','.join(str(item) for item in length_limited)}")
    protocol_limited = [
        record["environment"]
        for record in records
        if record["probe_coverage"]["horizon_curve"].get("unavailable_required_by_protocol")  # type: ignore[index,union-attr]
    ]
    if protocol_limited:
        warnings.append(f"Horizon protocol decision marks formal reports blocked for: {','.join(str(item) for item in protocol_limited)}")
    return warnings


def _next_actions(records: Sequence[Mapping[str, object]]) -> list[str]:
    actions: list[str] = []
    rerun_envs: list[str] = []
    length_limited_envs: list[str] = []
    protocol_blocked_envs: list[str] = []
    appearance_missing: list[str] = []
    action_missing: list[str] = []
    ood_missing: list[str] = []
    for record in records:
        horizon = record["probe_coverage"]["horizon_curve"]  # type: ignore[index]
        missing = set(horizon.get("missing_horizons", []))
        if missing:
            unavailable_by_protocol = set(horizon.get("unavailable_by_protocol", []))
            if unavailable_by_protocol and missing <= unavailable_by_protocol:
                protocol_blocked_envs.append(str(record["environment"]))
            else:
                unsupported = set(horizon.get("unsupported_by_availability", []))
                if unsupported and missing <= unsupported:
                    length_limited_envs.append(str(record["environment"]))
                else:
                    rerun_envs.append(str(record["environment"]))
        coverage = record["probe_coverage"]  # type: ignore[index]
        if coverage["appearance_drift"].get("state") != "measured":  # type: ignore[index,union-attr]
            appearance_missing.append(str(record["environment"]))
        if coverage["action_following"].get("no_action_delta_state") != "measured":  # type: ignore[index,union-attr]
            action_missing.append(str(record["environment"]))
        if coverage["ood_profile"].get("state") != "measured":  # type: ignore[index,union-attr]
            ood_missing.append(str(record["environment"]))
    if rerun_envs:
        actions.append(f"Run horizon_runtime to complete required horizons for: {','.join(rerun_envs)}")
    if length_limited_envs:
        actions.append(
            "Revise the horizon protocol or mark 64-frame raw horizon unavailable for dataset-limited environments: "
            + ",".join(length_limited_envs)
        )
    if protocol_blocked_envs:
        actions.append(
            "Keep formal failure_report emission blocked by the horizon protocol decision unless the goal/data protocol is revised: "
            + ",".join(protocol_blocked_envs)
        )
    if appearance_missing:
        actions.append("Add raw appearance_drift low-motion probe artifacts before emitting formal failure_report for: " + ",".join(appearance_missing))
    if action_missing:
        actions.append("Add action-following no-action delta rollout artifacts before verdict gating for: " + ",".join(action_missing))
    if ood_missing:
        actions.append("Add raw InD/OoD profile artifacts before ood_profile attribution for: " + ",".join(ood_missing))
    return actions


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Raw-Probe Coverage Report",
        "",
        f"State: `{report['state']}`",
        f"Ready horizon environments: `{report['ready_horizon_count']}/{report['environment_count']}`",
        f"Formal failure_report ready: `{report['formal_failure_report_ready_count']}/{report['environment_count']}`",
        "",
        "| Environment | Failure Report | Horizons | AUC Required Span | Segment Drift | Action Cache | Blockers |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        coverage = record["probe_coverage"]
        horizon = coverage["horizon_curve"]
        action = coverage["action_following"]
        blockers = ",".join(record["failure_report_blockers"])
        lines.append(
            "| {env} | {fr} | {horizons} | {auc} | {segment} | {action} | {blockers} |".format(
                env=record["environment"],
                fr=record["failure_report_contract_state"],
                horizons="/".join(horizon["available_horizons"]),
                auc=str(horizon["has_required_auc"]).lower(),
                segment=str(horizon["has_segment_drift"]).lower(),
                action=action["state"],
                blockers=blockers,
            )
        )
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


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
        raise RawProbeReportError("RAW_PROBE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "raw-probe-coverage.json", report_bytes)
        _write_bytes_atomic(temporary / "raw-probe-coverage.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("raw_probe_coverage_json", report_bytes, "application/json"),
                ("raw_probe_coverage_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-raw-probe-coverage-manifest",
            "state": report["state"],
            "report_path": str(destination / "raw-probe-coverage.json"),
            "markdown_path": str(destination / "raw-probe-coverage.md"),
            "cas_refs": cas_refs,
            "warnings": report["warnings"],
            "next_actions": report["next_actions"],
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
        raise RawProbeReportError("RAW_PROBE_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate an M1 raw-probe coverage report")
    generate.add_argument("--horizon-summary", type=Path, required=True)
    generate.add_argument("--inverse-summary", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--horizon-availability", type=Path)
    generate.add_argument("--horizon-protocol-decision", type=Path)
    generate.add_argument("--raw-probe-evidence", type=Path, nargs="+")
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_raw_probe_coverage_report(
            horizon_summary_path=args.horizon_summary,
            inverse_summary_path=args.inverse_summary,
            output_root=args.output_root,
            goal_config=args.goal_config,
            horizon_availability_path=args.horizon_availability,
            horizon_protocol_decision_path=args.horizon_protocol_decision,
            raw_probe_evidence_path=args.raw_probe_evidence,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
