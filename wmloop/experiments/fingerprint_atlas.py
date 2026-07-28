"""Audit ACWM fingerprint receipts and project matched selector inputs."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.acwm_fingerprint import load_campaign, sha256_file
from wmloop.geometry import estimate_response_chart


class FingerprintAtlasError(ValueError):
    """A fingerprint campaign output is incomplete or internally inconsistent."""


def build_fingerprint_atlas(
    *,
    campaign_path: Path,
    campaign_output_root: Path,
    output_root: Path,
    maximum_locality_residual: float = 0.5,
    path_calibration_policy: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Create an auditable eight-environment atlas and selector projections."""

    campaign = load_campaign(Path(campaign_path))
    policy = _load_path_policy(path_calibration_policy, campaign) if path_calibration_policy else None
    if policy is not None:
        maximum_locality_residual = float(policy["maximum_locality_residual"])
    if not math.isfinite(maximum_locality_residual) or maximum_locality_residual < 0.0:
        raise FingerprintAtlasError("FINGERPRINT_LOCALITY_THRESHOLD_INVALID")
    run_root = Path(campaign_output_root).resolve()
    status = _load_json(run_root / "status.json", "FINGERPRINT_CAMPAIGN_STATUS_INVALID")
    if status.get("campaign_id") != campaign["campaign_id"]:
        raise FingerprintAtlasError("FINGERPRINT_CAMPAIGN_ID_MISMATCH")
    if status.get("state") != "ready":
        raise FingerprintAtlasError(f"FINGERPRINT_CAMPAIGN_NOT_READY:{status.get('state')}")

    environments: dict[str, dict[str, object]] = {}
    selector_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    derived_chart_files: dict[str, bytes] = {}
    environment_names = tuple(str(value) for value in campaign["environments"])
    for environment_index, environment in enumerate(environment_names):
        environment_root = run_root / "environments" / environment
        manifest_path = environment_root / "manifest.json"
        chart_path = environment_root / "response-chart.json"
        measurements_path = environment_root / "measurements.jsonl"
        manifest = _load_json(manifest_path, "FINGERPRINT_ENVIRONMENT_MANIFEST_INVALID")
        chart = _load_json(chart_path, "FINGERPRINT_RESPONSE_CHART_INVALID")
        measurements = _load_jsonl(measurements_path)
        _validate_environment_receipt(
            campaign=campaign,
            environment=environment,
            manifest=manifest,
            chart=chart,
            measurements=measurements,
            chart_path=chart_path,
            measurements_path=measurements_path,
            protocol=str(status["protocol"]),
        )
        local_chart = _derive_calibration_chart(campaign, measurements, policy) if policy else chart
        path_locality = {name: float(value) for name, value in local_chart["locality_residuals"].items()}
        supported_paths = sorted(name for name, value in path_locality.items() if value <= maximum_locality_residual)
        locality = min(path_locality.values())
        locality_pass = bool(supported_paths)
        derived_chart_files[f"charts/{environment}.json"] = canonical_json(local_chart)
        projections = _selector_projections(
            campaign=campaign,
            environment=environment,
            environment_index=environment_index,
            environment_count=len(environment_names),
            measurements=measurements,
            chart=local_chart,
            supported_paths=supported_paths,
        )
        selector_rows.extend(projections)
        environments[environment] = {
            "measurement_state": "complete",
            "locality_state": "passed" if locality_pass else "failed",
            "best_locality_residual": locality,
            "path_locality_residuals": path_locality,
            "supported_local_paths": supported_paths,
            "locality_threshold": maximum_locality_residual,
            "measurement_count": len(measurements),
            "repeat_count": int(chart["repeat_count"]),
            "checkpoint_step": manifest["checkpoint_step"],
            "manifest_ref": str(manifest_path),
            "response_chart_ref": str(chart_path),
            "measurements_ref": str(measurements_path),
        }
        evidence_rows.append(
            {
                "environment": environment,
                "measurement_complete": True,
                "locality_pass": locality_pass,
                "best_locality_residual": locality,
                "supported_local_paths": ";".join(supported_paths),
                "measurement_count": len(measurements),
                "repeat_count": chart["repeat_count"],
                "checkpoint_step": manifest["checkpoint_step"],
            }
        )

    local_count = sum(row["locality_state"] == "passed" for row in environments.values())
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-fingerprint-atlas",
        "campaign_id": campaign["campaign_id"],
        "protocol": status["protocol"],
        "state": "ready",
        "claim_boundary": (
            "All entries are paired-dose response evidence. Measurement completeness does not imply "
            "a valid local linear chart; only environments passing the frozen locality threshold count "
            "as calibrated for IRG selector claims."
        ),
        "environment_count": len(environments),
        "measurement_complete_count": len(environments),
        "locality_calibrated_count": local_count,
        "maximum_locality_residual": maximum_locality_residual,
        "path_calibration_policy": str(Path(path_calibration_policy).resolve()) if path_calibration_policy else None,
        "calibration_mode": "one_sided_semantic_paths" if policy else "source_signed_path",
        "selector_projection_contract": {
            "conditions": ["environment_label", "static_probe", "raw_response", "irg"],
            "same_source_measurements": True,
            "same_environment_order": list(environment_names),
            "selector_row_count": len(selector_rows),
        },
        "environments": environments,
    }
    files = {
        "fingerprint-atlas.json": canonical_json(report),
        "fingerprint-atlas.md": _markdown(report).encode("utf-8"),
        "tables/environment-calibration.csv": _csv(evidence_rows).encode("utf-8"),
        "selector-input-projections.jsonl": b"".join(canonical_json(row) for row in selector_rows),
        "input-campaign.json": canonical_json({key: value for key, value in campaign.items() if not key.startswith("_")}),
        **derived_chart_files,
    }
    if policy is not None:
        files["input-path-calibration-policy.json"] = canonical_json(policy)
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-acwm-fingerprint-atlas-manifest",
            "state": "ready",
            "campaign_id": campaign["campaign_id"],
            "protocol": status["protocol"],
            "environment_count": len(environments),
            "measurement_complete_count": len(environments),
            "locality_calibrated_count": local_count,
            "selector_projection_row_count": len(selector_rows),
            "report_path": str(destination / "fingerprint-atlas.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_environment_receipt(
    *,
    campaign: Mapping[str, Any],
    environment: str,
    manifest: Mapping[str, Any],
    chart: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    chart_path: Path,
    measurements_path: Path,
    protocol: str,
) -> None:
    expected_count = len(campaign["probe"]["doses"]) * len(campaign["seeds"])
    checks = {
        "state": manifest.get("state") == "ready",
        "campaign": manifest.get("campaign_id") == campaign["campaign_id"],
        "environment": manifest.get("environment") == environment,
        "protocol": manifest.get("protocol") == protocol,
        "measurement_count": manifest.get("measurement_count") == expected_count == len(measurements),
        "measurement_sha": manifest.get("measurement_sha256") == sha256_file(measurements_path),
        "chart_sha": manifest.get("response_chart_sha256") == sha256_file(chart_path),
        "chart_id": chart.get("chart_id") == f"{campaign['campaign_id']}:{environment}",
        "repeat_count": chart.get("repeat_count") == len(campaign["seeds"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise FingerprintAtlasError(f"FINGERPRINT_ENVIRONMENT_RECEIPT_INVALID:{environment}:{','.join(failed)}")
    expected_pairs = {
        (float(dose), int(seed))
        for dose in campaign["probe"]["doses"]
        for seed in campaign["seeds"]
    }
    actual_pairs = {(float(row["dose"]), int(row["seed"])) for row in measurements}
    if actual_pairs != expected_pairs:
        raise FingerprintAtlasError(f"FINGERPRINT_PAIRED_FRAME_INVALID:{environment}")
    numeric = [
        value
        for key in ("jacobian", "covariance", "repair_metric")
        for row in chart[key]
        for value in row
    ] + list(chart["response_coordinate"]) + list(chart["locality_residuals"].values())
    if not numeric or any(not math.isfinite(float(value)) for value in numeric):
        raise FingerprintAtlasError(f"FINGERPRINT_CHART_NONFINITE:{environment}")


def _selector_projections(
    *,
    campaign: Mapping[str, Any],
    environment: str,
    environment_index: int,
    environment_count: int,
    measurements: Sequence[Mapping[str, Any]],
    chart: Mapping[str, Any],
    supported_paths: Sequence[str],
) -> list[dict[str, object]]:
    seeds = tuple(int(value) for value in campaign["seeds"])
    outcomes = tuple(str(value["name"]) for value in campaign["goal_oriented_outcomes"])
    by_pair = {(float(row["dose"]), int(row["seed"])): row for row in measurements}
    baseline_vectors = [
        _goal_vector(campaign, by_pair[(0.0, seed)]["metrics"])
        for seed in seeds
    ]
    static_mean = _mean_vector(baseline_vectors)
    static_variance = _variance_vector(baseline_vectors, static_mean)
    raw_features: list[float] = []
    raw_feature_names: list[str] = []
    for dose in sorted(float(value) for value in campaign["probe"]["doses"] if float(value) != 0.0):
        deltas = []
        for seed, baseline in zip(seeds, baseline_vectors, strict=True):
            treated = _goal_vector(campaign, by_pair[(dose, seed)]["metrics"])
            deltas.append(tuple(a - b for a, b in zip(treated, baseline, strict=True)))
        mean_delta = _mean_vector(deltas)
        raw_features.extend(mean_delta)
        raw_feature_names.extend(f"dose={dose:g}:{name}" for name in outcomes)

    label = [0.0] * environment_count
    label[environment_index] = 1.0
    intervention_names = [str(value) for value in chart["intervention_names"]]
    supported = set(supported_paths)
    coordinate_mask = [
        1.0 if intervention_names[index % len(intervention_names)] in supported else 0.0
        for index in range(len(chart["response_coordinate"]))
    ]
    chart_coordinate = [float(value) * coordinate_mask[index] for index, value in enumerate(chart["response_coordinate"])]
    covariance_diagonal = [
        float(row[index]) * coordinate_mask[index]
        for index, row in enumerate(chart["covariance"])
    ]
    locality = [float(chart["locality_residuals"][name]) for name in chart["intervention_names"]]
    path_mask = [1.0 if name in supported else 0.0 for name in intervention_names]
    common = {
        "schema_version": 1,
        "artifact_type": "verdiwm-selector-input-projection",
        "campaign_id": campaign["campaign_id"],
        "environment": environment,
        "source_measurement_count": len(measurements),
        "source_seed_count": len(seeds),
    }
    return [
        {
            **common,
            "selector": "environment_label",
            "feature_names": [f"environment={name}" for name in campaign["environments"]],
            "features": label,
            "uses_intervention_response": False,
            "uses_uncertainty": False,
        },
        {
            **common,
            "selector": "static_probe",
            "feature_names": [f"baseline_mean:{name}" for name in outcomes]
            + [f"baseline_variance:{name}" for name in outcomes],
            "features": list(static_mean) + list(static_variance),
            "uses_intervention_response": False,
            "uses_uncertainty": True,
        },
        {
            **common,
            "selector": "raw_response",
            "feature_names": raw_feature_names,
            "features": raw_features,
            "uses_intervention_response": True,
            "uses_uncertainty": False,
            "aligned_geometry": False,
        },
        {
            **common,
            "selector": "irg",
            "feature_names": [f"response_coordinate:{index}" for index in range(len(chart_coordinate))]
            + [f"covariance_diagonal:{index}" for index in range(len(covariance_diagonal))]
            + [f"locality:{name}" for name in intervention_names]
            + [f"path_supported:{name}" for name in intervention_names],
            "features": chart_coordinate + covariance_diagonal + locality + path_mask,
            "uses_intervention_response": True,
            "uses_uncertainty": True,
            "aligned_geometry": True,
        },
    ]


def _load_path_policy(path: Path, campaign: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = _load_json(Path(path), "FINGERPRINT_PATH_POLICY_INVALID")
    if policy.get("artifact_type") != "verdiwm-irg-path-calibration-policy":
        raise FingerprintAtlasError("FINGERPRINT_PATH_POLICY_TYPE_INVALID")
    if policy.get("source_campaign_id") != campaign["campaign_id"]:
        raise FingerprintAtlasError("FINGERPRINT_PATH_POLICY_CAMPAIGN_MISMATCH")
    if policy.get("source_probe_id") != campaign["probe"]["probe_id"]:
        raise FingerprintAtlasError("FINGERPRINT_PATH_POLICY_PROBE_MISMATCH")
    paths = policy.get("paths")
    if not isinstance(paths, list) or len(paths) < 2:
        raise FingerprintAtlasError("FINGERPRINT_PATH_POLICY_PATHS_INVALID")
    signs = {str(item.get("source_dose_sign")) for item in paths if isinstance(item, Mapping)}
    if signs != {"positive", "negative"}:
        raise FingerprintAtlasError("FINGERPRINT_PATH_POLICY_SIGNS_INVALID")
    return policy


def _derive_calibration_chart(
    campaign: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, object]:
    seeds = tuple(int(value) for value in campaign["seeds"])
    by_pair = {
        (float(row["dose"]), int(row["seed"])): _goal_vector(campaign, row["metrics"])
        for row in measurements
    }
    baseline = tuple(by_pair[(0.0, seed)] for seed in seeds)
    observations: dict[str, dict[float, tuple[tuple[float, ...], ...]]] = {}
    minimum_doses = int(policy["minimum_nonzero_doses_per_path"])
    source_doses = tuple(float(value) for value in campaign["probe"]["doses"] if float(value) != 0.0)
    for path in policy["paths"]:
        sign = str(path["source_dose_sign"])
        selected = sorted(
            (dose for dose in source_doses if (dose > 0.0) is (sign == "positive")),
            key=abs,
        )
        if len(selected) < minimum_doses:
            raise FingerprintAtlasError(f"FINGERPRINT_PATH_DOSES_INSUFFICIENT:{path['path_name']}")
        observations[str(path["path_name"])] = {
            abs(dose): tuple(by_pair[(dose, seed)] for seed in seeds)
            for dose in selected
        }
    chart = estimate_response_chart(
        chart_id=f"{campaign['campaign_id']}:{measurements[0]['environment']}:one_sided",
        goal_schema="acwm_phys_goal_oriented_pixel_metrics_v1",
        outcome_names=tuple(str(value["name"]) for value in campaign["goal_oriented_outcomes"]),
        outcome_weights=tuple(float(value["weight"]) for value in campaign["goal_oriented_outcomes"]),
        baseline_repeats=baseline,
        dose_observations=observations,
    ).to_dict()
    chart["source_signed_probe_id"] = campaign["probe"]["probe_id"]
    chart["path_calibration_policy_id"] = policy["policy_id"]
    chart["claim_boundary"] = policy["claim_boundary"]
    return chart


def _goal_vector(campaign: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(metrics[str(outcome["source_metric"])]) * float(outcome["sign"])
        for outcome in campaign["goal_oriented_outcomes"]
    )


def _mean_vector(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(fmean(row[index] for row in rows) for index in range(len(rows[0])))


def _variance_vector(rows: Sequence[Sequence[float]], mean: Sequence[float]) -> tuple[float, ...]:
    if len(rows) < 2:
        return tuple(0.0 for _ in mean)
    return tuple(
        sum((row[index] - mean[index]) ** 2 for row in rows) / (len(rows) - 1)
        for index in range(len(mean))
    )


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintAtlasError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise FingerprintAtlasError(f"{code}:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintAtlasError(f"FINGERPRINT_MEASUREMENTS_INVALID:{path}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise FingerprintAtlasError(f"FINGERPRINT_MEASUREMENTS_INVALID:{path}")
    return rows


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM-Phys Fingerprint Atlas",
        "",
        str(report["claim_boundary"]),
        "",
        "| Environment | Measurements | Locality residual | Threshold | Calibration |",
        "|---|---:|---:|---:|---|",
    ]
    for environment, row in report["environments"].items():
        lines.append(
            f"| {environment} | {row['measurement_count']} | {row['best_locality_residual']:.6f} | "
            f"{row['locality_threshold']:.6f} | {row['locality_state']} |"
        )
    lines.extend(
        [
            "",
            f"Measurement-complete environments: `{report['measurement_complete_count']}/{report['environment_count']}`.",
            f"Locality-calibrated environments: `{report['locality_calibrated_count']}/{report['environment_count']}`.",
            "",
            "The selector projection file contains four rows per environment, all derived from the same paired measurements.",
        ]
    )
    return "\n".join(lines) + "\n"
