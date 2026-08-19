"""Deterministic, evidence-bound checks for whether a metric is adequate.

This module evaluates a candidate metric in shadow mode. Its result is an
input to constitutional policy, never a scientific verdict and never a change
to the active evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.constitution_proposal import (
    ConstitutionProposalError,
    validate_metric_adequacy,
)


class MetricAdequacyError(ValueError):
    """A metric-adequacy input failed validation or was not evidence-bound."""


_CONTENT_REF_PREFIXES = ("cas://", "urn:", "sha256:")
_THRESHOLD_FIELDS = (
    "minimum_discrimination_lower_bound",
    "maximum_reference_cv",
    "minimum_anti_goodhart_coverage",
    "minimum_incremental_information_lower_bound",
)


def evaluate_metric_adequacy(
    metric_contract: Mapping[str, object],
    observation: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
    root: Path | None = None,
) -> dict[str, object]:
    """Evaluate metric usefulness with explicit, frozen-at-call thresholds."""

    try:
        validate_metric_adequacy(metric_contract, root=root)
    except ConstitutionProposalError as exc:
        raise MetricAdequacyError(str(exc)) from exc
    _validate_observation(observation, root=root)
    normalized_thresholds = _validate_thresholds(thresholds)
    if observation["metric_id"] != metric_contract["metric_id"]:
        raise MetricAdequacyError("METRIC_ADEQUACY_OBSERVATION_METRIC_MISMATCH")
    if observation["direction"] != metric_contract["direction"]:
        raise MetricAdequacyError("METRIC_ADEQUACY_OBSERVATION_DIRECTION_MISMATCH")
    if observation["evaluator_binding"] != metric_contract["evaluator_binding"]:
        raise MetricAdequacyError("METRIC_ADEQUACY_OBSERVATION_EVALUATOR_MISMATCH")

    reference = [float(value) for value in observation["reference_values"]]
    regression = [float(value) for value in observation["known_regression_values"]]
    _require_finite(reference, "REFERENCE")
    _require_finite(regression, "KNOWN_REGRESSION")
    reference_mean = statistics.fmean(reference)
    reference_cv = _coefficient_of_variation(reference, reference_mean)
    if not math.isfinite(reference_cv):
        raise MetricAdequacyError("METRIC_ADEQUACY_REFERENCE_CV_NONFINITE")
    separation_lower_bound = _separation_lower_bound(
        reference,
        regression,
        direction=str(metric_contract["direction"]),
    )
    anti_cases = observation["anti_goodhart_cases"]
    coverage = sum(1 for row in anti_cases if row["detected"] is True) / len(anti_cases)
    incremental_information = float(observation["incremental_information_lower_bound"])
    if not math.isfinite(incremental_information):
        raise MetricAdequacyError("METRIC_ADEQUACY_INCREMENTAL_INFORMATION_NONFINITE")

    gates = {
        "discriminative_power": separation_lower_bound
        >= normalized_thresholds["minimum_discrimination_lower_bound"],
        "seed_stability": reference_cv <= normalized_thresholds["maximum_reference_cv"],
        "anti_goodhart_coverage": coverage
        >= normalized_thresholds["minimum_anti_goodhart_coverage"],
        "incremental_information": incremental_information
        >= normalized_thresholds["minimum_incremental_information_lower_bound"],
        "protected_metric_non_regression": observation["protected_metric_non_regression"] is True,
        "fresh_heldout_evidence": bool(observation["fresh_heldout_evidence_refs"]),
    }
    evidence_refs = sorted(
        set(
            [
                *[str(value) for value in observation["calibration_evidence_refs"]],
                *[str(value) for value in observation["fresh_heldout_evidence_refs"]],
            ]
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-metric-adequacy-report",
        "state": "adequate" if all(gates.values()) else "inadequate",
        "metric_id": metric_contract["metric_id"],
        "metric_contract_digest": _digest(metric_contract),
        "evaluator_binding": metric_contract["evaluator_binding"],
        "thresholds": normalized_thresholds,
        "measurements": {
            "reference_mean": reference_mean,
            "reference_cv": reference_cv,
            "regression_separation_lower_bound": separation_lower_bound,
            "anti_goodhart_coverage": coverage,
            "incremental_information_lower_bound": incremental_information,
        },
        "gates": gates,
        "evidence_refs": evidence_refs,
        "verdict_authority": False,
    }
    report["report_id"] = "metric-adequacy-" + _digest(report)[:24]
    report["report_digest"] = _digest_without(report, "report_digest")
    validate_metric_adequacy_report(report, root=root)
    return report


def validate_metric_adequacy_report(
    report: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate a stable adequacy report before policy consumes its reference."""

    try:
        validate_document("metric_adequacy_report", report, root=root)
    except ContractValidationError as exc:
        raise MetricAdequacyError(f"METRIC_ADEQUACY_REPORT_SCHEMA_INVALID:{exc}") from exc
    if report["report_digest"] != _digest_without(report, "report_digest"):
        raise MetricAdequacyError("METRIC_ADEQUACY_REPORT_DIGEST_MISMATCH")
    expected_id = "metric-adequacy-" + _digest_without(report, "report_id", "report_digest")[:24]
    if report["report_id"] != expected_id:
        raise MetricAdequacyError("METRIC_ADEQUACY_REPORT_ID_MISMATCH")
    if report["state"] == "adequate" and not all(report["gates"].values()):
        raise MetricAdequacyError("METRIC_ADEQUACY_REPORT_STATE_MISMATCH")
    if report["state"] == "inadequate" and all(report["gates"].values()):
        raise MetricAdequacyError("METRIC_ADEQUACY_REPORT_STATE_MISMATCH")


def _validate_observation(observation: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_document("metric_adequacy_observation", observation, root=root)
    except ContractValidationError as exc:
        raise MetricAdequacyError(f"METRIC_ADEQUACY_OBSERVATION_SCHEMA_INVALID:{exc}") from exc
    case_ids = [str(row["case_id"]) for row in observation["anti_goodhart_cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise MetricAdequacyError("METRIC_ADEQUACY_ANTI_GOODHART_CASE_DUPLICATE")
    for field in ("calibration_evidence_refs", "fresh_heldout_evidence_refs"):
        if any(
            not str(value).startswith(_CONTENT_REF_PREFIXES)
            for value in observation[field]
        ):
            raise MetricAdequacyError("METRIC_ADEQUACY_EVIDENCE_REF_INVALID:" + field)


def _validate_thresholds(thresholds: Mapping[str, float]) -> dict[str, float]:
    if set(thresholds) != set(_THRESHOLD_FIELDS):
        raise MetricAdequacyError("METRIC_ADEQUACY_THRESHOLDS_INVALID")
    normalized = {name: float(thresholds[name]) for name in _THRESHOLD_FIELDS}
    if not all(math.isfinite(value) for value in normalized.values()):
        raise MetricAdequacyError("METRIC_ADEQUACY_THRESHOLDS_NONFINITE")
    if normalized["maximum_reference_cv"] < 0:
        raise MetricAdequacyError("METRIC_ADEQUACY_THRESHOLD_CV_INVALID")
    coverage = normalized["minimum_anti_goodhart_coverage"]
    if not 0 <= coverage <= 1:
        raise MetricAdequacyError("METRIC_ADEQUACY_THRESHOLD_COVERAGE_INVALID")
    return normalized


def _separation_lower_bound(
    reference: list[float], regression: list[float], *, direction: str
) -> float:
    separation = statistics.fmean(reference) - statistics.fmean(regression)
    if direction == "minimize":
        separation = -separation
    variance = statistics.variance(reference) / len(reference) + statistics.variance(regression) / len(regression)
    return separation - 1.96 * math.sqrt(variance)


def _coefficient_of_variation(values: list[float], mean: float) -> float:
    deviation = statistics.stdev(values)
    if mean == 0:
        return 0.0 if deviation == 0 else float("inf")
    return abs(deviation / mean)


def _require_finite(values: list[float], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise MetricAdequacyError("METRIC_ADEQUACY_" + label + "_NONFINITE")


def _digest(document: Mapping[str, object]) -> str:
    return _digest_without(document, "report_id", "report_digest")


def _digest_without(document: Mapping[str, object], *fields: str) -> str:
    body = {key: value for key, value in document.items() if key not in fields}
    try:
        payload = json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MetricAdequacyError("METRIC_ADEQUACY_PAYLOAD_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()
