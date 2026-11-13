"""Independent paired statistics for open-method evaluation observations.

This module computes contrasts only. Its caller must independently establish
provenance, execution validity and archive binding before issuing a verdict.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.verify.statistics import student_t_mean_interval

ROLES = ("baseline", "source_only", "target_only", "combined")

def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _interval(values: Sequence[float]) -> dict[str, float]:
    try:
        return student_t_mean_interval(values)
    except ValueError as exc:
        raise ValueError("OPEN_STATISTICS_NUMERIC_INVALID") from exc


def paired_statistics(*, study: Mapping[str, object], verifier: Mapping[str, object],
                      observations: Sequence[Mapping[str, object]], phase: str,
                      expected_units: Sequence[str]) -> dict[str, object]:
    """Recompute episode-clustered paired contrasts from evaluator documents.

    Means are paired over seeds within each episode before estimating uncertainty
    across episodes. The interval assumes independent, approximately normal
    episode-level contrasts; it does not correct for multiple comparisons.
    Inputs must already be independently bound to a run; this function alone
    does not establish split independence or a formal effect.
    """
    try:
        validate_document("open_method_verifier", verifier)
    except ContractValidationError as exc:
        raise ValueError("OPEN_STATISTICS_VERIFIER_INVALID") from exc
    if phase not in ("selection", "confirmation"):
        raise ValueError("OPEN_STATISTICS_PHASE_INVALID")
    seeds = study.get("paired_seeds" if phase == "selection" else "confirmation_seeds")
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("OPEN_STATISTICS_SEEDS_INVALID")
    if (not isinstance(expected_units, (list, tuple)) or len(expected_units) < 2
            or any(not isinstance(unit, str) or not unit.strip() for unit in expected_units)
            or len(set(expected_units)) != len(expected_units)):
        raise ValueError("OPEN_STATISTICS_UNITS_INVALID")
    arms = study.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ROLES):
        raise ValueError("OPEN_STATISTICS_ARMS_INVALID")
    if any(not isinstance(arm, Mapping) or not isinstance(arm.get("role"), str) for arm in arms):
        raise ValueError("OPEN_STATISTICS_ARMS_INVALID")
    methods = {arm["role"]: arm.get("method_id") for arm in arms}
    if set(methods) != set(ROLES) or any(not isinstance(value, str) or not value for value in methods.values()):
        raise ValueError("OPEN_STATISTICS_ARMS_INVALID")
    metrics = [*verifier["primary_metrics"], *verifier["protected_metrics"]]
    if (len(set(metrics)) != len(metrics) or set(verifier["metric_policy"]) != set(metrics)
            or study.get("primary_metrics") != verifier["primary_metrics"]
            or study.get("protected_metrics") != verifier["protected_metrics"]):
        raise ValueError("OPEN_STATISTICS_METRICS_INVALID")
    for policy in verifier["metric_policy"].values():
        if any(not _finite_number(policy[name])
               for name in ("minimum_effect", "maximum_regression")):
            raise ValueError("OPEN_STATISTICS_METRICS_INVALID")
    required = {(role, seed) for role in ROLES for seed in seeds}
    data: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for document in observations:
        try:
            validate_document("open_method_evaluation", document)
        except ContractValidationError as exc:
            raise ValueError("OPEN_STATISTICS_OBSERVATION_INVALID") from exc
        role, seed = document["role"], document["seed"]
        key = (role, seed)
        if (key not in required or key in data or document["phase"] != phase
                or document["study_id"] != study.get("study_id")
                or document["method_id"] != methods[role]
                or document["verifier_id"] != verifier["verifier_id"]):
            raise ValueError("OPEN_STATISTICS_IDENTITY_INVALID")
        if not all(document["validity_gates"].get(gate) is True for gate in verifier["required_validity_gates"]):
            raise ValueError("OPEN_STATISTICS_GATE_FAILED")
        rows = document["unit_metrics"]
        units = {row["unit_id"]: row["metrics"] for row in rows}
        if len(units) != len(rows) or set(units) != set(expected_units):
            raise ValueError("OPEN_STATISTICS_UNIT_MISMATCH")
        for values in units.values():
            for metric in metrics:
                value = values.get(metric)
                if not _finite_number(value):
                    raise ValueError("OPEN_STATISTICS_NUMERIC_INVALID")
        data[key] = units
    if set(data) != required:
        raise ValueError("OPEN_STATISTICS_INCOMPLETE")
    result = {}
    for metric in metrics:
        sign = 1 if verifier["metric_policy"][metric]["direction"] == "maximize" else -1
        contrasts = {role: [] for role in ROLES[1:]}
        combined_vs_component = {role: [] for role in ("source_only", "target_only")}
        interactions = []
        for unit in sorted(expected_units):
            try:
                values = {role: statistics.fmean(sign * data[(role, seed)][unit][metric] for seed in seeds) for role in ROLES}
                for role in ROLES[1:]:
                    contrasts[role].append(values[role] - values["baseline"])
                for role in combined_vs_component:
                    combined_vs_component[role].append(values["combined"] - values[role])
                interactions.append(values["combined"] - values["source_only"] - values["target_only"] + values["baseline"])
            except (ArithmeticError, ValueError) as exc:
                raise ValueError("OPEN_STATISTICS_NUMERIC_INVALID") from exc
        result[metric] = {"effects_vs_baseline": {role: _interval(values) for role, values in contrasts.items()},
                          "combined_vs_component": {role: _interval(values) for role, values in combined_vs_component.items()},
                          "interaction": _interval(interactions)}
    return {"phase": phase, "paired_units": len(expected_units), "paired_seeds": len(seeds),
            "confidence_interval": "student_t_two_sided_95_episode_clustered",
            "metrics": result}
