"""Replay ACWM selector choices against settled, non-leaking effect labels."""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.spec import SELECTORS


class SelectorReplayError(ValueError):
    """Selector replay inputs are malformed or violate the held-out contract."""


def run_selector_replay(
    *,
    plan_path: Path,
    projection_path: Path,
    effect_label_index: Path,
    output_root: Path,
    primitive_probe_affinity: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    plan = _load_json(plan_path, "SELECTOR_REPLAY_PLAN_INVALID")
    projections = _load_jsonl(projection_path, "SELECTOR_REPLAY_PROJECTIONS_INVALID")
    label_index = _load_json(effect_label_index, "SELECTOR_REPLAY_LABELS_INVALID")
    labels = label_index.get("labels")
    trials = plan.get("trials")
    if not isinstance(labels, list) or any(not isinstance(row, Mapping) for row in labels):
        raise SelectorReplayError("SELECTOR_REPLAY_LABELS_INVALID")
    if not isinstance(trials, list) or any(not isinstance(row, Mapping) for row in trials):
        raise SelectorReplayError("SELECTOR_REPLAY_TRIALS_INVALID")

    projection_map = _validate_projections(projections)
    affinity = _load_probe_affinity(primitive_probe_affinity) if primitive_probe_affinity is not None else None
    effects = _aggregate_effects(labels)
    distances = _fold_distances(trials, projection_map)
    cell_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    probe_work_orders: dict[tuple[str, str, str], dict[str, object]] = {}
    fold_dispositions: dict[tuple[str, str], dict[str, object]] = {}
    for trial in trials:
        target = str(trial["target_environment"])
        selector = str(trial["selector"])
        sources = tuple(str(value) for value in trial["source_environments"])
        if target in sources:
            raise SelectorReplayError("SELECTOR_REPLAY_TARGET_LEAK")
        target_effects = {
            primitive: effect
            for (environment, primitive), effect in effects.items()
            if environment == target and effect["consensus_state"] != "ambiguous"
        }
        ambiguous = sorted(
            primitive
            for (environment, primitive), effect in effects.items()
            if environment == target and effect["consensus_state"] == "ambiguous"
        )
        supported: dict[str, dict[str, object]] = {}
        unsupported: list[str] = []
        geometry_gaps: list[dict[str, object]] = []
        for primitive, target_effect in sorted(target_effects.items()):
            source_effects = [
                effects[(source, primitive)]
                for source in sources
                if (source, primitive) in effects
                and effects[(source, primitive)]["consensus_state"] != "ambiguous"
            ]
            if not source_effects:
                unsupported.append(primitive)
                continue
            geometry = _candidate_geometry(
                target=target,
                primitive=primitive,
                sources=sources,
                selector=selector,
                projections=projection_map,
                affinity=affinity,
            )
            if selector == "irg" and affinity is not None and geometry["state"] != "covered":
                gap = {
                    "primitive": primitive,
                    "state": geometry["state"],
                    "required_probe_paths": geometry["required_probe_paths"],
                    "reason": geometry["reason"],
                    "successor_probe_axis": geometry.get("successor_probe_axis"),
                }
                geometry_gaps.append(gap)
                key = (target, primitive, str(geometry["reason"]))
                probe_work_orders.setdefault(
                    key,
                    {
                        "target_environment": target,
                        "primitive": primitive,
                        "reason": geometry["reason"],
                        "required_probe_paths": geometry["required_probe_paths"],
                        "successor_probe_axis": geometry.get("successor_probe_axis"),
                        "action": "materialize_counterexample_driven_probe_then_replay",
                    },
                )
                continue
            source_rows = []
            for source_effect in source_effects:
                source = str(source_effect["environment"])
                candidate_distance = (
                    geometry["distances"].get(source)
                    if selector == "irg" and affinity is not None
                    else distances[(target, selector, source)]
                )
                if candidate_distance is None:
                    continue
                source_rows.append(
                    {
                        **source_effect,
                        "distance": candidate_distance,
                    }
                )
            if not source_rows:
                gap = {
                    "primitive": primitive,
                    "state": "source_path_support_missing",
                    "required_probe_paths": geometry["required_probe_paths"],
                    "reason": "no_nonleaking_source_has_all_required_local_paths",
                    "successor_probe_axis": geometry.get("successor_probe_axis"),
                }
                geometry_gaps.append(gap)
                key = (target, primitive, str(gap["reason"]))
                probe_work_orders.setdefault(
                    key,
                    {
                        "target_environment": target,
                        "primitive": primitive,
                        "reason": gap["reason"],
                        "required_probe_paths": gap["required_probe_paths"],
                        "successor_probe_axis": gap["successor_probe_axis"],
                        "action": "evolve_nonlocal_probe_path_then_replay",
                    },
                )
                continue
            probability = _distance_weighted_probability(source_rows)
            nearest = min(float(row["distance"]) for row in source_rows)
            supported[primitive] = {
                "primitive": primitive,
                "source_positive_probability": probability,
                "nearest_source_distance": nearest,
                "ranking_score": probability / (1.0 + nearest),
                "predicted_sign": "positive" if probability > 0.5 else "negative" if probability < 0.5 else "abstain",
                "target_positive": bool(target_effect["positive_rate"] == 1.0),
                "target_psnr_delta": target_effect["mean_psnr_delta"],
                "source_effects": source_rows,
                "geometry_coverage_state": geometry["state"],
                "required_probe_paths": geometry["required_probe_paths"],
            }
        disposition_key = (target, selector)
        minimum_geometry_candidates = int(affinity["minimum_covered_candidates_per_fold"]) if affinity else 0
        strict_geometry_coverage = bool(affinity["require_full_candidate_coverage"]) if affinity else False
        if selector == "irg" and affinity is not None and (
            (strict_geometry_coverage and geometry_gaps)
            or len(supported) < minimum_geometry_candidates
        ):
            reason = "candidate_probe_coverage_incomplete" if geometry_gaps else "insufficient_geometry_supported_candidates"
            cell = {
                "trial_id": trial["trial_id"],
                "fold_id": trial["fold_id"],
                "target_environment": target,
                "selector": selector,
                "seed": int(trial["seed"]),
                "state": "abstained",
                "abstention_reason": reason,
                "supported_candidate_count": len(supported),
                "unsupported_target_candidates": unsupported,
                "ambiguous_target_candidates": ambiguous,
                "geometry_gaps": geometry_gaps,
            }
            cell_rows.append(cell)
            fold_dispositions[disposition_key] = cell
            if not geometry_gaps:
                key = (target, "*", reason)
                probe_work_orders.setdefault(
                    key,
                    {
                        "target_environment": target,
                        "primitive": None,
                        "reason": reason,
                        "required_probe_paths": [],
                        "successor_probe_axis": None,
                        "action": "expand_mechanism_distinct_candidate_pool",
                    },
                )
            continue
        if not supported:
            cell = {
                "trial_id": trial["trial_id"],
                "fold_id": trial["fold_id"],
                "target_environment": target,
                "selector": selector,
                "seed": int(trial["seed"]),
                "state": "abstained",
                "abstention_reason": (
                    "target_labels_ambiguous" if ambiguous and not target_effects else "no_nonleaking_source_support"
                ),
                "supported_candidate_count": 0,
                "unsupported_target_candidates": unsupported,
                "ambiguous_target_candidates": ambiguous,
            }
            cell_rows.append(cell)
            fold_dispositions[disposition_key] = cell
            continue

        ranked = sorted(supported.values(), key=lambda row: (-float(row["ranking_score"]), str(row["primitive"])))
        for rank, candidate in enumerate(ranked, start=1):
            candidate_rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "target_environment": target,
                    "selector": selector,
                    "seed": int(trial["seed"]),
                    "rank": rank,
                    **candidate,
                }
            )
        selected = ranked[0]
        certificate = _transfer_certificate(selector=selector, affinity=affinity, selected=selected)
        if not certificate["passed"]:
            cell = {
                "trial_id": trial["trial_id"],
                "fold_id": trial["fold_id"],
                "target_environment": target,
                "selector": selector,
                "seed": int(trial["seed"]),
                "state": "abstained",
                "abstention_reason": "transfer_certificate_failed",
                "supported_candidate_count": len(ranked),
                "unsupported_target_candidates": unsupported,
                "ambiguous_target_candidates": ambiguous,
                "selected_primitive_before_certificate": selected["primitive"],
                "transfer_certificate": certificate,
            }
            cell_rows.append(cell)
            fold_dispositions[disposition_key] = cell
            continue
        positive_ranks = [rank for rank, row in enumerate(ranked, start=1) if row["target_positive"]]
        sign_pairs = [
            row
            for row in ranked
            if row["predicted_sign"] != "abstain"
        ]
        sign_accuracy = (
            fmean(
                float((row["predicted_sign"] == "positive") is bool(row["target_positive"]))
                for row in sign_pairs
            )
            if sign_pairs
            else None
        )
        target_order = sorted(
            ranked,
            key=lambda row: (
                -int(bool(row["target_positive"])),
                -float(row["target_psnr_delta"]),
                str(row["primitive"]),
            ),
        )
        cell = {
            "trial_id": trial["trial_id"],
            "fold_id": trial["fold_id"],
            "target_environment": target,
            "selector": selector,
            "seed": int(trial["seed"]),
            "state": "evaluated",
            "abstention_reason": None,
            "supported_candidate_count": len(ranked),
            "unsupported_target_candidates": unsupported,
            "ambiguous_target_candidates": ambiguous,
            "selected_primitive": selected["primitive"],
            "selected_target_positive": selected["target_positive"],
            "top1_positive_hit": float(bool(selected["target_positive"])),
            "negative_selection": float(not bool(selected["target_positive"])),
            "benefit_sign_accuracy": sign_accuracy,
            "ranking_kendall_tau": _kendall_tau(ranked, target_order),
            "selection_regret": float(any(row["target_positive"] for row in ranked) and not selected["target_positive"]),
            "trials_to_first_positive": min(positive_ranks) if positive_ranks else None,
            "gpu_hours": 0.0,
        }
        cell_rows.append(cell)
        fold_dispositions[disposition_key] = cell

    aggregate_rows = _aggregate_selectors(cell_rows)
    environments = sorted({str(row["target_environment"]) for row in trials})
    evaluated_environments = sorted(
        {str(row["target_environment"]) for row in cell_rows if row["state"] == "evaluated"}
    )
    fully_evaluated_environments = sorted(
        environment
        for environment in environments
        if all(
            any(
                row["target_environment"] == environment
                and row["selector"] == selector
                and row["state"] == "evaluated"
                for row in cell_rows
            )
            for selector in SELECTORS
        )
    )
    representative_cells = {
        (str(row["target_environment"]), str(row["selector"])): row
        for row in cell_rows
        if row["state"] == "evaluated"
    }
    multi_candidate_environments = sorted(
        {
            environment
            for (environment, _selector), row in representative_cells.items()
            if int(row["supported_candidate_count"]) >= 2
        }
    )
    choices_by_environment: dict[str, set[str]] = defaultdict(set)
    for (environment, _selector), row in representative_cells.items():
        choices_by_environment[environment].add(str(row["selected_primitive"]))
    choice_divergence_environments = sorted(
        environment for environment, choices in choices_by_environment.items() if len(choices) > 1
    )
    formal_ready = (
        len(fully_evaluated_environments) == len(environments)
        and len(multi_candidate_environments) == len(environments)
    )
    selection_discrimination_ready = formal_ready and bool(choice_divergence_environments)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-selector-cpu-replay",
        "state": "ready" if formal_ready else "partial",
        "claim_boundary": (
            "CPU replay of settled target-local labels under leave-one-environment-out source exclusion. "
            "It measures selector behavior on observed label support and cannot replace new GPU confirmation receipts."
        ),
        "planned_cell_count": len(trials),
        "dispositioned_cell_count": len(cell_rows),
        "evaluated_cell_count": sum(row["state"] == "evaluated" for row in cell_rows),
        "abstained_cell_count": sum(row["state"] == "abstained" for row in cell_rows),
        "environment_count": len(environments),
        "evaluated_environment_count": len(evaluated_environments),
        "evaluated_environments": evaluated_environments,
        "fully_evaluated_environment_count": len(fully_evaluated_environments),
        "fully_evaluated_environments": fully_evaluated_environments,
        "multi_candidate_environment_count": len(multi_candidate_environments),
        "multi_candidate_environments": multi_candidate_environments,
        "selector_choice_divergence_environment_count": len(choice_divergence_environments),
        "selector_choice_divergence_environments": choice_divergence_environments,
        "formal_comparison_ready": formal_ready,
        "selection_discrimination_ready": selection_discrimination_ready,
        "seed_replicates_identical_by_contract": True,
        "distance_contract": (
            "primitive-conditioned source-fold z-score on affinity-certified IRG paths followed by Euclidean distance"
            if affinity is not None
            else "source-fold z-score followed by Euclidean distance"
        ),
        "ranking_contract": "source positive rate divided by one plus nearest source distance",
        "target_label_contract": "unanimous settled gate sign; mixed checkpoint outcomes abstain",
        "primitive_probe_affinity_enabled": affinity is not None,
        "probe_coverage_ready": not probe_work_orders,
        "probe_evolution_work_order_count": len(probe_work_orders),
        "probe_evolution_work_orders": list(probe_work_orders.values()),
        "transfer_certificate_enabled": bool(affinity and affinity.get("transfer_certificate")),
        "transfer_certificate_abstention_count": sum(
            row.get("abstention_reason") == "transfer_certificate_failed" for row in cell_rows
        ),
        "selectors": aggregate_rows,
        "cells": cell_rows,
    }
    destination = Path(output_root).resolve()
    files = {
        "selector-replay.json": canonical_json(report),
        "selector-replay.md": _markdown(report).encode("utf-8"),
        "tables/cells.csv": _csv(cell_rows).encode("utf-8"),
        "tables/candidates.csv": _csv(candidate_rows).encode("utf-8"),
        "tables/selector-metrics.csv": _csv(aggregate_rows).encode("utf-8"),
        "tables/probe-evolution-work-orders.csv": _csv(list(probe_work_orders.values())).encode("utf-8"),
        "input-plan.json": canonical_json(plan),
        "input-effect-label-index.json": canonical_json(label_index),
    }
    if affinity is not None:
        files["input-primitive-probe-affinity.json"] = canonical_json(affinity)
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-acwm-selector-cpu-replay-manifest",
            "state": report["state"],
            "planned_cell_count": len(trials),
            "evaluated_cell_count": report["evaluated_cell_count"],
            "abstained_cell_count": report["abstained_cell_count"],
            "evaluated_environment_count": len(evaluated_environments),
            "formal_comparison_ready": report["formal_comparison_ready"],
            "selection_discrimination_ready": report["selection_discrimination_ready"],
            "probe_coverage_ready": report["probe_coverage_ready"],
            "probe_evolution_work_order_count": report["probe_evolution_work_order_count"],
            "transfer_certificate_abstention_count": report["transfer_certificate_abstention_count"],
            "multi_candidate_environment_count": len(multi_candidate_environments),
            "report_path": str(destination / "selector-replay.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_projections(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, object]]:
    projections: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("environment")), str(row.get("selector")))
        values = row.get("features")
        if key[1] not in SELECTORS or not isinstance(values, list) or not values:
            raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_ROW_INVALID")
        vector = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in vector) or key in projections:
            raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_ROW_INVALID")
        raw_names = row.get("feature_names")
        if raw_names is None:
            names = tuple(f"feature:{index}" for index in range(len(vector)))
        elif (
            not isinstance(raw_names, list)
            or len(raw_names) != len(vector)
            or any(not isinstance(value, str) or not value for value in raw_names)
        ):
            raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_ROW_INVALID")
        else:
            names = tuple(raw_names)
        projections[key] = {"features": vector, "feature_names": names}
    return projections


def _load_probe_affinity(path: Path) -> Mapping[str, Any]:
    payload = _load_json(Path(path), "SELECTOR_PROBE_AFFINITY_INVALID")
    if payload.get("artifact_type") != "verdiwm-primitive-probe-affinity":
        raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_TYPE_INVALID")
    path_order = payload.get("projection_path_order")
    primitives = payload.get("primitives")
    minimum = payload.get("minimum_covered_candidates_per_fold")
    if (
        not isinstance(path_order, list)
        or not path_order
        or len(set(path_order)) != len(path_order)
        or any(not isinstance(value, str) or not value for value in path_order)
        or not isinstance(primitives, Mapping)
        or not isinstance(minimum, int)
        or minimum < 2
        or not isinstance(payload.get("require_full_candidate_coverage"), bool)
    ):
        raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_INVALID")
    for primitive, record in primitives.items():
        if not isinstance(primitive, str) or not primitive or not isinstance(record, Mapping):
            raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_INVALID")
        required = record.get("required_probe_paths")
        state = record.get("coverage_state")
        if (
            not isinstance(required, list)
            or not required
            or any(not isinstance(value, str) or not value for value in required)
            or state not in {"covered", "probe_missing"}
        ):
            raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_INVALID")
        if state == "covered" and any(value not in path_order for value in required):
            raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_PATH_UNKNOWN")
        if state == "probe_missing" and not isinstance(record.get("successor_probe_axis"), str):
            raise SelectorReplayError("SELECTOR_PROBE_AFFINITY_SUCCESSOR_INVALID")
    certificate = payload.get("transfer_certificate")
    if certificate is not None:
        if not isinstance(certificate, Mapping):
            raise SelectorReplayError("SELECTOR_TRANSFER_CERTIFICATE_INVALID")
        minimum_sources = certificate.get("minimum_nonleaking_source_environments")
        minimum_probability = certificate.get("minimum_selected_positive_probability")
        if (
            not isinstance(minimum_sources, int)
            or minimum_sources < 2
            or not isinstance(minimum_probability, (int, float))
            or not 0.5 < float(minimum_probability) <= 1.0
            or certificate.get("failure_policy") != "abstain_fold"
        ):
            raise SelectorReplayError("SELECTOR_TRANSFER_CERTIFICATE_INVALID")
    return payload


def _transfer_certificate(
    *,
    selector: str,
    affinity: Mapping[str, Any] | None,
    selected: Mapping[str, Any],
) -> dict[str, object]:
    policy = affinity.get("transfer_certificate") if affinity is not None else None
    if selector != "irg" or not isinstance(policy, Mapping):
        return {"enabled": False, "passed": True, "checks": {}}
    source_count = len(selected["source_effects"])
    probability = float(selected["source_positive_probability"])
    minimum_sources = int(policy["minimum_nonleaking_source_environments"])
    minimum_probability = float(policy["minimum_selected_positive_probability"])
    checks = {
        "nonleaking_source_environment_count": {
            "observed": source_count,
            "required": minimum_sources,
            "passed": source_count >= minimum_sources,
        },
        "selected_positive_probability": {
            "observed": probability,
            "required": minimum_probability,
            "passed": probability >= minimum_probability,
        },
    }
    return {
        "enabled": True,
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "failure_policy": policy["failure_policy"],
    }


def _candidate_geometry(
    *,
    target: str,
    primitive: str,
    sources: Sequence[str],
    selector: str,
    projections: Mapping[tuple[str, str], Mapping[str, object]],
    affinity: Mapping[str, Any] | None,
) -> dict[str, object]:
    if selector != "irg" or affinity is None:
        return {
            "state": "not_applicable",
            "reason": None,
            "required_probe_paths": [],
            "distances": {},
        }
    records = affinity["primitives"]
    record = records.get(primitive) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        return {
            "state": "probe_contract_missing",
            "reason": "primitive_probe_affinity_missing",
            "required_probe_paths": [],
            "successor_probe_axis": f"{primitive}_mechanism_response",
            "distances": {},
        }
    required = tuple(str(value) for value in record["required_probe_paths"])
    if record.get("coverage_state") != "covered":
        return {
            "state": "probe_missing",
            "reason": "required_probe_axis_not_materialized",
            "required_probe_paths": list(required),
            "successor_probe_axis": record.get("successor_probe_axis"),
            "distances": {},
        }
    try:
        distances = _conditioned_irg_distances(
            target=target,
            sources=sources,
            projections=projections,
            required_paths=required,
            path_order=tuple(str(value) for value in affinity["projection_path_order"]),
        )
    except KeyError as exc:
        raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_MISSING") from exc
    if distances is None:
        successor = record.get("successor_probe_axis") or f"{required[0]}_locality_successor"
        return {
            "state": "target_path_support_missing",
            "reason": "target_required_probe_path_is_nonlocal_or_missing",
            "required_probe_paths": list(required),
            "successor_probe_axis": successor,
            "distances": {},
        }
    return {
        "state": "covered",
        "reason": None,
        "required_probe_paths": list(required),
        "successor_probe_axis": record.get("successor_probe_axis") or f"{required[0]}_locality_successor",
        "distances": distances,
    }


def _conditioned_irg_distances(
    *,
    target: str,
    sources: Sequence[str],
    projections: Mapping[tuple[str, str], Mapping[str, object]],
    required_paths: Sequence[str],
    path_order: Sequence[str],
) -> dict[str, float] | None:
    target_projection = projections[(target, "irg")]
    indices = _irg_path_feature_indices(target_projection, required_paths=required_paths, path_order=path_order)
    if not indices or not _projection_supports_paths(target_projection, required_paths):
        return None
    eligible_sources = [
        source
        for source in sources
        if _projection_supports_paths(projections[(source, "irg")], required_paths)
    ]
    if not eligible_sources:
        return {}
    target_vector = tuple(float(value) for value in target_projection["features"])
    source_vectors = [tuple(float(value) for value in projections[(source, "irg")]["features"]) for source in eligible_sources]
    means = tuple(fmean(vector[index] for vector in source_vectors) for index in indices)
    scales = tuple(
        math.sqrt(fmean((vector[index] - means[position]) ** 2 for vector in source_vectors))
        for position, index in enumerate(indices)
    )
    distances: dict[str, float] = {}
    for source, vector in zip(eligible_sources, source_vectors, strict=True):
        squared = 0.0
        for position, index in enumerate(indices):
            scale = scales[position]
            if scale > 1e-12:
                squared += ((target_vector[index] - vector[index]) / scale) ** 2
        distances[source] = math.sqrt(squared)
    return distances


def _projection_supports_paths(projection: Mapping[str, object], required_paths: Sequence[str]) -> bool:
    names = tuple(str(value) for value in projection["feature_names"])
    features = tuple(float(value) for value in projection["features"])
    support = {
        name.split(":", 1)[1]: features[index] >= 0.5
        for index, name in enumerate(names)
        if name.startswith("path_supported:")
    }
    return all(support.get(path) is True for path in required_paths)


def _irg_path_feature_indices(
    projection: Mapping[str, object],
    *,
    required_paths: Sequence[str],
    path_order: Sequence[str],
) -> tuple[int, ...]:
    selected = set(required_paths)
    names = tuple(str(value) for value in projection["feature_names"])
    result: list[int] = []
    for index, name in enumerate(names):
        path: str | None = None
        if name.startswith("response_coordinate:") or name.startswith("covariance_diagonal:"):
            suffix = name.rsplit(":", 1)[1]
            if suffix.isdigit():
                path = path_order[int(suffix) % len(path_order)]
            elif suffix in path_order:
                path = suffix
        elif name.startswith("locality:") or name.startswith("path_supported:"):
            path = name.split(":", 1)[1]
        if path in selected:
            result.append(index)
    return tuple(result)


def _aggregate_effects(labels: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in labels:
        if row.get("settled") is not True or not isinstance(row.get("environment"), str) or not isinstance(row.get("primitive"), str):
            continue
        if not isinstance(row.get("positive"), bool):
            continue
        grouped[(str(row["environment"]), str(row["primitive"]))].append(row)
    effects: dict[tuple[str, str], dict[str, object]] = {}
    for (environment, primitive), rows in sorted(grouped.items()):
        signs = [bool(row["positive"]) for row in rows]
        psnr = [
            float(row["delta_candidate_minus_baseline"]["psnr"])
            for row in rows
            if isinstance(row.get("delta_candidate_minus_baseline"), Mapping)
            and isinstance(row["delta_candidate_minus_baseline"].get("psnr"), (int, float))
        ]
        rate = fmean(float(value) for value in signs)
        effects[(environment, primitive)] = {
            "environment": environment,
            "primitive": primitive,
            "positive_rate": rate,
            "consensus_state": "positive" if rate == 1.0 else "negative" if rate == 0.0 else "ambiguous",
            "mean_psnr_delta": fmean(psnr) if psnr else 0.0,
            "receipt_count": len(rows),
        }
    return effects


def _fold_distances(
    trials: Sequence[Mapping[str, Any]],
    projections: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str, str], float]:
    distances: dict[tuple[str, str, str], float] = {}
    frames = {
        (str(row["target_environment"]), str(row["selector"])): tuple(str(value) for value in row["source_environments"])
        for row in trials
    }
    for (target, selector), sources in frames.items():
        try:
            target_vector = tuple(float(value) for value in projections[(target, selector)]["features"])
            source_vectors = [
                tuple(float(value) for value in projections[(source, selector)]["features"])
                for source in sources
            ]
        except KeyError as exc:
            raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_MISSING") from exc
        dimensions = {len(target_vector), *(len(vector) for vector in source_vectors)}
        if len(dimensions) != 1:
            raise SelectorReplayError("SELECTOR_REPLAY_PROJECTION_DIMENSION_MISMATCH")
        means = tuple(fmean(vector[index] for vector in source_vectors) for index in range(len(target_vector)))
        scales = tuple(
            math.sqrt(fmean((vector[index] - means[index]) ** 2 for vector in source_vectors))
            for index in range(len(target_vector))
        )
        for source, vector in zip(sources, source_vectors, strict=True):
            squared = 0.0
            for index, (target_value, source_value) in enumerate(zip(target_vector, vector, strict=True)):
                scale = scales[index]
                if scale > 1e-12:
                    squared += ((target_value - source_value) / scale) ** 2
            distances[(target, selector, source)] = math.sqrt(squared)
    return distances


def _distance_weighted_probability(rows: Sequence[Mapping[str, Any]]) -> float:
    weights = [1.0 / (1.0 + float(row["distance"])) for row in rows]
    return sum(weight * float(row["positive_rate"]) for weight, row in zip(weights, rows, strict=True)) / sum(weights)


def _kendall_tau(predicted: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]) -> float | None:
    if len(predicted) < 2:
        return None
    predicted_rank = {str(row["primitive"]): index for index, row in enumerate(predicted)}
    actual_rank = {str(row["primitive"]): index for index, row in enumerate(actual)}
    concordant = 0
    discordant = 0
    names = sorted(predicted_rank)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            product = (predicted_rank[left] - predicted_rank[right]) * (actual_rank[left] - actual_rank[right])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def _aggregate_selectors(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selector in SELECTORS:
        selector_cells = [row for row in cells if row["selector"] == selector]
        by_environment: dict[str, Mapping[str, Any]] = {}
        for row in selector_cells:
            by_environment.setdefault(str(row["target_environment"]), row)
        evaluated = [row for row in by_environment.values() if row["state"] == "evaluated"]
        aggregate: dict[str, object] = {
            "selector": selector,
            "planned_cell_count": len(selector_cells),
            "evaluated_cell_count": sum(row["state"] == "evaluated" for row in selector_cells),
            "abstained_cell_count": sum(row["state"] == "abstained" for row in selector_cells),
            "evaluated_fold_count": len(evaluated),
            "gpu_hours": 0.0,
        }
        for field in (
            "top1_positive_hit",
            "benefit_sign_accuracy",
            "ranking_kendall_tau",
            "selection_regret",
            "trials_to_first_positive",
            "negative_selection",
        ):
            values = [float(row[field]) for row in evaluated if isinstance(row.get(field), (int, float))]
            aggregate[field] = fmean(values) if values else None
            aggregate[f"{field}_fold_count"] = len(values)
            aggregate[f"{field}_bootstrap_95"] = _bootstrap_interval(values, seed=20260728) if values else None
        rows.append(aggregate)
    return rows


def _bootstrap_interval(values: Sequence[float], *, seed: int, repeats: int = 2000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(fmean(rng.choice(values) for _ in values) for _ in range(repeats))
    return [means[int(0.025 * (repeats - 1))], means[int(0.975 * (repeats - 1))]]


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorReplayError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SelectorReplayError(f"{code}:{path}")
    return payload


def _load_jsonl(path: Path, code: str) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorReplayError(f"{code}:{path}") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise SelectorReplayError(f"{code}:{path}")
    return rows


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM-Phys Selector CPU Replay",
        "",
        f"State: `{report['state']}`",
        f"Evaluated cells: `{report['evaluated_cell_count']}/{report['planned_cell_count']}`",
        f"Evaluated environments: `{report['evaluated_environment_count']}/{report['environment_count']}`",
        f"Multi-candidate environments: `{report['multi_candidate_environment_count']}/{report['environment_count']}`",
        f"Environments with selector choice divergence: `{report['selector_choice_divergence_environment_count']}`",
        f"Formal comparison ready: `{str(report['formal_comparison_ready']).lower()}`",
        f"Selection discrimination ready: `{str(report['selection_discrimination_ready']).lower()}`",
        "",
        "| Selector | Folds | Top-1 positive | Negative selection | Sign accuracy | Kendall tau | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["selectors"]:
        def value(name: str) -> str:
            item = row[name]
            return "NA" if item is None else f"{float(item):.4f}"

        lines.append(
            f"| {row['selector']} | {row['evaluated_fold_count']} | {value('top1_positive_hit')} | "
            f"{value('negative_selection')} | {value('benefit_sign_accuracy')} | "
            f"{value('ranking_kendall_tau')} | {value('selection_regret')} |"
        )
    lines.append("")
    if int(report["abstained_cell_count"]) > 0:
        lines.append(
            "Abstained folds remain evidence gaps. They must be filled with matched official-gate labels before GPU confirmation or a formal selector comparison."
        )
    elif report.get("selection_discrimination_ready") is not True:
        lines.append(
            "The evidence matrix is complete, but all selectors make the same top-1 choices. Ranking and sign metrics are reportable; selector-choice superiority is not identified."
        )
    return "\n".join(lines) + "\n"
