#!/usr/bin/env python3
"""Aggregate CCLVR v1 held-out shards into a fail-closed local promotion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ARM_DOSES = (-0.99, 0.0, 0.99)
PLAN_TYPES = {
    "verdiwm-ctrl-world-cclvr-heldout-plan-v1",
    "verdiwm-ctrl-world-cclvr-heldout-plan-v2",
}
QUALITY_METRICS = (
    "mean_l1",
    "final_interaction_l1",
    "horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class CCLVRHeldoutAggregationError(RuntimeError):
    """Held-out CCLVR artifacts do not match the frozen evaluation frame."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRHeldoutAggregationError(f"CCLVR_HELDOUT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRHeldoutAggregationError(f"CCLVR_HELDOUT_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: object, error: str = "CCLVR_HELDOUT_VALUE_INVALID") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CCLVRHeldoutAggregationError(error) from exc
    if not math.isfinite(result):
        raise CCLVRHeldoutAggregationError(error)
    return result


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    raw = row.get("identity")
    if not isinstance(raw, Mapping):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_IDENTITY_INVALID")
    identity = (
        str(raw.get("context_id", "")),
        str(raw.get("episode_id", "")),
        int(raw.get("start_idx", -1)),
        int(raw.get("seed", -1)),
    )
    if not identity[0] or identity[1] != "1799" or identity[2] < 0 or identity[3] < 0:
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_IDENTITY_INVALID")
    return identity


def _interactions(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = row.get("interactions")
    if not isinstance(records, list) or len(records) != 4 or any(not isinstance(value, Mapping) for value in records):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_INTERACTIONS_INVALID")
    for index, record in enumerate(records):
        if int(record.get("interaction", -1)) != index:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_INTERACTIONS_INVALID")
        for key in ("mean_l1", "final_l1", "mean_psnr", "final_psnr"):
            _finite(record.get(key), "CCLVR_HELDOUT_INTERACTIONS_INVALID")
    return list(records)


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def local_error(
    row: Mapping[str, Any],
    interaction: int,
    *,
    suffix_weight: float,
    terminal_weight: float,
    slope_weight: float,
) -> dict[str, float]:
    records = _interactions(row)
    suffix = [_finite(record["mean_l1"]) for record in records[interaction:]]
    terminal = _finite(records[-1]["final_l1"])
    slope = _slope(suffix)
    suffix_mean = float(np.mean(suffix))
    return {
        "loss": suffix_weight * suffix_mean + terminal_weight * terminal + slope_weight * slope,
        "suffix_mean_l1": suffix_mean,
        "terminal_interaction_l1": terminal,
        "suffix_horizon_l1_slope": slope,
    }


def _quality(row: Mapping[str, Any]) -> dict[str, float]:
    outcomes = row.get("outcomes")
    if not isinstance(outcomes, Mapping):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_OUTCOME_INVALID")
    return {
        "mean_l1": -_finite(outcomes.get("negative_mean_l1")),
        "final_interaction_l1": -_finite(outcomes.get("negative_final_interaction_l1")),
        "horizon_l1_slope": -_finite(outcomes.get("negative_horizon_l1_slope")),
        "mean_psnr": _finite(outcomes.get("mean_psnr")),
        "final_psnr": _finite(outcomes.get("final_psnr")),
    }


def _response(row: Mapping[str, Any]) -> np.ndarray:
    response = row.get("prediction_response")
    if not isinstance(response, Mapping) or not isinstance(response.get("values"), list):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_RESPONSE_INVALID")
    values = np.asarray(response["values"], dtype=np.float64)
    shape = response.get("shape")
    if not isinstance(shape, list) or int(np.prod(shape)) != values.size or not np.isfinite(values).all():
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_RESPONSE_INVALID")
    return values


def _mean_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROWS_EMPTY")
    return {name: float(np.mean([_finite(row[name]) for row in rows])) for name in QUALITY_METRICS}


def _load_routing_rows(
    *,
    plan: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, str]]]:
    cells = plan.get("cells")
    jobs = plan.get("routing_jobs")
    if not isinstance(cells, list) or not isinstance(jobs, list):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_PLAN_INVALID")
    cell_map = {str(row["id"]): row for row in cells if isinstance(row, Mapping)}
    expected_seeds = {int(value) for value in plan["seeds"]}
    output: dict[str, dict[int, dict[str, Any]]] = {cell_id: {} for cell_id in cell_map}
    receipts: list[dict[str, str]] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_PLAN_INVALID")
        cell_id = str(job.get("cell_id"))
        if cell_id not in cell_map:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_JOB_CELL_INVALID")
        path = root / str(job.get("output_rel")) / "result.json"
        result = _load(path)
        if (
            result.get("artifact_type") != "verdiwm-ctrl-world-cclvr-heldout-shard-v1"
            or result.get("state") != "ready"
            or result.get("mode") != "routing"
            or result.get("cell_id") != cell_id
            or result.get("route_scope") != cell_map[cell_id].get("route_scope")
        ):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_SHARD_INVALID")
        raw_rows = result.get("measurements")
        if not isinstance(raw_rows, list):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_SHARD_INVALID")
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_ROW_INVALID")
            grouped.setdefault(_identity(raw)[3], []).append(raw)
        expected_job_seeds = {int(value) for value in job.get("seeds", ())}
        if set(grouped) != expected_job_seeds:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_SEED_FRAME_INVALID")
        for seed, seed_rows in grouped.items():
            if seed in output[cell_id]:
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_SEED_DUPLICATE")
            zero_rows = [row for row in seed_rows if row.get("kind") == "fixed_zero"]
            learned_rows = [row for row in seed_rows if row.get("kind") == "learned_cached"]
            endpoints: dict[tuple[int, float], Mapping[str, Any]] = {}
            for row in seed_rows:
                if row.get("kind") != "fixed_endpoint":
                    continue
                key = (int(row.get("target_interaction", -1)), _finite(row.get("dose")))
                if key in endpoints:
                    raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ENDPOINT_DUPLICATE")
                endpoints[key] = row
            expected_endpoints = {(interaction, dose) for interaction in range(4) for dose in (ARM_DOSES[0], ARM_DOSES[2])}
            if len(zero_rows) != 1 or len(learned_rows) != 1 or set(endpoints) != expected_endpoints:
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_FRAME_INVALID")
            output[cell_id][seed] = {
                "zero": zero_rows[0],
                "learned": learned_rows[0],
                "endpoints": endpoints,
            }
        receipts.append({"path": str(path), "sha256": _sha256(path)})
    if any(set(rows) != expected_seeds for rows in output.values()):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ROUTING_COVERAGE_INVALID")
    return output, receipts


def _decisions_by_interaction(learned: Mapping[str, Any], route_scope: str) -> dict[int, Mapping[str, Any]]:
    audit = learned.get("route_audit")
    if not isinstance(audit, Mapping) or not isinstance(audit.get("decision_records"), list):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_DECISIONS_INVALID")
    records = audit["decision_records"]
    if route_scope == "episode":
        if len(records) != 1 or not isinstance(records[0], Mapping):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_DECISIONS_INVALID")
        return {interaction: records[0] for interaction in range(4)}
    if len(records) != 4 or any(not isinstance(record, Mapping) for record in records):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_DECISIONS_INVALID")
    decisions = {int(record["decision_interaction"]): record for record in records}
    if set(decisions) != set(range(4)):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_DECISIONS_INVALID")
    return decisions


def routing_report(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    route_scope: str,
    suffix_weight: float,
    terminal_weight: float,
    slope_weight: float,
    minimum_benefit: float,
    prefix_tolerance: float = 1e-6,
) -> dict[str, object]:
    per_seed: list[dict[str, object]] = []
    flat: list[dict[str, object]] = []
    quality_rows: list[dict[str, float]] = []
    for seed in sorted(rows):
        frame = rows[seed]
        zero = frame["zero"]
        learned = frame["learned"]
        endpoints = frame["endpoints"]
        decisions = _decisions_by_interaction(learned, route_scope)
        interaction_rows: list[dict[str, object]] = []
        zero_interactions = _interactions(zero)
        for interaction in range(4):
            arms = [
                endpoints[(interaction, ARM_DOSES[0])],
                zero,
                endpoints[(interaction, ARM_DOSES[2])],
            ]
            components = [
                local_error(
                    row,
                    interaction,
                    suffix_weight=suffix_weight,
                    terminal_weight=terminal_weight,
                    slope_weight=slope_weight,
                )
                for row in arms
            ]
            for endpoint in (arms[0], arms[2]):
                endpoint_interactions = _interactions(endpoint)
                for prefix in range(interaction):
                    for metric in ("mean_l1", "final_l1"):
                        if abs(
                            _finite(endpoint_interactions[prefix][metric])
                            - _finite(zero_interactions[prefix][metric])
                        ) > prefix_tolerance:
                            raise CCLVRHeldoutAggregationError(
                                "CCLVR_HELDOUT_LOCAL_ENDPOINT_CHANGED_PREFIX"
                            )
            losses = [component["loss"] for component in components]
            values = [losses[1] - loss for loss in losses]
            best_nonzero = 0 if values[0] >= values[2] else 2
            target_arm = best_nonzero if values[best_nonzero] >= minimum_benefit else 1
            decision = decisions[interaction]
            selected_arm = int(decision.get("selected_arm", -1))
            probabilities = [_finite(value) for value in decision.get("soft_probabilities", ())]
            if selected_arm not in (0, 1, 2) or len(probabilities) != 3 or not math.isclose(sum(probabilities), 1.0, abs_tol=2e-3):
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_POLICY_INVALID")
            selected_value = values[selected_arm]
            one_hot = [1.0 if index == target_arm else 0.0 for index in range(3)]
            brier = sum((probability - target) ** 2 for probability, target in zip(probabilities, one_hot, strict=True))
            interaction_row = {
                "interaction": interaction,
                "arm_losses": losses,
                "arm_values": values,
                "target_arm": target_arm,
                "selected_arm": selected_arm,
                "selected_dose": ARM_DOSES[selected_arm],
                "soft_probabilities": probabilities,
                "brier": brier,
                "active": selected_arm != 1,
                "oracle_active": target_arm != 1,
                "selected_counterfactual_value": selected_value,
                "harmful": bool(selected_arm != 1 and selected_value < -minimum_benefit),
                "direction_correct": bool(target_arm != 1 and selected_arm == target_arm),
                "arm_components": components,
            }
            interaction_rows.append(interaction_row)
            flat.append(interaction_row)
        beneficial_exists = any(bool(row["oracle_active"]) for row in interaction_rows)
        nonbeneficial_exists = any(not bool(row["oracle_active"]) for row in interaction_rows)
        executed_beneficial = any(
            bool(row["active"]) and _finite(row["selected_counterfactual_value"]) >= minimum_benefit
            for row in interaction_rows
        )
        abstained_nonbeneficial = any(
            not bool(row["oracle_active"]) and not bool(row["active"]) for row in interaction_rows
        )
        mixed_required = beneficial_exists and nonbeneficial_exists
        per_seed.append(
            {
                "seed": seed,
                "interactions": interaction_rows,
                "beneficial_and_nonbeneficial_oracle_present": mixed_required,
                "executed_beneficial_interaction": executed_beneficial,
                "abstained_nonbeneficial_interaction": abstained_nonbeneficial,
                "mixed_execution_rule_passed": (
                    not mixed_required or (executed_beneficial and abstained_nonbeneficial)
                ),
            }
        )
        quality = _quality(learned)
        quality["seed"] = float(seed)
        quality_rows.append(quality)
    active_rate = float(np.mean([bool(row["active"]) for row in flat]))
    oracle_coverage = float(np.mean([bool(row["oracle_active"]) for row in flat]))
    quality_per_seed = [
        {"seed": int(row["seed"]), **{name: row[name] for name in QUALITY_METRICS}}
        for row in quality_rows
    ]
    return {
        "quality": {"mean": _mean_rows(quality_per_seed), "per_seed": quality_per_seed},
        "routing": {
            "decision_count": len(flat),
            "active_coverage": active_rate,
            "oracle_coverage": oracle_coverage,
            "coverage_calibration_error": abs(active_rate - oracle_coverage),
            "policy_brier": float(np.mean([_finite(row["brier"]) for row in flat])),
            "harmful_routing_rate": float(np.mean([bool(row["harmful"]) for row in flat])),
            "counterfactual_policy_value": float(
                np.mean([_finite(row["selected_counterfactual_value"]) for row in flat])
            ),
            "direction_accuracy_when_needed": (
                float(np.mean([bool(row["direction_correct"]) for row in flat if bool(row["oracle_active"])]))
                if any(bool(row["oracle_active"]) for row in flat)
                else None
            ),
            "mixed_execution_rule_passed": all(
                bool(row["mixed_execution_rule_passed"]) for row in per_seed
            ),
            "per_seed": per_seed,
        },
    }


def _historical_action_rows(result: Mapping[str, Any]) -> dict[tuple[int, float], Mapping[str, Any]]:
    raw = result.get("measurements")
    if result.get("state") != "ready" or not isinstance(raw, list):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_HISTORICAL_ACTION_INVALID")
    rows: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_HISTORICAL_ACTION_INVALID")
        key = (_identity(row)[3], _finite(row.get("dose")))
        if key in rows:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_HISTORICAL_ACTION_INVALID")
        rows[key] = row
    return rows


def _action_report(rows: Mapping[tuple[int, float], Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, object]:
    per_seed = []
    for seed in seeds:
        negative = rows[(seed, -0.1)]
        zero = rows[(seed, 0.0)]
        positive = rows[(seed, 0.1)]
        negative_response = _response(negative)
        zero_response = _response(zero)
        positive_response = _response(positive)
        if negative_response.shape != zero_response.shape or positive_response.shape != zero_response.shape:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_RESPONSE_FRAME_INVALID")
        per_seed.append(
            {
                "seed": seed,
                "zero_centered_l1": float(
                    (
                        np.mean(np.abs(positive_response - zero_response))
                        + np.mean(np.abs(negative_response - zero_response))
                    )
                    / 2.0
                ),
                "endpoint_l1": float(np.mean(np.abs(positive_response - negative_response))),
                "quality_central_slope": float(
                    (_quality(positive)["mean_l1"] - _quality(negative)["mean_l1"]) / 0.2
                ),
            }
        )
    return {
        "mean_zero_centered_prediction_l1": float(np.mean([row["zero_centered_l1"] for row in per_seed])),
        "mean_endpoint_prediction_l1": float(np.mean([row["endpoint_l1"] for row in per_seed])),
        "mean_abs_quality_central_slope": float(
            np.mean([abs(row["quality_central_slope"]) for row in per_seed])
        ),
        "per_seed": per_seed,
    }


def _load_action_rows(
    *,
    plan: Mapping[str, Any],
    root: Path,
) -> tuple[dict[tuple[int, float], Mapping[str, Any]], list[dict[str, str]]]:
    jobs = plan.get("action_jobs")
    if not isinstance(jobs, list):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_PLAN_INVALID")
    rows: dict[tuple[int, float], Mapping[str, Any]] = {}
    receipts: list[dict[str, str]] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_PLAN_INVALID")
        seed = int(job.get("seed", -1))
        path = root / str(job.get("output_rel")) / "result.json"
        result = _load(path)
        raw = result.get("measurements")
        if (
            result.get("artifact_type") != "verdiwm-ctrl-world-cclvr-heldout-shard-v1"
            or result.get("state") != "ready"
            or result.get("mode") != "action_sensitivity"
            or result.get("cell_id") != "d4"
            or not isinstance(raw, list)
            or len(raw) != 3
        ):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_SHARD_INVALID")
        route_signatures = []
        for row in raw:
            if not isinstance(row, Mapping) or _identity(row)[3] != seed:
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_ROW_INVALID")
            key = (seed, _finite(row.get("action_dose")))
            if key in rows:
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_ROW_DUPLICATE")
            rows[key] = row
            audit = row.get("route_audit")
            if not isinstance(audit, Mapping) or not isinstance(audit.get("decision_records"), list):
                raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_ROUTE_INVALID")
            route_signatures.append(
                tuple(int(record["selected_arm"]) for record in audit["decision_records"])
            )
        if len(set(route_signatures)) != 1:
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_ROUTE_NOT_PAIRED")
        receipts.append({"path": str(path), "sha256": _sha256(path)})
    expected = {(int(seed), dose) for seed in plan["seeds"] for dose in (-0.1, 0.0, 0.1)}
    if set(rows) != expected:
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_ACTION_FRAME_INVALID")
    return rows, receipts


def aggregate(*, plan_path: Path, output_path: Path) -> dict[str, object]:
    plan_path = plan_path.resolve(strict=True)
    plan = _load(plan_path)
    if (
        plan.get("artifact_type") not in PLAN_TYPES
        or plan.get("state") != "frozen_before_execution"
        or str(plan.get("promotion_episode")) != "1799"
    ):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_PLAN_INVALID")
    root = Path(str(plan["output_root"])).resolve(strict=True)
    routing_rows, receipts = _load_routing_rows(plan=plan, root=root)
    local_return = plan["local_return"]
    suffix_weight = _finite(local_return["suffix_mean_l1_weight"])
    terminal_weight = _finite(local_return["terminal_interaction_l1_weight"])
    slope_weight = _finite(local_return["suffix_horizon_l1_slope_weight"])
    minimum_benefit = _finite(local_return["minimum_benefit"])
    prefix_tolerance = _finite(local_return["prefix_identity_tolerance"])
    cells = {str(row["id"]): row for row in plan["cells"]}
    models = {
        cell_id: routing_report(
            routing_rows[cell_id],
            route_scope=str(cells[cell_id]["route_scope"]),
            suffix_weight=suffix_weight,
            terminal_weight=terminal_weight,
            slope_weight=slope_weight,
            minimum_benefit=minimum_benefit,
            prefix_tolerance=prefix_tolerance,
        )
        for cell_id in sorted(cells)
    }

    action_rows, action_receipts = _load_action_rows(plan=plan, root=root)
    receipts.extend(action_receipts)
    d4_action = _action_report(action_rows, [int(value) for value in plan["seeds"]])
    models["d4"]["action_sensitivity"] = d4_action

    references = plan["references"]
    a0_action_path = Path(str(references["a0_action_result"])).resolve(strict=True)
    if _sha256(a0_action_path) != str(references["a0_action_result_sha256"]):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_A0_ACTION_HASH_MISMATCH")
    a0_rows = _historical_action_rows(_load(a0_action_path))
    seeds = [int(value) for value in plan["seeds"]]
    a0_action = _action_report(a0_rows, seeds)
    historical_report_path = Path(str(references["cbma_report"])).resolve(strict=True)
    if _sha256(historical_report_path) != str(references["cbma_report_sha256"]):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_CBMA_REPORT_HASH_MISMATCH")
    historical = _load(historical_report_path)
    historical_models = historical.get("models")
    if not isinstance(historical_models, Mapping):
        raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_CBMA_REPORT_INVALID")

    zero_tolerance = _finite(plan["promotion_gate"]["zero_identity_tolerance"])
    zero_identity: list[dict[str, object]] = []
    for cell_id in sorted(cells):
        for seed in seeds:
            candidate = routing_rows[cell_id][seed]["zero"]
            reference = a0_rows[(seed, 0.0)]
            candidate_quality = _quality(candidate)
            reference_quality = _quality(reference)
            difference = max(
                abs(candidate_quality[name] - reference_quality[name]) for name in QUALITY_METRICS
            )
            zero_identity.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "maximum_quality_abs_difference": difference,
                    "tolerance": zero_tolerance,
                    "state": "passed" if difference <= zero_tolerance else "failed",
                }
            )

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "state": "passed" if passed else "failed", "evidence": evidence})

    gate = plan["promotion_gate"]
    d4_quality = models["d4"]["quality"]
    d4_routing = models["d4"]["routing"]
    thresholds = gate["quality_thresholds"]
    for metric in ("mean_l1", "final_interaction_l1", "horizon_l1_slope"):
        observed = _finite(d4_quality["mean"][metric])
        threshold = _finite(thresholds[metric])
        check(
            f"d4_{metric}_beats_frozen_cbma_threshold",
            observed < threshold,
            {"observed": observed, "threshold": threshold, "lower_is_better": True},
        )
    d3_quality = models["d3"]["quality"]["mean"]
    for metric in ("final_interaction_l1", "horizon_l1_slope"):
        check(
            f"d4_beats_d3_{metric}",
            _finite(d4_quality["mean"][metric]) < _finite(d3_quality[metric]),
            {"d4": d4_quality["mean"][metric], "d3": d3_quality[metric]},
        )
    minimum_wins = int(gate["minimum_paired_seed_wins"])
    d4_seed = {int(row["seed"]): row for row in d4_quality["per_seed"]}
    for comparator in ("b3", "c1", "c3"):
        comparator_model = historical_models.get(comparator)
        if not isinstance(comparator_model, Mapping):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_COMPARATOR_INVALID")
        comparator_quality = comparator_model.get("quality")
        if not isinstance(comparator_quality, Mapping) or not isinstance(comparator_quality.get("per_seed"), list):
            raise CCLVRHeldoutAggregationError("CCLVR_HELDOUT_COMPARATOR_INVALID")
        comparator_seed = {int(row["seed"]): row for row in comparator_quality["per_seed"]}
        wins = sum(
            _finite(d4_seed[seed]["mean_l1"]) < _finite(comparator_seed[seed]["mean_l1"])
            for seed in seeds
        )
        check(
            f"d4_paired_mean_l1_wins_vs_{comparator}",
            wins >= minimum_wins,
            {"wins": wins, "required": minimum_wins, "seed_count": len(seeds)},
        )
    check(
        "d4_policy_brier_below_cbma_b3",
        _finite(d4_routing["policy_brier"]) < _finite(gate["maximum_policy_brier"]),
        {"observed": d4_routing["policy_brier"], "threshold": gate["maximum_policy_brier"]},
    )
    check(
        "d4_harmful_routing_bounded",
        _finite(d4_routing["harmful_routing_rate"]) <= _finite(gate["maximum_harmful_routing_rate"]),
        {
            "observed": d4_routing["harmful_routing_rate"],
            "maximum": gate["maximum_harmful_routing_rate"],
        },
    )
    check(
        "d4_counterfactual_policy_value_positive",
        _finite(d4_routing["counterfactual_policy_value"]) > 0.0,
        d4_routing["counterfactual_policy_value"],
    )
    check(
        "d4_mixed_benefit_execution_and_abstention",
        bool(d4_routing["mixed_execution_rule_passed"]),
        [
            {
                "seed": row["seed"],
                "mixed_required": row["beneficial_and_nonbeneficial_oracle_present"],
                "executed_beneficial": row["executed_beneficial_interaction"],
                "abstained_nonbeneficial": row["abstained_nonbeneficial_interaction"],
            }
            for row in d4_routing["per_seed"]
        ],
    )
    action_ratio = _finite(d4_action["mean_zero_centered_prediction_l1"]) / max(
        _finite(a0_action["mean_zero_centered_prediction_l1"]), 1e-12
    )
    check(
        "d4_action_sensitivity_retained_vs_a0",
        action_ratio >= _finite(gate["minimum_action_sensitivity_ratio"]),
        {
            "observed_ratio": action_ratio,
            "minimum": gate["minimum_action_sensitivity_ratio"],
            "d4": d4_action["mean_zero_centered_prediction_l1"],
            "a0": a0_action["mean_zero_centered_prediction_l1"],
        },
    )
    d1_routing = models["d1"]["routing"]
    d2_routing = models["d2"]["routing"]
    check(
        "d2_improves_d1_coverage_calibration",
        _finite(d2_routing["coverage_calibration_error"])
        < _finite(d1_routing["coverage_calibration_error"]),
        {
            "d1": d1_routing["coverage_calibration_error"],
            "d2": d2_routing["coverage_calibration_error"],
            "lower_is_better": True,
        },
    )
    check(
        "d2_does_not_increase_d1_harmful_routing",
        _finite(d2_routing["harmful_routing_rate"]) <= _finite(d1_routing["harmful_routing_rate"]),
        {"d1": d1_routing["harmful_routing_rate"], "d2": d2_routing["harmful_routing_rate"]},
    )
    check(
        "all_cclvr_zero_routes_reproduce_a0_identity",
        all(row["state"] == "passed" for row in zero_identity),
        zero_identity,
    )
    passed = all(row["state"] == "passed" for row in checks)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-report-v1",
        "state": "ready",
        "experiment_id": plan["experiment_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "input_receipts": receipts,
        "references": {
            "a0_action_sensitivity": a0_action,
            "cbma_report": str(historical_report_path),
        },
        "models": models,
        "zero_identity_checks": zero_identity,
        "promotion": {
            "candidate": "d4",
            "state": "locally_promoted_for_mechanism_retention" if passed else "not_promoted",
            "confirmation_authorized": False,
            "checks": checks,
            "failed_checks": [row["name"] for row in checks if row["state"] == "failed"],
            "claim_boundary": (
                "This is a one-episode, three-seed held-out mechanism screen. Passing retains D4 as the "
                "current local mechanism primitive; it does not authorize a 512-step run or establish a "
                "dataset-wide or publication-level improvement claim."
            ),
        },
        "promotion_episode_reuse_policy": (
            "Episode 1799 remains forbidden from training, supervision labels, coverage control, "
            "calibration fitting, and method selection after this report."
        ),
    }
    _atomic_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = aggregate(plan_path=args.plan, output_path=args.output.resolve())
    print(json.dumps(result["promotion"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
