#!/usr/bin/env python3
"""Aggregate held-out FSHC ablations and emit a fail-closed promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OUTCOMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class FSHCHeldoutAggregationError(RuntimeError):
    """Held-out artifacts do not match the frozen evaluation frame."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FSHCHeldoutAggregationError(f"FSHC_HELDOUT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise FSHCHeldoutAggregationError(f"FSHC_HELDOUT_JSON_INVALID:{path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, error: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FSHCHeldoutAggregationError(error)
    return number


def _rows(result: Mapping[str, Any]) -> dict[tuple[int, float], Mapping[str, Any]]:
    raw = result.get("measurements")
    if result.get("state") != "ready" or not isinstance(raw, list):
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_RESULT_NOT_READY")
    rows: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("identity"), Mapping):
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_MEASUREMENT_INVALID")
        key = (int(row["identity"]["seed"]), float(row["dose"]))
        if key in rows:
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_MEASUREMENT_DUPLICATE")
        rows[key] = row
    return rows


def _outcome(row: Mapping[str, Any], name: str) -> float:
    outcomes = row.get("outcomes")
    if not isinstance(outcomes, Mapping) or name not in outcomes:
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_OUTCOME_INVALID")
    return _finite(outcomes[name], "FSHC_HELDOUT_OUTCOME_NONFINITE")


def _response(row: Mapping[str, Any]) -> np.ndarray:
    response = row.get("prediction_response")
    if not isinstance(response, Mapping) or not isinstance(response.get("values"), list):
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_PREDICTION_RESPONSE_MISSING")
    values = np.asarray(response["values"], dtype=np.float64)
    shape = response.get("shape")
    if not isinstance(shape, list) or int(np.prod(shape)) != values.size or not np.isfinite(values).all():
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_PREDICTION_RESPONSE_INVALID")
    return values


def _quality(rows: Mapping[tuple[int, float], Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, object]:
    per_seed = []
    for seed in seeds:
        row = rows[(seed, 0.0)]
        per_seed.append(
            {
                "seed": seed,
                "mean_l1": -_outcome(row, "negative_mean_l1"),
                "final_interaction_l1": -_outcome(row, "negative_final_interaction_l1"),
                "horizon_l1_slope": -_outcome(row, "negative_horizon_l1_slope"),
                "mean_psnr": _outcome(row, "mean_psnr"),
                "final_psnr": _outcome(row, "final_psnr"),
            }
        )
    names = tuple(name for name in per_seed[0] if name != "seed")
    return {
        "mean": {name: float(np.mean([row[name] for row in per_seed])) for name in names},
        "per_seed": per_seed,
    }


def _action_report(
    rows: Mapping[tuple[int, float], Mapping[str, Any]], seeds: Sequence[int], radius: float
) -> dict[str, object]:
    response_rows = []
    for seed in seeds:
        negative = _response(rows[(seed, -radius)])
        zero = _response(rows[(seed, 0.0)])
        positive = _response(rows[(seed, radius)])
        if negative.shape != zero.shape or positive.shape != zero.shape:
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_PREDICTION_RESPONSE_FRAME_MISMATCH")
        response_rows.append(
            {
                "seed": seed,
                "zero_centered_l1": float(
                    (np.mean(np.abs(positive - zero)) + np.mean(np.abs(negative - zero))) / 2.0
                ),
                "endpoint_l1": float(np.mean(np.abs(positive - negative))),
                "quality_central_slope": float(
                    (_outcome(rows[(seed, radius)], "negative_mean_l1")
                    - _outcome(rows[(seed, -radius)], "negative_mean_l1"))
                    / (2.0 * radius)
                ),
            }
        )
    return {
        "mean_zero_centered_prediction_l1": float(
            np.mean([row["zero_centered_l1"] for row in response_rows])
        ),
        "mean_endpoint_prediction_l1": float(np.mean([row["endpoint_l1"] for row in response_rows])),
        "mean_abs_quality_central_slope": float(
            np.mean([abs(row["quality_central_slope"]) for row in response_rows])
        ),
        "per_seed": response_rows,
    }


def _flatten_gate(row: Mapping[str, Any], key: str) -> np.ndarray:
    records = row.get("learned_gate_records")
    if not isinstance(records, list) or not records:
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_GATE_RECORDS_MISSING")
    values = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get(key), list):
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_GATE_RECORD_INVALID")
        values.extend(float(value) for value in record[key])
    output = np.asarray(values, dtype=np.float64)
    if not output.size or not np.isfinite(output).all():
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_GATE_RECORD_INVALID")
    return output


def _routing_report(
    rows: Mapping[tuple[int, float], Mapping[str, Any]],
    seeds: Sequence[int],
    radius: float,
    weights: Sequence[float],
    minimum_benefit: float,
    active_threshold: float,
) -> dict[str, object]:
    per_seed = []
    for seed in seeds:
        zero = rows[(seed, 0.0)]
        scores = {
            dose: float(sum(weight * _outcome(rows[(seed, dose)], name) for weight, name in zip(weights, OUTCOMES, strict=True)))
            for dose in (-radius, 0.0, radius)
        }
        gains = {-1: scores[-radius] - scores[0.0], 1: scores[radius] - scores[0.0]}
        oracle_sign = max(gains, key=gains.get)
        oracle_gain = max(gains.values())
        signed_gain = float(np.mean(_flatten_gate(zero, "signed_gain")))
        confidence = float(np.mean(_flatten_gate(zero, "confidence")))
        use_probability = float(np.mean(_flatten_gate(zero, "use_probability")))
        active = use_probability >= active_threshold and abs(signed_gain) > 1e-8
        predicted_sign = 1 if signed_gain > 0.0 else -1 if signed_gain < 0.0 else 0
        routed_gain = gains[predicted_sign] if active and predicted_sign else 0.0
        target_active = oracle_gain > minimum_benefit
        per_seed.append(
            {
                "seed": seed,
                "negative_fixed_gain": gains[-1],
                "positive_fixed_gain": gains[1],
                "oracle_sign": oracle_sign if target_active else 0,
                "oracle_gain": oracle_gain,
                "learned_signed_gain": signed_gain,
                "mean_confidence": confidence,
                "mean_use_probability": use_probability,
                "active": active,
                "predicted_sign": predicted_sign if active else 0,
                "routed_fixed_dose_gain": routed_gain,
                "harmful": bool(active and routed_gain < -minimum_benefit),
                "direction_correct": bool(active and target_active and predicted_sign == oracle_sign),
                "activation_target": target_active,
                "activation_brier": (use_probability - float(target_active)) ** 2,
            }
        )
    active_rows = [row for row in per_seed if row["active"]]
    target_rows = [row for row in per_seed if row["activation_target"]]
    return {
        "harmful_routing_rate": float(np.mean([row["harmful"] for row in per_seed])),
        "active_rate": float(np.mean([row["active"] for row in per_seed])),
        "abstention_rate": float(np.mean([not row["active"] for row in per_seed])),
        "abstention_calibration_brier": float(np.mean([row["activation_brier"] for row in per_seed])),
        "direction_accuracy_when_needed": (
            float(np.mean([row["direction_correct"] for row in target_rows])) if target_rows else None
        ),
        "mean_routed_fixed_dose_gain": float(np.mean([row["routed_fixed_dose_gain"] for row in per_seed])),
        "active_count": len(active_rows),
        "per_seed": per_seed,
    }


def aggregate(*, plan_path: Path, output_path: Path) -> dict[str, object]:
    plan_path = plan_path.resolve(strict=True)
    plan = _load_json(plan_path)
    if plan.get("artifact_type") != "ctrl-world-fshc-heldout-evaluation-plan":
        raise FSHCHeldoutAggregationError("FSHC_HELDOUT_PLAN_INVALID")
    root = Path(str(plan["output_root"])).resolve(strict=True)
    phases = plan["phases"]
    seeds = [int(seed) for seed in plan["seeds"]]
    model_ids = [str(row["id"]) for row in plan["models"]]
    action_phase = phases["action_sensitivity"]
    action_radius = max(float(value) for value in action_phase["doses"])
    models: dict[str, object] = {}
    input_receipts = []
    for model_id in model_ids:
        path = root / "action_sensitivity" / model_id / "result.json"
        result = _load_json(path)
        rows = _rows(result)
        expected = {(seed, dose) for seed in seeds for dose in (-action_radius, 0.0, action_radius)}
        if set(rows) != expected:
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_ACTION_FRAME_MISMATCH")
        models[model_id] = {
            "quality": _quality(rows, seeds),
            "action_sensitivity": _action_report(rows, seeds, action_radius),
        }
        input_receipts.append({"path": str(path), "sha256": _sha256(path)})

    routing_phase = phases["routing"]
    routing_radius = max(float(value) for value in routing_phase["doses"])
    weights = [float(value) for value in routing_phase["outcome_weights"]]
    for model_id in routing_phase["models"]:
        path = root / "routing" / str(model_id) / "result.json"
        result = _load_json(path)
        rows = _rows(result)
        expected = {(seed, dose) for seed in seeds for dose in (-routing_radius, 0.0, routing_radius)}
        if set(rows) != expected:
            raise FSHCHeldoutAggregationError("FSHC_HELDOUT_ROUTING_FRAME_MISMATCH")
        models[str(model_id)]["routing"] = _routing_report(
            rows,
            seeds,
            routing_radius,
            weights,
            float(routing_phase["minimum_composite_benefit"]),
            float(routing_phase["active_probability_threshold"]),
        )
        input_receipts.append({"path": str(path), "sha256": _sha256(path)})

    gate = plan["promotion_gate"]
    candidate = str(gate["candidate"])
    comparators = [str(value) for value in gate["quality_comparators"]]
    candidate_model = models[candidate]
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "state": "passed" if passed else "failed", "evidence": evidence})

    candidate_quality = candidate_model["quality"]["mean"]
    for metric in ("mean_l1", "final_interaction_l1", "horizon_l1_slope"):
        values = {model_id: models[model_id]["quality"]["mean"][metric] for model_id in comparators}
        check(
            f"candidate_best_{metric}",
            candidate_quality[metric] < min(values.values()),
            {"candidate": candidate_quality[metric], "comparators": values, "lower_is_better": True},
        )
    minimum_wins = int(gate["minimum_paired_seed_wins"])
    candidate_seed = {row["seed"]: row for row in candidate_model["quality"]["per_seed"]}
    for comparator in comparators:
        comparator_seed = {row["seed"]: row for row in models[comparator]["quality"]["per_seed"]}
        wins = sum(candidate_seed[seed]["mean_l1"] < comparator_seed[seed]["mean_l1"] for seed in seeds)
        check(
            f"candidate_paired_wins_vs_{comparator}",
            wins >= minimum_wins,
            {"wins": wins, "required": minimum_wins, "seed_count": len(seeds)},
        )
    retention = float(gate["minimum_action_sensitivity_retention"])
    candidate_sensitivity = candidate_model["action_sensitivity"]["mean_zero_centered_prediction_l1"]
    for reference in gate["action_sensitivity_references"]:
        reference = str(reference)
        reference_sensitivity = models[reference]["action_sensitivity"]["mean_zero_centered_prediction_l1"]
        check(
            f"action_sensitivity_retained_vs_{reference}",
            candidate_sensitivity >= retention * reference_sensitivity,
            {
                "candidate": candidate_sensitivity,
                "reference": reference_sensitivity,
                "required_ratio": retention,
                "observed_ratio": candidate_sensitivity / max(reference_sensitivity, 1e-12),
            },
        )
    candidate_routing = candidate_model["routing"]
    check(
        "harmful_routing_bounded",
        candidate_routing["harmful_routing_rate"] <= float(gate["maximum_harmful_routing_rate"]),
        candidate_routing["harmful_routing_rate"],
    )
    for comparator in gate["routing_comparators"]:
        comparator = str(comparator)
        check(
            f"routing_calibration_vs_{comparator}",
            candidate_routing["abstention_calibration_brier"]
            <= models[comparator]["routing"]["abstention_calibration_brier"],
            {
                "candidate": candidate_routing["abstention_calibration_brier"],
                "comparator": models[comparator]["routing"]["abstention_calibration_brier"],
            },
        )
    promoted = all(row["state"] == "passed" for row in checks)
    report = {
        "schema_version": 1,
        "artifact_type": "ctrl-world-fshc-heldout-evaluation-report",
        "state": "ready",
        "experiment_id": plan["experiment_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "input_receipts": input_receipts,
        "models": models,
        "promotion": {
            "candidate": candidate,
            "state": "promoted_to_512_step_confirmation" if promoted else "not_promoted",
            "checks": checks,
            "failed_checks": [row["name"] for row in checks if row["state"] == "failed"],
            "claim_boundary": (
                "This is a one-context, three-seed screen. Passing licenses a 512-step confirmation run; "
                "it does not establish a general model improvement claim."
            ),
        },
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
