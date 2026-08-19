"""Fail-closed Pareto assessment for the two-surface ACWM experiment loop.

The module deliberately evaluates scalar outputs only.  GPU scheduling remains
with the existing campaign daemon, and no downstream policy/task-success
signal can enter this ACWM contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class ACWMDualEvaluationError(ValueError):
    """An ACWM dual-evaluation contract or result failed closed."""


_SURFACE_ROLES = {
    "paired_prediction": "primary",
    "giga_style_rollout": "protected",
}
_REQUIRED_PROHIBITED_SIGNALS = {"task_success", "task_progress", "safety_events"}
_STAGE_SPLITS = {"screen": "dev", "confirm": "accept"}


def validate_acwm_dual_evaluation_contract(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate the fixed two-surface ACWM boundary and content digest."""

    try:
        validate_document("acwm_dual_evaluation", document, root=root)
    except ContractValidationError as exc:
        raise ACWMDualEvaluationError(f"ACWM_DUAL_EVALUATION_SCHEMA_INVALID:{exc}") from exc

    if document["contract_digest"] != _digest_without(document, "contract_digest"):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_DIGEST_MISMATCH")
    prohibited = {str(value) for value in document["prohibited_signal_ids"]}
    if not _REQUIRED_PROHIBITED_SIGNALS.issubset(prohibited):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_WAM_EXCLUSION_MISSING")
    if len(prohibited) != len(document["prohibited_signal_ids"]):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROHIBITED_SIGNAL_DUPLICATE")

    surfaces = _rows(document, "surfaces")
    surface_ids = [str(row["surface_id"]) for row in surfaces]
    if set(surface_ids) != set(_SURFACE_ROLES) or len(surface_ids) != len(_SURFACE_ROLES):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_SURFACE_SET_INVALID")
    for row in surfaces:
        surface_id = str(row["surface_id"])
        if row["role"] != _SURFACE_ROLES[surface_id]:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_SURFACE_ROLE_INVALID")

    metrics = _rows(document, "metrics")
    metric_ids = [str(row["metric_id"]) for row in metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_METRIC_DUPLICATE")
    primary_count = 0
    for metric in metrics:
        metric_id = str(metric["metric_id"])
        if metric_id in prohibited:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_WAM_METRIC_FORBIDDEN")
        surface_id = str(metric["surface_id"])
        if surface_id == "giga_style_rollout" and metric["role"] != "protected":
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_METRIC_ROLE_INVALID")
        if metric["role"] == "primary":
            primary_count += 1
            if float(metric["maximum_regression"]) != 0.0:
                raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PRIMARY_REGRESSION_INVALID")
        elif float(metric["minimum_improvement"]) != 0.0:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTECTED_IMPROVEMENT_INVALID")
    if primary_count != 1:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PRIMARY_COUNT_INVALID")

    stages = _rows(document, "stages")
    stage_names = [str(row["stage"]) for row in stages]
    if set(stage_names) != set(_STAGE_SPLITS) or len(stage_names) != len(_STAGE_SPLITS):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_STAGE_SET_INVALID")
    for stage in stages:
        if stage["split"] != _STAGE_SPLITS[str(stage["stage"])]:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_STAGE_SPLIT_INVALID")

    protocol = document["protocol"]
    if not isinstance(protocol, Mapping) or protocol.get("metric_version") != 1:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_INVALID")
    protocol_stages = _rows(protocol, "stages")
    protocol_names = [str(row["stage"]) for row in protocol_stages]
    if set(protocol_names) != set(_STAGE_SPLITS) or len(protocol_names) != len(_STAGE_SPLITS):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_STAGE_SET_INVALID")
    for protocol_stage in protocol_stages:
        if int(protocol_stage["paired_prediction_interactions"]) < 1:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_PAIRED_INVALID")
        if int(protocol_stage["rollout_interactions"]) < 2:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_ROLLOUT_INVALID")
        if int(protocol_stage["num_inference_steps"]) < 1:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_STEPS_INVALID")
        _validate_contexts(protocol_stage["paired_prediction_contexts"])
        _validate_contexts(protocol_stage["rollout_contexts"])

    resource_policy = document["resource_policy"]
    if not isinstance(resource_policy, Mapping):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_RESOURCE_POLICY_INVALID")
    total_gpus = int(resource_policy["total_gpus"])
    per_candidate_gpus = int(resource_policy["per_candidate_gpus"])
    parallel_candidates = int(resource_policy["max_parallel_candidates"])
    if per_candidate_gpus * parallel_candidates > total_gpus:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_RESOURCE_POLICY_OVERSUBSCRIBED")


def assess_acwm_dual_evaluation(
    contract: Mapping[str, object],
    *,
    stage: str,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    root: Path | None = None,
) -> dict[str, object]:
    """Assess one candidate without collapsing incompatible metrics into a score."""

    validate_acwm_dual_evaluation_contract(contract, root=root)
    if stage not in _STAGE_SPLITS:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_STAGE_INVALID")
    rules = _rows(contract, "metrics")
    expected = {str(rule["metric_id"]) for rule in rules}
    _validate_measurement_binding(contract, stage, baseline, "BASELINE")
    _validate_measurement_binding(contract, stage, candidate, "CANDIDATE")
    baseline_values = _metric_values(baseline, expected, "BASELINE")
    candidate_values = _metric_values(candidate, expected, "CANDIDATE")

    deltas: dict[str, float] = {}
    blockers: list[str] = []
    for rule in rules:
        metric_id = str(rule["metric_id"])
        if rule["direction"] == "minimize":
            delta = baseline_values[metric_id] - candidate_values[metric_id]
        else:
            delta = candidate_values[metric_id] - baseline_values[metric_id]
        deltas[metric_id] = delta
        if rule["role"] == "primary":
            if delta <= float(rule["minimum_improvement"]):
                blockers.append(f"PRIMARY_NOT_IMPROVED:{metric_id}")
        elif delta < -float(rule["maximum_regression"]):
            blockers.append(f"PROTECTED_REGRESSION:{metric_id}")

    accepted = not blockers
    next_state = (
        "eligible_for_confirm" if stage == "screen" else "qualified_for_frozen_verifier"
    ) if accepted else "abstained"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-dual-evaluation-assessment",
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": stage,
        "split": _STAGE_SPLITS[stage],
        "state": next_state,
        "accepted": accepted,
        "metric_deltas": dict(sorted(deltas.items())),
        "blockers": blockers,
        "verdict_authority": False,
        "claim_boundary": "Only the separately frozen evaluator may promote a confirmed ACWM result.",
    }


def _rows(document: Mapping[str, object], field: str) -> list[Mapping[str, object]]:
    value = document[field]
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ACWMDualEvaluationError(f"ACWM_DUAL_EVALUATION_{field.upper()}_INVALID")
    return list(value)


def _metric_values(
    values: Mapping[str, object], expected: set[str], label: str
) -> dict[str, float]:
    metric_values: Mapping[str, object] = values
    nested = values.get("metrics")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ACWMDualEvaluationError(
                f"ACWM_DUAL_EVALUATION_{label}_METRICS_INVALID"
            )
        metric_values = nested
    actual = {str(key) for key in metric_values}
    if actual != expected:
        raise ACWMDualEvaluationError(f"ACWM_DUAL_EVALUATION_{label}_METRIC_SET_INVALID")
    normalized: dict[str, float] = {}
    for metric_id in expected:
        raw = metric_values[metric_id]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ACWMDualEvaluationError(
                f"ACWM_DUAL_EVALUATION_{label}_METRIC_NONFINITE:{metric_id}"
            )
        normalized[metric_id] = float(raw)
    return normalized


def _validate_measurement_binding(
    contract: Mapping[str, object], stage: str, values: Mapping[str, object], label: str
) -> None:
    if "metrics" not in values:
        return
    if values.get("contract_id") != contract["contract_id"]:
        raise ACWMDualEvaluationError(
            f"ACWM_DUAL_EVALUATION_{label}_CONTRACT_ID_MISMATCH"
        )
    if values.get("contract_digest") != contract["contract_digest"]:
        raise ACWMDualEvaluationError(
            f"ACWM_DUAL_EVALUATION_{label}_CONTRACT_DIGEST_MISMATCH"
        )
    if values.get("stage") != stage:
        raise ACWMDualEvaluationError(
            f"ACWM_DUAL_EVALUATION_{label}_STAGE_MISMATCH"
        )


def _validate_contexts(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_CONTEXTS_INVALID")
    seen: set[tuple[str, int]] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_CONTEXTS_INVALID")
        key = (str(row["episode_id"]), int(row["start_idx"]))
        if key in seen:
            raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PROTOCOL_CONTEXT_DUPLICATE")
        seen.add(key)


def _digest_without(document: Mapping[str, object], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_PAYLOAD_INVALID") from exc
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise ACWMDualEvaluationError("ACWM_DUAL_EVALUATION_FILE_INVALID")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "assess"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(_STAGE_SPLITS))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    contract = _load_mapping(args.contract)
    if args.command == "validate":
        validate_acwm_dual_evaluation_contract(contract)
        result: Mapping[str, Any] = {"valid": True, "contract_id": contract["contract_id"]}
    else:
        if args.stage is None or args.baseline is None or args.candidate is None:
            parser.error("assess requires --stage, --baseline, and --candidate")
        result = assess_acwm_dual_evaluation(
            contract,
            stage=args.stage,
            baseline=_load_mapping(args.baseline),
            candidate=_load_mapping(args.candidate),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
