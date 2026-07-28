"""Settled stage-receipt ingestion for cross-backbone experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments.lobo import build_lobo_plan


class ExperimentLedgerError(ValueError):
    """A receipt cannot enter the experiment ledger."""


def load_settled_receipts(
    *,
    spec: Mapping[str, Any],
    receipt_paths: Sequence[Path],
) -> tuple[dict[str, Any], ...]:
    """Load receipts and reject every non-settled or semantically invalid row."""

    plan = build_lobo_plan(spec)
    trials = {str(item["trial_id"]): item for item in plan["trials"]}
    stages = {str(item["stage"]): item for item in spec["stages"]}
    loaded: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted((Path(value).resolve() for value in receipt_paths), key=str):
        receipt = _read_receipt(path)
        try:
            validate_document("experiment_stage_receipt", receipt)
        except ContractValidationError as exc:
            raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_CONTRACT_INVALID:{path}:{exc}") from exc
        if receipt["settlement_state"] != "settled":
            raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_NOT_SETTLED:{path}")
        trial_id = str(receipt["trial_id"])
        stage = str(receipt["stage"])
        key = (trial_id, stage)
        if key in seen:
            raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_DUPLICATE_STAGE:{trial_id}:{stage}")
        seen.add(key)
        if trial_id not in trials:
            raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_UNKNOWN_TRIAL:{trial_id}")
        _validate_against_trial(spec=spec, trial=trials[trial_id], receipt=receipt, stage_config=stages[stage])
        receipt["_path"] = str(path)
        loaded.append(receipt)
    _validate_stage_chain(loaded)
    return tuple(sorted(loaded, key=lambda item: (int(item["sequence_index"]), str(item["stage"]))))


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_READ_FAILED:{path}") from exc
    if not isinstance(payload, dict):
        raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_ROOT_OBJECT_REQUIRED:{path}")
    return payload


def _validate_against_trial(
    *,
    spec: Mapping[str, Any],
    trial: Mapping[str, Any],
    receipt: Mapping[str, Any],
    stage_config: Mapping[str, Any],
) -> None:
    identity_fields = ("fold_id", "target_backbone", "scenario", "arm", "selector", "seed", "sequence_index")
    for field in identity_fields:
        if receipt[field] != trial[field]:
            raise ExperimentLedgerError(f"EXPERIMENT_RECEIPT_TRIAL_MISMATCH:{receipt['trial_id']}:{field}")
    if receipt["experiment_id"] != spec["experiment_id"]:
        raise ExperimentLedgerError("EXPERIMENT_RECEIPT_EXPERIMENT_MISMATCH")
    metric = spec["metric_contract"]
    if receipt["metric_name"] != metric["metric_name"]:
        raise ExperimentLedgerError("EXPERIMENT_RECEIPT_METRIC_MISMATCH")
    numeric = {
        name: float(receipt[name])
        for name in ("baseline_value", "candidate_value", "delta", "threshold", "gpu_hours")
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ExperimentLedgerError("EXPERIMENT_RECEIPT_NONFINITE_VALUE")
    if not math.isclose(numeric["threshold"], float(metric["positive_delta_threshold"]), abs_tol=1e-12):
        raise ExperimentLedgerError("EXPERIMENT_RECEIPT_THRESHOLD_MISMATCH")
    expected_delta = (
        numeric["candidate_value"] - numeric["baseline_value"]
        if metric["direction"] == "higher_is_better"
        else numeric["baseline_value"] - numeric["candidate_value"]
    )
    if not math.isclose(numeric["delta"], expected_delta, rel_tol=1e-9, abs_tol=1e-12):
        raise ExperimentLedgerError("EXPERIMENT_RECEIPT_DELTA_INCONSISTENT")
    if numeric["gpu_hours"] > float(stage_config["max_gpu_hours"]) + 1e-12:
        raise ExperimentLedgerError(f"EXPERIMENT_STAGE_BUDGET_EXCEEDED:{receipt['trial_id']}:{receipt['stage']}")
    threshold = numeric["threshold"]
    delta = numeric["delta"]
    outcome = str(receipt["outcome"])
    if outcome == "positive" and delta <= threshold:
        raise ExperimentLedgerError("EXPERIMENT_POSITIVE_OUTCOME_WITHOUT_THRESHOLD_GAIN")
    if outcome == "negative" and delta > threshold:
        raise ExperimentLedgerError("EXPERIMENT_NEGATIVE_OUTCOME_WITH_POSITIVE_GAIN")
    certificate = str(receipt["certificate_status"])
    if receipt["arm"] == "warm_start":
        expected_certificate = "abstain" if outcome == "abstain" else "licensed"
        if outcome in {"positive", "negative", "abstain"} and certificate != expected_certificate:
            raise ExperimentLedgerError("EXPERIMENT_WARM_START_CERTIFICATE_MISMATCH")
    elif certificate != "not_applicable":
        raise ExperimentLedgerError("EXPERIMENT_NON_TRANSFER_CERTIFICATE_FORBIDDEN")


def _validate_stage_chain(receipts: Sequence[Mapping[str, Any]]) -> None:
    by_trial: dict[str, dict[str, Mapping[str, Any]]] = {}
    for receipt in receipts:
        by_trial.setdefault(str(receipt["trial_id"]), {})[str(receipt["stage"])] = receipt
    for trial_id, stages in by_trial.items():
        if "gate" in stages and ("screen" not in stages or stages["screen"]["outcome"] != "positive"):
            raise ExperimentLedgerError(f"EXPERIMENT_GATE_WITHOUT_POSITIVE_SCREEN:{trial_id}")
        if "confirm" in stages and ("gate" not in stages or stages["gate"]["outcome"] != "positive"):
            raise ExperimentLedgerError(f"EXPERIMENT_CONFIRM_WITHOUT_POSITIVE_GATE:{trial_id}")
