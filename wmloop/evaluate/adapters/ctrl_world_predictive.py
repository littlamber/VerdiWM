"""Read-only Ctrl-World ACWM predictive-quality receipt adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, validate_document


class CtrlWorldPredictiveEvaluationError(ValueError):
    """Ctrl-World predictive evidence is malformed or outside the frozen split."""


def evaluate_ctrl_world_prediction_receipt(
    *,
    receipt_path: Path,
    heldout_split_path: Path,
    split_name: str,
) -> dict[str, object]:
    receipt = _load_json(receipt_path, "CTRL_WORLD_PREDICTION_RECEIPT_READ_FAILED")
    split = _load_json(heldout_split_path, "CTRL_WORLD_SPLIT_READ_FAILED")
    try:
        validate_document("ctrl_world_prediction_receipt", receipt)
        validate_document("ctrl_world_heldout_split", split)
    except ContractValidationError as exc:
        raise CtrlWorldPredictiveEvaluationError(f"CTRL_WORLD_PREDICTION_CONTRACT_INVALID:{exc}") from exc
    if split_name not in {"dev", "accept"}:
        raise CtrlWorldPredictiveEvaluationError("CTRL_WORLD_SPLIT_NAME_INVALID")
    identity = (str(receipt["task_id"]), str(receipt["episode_id"]), int(receipt["seed"]))
    allowed = {(str(item["task_id"]), str(item["episode_id"]), int(item["seed"])) for item in split[split_name]}
    if identity not in allowed:
        raise CtrlWorldPredictiveEvaluationError("CTRL_WORLD_PREDICTION_RECEIPT_OUTSIDE_FROZEN_SPLIT")
    if receipt["evidence_source"] != "paired_ground_truth_rollout":
        raise CtrlWorldPredictiveEvaluationError("CTRL_WORLD_DOWNSTREAM_SUCCESS_FORBIDDEN")
    if receipt["action_conditioned"] is not True:
        raise CtrlWorldPredictiveEvaluationError("CTRL_WORLD_ACTION_CONDITIONING_MISSING")
    metrics = {str(key): float(value) for key, value in receipt["metrics"].items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise CtrlWorldPredictiveEvaluationError("CTRL_WORLD_PREDICTION_METRIC_NONFINITE")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-predictive-verdict-evidence",
        "task_id": receipt["task_id"],
        "episode_id": receipt["episode_id"],
        "seed": receipt["seed"],
        "split": split_name,
        "horizon_frames": receipt["horizon_frames"],
        "metrics": metrics,
        "action_conditioned": True,
        "rollout_ref": receipt["rollout_ref"],
        "evidence_source": receipt["evidence_source"],
        "downstream_task_success_used_for_verdict": False,
    }


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldPredictiveEvaluationError(code) from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldPredictiveEvaluationError(code)
    return payload
