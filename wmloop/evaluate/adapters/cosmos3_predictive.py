"""Read-only ACWM predictive receipt adapter for Cosmos3 forward dynamics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class Cosmos3PredictiveEvaluationError(ValueError):
    """Cosmos3 evidence is malformed, policy-side, or outside the frozen split."""


def evaluate_cosmos3_prediction_receipt(
    *,
    receipt_path: Path,
    heldout_split_path: Path,
    split_name: str,
) -> dict[str, object]:
    receipt = _load_json(receipt_path, "COSMOS3_PREDICTION_RECEIPT_READ_FAILED")
    split = _load_json(heldout_split_path, "COSMOS3_SPLIT_READ_FAILED")
    try:
        validate_document("cosmos3_prediction_receipt", receipt)
        validate_document("cosmos3_forward_dynamics_split", split)
    except ContractValidationError as exc:
        raise Cosmos3PredictiveEvaluationError(f"COSMOS3_PREDICTION_CONTRACT_INVALID:{exc}") from exc
    if split_name not in {"dev", "accept"}:
        raise Cosmos3PredictiveEvaluationError("COSMOS3_SPLIT_NAME_INVALID")
    identity = (int(receipt["sample_index"]), int(receipt["seed"]))
    allowed = {(int(item["sample_index"]), int(item["seed"])) for item in split[split_name]}
    if identity not in allowed:
        raise Cosmos3PredictiveEvaluationError("COSMOS3_PREDICTION_RECEIPT_OUTSIDE_FROZEN_SPLIT")
    if receipt["split_id"] != split["split_id"] or receipt["split_name"] != split_name:
        raise Cosmos3PredictiveEvaluationError("COSMOS3_PREDICTION_SPLIT_IDENTITY_MISMATCH")
    if receipt["dataset_freeze_id"] != "cosmos3_droid_lerobot_cookbook_sample_v1":
        raise Cosmos3PredictiveEvaluationError("COSMOS3_DATASET_FREEZE_IDENTITY_MISMATCH")
    if receipt["model_mode"] != "forward_dynamics":
        raise Cosmos3PredictiveEvaluationError("COSMOS3_POLICY_MODE_FORBIDDEN")
    if receipt["evidence_source"] != "paired_ground_truth_rollout":
        raise Cosmos3PredictiveEvaluationError("COSMOS3_DOWNSTREAM_SUCCESS_FORBIDDEN")
    if receipt["action_conditioned"] is not True:
        raise Cosmos3PredictiveEvaluationError("COSMOS3_ACTION_CONDITIONING_MISSING")
    if receipt["viewpoint"] != "concat_view" or receipt["action_shape"] != [16, 10]:
        raise Cosmos3PredictiveEvaluationError("COSMOS3_FORWARD_DYNAMICS_SHAPE_OR_VIEWPOINT_INVALID")
    alignment = receipt["frame_alignment"]
    if (
        receipt["horizon_frames"] != 16
        or len(alignment["ground_truth_shape"]) != 4
        or len(alignment["rollout_shape"]) != 4
        or alignment["condition_frame_index"] != 0
        or alignment["future_start_index"] != 1
        or alignment["future_frame_count"] != 16
        or alignment["spatial_policy"] != "top_left_content_crop_to_rollout"
        or alignment["conditioning_frame_mae"] > alignment["max_conditioning_frame_mae"]
    ):
        raise Cosmos3PredictiveEvaluationError("COSMOS3_FRAME_ALIGNMENT_INVALID")
    metrics = {str(key): float(value) for key, value in receipt["metrics"].items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise Cosmos3PredictiveEvaluationError("COSMOS3_PREDICTION_METRIC_NONFINITE")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-predictive-verdict-evidence",
        "sample_index": receipt["sample_index"],
        "seed": receipt["seed"],
        "split": split_name,
        "horizon_frames": receipt["horizon_frames"],
        "metrics": metrics,
        "action_conditioned": True,
        "model_mode": "forward_dynamics",
        "rollout_ref": receipt["rollout_ref"],
        "evidence_source": receipt["evidence_source"],
        "downstream_task_success_used_for_verdict": False,
    }


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Cosmos3PredictiveEvaluationError(code) from exc
    if not isinstance(payload, Mapping):
        raise Cosmos3PredictiveEvaluationError(code)
    return payload
