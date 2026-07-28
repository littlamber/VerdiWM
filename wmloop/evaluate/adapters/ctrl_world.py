"""Read-only Ctrl-World rollout receipt adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, validate_document


class CtrlWorldEvaluationError(ValueError):
    """Ctrl-World evidence is malformed or outside the frozen split."""


def evaluate_ctrl_world_receipt(*, receipt_path: Path, heldout_split_path: Path, split_name: str) -> dict[str, object]:
    receipt = _load_json(receipt_path, "CTRL_WORLD_RECEIPT_READ_FAILED")
    split = _load_json(heldout_split_path, "CTRL_WORLD_SPLIT_READ_FAILED")
    try:
        validate_document("ctrl_world_rollout_receipt", receipt)
        validate_document("ctrl_world_heldout_split", split)
    except ContractValidationError as exc:
        raise CtrlWorldEvaluationError(f"CTRL_WORLD_CONTRACT_INVALID:{exc}") from exc
    if split_name not in {"dev", "accept"}:
        raise CtrlWorldEvaluationError("CTRL_WORLD_SPLIT_NAME_INVALID")
    identity = (str(receipt["task_id"]), str(receipt["episode_id"]), int(receipt["seed"]))
    allowed = {(str(item["task_id"]), str(item["episode_id"]), int(item["seed"])) for item in split[split_name]}
    if identity not in allowed:
        raise CtrlWorldEvaluationError("CTRL_WORLD_RECEIPT_OUTSIDE_FROZEN_SPLIT")
    if receipt["evidence_source"] != "environment_task_receipt":
        raise CtrlWorldEvaluationError("CTRL_WORLD_VERDICT_SOURCE_FORBIDDEN")
    safety = list(receipt["safety_events"])
    validity = bool(receipt["action_valid"]) and not safety
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-verdict-evidence",
        "task_id": receipt["task_id"],
        "episode_id": receipt["episode_id"],
        "seed": receipt["seed"],
        "split": split_name,
        "task_success": receipt["task_success"],
        "task_progress": receipt["task_progress"],
        "safety_events": safety,
        "action_valid": receipt["action_valid"],
        "validity_gate_pass": validity,
        "accept_eligible": validity and bool(receipt["task_success"]),
        "rollout_ref": receipt["rollout_ref"],
        "evidence_source": receipt["evidence_source"],
        "generated_video_scores_used_for_verdict": False,
    }


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldEvaluationError(code) from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldEvaluationError(code)
    return payload
