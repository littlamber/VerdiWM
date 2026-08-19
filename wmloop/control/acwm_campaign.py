"""Small fail-closed control plane for frozen Ctrl-World ACWM candidate batches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_dual_evaluation import (
    ACWMDualEvaluationError,
    assess_acwm_dual_evaluation,
    validate_acwm_dual_evaluation_contract,
)


class ACWMCampaignError(ValueError):
    """A candidate batch, receipt, or settlement crossed an ACWM boundary."""


_CANDIDATE_KIND = "inference_guidance_scale"
_TERMINAL_STATES = {"eligible_for_confirm", "abstained", "failed"}


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Encode an evidence record deterministically."""

    try:
        return (
            json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ACWMCampaignError("ACWM_CAMPAIGN_PAYLOAD_INVALID") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise ACWMCampaignError("ACWM_CAMPAIGN_FILE_UNREADABLE") from exc


def load_mapping(path: Path, *, error_code: str) -> dict[str, object]:
    """Load one JSON object without accepting symlinked evidence."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ACWMCampaignError(error_code)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMCampaignError(error_code) from exc
    if not isinstance(payload, dict):
        raise ACWMCampaignError(error_code)
    return payload


def batch_digest(batch: Mapping[str, object]) -> str:
    payload = {key: value for key, value in batch.items() if key != "batch_digest"}
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_PAYLOAD_INVALID") from exc
    return sha256_bytes(encoded)


def validate_acwm_candidate_batch(
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    root: Path | None = None,
) -> None:
    """Validate immutable, one-GPU ACWM inference candidates against a contract."""

    try:
        validate_document("acwm_candidate_batch", batch, root=root)
    except ContractValidationError as exc:
        raise ACWMCampaignError(f"ACWM_CANDIDATE_BATCH_SCHEMA_INVALID:{exc}") from exc
    validate_acwm_dual_evaluation_contract(contract, root=root)
    if batch.get("batch_digest") != batch_digest(batch):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_DIGEST_MISMATCH")
    if batch.get("contract_id") != contract.get("contract_id"):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CONTRACT_ID_MISMATCH")
    if batch.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CONTRACT_DIGEST_MISMATCH")

    stage = str(batch["stage"])
    stage_rows = [row for row in contract["stages"] if isinstance(row, Mapping) and row.get("stage") == stage]
    if len(stage_rows) != 1:
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_STAGE_INVALID")
    expected_cost = float(batch["expected_gpu_hours_per_candidate"])
    if not math.isfinite(expected_cost) or expected_cost > float(stage_rows[0]["max_gpu_hours_per_candidate"]):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_COST_EXCEEDS_STAGE_LIMIT")

    candidates = batch["candidates"]
    if not isinstance(candidates, list):  # Schema validation makes this defensive branch unreachable.
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CANDIDATES_INVALID")
    resource_policy = contract["resource_policy"]
    assert isinstance(resource_policy, Mapping)
    if len(candidates) > int(resource_policy["max_parallel_candidates"]):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_OVERSUBSCRIBED")
    if len(candidates) * int(resource_policy["per_candidate_gpus"]) > int(resource_policy["total_gpus"]):
        raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_GPU_POLICY_INVALID")

    candidate_ids: set[str] = set()
    scales: set[float] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CANDIDATE_INVALID")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in candidate_ids:
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CANDIDATE_ID_DUPLICATE")
        candidate_ids.add(candidate_id)
        if candidate.get("candidate_kind") != _CANDIDATE_KIND:
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_CANDIDATE_KIND_INVALID")
        guidance_scale = candidate.get("guidance_scale")
        if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)):
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_GUIDANCE_SCALE_INVALID")
        normalized = float(guidance_scale)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_GUIDANCE_SCALE_INVALID")
        if normalized in scales:
            raise ACWMCampaignError("ACWM_CANDIDATE_BATCH_GUIDANCE_SCALE_DUPLICATE")
        scales.add(normalized)


def load_measurement_receipt(
    path: Path,
    *,
    contract: Mapping[str, object],
    stage: str,
    root: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load a hashed evaluator receipt and ensure it belongs to this campaign."""

    measurement_path = Path(path).expanduser().resolve()
    measurement = load_mapping(measurement_path, error_code="ACWM_MEASUREMENT_INVALID")
    manifest_path = measurement_path.parent / "manifest.json"
    manifest = load_mapping(manifest_path, error_code="ACWM_MEASUREMENT_MANIFEST_INVALID")
    measurement_sha256 = sha256_file(measurement_path)
    if manifest.get("measurement_sha256") != measurement_sha256:
        raise ACWMCampaignError("ACWM_MEASUREMENT_HASH_MISMATCH")
    if manifest.get("stage") != stage or manifest.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMCampaignError("ACWM_MEASUREMENT_MANIFEST_BINDING_MISMATCH")
    try:
        # A self-comparison is deliberately rejected by the Pareto gate, but validates all receipt bindings.
        assess_acwm_dual_evaluation(
            contract, stage=stage, baseline=measurement, candidate=measurement, root=root
        )
    except ACWMDualEvaluationError as exc:
        raise ACWMCampaignError(f"ACWM_MEASUREMENT_CONTRACT_INVALID:{exc}") from exc
    return measurement, measurement_sha256


def build_evaluator_command(
    *,
    runtime_python: Path,
    evaluator: Path,
    contract: Path,
    stage: str,
    candidate: Mapping[str, object],
    output_root: Path,
    ctrl_world_root: Path,
    dataset_root: Path,
    data_stat: Path,
    checkpoint: Path,
    svd_model: Path,
    clip_model: Path,
) -> list[str]:
    """Build the source-preserving evaluator command for exactly one candidate."""

    if candidate.get("candidate_kind") != _CANDIDATE_KIND:
        raise ACWMCampaignError("ACWM_COMMAND_CANDIDATE_KIND_INVALID")
    scale = candidate.get("guidance_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ACWMCampaignError("ACWM_COMMAND_GUIDANCE_SCALE_INVALID")
    return [
        # Preserve the caller's interpreter path.  Resolving a venv's Python
        # symlink can bypass its isolated site-packages and silently select the
        # host environment instead.
        str(Path(runtime_python).expanduser()),
        str(Path(evaluator).expanduser().resolve()),
        "--contract",
        str(Path(contract).expanduser().resolve()),
        "--stage",
        stage,
        "--ctrl-world-root",
        str(Path(ctrl_world_root).expanduser().resolve()),
        "--dataset-root",
        str(Path(dataset_root).expanduser().resolve()),
        "--data-stat",
        str(Path(data_stat).expanduser().resolve()),
        "--checkpoint",
        str(Path(checkpoint).expanduser().resolve()),
        "--svd-model",
        str(Path(svd_model).expanduser().resolve()),
        "--clip-model",
        str(Path(clip_model).expanduser().resolve()),
        "--candidate-id",
        str(candidate["candidate_id"]),
        "--guidance-scale",
        str(float(scale)),
        "--output-root",
        str(Path(output_root).expanduser().resolve()),
    ]


def settle_acwm_candidate(
    *,
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    baseline: Mapping[str, object],
    baseline_sha256: str,
    baseline_path: Path,
    candidate: Mapping[str, object],
    measurement: Mapping[str, object],
    measurement_sha256: str,
    measurement_path: Path,
    gpu_index: int,
    root: Path | None = None,
) -> dict[str, object]:
    """Settle one valid screen or confirm receipt and preserve its claim boundary."""

    _validate_candidate_measurement(candidate, measurement)
    try:
        assessment = assess_acwm_dual_evaluation(
            contract,
            stage=str(batch["stage"]),
            baseline=baseline,
            candidate=measurement,
            root=root,
        )
    except ACWMDualEvaluationError as exc:
        raise ACWMCampaignError(f"ACWM_CANDIDATE_ASSESSMENT_INVALID:{exc}") from exc
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-candidate-settlement",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": batch["stage"],
        "candidate": _candidate_descriptor(candidate),
        "gpu_index": gpu_index,
        "state": assessment["state"],
        "accepted": assessment["accepted"],
        "metrics": measurement["metrics"],
        "metric_deltas": assessment["metric_deltas"],
        "blockers": assessment["blockers"],
        "baseline_receipt": {"path": str(Path(baseline_path).resolve()), "sha256": baseline_sha256},
        "candidate_receipt": {"path": str(Path(measurement_path).resolve()), "sha256": measurement_sha256},
        "verdict_authority": False,
        "claim_boundary": "A screen or confirm settlement is reusable ACWM evidence only; promotion requires the separately frozen verifier.",
    }


def failed_acwm_candidate(
    *,
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: Mapping[str, object],
    gpu_index: int,
    failure_code: str,
    worker_receipt_path: Path,
) -> dict[str, object]:
    """Create a durable terminal row for a worker or receipt failure."""

    if not failure_code:
        raise ACWMCampaignError("ACWM_CANDIDATE_FAILURE_CODE_MISSING")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-candidate-settlement",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": batch["stage"],
        "candidate": _candidate_descriptor(candidate),
        "gpu_index": gpu_index,
        "state": "failed",
        "accepted": False,
        "metrics": None,
        "metric_deltas": {},
        "blockers": [failure_code],
        "worker_receipt_path": str(Path(worker_receipt_path).resolve()),
        "verdict_authority": False,
        "claim_boundary": "A failed worker is archived as an operational result and makes no ACWM performance claim.",
    }


def terminal_state(value: Mapping[str, object]) -> bool:
    return str(value.get("state")) in _TERMINAL_STATES


def _candidate_descriptor(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "parameters": {"guidance_scale": float(candidate["guidance_scale"])},
    }


def _validate_candidate_measurement(candidate: Mapping[str, object], measurement: Mapping[str, object]) -> None:
    embedded = measurement.get("candidate")
    if not isinstance(embedded, Mapping):
        raise ACWMCampaignError("ACWM_CANDIDATE_MEASUREMENT_METADATA_MISSING")
    if embedded.get("candidate_id") != candidate.get("candidate_id"):
        raise ACWMCampaignError("ACWM_CANDIDATE_MEASUREMENT_ID_MISMATCH")
    if embedded.get("candidate_kind") != candidate.get("candidate_kind"):
        raise ACWMCampaignError("ACWM_CANDIDATE_MEASUREMENT_KIND_MISMATCH")
    parameters = embedded.get("parameters")
    if not isinstance(parameters, Mapping) or parameters.get("guidance_scale") != float(candidate["guidance_scale"]):
        raise ACWMCampaignError("ACWM_CANDIDATE_MEASUREMENT_PARAMETERS_MISMATCH")
