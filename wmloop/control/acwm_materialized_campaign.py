"""Fail-closed campaign boundary for source-grounded materialized ACWM methods."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import (
    canonical_json_bytes,
    load_mapping,
    load_measurement_receipt as load_baseline_measurement,
    sha256_file,
)
from wmloop.control.acwm_dual_evaluation import (
    ACWMDualEvaluationError,
    assess_acwm_dual_evaluation,
    validate_acwm_dual_evaluation_contract,
)
from wmloop.control.method_candidate_compiler import compile_method_candidates
from wmloop.control.masked_adapter_training import (
    MaskedAdapterTrainingError,
    bind_training_artifacts,
    training_binding_from_receipt,
)


class ACWMMaterializedCampaignError(ValueError):
    """A materialized method crossed an evidence or execution boundary."""


_HISTORY_CANDIDATE_KIND = "materialized_history_retrieval"
_MASKED_ACTION_CANDIDATE_KIND = "materialized_masked_intermediate_action_adapter"
_CANDIDATE_KINDS = {_HISTORY_CANDIDATE_KIND, _MASKED_ACTION_CANDIDATE_KIND}
_TERMINAL_STATES = {"eligible_for_confirm", "qualified_for_frozen_verifier", "abstained", "failed"}


def batch_digest(batch: Mapping[str, object]) -> str:
    payload = {key: value for key, value in batch.items() if key != "batch_digest"}
    return hashlib.sha256(canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def candidate_binding_digest(candidate: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(candidate).rstrip(b"\n")).hexdigest()


def compile_materialized_candidate_batch(
    *,
    catalog_path: Path,
    assessment_path: Path,
    contract: Mapping[str, object],
    stage: str,
    batch_id: str,
    objective: str,
    hypothesis: str,
    falsification_criterion: str,
    selection_reason: str,
    expected_gpu_hours_per_candidate: float,
    root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compile one admitted catalog candidate into an immutable execution batch."""

    catalog_source = _require_file(catalog_path, "ACWM_MATERIALIZED_CATALOG_INVALID")
    assessment_source = _require_file(
        assessment_path, "ACWM_MATERIALIZED_SOURCE_ASSESSMENT_INVALID"
    )
    catalog = _load(catalog_source, "ACWM_MATERIALIZED_CATALOG_INVALID")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CATALOG_CANDIDATE_SET_INVALID")
    catalog_candidate = candidates[0]
    compiled_batch: dict[str, object] = {"candidates": []}
    compilation = compile_method_candidates(
        batch=compiled_batch,
        catalog_path=catalog_source,
        diagnostic_probe=None,
    )
    if compilation.get("state") != "ready" or compilation.get("compiled_candidate_count") != 1:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CANDIDATE_COMPILATION_BLOCKED")
    compiled = compiled_batch["candidates"]
    assert isinstance(compiled, list) and len(compiled) == 1
    template = compiled[0]
    assert isinstance(template, Mapping)
    assessment = _load(assessment_source, "ACWM_MATERIALIZED_SOURCE_ASSESSMENT_INVALID")
    receipt_path = _require_file(
        Path(str(catalog_candidate["materialization_receipt_path"])),
        "ACWM_MATERIALIZED_RECEIPT_INVALID",
    )
    receipt = _load(receipt_path, "ACWM_MATERIALIZED_RECEIPT_INVALID")
    descriptor_path = _descriptor_from_receipt(receipt_path, receipt)
    descriptor = _load(descriptor_path, "ACWM_MATERIALIZED_DESCRIPTOR_INVALID")
    provenance = {
        "candidate_catalog_path": str(catalog_source),
        "candidate_catalog_sha256": sha256_file(catalog_source),
        "materialization_receipt_path": str(receipt_path),
        "materialization_receipt_sha256": sha256_file(receipt_path),
        "implementation_revision": receipt.get("implementation_revision"),
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": sha256_file(descriptor_path),
        "source_assessment_path": str(assessment_source),
        "source_assessment_sha256": sha256_file(assessment_source),
        "source_id": assessment.get("source_id"),
        "source_digest": assessment.get("source_digest"),
        "assessment_digest": assessment.get("assessment_digest"),
        "source_component_mapping": [
            {
                "source_component_id": row.get("source_component_id"),
                "touchpoint": row.get("touchpoint"),
            }
            for row in descriptor.get("intent_to_code", [])
            if isinstance(row, Mapping)
        ],
        "required_files": [dict(row) for row in catalog_candidate.get("required_files", [])],
    }
    candidate = {
        "candidate_id": template.get("candidate_id"),
        "candidate_kind": template.get("candidate_kind"),
        "parameters": dict(template.get("parameters", {})),
        "provenance": provenance,
    }
    batch = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-candidate-batch",
        "batch_id": batch_id,
        "batch_digest": "",
        "contract_id": contract.get("contract_id"),
        "contract_digest": contract.get("contract_digest"),
        "stage": stage,
        "objective": objective,
        "hypothesis": hypothesis,
        "falsification_criterion": falsification_criterion,
        "selection_reason": selection_reason,
        "expected_gpu_hours_per_candidate": expected_gpu_hours_per_candidate,
        "artifact_policy": (
            "Preserve candidate, source, implementation, measurement, worker, and settlement "
            "evidence outside both source repositories; only a frozen verifier may settle a claim."
        ),
        "candidates": [candidate],
    }
    batch["batch_digest"] = batch_digest(batch)
    # Compilation may produce a pending training skeleton.  The public validator
    # remains strict and is called again after the adapter-only fit stage.
    validate_materialized_candidate_batch(
        batch, contract, root=root, allow_training_pending=True
    )
    return batch, compilation


def validate_materialized_candidate_batch(
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    root: Path | None = None,
    allow_training_pending: bool = False,
) -> None:
    try:
        validate_document("acwm_materialized_candidate_batch", batch, root=root)
    except ContractValidationError as exc:
        raise ACWMMaterializedCampaignError(
            f"ACWM_MATERIALIZED_BATCH_SCHEMA_INVALID:{exc}"
        ) from exc
    validate_acwm_dual_evaluation_contract(contract, root=root)
    if batch.get("batch_digest") != batch_digest(batch):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_DIGEST_MISMATCH")
    if batch.get("contract_id") != contract.get("contract_id"):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_CONTRACT_ID_MISMATCH")
    if batch.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_CONTRACT_DIGEST_MISMATCH")
    stage = str(batch["stage"])
    stage_rows = [
        row
        for row in contract["stages"]
        if isinstance(row, Mapping) and row.get("stage") == stage
    ]
    if len(stage_rows) != 1:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_STAGE_INVALID")
    cost = float(batch["expected_gpu_hours_per_candidate"])
    if not math.isfinite(cost) or cost > float(stage_rows[0]["max_gpu_hours_per_candidate"]):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_COST_EXCEEDS_LIMIT")
    candidates = batch["candidates"]
    assert isinstance(candidates, list)
    resource = contract["resource_policy"]
    assert isinstance(resource, Mapping)
    if len(candidates) > int(resource["max_parallel_candidates"]):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_OVERSUBSCRIBED")
    if len(candidates) * int(resource["per_candidate_gpus"]) > int(resource["total_gpus"]):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_BATCH_GPU_POLICY_INVALID")
    identifiers: set[str] = set()
    for candidate in candidates:
        assert isinstance(candidate, Mapping)
        identifier = str(candidate["candidate_id"])
        if identifier in identifiers:
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CANDIDATE_DUPLICATE")
        identifiers.add(identifier)
        validate_materialized_candidate(
            candidate, allow_training_pending=allow_training_pending
        )


def validate_materialized_candidate(
    candidate: Mapping[str, object], *, allow_training_pending: bool = False
) -> None:
    kind = candidate.get("candidate_kind")
    if kind not in _CANDIDATE_KINDS:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CANDIDATE_KIND_INVALID")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_PARAMETERS_INVALID")
    if kind == _HISTORY_CANDIDATE_KIND:
        expected = {"max_items", "token_grid_size", "spatial_weight", "action_weight", "temporal_weight"}
        if set(parameters) != expected:
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_HISTORY_PARAMETERS_INVALID")
        weights = [parameters.get(name) for name in ("spatial_weight", "action_weight", "temporal_weight")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in weights):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_RELEVANCE_WEIGHT_INVALID")
        if sum(float(value) for value in weights) <= 0:
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_RELEVANCE_WEIGHT_INVALID")
    else:
        _validate_masked_action_parameters(parameters)
        if not isinstance(candidate.get("provenance"), Mapping):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_PROVENANCE_INVALID")
        training_binding = candidate["provenance"].get("training_binding")
        if training_binding is None:
            if not allow_training_pending:
                raise ACWMMaterializedCampaignError(
                    "ACWM_MATERIALIZED_TRAINING_BINDING_REQUIRED"
                )
        else:
            _validate_training_binding(
                candidate_id=str(candidate.get("candidate_id")),
                parameters=parameters,
                training_binding=training_binding,
            )

    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_PROVENANCE_INVALID")
    catalog_path = _bound_file(provenance, "candidate_catalog", "ACWM_MATERIALIZED_CATALOG")
    receipt_path = _bound_file(provenance, "materialization_receipt", "ACWM_MATERIALIZED_RECEIPT")
    descriptor_path = _bound_file(provenance, "descriptor", "ACWM_MATERIALIZED_DESCRIPTOR")
    assessment_path = _bound_file(provenance, "source_assessment", "ACWM_MATERIALIZED_SOURCE_ASSESSMENT")
    catalog = _load(catalog_path, "ACWM_MATERIALIZED_CATALOG_INVALID")
    receipt = _load(receipt_path, "ACWM_MATERIALIZED_RECEIPT_INVALID")
    descriptor = _load(descriptor_path, "ACWM_MATERIALIZED_DESCRIPTOR_INVALID")
    assessment = _load(assessment_path, "ACWM_MATERIALIZED_SOURCE_ASSESSMENT_INVALID")
    if (
        receipt.get("state") != "ready_for_candidate_compilation"
        or receipt.get("candidate_id") != candidate.get("candidate_id")
        or receipt.get("implementation_revision") != provenance.get("implementation_revision")
        or receipt.get("descriptor_sha256") != provenance.get("descriptor_sha256")
    ):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_RECEIPT_BINDING_MISMATCH")
    if descriptor.get("candidate_id") != candidate.get("candidate_id") or descriptor.get("declared_compromises") != []:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_DESCRIPTOR_BINDING_MISMATCH")
    mappings = [
        {"source_component_id": row.get("source_component_id"), "touchpoint": row.get("touchpoint")}
        for row in descriptor.get("intent_to_code", [])
        if isinstance(row, Mapping)
    ]
    if mappings != provenance.get("source_component_mapping"):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_COMPONENT_MAPPING_MISMATCH")
    if (
        assessment.get("source_id") != provenance.get("source_id")
        or assessment.get("source_digest") != provenance.get("source_digest")
        or assessment.get("assessment_digest") != provenance.get("assessment_digest")
        or assessment.get("execution_state") != "materialization_required"
        or assessment.get("source_role") != "transferable_optimizer"
    ):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_SOURCE_BINDING_MISMATCH")
    catalog_rows = catalog.get("candidates")
    if not isinstance(catalog_rows, list) or len(catalog_rows) != 1 or not isinstance(catalog_rows[0], Mapping):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CATALOG_BINDING_MISMATCH")
    catalog_candidate = catalog_rows[0]
    if catalog_candidate.get("candidate_id") != candidate.get("candidate_id"):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CATALOG_BINDING_MISMATCH")
    expected_source = (
        f"{provenance['source_id']}@sha256:{provenance['source_digest']}"
        f"#assessment-sha256:{provenance['assessment_digest']}"
    )
    if catalog_candidate.get("source") != expected_source:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CATALOG_SOURCE_MISMATCH")
    required = provenance.get("required_files")
    if not isinstance(required, list) or not required:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_REQUIRED_FILES_INVALID")
    for binding in required:
        if not isinstance(binding, Mapping):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_REQUIRED_FILES_INVALID")
        _bound_named_file(binding)


def _validate_masked_action_parameters(parameters: Mapping[str, object]) -> None:
    expected = {"action_dim", "hidden_dim", "max_residual", "mask_temperature", "adapter_scale"}
    if set(parameters) != expected:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MASKED_ACTION_PARAMETERS_INVALID")
    for name in ("action_dim", "hidden_dim"):
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "action_dim" else 8):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MASKED_ACTION_PARAMETERS_INVALID")
    for name, lower, upper in (
        ("max_residual", 0.0, 1.0),
        ("mask_temperature", 0.05, 10.0),
        ("adapter_scale", 0.0, 1.0),
    ):
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MASKED_ACTION_PARAMETERS_INVALID")
        normalized = float(value)
        strictly_positive = name in {"max_residual", "adapter_scale"}
        if (
            not math.isfinite(normalized)
            or (normalized <= lower if strictly_positive else normalized < lower)
            or normalized > upper
        ):
            raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MASKED_ACTION_PARAMETERS_INVALID")


def _validate_training_binding(
    *,
    candidate_id: str,
    parameters: Mapping[str, object],
    training_binding: object,
) -> None:
    if not isinstance(training_binding, Mapping):
        raise ACWMMaterializedCampaignError(
            "ACWM_MATERIALIZED_TRAINING_BINDING_INVALID"
        )
    receipt_path = Path(str(training_binding.get("training_receipt_path", "")))
    try:
        expected = training_binding_from_receipt(
            receipt_path,
            expected_candidate_id=candidate_id,
            expected_parameters=parameters,
        )
    except (MaskedAdapterTrainingError, OSError, ValueError) as exc:
        raise ACWMMaterializedCampaignError(str(exc)) from exc
    if dict(training_binding) != expected:
        raise ACWMMaterializedCampaignError(
            "ACWM_MATERIALIZED_TRAINING_BINDING_MISMATCH"
        )


def bind_training_to_batch(
    batch: Mapping[str, object],
    *,
    receipt_path: Path,
    contract: Mapping[str, object],
    root: Path | None = None,
) -> dict[str, object]:
    """Attach a completed adapter fit and re-seal the candidate batch digest."""

    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_CANDIDATE_SET_INVALID")
    candidate = candidates[0]
    if candidate.get("candidate_kind") != _MASKED_ACTION_CANDIDATE_KIND:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_TRAINING_NOT_APPLICABLE")
    try:
        updated_candidate = bind_training_artifacts(candidate, receipt_path)
    except MaskedAdapterTrainingError as exc:
        raise ACWMMaterializedCampaignError(str(exc)) from exc
    updated = dict(batch)
    updated["candidates"] = [updated_candidate]
    updated["batch_digest"] = batch_digest({**updated, "batch_digest": ""})
    validate_materialized_candidate_batch(updated, contract, root=root)
    return updated


def build_evaluator_command(
    *,
    runtime_python: Path,
    evaluator: Path,
    base_evaluator: Path,
    contract: Path,
    stage: str,
    candidate_path: Path,
    output_root: Path,
    ctrl_world_root: Path,
    dataset_root: Path,
    dataset_name: str = "droid_subset",
    data_stat: Path,
    checkpoint: Path,
    svd_model: Path,
    clip_model: Path,
) -> list[str]:
    return [
        str(Path(runtime_python).resolve()),
        str(Path(evaluator).resolve()),
        "--base-evaluator", str(Path(base_evaluator).resolve()),
        "--contract", str(Path(contract).resolve()),
        "--stage", stage,
        "--candidate", str(Path(candidate_path).resolve()),
        "--ctrl-world-root", str(Path(ctrl_world_root).resolve()),
        "--dataset-root", str(Path(dataset_root).resolve()),
        "--dataset-name", dataset_name,
        "--data-stat", str(Path(data_stat).resolve()),
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--svd-model", str(Path(svd_model).resolve()),
        "--clip-model", str(Path(clip_model).resolve()),
        "--output-root", str(Path(output_root).resolve()),
    ]


def load_materialized_measurement(
    path: Path,
    *,
    contract: Mapping[str, object],
    stage: str,
    candidate: Mapping[str, object],
    root: Path | None = None,
) -> tuple[dict[str, object], str]:
    measurement_path = _require_file(path, "ACWM_MATERIALIZED_MEASUREMENT_INVALID")
    measurement = _load(measurement_path, "ACWM_MATERIALIZED_MEASUREMENT_INVALID")
    manifest = _load(
        _require_file(measurement_path.parent / "manifest.json", "ACWM_MATERIALIZED_MEASUREMENT_MANIFEST_INVALID"),
        "ACWM_MATERIALIZED_MEASUREMENT_MANIFEST_INVALID",
    )
    digest = sha256_file(measurement_path)
    if (
        manifest.get("measurement_sha256") != digest
        or manifest.get("candidate_binding_sha256") != candidate_binding_digest(candidate)
        or manifest.get("stage") != stage
        or manifest.get("contract_digest") != contract.get("contract_digest")
    ):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MEASUREMENT_MANIFEST_MISMATCH")
    if measurement.get("candidate") != candidate:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_MEASUREMENT_CANDIDATE_MISMATCH")
    try:
        assess_acwm_dual_evaluation(
            contract, stage=stage, baseline=measurement, candidate=measurement, root=root
        )
    except ACWMDualEvaluationError as exc:
        raise ACWMMaterializedCampaignError(
            f"ACWM_MATERIALIZED_MEASUREMENT_CONTRACT_INVALID:{exc}"
        ) from exc
    return measurement, digest


def settle_materialized_candidate(
    *,
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    baseline: Mapping[str, object],
    baseline_path: Path,
    baseline_sha256: str,
    candidate: Mapping[str, object],
    measurement: Mapping[str, object],
    measurement_path: Path,
    measurement_sha256: str,
    gpu_index: int,
    root: Path | None = None,
) -> dict[str, object]:
    if measurement.get("candidate") != candidate:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_SETTLEMENT_CANDIDATE_MISMATCH")
    try:
        assessment = assess_acwm_dual_evaluation(
            contract,
            stage=str(batch["stage"]),
            baseline=baseline,
            candidate=measurement,
            root=root,
        )
    except ACWMDualEvaluationError as exc:
        raise ACWMMaterializedCampaignError(
            f"ACWM_MATERIALIZED_ASSESSMENT_INVALID:{exc}"
        ) from exc
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-candidate-settlement",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": batch["stage"],
        "candidate": dict(candidate),
        "candidate_binding_sha256": candidate_binding_digest(candidate),
        "gpu_index": gpu_index,
        "state": assessment["state"],
        "accepted": assessment["accepted"],
        "metrics": measurement["metrics"],
        "metric_deltas": assessment["metric_deltas"],
        "blockers": assessment["blockers"],
        "baseline_receipt": {"path": str(Path(baseline_path).resolve()), "sha256": baseline_sha256},
        "candidate_receipt": {"path": str(Path(measurement_path).resolve()), "sha256": measurement_sha256},
        "verdict_authority": False,
        "claim_boundary": (
            "This settlement records one source-grounded screen or confirmation result. "
            "Only the independently frozen materialized-method verifier may settle knowledge."
        ),
    }


def failed_materialized_candidate(
    *,
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: Mapping[str, object],
    gpu_index: int,
    failure_code: str,
    worker_receipt_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-candidate-settlement",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": batch["stage"],
        "candidate": dict(candidate),
        "candidate_binding_sha256": candidate_binding_digest(candidate),
        "gpu_index": gpu_index,
        "state": "failed",
        "accepted": False,
        "metrics": None,
        "metric_deltas": {},
        "blockers": [failure_code],
        "worker_receipt_path": str(Path(worker_receipt_path).resolve()),
        "verdict_authority": False,
        "claim_boundary": "Operational failure evidence only; no performance claim is made.",
    }


def terminal_state(value: Mapping[str, object]) -> bool:
    return str(value.get("state")) in _TERMINAL_STATES


def load_baseline(
    path: Path,
    *,
    contract: Mapping[str, object],
    stage: str,
    root: Path | None = None,
) -> tuple[dict[str, object], str]:
    return load_baseline_measurement(path, contract=contract, stage=stage, root=root)


def _bound_file(provenance: Mapping[str, object], stem: str, code: str) -> Path:
    path = _require_file(Path(str(provenance.get(f"{stem}_path", ""))), f"{code}_INVALID")
    if sha256_file(path) != provenance.get(f"{stem}_sha256"):
        raise ACWMMaterializedCampaignError(f"{code}_HASH_MISMATCH")
    return path


def _bound_named_file(binding: Mapping[str, object]) -> Path:
    path = _require_file(
        Path(str(binding.get("path", ""))), "ACWM_MATERIALIZED_REQUIRED_FILE_INVALID"
    )
    if sha256_file(path) != binding.get("sha256"):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_REQUIRED_FILE_HASH_MISMATCH")
    return path


def _descriptor_from_receipt(
    receipt_path: Path, receipt: Mapping[str, object]
) -> Path:
    expected = receipt.get("descriptor_sha256")
    changed_paths = receipt.get("changed_paths")
    if not isinstance(expected, str) or not isinstance(changed_paths, list):
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_DESCRIPTOR_INVALID")
    workspace = receipt_path.parent / "workspace"
    matches = []
    for raw in changed_paths:
        if not isinstance(raw, str):
            continue
        candidate = workspace / raw
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if sha256_file(candidate) != expected:
            continue
        payload = _load(candidate, "ACWM_MATERIALIZED_DESCRIPTOR_INVALID")
        if payload.get("artifact_type") == "verdiwm-materialized-method-descriptor":
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ACWMMaterializedCampaignError("ACWM_MATERIALIZED_DESCRIPTOR_INVALID")
    return matches[0]


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ACWMMaterializedCampaignError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise ACWMMaterializedCampaignError(code)
    return resolved


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMMaterializedCampaignError(code) from exc
    if not isinstance(payload, dict):
        raise ACWMMaterializedCampaignError(code)
    return payload


__all__: Sequence[str] = (
    "ACWMMaterializedCampaignError",
    "batch_digest",
    "bind_training_to_batch",
    "build_evaluator_command",
    "candidate_binding_digest",
    "compile_materialized_candidate_batch",
    "failed_materialized_candidate",
    "load_baseline",
    "load_mapping",
    "load_materialized_measurement",
    "settle_materialized_candidate",
    "sha256_file",
    "terminal_state",
    "validate_materialized_candidate",
    "validate_materialized_candidate_batch",
)
