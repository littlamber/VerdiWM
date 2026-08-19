"""Compile portfolio-aware GPU admissions without embedding campaign policy in the Kernel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.experiment_portfolio import (
    ExperimentPortfolioError,
    validate_experiment_portfolio,
)


class ResourcePortfolioError(RuntimeError):
    """A resource partition or evidence-driven reallocation failed closed."""


_CTRL_WORLD_GPU_INVENTORY = list(range(8))
_CTRL_WORLD_EXPERIMENT_GPUS = list(range(6))
_CTRL_WORLD_DROID_GPUS = [6, 7]


def build_screen_resource_portfolio_receipt(
    *,
    portfolio: Mapping[str, object],
    work_id: str,
    candidate_id: str,
    config_digest: str,
    resource_allocation: Mapping[str, object],
    experiment_scale_plan: Mapping[str, object],
    conversion_scale_plan: Mapping[str, object],
    policy_id: str,
    portfolio_entry_ids: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Admit one candidate screen while retaining the complete portfolio queue."""

    _validate_portfolio(portfolio, root=root)
    partition = _role_partition(
        resource_allocation,
        experiment_scale_plan=experiment_scale_plan,
        conversion_scale_plan=conversion_scale_plan,
    )
    _require_digest(config_digest, "RESOURCE_PORTFOLIO_CONFIG_DIGEST_INVALID")
    if not isinstance(work_id, str) or not work_id:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_WORK_ID_INVALID")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_CANDIDATE_ID_INVALID")
    if not isinstance(policy_id, str) or not policy_id:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_POLICY_ID_INVALID")
    trials = _screen_trials(portfolio)
    selected = _selected_screen_trial(
        trials,
        portfolio_entry_ids=tuple(portfolio_entry_ids),
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-resource-portfolio-receipt",
        "state": "ready_for_screen",
        "phase": "screen_admission",
        "work_id": work_id,
        "bindings": {
            "config_digest": config_digest,
            "portfolio_id": portfolio["portfolio_id"],
            "portfolio_digest": portfolio["portfolio_digest"],
            "candidate_id": candidate_id,
            "parent_receipt_id": None,
            "parent_receipt_digest": None,
            "screen_evidence_digest": None,
            "training_scale_plan_digest": None,
        },
        "role_partition": partition,
        "portfolio_trials": [
            {
                **row,
                "state": (
                    "admitted"
                    if row["trial_id"] == selected["trial_id"]
                    else "conditional"
                ),
            }
            for row in trials
        ],
        "allocation": {
            "selected_trial_id": selected["trial_id"],
            "stage": "screen",
            "allowed_gpu_indices": partition["experiment"]["gpu_indices"],
            "requested_gpu_count": 1,
            "max_parallel_jobs": partition["experiment"]["max_parallel_jobs"],
            "distributed_training": False,
            "scale_rationale": None,
        },
        "reallocation_policy": _reallocation_policy(policy_id),
        "authority": _authority(),
        "artifact_policy": _artifact_policy(),
        "claim_boundary": (
            "This receipt admits one independent screen lease request from a frozen "
            "experiment portfolio. It reserves the DROID conversion role and grants no "
            "evaluator, metric, verdict, or promotion authority."
        ),
    }
    return _finalize(body, root=root)


def build_confirm_resource_portfolio_receipt(
    *,
    screen_receipt: Mapping[str, object],
    screen_evidence: Mapping[str, object],
    requested_gpu_count: int = 1,
    training_scale_plan: Mapping[str, object] | None = None,
    scale_rationale: Mapping[str, object] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Reallocate a screened trial to confirmation only from bound evidence."""

    validate_resource_portfolio_receipt(screen_receipt, root=root)
    if screen_receipt.get("phase") != "screen_admission":
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_PARENT_PHASE_INVALID")
    evidence_digest = _validate_screen_evidence(screen_receipt, screen_evidence)
    if isinstance(requested_gpu_count, bool) or not isinstance(requested_gpu_count, int):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_GPU_COUNT_INVALID")
    allowed = screen_receipt["allocation"]["allowed_gpu_indices"]
    assert isinstance(allowed, list)
    if requested_gpu_count < 1 or requested_gpu_count > len(allowed):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_GPU_COUNT_INVALID")
    scale_digest = None
    normalized_rationale = None
    if requested_gpu_count > 1:
        scale_digest, normalized_rationale = _distributed_scale_binding(
            requested_gpu_count=requested_gpu_count,
            training_scale_plan=training_scale_plan,
            scale_rationale=scale_rationale,
        )
    elif training_scale_plan is not None or scale_rationale is not None:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_UNUSED_SCALE_BINDING")
    selected_trial_id = str(screen_receipt["allocation"]["selected_trial_id"])
    selected_rows = [
        row
        for row in screen_receipt["portfolio_trials"]
        if row.get("trial_id") == selected_trial_id
    ]
    if len(selected_rows) != 1:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SELECTED_TRIAL_INVALID")
    source = selected_rows[0]
    confirm_trial = {
        **dict(source),
        "trial_id": _stable_id(
            "resource-trial",
            {
                "parent_trial_id": selected_trial_id,
                "stage": "confirm",
                "screen_evidence_digest": evidence_digest,
            },
        ),
        "stage": "confirm",
        "state": "admitted",
    }
    bindings = screen_receipt["bindings"]
    partition = screen_receipt["role_partition"]
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-resource-portfolio-receipt",
        "state": "ready_for_confirm",
        "phase": "confirm_reallocation",
        "work_id": screen_receipt["work_id"],
        "bindings": {
            "config_digest": bindings["config_digest"],
            "portfolio_id": bindings["portfolio_id"],
            "portfolio_digest": bindings["portfolio_digest"],
            "candidate_id": bindings["candidate_id"],
            "parent_receipt_id": screen_receipt["receipt_id"],
            "parent_receipt_digest": screen_receipt["receipt_digest"],
            "screen_evidence_digest": evidence_digest,
            "training_scale_plan_digest": scale_digest,
        },
        "role_partition": partition,
        "portfolio_trials": [confirm_trial],
        "allocation": {
            "selected_trial_id": confirm_trial["trial_id"],
            "stage": "confirm",
            "allowed_gpu_indices": partition["experiment"]["gpu_indices"],
            "requested_gpu_count": requested_gpu_count,
            "max_parallel_jobs": min(
                int(partition["experiment"]["max_parallel_jobs"]),
                max(1, len(allowed) // requested_gpu_count),
            ),
            "distributed_training": requested_gpu_count > 1,
            "scale_rationale": normalized_rationale,
        },
        "reallocation_policy": screen_receipt["reallocation_policy"],
        "authority": _authority(),
        "artifact_policy": _artifact_policy(),
        "claim_boundary": (
            "This receipt reallocates an accepted screen to confirmation. Multi-GPU "
            "execution is admitted only from a ready training-scale plan and an explicit "
            "information-gain rationale; the frozen verifier retains claim authority."
        ),
    }
    return _finalize(body, root=root)


def validate_resource_portfolio_receipt(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate resource identity, role isolation, and phase-specific authority."""

    try:
        validate_document("resource_portfolio_receipt", document, root=root)
    except ContractValidationError as exc:
        raise ResourcePortfolioError(f"RESOURCE_PORTFOLIO_SCHEMA_INVALID:{exc}") from exc
    body = dict(document)
    received_digest = body.pop("receipt_digest", None)
    if received_digest != _digest(body):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_DIGEST_MISMATCH")
    identity = dict(body)
    received_id = identity.pop("receipt_id", None)
    if received_id != _stable_id("resource-portfolio", identity):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ID_MISMATCH")
    partition = document["role_partition"]
    experiment = partition["experiment"]
    preparation = partition["data_preparation"]
    experiment_gpus = _indices(experiment["gpu_indices"])
    preparation_gpus = _indices(preparation["gpu_indices"])
    inventory = _indices(partition["inventory_gpu_indices"])
    _validate_current_ctrl_world_partition(
        inventory=inventory,
        experiment_gpus=experiment_gpus,
        preparation_gpus=preparation_gpus,
    )
    allocation = document["allocation"]
    if _indices(allocation["allowed_gpu_indices"]) != experiment_gpus:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ALLOWED_GPUS_MISMATCH")
    phase = document["phase"]
    bindings = document["bindings"]
    if phase == "screen_admission":
        if document["state"] != "ready_for_screen" or allocation["stage"] != "screen":
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCREEN_STATE_INVALID")
        if any(
            bindings[key] is not None
            for key in (
                "parent_receipt_id",
                "parent_receipt_digest",
                "screen_evidence_digest",
                "training_scale_plan_digest",
            )
        ):
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCREEN_BINDING_INVALID")
        if allocation["requested_gpu_count"] != 1 or allocation["distributed_training"] is not False:
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCREEN_DISTRIBUTED_FORBIDDEN")
    else:
        if document["state"] != "ready_for_confirm" or allocation["stage"] != "confirm":
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_CONFIRM_STATE_INVALID")
        if any(
            bindings[key] is None
            for key in (
                "parent_receipt_id",
                "parent_receipt_digest",
                "screen_evidence_digest",
            )
        ):
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_CONFIRM_BINDING_INVALID")
        distributed = int(allocation["requested_gpu_count"]) > 1
        if allocation["distributed_training"] is not distributed:
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_DISTRIBUTED_FLAG_INVALID")
        if distributed != (bindings["training_scale_plan_digest"] is not None):
            raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_BINDING_INVALID")


def load_resource_portfolio_receipt(
    path: Path,
    *,
    expected_sha256: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Load an exact resource receipt from local execution state."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_RECEIPT_INVALID")
    payload = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_FILE_HASH_MISMATCH")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_RECEIPT_INVALID") from exc
    if not isinstance(document, dict):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_RECEIPT_INVALID")
    validate_resource_portfolio_receipt(document, root=root)
    return document


def _validate_portfolio(portfolio: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_experiment_portfolio(portfolio, root=root)
    except ExperimentPortfolioError as exc:
        raise ResourcePortfolioError(f"RESOURCE_PORTFOLIO_EXPERIMENT_INVALID:{exc}") from exc
    if portfolio.get("state") != "ready_for_resource_admission":
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_EXPERIMENT_NOT_READY")


def _role_partition(
    allocation: Mapping[str, object],
    *,
    experiment_scale_plan: Mapping[str, object],
    conversion_scale_plan: Mapping[str, object],
) -> dict[str, object]:
    body = dict(allocation)
    allocation_id = body.pop("allocation_id", None)
    if allocation_id != _digest(body):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ALLOCATION_DIGEST_MISMATCH")
    if allocation.get("artifact_type") != "verdiwm-gpu-resource-allocation" or allocation.get("state") != "ready":
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ALLOCATION_INVALID")
    roles = allocation.get("roles")
    if not isinstance(roles, list):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ROLES_INVALID")
    experiment = _one_role(roles, "autonomous_candidate_evaluation")
    preparation = _one_role(roles, "droid_data_preparation")
    inventory = _indices(allocation.get("gpu_indices"))
    experiment_gpus = _indices(experiment.get("gpu_indices"))
    preparation_gpus = _indices(preparation.get("gpu_indices"))
    _validate_current_ctrl_world_partition(
        inventory=inventory,
        experiment_gpus=experiment_gpus,
        preparation_gpus=preparation_gpus,
    )
    if int(experiment.get("max_parallel_jobs", -1)) > len(experiment_gpus):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_EXPERIMENT_PARALLELISM_INVALID")
    if int(preparation.get("max_parallel_jobs", -1)) != 2:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_CONVERSION_PARALLELISM_INVALID")
    _validate_scale_plan(
        experiment_scale_plan,
        artifact_type="verdiwm-autonomous-controller-scale-plan",
        gpu_indices=experiment_gpus,
        max_parallel_jobs=int(experiment["max_parallel_jobs"]),
    )
    _validate_scale_plan(
        conversion_scale_plan,
        artifact_type="verdiwm-droid-conversion-scale-plan",
        gpu_indices=preparation_gpus,
        max_parallel_jobs=2,
    )
    return {
        "allocation_id": allocation_id,
        "inventory_gpu_indices": inventory,
        "experiment": {
            **dict(experiment),
            "scale_plan_digest": _digest(experiment_scale_plan),
        },
        "data_preparation": {
            **dict(preparation),
            "workload_id": "droid-ctrl-world-conversion-v1",
            "scale_plan_digest": _digest(conversion_scale_plan),
        },
    }


def _validate_scale_plan(
    plan: Mapping[str, object],
    *,
    artifact_type: str,
    gpu_indices: Sequence[int],
    max_parallel_jobs: int,
) -> None:
    if plan.get("schema_version") != 1 or plan.get("artifact_type") != artifact_type:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_PLAN_INVALID")
    if _indices(plan.get("physical_gpu_indices")) != list(gpu_indices):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_PLAN_GPU_MISMATCH")
    if int(plan.get("max_parallel_gpu_jobs", -1)) != max_parallel_jobs:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_PLAN_PARALLELISM_MISMATCH")


def _screen_trials(portfolio: Mapping[str, object]) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for entry in portfolio["entries"]:
        seeds = entry["replication"]["seeds"]
        for index, seed in enumerate(seeds):
            identity = {
                "portfolio_id": portfolio["portfolio_id"],
                "entry_id": entry["entry_id"],
                "replication_index": index,
                "seed": seed,
                "stage": "screen",
            }
            trials.append(
                {
                    "trial_id": _stable_id("resource-trial", identity),
                    "entry_id": entry["entry_id"],
                    "candidate_id": entry["candidate_id"],
                    "mechanism_id": entry["mechanism_id"],
                    "role": entry["role"],
                    "replication_index": index,
                    "seed": seed,
                    "stage": "screen",
                    "state": "conditional",
                    "estimated_gpu_hours": entry["cost"]["per_replication_gpu_hours"],
                    "dependencies": list(entry["dependencies"]),
                }
            )
    return sorted(
        trials,
        key=lambda row: (
            str(row["entry_id"]),
            int(row["replication_index"]),
            str(row["trial_id"]),
        ),
    )


def _selected_screen_trial(
    trials: Sequence[Mapping[str, object]], *, portfolio_entry_ids: Sequence[str]
) -> Mapping[str, object]:
    bound = set(portfolio_entry_ids)
    candidates = [
        row
        for row in trials
        if row.get("role") == "mechanism_test"
        and int(row["replication_index"]) == 0
        and (not bound or row.get("entry_id") in bound)
    ]
    if len(candidates) != 1:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_EXECUTION_ENTRY_AMBIGUOUS")
    return candidates[0]


def _validate_screen_evidence(
    screen_receipt: Mapping[str, object], screen_evidence: Mapping[str, object]
) -> str:
    if (
        screen_evidence.get("schema_version") != 1
        or screen_evidence.get("artifact_type") != "verdiwm-resource-screen-evidence"
        or screen_evidence.get("state") != "settled"
        or screen_evidence.get("decision") != "accepted"
        or screen_evidence.get("work_id") != screen_receipt.get("work_id")
        or screen_evidence.get("trial_id")
        != screen_receipt["allocation"]["selected_trial_id"]
    ):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCREEN_EVIDENCE_INVALID")
    evidence_ref = screen_evidence.get("evidence_ref")
    if not isinstance(evidence_ref, str) or not (
        evidence_ref.startswith("sha256:") or evidence_ref.startswith("cas://sha256/")
    ):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCREEN_EVIDENCE_REF_INVALID")
    return _digest(screen_evidence)


def _distributed_scale_binding(
    *,
    requested_gpu_count: int,
    training_scale_plan: Mapping[str, object] | None,
    scale_rationale: Mapping[str, object] | None,
) -> tuple[str, dict[str, object]]:
    if training_scale_plan is None or scale_rationale is None:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_DISTRIBUTED_SCALE_REQUIRED")
    if (
        training_scale_plan.get("artifact_type") != "verdiwm-training-scale-plan"
        or training_scale_plan.get("state") != "ready"
        or training_scale_plan.get("stage") != "confirm"
        or training_scale_plan.get("parallelism", {}).get("world_size")
        != requested_gpu_count
    ):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_DISTRIBUTED_SCALE_INVALID")
    required = {
        "decision",
        "expected_information_gain",
        "independent_screen_opportunity_cost",
        "reason",
    }
    if set(scale_rationale) != required:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_RATIONALE_INVALID")
    if scale_rationale.get("decision") != "distributed_confirm":
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_RATIONALE_INVALID")
    gain = scale_rationale.get("expected_information_gain")
    opportunity = scale_rationale.get("independent_screen_opportunity_cost")
    reason = scale_rationale.get("reason")
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or isinstance(opportunity, bool)
        or not isinstance(opportunity, (int, float))
        or float(gain) <= float(opportunity)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_SCALE_RATIONALE_INVALID")
    return _digest(training_scale_plan), {
        "decision": "distributed_confirm",
        "expected_information_gain": float(gain),
        "independent_screen_opportunity_cost": float(opportunity),
        "reason": reason.strip(),
        "world_size": requested_gpu_count,
        "planned_steps": training_scale_plan["updates"]["planned_steps"],
        "effective_batch_size": training_scale_plan["parallelism"]["effective_batch_size"],
    }


def _one_role(roles: Sequence[object], name: str) -> Mapping[str, object]:
    matches = [row for row in roles if isinstance(row, Mapping) and row.get("role") == name]
    if len(matches) != 1:
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_ROLE_INVALID:" + name)
    return matches[0]


def _indices(value: object) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_GPU_INDICES_INVALID")
    if len(value) != len(set(value)):
        raise ResourcePortfolioError("RESOURCE_PORTFOLIO_GPU_INDICES_INVALID")
    return list(value)


def _validate_current_ctrl_world_partition(
    *,
    inventory: Sequence[int],
    experiment_gpus: Sequence[int],
    preparation_gpus: Sequence[int],
) -> None:
    """Keep the v1 campaign's DROID reservation outside experiment admission."""

    if (
        list(inventory) != _CTRL_WORLD_GPU_INVENTORY
        or list(experiment_gpus) != _CTRL_WORLD_EXPERIMENT_GPUS
        or list(preparation_gpus) != _CTRL_WORLD_DROID_GPUS
    ):
        raise ResourcePortfolioError(
            "RESOURCE_PORTFOLIO_CTRL_WORLD_GPU_ROLE_ASSIGNMENT_INVALID"
        )


def _reallocation_policy(policy_id: str) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "screens_before_confirms": True,
        "require_screen_acceptance": True,
        "distributed_requires_ready_scale_plan": True,
        "distributed_requires_information_gain_rationale": True,
    }


def _authority() -> dict[str, object]:
    return {
        "gpu_lease_request": True,
        "evaluator_mutation": False,
        "metric_mutation": False,
        "promotion": False,
    }


def _artifact_policy() -> dict[str, object]:
    return {
        "archive_policy": "archive_all_terminal",
        "cleanup_policy": "after_content_addressed_receipt",
        "retain_failed_terminal_evidence": True,
    }


def _finalize(body: Mapping[str, object], *, root: Path | None) -> dict[str, object]:
    receipt: dict[str, object] = dict(body)
    receipt["receipt_id"] = _stable_id("resource-portfolio", receipt)
    receipt["receipt_digest"] = _digest(receipt)
    validate_resource_portfolio_receipt(receipt, root=root)
    return receipt


def _require_digest(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResourcePortfolioError(code)
    return value


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_digest(value)[:24]}"


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
