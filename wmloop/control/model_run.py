"""Compile a profile-backed model training run without launching it.

The model source remains an external, read-only dependency.  This module is
the single point where a training-scale receipt, GPU allocation, frozen
evaluator, and fresh output root become one durable run contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.agibot_native_config import (
    AgibotNativeConfigError,
    bind_agibot_native_config,
    validate_agibot_native_config,
)
from wmloop.execute.model_run_budget import (
    ModelRunBudgetError,
    compile_model_run_budget_binding,
    estimate_model_run_gpu_hours,
    validate_model_run_budget_binding,
    validate_model_run_budget_limit,
)
from wmloop.experiments.training_scale import (
    TrainingScaleError,
    validate_training_scale_checkpoint_schedule,
)


class ModelRunError(ValueError):
    """A model-run manifest cannot be safely compiled."""


_MANIFEST_TOKEN = "{model_run_manifest}"
_SHELL_META = re.compile(r"[;&|`\n\r]")
_ASSET_TOKEN = re.compile(r"^\{asset:(?P<parameter>--[A-Za-z0-9_-]+)\}$")


def compile_model_run(
    *,
    campaign_id: str,
    execution: Mapping[str, object],
    model: Path,
    data: Path,
    training_scale_plan: Mapping[str, object],
    resource_allocation: Mapping[str, object],
    root: Path | None = None,
) -> dict[str, object]:
    """Bind an adapter profile to a receipt-first, non-executing train run."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ModelRunError("MODEL_RUN_CAMPAIGN_ID_INVALID")
    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    adapter = _mapping(execution.get("model_run_adapter"), "MODEL_RUN_ADAPTER_REQUIRED")
    runner = _mapping(adapter.get("runner"), "MODEL_RUN_RUNNER_REQUIRED")
    if runner.get("requires_training") is not True:
        raise ModelRunError("MODEL_RUN_TRAINING_RUNNER_REQUIRED")
    protocol_version = runner.get("protocol_version")
    if protocol_version not in {1, 2, 3} or isinstance(protocol_version, bool):
        raise ModelRunError("MODEL_RUN_PROTOCOL_VERSION_INVALID")
    _validate_scale_plan(training_scale_plan, root=base)
    _validate_allocation(
        resource_allocation, world_size=_world_size(training_scale_plan)
    )

    model_root = _directory(model, "MODEL_RUN_MODEL_INVALID")
    dataset_root = _existing(data, "MODEL_RUN_DATASET_INVALID")
    source_root = _directory(
        Path(_text(execution.get("source_root") or execution.get("repo_root") or model_root, "MODEL_RUN_SOURCE_INVALID")),
        "MODEL_RUN_SOURCE_INVALID",
    )
    output_root = _output_root(execution, model_root, dataset_root, base)
    manifest_path = output_root / "model-run.json"
    evaluator = _file(
        execution.get("evaluator_contract"), "MODEL_RUN_EVALUATOR_INVALID"
    )
    runtime = _executable(execution.get("runtime_python"), "MODEL_RUN_RUNTIME_INVALID")
    assets = _asset_bindings(execution.get("asset_bindings"))
    selected_vae = execution.get("agibot_vae_checkpoint")
    if selected_vae is not None:
        if not isinstance(selected_vae, (str, Path)) or not selected_vae:
            raise ModelRunError("MODEL_RUN_AGIBOT_VAE_CHECKPOINT_INVALID")
        assets["--vae-checkpoint"] = str(
            _existing(
                Path(selected_vae), "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_INVALID"
            )
        )
    adapter_root = _optional_adapter_root(
        adapter.get("adapter_root"), output_root, model_root, dataset_root, base
    )
    train_command = _bind_command(
        runner.get("train"),
        model_root=model_root,
        source_root=source_root,
        adapter_root=adapter_root,
        manifest_path=manifest_path,
        root=base,
        code="MODEL_RUN_TRAIN_COMMAND_INVALID",
        native_config_path=(
            output_root / "config" / "agibot-training.yaml"
            if protocol_version == 3
            else None
        ),
        allow_native_config_placeholder=protocol_version == 3,
        asset_bindings=assets,
    )
    evaluate_command = _bind_command(
        runner.get("evaluate"),
        model_root=model_root,
        source_root=source_root,
        adapter_root=adapter_root,
        manifest_path=manifest_path,
        root=base,
        code="MODEL_RUN_EVALUATE_COMMAND_INVALID",
        allow_checkpoint_placeholder=protocol_version in {2, 3},
    )
    checkpoint_receipt = _output_receipt(
        output_root,
        runner.get("checkpoint_receipt"),
        "MODEL_RUN_CHECKPOINT_RECEIPT_INVALID",
    )
    evidence_receipt = _output_receipt(
        output_root,
        runner.get("evaluation_receipt"),
        "MODEL_RUN_EVIDENCE_RECEIPT_INVALID",
    )
    stage_timeouts: dict[str, float] | None = None
    estimated_gpu_hours: float | None = None
    agibot_support: dict[str, object] | None = None
    agibot_native_config: dict[str, object] | None = None
    if protocol_version == 3:
        stage_timeouts = _protocol_v3_stage_timeouts(
            execution.get("stage_timeouts_seconds")
        )
        try:
            estimated_gpu_hours = estimate_model_run_gpu_hours(
                stage_timeouts, _world_size(training_scale_plan)
            )
            validate_model_run_budget_limit(
                estimated_gpu_hours, execution.get("budget_total_gpu_hours")
            )
        except ModelRunBudgetError as exc:
            raise ModelRunError(str(exc)) from exc
        try:
            from wmloop.execute.agibot_support import (
                AgibotSupportError,
                agibot_receipt_bindings,
                build_agibot_training_support,
                validate_agibot_training_support,
            )

            raw_support_value = execution.get("agibot_training_support")
            if raw_support_value is None:
                loader = _mapping(
                    execution.get("agibot_support_loader"),
                    "MODEL_RUN_AGIBOT_LOADER_REQUIRED",
                )
                raw_support = build_agibot_training_support(
                    dataset_root,
                    source_root,
                    loader=loader,
                    validation_sample_count=execution.get(
                        "agibot_validation_sample_count", 128
                    ),  # type: ignore[arg-type]
                )
            else:
                raw_support = _mapping(
                    raw_support_value, "MODEL_RUN_AGIBOT_SUPPORT_REQUIRED"
                )
            validate_agibot_training_support(
                raw_support,
                expected_dataset_root=dataset_root,
                expected_native_source_root=source_root,
            )
        except AgibotSupportError as exc:
            raise ModelRunError(f"MODEL_RUN_AGIBOT_SUPPORT_INVALID:{exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ModelRunError(f"MODEL_RUN_AGIBOT_SUPPORT_INVALID:{exc}") from exc
        agibot_support = dict(raw_support)
        agibot_bindings = agibot_receipt_bindings(raw_support)
        try:
            agibot_native_config = bind_agibot_native_config(
                execution.get("agibot_native_config"), training_scale_plan,
                output_root=output_root,
                dataset_root=dataset_root,
                asset_bindings=assets,
            )
        except AgibotNativeConfigError as exc:
            raise ModelRunError(str(exc)) from exc
        if isinstance(execution.get("agibot_native_config"), (str, Path)):
            if agibot_native_config.get("source_path") != str(Path(str(execution["agibot_native_config"])).expanduser().resolve()):
                raise ModelRunError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_SOURCE_INVALID")
    else:
        agibot_bindings = None

    isolation: dict[str, object] = {
        "model_source_read_only": True,
        "dataset_read_only": True,
        "evaluator_read_only": True,
        "output_root_fresh": True,
        "network_access": False,
    }
    if protocol_version == 3:
        # Protocol v3 disables known telemetry clients and runs in campaign
        # scratch, but native code still executes as a host process. Describe
        # its offline requirement and best-effort policy without claiming an
        # OS-level network or filesystem sandbox.
        isolation.pop("network_access")
        isolation.update(
            {
                "network_required": False,
                "network_policy": "known_clients_offline_only",
                "os_sandbox_enforced": False,
            }
        )

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-run-manifest",
        "model_run_id": "",
        "campaign_id": campaign_id,
        "manifest_path": str(manifest_path),
        "adapter": {
            "profile_id": _text(adapter.get("profile_id"), "MODEL_RUN_PROFILE_INVALID"),
            "model_family": _text(
                adapter.get("model_family"), "MODEL_RUN_PROFILE_INVALID"
            ),
            "capability_level": _text(
                adapter.get("capability_level"), "MODEL_RUN_PROFILE_INVALID"
            ),
            "protocol_version": protocol_version,
        },
        "runner": {"adapter_id": runner.get("adapter_id")},
        "source": {
            "model_root": str(model_root),
            "dataset_root": str(dataset_root),
            "source_root": str(source_root),
            "asset_bindings": assets,
            **({"adapter_root": str(adapter_root)} if adapter_root is not None else {}),
        },
        "runtime": {"python": str(runtime), "train_command": train_command},
        "training": {
            "scale_plan": dict(training_scale_plan),
            "scale_plan_digest": _digest(training_scale_plan),
            "checkpoint_receipt": str(checkpoint_receipt),
            **(
                {"agibot_training_support": agibot_support}
                if agibot_support is not None
                else {}
            ),
            **(
                {"agibot_receipt_bindings": agibot_bindings}
                if agibot_bindings is not None
                else {}
            ),
            **(
                {"agibot_native_config": agibot_native_config}
                if agibot_native_config is not None
                else {}
            ),
        },
        "evaluation": {
            "evaluator_contract": str(evaluator),
            "command": evaluate_command,
            "evidence_receipt": str(evidence_receipt),
            **(
                {
                    "stage": "engineering_diagnostic",
                    "evidence_class": "runtime_only",
                    "quality_eligible": False,
                }
                if protocol_version == 3
                else {}
            ),
        },
        "resource_binding": {
            "allocation": dict(resource_allocation),
            "role": "autonomous_candidate_evaluation",
            "gpu_lease_required": True,
        },
        "output": {
            "root": str(output_root),
            "checkpoint_receipt": str(checkpoint_receipt),
            "evidence_receipt": str(evidence_receipt),
        },
        "isolation": isolation,
        "execution_policy": {
            "stage_timeout_seconds": max(stage_timeouts.values()) if stage_timeouts else 24.0 * 3600.0,
            **(
                {"stage_timeouts_seconds": stage_timeouts}
                if stage_timeouts is not None
                else {}
            ),
            "timeout_receipt": "execution-timeout-receipt.json",
        },
        "claim_boundary": (
            (
                "This manifest admits a bounded Agibot engineering diagnostic only. "
                "Its runtime-only result is not a model-effect result or formal verdict."
            )
            if protocol_version == 3
            else (
                "This manifest admits a bounded external model run only after a GPU lease. "
                "Checkpoint output is not promotion evidence without the declared held-out receipt."
            )
        ),
    }
    if protocol_version == 3:
        try:
            if estimated_gpu_hours is None:
                raise ModelRunBudgetError("MODEL_RUN_BUDGET_ESTIMATE_INVALID")
            body["budget_binding"] = compile_model_run_budget_binding(
                execution,
                plan_document=body,
                estimated_gpu_hours=estimated_gpu_hours,
            )
        except ModelRunBudgetError as exc:
            raise ModelRunError(str(exc)) from exc
    body["model_run_id"] = "model-run-" + model_run_digest(body)[:24]
    validate_model_run(body, root=base)
    return body


def validate_model_run(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate the persisted manifest and its non-overlap invariants."""

    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    try:
        validate_document("model_run_manifest", document, root=base)
    except ContractValidationError as exc:
        raise ModelRunError(f"MODEL_RUN_SCHEMA_INVALID:{exc}") from exc
    expected = "model-run-" + model_run_digest(document)[:24]
    if document.get("model_run_id") != expected:
        raise ModelRunError("MODEL_RUN_DIGEST_MISMATCH")
    output = _directory(Path(_text(document["output"]["root"], "MODEL_RUN_OUTPUT_INVALID")), "MODEL_RUN_OUTPUT_INVALID", require_exists=False)  # type: ignore[index]
    source = document["source"]
    model_root = _directory(Path(_text(source["model_root"], "MODEL_RUN_MODEL_INVALID")), "MODEL_RUN_MODEL_INVALID")  # type: ignore[index]
    dataset_root = _existing(Path(_text(source["dataset_root"], "MODEL_RUN_DATASET_INVALID")), "MODEL_RUN_DATASET_INVALID")  # type: ignore[index]
    source_root = _directory(
        Path(_text(source.get("source_root") or source.get("model_root"), "MODEL_RUN_SOURCE_INVALID")),
        "MODEL_RUN_SOURCE_INVALID",
    )
    _require_disjoint(output, model_root, "MODEL_RUN_OUTPUT_OVERLAPS_MODEL")
    _require_disjoint(output, dataset_root, "MODEL_RUN_OUTPUT_OVERLAPS_DATA")
    _require_disjoint(output, source_root, "MODEL_RUN_OUTPUT_OVERLAPS_SOURCE")
    adapter_root = _optional_adapter_root(
        source.get("adapter_root"), output, model_root, dataset_root, base  # type: ignore[union-attr]
    )
    if adapter_root is not None:
        for command in (
            document["runtime"]["train_command"],  # type: ignore[index]
            document["evaluation"]["command"],  # type: ignore[index]
        ):
            entrypoint = Path(_text(command[0], "MODEL_RUN_ADAPTER_COMMAND_INVALID"))
            if not _within(entrypoint.resolve(), adapter_root):
                raise ModelRunError("MODEL_RUN_ADAPTER_COMMAND_OUTSIDE_ROOT")
    manifest = Path(_text(document["manifest_path"], "MODEL_RUN_MANIFEST_PATH_INVALID")).resolve()  # type: ignore[index]
    if not _within(manifest, output):
        raise ModelRunError("MODEL_RUN_MANIFEST_OUTSIDE_OUTPUT")
    _validate_scale_plan(_mapping(document["training"]["scale_plan"], "MODEL_RUN_SCALE_PLAN_INVALID"), root=root)  # type: ignore[index]
    protocol_version = document["adapter"].get("protocol_version")  # type: ignore[index]
    if protocol_version == 2:
        _validate_protocol_v2(document)
    elif protocol_version == 3:
        _validate_protocol_v3(document)
        try:
            validate_model_run_budget_binding(document)
            _validate_protocol_v3_budget(document)
        except ModelRunBudgetError as exc:
            raise ModelRunError(str(exc)) from exc
    if document["training"].get("scale_plan_digest") != _digest(document["training"]["scale_plan"]):  # type: ignore[index]
        raise ModelRunError("MODEL_RUN_SCALE_PLAN_DIGEST_MISMATCH")
    _validate_allocation(
        _mapping(document["resource_binding"]["allocation"], "MODEL_RUN_RESOURCE_ALLOCATION_INVALID"),  # type: ignore[index]
        world_size=_world_size(document["training"]["scale_plan"]),  # type: ignore[index]
    )
    execution_policy = document.get("execution_policy")
    if execution_policy is not None:
        if not isinstance(execution_policy, Mapping):
            raise ModelRunError("MODEL_RUN_EXECUTION_POLICY_INVALID")
        timeout = execution_policy.get("stage_timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ModelRunError("MODEL_RUN_TIMEOUT_INVALID")
        receipt_name = execution_policy.get("timeout_receipt")
        if (
            not isinstance(receipt_name, str)
            or not receipt_name
            or Path(receipt_name).is_absolute()
            or ".." in Path(receipt_name).parts
        ):
            raise ModelRunError("MODEL_RUN_TIMEOUT_RECEIPT_INVALID")


def model_run_digest(document: Mapping[str, object]) -> str:
    """Return the immutable identity digest excluding the derived run id."""

    payload = {key: value for key, value in document.items() if key != "model_run_id"}
    return _digest(payload)


def _validate_scale_plan(plan: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_document("training_scale_plan", plan, root=root)
    except ContractValidationError as exc:
        raise ModelRunError("MODEL_RUN_SCALE_PLAN_INVALID") from exc
    if plan.get("state") != "ready":
        raise ModelRunError("MODEL_RUN_SCALE_PLAN_NOT_READY")
    try:
        validate_training_scale_checkpoint_schedule(plan)
    except TrainingScaleError as exc:
        raise ModelRunError(
            "MODEL_RUN_SCALE_PLAN_CHECKPOINT_SCHEDULE_INVALID"
        ) from exc
    stopping = _mapping(plan.get("stopping_policy"), "MODEL_RUN_SCALE_PLAN_INVALID")
    if stopping.get("evaluate_each_checkpoint_on_heldout") is not True:
        raise ModelRunError("MODEL_RUN_HELDOUT_EVALUATION_REQUIRED")
    if plan.get("schema_version") == 3:
        dataset = _mapping(plan.get("dataset"), "MODEL_RUN_SCALE_PLAN_INVALID")
        replication = _mapping(plan.get("replication"), "MODEL_RUN_SCALE_PLAN_INVALID")
        native_agibot = dataset.get("input_format") == "agibot_native_index_jsonl"
        record_ids = dataset.get("selected_train_record_ids")
        record_binding = dataset.get("selected_train_record_ids_binding")
        episode_ids = dataset.get("selected_train_episode_ids")
        episode_binding = dataset.get("selected_train_episode_ids_binding")
        seeds = replication.get("seeds")
        seed_count = replication.get("requested_seed_count")
        record_count_valid = (
            _valid_compact_identity_binding(
                record_binding, expected_count=dataset.get("selected_train_examples")
            )
            if native_agibot
            else isinstance(record_ids, list)
            and bool(record_ids)
            and len(set(record_ids)) == len(record_ids)
            and len(record_ids) == dataset.get("selected_train_examples")
        )
        episode_count_valid = (
            _valid_compact_identity_binding(
                episode_binding,
                expected_count=dataset.get("selected_train_episode_count"),
            )
            if native_agibot
            else isinstance(episode_ids, list)
            and len(episode_ids) == dataset.get("selected_train_episode_count")
        )
        if (
            not record_count_valid
            or not episode_count_valid
            or not isinstance(seeds, list)
            or isinstance(seed_count, bool)
            or not isinstance(seed_count, int)
            or len(seeds) != seed_count
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ModelRunError("MODEL_RUN_SCALE_PLAN_BINDING_INVALID")


def _validate_protocol_v2(document: Mapping[str, object]) -> None:
    training = _mapping(document.get("training"), "MODEL_RUN_TRAINING_INVALID")
    plan = _mapping(training.get("scale_plan"), "MODEL_RUN_SCALE_PLAN_INVALID")
    if plan.get("schema_version") != 3:
        raise ModelRunError("MODEL_RUN_PROTOCOL_V2_SCALE_PLAN_REQUIRED")
    replication = _mapping(plan.get("replication"), "MODEL_RUN_SCALE_PLAN_INVALID")
    seeds = replication.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise ModelRunError("MODEL_RUN_PROTOCOL_V2_SINGLE_SEED_REQUIRED")
    evaluation = _mapping(document.get("evaluation"), "MODEL_RUN_EVALUATION_INVALID")
    command = evaluation.get("command")
    if (
        not isinstance(command, list)
        or command.count("{trained_checkpoint}") != 1
    ):
        raise ModelRunError("MODEL_RUN_TRAINED_CHECKPOINT_PLACEHOLDER_REQUIRED")


def _validate_protocol_v3(document: Mapping[str, object]) -> None:
    training = _mapping(document.get("training"), "MODEL_RUN_TRAINING_INVALID")
    plan = _mapping(training.get("scale_plan"), "MODEL_RUN_SCALE_PLAN_INVALID")
    if plan.get("schema_version") != 3:
        raise ModelRunError("MODEL_RUN_PROTOCOL_V3_SCALE_PLAN_REQUIRED")
    dataset = _mapping(plan.get("dataset"), "MODEL_RUN_SCALE_PLAN_INVALID")
    updates = _mapping(plan.get("updates"), "MODEL_RUN_SCALE_PLAN_INVALID")
    replication = _mapping(plan.get("replication"), "MODEL_RUN_SCALE_PLAN_INVALID")
    seeds = replication.get("seeds")
    if (
        plan.get("stage") != "engineering_diagnostic"
        or plan.get("training_mode") != "probe"
        or plan.get("evidence_class") != "runtime_only"
        or plan.get("quality_eligible") is not False
        or dataset.get("selection_fraction") != 1.0
        or updates.get("planned_steps") is None
        or isinstance(updates.get("planned_steps"), bool)
        or not isinstance(updates.get("planned_steps"), int)
        or updates["planned_steps"] > 8
        or replication.get("requested_seed_count") != 1
        or not isinstance(seeds, list)
        or len(seeds) != 1
    ):
        raise ModelRunError("MODEL_RUN_PROTOCOL_V3_ENGINEERING_DIAGNOSTIC_POLICY_INVALID")
    support = _mapping(
        training.get("agibot_training_support"),
        "MODEL_RUN_AGIBOT_SUPPORT_REQUIRED",
    )
    source = _mapping(document.get("source"), "MODEL_RUN_SOURCE_INVALID")
    if dataset.get("input_format") != "agibot_native_index_jsonl":
        raise ModelRunError("MODEL_RUN_PROTOCOL_V3_NATIVE_AGIBOT_PLAN_REQUIRED")
    try:
        from wmloop.execute.agibot_support import (
            AgibotSupportError,
            agibot_receipt_bindings,
            validate_agibot_training_support,
        )

        validate_agibot_training_support(
            support,
            expected_dataset_root=Path(str(source["dataset_root"])),
            expected_native_source_root=Path(str(source["source_root"])),
        )
    except AgibotSupportError as exc:
        raise ModelRunError(f"MODEL_RUN_AGIBOT_SUPPORT_INVALID:{exc}") from exc
    if training.get("agibot_receipt_bindings") != agibot_receipt_bindings(support):
        raise ModelRunError("MODEL_RUN_AGIBOT_RECEIPT_BINDINGS_MISMATCH")
    _validate_agibot_plan_support_binding(dataset, support)
    try:
        validate_agibot_native_config(
            _mapping(training.get("agibot_native_config"), "MODEL_RUN_AGIBOT_NATIVE_CONFIG_REQUIRED"),
            plan,
            output_root=Path(str(_mapping(document.get("output"), "MODEL_RUN_OUTPUT_INVALID")["root"])),
            dataset_root=Path(str(source["dataset_root"])),
            asset_bindings=_mapping(source.get("asset_bindings"), "MODEL_RUN_ASSET_BINDINGS_INVALID"),
        )
    except AgibotNativeConfigError as exc:
        raise ModelRunError(str(exc)) from exc
    evaluation = _mapping(document.get("evaluation"), "MODEL_RUN_EVALUATION_INVALID")
    runtime = _mapping(document.get("runtime"), "MODEL_RUN_RUNTIME_INVALID")
    train_command = runtime.get("train_command")
    source_root = Path(str(source.get("source_root", ""))).resolve()
    output = Path(str(_mapping(document.get("output"), "MODEL_RUN_OUTPUT_INVALID")["root"])).resolve()
    expected_native_script = source_root / "scripts" / "training" / "train_world_model.py"
    expected_native_config = output / "config" / "agibot-training.yaml"
    if (
        not isinstance(train_command, list)
        or len(train_command) != 5
        or Path(str(train_command[0])).resolve() != expected_native_script
        or train_command[1] != "--config"
        or train_command[2] != str(expected_native_config)
        or train_command[3] != "--init_from"
        or train_command[4] != source.get("asset_bindings", {}).get("--init_from")
    ):
        raise ModelRunError("MODEL_RUN_AGIBOT_NATIVE_TRAIN_COMMAND_INVALID")
    command = evaluation.get("command")
    if (
        evaluation.get("stage") != "engineering_diagnostic"
        or evaluation.get("evidence_class") != "runtime_only"
        or evaluation.get("quality_eligible") is not False
        or not isinstance(command, list)
        or command.count("{trained_checkpoint}") != 1
    ):
        raise ModelRunError("MODEL_RUN_ENGINEERING_DIAGNOSTIC_BINDING_INVALID")
    policy = _mapping(document.get("execution_policy"), "MODEL_RUN_EXECUTION_POLICY_INVALID")
    _protocol_v3_stage_timeouts(policy.get("stage_timeouts_seconds"))


def _protocol_v3_stage_timeouts(value: object) -> dict[str, float]:
    if value is None:
        return {"train": 24.0 * 3600.0, "engineering_diagnostic": 3600.0}
    if not isinstance(value, Mapping) or set(value) != {"train", "engineering_diagnostic"}:
        raise ModelRunError("MODEL_RUN_STAGE_TIMEOUTS_INVALID")
    result: dict[str, float] = {}
    for stage, raw_timeout in value.items():
        if (
            isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or not math.isfinite(float(raw_timeout))
            or float(raw_timeout) <= 0
        ):
            raise ModelRunError("MODEL_RUN_STAGE_TIMEOUTS_INVALID")
        result[str(stage)] = float(raw_timeout)
    return result


def _validate_protocol_v3_budget(document: Mapping[str, object]) -> None:
    """Recompute the runtime budget from the immutable execution plan."""

    execution_policy = _mapping(
        document.get("execution_policy"), "MODEL_RUN_EXECUTION_POLICY_INVALID"
    )
    stage_timeouts = _protocol_v3_stage_timeouts(
        execution_policy.get("stage_timeouts_seconds")
    )
    training = _mapping(document.get("training"), "MODEL_RUN_TRAINING_INVALID")
    scale_plan = _mapping(training.get("scale_plan"), "MODEL_RUN_SCALE_PLAN_INVALID")
    world_size = _world_size(scale_plan)
    resource_binding = _mapping(
        document.get("resource_binding"), "MODEL_RUN_RESOURCE_BINDING_INVALID"
    )
    allocation = _mapping(
        resource_binding.get("allocation"), "MODEL_RUN_RESOURCE_ALLOCATION_INVALID"
    )
    _validate_allocation(allocation, world_size=world_size)

    binding = _mapping(
        document.get("budget_binding"), "MODEL_RUN_BUDGET_BINDING_REQUIRED"
    )
    policy = _mapping(
        binding.get("policy"), "MODEL_RUN_BUDGET_POLICY_INVALID"
    )
    expected_estimate = estimate_model_run_gpu_hours(stage_timeouts, world_size)
    validate_model_run_budget_limit(
        expected_estimate, policy.get("total_gpu_hours")
    )
    if binding.get("estimated_gpu_hours") != expected_estimate:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_ESTIMATE_MISMATCH")


def _valid_compact_identity_binding(value: object, *, expected_count: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    count = value.get("count")
    digest = value.get("sha256")
    return (
        not isinstance(count, bool)
        and isinstance(count, int)
        and count > 0
        and count == expected_count
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    )


def _validate_agibot_plan_support_binding(
    dataset: Mapping[str, object], support: Mapping[str, object]
) -> None:
    inputs = _mapping(support.get("inputs"), "MODEL_RUN_AGIBOT_SUPPORT_INVALID")
    train_support = _mapping(
        support.get("train_support"), "MODEL_RUN_AGIBOT_SUPPORT_INVALID"
    )
    validation_support = _mapping(
        support.get("validation_support"), "MODEL_RUN_AGIBOT_SUPPORT_INVALID"
    )
    manifest_paths = _mapping(
        dataset.get("manifest_paths"), "MODEL_RUN_SCALE_PLAN_BINDING_INVALID"
    )
    support_root = Path(str(support.get("dataset_root", ""))).resolve()
    expected_train = support_root / "train.index.jsonl"
    expected_validation = support_root / "val.index.jsonl"
    try:
        plan_train = Path(str(manifest_paths.get("train", ""))).resolve(strict=True)
        plan_validation = Path(
            str(manifest_paths.get("validation", ""))
        ).resolve(strict=True)
    except OSError as exc:
        raise ModelRunError("MODEL_RUN_SCALE_PLAN_BINDING_INVALID") from exc
    train_binding = dataset.get("selected_train_episode_ids_binding")
    if (
        inputs.get("train_index") != dataset.get("train_manifest_sha256")
        or inputs.get("validation_index") != dataset.get("validation_manifest_sha256")
        or dataset.get("train_examples") != train_support.get("episode_count")
        or dataset.get("selected_train_examples") != train_support.get("episode_count")
        or dataset.get("train_episode_count") != train_support.get("episode_count")
        or dataset.get("validation_examples") != validation_support.get("episode_count")
        or dataset.get("validation_episode_count") != validation_support.get("episode_count")
        or not _valid_compact_identity_binding(
            train_binding, expected_count=train_support.get("episode_count")
        )
        or not isinstance(train_binding, Mapping)
        or train_binding.get("sha256") != train_support.get("episode_uids_sha256")
        or plan_train != expected_train.resolve()
        or plan_validation != expected_validation.resolve()
    ):
        raise ModelRunError("MODEL_RUN_AGIBOT_PLAN_SUPPORT_BINDING_MISMATCH")


def _validate_allocation(allocation: Mapping[str, object], *, world_size: int) -> None:
    if (
        allocation.get("artifact_type") != "verdiwm-gpu-resource-allocation"
        or allocation.get("state") != "ready"
    ):
        raise ModelRunError("MODEL_RUN_RESOURCE_ALLOCATION_INVALID")
    roles = allocation.get("roles")
    if not isinstance(roles, list):
        raise ModelRunError("MODEL_RUN_RESOURCE_ALLOCATION_INVALID")
    matches = [
        row
        for row in roles
        if isinstance(row, Mapping)
        and row.get("role") == "autonomous_candidate_evaluation"
    ]
    if len(matches) != 1:
        raise ModelRunError("MODEL_RUN_RESOURCE_ROLE_INVALID")
    indices = matches[0].get("gpu_indices")
    if (
        not isinstance(indices, list)
        or len(indices) < world_size
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in indices
        )
    ):
        raise ModelRunError("MODEL_RUN_RESOURCE_CAPACITY_INVALID")


def _world_size(plan: Mapping[str, object]) -> int:
    parallelism = _mapping(plan.get("parallelism"), "MODEL_RUN_SCALE_PLAN_INVALID")
    value = parallelism.get("world_size")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelRunError("MODEL_RUN_SCALE_PLAN_INVALID")
    return value


def _output_root(
    execution: Mapping[str, object], model: Path, data: Path, root: Path
) -> Path:
    output = _directory(
        Path(_text(execution.get("output_root"), "MODEL_RUN_OUTPUT_INVALID")),
        "MODEL_RUN_OUTPUT_INVALID",
        require_exists=False,
    )
    _require_disjoint(output, model, "MODEL_RUN_OUTPUT_OVERLAPS_MODEL")
    _require_disjoint(output, data, "MODEL_RUN_OUTPUT_OVERLAPS_DATA")
    _require_disjoint(output, root, "MODEL_RUN_OUTPUT_OVERLAPS_SYSTEM")
    return output


def _bind_command(
    value: object,
    *,
    model_root: Path,
    source_root: Path,
    adapter_root: Path | None,
    manifest_path: Path,
    root: Path,
    code: str,
    native_config_path: Path | None = None,
    allow_native_config_placeholder: bool = False,
    allow_checkpoint_placeholder: bool = False,
    asset_bindings: Mapping[str, str] | None = None,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ModelRunError(code)
    command = [_text(token, code) for token in value]
    manifest_tokens = command.count(_MANIFEST_TOKEN)
    if manifest_tokens != (0 if allow_native_config_placeholder else 1):
        raise ModelRunError("MODEL_RUN_MANIFEST_PLACEHOLDER_REQUIRED")
    native_config_tokens = command.count("{agibot_native_config}")
    if allow_native_config_placeholder:
        if native_config_path is None or native_config_tokens != 1:
            raise ModelRunError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_PLACEHOLDER_REQUIRED")
    elif native_config_tokens:
        raise ModelRunError("MODEL_RUN_COMMAND_PLACEHOLDER_INVALID")
    checkpoint_tokens = command.count("{trained_checkpoint}")
    if checkpoint_tokens and (not allow_checkpoint_placeholder or checkpoint_tokens != 1):
        raise ModelRunError("MODEL_RUN_TRAINED_CHECKPOINT_PLACEHOLDER_INVALID")
    bound: list[str] = []
    for token in command:
        if _SHELL_META.search(token):
            raise ModelRunError(code)
        asset_match = _ASSET_TOKEN.fullmatch(token)
        if asset_match is not None:
            parameter = asset_match.group("parameter")
            asset_path = (asset_bindings or {}).get(parameter)
            if not asset_path:
                raise ModelRunError("MODEL_RUN_COMMAND_ASSET_PLACEHOLDER_UNBOUND")
            bound.append(asset_path)
            continue
        if "{asset:" in token:
            raise ModelRunError("MODEL_RUN_COMMAND_ASSET_PLACEHOLDER_INVALID")
        stripped = token.replace(_MANIFEST_TOKEN, "manifest.json")
        stripped = stripped.replace("{trained_checkpoint}", "trained-checkpoint")
        stripped = stripped.replace("{verdiwm_root}", "verdiwm-root")
        stripped = stripped.replace("{adapter_root}", "adapter-root")
        stripped = stripped.replace("{source_root}", "source-root")
        stripped = stripped.replace("{agibot_native_config}", "agibot-training.yaml")
        if "{" in stripped or "}" in stripped:
            raise ModelRunError("MODEL_RUN_COMMAND_PLACEHOLDER_INVALID")
        path = PurePosixPath(stripped)
        if not any(marker in token for marker in ("{verdiwm_root}", "{source_root}")) and (path.is_absolute() or ".." in path.parts):
            raise ModelRunError("MODEL_RUN_COMMAND_PATH_INVALID")
        if token.startswith("{verdiwm_root}/"):
            bound.append(str(root / token.removeprefix("{verdiwm_root}/")))
        elif token.startswith("{source_root}/"):
            bound.append(str(source_root / token.removeprefix("{source_root}/")))
        elif token.startswith("{adapter_root}/"):
            if adapter_root is None:
                raise ModelRunError("MODEL_RUN_COMMAND_PLACEHOLDER_INVALID")
            bound.append(str(adapter_root / token.removeprefix("{adapter_root}/")))
        elif token == "{agibot_native_config}" and native_config_path is not None:
            bound.append(str(native_config_path))
        else:
            bound.append(token)
    entrypoint = Path(bound[0]) if bound[0].startswith("/") else model_root / bound[0]
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise ModelRunError(code)
    return [
        str(manifest_path) if token == _MANIFEST_TOKEN else token for token in bound
    ]


def _optional_adapter_root(
    value: object,
    output: Path,
    model: Path,
    data: Path,
    project: Path,
) -> Path | None:
    if value is None:
        return None
    root = _directory(
        Path(_text(value, "MODEL_RUN_ADAPTER_ROOT_INVALID")),
        "MODEL_RUN_ADAPTER_ROOT_INVALID",
    )
    _require_disjoint(root, output, "MODEL_RUN_ADAPTER_ROOT_OVERLAPS_OUTPUT")
    _require_disjoint(root, model, "MODEL_RUN_ADAPTER_ROOT_OVERLAPS_MODEL")
    _require_disjoint(root, data, "MODEL_RUN_ADAPTER_ROOT_OVERLAPS_DATA")
    _require_disjoint(root, project, "MODEL_RUN_ADAPTER_ROOT_OVERLAPS_SYSTEM")
    return root


def _output_receipt(output: Path, value: object, code: str) -> Path:
    relative = _text(value, code)
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ModelRunError(code)
    return output / Path(path)


def _asset_bindings(value: object) -> dict[str, str]:
    bindings = _mapping(value, "MODEL_RUN_ASSET_BINDINGS_INVALID")
    if not bindings:
        raise ModelRunError("MODEL_RUN_ASSET_BINDINGS_INVALID")
    result: dict[str, str] = {}
    for parameter, raw_path in bindings.items():
        if not isinstance(parameter, str) or not parameter.startswith("--"):
            raise ModelRunError("MODEL_RUN_ASSET_BINDINGS_INVALID")
        result[parameter] = str(
            _existing(
                Path(_text(raw_path, "MODEL_RUN_ASSET_BINDINGS_INVALID")),
                "MODEL_RUN_ASSET_BINDINGS_INVALID",
            )
        )
    return result


def _directory(path: Path, code: str, *, require_exists: bool = True) -> Path:
    candidate = path.expanduser().resolve()
    if require_exists and (not candidate.is_dir() or path.is_symlink()):
        raise ModelRunError(code)
    return candidate


def _existing(path: Path, code: str) -> Path:
    candidate = path.expanduser()
    if not candidate.exists() or candidate.is_symlink():
        raise ModelRunError(code)
    return candidate.resolve()


def _file(value: object, code: str) -> Path:
    candidate = Path(_text(value, code)).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        raise ModelRunError(code)
    return candidate.resolve()


def _executable(value: object, code: str) -> Path:
    candidate = Path(_text(value, code)).expanduser()
    if not candidate.exists():
        raise ModelRunError(code)
    resolved = candidate.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ModelRunError(code)
    # Preserve a virtualenv's launcher symlink. Resolving it to the base
    # interpreter can discard the environment prefix and hide installed deps.
    return candidate.absolute()


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelRunError(code)
    return value


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelRunError(code)
    return value


def _require_disjoint(first: Path, second: Path, code: str) -> None:
    if _within(first, second) or _within(second, first):
        raise ModelRunError(code)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
