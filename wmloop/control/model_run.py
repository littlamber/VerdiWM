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


class ModelRunError(ValueError):
    """A model-run manifest cannot be safely compiled."""


_MANIFEST_TOKEN = "{model_run_manifest}"
_SHELL_META = re.compile(r"[;&|`\n\r]")


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
    _validate_scale_plan(training_scale_plan, root=base)
    _validate_allocation(
        resource_allocation, world_size=_world_size(training_scale_plan)
    )

    model_root = _directory(model, "MODEL_RUN_MODEL_INVALID")
    dataset_root = _existing(data, "MODEL_RUN_DATASET_INVALID")
    output_root = _output_root(execution, model_root, dataset_root, base)
    manifest_path = output_root / "model-run.json"
    evaluator = _file(
        execution.get("evaluator_contract"), "MODEL_RUN_EVALUATOR_INVALID"
    )
    runtime = _executable(execution.get("runtime_python"), "MODEL_RUN_RUNTIME_INVALID")
    assets = _asset_bindings(execution.get("asset_bindings"))
    adapter_root = _optional_adapter_root(
        adapter.get("adapter_root"), output_root, model_root, dataset_root, base
    )
    train_command = _bind_command(
        runner.get("train"),
        model_root=model_root,
        adapter_root=adapter_root,
        manifest_path=manifest_path,
        root=base,
        code="MODEL_RUN_TRAIN_COMMAND_INVALID",
    )
    evaluate_command = _bind_command(
        runner.get("evaluate"),
        model_root=model_root,
        adapter_root=adapter_root,
        manifest_path=manifest_path,
        root=base,
        code="MODEL_RUN_EVALUATE_COMMAND_INVALID",
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
            "protocol_version": runner.get("protocol_version"),
        },
        "runner": {"adapter_id": runner.get("adapter_id")},
        "source": {
            "model_root": str(model_root),
            "dataset_root": str(dataset_root),
            "asset_bindings": assets,
            **({"adapter_root": str(adapter_root)} if adapter_root is not None else {}),
        },
        "runtime": {"python": str(runtime), "train_command": train_command},
        "training": {
            "scale_plan": dict(training_scale_plan),
            "scale_plan_digest": _digest(training_scale_plan),
            "checkpoint_receipt": str(checkpoint_receipt),
        },
        "evaluation": {
            "evaluator_contract": str(evaluator),
            "command": evaluate_command,
            "evidence_receipt": str(evidence_receipt),
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
        "isolation": {
            "model_source_read_only": True,
            "dataset_read_only": True,
            "evaluator_read_only": True,
            "output_root_fresh": True,
            "network_access": False,
        },
        "execution_policy": {
            "stage_timeout_seconds": 24.0 * 3600.0,
            "timeout_receipt": "execution-timeout-receipt.json",
        },
        "claim_boundary": (
            "This manifest admits a bounded external model run only after a GPU lease. "
            "Checkpoint output is not promotion evidence without the declared held-out receipt."
        ),
    }
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
    _require_disjoint(output, model_root, "MODEL_RUN_OUTPUT_OVERLAPS_MODEL")
    _require_disjoint(output, dataset_root, "MODEL_RUN_OUTPUT_OVERLAPS_DATA")
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
    stopping = _mapping(plan.get("stopping_policy"), "MODEL_RUN_SCALE_PLAN_INVALID")
    if stopping.get("evaluate_each_checkpoint_on_heldout") is not True:
        raise ModelRunError("MODEL_RUN_HELDOUT_EVALUATION_REQUIRED")


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
    adapter_root: Path | None,
    manifest_path: Path,
    root: Path,
    code: str,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ModelRunError(code)
    command = [_text(token, code) for token in value]
    if command.count(_MANIFEST_TOKEN) != 1:
        raise ModelRunError("MODEL_RUN_MANIFEST_PLACEHOLDER_REQUIRED")
    bound: list[str] = []
    for token in command:
        if _SHELL_META.search(token):
            raise ModelRunError(code)
        stripped = token.replace(_MANIFEST_TOKEN, "manifest.json")
        stripped = stripped.replace("{verdiwm_root}", "verdiwm-root")
        stripped = stripped.replace("{adapter_root}", "adapter-root")
        if "{" in stripped or "}" in stripped:
            raise ModelRunError("MODEL_RUN_COMMAND_PLACEHOLDER_INVALID")
        path = PurePosixPath(stripped)
        if "{verdiwm_root}" not in token and (path.is_absolute() or ".." in path.parts):
            raise ModelRunError("MODEL_RUN_COMMAND_PATH_INVALID")
        bound.append(
            str(root / token.replace("{verdiwm_root}", "").lstrip("/"))
            if token.startswith("{verdiwm_root}/")
            else (
                str(adapter_root / token.replace("{adapter_root}", "").lstrip("/"))
                if token.startswith("{adapter_root}/") and adapter_root is not None
                else token
            )
        )
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
