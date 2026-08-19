"""Portable intermediate representations between model, experiment, and kernel.

The IRs contain semantic capabilities and content digests.  Local paths stay in
the onboarding connector or execution receipt, so reusable compilation does not
bind shared knowledge to one checkout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.workflow_plugins import (
    WorkflowPlugin,
    required_model_capabilities,
    workflow_capability_digest,
)


class IntermediateRepresentationError(ValueError):
    """An IR source or semantic binding is invalid."""


_ENTRYPOINT_KINDS = ("training", "evaluation", "rollout", "inference")
_CONTENT_BINDING_PREFIXES = ("cas://", "sha256:", "urn:")
_FILE_LIKE_SUFFIX = re.compile(
    r"\.(?:bin|ckpt|jsonl?|onnx|pt|pth|safetensors|toml|ya?ml)$",
    re.IGNORECASE,
)


def build_model_capability_ir(
    onboarding_report: Mapping[str, object],
    *,
    model_family: str | None = None,
    hooks: Sequence[Mapping[str, object]] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Project a read-only onboarding report into path-independent capabilities."""

    if onboarding_report.get("artifact_type") != "wmloop-model-onboarding-report":
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_SOURCE_INVALID")
    connector = _mapping(onboarding_report, "connector")
    source_revision = _mapping(onboarding_report, "source_revision")
    capabilities = []
    for row in _mapping_rows(onboarding_report.get("capabilities")):
        state = str(row.get("state"))
        capabilities.append(
            {
                "capability": _required_text(
                    row.get("capability"),
                    "MODEL_CAPABILITY_IR_CAPABILITY_INVALID",
                ),
                "state": "available" if state == "discovered" else "unknown",
                "evidence_count": (
                    len(row.get("evidence", []))
                    if isinstance(row.get("evidence"), list)
                    else 0
                ),
            }
        )
    if not capabilities:
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_CAPABILITY_MISSING")

    grouped = connector.get("entrypoints_by_kind")
    if not isinstance(grouped, Mapping):
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_ENTRYPOINTS_INVALID")
    interfaces = [
        {
            "kind": kind,
            "contract_id": f"model-entrypoint:{kind}:v1",
            "state": (
                "available"
                if isinstance(grouped.get(kind), list) and bool(grouped[kind])
                else "unavailable"
            ),
        }
        for kind in _ENTRYPOINT_KINDS
    ]
    hook_rows = [_normalized_hook(row) for row in hooks]
    hook_ids = [str(row["hook"]) for row in hook_rows]
    if len(hook_ids) != len(set(hook_ids)):
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_HOOK_DUPLICATE")

    asset_classes = sorted(
        {
            str(row.get("kind"))
            for row in _mapping_rows(connector.get("asset_bindings"))
            if isinstance(row.get("kind"), str) and str(row.get("kind")).strip()
        }
    )
    evaluator = _mapping(onboarding_report, "evaluator_contract")
    evaluator_state = str(evaluator.get("state", "blocked"))
    if evaluator_state not in {"ready", "binding_required", "blocked"}:
        evaluator_state = "blocked"
    family = _slug(model_family or str(onboarding_report.get("repo_name", "model")))
    source_kind = str(source_revision.get("kind", "none"))
    if source_kind not in {"git_commit", "source_tree_sha256", "none"}:
        source_kind = "none"
    source_value = _required_text(
        source_revision.get("revision"), "MODEL_CAPABILITY_IR_SOURCE_REVISION_INVALID"
    )
    semantic_source = {
        "model_family": family,
        "source_revision": {"kind": source_kind, "value": source_value},
    }
    verifier = _optional_text(evaluator.get("verifier"))
    if verifier is not None and _looks_like_runtime_binding(verifier):
        verifier = None
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-capability-ir",
        "model_family": family,
        "source_revision": {"kind": source_kind, "value": source_value},
        "capabilities": sorted(capabilities, key=lambda row: str(row["capability"])),
        "execution_interfaces": interfaces,
        "hooks": sorted(hook_rows, key=lambda row: str(row["hook"])),
        "asset_classes": asset_classes,
        "evaluator": {
            "state": evaluator_state,
            "evaluator_id": _optional_text(evaluator.get("evaluator_id")),
            "contract_digest": _optional_text(evaluator.get("contract_sha256")),
            "verifier": verifier,
        },
        "authority": {
            "source_mutation_allowed": False,
            "claim_authority": "target_verifier" if evaluator_state == "ready" else "none",
        },
        "provenance": {
            "source_artifact_type": str(onboarding_report["artifact_type"]),
            "source_digest": ir_digest(semantic_source),
        },
    }
    body["capability_id"] = _derived_id("model-capability", body)
    validate_model_capability_ir(body, root=root)
    return body


def validate_model_capability_ir(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        validate_document("model_capability_ir", document, root=root)
    except ContractValidationError as exc:
        raise IntermediateRepresentationError(
            f"MODEL_CAPABILITY_IR_SCHEMA_INVALID:{exc}"
        ) from exc
    _reject_runtime_bindings(document, "MODEL_CAPABILITY_IR_RUNTIME_BINDING_FORBIDDEN")
    _require_unique_rows(document, "capabilities", "capability")
    _require_unique_rows(document, "execution_interfaces", "kind")
    _require_unique_rows(document, "hooks", "hook")
    assets = document.get("asset_classes")
    if not isinstance(assets, list) or len(assets) != len(set(assets)):
        raise IntermediateRepresentationError(
            "MODEL_CAPABILITY_IR_ASSET_CLASS_DUPLICATE"
        )
    body = dict(document)
    body.pop("capability_id", None)
    if document.get("capability_id") != _derived_id("model-capability", body):
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_ID_MISMATCH")


def load_model_capability_ir(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_FILE_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_FILE_INVALID")
    validate_model_capability_ir(payload, root=root)
    return payload


def build_experiment_ir(
    *,
    proposal: Mapping[str, object],
    engineering: Mapping[str, object],
    training_plan: Mapping[str, object],
    plugins: Sequence[WorkflowPlugin],
    model_capability: Mapping[str, object],
    dataset_freeze_binding: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Compile one model-neutral experiment contract against declared capabilities."""

    validate_model_capability_ir(model_capability, root=root)
    workflow = _mapping(proposal, "workflow")
    proposal_dataset = _mapping(proposal, "dataset")
    engineering_dataset = _mapping(engineering, "dataset")
    scale = _mapping(proposal, "scale_profile")
    evaluation = _mapping(proposal, "evaluation")
    budget = _mapping(proposal, "budget")
    available = {
        str(row["capability"])
        for row in _mapping_rows(model_capability.get("capabilities"))
        if row.get("state") == "available"
    }
    required = required_model_capabilities(plugins)
    blockers = [
        f"MODEL_CAPABILITY_MISSING:{capability}"
        for capability in required
        if capability not in available
    ]
    evaluator = _mapping(model_capability, "evaluator")
    if "evaluation" in required and evaluator.get("state") != "ready":
        blockers.append("MODEL_EVALUATOR_UNBOUND")

    freeze_source = _required_text(
        engineering_dataset.get("dataset_freeze"), "EXPERIMENT_IR_DATASET_FREEZE_INVALID"
    )
    if dataset_freeze_binding is None:
        if _looks_like_runtime_binding(freeze_source):
            raise IntermediateRepresentationError(
                "EXPERIMENT_IR_DATASET_FREEZE_CONTENT_BINDING_REQUIRED"
            )
        freeze_id = (
            "dataset-freeze:semantic:"
            + hashlib.sha256(freeze_source.encode("utf-8")).hexdigest()
        )
    else:
        freeze_id = _required_text(
            dataset_freeze_binding, "EXPERIMENT_IR_DATASET_FREEZE_INVALID"
        )
        if re.fullmatch(r"dataset-freeze:(?:semantic|sha256):[0-9a-f]{64}", freeze_id) is None:
            raise IntermediateRepresentationError(
                "EXPERIMENT_IR_DATASET_FREEZE_INVALID"
            )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-experiment-ir",
        "experiment_id": _required_text(proposal.get("proposal_id"), "EXPERIMENT_IR_ID_INVALID"),
        "version": _required_text(proposal.get("version"), "EXPERIMENT_IR_VERSION_INVALID"),
        "model_capability_digest": ir_digest(model_capability),
        "objective": _required_text(proposal.get("objective"), "EXPERIMENT_IR_OBJECTIVE_INVALID"),
        "hypothesis": _required_text(
            proposal.get("hypothesis"), "EXPERIMENT_IR_HYPOTHESIS_INVALID"
        ),
        "falsification_criterion": _required_text(
            proposal.get("falsification_criterion"), "EXPERIMENT_IR_FALSIFICATION_INVALID"
        ),
        "workflow": {
            "workflow_id": _required_text(
                workflow.get("workflow_id"), "EXPERIMENT_IR_WORKFLOW_INVALID"
            ),
            "workflow_version": _required_text(
                workflow.get("workflow_version"),
                "EXPERIMENT_IR_WORKFLOW_INVALID",
            ),
            "skills": sorted(plugin.plugin_id for plugin in plugins),
            "capability_digest": workflow_capability_digest(plugins),
        },
        "dataset": {
            "freeze_id": freeze_id,
            "split_policy": _required_text(
                proposal_dataset.get("split_policy"),
                "EXPERIMENT_IR_SPLIT_INVALID",
            ),
            "mutation_policy": "frozen",
        },
        "training": {
            "stage": training_plan.get("stage"),
            "batch_size": scale.get("batch_size"),
            "gradient_accumulation": scale.get("gradient_accumulation"),
            "world_size": scale.get("world_size"),
            "sequence_length": scale.get("sequence_length"),
            "seed_count": scale.get("requested_seed_count"),
            "plan_digest": training_plan_semantic_digest(training_plan),
        },
        "evaluation": {
            "metrics": list(evaluation.get("metrics", [])),
            "horizons": list(evaluation.get("horizons", [])),
            "seeds": list(evaluation.get("seeds", [])),
            "heldout_protocol": evaluation.get("heldout_protocol"),
            "evaluator_digest": evaluator.get("contract_digest"),
        },
        "interventions": [],
        "budget": {
            "max_gpu_hours": budget.get("max_gpu_hours"),
            "max_trials": budget.get("max_trials"),
        },
        "artifacts": _artifact_classes(proposal.get("expected_artifacts")),
        "authority": dict(_mapping(proposal, "authority")),
        "launch": {
            "state": "blocked" if blockers else "not_started",
            "blockers": sorted(set(blockers)),
        },
    }
    validate_experiment_ir(body, root=root)
    return body


def validate_experiment_ir(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        validate_document("experiment_ir", document, root=root)
    except ContractValidationError as exc:
        raise IntermediateRepresentationError(
            f"EXPERIMENT_IR_SCHEMA_INVALID:{exc}"
        ) from exc
    _validate_experiment_semantics(document)
    launch = _mapping(document, "launch")
    blockers = launch.get("blockers")
    expected_state = "blocked" if blockers else "not_started"
    if launch.get("state") != expected_state:
        raise IntermediateRepresentationError("EXPERIMENT_IR_LAUNCH_STATE_INVALID")


def training_plan_semantic_digest(training_plan: Mapping[str, object]) -> str:
    """Hash a training plan after removing local manifest locations."""

    projection = dict(training_plan)
    dataset = dict(_mapping(training_plan, "dataset"))
    dataset.pop("manifest_paths", None)
    projection["dataset"] = dataset
    return ir_digest(projection)


def ir_digest(value: object) -> str:
    """Return a deterministic content digest for a JSON-compatible IR source."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IntermediateRepresentationError("IR_CANONICALIZATION_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise IntermediateRepresentationError(f"IR_MAPPING_REQUIRED:{name}")
    return result


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise IntermediateRepresentationError("IR_MAPPING_ROWS_INVALID")
    return list(value)


def _normalized_hook(row: Mapping[str, object]) -> dict[str, object]:
    state = str(row.get("state", "unknown"))
    if state not in {"available", "unavailable", "unknown"}:
        raise IntermediateRepresentationError("MODEL_CAPABILITY_IR_HOOK_STATE_INVALID")
    return {
        "hook": _required_text(row.get("hook"), "MODEL_CAPABILITY_IR_HOOK_INVALID"),
        "state": state,
        "semantic_role": _required_text(
            row.get("semantic_role"), "MODEL_CAPABILITY_IR_HOOK_INVALID"
        ),
        "binding_contract": _required_text(
            row.get("binding_contract"), "MODEL_CAPABILITY_IR_HOOK_INVALID"
        ),
    }


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntermediateRepresentationError(code)
    return value.strip()


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "model"


def _derived_id(prefix: str, body: Mapping[str, object]) -> str:
    return f"{prefix}-{ir_digest(body)[:24]}"


def _require_unique_rows(
    document: Mapping[str, object], field: str, identity: str
) -> None:
    rows = _mapping_rows(document.get(field))
    values = [str(row.get(identity)) for row in rows]
    if len(values) != len(set(values)):
        raise IntermediateRepresentationError(
            f"MODEL_CAPABILITY_IR_{identity.upper()}_DUPLICATE"
        )


def _reject_runtime_bindings(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_runtime_bindings(child, code)
    elif isinstance(value, list):
        for child in value:
            _reject_runtime_bindings(child, code)
    elif isinstance(value, str) and _looks_like_runtime_binding(value):
        raise IntermediateRepresentationError(code)


def _looks_like_runtime_binding(value: str) -> bool:
    if value.startswith(_CONTENT_BINDING_PREFIXES):
        return False
    return bool(
        value.startswith(("/", "~/", "file://"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or "/" in value
        or "\\" in value
        or _FILE_LIKE_SUFFIX.search(value)
    )


def _artifact_classes(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise IntermediateRepresentationError("EXPERIMENT_IR_ARTIFACTS_INVALID")
    classes = []
    for raw in value:
        text = _required_text(raw, "EXPERIMENT_IR_ARTIFACTS_INVALID")
        leaf = text.replace("\\", "/").rsplit("/", 1)[-1]
        stem = leaf.rsplit(".", 1)[0] if "." in leaf else leaf
        classes.append("artifact:" + _slug(stem))
    return sorted(set(classes))


def _validate_experiment_semantics(document: Mapping[str, object]) -> None:
    workflow = _mapping(document, "workflow")
    dataset = _mapping(document, "dataset")
    evaluation = _mapping(document, "evaluation")
    values: list[object] = [
        workflow.get("workflow_id"),
        workflow.get("workflow_version"),
        dataset.get("split_policy"),
        evaluation.get("heldout_protocol"),
        *list(workflow.get("skills", [])),
        *list(evaluation.get("metrics", [])),
        *list(document.get("artifacts", [])),
    ]
    for row in _mapping_rows(document.get("interventions")):
        values.append(row.get("primitive_id"))
        values.extend(list(row.get("target_hooks", [])))
    if any(
        isinstance(value, str) and _looks_like_runtime_binding(value)
        for value in values
    ):
        raise IntermediateRepresentationError(
            "EXPERIMENT_IR_RUNTIME_BINDING_FORBIDDEN"
        )
