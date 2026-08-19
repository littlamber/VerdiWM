"""Compile LLM research proposals into bounded experiment manifests.

The compiler is intentionally non-executing.  It binds a proposal to an
already linted experiment package and a manifest-derived training plan, then
emits a receipt that a later dispatcher can consume.  A language model can
choose a hypothesis and workflow, but it cannot grant itself evaluator,
dataset, budget, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.workflow_plugins import (
    WorkflowPluginError,
    load_workflow_registry,
    require_workflow_plugins,
    select_workflow_plugins,
    workflow_capability_digest,
)
from wmloop.control.intermediate_ir import (
    IntermediateRepresentationError,
    build_experiment_ir,
    ir_digest,
    load_model_capability_ir,
    validate_experiment_ir,
)
from wmloop.experiments.engineering import lint_experiment_manifest


class ResearchProposalError(ValueError):
    """A proposal cannot be admitted to the deterministic control plane."""


def compile_proposal_to_experiment_manifest(
    proposal_path: Path,
    *,
    engineering_manifest_path: Path,
    training_scale_plan_path: Path,
    model_capability_ir_path: Path | None = None,
    workflow_registry_path: Path | None = None,
    engineering_repo_root: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Compile a proposal without launching code, workers, or GPU processes."""

    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    proposal_file = _resolve_file(proposal_path, "RESEARCH_PROPOSAL_INVALID")
    engineering_file = _resolve_file(
        engineering_manifest_path, "RESEARCH_PROPOSAL_ENGINEERING_MANIFEST_INVALID"
    )
    plan_file = _resolve_file(
        training_scale_plan_path, "RESEARCH_PROPOSAL_SCALE_PLAN_INVALID"
    )
    proposal = _load_document(proposal_file, "RESEARCH_PROPOSAL_INVALID")
    try:
        validate_document("research_proposal", proposal, root=base)
    except ContractValidationError as exc:
        raise ResearchProposalError(f"RESEARCH_PROPOSAL_SCHEMA_INVALID:{exc}") from exc
    engineering = _load_document(
        engineering_file, "RESEARCH_PROPOSAL_ENGINEERING_MANIFEST_INVALID"
    )
    try:
        validate_document("experiment_engineering_manifest", engineering, root=base)
    except ContractValidationError as exc:
        raise ResearchProposalError(
            f"RESEARCH_PROPOSAL_ENGINEERING_SCHEMA_INVALID:{exc}"
        ) from exc
    plan = _load_document(plan_file, "RESEARCH_PROPOSAL_SCALE_PLAN_INVALID")
    try:
        validate_document("training_scale_plan", plan, root=base)
    except ContractValidationError as exc:
        raise ResearchProposalError(f"RESEARCH_PROPOSAL_SCALE_SCHEMA_INVALID:{exc}") from exc

    workflow = _mapping(proposal, "workflow")
    try:
        registry = load_workflow_registry(
            registry_path=workflow_registry_path, root=base
        )
        plugins = select_workflow_plugins(
            workflow["skills"], registry_path=workflow_registry_path, root=base
        )
        require_workflow_plugins(
            workflow["workflow_id"],
            workflow["skills"],
            workflow_version=workflow["workflow_version"],
            registry_path=workflow_registry_path,
            root=base,
        )
    except WorkflowPluginError as exc:
        raise ResearchProposalError(str(exc)) from exc
    skills = {plugin.plugin_id for plugin in plugins}

    _require_authority(proposal)
    _bind_scientific_intent(proposal, engineering)
    _bind_dataset(proposal, engineering)
    _bind_evaluation(proposal, engineering)
    _bind_expected_artifacts(proposal, engineering)
    _bind_plan_dataset(engineering, plan, package_root=engineering_file.parent)
    _require_admitted_training_profile(plan)
    _bind_scale_profile(proposal, plan)
    expected_plan = (engineering_file.parent / str(engineering["scale_plan"])).resolve()
    if expected_plan != plan_file:
        raise ResearchProposalError("RESEARCH_PROPOSAL_SCALE_PLAN_BINDING_MISMATCH")

    lint = lint_experiment_manifest(
        engineering_file,
        repo_root=engineering_repo_root,
        root=base,
    )
    blockers: list[str] = []
    if lint["state"] != "ready":
        blockers.extend(f"ENGINEERING:{item}" for item in lint["blockers"])
    if plan["state"] != "ready":
        blockers.extend(f"TRAINING:{item}" for item in plan.get("blockers", []))
    capability_file: Path | None = None
    capability_ir: Mapping[str, object] | None = None
    experiment_ir: dict[str, object] | None = None
    if model_capability_ir_path is not None:
        capability_file = _resolve_file(
            model_capability_ir_path, "RESEARCH_PROPOSAL_MODEL_CAPABILITY_IR_INVALID"
        )
        try:
            capability_ir = load_model_capability_ir(capability_file, root=base)
            experiment_ir = build_experiment_ir(
                proposal=proposal,
                engineering=engineering,
                training_plan=plan,
                plugins=plugins,
                model_capability=capability_ir,
                dataset_freeze_binding=_dataset_freeze_binding(
                    engineering_dataset=_mapping(engineering, "dataset"),
                    package_root=engineering_file.parent,
                ),
                root=base,
            )
        except IntermediateRepresentationError as exc:
            raise ResearchProposalError(str(exc)) from exc
        launch = _mapping(experiment_ir, "launch")
        blockers.extend(
            f"IR:{item}" for item in launch.get("blockers", [])
        )
    if not blockers:
        state = "ready"
    else:
        state = "blocked"
    training_profile = plan.get("training_profile")
    if not isinstance(training_profile, Mapping):
        training_profile = {}
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-compiled-experiment-manifest",
        "state": state,
        "proposal": {
            "proposal_id": proposal["proposal_id"],
            "version": proposal["version"],
            "sha256": _sha256(proposal_file),
        },
        "workflow": {
            "workflow_id": workflow["workflow_id"],
            "workflow_version": workflow["workflow_version"],
            "skills": sorted(skills),
            "capability_digest": workflow_capability_digest(plugins),
            "registry_id": registry.registry_id,
            "registry_digest": registry.digest,
            "registry_path": str(registry.source_path),
            "registry_sha256": _sha256(registry.source_path),
        },
        "engineering": {
            "manifest_path": str(engineering_file),
            "manifest_sha256": _sha256(engineering_file),
            "lint_state": lint["state"],
            "source_revision": lint["source"].get("revision"),
            "source_dirty": lint["source"].get("dirty"),
        },
        "training": {
            "plan_path": str(plan_file),
            "plan_sha256": _sha256(plan_file),
            "plan_state": plan["state"],
            "stage": plan["stage"],
            "recipe": {
                "id": training_profile.get("recipe_id"),
                "status": training_profile.get("status"),
                "planner_policy": training_profile.get("planner_policy"),
            },
            "blockers": list(plan.get("blockers", [])),
        },
        "execution": {
            "launch_state": "not_started",
            "requires": [
                "gpu_lease_receipt",
                "budget_admission_receipt",
                "heldout_checkpoint_verifier",
                "archive_receipt",
            ],
            "authority": dict(_mapping(proposal, "authority")),
            "blockers": blockers,
        },
        "claim_boundary": (
            "Compilation proves proposal, repository, and training-plan contracts are bound. "
            "It does not launch training or establish a scientific effect."
        ),
    }
    if capability_file is not None and capability_ir is not None and experiment_ir is not None:
        result["model_capability"] = {
            "path": str(capability_file),
            "sha256": _sha256(capability_file),
            "semantic_digest": ir_digest(capability_ir),
        }
        result["experiment_ir"] = experiment_ir
        result["experiment_ir_sha256"] = ir_digest(experiment_ir)
    try:
        validate_document("compiled_experiment_manifest", result, root=base)
    except ContractValidationError as exc:
        raise ResearchProposalError(f"RESEARCH_PROPOSAL_COMPILED_SCHEMA_INVALID:{exc}") from exc
    return result


def write_compiled_experiment_manifest(
    manifest: Mapping[str, object], output: Path
) -> None:
    """Write a compiled manifest atomically for resumable dispatch."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_compiled_experiment_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    require_ready: bool = True,
    root: Path | None = None,
) -> Mapping[str, Any]:
    """Load and re-verify a compiled manifest before campaign admission."""

    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    source = _resolve_file(path, "COMPILED_EXPERIMENT_MANIFEST_INVALID")
    if expected_sha256 is not None and _sha256(source) != expected_sha256:
        raise ResearchProposalError("COMPILED_EXPERIMENT_MANIFEST_HASH_MISMATCH")
    payload = _load_document(source, "COMPILED_EXPERIMENT_MANIFEST_INVALID")
    try:
        validate_document("compiled_experiment_manifest", payload, root=base)
    except ContractValidationError as exc:
        raise ResearchProposalError(
            f"COMPILED_EXPERIMENT_MANIFEST_SCHEMA_INVALID:{exc}"
        ) from exc
    if require_ready and payload["state"] != "ready":
        raise ResearchProposalError("COMPILED_EXPERIMENT_MANIFEST_NOT_READY")
    workflow = _mapping(payload, "workflow")
    try:
        registry_path = _resolve_file(
            Path(str(workflow["registry_path"])),
            "COMPILED_EXPERIMENT_REGISTRY_HASH_MISMATCH",
        )
        _verify_dependency_hash(
            str(registry_path),
            workflow["registry_sha256"],
            "COMPILED_EXPERIMENT_REGISTRY_HASH_MISMATCH",
        )
        registry = load_workflow_registry(registry_path=registry_path, root=base)
        if (
            registry.registry_id != workflow["registry_id"]
            or registry.digest != workflow["registry_digest"]
        ):
            raise ResearchProposalError("COMPILED_EXPERIMENT_REGISTRY_DIGEST_MISMATCH")
        plugins = select_workflow_plugins(
            workflow["skills"], registry_path=registry_path, root=base
        )
        require_workflow_plugins(
            workflow["workflow_id"],
            workflow["skills"],
            workflow_version=workflow["workflow_version"],
            registry_path=registry_path,
            root=base,
        )
    except WorkflowPluginError as exc:
        raise ResearchProposalError(str(exc)) from exc
    if workflow["capability_digest"] != workflow_capability_digest(plugins):
        raise ResearchProposalError("COMPILED_EXPERIMENT_CAPABILITY_DIGEST_MISMATCH")
    engineering = _mapping(payload, "engineering")
    training = _mapping(payload, "training")
    model_capability = payload.get("model_capability")
    experiment_ir = payload.get("experiment_ir")
    experiment_ir_sha256 = payload.get("experiment_ir_sha256")
    if any(value is not None for value in (model_capability, experiment_ir, experiment_ir_sha256)):
        if (
            not isinstance(model_capability, Mapping)
            or not isinstance(experiment_ir, Mapping)
            or not isinstance(experiment_ir_sha256, str)
        ):
            raise ResearchProposalError("COMPILED_EXPERIMENT_IR_BINDING_INVALID")
        capability_path = _resolve_file(
            Path(str(model_capability.get("path", ""))),
            "COMPILED_EXPERIMENT_MODEL_CAPABILITY_HASH_MISMATCH",
        )
        _verify_dependency_hash(
            str(capability_path),
            model_capability.get("sha256"),
            "COMPILED_EXPERIMENT_MODEL_CAPABILITY_HASH_MISMATCH",
        )
        capability = load_model_capability_ir(capability_path, root=base)
        capability_digest = ir_digest(capability)
        if capability_digest != model_capability.get("semantic_digest"):
            raise ResearchProposalError("COMPILED_EXPERIMENT_MODEL_CAPABILITY_DIGEST_MISMATCH")
        try:
            validate_experiment_ir(experiment_ir, root=base)
        except IntermediateRepresentationError as exc:
            raise ResearchProposalError(str(exc)) from exc
        if ir_digest(experiment_ir) != experiment_ir_sha256:
            raise ResearchProposalError("COMPILED_EXPERIMENT_IR_HASH_MISMATCH")
        ir_workflow = _mapping(experiment_ir, "workflow")
        if (
            ir_workflow.get("workflow_id") != workflow["workflow_id"]
            or ir_workflow.get("workflow_version") != workflow["workflow_version"]
            or ir_workflow.get("capability_digest") != workflow["capability_digest"]
            or list(ir_workflow.get("skills", [])) != list(workflow["skills"])
        ):
            raise ResearchProposalError("COMPILED_EXPERIMENT_IR_WORKFLOW_MISMATCH")
        if experiment_ir.get("model_capability_digest") != capability_digest:
            raise ResearchProposalError(
                "COMPILED_EXPERIMENT_IR_MODEL_CAPABILITY_MISMATCH"
            )
        ir_launch = _mapping(experiment_ir, "launch")
        ir_blockers = sorted(f"IR:{item}" for item in ir_launch["blockers"])
        execution_blockers = sorted(
            item
            for item in _mapping(payload, "execution")["blockers"]
            if str(item).startswith("IR:")
        )
        if ir_blockers != execution_blockers:
            raise ResearchProposalError(
                "COMPILED_EXPERIMENT_IR_EXECUTION_MISMATCH"
            )
    _verify_dependency_hash(
        engineering["manifest_path"],
        engineering["manifest_sha256"],
        "COMPILED_EXPERIMENT_ENGINEERING_HASH_MISMATCH",
    )
    _verify_dependency_hash(
        training["plan_path"],
        training["plan_sha256"],
        "COMPILED_EXPERIMENT_SCALE_PLAN_HASH_MISMATCH",
    )
    if require_ready and (
        engineering["lint_state"] != "ready" or training["plan_state"] != "ready"
    ):
        raise ResearchProposalError("COMPILED_EXPERIMENT_DEPENDENCY_NOT_READY")
    return payload


def _require_authority(proposal: Mapping[str, Any]) -> None:
    authority = _mapping(proposal, "authority")
    expected = {
        "evaluator_policy": "frozen",
        "data_split_policy": "frozen",
        "promotion_policy": "verifier_only",
    }
    if dict(authority) != expected:
        raise ResearchProposalError("RESEARCH_PROPOSAL_AUTHORITY_MUTATION_FORBIDDEN")


def _bind_scientific_intent(
    proposal: Mapping[str, Any], engineering: Mapping[str, Any]
) -> None:
    if proposal["proposal_id"] != engineering["experiment_id"]:
        raise ResearchProposalError("RESEARCH_PROPOSAL_EXPERIMENT_ID_MISMATCH")
    for field in (
        "version",
        "objective",
        "hypothesis",
        "falsification_criterion",
        "selection_reason",
    ):
        if proposal[field] != engineering[field]:
            raise ResearchProposalError(
                f"RESEARCH_PROPOSAL_SCIENTIFIC_INTENT_MISMATCH:{field}"
            )


def _bind_dataset(
    proposal: Mapping[str, Any], engineering: Mapping[str, Any]
) -> None:
    proposed = _mapping(proposal, "dataset")
    declared = _mapping(engineering, "dataset")
    for field in ("train_manifest", "validation_manifest", "dataset_freeze", "split_policy"):
        if proposed[field] != declared[field]:
            raise ResearchProposalError(f"RESEARCH_PROPOSAL_DATASET_BINDING_MISMATCH:{field}")
    if proposed["mutation_policy"] != "frozen":
        raise ResearchProposalError("RESEARCH_PROPOSAL_DATASET_MUTATION_FORBIDDEN")


def _bind_evaluation(
    proposal: Mapping[str, Any], engineering: Mapping[str, Any]
) -> None:
    if dict(_mapping(proposal, "evaluation")) != dict(_mapping(engineering, "evaluation")):
        raise ResearchProposalError("RESEARCH_PROPOSAL_EVALUATION_MUTATION_FORBIDDEN")


def _bind_expected_artifacts(
    proposal: Mapping[str, Any], engineering: Mapping[str, Any]
) -> None:
    artifact_policy = _mapping(engineering, "artifact_policy")
    proposed = sorted(str(value) for value in proposal["expected_artifacts"])
    declared = sorted(str(value) for value in artifact_policy["required_artifacts"])
    if proposed != declared:
        raise ResearchProposalError("RESEARCH_PROPOSAL_ARTIFACT_POLICY_MISMATCH")


def _bind_plan_dataset(
    engineering: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    package_root: Path,
) -> None:
    declared = _mapping(engineering, "dataset")
    plan_dataset = _mapping(plan, "dataset")
    manifest_paths = _mapping(plan_dataset, "manifest_paths")
    train = _declared_file(
        package_root,
        declared["train_manifest"],
        "RESEARCH_PROPOSAL_TRAIN_MANIFEST_INVALID",
    )
    validation = _declared_file(
        package_root,
        declared["validation_manifest"],
        "RESEARCH_PROPOSAL_VALIDATION_MANIFEST_INVALID",
    )
    if _normalized_path(manifest_paths.get("train")) != train:
        raise ResearchProposalError("RESEARCH_PROPOSAL_SCALE_TRAIN_MANIFEST_MISMATCH")
    if _normalized_path(manifest_paths.get("validation")) != validation:
        raise ResearchProposalError("RESEARCH_PROPOSAL_SCALE_VALIDATION_MANIFEST_MISMATCH")
    if _sha256(train) != plan_dataset["train_manifest_sha256"]:
        raise ResearchProposalError("RESEARCH_PROPOSAL_SCALE_TRAIN_MANIFEST_HASH_MISMATCH")
    if _sha256(validation) != plan_dataset["validation_manifest_sha256"]:
        raise ResearchProposalError(
            "RESEARCH_PROPOSAL_SCALE_VALIDATION_MANIFEST_HASH_MISMATCH"
        )


def _require_admitted_training_profile(plan: Mapping[str, Any]) -> None:
    raw_profile = plan.get("training_profile")
    if raw_profile is None:
        return
    if not isinstance(raw_profile, Mapping):
        raise ResearchProposalError("RESEARCH_PROPOSAL_TRAINING_PROFILE_INVALID")
    if (
        raw_profile.get("status")
        not in {"local_validated", "reusable_optimization_memory"}
        or raw_profile.get("planner_policy") != "admitted"
    ):
        raise ResearchProposalError("RESEARCH_PROPOSAL_TRAINING_RECIPE_NOT_ADMITTED")


def _bind_scale_profile(proposal: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    profile = _mapping(proposal, "scale_profile")
    parallelism = _mapping(plan, "parallelism")
    dataset = _mapping(plan, "dataset")
    replication = _mapping(plan, "replication")
    training_profile = plan.get("training_profile")
    if not isinstance(training_profile, Mapping):
        training_profile = {}
    expected = {
        "stage": plan["stage"],
        "batch_size": parallelism["batch_size"],
        "gradient_accumulation": parallelism["gradient_accumulation"],
        "world_size": parallelism["world_size"],
        "sequence_length": dataset.get("sequence_length"),
        "requested_seed_count": replication["requested_seed_count"],
        "training_recipe_id": training_profile.get("recipe_id"),
    }
    if dict(profile) != expected:
        raise ResearchProposalError("RESEARCH_PROPOSAL_SCALE_PROFILE_MISMATCH")


def _dataset_freeze_binding(
    *, engineering_dataset: Mapping[str, Any], package_root: Path
) -> str:
    value = engineering_dataset.get("dataset_freeze")
    if not isinstance(value, str) or not value.strip():
        raise ResearchProposalError("RESEARCH_PROPOSAL_DATASET_FREEZE_INVALID")
    text = value.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        return "dataset-freeze:sha256:" + text.removeprefix("sha256:")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = package_root / candidate
    if candidate.is_file() and not candidate.is_symlink():
        return "dataset-freeze:sha256:" + _sha256(candidate.resolve())
    if (
        Path(text).is_absolute()
        or "/" in text
        or "\\" in text
        or Path(text).suffix.lower() in {".json", ".jsonl", ".yaml", ".yml"}
    ):
        raise ResearchProposalError("RESEARCH_PROPOSAL_DATASET_FREEZE_INVALID")
    return (
        "dataset-freeze:semantic:"
        + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def _mapping(document: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        raise ResearchProposalError(f"RESEARCH_PROPOSAL_{field.upper()}_OBJECT_REQUIRED")
    return value


def _load_document(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchProposalError(code) from exc
    if not isinstance(payload, Mapping):
        raise ResearchProposalError(code)
    return payload


def _resolve_file(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ResearchProposalError(code)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ResearchProposalError(code) from exc
    if not resolved.is_file():
        raise ResearchProposalError(code)
    return resolved


def _declared_file(root: Path, value: object, code: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResearchProposalError(code)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return _resolve_file(candidate, code)


def _normalized_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).expanduser().resolve(strict=True)
    except OSError:
        return None


def _verify_dependency_hash(value: object, expected: object, code: str) -> None:
    if not isinstance(value, str) or not isinstance(expected, str):
        raise ResearchProposalError(code)
    path = _resolve_file(Path(value), code)
    if _sha256(path) != expected:
        raise ResearchProposalError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
