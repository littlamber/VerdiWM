"""Compile an LLM research idea into a trusted ABI-bound materialization plan."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.intermediate_ir import load_model_capability_ir
from wmloop.control.module_composition import (
    ModuleCompositionError,
    validate_module_abi_registry,
)
from wmloop.control.module_manufacturing import (
    ModuleManufacturingError,
    load_module_manufacturing_work_order,
)
from wmloop.execute.automatic_materialization import materialization_plan_digest
from wmloop.execute.llm_task_adapter import run_llm_task


class AutomaticModulePlanError(RuntimeError):
    """An automatic module specification crossed a trusted compiler boundary."""


_PROMPT_TEMPLATE = """Design one faithful implementation for a pre-approved module ABI.
Select an ABI only when the source mechanism fits it without substitution.
Return a capability-gap state when an interface, architecture, or data regime is missing.
Generated code may implement only the declared symbol and may not access files, networks,
processes, environment variables, evaluators, datasets, GPU policy, or promotion policy.
The kernel chooses candidate identity, paths, parameters, tests, evaluator, budget, and lifecycle.
"""
_PROMPT_TEMPLATE_DIGEST = hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
_EXECUTABLE_STATES = {"source_faithful", "derived_embodiment"}
_GAP_STATES = {
    "requires_interface_extension",
    "architecture_bound",
    "missing_data_regime",
}
_BAD_CALL_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_BAD_ATTRIBUTE_ROOTS = {
    "builtins",
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}
_BAD_ATTRIBUTE_NAMES = {
    "__class__",
    "__dict__",
    "__globals__",
    "__subclasses__",
    "compile",
    "cuda",
    "distributed",
    "hub",
    "jit",
    "load",
    "save",
}
_ALLOWED_ATTRIBUTE_CALLS = {
    "adaptive_avg_pool2d",
    "all",
    "argsort",
    "clamp",
    "clamp_min",
    "exp",
    "expand",
    "flip",
    "float",
    "gather",
    "is_floating_point",
    "isfinite",
    "linspace",
    "norm",
    "reshape",
    "softmax",
    "sort",
    "sum",
    "to",
    "unsqueeze",
    "view",
}


def abi_registry_digest(registry: Mapping[str, object]) -> str:
    payload = {key: value for key, value in registry.items() if key != "registry_digest"}
    return _sha256(_canonical_json(payload))


def compile_automatic_module_plan(
    *,
    idea_path: Path,
    work_order_path: Path,
    assessment_path: Path,
    model_capability_ir_path: Path,
    abi_registry_path: Path,
    adapter: Mapping[str, object],
    runtime_python: Path,
    output_root: Path,
    project_root: Path,
    manufacturing_work_order_path: Path | None = None,
) -> dict[str, object]:
    """Produce a content-addressed plan or a differentiated capability gap."""

    root = Path(project_root).resolve()
    idea_file = _require_file(idea_path, "AUTOMATIC_MODULE_IDEA_INVALID")
    work_file = _require_file(work_order_path, "AUTOMATIC_MODULE_WORK_ORDER_INVALID")
    assessment_file = _require_file(assessment_path, "AUTOMATIC_MODULE_ASSESSMENT_INVALID")
    capability_file = _require_file(
        model_capability_ir_path, "AUTOMATIC_MODULE_CAPABILITY_IR_INVALID"
    )
    registry_file = _require_file(abi_registry_path, "AUTOMATIC_MODULE_ABI_REGISTRY_INVALID")
    python = _require_runtime_file(
        runtime_python, "AUTOMATIC_MODULE_RUNTIME_PYTHON_INVALID"
    )
    idea = _load(idea_file, "AUTOMATIC_MODULE_IDEA_INVALID")
    work_order = _load(work_file, "AUTOMATIC_MODULE_WORK_ORDER_INVALID")
    assessment = _load(assessment_file, "AUTOMATIC_MODULE_ASSESSMENT_INVALID")
    capability = load_model_capability_ir(capability_file, root=root)
    registry = _load(registry_file, "AUTOMATIC_MODULE_ABI_REGISTRY_INVALID")
    _validate_registry(registry, root=root)
    manufacturing_file = None
    manufacturing_order = None
    if manufacturing_work_order_path is not None:
        manufacturing_file = _require_file(
            manufacturing_work_order_path,
            "AUTOMATIC_MODULE_MANUFACTURING_WORK_ORDER_INVALID",
        )
        try:
            manufacturing_order = load_module_manufacturing_work_order(
                manufacturing_file,
                root=root,
            )
        except ModuleManufacturingError as exc:
            raise AutomaticModulePlanError(
                f"AUTOMATIC_MODULE_MANUFACTURING_WORK_ORDER_INVALID:{exc}"
            ) from exc
        _validate_manufacturing_registry_binding(
            manufacturing_order,
            registry=registry,
        )
    input_lock = {
        "idea_sha256": _sha256(idea_file.read_bytes()),
        "work_order_sha256": _sha256(work_file.read_bytes()),
        "assessment_sha256": _sha256(assessment_file.read_bytes()),
        "model_capability_ir_sha256": _sha256(capability_file.read_bytes()),
        "abi_registry_sha256": _sha256(registry_file.read_bytes()),
        "abi_registry_digest": registry["registry_digest"],
        "runtime_python": str(python),
        "adapter": _adapter_identity(adapter),
    }
    if manufacturing_file is not None and manufacturing_order is not None:
        input_lock.update(
            {
                "manufacturing_work_order_sha256": _sha256(
                    manufacturing_file.read_bytes()
                ),
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_work_order_digest": manufacturing_order[
                    "work_order_digest"
                ],
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
            }
        )
    input_digest = _sha256(_canonical_json(input_lock))
    destination = Path(output_root).expanduser().resolve()
    resumed = _resume(destination, input_digest=input_digest)
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_OUTPUT_INVALID")
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "input-lock.json", {**input_lock, "input_digest": input_digest})

    abi_rows = registry.get("abis")
    assert isinstance(abi_rows, list)
    requested_abi_id = (
        str(manufacturing_order["target_abi"]["abi_id"])
        if manufacturing_order is not None
        else None
    )
    candidates = [
        _abi_for_prompt(row, capability=capability, idea=idea)
        for row in abi_rows
        if isinstance(row, Mapping)
        and str(row.get("model_family")) == str(capability.get("model_family"))
        and (requested_abi_id is None or row.get("abi_id") == requested_abi_id)
    ]
    if manufacturing_order is not None and len(candidates) != 1:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_UNAVAILABLE")
    request = _module_request(
        idea=idea,
        work_order=work_order,
        assessment=assessment,
        capability=capability,
        abi_candidates=candidates,
        registry=registry,
        manufacturing_order=manufacturing_order,
    )
    task_manifest = run_llm_task(
        request=request,
        adapter=adapter,
        output_root=destination / "llm-task",
        project_root=root,
    )
    if task_manifest.get("state") != "completed":
        return _write_gap(
            destination,
            input_digest=input_digest,
            idea=idea,
            state="operational_failure",
            blockers=[
                {
                    "code": "LLM_MODULE_TASK_BLOCKED",
                    "llm_task_receipt_path": task_manifest.get("receipt_path"),
                }
            ],
            task_manifest=task_manifest,
        )
    response = _load(
        Path(str(task_manifest["response_path"])), "AUTOMATIC_MODULE_RESPONSE_INVALID"
    )
    spec = response.get("output")
    if not isinstance(spec, dict):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_SPEC_INVALID")
    _validate_document("automatic_module_spec", spec, root=root)
    if spec["state"] in _GAP_STATES:
        return _write_gap(
            destination,
            input_digest=input_digest,
            idea=idea,
            state=str(spec["state"]),
            blockers=[dict(row) for row in spec["blockers"] if isinstance(row, Mapping)],
            task_manifest=task_manifest,
        )
    if spec["state"] not in _EXECUTABLE_STATES:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_SPEC_STATE_INVALID")
    abi = _selected_abi(spec=spec, registry=registry, capability=capability, idea=idea)
    if manufacturing_order is not None and abi["abi_id"] != requested_abi_id:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_MISMATCH")
    module_source = str(spec.get("module_source") or "")
    _validate_module_source(module_source, abi=abi)
    preserved = _preserved_components(idea)
    touchpoints = _component_touchpoints(spec, preserved=preserved)
    spec_digest = _sha256(_canonical_json(spec))
    candidate_id = _candidate_id(
        str(idea["idea_id"]),
        spec_digest=spec_digest,
        manufacturing_work_order_digest=(
            str(manufacturing_order["work_order_digest"])
            if manufacturing_order is not None
            else None
        ),
    )
    module_path = _repository_path(abi["module_path"], "AUTOMATIC_MODULE_ABI_MODULE_PATH_INVALID")
    test_path = _repository_path(abi["test_path"], "AUTOMATIC_MODULE_ABI_TEST_PATH_INVALID")
    descriptor_path = f"configs/verdiwm_generated_{candidate_id}.json"
    test_template = _resolve_project_file(
        root, abi["test_template"], "AUTOMATIC_MODULE_ABI_TEST_TEMPLATE_INVALID"
    ).read_text(encoding="utf-8")
    descriptor = {
        "schema_version": 1,
        "artifact_type": "verdiwm-materialized-method-descriptor",
        "candidate_id": candidate_id,
        "idea_id": idea["idea_id"],
        "implementation_files": [module_path, test_path],
        "intent_to_code": touchpoints,
        "runtime_contract": spec["runtime_contract"],
        "negative_check": spec["negative_check"],
        "applicability_conditions": list(spec["applicability_conditions"]),
        "failure_boundaries": list(spec["failure_boundaries"]),
        "declared_compromises": [],
    }
    _validate_document("materialized_method_descriptor", descriptor, root=root)
    bundle = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-bundle",
        "bundle_digest": "",
        "candidate_id": candidate_id,
        "idea_id": idea["idea_id"],
        "abi_id": abi["abi_id"],
        "module_path": module_path,
        "test_path": test_path,
        "descriptor_path": descriptor_path,
        "module_source": module_source,
        "module_source_sha256": _sha256(module_source.encode("utf-8")),
        "test_source": test_template,
        "test_source_sha256": _sha256(test_template.encode("utf-8")),
        "descriptor": descriptor,
    }
    bundle["bundle_digest"] = _document_digest(bundle, excluded="bundle_digest")
    _validate_document("automatic_module_bundle", bundle, root=root)
    bundle_path = destination / "module-bundle.json"
    _write_json(bundle_path, bundle)
    evaluator = _resolve_project_file(
        root, abi["evaluator"], "AUTOMATIC_MODULE_ABI_EVALUATOR_INVALID"
    )
    plan = _materialization_plan(
        idea=idea,
        abi=abi,
        candidate_id=candidate_id,
        module_path=module_path,
        test_path=test_path,
        descriptor_path=descriptor_path,
        bundle_path=bundle_path,
        runtime_python=python,
        project_root=root,
        manufacturing_order=manufacturing_order,
        manufacturing_work_order_sha256=(
            str(input_lock["manufacturing_work_order_sha256"])
            if manufacturing_order is not None
            else None
        ),
    )
    plan_path = destination / "automatic-materialization-plan.json"
    _write_json(plan_path, plan)
    admission = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-admission",
        "state": "admitted_for_materialization",
        "candidate_id": candidate_id,
        "idea_id": idea["idea_id"],
        "abi_id": abi["abi_id"],
        "embodiment_class": spec["state"],
        "spec_digest": spec_digest,
        "input_digest": input_digest,
        "model_capability_ir_sha256": input_lock["model_capability_ir_sha256"],
        "abi_registry_digest": registry["registry_digest"],
        "llm_task_receipt_path": task_manifest["receipt_path"],
        "llm_task_receipt_sha256": task_manifest["receipt_sha256"],
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path.read_bytes()),
        "materialization_plan_path": str(plan_path),
        "materialization_plan_sha256": _sha256(plan_path.read_bytes()),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": _sha256(evaluator.read_bytes()),
        "side_effects": {
            "source_mutated": False,
            "gpu_execution_started": False,
            "candidate_materialization_authority": True,
            "gpu_scheduling_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "Admission authorizes one isolated implementation transaction for a fixed ABI. "
            "Only the controller may issue a GPU lease and only the frozen verifier may settle evidence."
        ),
    }
    if manufacturing_order is not None and manufacturing_file is not None:
        admission.update(
            {
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_work_order_digest": manufacturing_order[
                    "work_order_digest"
                ],
                "manufacturing_work_order_path": str(manufacturing_file),
                "manufacturing_work_order_sha256": input_lock[
                    "manufacturing_work_order_sha256"
                ],
                "manufacturing_bindings": dict(manufacturing_order["bindings"]),
                "manufacturing_request": dict(
                    manufacturing_order["manufacturing_request"]
                ),
                "dependency_closure": dict(
                    manufacturing_order["dependency_closure"]
                ),
                "portfolio_entry_ids": list(
                    manufacturing_order["portfolio_entry_ids"]
                ),
                "evaluator_binding": dict(
                    manufacturing_order["evaluator_binding"]
                ),
                "protected_metrics": list(manufacturing_order["protected_metrics"]),
            }
        )
    _validate_document("automatic_module_admission", admission, root=root)
    admission_path = destination / "admission.json"
    _write_json(admission_path, admission)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-plan-manifest",
        "state": "ready_for_materialization",
        "candidate_id": candidate_id,
        "abi_id": abi["abi_id"],
        "input_digest": input_digest,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path.read_bytes()),
        "admission_path": str(admission_path),
        "admission_sha256": _sha256(admission_path.read_bytes()),
        "evaluator_path": str(evaluator),
        "llm_task_manifest": task_manifest,
        "capability_gap_path": None,
    }
    if manufacturing_order is not None and manufacturing_file is not None:
        manifest.update(
            {
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_work_order_path": str(manufacturing_file),
                "manufacturing_work_order_sha256": input_lock[
                    "manufacturing_work_order_sha256"
                ],
            }
        )
    _write_json(destination / "manifest.json", manifest)
    return manifest


def load_automatic_module_admission(
    path: Path, *, expected_sha256: str, project_root: Path
) -> dict[str, object]:
    source = _require_file(path, "AUTOMATIC_MODULE_ADMISSION_INVALID")
    if _sha256(source.read_bytes()) != expected_sha256:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ADMISSION_HASH_MISMATCH")
    admission = _load(source, "AUTOMATIC_MODULE_ADMISSION_INVALID")
    _validate_document("automatic_module_admission", admission, root=project_root)
    if admission.get("state") != "admitted_for_materialization":
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ADMISSION_STATE_INVALID")
    evaluator = _require_file(
        Path(str(admission["evaluator_path"])), "AUTOMATIC_MODULE_ABI_EVALUATOR_INVALID"
    )
    if _sha256(evaluator.read_bytes()) != admission.get("evaluator_sha256"):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_EVALUATOR_HASH_MISMATCH")
    plan = _require_file(
        Path(str(admission["materialization_plan_path"])),
        "AUTOMATIC_MODULE_MATERIALIZATION_PLAN_INVALID",
    )
    if _sha256(plan.read_bytes()) != admission.get("materialization_plan_sha256"):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_PLAN_HASH_MISMATCH")
    manufacturing_path = admission.get("manufacturing_work_order_path")
    if manufacturing_path is not None:
        required = (
            "manufacturing_mode",
            "manufacturing_work_order_id",
            "manufacturing_work_order_digest",
            "manufacturing_work_order_sha256",
            "manufacturing_bindings",
            "manufacturing_request",
            "dependency_closure",
            "portfolio_entry_ids",
            "evaluator_binding",
            "protected_metrics",
        )
        if any(admission.get(field) is None for field in required):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_BINDING_INCOMPLETE")
        try:
            manufacturing = load_module_manufacturing_work_order(
                Path(str(manufacturing_path)),
                expected_sha256=str(admission["manufacturing_work_order_sha256"]),
                root=project_root,
            )
        except ModuleManufacturingError as exc:
            raise AutomaticModulePlanError(
                f"AUTOMATIC_MODULE_MANUFACTURING_WORK_ORDER_INVALID:{exc}"
            ) from exc
        expected = {
            "manufacturing_mode": manufacturing["manufacturing_mode"],
            "manufacturing_work_order_id": manufacturing["work_order_id"],
            "manufacturing_work_order_digest": manufacturing["work_order_digest"],
            "manufacturing_bindings": manufacturing["bindings"],
            "manufacturing_request": manufacturing["manufacturing_request"],
            "dependency_closure": manufacturing["dependency_closure"],
            "portfolio_entry_ids": manufacturing["portfolio_entry_ids"],
            "evaluator_binding": manufacturing["evaluator_binding"],
            "protected_metrics": manufacturing["protected_metrics"],
        }
        if any(admission.get(field) != value for field, value in expected.items()):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_BINDING_MISMATCH")
        if (
            manufacturing["target_abi"]["abi_id"] != admission.get("abi_id")
            or manufacturing["manufacturing_mode"] != "intervention"
        ):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_MISMATCH")
    return admission


def _module_request(
    *,
    idea: Mapping[str, object],
    work_order: Mapping[str, object],
    assessment: Mapping[str, object],
    capability: Mapping[str, object],
    abi_candidates: Sequence[Mapping[str, object]],
    registry: Mapping[str, object],
    manufacturing_order: Mapping[str, object] | None,
) -> dict[str, object]:
    seed = {
        "idea_id": idea["idea_id"],
        "source": idea.get("source"),
        "capability_id": capability["capability_id"],
        "registry_digest": registry["registry_digest"],
        "manufacturing_work_order_digest": (
            manufacturing_order["work_order_digest"]
            if manufacturing_order is not None
            else None
        ),
    }
    task_id = f"module-{_sha256(_canonical_json(seed))[:24]}"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task",
        "task_id": task_id,
        "task_type": "module_generation",
        "prompt_template_digest": _PROMPT_TEMPLATE_DIGEST,
        "output_schema": "automatic_module_spec",
        "input": {
            "instructions": _PROMPT_TEMPLATE,
            "idea": dict(idea),
            "work_order": dict(work_order),
            "source_assessment": dict(assessment),
            "model_capability_ir": dict(capability),
            "abi_candidates": [dict(row) for row in abi_candidates],
            "module_manufacturing_work_order": (
                dict(manufacturing_order) if manufacturing_order is not None else None
            ),
        },
    }


def _abi_for_prompt(
    abi: Mapping[str, object], *, capability: Mapping[str, object], idea: Mapping[str, object]
) -> dict[str, object]:
    blockers = _abi_capability_blockers(abi, capability=capability, idea=idea)
    return {
        "abi_id": abi["abi_id"],
        "eligible": not blockers,
        "eligibility_blockers": blockers,
        "required_symbol": abi["required_symbol"],
        "positional_parameters": list(abi["positional_parameters"]),
        "keyword_parameters": list(abi["keyword_parameters"]),
        "required_model_capabilities": list(abi["required_model_capabilities"]),
        "required_hooks": list(abi["required_hooks"]),
        "implementation_contract": dict(abi["implementation_contract"]),
        "allowed_imports": list(abi["allowed_imports"]),
        "maximum_source_bytes": abi["maximum_source_bytes"],
        "fixed_parameters": dict(abi["candidate_parameters"]),
    }


def _selected_abi(
    *,
    spec: Mapping[str, object],
    registry: Mapping[str, object],
    capability: Mapping[str, object],
    idea: Mapping[str, object],
) -> dict[str, object]:
    abi_id = spec.get("abi_id")
    rows = [
        dict(row)
        for row in registry["abis"]
        if isinstance(row, Mapping) and row.get("abi_id") == abi_id
    ]
    if len(rows) != 1:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ABI_UNKNOWN")
    abi = rows[0]
    blockers = _abi_capability_blockers(abi, capability=capability, idea=idea)
    if blockers:
        raise AutomaticModulePlanError(
            "AUTOMATIC_MODULE_ABI_NOT_ELIGIBLE:" + ",".join(str(row["code"]) for row in blockers)
        )
    return abi


def _abi_capability_blockers(
    abi: Mapping[str, object], *, capability: Mapping[str, object], idea: Mapping[str, object]
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if abi.get("model_family") != capability.get("model_family"):
        blockers.append({"code": "MODEL_FAMILY_MISMATCH"})
    available = {
        str(row["capability"])
        for row in capability.get("capabilities", [])
        if isinstance(row, Mapping) and row.get("state") == "available"
    }
    missing = sorted(set(str(value) for value in abi["required_model_capabilities"]) - available)
    aliases = abi.get("capability_aliases")
    alias_map = dict(aliases) if isinstance(aliases, Mapping) else {}
    contract = idea.get("materialization_contract")
    required_target = (
        contract.get("required_target_capabilities", []) if isinstance(contract, Mapping) else []
    )
    target_missing = sorted(
        str(value)
        for value in required_target
        if str(alias_map.get(str(value), str(value))) not in available
    )
    if missing:
        blockers.append({"code": "MODEL_CAPABILITY_MISSING", "missing": missing})
    if target_missing:
        blockers.append({"code": "IDEA_TARGET_CAPABILITY_MISSING", "missing": target_missing})
    evaluator = capability.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("state") != "ready":
        blockers.append({"code": "MODEL_EVALUATOR_NOT_READY"})
    return blockers


def _validate_module_source(source: str, *, abi: Mapping[str, object]) -> None:
    if not source.strip() or len(source.encode("utf-8")) > int(abi["maximum_source_bytes"]):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_SOURCE_SIZE_INVALID")
    if re.search(r"\bcuda\b", source, flags=re.IGNORECASE):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_GPU_SURFACE_FORBIDDEN")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_SOURCE_SYNTAX_INVALID") from exc
    allowed_imports = set(str(value) for value in abi["allowed_imports"])
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise AutomaticModulePlanError("AUTOMATIC_MODULE_RELATIVE_IMPORT_FORBIDDEN")
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".", 1)[0] not in allowed_imports for name in names):
                raise AutomaticModulePlanError("AUTOMATIC_MODULE_IMPORT_FORBIDDEN")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BAD_CALL_NAMES:
                raise AutomaticModulePlanError("AUTOMATIC_MODULE_CALL_FORBIDDEN")
            if isinstance(node.func, ast.Attribute):
                root = _attribute_root(node.func)
                if (
                    root in _BAD_ATTRIBUTE_ROOTS
                    or node.func.attr in _BAD_ATTRIBUTE_NAMES
                    or node.func.attr not in _ALLOWED_ATTRIBUTE_CALLS
                ):
                    raise AutomaticModulePlanError("AUTOMATIC_MODULE_CALL_FORBIDDEN")
        if isinstance(node, ast.Attribute) and (
            _attribute_root(node) in _BAD_ATTRIBUTE_ROOTS
            or node.attr in _BAD_ATTRIBUTE_NAMES
            or node.attr.startswith("__")
        ):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_ATTRIBUTE_FORBIDDEN")
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Global,
                ast.Lambda,
                ast.Nonlocal,
                ast.While,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_LANGUAGE_SURFACE_FORBIDDEN")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_RESOURCE_EXPRESSION_FORBIDDEN")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and abs(float(node.value)) > 1_000_000
        ):
            raise AutomaticModulePlanError("AUTOMATIC_MODULE_NUMERIC_LITERAL_TOO_LARGE")
    allowed_top = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    if any(
        not isinstance(node, allowed_top)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        for node in tree.body
    ):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_TOP_LEVEL_EFFECT_FORBIDDEN")
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == abi["required_symbol"]
    ]
    if len(functions) != 1:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_REQUIRED_SYMBOL_INVALID")
    function = functions[0]
    positional = [argument.arg for argument in function.args.args]
    keyword = [argument.arg for argument in function.args.kwonlyargs]
    if (
        positional != list(abi["positional_parameters"])
        or keyword != list(abi["keyword_parameters"])
        or function.args.vararg is not None
        or function.args.kwarg is not None
    ):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_REQUIRED_SIGNATURE_INVALID")


def _component_touchpoints(
    spec: Mapping[str, object], *, preserved: Sequence[str]
) -> list[dict[str, object]]:
    rows = spec.get("component_touchpoints")
    assert isinstance(rows, list)
    mapped = [str(row.get("source_component_id")) for row in rows if isinstance(row, Mapping)]
    if sorted(mapped) != sorted(preserved) or len(mapped) != len(set(mapped)):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_COMPONENT_MAPPING_INVALID")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _materialization_plan(
    *,
    idea: Mapping[str, object],
    abi: Mapping[str, object],
    candidate_id: str,
    module_path: str,
    test_path: str,
    descriptor_path: str,
    bundle_path: Path,
    runtime_python: Path,
    project_root: Path,
    manufacturing_order: Mapping[str, object] | None,
    manufacturing_work_order_sha256: str | None,
) -> dict[str, object]:
    applier = project_root / "wmloop" / "execute" / "generated_module_bundle.py"
    _require_file(applier, "AUTOMATIC_MODULE_BUNDLE_APPLIER_INVALID")
    test_node = test_path.replace("\\", "/")
    plan = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-materialization-plan",
        "plan_id": f"automatic-{abi['abi_id']}-{candidate_id[-12:]}",
        "plan_digest": "",
        "idea_id": idea["idea_id"],
        "candidate_id": candidate_id,
        "model_family": abi["model_family"],
        "required_hooks": list(abi["required_hooks"]),
        "estimated_gpu_hours": abi["estimated_gpu_hours"],
        "descriptor_path": descriptor_path,
        "allowed_changed_paths": [module_path, test_path, descriptor_path],
        "forbidden_changed_paths": list(abi["forbidden_changed_paths"]),
        "source_overlay": {
            "include_untracked_globs": [
                "config.py",
                "configs/*.json",
                "models/*.py",
                "scripts/*.py",
                "tests/*.py",
            ],
            "max_file_bytes": 16777216,
            "max_total_bytes": 67108864,
        },
        "agent_command": [
            sys.executable,
            str(applier),
            "--bundle",
            str(bundle_path),
            "--workspace",
            "{workspace}",
            "--descriptor-path",
            "{descriptor_path}",
            "--candidate-id",
            "{candidate_id}",
            "--idea-id",
            "{idea_id}",
        ],
        "agent_timeout_seconds": 60,
        "fixed_checks": [
            {
                "label": "static",
                "argv": [str(runtime_python), "-m", "py_compile", module_path, test_path],
                "timeout_seconds": 120,
            },
            {
                "label": "semantic",
                "argv": [
                    str(runtime_python),
                    "-m",
                    "pytest",
                    "-q",
                    f"{test_node}::test_history_selection_abi_contract",
                ],
                "timeout_seconds": 180,
            },
            {
                "label": "negative",
                "argv": [
                    str(runtime_python),
                    "-m",
                    "pytest",
                    "-q",
                    f"{test_node}::test_history_selection_invalid_inputs_fail_closed",
                ],
                "timeout_seconds": 180,
            },
            {
                "label": "integration",
                "argv": [str(runtime_python), "-m", "pytest", "-q", test_node],
                "timeout_seconds": 300,
            },
        ],
        "inherit_environment_keys": [],
        "preserve_codex_auth": False,
        "candidate_template": {
            "candidate_id": candidate_id,
            "candidate_kind": abi["candidate_kind"],
            "parameters": dict(abi["candidate_parameters"]),
        },
        "implementation_contract": {
            **dict(abi["implementation_contract"]),
            "abi_id": abi["abi_id"],
            "module": module_path,
            "required_symbol": abi["required_symbol"],
        },
    }
    plan["plan_digest"] = materialization_plan_digest(plan)
    if manufacturing_order is not None:
        plan.update(
            {
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_work_order_digest": manufacturing_order[
                    "work_order_digest"
                ],
                "manufacturing_work_order_sha256": manufacturing_work_order_sha256,
                "manufacturing_request_id": manufacturing_order[
                    "manufacturing_request"
                ]["request_id"],
                "manufacturing_abi_id": manufacturing_order["target_abi"]["abi_id"],
                "dependency_abi_ids": list(
                    manufacturing_order["manufacturing_request"]["dependency_abi_ids"]
                ),
                "portfolio_entry_ids": list(
                    manufacturing_order["portfolio_entry_ids"]
                ),
                "protected_metrics": list(manufacturing_order["protected_metrics"]),
                "evaluator_binding": dict(
                    manufacturing_order["evaluator_binding"]
                ),
            }
        )
        plan["plan_digest"] = materialization_plan_digest(plan)
    return plan


def _validate_registry(registry: Mapping[str, object], *, root: Path) -> None:
    if registry.get("schema_version") == 2:
        try:
            validate_module_abi_registry(registry, root=root)
        except ModuleCompositionError as exc:
            raise AutomaticModulePlanError(
                f"AUTOMATIC_MODULE_ABI_REGISTRY_INVALID:{exc}"
            ) from exc
        return
    _validate_document("automatic_module_abi_registry", registry, root=root)
    if registry.get("registry_digest") != abi_registry_digest(registry):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ABI_REGISTRY_DIGEST_MISMATCH")
    rows = registry.get("abis")
    assert isinstance(rows, list)
    ids = [str(row["abi_id"]) for row in rows if isinstance(row, Mapping)]
    if len(ids) != len(rows) or len(ids) != len(set(ids)):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ABI_DUPLICATE")


def _write_gap(
    destination: Path,
    *,
    input_digest: str,
    idea: Mapping[str, object],
    state: str,
    blockers: Sequence[Mapping[str, object]],
    task_manifest: Mapping[str, object],
) -> dict[str, object]:
    gap = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-capability-gap",
        "state": state,
        "idea_id": idea["idea_id"],
        "input_digest": input_digest,
        "blockers": [dict(row) for row in blockers],
        "llm_task_receipt_path": task_manifest.get("receipt_path"),
        "side_effects": {
            "source_mutated": False,
            "gpu_execution_started": False,
            "candidate_materialization_authority": False,
            "gpu_scheduling_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "This gap is an executable boundary, not a substitute implementation. "
            "It grants no materialization, GPU, evaluator, or promotion authority."
        ),
    }
    gap_path = destination / "capability-gap.json"
    _write_json(gap_path, gap)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-plan-manifest",
        "state": "capability_gap",
        "candidate_id": None,
        "abi_id": None,
        "input_digest": input_digest,
        "plan_path": None,
        "plan_sha256": None,
        "admission_path": None,
        "admission_sha256": None,
        "evaluator_path": None,
        "llm_task_manifest": dict(task_manifest),
        "capability_gap_path": str(gap_path),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _preserved_components(idea: Mapping[str, object]) -> list[str]:
    contract = idea.get("materialization_contract")
    raw = contract.get("preserved_components") if isinstance(contract, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_PRESERVED_COMPONENTS_MISSING")
    values = [str(value) for value in raw]
    if len(values) != len(set(values)) or any(not value for value in values):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_PRESERVED_COMPONENTS_INVALID")
    return values


def _candidate_id(
    idea_id: str,
    *,
    spec_digest: str,
    manufacturing_work_order_digest: str | None = None,
) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", idea_id).strip("-._") or "candidate"
    identity = spec_digest
    if manufacturing_work_order_digest is not None:
        identity = _sha256(
            _canonical_json(
                {
                    "spec_digest": spec_digest,
                    "manufacturing_work_order_digest": manufacturing_work_order_digest,
                }
            )
        )
    return f"auto-{slug[:88]}-{identity[:12]}"[:127]


def _validate_manufacturing_registry_binding(
    order: Mapping[str, object], *, registry: Mapping[str, object]
) -> None:
    if order.get("manufacturing_mode") != "intervention":
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_MODE_UNSUPPORTED")
    target = order.get("target_abi")
    if not isinstance(target, Mapping):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_INVALID")
    if (
        target.get("registry_id") != registry.get("registry_id")
        or target.get("registry_digest") != registry.get("registry_digest")
    ):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_REGISTRY_DRIFT")
    rows = [
        row
        for row in registry["abis"]
        if isinstance(row, Mapping) and row.get("abi_id") == target.get("abi_id")
    ]
    if len(rows) != 1:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_UNAVAILABLE")
    abi = rows[0]
    if (
        abi.get("abi_version") != target.get("abi_version")
        or abi.get("abi_digest") != target.get("abi_digest")
    ):
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_MANUFACTURING_ABI_DRIFT")


def _repository_path(raw: object, code: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise AutomaticModulePlanError(code)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or str(path) != raw:
        raise AutomaticModulePlanError(code)
    return raw


def _resolve_project_file(root: Path, raw: object, code: str) -> Path:
    relative = _repository_path(raw, code)
    path = (root / relative).resolve()
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise AutomaticModulePlanError(code)
    return path


def _adapter_identity(adapter: Mapping[str, object]) -> dict[str, object]:
    allowed = {
        "command",
        "provider_alias",
        "model_alias",
        "timeout_seconds",
        "max_output_bytes",
        "credential_environment_keys",
    }
    if set(adapter) != allowed:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_ADAPTER_CONFIG_INVALID")
    return {key: adapter[key] for key in sorted(allowed)}


def _attribute_root(node: ast.Attribute) -> str | None:
    value: ast.expr = node
    while isinstance(value, ast.Attribute):
        value = value.value
    return value.id if isinstance(value, ast.Name) else None


def _document_digest(document: Mapping[str, object], *, excluded: str) -> str:
    return _sha256(_canonical_json({key: value for key, value in document.items() if key != excluded}))


def _resume(destination: Path, *, input_digest: str) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_OUTPUT_INVALID")
    lock = _load(destination / "input-lock.json", "AUTOMATIC_MODULE_INPUT_LOCK_INVALID")
    if lock.get("input_digest") != input_digest:
        raise AutomaticModulePlanError("AUTOMATIC_MODULE_INPUT_LOCK_MISMATCH")
    return _load(destination / "manifest.json", "AUTOMATIC_MODULE_MANIFEST_INVALID")


def _validate_document(name: str, document: Mapping[str, object], *, root: Path) -> None:
    try:
        validate_document(name, document, root=root)
    except ContractValidationError as exc:
        raise AutomaticModulePlanError(f"AUTOMATIC_MODULE_CONTRACT_INVALID:{name}:{exc}") from exc


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutomaticModulePlanError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise AutomaticModulePlanError(code)
    return resolved


def _require_runtime_file(path: Path, code: str) -> Path:
    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise AutomaticModulePlanError(code)
    return source


def _load(path: Path, code: str) -> dict[str, object]:
    source = _require_file(path, code)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomaticModulePlanError(code) from exc
    if not isinstance(payload, dict):
        raise AutomaticModulePlanError(code)
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(body, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
