"""Compile portrait-bound module manufacturing work orders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adaptive_observation import (
    AdaptiveObservationError,
    validate_adaptive_probe_plan,
)
from wmloop.control.capability_gap_planner import (
    CapabilityGapPlannerError,
    validate_capability_requirement_graph,
    validate_gap_plan_against_requirement_graph,
    validate_goal_ir,
)
from wmloop.control.experiment_portfolio import (
    ExperimentPortfolioError,
    validate_experiment_portfolio,
)
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.control.module_composition import (
    ModuleCompositionError,
    compose_module_abis,
    load_module_abi_registry,
    module_composition_receipt_digest,
)
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class ModuleManufacturingError(RuntimeError):
    """A module manufacturing order crossed an evidence or authority boundary."""


def build_intervention_manufacturing_work_order(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    requirement_graph: Mapping[str, object],
    gap_plan: Mapping[str, object],
    portfolio: Mapping[str, object],
    manufacturing_request_id: str,
    composition_receipt: Mapping[str, object],
    abi_registry_path: Path,
    root: Path | None = None,
) -> dict[str, object]:
    """Authorize one isolated implementation for one exact intervention ABI leaf."""

    _validate_intervention_inputs(
        goal_ir=goal_ir,
        portrait=portrait,
        requirement_graph=requirement_graph,
        gap_plan=gap_plan,
        portfolio=portfolio,
        root=root,
    )
    request = _selected_request(gap_plan, manufacturing_request_id)
    abi_id = request.get("abi_id")
    if not isinstance(abi_id, str) or not abi_id:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_INTERFACE_EXTENSION_REQUIRED")
    registry = load_module_abi_registry(abi_registry_path, root=root)
    if (
        registry["registry_id"] != gap_plan["registry_id"]
        or registry["registry_digest"] != gap_plan["registry_digest"]
        or registry["registry_id"] != requirement_graph["registry_id"]
        or registry["registry_digest"] != requirement_graph["registry_digest"]
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REGISTRY_BINDING_MISMATCH")
    composition = _validated_composition(
        composition_receipt,
        registry_path=abi_registry_path,
        expected_id=_composition_id(requirement_graph, request=request),
        expected_digest=_composition_digest(requirement_graph, request=request),
        root=root,
    )
    selected_modules = composition["selected_modules"]
    assert isinstance(selected_modules, list)
    target_rows = [row for row in selected_modules if row.get("abi_id") == abi_id]
    if len(target_rows) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_TARGET_ABI_NOT_IN_CLOSURE")
    target = target_rows[0]
    if (
        request.get("abi_version") != target.get("abi_version")
        or request.get("abi_digest") != target.get("abi_digest")
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_TARGET_ABI_DRIFT")
    direct_dependencies = sorted(
        str(row["abi_id"])
        for row in selected_modules
        if any(
            edge.get("consumer_module_id") == target["module_id"]
            and edge.get("provider_module_id") == row["module_id"]
            for edge in composition["dependency_edges"]
        )
    )
    if direct_dependencies != sorted(str(value) for value in request["dependency_abi_ids"]):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_DEPENDENCY_CLOSURE_MISMATCH")
    closure_module_ids = _target_dependency_module_ids(
        composition,
        target_module_id=str(target["module_id"]),
    )
    closure_modules = [
        row
        for row in selected_modules
        if str(row["module_id"]) in closure_module_ids
    ]
    node = _requirement_node(requirement_graph, str(request["requirement_id"]))
    if (
        node.get("capability") != request.get("capability")
        or abi_id not in node["resolution"]["manufacturing_abi_ids"]
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REQUEST_GRAPH_MISMATCH")
    matching_entries = [
        row
        for row in portfolio["entries"]
        if row.get("candidate_id") is not None
        and request["capability"] in row["required_module_capabilities"]
    ]
    if not matching_entries:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_PORTFOLIO_ENTRY_MISSING")
    evaluator = _shared_mapping(
        matching_entries,
        "evaluator",
        "MODULE_MANUFACTURING_EVALUATOR_BINDING_MISMATCH",
    )
    protected_metrics = _shared_strings(
        matching_entries,
        "protected_metrics",
        "MODULE_MANUFACTURING_PROTECTED_METRIC_MISMATCH",
    )
    target_abi = _registry_abi(registry, abi_id)
    if (
        target_abi.get("abi_version") != target.get("abi_version")
        or target_abi.get("abi_digest") != target.get("abi_digest")
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REGISTRY_ABI_DRIFT")
    closure = {
        "modules": sorted(
            (
                {
                    "role": "target" if row["abi_id"] == abi_id else "dependency",
                    "module_id": row["module_id"],
                    "abi_id": row["abi_id"],
                    "abi_version": row["abi_version"],
                    "abi_digest": row["abi_digest"],
                    "provides": sorted(str(value) for value in row["provides"]),
                    "requires": sorted(str(value) for value in row["requires"]),
                    "admission_state": (
                        "implementation_required"
                        if row["abi_id"] == abi_id
                        else "admitted_dependency"
                    ),
                }
                for row in closure_modules
            ),
            key=lambda row: str(row["abi_id"]),
        ),
        "edges": sorted(
            (
                {
                    "consumer_module_id": row["consumer_module_id"],
                    "required_capability": row["required_capability"],
                    "provider_module_id": row["provider_module_id"],
                    "contract": row["contract"],
                }
                for row in composition["dependency_edges"]
                if str(row["consumer_module_id"]) in closure_module_ids
                and str(row["provider_module_id"]) in closure_module_ids
            ),
            key=lambda row: (
                str(row["consumer_module_id"]),
                str(row["required_capability"]),
                str(row["provider_module_id"]),
            ),
        ),
    }
    body = {
        "schema_version": 1,
        "artifact_type": "verdiwm-module-manufacturing-work-order",
        "manufacturing_mode": "intervention",
        "decision": "manufacture_registered_abi",
        "bindings": {
            "portrait_id": portrait["portrait_id"],
            "portrait_digest": _digest(portrait),
            "goal_binding": goal_ir["goal_binding"],
            "goal_ir_id": goal_ir["goal_ir_id"],
            "goal_ir_digest": _digest(goal_ir),
            "requirement_graph_id": requirement_graph["graph_id"],
            "requirement_graph_digest": requirement_graph["graph_digest"],
            "gap_plan_id": gap_plan["plan_id"],
            "gap_plan_digest": gap_plan["plan_digest"],
            "portfolio_id": portfolio["portfolio_id"],
            "portfolio_digest": portfolio["portfolio_digest"],
            "observation_plan_id": None,
            "observation_plan_digest": None,
            "observation_task_id": None,
        },
        "manufacturing_request": {
            **dict(request),
            "probe_coverage_key": None,
        },
        "target_abi": _intervention_target(registry, target_abi),
        "dependency_closure": closure,
        "portfolio_entry_ids": sorted(str(row["entry_id"]) for row in matching_entries),
        "evaluator_binding": evaluator,
        "protected_metrics": protected_metrics,
        "authority": _authority("implementation_only", isolated_workspace_write=True),
        "artifact_policy": {
            "archive_policy": "archive_all_terminal",
            "cleanup_policy": "after_content_addressed_receipt",
        },
        "claim_boundary": (
            "This order authorizes one isolated implementation of one exact ABI leaf. "
            "It cannot choose paths, evaluators, splits, metrics, budgets, GPU work, or promotion."
        ),
    }
    return _finalize(body, root=root)


def build_observation_manufacturing_work_order(
    *,
    portrait: Mapping[str, object],
    adaptive_probe_plan: Mapping[str, object],
    observation_task_id: str,
    observation_registry: Mapping[str, object],
    root: Path | None = None,
) -> dict[str, object]:
    """Authorize one read-only, shadow-only observation module implementation."""

    try:
        validate_model_portrait(portrait, root=root)
        validate_adaptive_probe_plan(adaptive_probe_plan, root=root)
        validate_document("observation_module_abi_registry", observation_registry, root=root)
    except (ModelPortraitError, AdaptiveObservationError, ContractValidationError) as exc:
        raise ModuleManufacturingError(
            f"MODULE_MANUFACTURING_OBSERVATION_INPUT_INVALID:{exc}"
        ) from exc
    registry_body = dict(observation_registry)
    received_registry_digest = registry_body.pop("registry_digest", None)
    if received_registry_digest != _digest(registry_body):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_REGISTRY_DRIFT")
    if adaptive_probe_plan.get("portrait_id") != portrait.get("portrait_id"):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_PORTRAIT_MISMATCH")
    tasks = adaptive_probe_plan["tasks"]
    assert isinstance(tasks, list)
    selected = [row for row in tasks if row.get("task_id") == observation_task_id]
    if len(selected) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_TASK_UNKNOWN")
    task = selected[0]
    if (
        task.get("task_type") != "manufacture_shadow_probe"
        or task.get("execution_authority") != "shadow_only"
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_TASK_NOT_MANUFACTURABLE")
    abi_id = task.get("abi_id")
    rows = [row for row in observation_registry["abis"] if row.get("abi_id") == abi_id]
    if len(rows) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_ABI_UNKNOWN")
    abi = rows[0]
    if (
        abi.get("admission_state") != "shadow_template"
        or abi.get("execution_mode") != "shadow_via_controller"
        or abi.get("verdict_exposure_allowed") is not False
        or abi.get("active_metric_mutation_allowed") is not False
        or abi.get("active_evaluator_mutation_allowed") is not False
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_AUTHORITY_INVALID")
    abi_digest = _digest(abi)
    target = {
        "registry_id": observation_registry["registry_id"],
        "registry_digest": observation_registry["registry_digest"],
        "module_id": abi["abi_id"],
        "abi_id": abi["abi_id"],
        "abi_version": None,
        "abi_digest": abi_digest,
        "provides": sorted(str(value) for value in abi["provides"]),
        "requires": sorted(str(value) for value in abi["requires"]),
        "side_effect_class": abi["side_effect_class"],
        "authority_level": abi["authority_level"],
        "execution_mode": abi["execution_mode"],
    }
    capability = sorted(str(value) for value in abi["provides"])[0]
    body = {
        "schema_version": 1,
        "artifact_type": "verdiwm-module-manufacturing-work-order",
        "manufacturing_mode": "observation",
        "decision": "manufacture_registered_abi",
        "bindings": {
            "portrait_id": portrait["portrait_id"],
            "portrait_digest": _digest(portrait),
            "goal_binding": adaptive_probe_plan["goal_binding"],
            "goal_ir_id": None,
            "goal_ir_digest": None,
            "requirement_graph_id": None,
            "requirement_graph_digest": None,
            "gap_plan_id": None,
            "gap_plan_digest": None,
            "portfolio_id": None,
            "portfolio_digest": None,
            "observation_plan_id": adaptive_probe_plan["plan_id"],
            "observation_plan_digest": _digest(adaptive_probe_plan),
            "observation_task_id": task["task_id"],
        },
        "manufacturing_request": {
            "request_id": task["task_id"],
            "requirement_id": None,
            "capability": capability,
            "abi_id": abi["abi_id"],
            "abi_version": None,
            "abi_digest": abi_digest,
            "reason": task["reason"],
            "dependency_abi_ids": [],
            "external_ports": [],
            "probe_coverage_key": task["probe_coverage_key"],
        },
        "target_abi": target,
        "dependency_closure": {
            "modules": [
                {
                    "role": "target",
                    "module_id": abi["abi_id"],
                    "abi_id": abi["abi_id"],
                    "abi_version": None,
                    "abi_digest": abi_digest,
                    "provides": target["provides"],
                    "requires": target["requires"],
                    "admission_state": "shadow_template",
                }
            ],
            "edges": [],
        },
        "portfolio_entry_ids": [],
        "evaluator_binding": None,
        "protected_metrics": [],
        "authority": _authority("shadow_only", isolated_workspace_write=True),
        "artifact_policy": {
            "archive_policy": "archive_all_terminal",
            "cleanup_policy": "after_content_addressed_receipt",
        },
        "claim_boundary": (
            "This order authorizes one read-only shadow observation candidate. It cannot "
            "register active metrics, alter evaluators, expose verdicts, intervene, or schedule GPUs."
        ),
    }
    return _finalize(body, root=root)


def validate_module_manufacturing_work_order(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate content identity and mode-specific manufacturing authority."""

    try:
        reject_runtime_bindings(document)
        validate_document("module_manufacturing_work_order", document, root=root)
    except (GeometryValidationError, ContractValidationError) as exc:
        raise ModuleManufacturingError(f"MODULE_MANUFACTURING_WORK_ORDER_INVALID:{exc}") from exc
    body = dict(document)
    received_digest = body.pop("work_order_digest", None)
    if received_digest != _digest(body):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_DIGEST_MISMATCH")
    identity = dict(body)
    received_id = identity.pop("work_order_id", None)
    if received_id != _stable_id("module-manufacturing", identity):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_ID_MISMATCH")
    bindings = document["bindings"]
    request = document["manufacturing_request"]
    target = document["target_abi"]
    closure = document["dependency_closure"]
    authority = document["authority"]
    assert all(isinstance(row, Mapping) for row in (bindings, request, target, closure, authority))
    modules = closure["modules"]
    edges = closure["edges"]
    assert isinstance(modules, list) and isinstance(edges, list)
    abi_ids = [str(row["abi_id"]) for row in modules]
    if len(abi_ids) != len(set(abi_ids)):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_CLOSURE_DUPLICATE")
    targets = [row for row in modules if row["role"] == "target"]
    if len(targets) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_CLOSURE_TARGET_INVALID")
    target_row = targets[0]
    for field in ("module_id", "abi_id", "abi_version", "abi_digest", "provides", "requires"):
        if target_row[field] != target[field]:
            raise ModuleManufacturingError("MODULE_MANUFACTURING_TARGET_CLOSURE_MISMATCH")
    for field in ("abi_id", "abi_version", "abi_digest"):
        if request[field] != target[field]:
            raise ModuleManufacturingError("MODULE_MANUFACTURING_REQUEST_TARGET_MISMATCH")
    module_ids = {str(row["module_id"]) for row in modules}
    if any(
        str(row["consumer_module_id"]) not in module_ids
        or str(row["provider_module_id"]) not in module_ids
        for row in edges
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_CLOSURE_EDGE_INVALID")
    _unique_strings(document["portfolio_entry_ids"], "MODULE_MANUFACTURING_PORTFOLIO_ENTRY_DUPLICATE")
    _unique_strings(document["protected_metrics"], "MODULE_MANUFACTURING_PROTECTED_METRIC_DUPLICATE")
    mode = document["manufacturing_mode"]
    expected_false = {
        "target_tree_mutation": False,
        "evaluator_selection": False,
        "split_selection": False,
        "metric_selection": False,
        "budget_selection": False,
        "gpu_scheduling": False,
        "promotion": False,
        "verdict_exposure": False,
    }
    if any(authority.get(key) is not value for key, value in expected_false.items()):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_AUTHORITY_INVALID")
    if authority.get("isolated_workspace_write") is not True:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_ISOLATION_REQUIRED")
    if mode == "observation":
        nullable = (
            "goal_ir_id",
            "goal_ir_digest",
            "requirement_graph_id",
            "requirement_graph_digest",
            "gap_plan_id",
            "gap_plan_digest",
            "portfolio_id",
            "portfolio_digest",
        )
        if any(bindings.get(key) is not None for key in nullable):
            raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_BINDING_INVALID")
        if (
            not isinstance(bindings.get("observation_plan_id"), str)
            or not isinstance(bindings.get("observation_plan_digest"), str)
            or not isinstance(bindings.get("observation_task_id"), str)
            or document["portfolio_entry_ids"]
            or document["evaluator_binding"] is not None
            or document["protected_metrics"]
            or authority.get("execution_authority") not in {"read_only", "shadow_only"}
            or target_row["admission_state"] != "shadow_template"
        ):
            raise ModuleManufacturingError("MODULE_MANUFACTURING_OBSERVATION_POLICY_INVALID")
    else:
        required = (
            "goal_ir_id",
            "goal_ir_digest",
            "requirement_graph_id",
            "requirement_graph_digest",
            "gap_plan_id",
            "gap_plan_digest",
            "portfolio_id",
            "portfolio_digest",
        )
        if (
            any(not isinstance(bindings.get(key), str) for key in required)
            or bindings.get("observation_plan_id") is not None
            or bindings.get("observation_plan_digest") is not None
            or bindings.get("observation_task_id") is not None
            or not document["portfolio_entry_ids"]
            or not isinstance(document["evaluator_binding"], Mapping)
            or not document["protected_metrics"]
            or authority.get("execution_authority") != "implementation_only"
            or target_row["admission_state"] != "implementation_required"
        ):
            raise ModuleManufacturingError("MODULE_MANUFACTURING_INTERVENTION_POLICY_INVALID")


def load_module_manufacturing_work_order(
    path: Path,
    *,
    expected_sha256: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_FILE_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_FILE_INVALID")
    if expected_sha256 is not None and hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_WORK_ORDER_HASH_MISMATCH")
    validate_module_manufacturing_work_order(payload, root=root)
    return payload


def _validate_intervention_inputs(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    requirement_graph: Mapping[str, object],
    gap_plan: Mapping[str, object],
    portfolio: Mapping[str, object],
    root: Path | None,
) -> None:
    try:
        validate_goal_ir(goal_ir, root=root)
        validate_model_portrait(portrait, root=root)
        validate_capability_requirement_graph(requirement_graph, root=root)
        validate_gap_plan_against_requirement_graph(gap_plan, requirement_graph, root=root)
        validate_experiment_portfolio(portfolio, root=root)
    except (
        CapabilityGapPlannerError,
        ModelPortraitError,
        ExperimentPortfolioError,
    ) as exc:
        raise ModuleManufacturingError(
            f"MODULE_MANUFACTURING_INTERVENTION_INPUT_INVALID:{exc}"
        ) from exc
    if (
        requirement_graph.get("goal_ir_id") != goal_ir.get("goal_ir_id")
        or requirement_graph.get("goal_ir_digest") != _digest(goal_ir)
        or gap_plan.get("goal_ir_id") != goal_ir.get("goal_ir_id")
        or gap_plan.get("goal_ir_digest") != _digest(goal_ir)
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_GOAL_BINDING_MISMATCH")
    if (
        requirement_graph.get("portrait_id") != portrait.get("portrait_id")
        or requirement_graph.get("portrait_digest") != _digest(portrait)
        or gap_plan.get("portrait_id") != portrait.get("portrait_id")
        or gap_plan.get("portrait_digest") != _digest(portrait)
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_PORTRAIT_BINDING_MISMATCH")
    bindings = portfolio["bindings"]
    assert isinstance(bindings, Mapping)
    expected = {
        "goal_ir_id": goal_ir["goal_ir_id"],
        "goal_ir_digest": _digest(goal_ir),
        "portrait_id": portrait["portrait_id"],
        "portrait_digest": _digest(portrait),
        "requirement_graph_id": requirement_graph["graph_id"],
        "requirement_graph_digest": requirement_graph["graph_digest"],
        "gap_plan_id": gap_plan["plan_id"],
        "gap_plan_digest": gap_plan["plan_digest"],
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_PORTFOLIO_BINDING_MISMATCH")
    if (
        gap_plan.get("state") != "requires_manufacturing"
        or portfolio.get("state") != "ready_for_resource_admission"
        or portfolio.get("next_action") != "manufacture_modules"
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_INTERVENTION_NOT_AUTHORIZED")


def _selected_request(
    gap_plan: Mapping[str, object], request_id: str
) -> Mapping[str, object]:
    if not isinstance(request_id, str) or not request_id:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REQUEST_ID_INVALID")
    requests = gap_plan["manufacturing_requests"]
    assert isinstance(requests, list)
    selected = [row for row in requests if row.get("request_id") == request_id]
    if len(selected) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REQUEST_UNKNOWN")
    return selected[0]


def _requirement_node(
    graph: Mapping[str, object], requirement_id: str
) -> Mapping[str, object]:
    nodes = graph["nodes"]
    assert isinstance(nodes, list)
    selected = [row for row in nodes if row.get("requirement_id") == requirement_id]
    if len(selected) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_REQUIREMENT_UNKNOWN")
    return selected[0]


def _composition_id(
    graph: Mapping[str, object], *, request: Mapping[str, object]
) -> str:
    node = _requirement_node(graph, str(request["requirement_id"]))
    value = node["resolution"].get("composition_id")
    if not isinstance(value, str):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_COMPOSITION_REQUIRED")
    return value


def _composition_digest(
    graph: Mapping[str, object], *, request: Mapping[str, object]
) -> str:
    node = _requirement_node(graph, str(request["requirement_id"]))
    value = node["resolution"].get("composition_digest")
    if not isinstance(value, str):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_COMPOSITION_REQUIRED")
    return value


def _validated_composition(
    receipt: Mapping[str, object],
    *,
    registry_path: Path,
    expected_id: str,
    expected_digest: str,
    root: Path | None,
) -> Mapping[str, object]:
    try:
        validate_document("module_composition_receipt", receipt, root=root)
    except ContractValidationError as exc:
        raise ModuleManufacturingError(
            f"MODULE_MANUFACTURING_COMPOSITION_INVALID:{exc}"
        ) from exc
    if (
        receipt.get("composition_id") != expected_id
        or receipt.get("receipt_digest") != expected_digest
        or receipt.get("receipt_digest") != module_composition_receipt_digest(receipt)
    ):
        raise ModuleManufacturingError("MODULE_MANUFACTURING_COMPOSITION_BINDING_MISMATCH")
    selected = receipt["selected_modules"]
    assert isinstance(selected, list)
    locks = {
        str(row["module_id"]): {
            "abi_id": str(row["abi_id"]),
            "abi_version": str(row["abi_version"]),
            "abi_digest": str(row["abi_digest"]),
        }
        for row in selected
    }
    external_ports = {
        f"{row['module_id']}.{row['input_port']}": str(row["contract"])
        for row in receipt["external_bindings"]
    }
    try:
        rebuilt = compose_module_abis(
            registry_path=registry_path,
            requested_capabilities=tuple(receipt["requested_capabilities"]),
            external_ports=external_ports,
            maximum_authority_level=str(receipt["maximum_authority_level"]),
            expected_registry_digest=str(receipt["registry_digest"]),
            eligible_abi_ids=tuple(receipt["eligible_abi_ids"]),
            abi_locks=locks,
            root=root,
        )
    except ModuleCompositionError as exc:
        raise ModuleManufacturingError(
            f"MODULE_MANUFACTURING_COMPOSITION_INVALID:{exc}"
        ) from exc
    if rebuilt != receipt:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_COMPOSITION_REBUILD_MISMATCH")
    return receipt


def _registry_abi(
    registry: Mapping[str, object], abi_id: str
) -> Mapping[str, object]:
    rows = registry["abis"]
    assert isinstance(rows, list)
    selected = [row for row in rows if row.get("abi_id") == abi_id]
    if len(selected) != 1:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_ABI_UNKNOWN")
    return selected[0]


def _target_dependency_module_ids(
    composition: Mapping[str, object], *, target_module_id: str
) -> set[str]:
    selected = composition["selected_modules"]
    edges = composition["dependency_edges"]
    assert isinstance(selected, list) and isinstance(edges, list)
    known = {str(row["module_id"]) for row in selected}
    if target_module_id not in known:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_TARGET_ABI_NOT_IN_CLOSURE")
    closure = {target_module_id}
    pending = [target_module_id]
    while pending:
        consumer = pending.pop()
        providers = sorted(
            str(row["provider_module_id"])
            for row in edges
            if row.get("consumer_module_id") == consumer
        )
        for provider in providers:
            if provider not in known:
                raise ModuleManufacturingError(
                    "MODULE_MANUFACTURING_DEPENDENCY_CLOSURE_MISMATCH"
                )
            if provider not in closure:
                closure.add(provider)
                pending.append(provider)
    return closure


def _intervention_target(
    registry: Mapping[str, object], abi: Mapping[str, object]
) -> dict[str, object]:
    return {
        "registry_id": registry["registry_id"],
        "registry_digest": registry["registry_digest"],
        "module_id": abi["module_id"],
        "abi_id": abi["abi_id"],
        "abi_version": abi["abi_version"],
        "abi_digest": abi["abi_digest"],
        "provides": sorted(str(value) for value in abi["provides"]),
        "requires": sorted(str(value) for value in abi["requires"]),
        "side_effect_class": abi["side_effect_class"],
        "authority_level": abi["authority_level"],
        "execution_mode": None,
    }


def _shared_mapping(
    rows: Sequence[Mapping[str, object]], field: str, code: str
) -> dict[str, object]:
    values = [row[field] for row in rows]
    if not all(isinstance(value, Mapping) for value in values):
        raise ModuleManufacturingError(code)
    canonical = {_canonical_json(value) for value in values}
    if len(canonical) != 1:
        raise ModuleManufacturingError(code)
    return dict(values[0])


def _shared_strings(
    rows: Sequence[Mapping[str, object]], field: str, code: str
) -> list[str]:
    values = [tuple(sorted(str(value) for value in row[field])) for row in rows]
    if len(set(values)) != 1 or not values[0]:
        raise ModuleManufacturingError(code)
    return list(values[0])


def _authority(execution: str, *, isolated_workspace_write: bool) -> dict[str, object]:
    return {
        "execution_authority": execution,
        "isolated_workspace_write": isolated_workspace_write,
        "target_tree_mutation": False,
        "evaluator_selection": False,
        "split_selection": False,
        "metric_selection": False,
        "budget_selection": False,
        "gpu_scheduling": False,
        "promotion": False,
        "verdict_exposure": False,
    }


def _finalize(body: Mapping[str, object], *, root: Path | None) -> dict[str, object]:
    work_order: dict[str, object] = {
        **body,
        "work_order_id": _stable_id("module-manufacturing", body),
    }
    work_order["work_order_digest"] = _digest(work_order)
    validate_module_manufacturing_work_order(work_order, root=root)
    return work_order


def _unique_strings(values: object, code: str) -> None:
    if not isinstance(values, list):
        raise ModuleManufacturingError(code)
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ModuleManufacturingError(code)


def _stable_id(prefix: str, value: Mapping[str, object]) -> str:
    return f"{prefix}-{_digest(value)[:24]}"


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ModuleManufacturingError("MODULE_MANUFACTURING_CANONICAL_INVALID") from exc
