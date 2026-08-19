"""Compile a portrait-bound goal into exact capability gaps and leaf work."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.control.module_composition import (
    ModuleCompositionError,
    compose_module_abis,
    load_module_abi_registry,
    module_composition_receipt_digest,
)
from wmloop.geometry.evidence_ir import is_content_addressed, reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class CapabilityGapPlannerError(RuntimeError):
    """A goal, portrait, registry, or gap decision failed closed."""


_CLASSIFICATIONS = (
    "satisfied",
    "reusable",
    "composable",
    "manufacturable",
    "interface_extension_required",
    "data_blocked",
    "architecture_bound",
)
_AUTHORITY_RANK = {"L0": 0, "L1": 1, "L2": 2}
_BLOCKER_PRECEDENCE = {
    "architecture_bound": 3,
    "data_blocked": 2,
    "interface_extension_required": 1,
}


def build_goal_ir(
    *,
    goal_id: str,
    goal_binding: str,
    model_family: str,
    objective: str,
    requirements: Sequence[Mapping[str, object]],
    root: Path | None = None,
) -> dict[str, object]:
    """Build a path-free, content-derived Goal IR."""

    for value, code in (
        (goal_id, "GOAL_IR_GOAL_ID_INVALID"),
        (model_family, "GOAL_IR_MODEL_FAMILY_INVALID"),
        (objective, "GOAL_IR_OBJECTIVE_INVALID"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CapabilityGapPlannerError(code)
    if not isinstance(goal_binding, str) or not is_content_addressed(goal_binding):
        raise CapabilityGapPlannerError("GOAL_IR_BINDING_INVALID")
    normalized = [_normalize_requirement(row) for row in requirements]
    if not normalized:
        raise CapabilityGapPlannerError("GOAL_IR_REQUIREMENT_REQUIRED")
    capabilities = [str(row["capability"]) for row in normalized]
    if len(capabilities) != len(set(capabilities)):
        raise CapabilityGapPlannerError("GOAL_IR_CAPABILITY_DUPLICATE")
    known = set(capabilities)
    for row in normalized:
        unknown = sorted(set(row["dependencies"]) - known)
        if unknown:
            raise CapabilityGapPlannerError(
                f"GOAL_IR_DEPENDENCY_UNKNOWN:{row['capability']}:{unknown[0]}"
            )
    _topological_requirements(normalized)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-goal-ir",
        "goal_id": goal_id.strip(),
        "goal_binding": goal_binding,
        "model_family": model_family.strip(),
        "objective": objective.strip(),
        "requirements": sorted(normalized, key=lambda row: str(row["capability"])),
    }
    body["goal_ir_id"] = _stable_id("goal-ir", body)
    validate_goal_ir(body, root=root)
    return body


def validate_goal_ir(document: Mapping[str, object], *, root: Path | None = None) -> None:
    try:
        reject_runtime_bindings(document)
        validate_document("goal_ir", document, root=root)
    except (GeometryValidationError, ContractValidationError) as exc:
        raise CapabilityGapPlannerError(f"GOAL_IR_INVALID:{exc}") from exc
    requirements = document.get("requirements")
    assert isinstance(requirements, list)
    capabilities = [str(row["capability"]) for row in requirements]
    if len(capabilities) != len(set(capabilities)):
        raise CapabilityGapPlannerError("GOAL_IR_CAPABILITY_DUPLICATE")
    for row in requirements:
        assert isinstance(row, Mapping)
        body = dict(row)
        received = body.pop("requirement_id", None)
        if received != _stable_id("requirement", body):
            raise CapabilityGapPlannerError("GOAL_IR_REQUIREMENT_ID_MISMATCH")
    _topological_requirements(requirements)
    body = dict(document)
    received = body.pop("goal_ir_id", None)
    if received != _stable_id("goal-ir", body):
        raise CapabilityGapPlannerError("GOAL_IR_ID_MISMATCH")


def compile_capability_gap_plan(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    abi_registry_path: Path,
    maximum_authority_level: str,
    admitted_abi_ids: Sequence[str] = (),
    manufacturable_capabilities: Sequence[str] = (),
    available_data_regimes: Sequence[str] = (),
    kernel_capabilities: Sequence[str] = (),
    portable_knowledge_graph: Mapping[str, object] | Path | None = None,
    expected_registry_digest: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Return a requirement graph, gap receipt, and compatible compositions."""

    validate_goal_ir(goal_ir, root=root)
    try:
        validate_model_portrait(portrait, root=root)
    except ModelPortraitError as exc:
        raise CapabilityGapPlannerError(f"CAPABILITY_GAP_PORTRAIT_INVALID:{exc}") from exc
    if maximum_authority_level not in _AUTHORITY_RANK:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_AUTHORITY_INVALID")
    registry = load_module_abi_registry(abi_registry_path, root=root)
    if (
        expected_registry_digest is not None
        and expected_registry_digest != registry["registry_digest"]
    ):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_REGISTRY_DRIFT")
    admitted = _string_set(admitted_abi_ids, "CAPABILITY_GAP_ADMITTED_ABI_INVALID")
    known_abis = {str(row["abi_id"]) for row in registry["abis"]}
    unknown_admitted = sorted(admitted - known_abis)
    if unknown_admitted:
        raise CapabilityGapPlannerError(
            f"CAPABILITY_GAP_ADMITTED_ABI_UNKNOWN:{unknown_admitted[0]}"
        )
    manufacturable = _string_set(
        manufacturable_capabilities,
        "CAPABILITY_GAP_MANUFACTURABLE_CAPABILITY_INVALID",
    )
    data_regimes = _string_set(
        available_data_regimes, "CAPABILITY_GAP_DATA_REGIME_INVALID"
    )
    kernel = _string_set(kernel_capabilities, "CAPABILITY_GAP_KERNEL_CAPABILITY_INVALID")
    knowledge, knowledge_digest = _load_knowledge_graph(portable_knowledge_graph)
    portrait_state = _portrait_state(portrait)
    eligible_abis = _eligible_abi_ids(
        registry,
        portrait_state=portrait_state,
        maximum_authority_level=maximum_authority_level,
    )
    requirements = goal_ir["requirements"]
    assert isinstance(requirements, list)
    ordered = _topological_requirements(requirements)
    node_ids = {
        str(row["capability"]): _stable_id(
            "capability-node",
            {
                "goal_ir_id": goal_ir["goal_ir_id"],
                "requirement_id": row["requirement_id"],
                "capability": row["capability"],
            },
        )
        for row in requirements
    }
    decisions: dict[str, dict[str, object]] = {}
    composition_receipts: list[dict[str, object]] = []
    manufacturing_requests: list[dict[str, object]] = []
    knowledge_priors: list[dict[str, object]] = []
    for requirement in ordered:
        capability = str(requirement["capability"])
        priors = _knowledge_priors(
            knowledge,
            requirement=requirement,
            model_family=str(goal_ir["model_family"]),
        )
        knowledge_priors.extend(priors)
        dependency_decisions = [decisions[str(value)] for value in requirement["dependencies"]]
        decision, composition, requests = _classify_requirement(
            requirement=requirement,
            dependency_decisions=dependency_decisions,
            portrait_state=portrait_state,
            registry=registry,
            registry_path=abi_registry_path,
            eligible_abi_ids=eligible_abis,
            admitted_abi_ids=admitted,
            manufacturable_capabilities=manufacturable,
            available_data_regimes=data_regimes,
            kernel_capabilities=kernel,
            maximum_authority_level=maximum_authority_level,
            ranking_prior_ids=[str(row["prior_id"]) for row in priors],
            root=root,
        )
        decision["node_id"] = node_ids[capability]
        decision["requirement_id"] = requirement["requirement_id"]
        decision["capability"] = capability
        decision["kind"] = requirement["kind"]
        decision["dependencies"] = [node_ids[str(value)] for value in requirement["dependencies"]]
        decisions[capability] = decision
        if composition is not None:
            composition_receipts.append(composition)
        manufacturing_requests.extend(requests)
    nodes = [decisions[str(row["capability"])] for row in requirements]
    nodes.sort(key=lambda row: str(row["node_id"]))
    edges = sorted(
        [
            {
                "source": node_ids[str(dependency)],
                "relation": "requires",
                "target": node_ids[str(row["capability"])],
            }
            for row in requirements
            for dependency in row["dependencies"]
        ],
        key=lambda row: (row["source"], row["target"]),
    )
    graph = _build_graph(
        goal_ir=goal_ir,
        portrait=portrait,
        registry=registry,
        nodes=nodes,
        edges=edges,
        root=root,
    )
    receipt = _build_gap_receipt(
        goal_ir=goal_ir,
        portrait=portrait,
        registry=registry,
        graph=graph,
        nodes=nodes,
        manufacturing_requests=manufacturing_requests,
        composition_receipts=composition_receipts,
        knowledge_digest=knowledge_digest,
        knowledge_priors=knowledge_priors,
        root=root,
    )
    return {
        "goal_ir": dict(goal_ir),
        "requirement_graph": graph,
        "gap_plan_receipt": receipt,
        "composition_receipts": composition_receipts,
    }


def validate_capability_requirement_graph(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate_path_free_schema("capability_requirement_graph", document, root=root)
    body = dict(document)
    received_digest = body.pop("graph_digest", None)
    if received_digest != _digest(body):
        raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_DIGEST_MISMATCH")
    identity = dict(body)
    received_id = identity.pop("graph_id", None)
    if received_id != _stable_id("capability-graph", identity):
        raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_ID_MISMATCH")
    nodes = document.get("nodes")
    assert isinstance(nodes, list)
    node_ids = {str(row["node_id"]) for row in nodes}
    if len(node_ids) != len(nodes):
        raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_NODE_DUPLICATE")
    _validate_node_dag(nodes)
    edges = document.get("edges")
    assert isinstance(edges, list)
    expected_edges = {
        (str(dependency), "requires", str(row["node_id"]))
        for row in nodes
        for dependency in row["dependencies"]
    }
    received_edges = {
        (str(row["source"]), str(row["relation"]), str(row["target"]))
        for row in edges
    }
    if len(received_edges) != len(edges) or received_edges != expected_edges:
        raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_EDGE_MISMATCH")


def validate_capability_gap_plan_receipt(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _validate_path_free_schema("capability_gap_plan_receipt", document, root=root)
    body = dict(document)
    received_digest = body.pop("plan_digest", None)
    if received_digest != _digest(body):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_DIGEST_MISMATCH")
    identity = dict(body)
    received_id = identity.pop("plan_id", None)
    if received_id != _stable_id("gap-plan", identity):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_ID_MISMATCH")
    counts = document["classification_counts"]
    assert isinstance(counts, Mapping)
    manufacturing = document["manufacturing_requests"]
    assert isinstance(manufacturing, list)
    if sum(int(counts[name]) for name in _CLASSIFICATIONS) < 1:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_CLASSIFICATION_EMPTY")
    request_ids = [str(row["request_id"]) for row in manufacturing]
    if len(request_ids) != len(set(request_ids)):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_REQUEST_DUPLICATE")
    manufacturing_classifications = int(counts["manufacturable"]) + int(
        counts["composable"]
    )
    if manufacturing and manufacturing_classifications == 0:
        raise CapabilityGapPlannerError(
            "CAPABILITY_GAP_PLAN_MANUFACTURING_CLASSIFICATION_MISSING"
        )
    if int(counts["manufacturable"]) > 0 and not manufacturing:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_MANUFACTURING_REQUEST_MISSING")
    expected_state, expected_action = _gap_state(
        {name: int(counts[name]) for name in _CLASSIFICATIONS},
        bool(manufacturing),
    )
    if document.get("state") != expected_state or document.get("next_action") != expected_action:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_STATE_INCONSISTENT")


def validate_gap_plan_against_requirement_graph(
    receipt: Mapping[str, object],
    graph: Mapping[str, object],
    *,
    root: Path | None = None,
) -> None:
    """Validate that a gap receipt exactly reflects its bound requirement DAG."""

    validate_capability_gap_plan_receipt(receipt, root=root)
    validate_capability_requirement_graph(graph, root=root)
    if (
        receipt.get("graph_id") != graph.get("graph_id")
        or receipt.get("graph_digest") != graph.get("graph_digest")
    ):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_GRAPH_BINDING_MISMATCH")
    nodes = graph["nodes"]
    assert isinstance(nodes, list)
    expected_counts = Counter(str(row["classification"]) for row in nodes)
    received_counts = receipt["classification_counts"]
    assert isinstance(received_counts, Mapping)
    if {
        name: expected_counts.get(name, 0) for name in _CLASSIFICATIONS
    } != {name: int(received_counts[name]) for name in _CLASSIFICATIONS}:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_CLASSIFICATION_MISMATCH")

    nodes_by_requirement = {str(row["requirement_id"]): row for row in nodes}
    requests = receipt["manufacturing_requests"]
    assert isinstance(requests, list)
    requested_abis: dict[str, set[str]] = {}
    requests_by_requirement: Counter[str] = Counter()
    request_keys: set[tuple[str, str | None]] = set()
    for request in requests:
        requirement_id = str(request["requirement_id"])
        node = nodes_by_requirement.get(requirement_id)
        if node is None:
            raise CapabilityGapPlannerError(
                "CAPABILITY_GAP_PLAN_REQUEST_REQUIREMENT_UNKNOWN"
            )
        if node["classification"] not in {"manufacturable", "composable"}:
            raise CapabilityGapPlannerError(
                "CAPABILITY_GAP_PLAN_REQUEST_CLASSIFICATION_INVALID"
            )
        abi_value = request.get("abi_id")
        abi_id = str(abi_value) if abi_value is not None else None
        request_key = (requirement_id, abi_id)
        if request_key in request_keys:
            raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_REQUEST_DUPLICATE")
        request_keys.add(request_key)
        requests_by_requirement[requirement_id] += 1
        resolution = node["resolution"]
        assert isinstance(resolution, Mapping)
        manufacturing_abi_ids = {
            str(value) for value in resolution["manufacturing_abi_ids"]
        }
        if abi_id is None:
            if (
                node["classification"] != "manufacturable"
                or manufacturing_abi_ids
                or request["capability"] != node["capability"]
            ):
                raise CapabilityGapPlannerError(
                    "CAPABILITY_GAP_PLAN_UNBOUND_MATERIALIZER_REQUEST"
                )
            continue
        if abi_id not in manufacturing_abi_ids:
            raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_REQUEST_ABI_MISMATCH")
        requested_abis.setdefault(requirement_id, set()).add(abi_id)

    for requirement_id, node in nodes_by_requirement.items():
        resolution = node["resolution"]
        assert isinstance(resolution, Mapping)
        expected_abis = {str(value) for value in resolution["manufacturing_abi_ids"]}
        if requested_abis.get(requirement_id, set()) != expected_abis:
            raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_REQUEST_CLOSURE_MISMATCH")
        if (
            node["classification"] == "manufacturable"
            and requests_by_requirement[requirement_id] == 0
        ):
            raise CapabilityGapPlannerError("CAPABILITY_GAP_PLAN_MANUFACTURING_REQUEST_MISSING")


def _classify_requirement(
    *,
    requirement: Mapping[str, object],
    dependency_decisions: Sequence[Mapping[str, object]],
    portrait_state: Mapping[str, object],
    registry: Mapping[str, object],
    registry_path: Path,
    eligible_abi_ids: set[str],
    admitted_abi_ids: set[str],
    manufacturable_capabilities: set[str],
    available_data_regimes: set[str],
    kernel_capabilities: set[str],
    maximum_authority_level: str,
    ranking_prior_ids: Sequence[str],
    root: Path | None,
) -> tuple[dict[str, object], dict[str, object] | None, list[dict[str, object]]]:
    capability = str(requirement["capability"])
    external_ports = [dict(row) for row in requirement["external_ports"]]
    base_resolution = {
        "native": False,
        "selected_abi_ids": [],
        "composition_id": None,
        "composition_digest": None,
        "manufacturing_abi_ids": [],
        "external_ports": external_ports,
        "ranking_prior_ids": list(ranking_prior_ids),
    }
    dependency_blocker = _dependency_blocker(dependency_decisions)
    if dependency_blocker is not None:
        classification, reason = dependency_blocker
        return _decision(classification, [reason], base_resolution), None, []
    waiting = [
        row
        for row in dependency_decisions
        if row["classification"] in {"manufacturable", "composable"}
        and row["resolution"]["manufacturing_abi_ids"]
    ]
    if waiting:
        return _decision(
            "composable",
            ["CAPABILITY_DEPENDENCY_MANUFACTURING_REQUIRED"],
            base_resolution,
        ), None, []
    structural = _structural_blocker(
        requirement,
        portrait_state=portrait_state,
        available_data_regimes=available_data_regimes,
    )
    if structural is not None:
        classification, reasons = structural
        return _decision(classification, reasons, base_resolution), None, []
    kind = str(requirement["kind"])
    available = portrait_state["available_capabilities"]
    assert isinstance(available, set)
    if capability in available or capability in kernel_capabilities:
        base_resolution["native"] = True
        return _decision("satisfied", ["CAPABILITY_AVAILABLE_NATIVE"], base_resolution), None, []
    if kind in {"observation", "interface", "data", "evaluation"}:
        if kind == "evaluation" and portrait_state["evaluator_ready"]:
            base_resolution["native"] = True
            return (
                _decision("satisfied", ["EVALUATOR_BINDING_AVAILABLE"], base_resolution),
                None,
                [],
            )
        if kind == "data":
            return _decision("data_blocked", ["DATA_REGIME_UNAVAILABLE"], base_resolution), None, []
        return _decision(
            "interface_extension_required",
            ["CAPABILITY_INTERFACE_NOT_AVAILABLE"],
            base_resolution,
        ), None, []
    if kind == "evidence":
        return _decision(
            "architecture_bound",
            ["KERNEL_EVIDENCE_CAPABILITY_UNAVAILABLE"],
            base_resolution,
        ), None, []
    providers = [row for row in registry["abis"] if capability in row["provides"]]
    eligible_providers = [row for row in providers if str(row["abi_id"]) in eligible_abi_ids]
    if not eligible_providers:
        if bool(requirement["manufacturable"]) and capability in manufacturable_capabilities:
            request = _manufacturing_request(
                requirement=requirement,
                capability=capability,
                abi=None,
                dependency_abi_ids=(),
                reason="REGISTERED_MATERIALIZER_LEAF",
            )
            base_resolution["manufacturing_abi_ids"] = []
            return _decision(
                "manufacturable",
                ["REGISTERED_MATERIALIZER_AVAILABLE"],
                base_resolution,
            ), None, [request]
        blocker = _provider_blocker(providers, portrait_state=portrait_state)
        return _decision(blocker[0], blocker[1], base_resolution), None, []
    external = {str(row["port"]): str(row["contract"]) for row in external_ports}
    try:
        composition = compose_module_abis(
            registry_path=registry_path,
            requested_capabilities=[capability],
            external_ports=external,
            maximum_authority_level=maximum_authority_level,
            expected_registry_digest=str(registry["registry_digest"]),
            eligible_abi_ids=sorted(eligible_abi_ids),
            root=root,
        )
    except ModuleCompositionError as exc:
        detail = str(exc)
        if "PORT" in detail:
            return _decision(
                "interface_extension_required",
                ["MODULE_PORT_UNSATISFIED"],
                base_resolution,
            ), None, []
        if "AUTHORITY" in detail or "CYCLE" in detail or "CONSTRAINT" in detail:
            return _decision(
                "architecture_bound",
                ["MODULE_COMPOSITION_UNSUPPORTED:" + detail.split(":", 1)[0]],
                base_resolution,
            ), None, []
        if "DRIFT" in detail or "DIGEST" in detail:
            raise CapabilityGapPlannerError(
                f"CAPABILITY_GAP_COMPOSITION_TAMPERED:{detail}"
            ) from exc
        return _decision(
            "architecture_bound",
            ["MODULE_CAPABILITY_CLOSURE_UNSATISFIED"],
            base_resolution,
        ), None, []
    selected = composition["selected_modules"]
    selected_ids = [str(row["abi_id"]) for row in selected]
    base_resolution.update(
        {
            "selected_abi_ids": selected_ids,
            "composition_id": composition["composition_id"],
            "composition_digest": composition["receipt_digest"],
        }
    )
    missing = [row for row in selected if str(row["abi_id"]) not in admitted_abi_ids]
    if not missing:
        classification = "reusable" if len(selected) == 1 else "composable"
        return _decision(
            classification,
            ["ADMITTED_MODULE_REUSED" if len(selected) == 1 else "ADMITTED_MODULES_COMPOSED"],
            base_resolution,
        ), composition, []
    if not bool(requirement["manufacturable"]):
        return _decision(
            "architecture_bound",
            ["MODULE_IMPLEMENTATION_MISSING_AND_MANUFACTURING_FORBIDDEN"],
            base_resolution,
        ), composition, []
    leaf_rows = _missing_leaf_modules(
        composition,
        admitted_abi_ids=admitted_abi_ids,
    )
    if not leaf_rows:
        return _decision(
            "architecture_bound",
            ["MODULE_MANUFACTURING_LEAF_UNRESOLVED"],
            base_resolution,
        ), composition, []
    requests = [
        _manufacturing_request(
            requirement=requirement,
            capability=(
                capability
                if capability in row["provides"]
                else sorted(str(value) for value in row["provides"])[0]
            ),
            abi=row,
            dependency_abi_ids=_dependency_abi_ids(composition, str(row["module_id"])),
            reason="MISSING_ADMITTED_ABI_LEAF",
        )
        for row in leaf_rows
    ]
    leaf_ids = [str(row["abi_id"]) for row in leaf_rows]
    base_resolution["manufacturing_abi_ids"] = leaf_ids
    root_abi_id = _root_provider_abi_id(composition, capability)
    classification = (
        "manufacturable"
        if len(missing) == 1 and root_abi_id in leaf_ids
        else "composable"
    )
    return _decision(
        classification,
        ["MISSING_LEAF_MODULE_MANUFACTURING_REQUIRED"],
        base_resolution,
    ), composition, requests


def _build_graph(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    registry: Mapping[str, object],
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
    root: Path | None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-capability-requirement-graph",
        "goal_ir_id": goal_ir["goal_ir_id"],
        "goal_ir_digest": _digest(goal_ir),
        "portrait_id": portrait["portrait_id"],
        "portrait_digest": _digest(portrait),
        "registry_id": registry["registry_id"],
        "registry_digest": registry["registry_digest"],
        "nodes": [dict(row) for row in nodes],
        "edges": [dict(row) for row in edges],
        "claim_boundary": (
            "Node classifications require exact portrait and ABI compatibility. "
            "Portable knowledge affects ranking only and grants no execution authority."
        ),
    }
    graph: dict[str, object] = {**identity, "graph_id": _stable_id("capability-graph", identity)}
    graph["graph_digest"] = _digest(graph)
    validate_capability_requirement_graph(graph, root=root)
    return graph


def _build_gap_receipt(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    registry: Mapping[str, object],
    graph: Mapping[str, object],
    nodes: Sequence[Mapping[str, object]],
    manufacturing_requests: Sequence[Mapping[str, object]],
    composition_receipts: Sequence[Mapping[str, object]],
    knowledge_digest: str | None,
    knowledge_priors: Sequence[Mapping[str, object]],
    root: Path | None,
) -> dict[str, object]:
    counts = Counter(str(row["classification"]) for row in nodes)
    classification_counts = {name: counts.get(name, 0) for name in _CLASSIFICATIONS}
    state, next_action = _gap_state(classification_counts, bool(manufacturing_requests))
    composition_bindings = sorted(
        [
            {
                "requirement_id": node["requirement_id"],
                "capability": node["capability"],
                "composition_id": node["resolution"]["composition_id"],
                "composition_digest": node["resolution"]["composition_digest"],
                "selected_abi_ids": list(node["resolution"]["selected_abi_ids"]),
            }
            for node in nodes
            if node["resolution"]["composition_id"] is not None
        ],
        key=lambda row: str(row["requirement_id"]),
    )
    known_compositions = {
        (str(row["composition_id"]), str(row["receipt_digest"]))
        for row in composition_receipts
    }
    if any(
        (str(row["composition_id"]), str(row["composition_digest"]))
        not in known_compositions
        for row in composition_bindings
    ):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_COMPOSITION_BINDING_INVALID")
    identity: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-capability-gap-plan-receipt",
        "state": state,
        "goal_ir_id": goal_ir["goal_ir_id"],
        "goal_ir_digest": _digest(goal_ir),
        "portrait_id": portrait["portrait_id"],
        "portrait_digest": _digest(portrait),
        "registry_id": registry["registry_id"],
        "registry_digest": registry["registry_digest"],
        "graph_id": graph["graph_id"],
        "graph_digest": graph["graph_digest"],
        "classification_counts": classification_counts,
        "manufacturing_requests": sorted(
            (dict(row) for row in manufacturing_requests),
            key=lambda row: str(row["request_id"]),
        ),
        "composition_bindings": composition_bindings,
        "knowledge_graph_digest": knowledge_digest,
        "knowledge_priors": sorted(
            (dict(row) for row in knowledge_priors),
            key=lambda row: (str(row["requirement_id"]), int(row["rank"])),
        ),
        "next_action": next_action,
        "authority": {
            "module_manufacturing_authority": False,
            "gpu_authority": False,
            "evaluator_authority": False,
            "promotion_authority": False,
        },
        "side_effects": {
            "source_mutated": False,
            "generated_module_imported": False,
            "gpu_execution_started": False,
        },
        "claim_boundary": (
            "This receipt ranks reusable evidence and identifies missing leaves. "
            "It does not authorize code generation, GPU work, evaluator changes, or promotion."
        ),
    }
    receipt: dict[str, object] = {**identity, "plan_id": _stable_id("gap-plan", identity)}
    receipt["plan_digest"] = _digest(receipt)
    validate_capability_gap_plan_receipt(receipt, root=root)
    return receipt


def _normalize_requirement(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CapabilityGapPlannerError("GOAL_IR_REQUIREMENT_INVALID")
    capability = _required_text(value.get("capability"), "GOAL_IR_CAPABILITY_INVALID")
    kind = _required_text(value.get("kind"), "GOAL_IR_REQUIREMENT_KIND_INVALID")
    if kind not in {"observation", "interface", "data", "intervention", "evaluation", "evidence"}:
        raise CapabilityGapPlannerError("GOAL_IR_REQUIREMENT_KIND_INVALID")
    interfaces = []
    raw_interfaces = value.get("required_interfaces", ())
    if isinstance(raw_interfaces, (str, bytes)) or not isinstance(raw_interfaces, Sequence):
        raise CapabilityGapPlannerError("GOAL_IR_INTERFACE_INVALID")
    for raw in raw_interfaces:
        if not isinstance(raw, Mapping):
            raise CapabilityGapPlannerError("GOAL_IR_INTERFACE_INVALID")
        interfaces.append(
            {
                "kind": _required_text(raw.get("kind"), "GOAL_IR_INTERFACE_INVALID"),
                "contract_id": _required_text(
                    raw.get("contract_id"), "GOAL_IR_INTERFACE_INVALID"
                ),
            }
        )
    external_ports = []
    raw_external_ports = value.get("external_ports", ())
    if isinstance(raw_external_ports, (str, bytes)) or not isinstance(
        raw_external_ports, Sequence
    ):
        raise CapabilityGapPlannerError("GOAL_IR_EXTERNAL_PORT_INVALID")
    for raw in raw_external_ports:
        if not isinstance(raw, Mapping):
            raise CapabilityGapPlannerError("GOAL_IR_EXTERNAL_PORT_INVALID")
        external_ports.append(
            {
                "port": _required_text(raw.get("port"), "GOAL_IR_EXTERNAL_PORT_INVALID"),
                "contract": _required_text(
                    raw.get("contract"), "GOAL_IR_EXTERNAL_PORT_INVALID"
                ),
            }
        )
    manufacturable = value.get("manufacturable", False)
    if not isinstance(manufacturable, bool):
        raise CapabilityGapPlannerError("GOAL_IR_MANUFACTURABLE_INVALID")
    body: dict[str, object] = {
        "capability": capability,
        "kind": kind,
        "dependencies": sorted(
            _strings(value.get("dependencies", ()), "GOAL_IR_DEPENDENCY_INVALID")
        ),
        "required_model_capabilities": sorted(
            _strings(
                value.get("required_model_capabilities", ()),
                "GOAL_IR_MODEL_CAPABILITY_INVALID",
            )
        ),
        "required_hooks": sorted(
            _strings(value.get("required_hooks", ()), "GOAL_IR_HOOK_INVALID")
        ),
        "required_interfaces": sorted(
            interfaces, key=lambda row: (row["kind"], row["contract_id"])
        ),
        "required_probe_keys": sorted(
            _strings(value.get("required_probe_keys", ()), "GOAL_IR_PROBE_KEY_INVALID")
        ),
        "required_data_regimes": sorted(
            _strings(value.get("required_data_regimes", ()), "GOAL_IR_DATA_REGIME_INVALID")
        ),
        "compatible_model_families": sorted(
            _strings(
                value.get("compatible_model_families", ()),
                "GOAL_IR_COMPATIBLE_MODEL_FAMILY_INVALID",
                nonempty=True,
            )
        ),
        "manufacturable": manufacturable,
        "external_ports": sorted(external_ports, key=lambda row: row["port"]),
        "source": _required_text(value.get("source"), "GOAL_IR_SOURCE_INVALID"),
    }
    for field in (
        "dependencies",
        "required_model_capabilities",
        "required_hooks",
        "required_probe_keys",
        "required_data_regimes",
        "compatible_model_families",
    ):
        values = body[field]
        assert isinstance(values, list)
        if len(values) != len(set(values)):
            raise CapabilityGapPlannerError(f"GOAL_IR_DUPLICATE:{field}")
    if len(interfaces) != len({(row["kind"], row["contract_id"]) for row in interfaces}):
        raise CapabilityGapPlannerError("GOAL_IR_INTERFACE_DUPLICATE")
    if len(external_ports) != len({row["port"] for row in external_ports}):
        raise CapabilityGapPlannerError("GOAL_IR_EXTERNAL_PORT_DUPLICATE")
    body["requirement_id"] = _stable_id("requirement", body)
    return body


def _topological_requirements(
    requirements: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_capability = {str(row["capability"]): row for row in requirements}
    state: dict[str, str] = {}
    ordered: list[Mapping[str, object]] = []

    def visit(capability: str) -> None:
        if state.get(capability) == "visiting":
            raise CapabilityGapPlannerError(f"GOAL_IR_DEPENDENCY_CYCLE:{capability}")
        if state.get(capability) == "done":
            return
        row = by_capability.get(capability)
        if row is None:
            raise CapabilityGapPlannerError(f"GOAL_IR_DEPENDENCY_UNKNOWN:{capability}")
        state[capability] = "visiting"
        for dependency in sorted(str(value) for value in row["dependencies"]):
            visit(dependency)
        state[capability] = "done"
        ordered.append(row)

    for capability in sorted(by_capability):
        visit(capability)
    return ordered


def _portrait_state(portrait: Mapping[str, object]) -> dict[str, object]:
    coverage = portrait["coverage"]
    structural = portrait["structural_profile"]
    assert isinstance(coverage, Mapping) and isinstance(structural, Mapping)
    interfaces = structural["execution_interfaces"]
    assert isinstance(interfaces, list)
    evaluator = structural["evaluator"]
    assert isinstance(evaluator, Mapping)
    return {
        "model_family": str(portrait["model_family"]),
        "available_capabilities": set(str(value) for value in coverage["available_capabilities"]),
        "unknown_capabilities": set(str(value) for value in coverage["unknown_capabilities"]),
        "unavailable_capabilities": set(
            str(value) for value in coverage["unavailable_capabilities"]
        ),
        "available_hooks": set(str(value) for value in coverage["available_hooks"]),
        "unknown_hooks": set(str(value) for value in coverage["unknown_hooks"]),
        "unavailable_hooks": set(str(value) for value in coverage["unavailable_hooks"]),
        "interfaces": {
            str(row["kind"]): row
            for row in interfaces
            if isinstance(row, Mapping)
        },
        "probe_keys": set(str(value) for value in coverage["observed_probe_keys"]),
        "evaluator_ready": evaluator.get("state") == "ready",
    }


def _structural_blocker(
    requirement: Mapping[str, object],
    *,
    portrait_state: Mapping[str, object],
    available_data_regimes: set[str],
) -> tuple[str, list[str]] | None:
    model_family = str(portrait_state["model_family"])
    if model_family not in requirement["compatible_model_families"]:
        return "architecture_bound", ["MODEL_FAMILY_INCOMPATIBLE"]
    unavailable = portrait_state["unavailable_capabilities"]
    unknown = portrait_state["unknown_capabilities"]
    assert isinstance(unavailable, set) and isinstance(unknown, set)
    required_caps = set(str(value) for value in requirement["required_model_capabilities"])
    if required_caps & unavailable:
        return "architecture_bound", ["REQUIRED_MODEL_CAPABILITY_UNAVAILABLE"]
    if required_caps & unknown or not required_caps.issubset(
        portrait_state["available_capabilities"]
    ):
        return "interface_extension_required", ["REQUIRED_MODEL_CAPABILITY_UNKNOWN"]
    required_hooks = set(str(value) for value in requirement["required_hooks"])
    if not required_hooks.issubset(portrait_state["available_hooks"]):
        return "interface_extension_required", ["REQUIRED_HOOK_UNAVAILABLE"]
    interfaces = portrait_state["interfaces"]
    assert isinstance(interfaces, Mapping)
    for required in requirement["required_interfaces"]:
        row = interfaces.get(str(required["kind"]))
        if (
            not isinstance(row, Mapping)
            or row.get("state") != "available"
            or row.get("contract_id") != required["contract_id"]
        ):
            return "interface_extension_required", ["REQUIRED_INTERFACE_UNAVAILABLE"]
    if not set(requirement["required_probe_keys"]).issubset(portrait_state["probe_keys"]):
        return "interface_extension_required", ["REQUIRED_PROBE_COVERAGE_MISSING"]
    if not set(requirement["required_data_regimes"]).issubset(available_data_regimes):
        return "data_blocked", ["REQUIRED_DATA_REGIME_MISSING"]
    return None


def _eligible_abi_ids(
    registry: Mapping[str, object],
    *,
    portrait_state: Mapping[str, object],
    maximum_authority_level: str,
) -> set[str]:
    result = set()
    for abi in registry["abis"]:
        portability = abi["portability"]
        if portrait_state["model_family"] not in portability["model_family_scope"]:
            continue
        if not set(abi["required_model_capabilities"]).issubset(
            portrait_state["available_capabilities"]
        ):
            continue
        if not set(abi["required_hooks"]).issubset(portrait_state["available_hooks"]):
            continue
        if _AUTHORITY_RANK[str(abi["authority_level"])] > _AUTHORITY_RANK[maximum_authority_level]:
            continue
        result.add(str(abi["abi_id"]))
    return result


def _provider_blocker(
    providers: Sequence[Mapping[str, object]],
    *,
    portrait_state: Mapping[str, object],
) -> tuple[str, list[str]]:
    if not providers:
        return "architecture_bound", ["NO_ADMITTED_ABI_FOR_CAPABILITY"]
    model_family = str(portrait_state["model_family"])
    if all(model_family not in row["portability"]["model_family_scope"] for row in providers):
        return "architecture_bound", ["ABI_MODEL_FAMILY_INCOMPATIBLE"]
    if any(
        not set(row["required_hooks"]).issubset(portrait_state["available_hooks"])
        for row in providers
    ):
        return "interface_extension_required", ["ABI_REQUIRED_HOOK_UNAVAILABLE"]
    return "architecture_bound", ["ABI_STRUCTURAL_REQUIREMENTS_UNSATISFIED"]


def _dependency_blocker(
    decisions: Sequence[Mapping[str, object]],
) -> tuple[str, str] | None:
    blockers = [
        str(row["classification"])
        for row in decisions
        if row["classification"] in _BLOCKER_PRECEDENCE
    ]
    if not blockers:
        return None
    strongest = max(blockers, key=lambda value: _BLOCKER_PRECEDENCE[value])
    return strongest, "CAPABILITY_DEPENDENCY_BLOCKED"


def _missing_leaf_modules(
    composition: Mapping[str, object],
    *,
    admitted_abi_ids: set[str],
) -> list[Mapping[str, object]]:
    selected = composition["selected_modules"]
    by_module = {str(row["module_id"]): row for row in selected}
    providers: dict[str, set[str]] = {}
    for edge in composition["dependency_edges"]:
        providers.setdefault(str(edge["consumer_module_id"]), set()).add(
            str(edge["provider_module_id"])
        )
    leaves = []
    for row in selected:
        if str(row["abi_id"]) in admitted_abi_ids:
            continue
        dependency_rows = [
            by_module[module_id]
            for module_id in providers.get(str(row["module_id"]), set())
        ]
        if all(str(value["abi_id"]) in admitted_abi_ids for value in dependency_rows):
            leaves.append(row)
    return leaves


def _dependency_abi_ids(
    composition: Mapping[str, object], consumer_module_id: str
) -> list[str]:
    selected = {
        str(row["module_id"]): str(row["abi_id"])
        for row in composition["selected_modules"]
    }
    return sorted(
        selected[str(edge["provider_module_id"])]
        for edge in composition["dependency_edges"]
        if str(edge["consumer_module_id"]) == consumer_module_id
    )


def _root_provider_abi_id(composition: Mapping[str, object], capability: str) -> str:
    binding = next(
        row
        for row in composition["capability_bindings"]
        if row["requester_id"] == "composition-root" and row["capability"] == capability
    )
    return str(binding["provider_abi_id"])


def _manufacturing_request(
    *,
    requirement: Mapping[str, object],
    capability: str,
    abi: Mapping[str, object] | None,
    dependency_abi_ids: Sequence[str],
    reason: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "requirement_id": requirement["requirement_id"],
        "capability": capability,
        "abi_id": abi["abi_id"] if abi is not None else None,
        "abi_version": abi["abi_version"] if abi is not None else None,
        "abi_digest": abi["abi_digest"] if abi is not None else None,
        "reason": reason,
        "dependency_abi_ids": list(dependency_abi_ids),
        "external_ports": [dict(row) for row in requirement["external_ports"]],
    }
    return {**body, "request_id": _stable_id("manufacture", body)}


def _decision(
    classification: str,
    reason_codes: Sequence[str],
    resolution: Mapping[str, object],
) -> dict[str, object]:
    return {
        "classification": classification,
        "reason_codes": sorted(set(reason_codes)),
        "resolution": dict(resolution),
    }


def _gap_state(counts: Mapping[str, int], has_manufacturing: bool) -> tuple[str, str]:
    if counts["architecture_bound"]:
        return "architecture_bound", "stop_architecture_bound"
    if counts["data_blocked"]:
        return "missing_data_regime", "bind_data_regime"
    if counts["interface_extension_required"]:
        return "requires_interface_extension", "extend_interface"
    if has_manufacturing:
        return "requires_manufacturing", "manufacture_leaf_modules"
    return "ready_for_portfolio", "compile_experiment_portfolio"


def _load_knowledge_graph(
    value: Mapping[str, object] | Path | None,
) -> tuple[Mapping[str, object] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        graph = value
    else:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            path = path / "graph.json"
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CapabilityGapPlannerError("CAPABILITY_GAP_KNOWLEDGE_GRAPH_INVALID") from exc
    if (
        not isinstance(graph, Mapping)
        or graph.get("artifact_type") != "verdiwm-portable-knowledge-graph"
    ):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_KNOWLEDGE_GRAPH_INVALID")
    try:
        reject_runtime_bindings(graph)
    except GeometryValidationError as exc:
        raise CapabilityGapPlannerError("CAPABILITY_GAP_KNOWLEDGE_GRAPH_PATH_BOUND") from exc
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise CapabilityGapPlannerError("CAPABILITY_GAP_KNOWLEDGE_GRAPH_INVALID")
    return graph, _digest(graph)


def _knowledge_priors(
    graph: Mapping[str, object] | None,
    *,
    requirement: Mapping[str, object],
    model_family: str,
) -> list[dict[str, object]]:
    if graph is None:
        return []
    nodes = {str(row["id"]): row for row in graph["nodes"] if isinstance(row, Mapping)}
    target_ids = {
        node_id
        for node_id, row in nodes.items()
        if row.get("kind") in {"capability", "capability_class"}
        and row.get("key") == requirement["capability"]
    }
    candidates = []
    for edge in graph["edges"]:
        if not isinstance(edge, Mapping):
            continue
        if edge.get("target") not in target_ids or edge.get("relation") not in {
            "requires_capability",
            "requires_capability_class",
        }:
            continue
        source = nodes.get(str(edge.get("source")))
        if source is None:
            continue
        source_id = str(source["id"])
        model_match = any(
            row.get("source") == source_id
            and row.get("relation") == "observed_on_model_family"
            and nodes.get(str(row.get("target")), {}).get("key") == model_family
            for row in graph["edges"]
            if isinstance(row, Mapping)
        )
        claim_scope = str(source.get("claim_scope") or "unknown")
        if claim_scope not in {"ranking_only", "target_local", "transfer_prior"}:
            claim_scope = (
                "ranking_only"
                if source.get("kind") in {"mechanism", "portable_experience"}
                else "unknown"
            )
        candidates.append(
            {
                "requirement_id": requirement["requirement_id"],
                "capability": requirement["capability"],
                "source_node_id": source_id,
                "source_kind": str(source.get("kind") or "unknown"),
                "claim_scope": claim_scope,
                "model_family_match": model_match,
            }
        )
    scope_rank = {"transfer_prior": 3, "target_local": 2, "ranking_only": 1, "unknown": 0}
    candidates.sort(
        key=lambda row: (
            not bool(row["model_family_match"]),
            -scope_rank[str(row["claim_scope"])],
            str(row["source_kind"]),
            str(row["source_node_id"]),
        )
    )
    result = []
    for rank, row in enumerate(candidates, start=1):
        identity = dict(row)
        result.append(
            {
                **row,
                "rank": rank,
                "prior_id": _stable_id("knowledge-prior", identity),
            }
        )
    return result


def _validate_path_free_schema(
    schema: str, document: Mapping[str, object], *, root: Path | None
) -> None:
    try:
        reject_runtime_bindings(document)
        validate_document(schema, document, root=root)
    except (GeometryValidationError, ContractValidationError) as exc:
        raise CapabilityGapPlannerError(f"CAPABILITY_GAP_DOCUMENT_INVALID:{exc}") from exc


def _validate_node_dag(nodes: Sequence[Mapping[str, object]]) -> None:
    by_id = {str(row["node_id"]): row for row in nodes}
    state: dict[str, str] = {}

    def visit(node_id: str) -> None:
        if state.get(node_id) == "visiting":
            raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_CYCLE")
        if state.get(node_id) == "done":
            return
        row = by_id.get(node_id)
        if row is None:
            raise CapabilityGapPlannerError("CAPABILITY_REQUIREMENT_GRAPH_DEPENDENCY_UNKNOWN")
        state[node_id] = "visiting"
        for dependency in row["dependencies"]:
            visit(str(dependency))
        state[node_id] = "done"

    for node_id in sorted(by_id):
        visit(node_id)


def _strings(value: object, code: str, *, nonempty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CapabilityGapPlannerError(code)
    result = []
    for row in value:
        if not isinstance(row, str) or not row:
            raise CapabilityGapPlannerError(code)
        result.append(row)
    if nonempty and not result:
        raise CapabilityGapPlannerError(code)
    return result


def _string_set(value: Sequence[str], code: str) -> set[str]:
    rows = _strings(value, code)
    if len(rows) != len(set(rows)):
        raise CapabilityGapPlannerError(code)
    return set(rows)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityGapPlannerError(code)
    return value.strip()


def _stable_id(prefix: str, value: Mapping[str, object]) -> str:
    return f"{prefix}-{_digest(value)[:24]}"


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
