"""Strict stage drivers for the Ctrl-World autonomous transfer controller."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.ctrl_world_hybrid_memory_transfer_v1 import run as hybrid_campaign
from experiments.ctrl_world_research_loop_v2.research_intake import (
    run_research_intake_v2,
)
from experiments.ctrl_world_autonomous_transfer_v1.local_method_intake import (
    run_local_method_intake,
)
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.automatic_materialization import (
    materialization_plan_digest,
    run_automatic_materialization,
)
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.llm_task_adapter import run_llm_task
from wmloop.execute.observation_adapter import (
    ObservationAdapterError,
    run_observation_adapter,
)
from wmloop.evaluate.shadow_metric_evolution import (
    ShadowMetricEvolutionError,
    compile_shadow_metric_proposals,
)
from wmloop.control.acwm_materialized_campaign import (
    bind_training_to_batch,
    validate_materialized_candidate_batch,
)
from wmloop.control.adaptive_observation import (
    AdaptiveObservationError,
    build_adaptive_probe_plan,
    load_observation_abi_registry,
    validate_adaptive_probe_plan,
)
from wmloop.control.automatic_module_plan import (
    AutomaticModulePlanError,
    compile_automatic_module_plan,
    load_automatic_module_admission,
)
from wmloop.control.capability_gap_planner import (
    CapabilityGapPlannerError,
    build_goal_ir,
    compile_capability_gap_plan,
    validate_capability_requirement_graph,
    validate_gap_plan_against_requirement_graph,
)
from wmloop.control.experiment_portfolio import (
    ExperimentPortfolioError,
    compile_experiment_portfolio,
    validate_experiment_portfolio,
    validate_hypothesis_batch,
)
from wmloop.control.model_portrait import (
    ModelPortraitError,
    build_portrait_readiness_receipt,
    derive_model_portrait,
    probe_coverage_key,
    update_model_portrait_from_observation,
    validate_model_portrait,
)
from wmloop.control.open_method_pipeline import (
    OpenMethodPipelineError,
    build_open_method_request,
    compile_open_method_proposal,
)
from wmloop.control.shadow_probe_admission import (
    ShadowProbeAdmissionError,
    compile_shadow_probe_admission,
)
from wmloop.geometry.community_knowledge import (
    CommunityKnowledgeError,
    build_portrait_transition,
    validate_portrait_transition,
)
from wmloop.geometry.portable_transfer_knowledge import (
    build_probe_fingerprint_summary,
)
from wmloop.control.module_composition import (
    ModuleCompositionError,
    load_module_abi_registry,
)
from wmloop.control.module_manufacturing import (
    ModuleManufacturingError,
    build_intervention_manufacturing_work_order,
    build_observation_manufacturing_work_order,
    load_module_manufacturing_work_order,
)
from wmloop.control.resource_portfolio import (
    ResourcePortfolioError,
    build_confirm_resource_portfolio_receipt,
    build_screen_resource_portfolio_receipt,
    load_resource_portfolio_receipt,
)
from wmloop.experiments.evidence_graph import write_evidence_graph
from wmloop.experiments.materialized_transfer_evidence import (
    project_materialized_transfer_evidence,
)
from wmloop.experiments.portable_knowledge_graph import (
    stage_portable_knowledge_records,
    write_portable_knowledge_graph,
)
from wmloop.experiments.verified_transfer_knowledge import (
    stage_verified_transfer_knowledge,
)
from wmloop.experiments.training_resource_planner import (
    method_class_from_candidate,
    plan_training_resources,
    TrainingResourcePlanningError,
)
from experiments.ctrl_world_autonomous_transfer_v1.replanning import (
    ClosedLoopReplanningError,
    archive_work,
    build_next_task_decision,
    build_quality_audit,
    load_archive_receipts,
)
from wmloop.verify.acwm_materialized_frozen_verifier import (
    run_materialized_frozen_verifier,
)


class AutonomousTransferWorkflowError(RuntimeError):
    """A strict workflow stage could not preserve its declared boundary."""


@dataclass(frozen=True)
class StageResult:
    """Terminal result for one durable controller stage attempt."""

    state: str
    outcome: str
    payload: dict[str, object]
    receipt_path: Path | None = None


def config_digest(config: Mapping[str, object]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_digest"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def load_and_validate_config(path: Path, *, project_root: Path) -> dict[str, object]:
    config_path = _require_file(path, "AUTONOMOUS_CONFIG_INVALID")
    config = _load(config_path, "AUTONOMOUS_CONFIG_INVALID")
    try:
        validate_document("ctrl_world_autonomous_transfer_loop", config, root=project_root)
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_CONFIG_SCHEMA_INVALID:{exc}"
        ) from exc
    if config.get("config_digest") != config_digest(config):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CONFIG_DIGEST_MISMATCH")
    gpu_indices = config["gpu_indices"]
    assert isinstance(gpu_indices, list)
    if (
        len(gpu_indices) > 8
        or len(gpu_indices) != len(set(gpu_indices))
        or int(config["max_parallel_gpu_jobs"]) > len(gpu_indices)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GPU_PARALLELISM_INVALID")
    if isinstance(config.get("portfolio_planning"), Mapping) and not isinstance(
        config.get("resource_portfolio"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_PORTFOLIO_REQUIRED"
        )
    if isinstance(config.get("closed_loop"), Mapping) and not isinstance(
        config.get("portrait_gate"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CLOSED_LOOP_PORTRAIT_GATE_REQUIRED"
        )
    if isinstance(config.get("observation_planning"), Mapping) != isinstance(
        config.get("observation_execution"), Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_PAIR_REQUIRED"
        )
    if isinstance(config.get("resource_portfolio"), Mapping):
        _load_resource_policy(config, project_root=project_root)
    _validate_local_method_discovery(config, project_root=project_root)
    _validate_metric_evolution(config, project_root=project_root)
    _validate_paths(config, project_root=project_root)
    graph_root = Path(str(config["knowledge_graph_root"])).expanduser().resolve()
    if graph_root == project_root or project_root in graph_root.parents:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GRAPH_INSIDE_SOURCE")
    portable_graph_value = config.get("portable_knowledge_root")
    if portable_graph_value is not None:
        portable_graph_root = Path(str(portable_graph_value)).expanduser().resolve()
        if (
            portable_graph_root == project_root
            or project_root in portable_graph_root.parents
            or portable_graph_root == graph_root
            or graph_root in portable_graph_root.parents
            or portable_graph_root in graph_root.parents
        ):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_GRAPH_ROOT_INVALID")
    portable_records_value = config.get("portable_knowledge_records_root")
    if portable_records_value is not None:
        portable_records_root = Path(str(portable_records_value)).expanduser().resolve()
        if portable_records_root == project_root or project_root in portable_records_root.parents:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INSIDE_SOURCE")
    return config


def _validate_local_method_discovery(
    config: Mapping[str, object], *, project_root: Path
) -> None:
    settings = config.get("local_method_discovery")
    if settings is None:
        return
    if not isinstance(settings, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_DISCOVERY_INVALID")
    if settings.get("enabled") is not True:
        return
    intake_path = _require_file(
        Path(str(settings.get("intake_config") or "")),
        "AUTONOMOUS_LOCAL_METHOD_INTAKE_CONFIG_INVALID",
    )
    profiles = settings.get("materializer_profiles")
    materializers = config.get("materializers")
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(value, str) or not value for value in profiles)
        or len(set(profiles)) != len(profiles)
        or not isinstance(materializers, Mapping)
        or any(str(value) not in materializers for value in profiles)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_PROFILE_BINDING_INVALID")
    maximum = settings.get("max_methods")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_LOCAL_METHOD_MAX_INVALID")


def _validate_metric_evolution(
    config: Mapping[str, object], *, project_root: Path
) -> None:
    settings = config.get("metric_evolution")
    if settings is None:
        return
    if not isinstance(settings, Mapping) or settings.get("mode") != "shadow_only":
        raise AutonomousTransferWorkflowError("AUTONOMOUS_METRIC_EVOLUTION_SHADOW_ONLY_REQUIRED")
    protected = settings.get("protected_metric_ids")
    candidates = settings.get("candidate_metrics")
    if (
        not isinstance(protected, list)
        or any(not isinstance(value, str) or not value for value in protected)
        or not isinstance(candidates, list)
        or any(not isinstance(value, Mapping) for value in candidates)
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_METRIC_EVOLUTION_CONFIG_INVALID")
    for candidate in candidates:
        try:
            validate_document("metric_adequacy", candidate, root=project_root)
        except ContractValidationError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_METRIC_EVOLUTION_CONTRACT_INVALID:{exc}"
            ) from exc


def run_discovery(
    config: Mapping[str, object], *, output_root: Path, failure_context: tuple[str, ...]
) -> dict[str, object]:
    paths = _paths(config)
    intake = _load(paths["intake_config"], "AUTONOMOUS_INTAKE_CONFIG_INVALID")
    queries = [str(value) for value in intake.get("queries", [])]
    for value in failure_context:
        for query in _failure_queries(value):
            if query not in queries:
                queries.append(query)
    intake["queries"] = queries[:12]
    intake["policy_digest"] = ""
    intake["policy_digest"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in intake.items() if key != "policy_digest"}
        )
    ).hexdigest()
    cycle_config = output_root.with_name(output_root.name + "-intake-config.json")
    _write_json_idempotent(cycle_config, intake)
    external_error: str | None = None
    try:
        external = run_research_intake_v2(
            config_path=cycle_config,
            contract_path=paths["contract"],
            output_root=output_root,
            project_root=paths["project_root"],
            failure_context=failure_context,
            source_bundle_path=(
                Path(str(config["source_bundle_path"]))
                if config.get("source_bundle_path")
                else None
            ),
        )
    except Exception as exc:
        # A network/replay outage is an intake failure, not a reason to discard
        # the controller's local reviewed method inventory. Preserve the error
        # in the cycle manifest while keeping the source boundary explicit.
        external = {
            "state": "failed",
            "assessment_paths": [],
            "idea_paths": [],
            "work_order_paths": [],
        }
        external_error = f"{type(exc).__name__}:{str(exc)[:800]}"
    strict_network = _external_research_requires_network(config)
    if strict_network and external_error is None:
        reason = _network_readiness_error(
            external,
            minimum_successful_sources=_external_research_minimum_sources(config),
        )
        if reason is not None:
            external_error = reason
    if strict_network and external_error is not None:
        return {
            "state": "failed",
            "assessment_paths": [],
            "idea_paths": [],
            "work_order_paths": [],
            "external": external,
            "local": {
                "state": "not_run_network_required",
                "local_method_count": 0,
                "assessment_paths": [],
                "idea_paths": [],
                "work_order_paths": [],
            },
            "local_method_count": 0,
            "network_required": True,
            "external_error": external_error,
            "claim_boundary": (
                "External retrieval was required for this fresh deployment. No local reviewed "
                "profile was promoted into an executable discovery queue after the network gate failed."
            ),
        }
    local = run_local_method_intake(
        config=config,
        output_root=output_root / "local-methods",
        project_root=paths["project_root"],
        failure_context=failure_context,
    )
    merged = {
        "state": (
            "ready_for_materialization"
            if local.get("idea_paths") or external.get("idea_paths")
            else ("failed" if external_error else str(external.get("state") or "empty"))
        ),
        "assessment_paths": [
            *[str(value) for value in external.get("assessment_paths", [])],
            *[str(value) for value in local.get("assessment_paths", [])],
        ],
        "idea_paths": [
            *[str(value) for value in external.get("idea_paths", [])],
            *[str(value) for value in local.get("idea_paths", [])],
        ],
        "work_order_paths": [
            *[str(value) for value in external.get("work_order_paths", [])],
            *[str(value) for value in local.get("work_order_paths", [])],
        ],
        "external": external,
        "local": local,
        "local_method_count": int(local.get("local_method_count", 0)),
        "claim_boundary": (
            "External and local method sources share only the immutable intake contract. "
            "Local profiles are not external source claims."
        ),
    }
    if external_error is not None:
        merged["external_error"] = external_error
    return merged


def evaluate_portrait_gate(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Bind one idea to a goal-relative portrait readiness decision."""

    gate = config.get("portrait_gate")
    if not isinstance(gate, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_GATE_REQUIRED")
    portrait = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=_state_root_from_attempt(attempt_root),
    )
    idea = _load(
        _require_file(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID"),
        "AUTONOMOUS_IDEA_INVALID",
    )
    work_order = _load(
        _require_file(
            Path(str(work["work_order_path"])), "AUTONOMOUS_WORK_ORDER_INVALID"
        ),
        "AUTONOMOUS_WORK_ORDER_INVALID",
    )
    goal_binding = "sha256:" + hashlib.sha256(
        _canonical_bytes({"idea": idea, "work_order": work_order})
    ).hexdigest()
    requirements = gate.get("requirements")
    if not isinstance(requirements, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_REQUIREMENTS_INVALID")
    try:
        receipt = build_portrait_readiness_receipt(
            portrait=portrait,
            goal_binding=goal_binding,
            coverage_policy_id=str(gate["coverage_policy_id"]),
            required_capabilities=tuple(requirements.get("capabilities", ())),
            required_execution_interfaces=tuple(
                requirements.get("execution_interfaces", ())
            ),
            required_hooks=tuple(requirements.get("hooks", ())),
            required_probe_coverage=tuple(requirements.get("probe_coverage", ())),
            required_operational_metrics=tuple(
                requirements.get("operational_metrics", ())
            ),
            evaluator_required=bool(requirements.get("evaluator_required")),
            root=_paths(config)["project_root"],
        )
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_PORTRAIT_READINESS_INVALID:{exc}"
        ) from exc
    receipt_path = attempt_root / "portrait-readiness.json"
    _write_json_idempotent(receipt_path, receipt)
    payload: dict[str, object] = {
        "portrait_gate_state": receipt["state"],
        "portrait_id": receipt["portrait_id"],
        "portrait_readiness_id": receipt["readiness_id"],
        "portrait_readiness_receipt_path": str(receipt_path),
        "portrait_readiness_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "portrait_goal_binding": goal_binding,
    }
    state = str(receipt["state"])
    if state == "ready_for_gap_planning":
        return StageResult(
            state="completed",
            outcome=state,
            payload=payload,
            receipt_path=receipt_path,
        )

    observation_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-portrait-observation-work-order",
        "state": "pending",
        "work_id": str(work["work_id"]),
        "idea_id": str(work["idea_id"]),
        "portrait_id": receipt["portrait_id"],
        "readiness_id": receipt["readiness_id"],
        "readiness_state": state,
        "goal_binding": goal_binding,
        "blockers": list(receipt["blockers"]),
        "requirements": dict(receipt["requirements"]),
        "authority": {
            "gpu_authority": False,
            "intervention_authority": False,
            "promotion_authority": False,
        },
    }
    observation_body["observation_id"] = (
        "portrait-observation-"
        + hashlib.sha256(_canonical_bytes(observation_body)).hexdigest()[:24]
    )
    try:
        validate_document(
            "portrait_observation_work_order",
            observation_body,
            root=_paths(config)["project_root"],
        )
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_PORTRAIT_OBSERVATION_ORDER_INVALID:{exc}"
        ) from exc
    observation_path = attempt_root.parent / "observation-work-order.json"
    _write_json_idempotent(observation_path, observation_body)
    payload.update(
        {
            "portrait_observation_id": observation_body["observation_id"],
            "portrait_observation_work_order_path": str(observation_path),
            "portrait_observation_work_order_sha256": hashlib.sha256(
                observation_path.read_bytes()
            ).hexdigest(),
            "portrait_blockers": list(receipt["blockers"]),
        }
    )
    return StageResult(
        state="blocked",
        outcome=state,
        payload=payload,
        receipt_path=receipt_path,
    )


def plan_observation(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Turn a portrait blocker into a bounded read-only or shadow work plan."""

    planning = config.get("observation_planning")
    if not isinstance(planning, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_PLANNING_REQUIRED"
        )
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    order_path = _require_file(
        Path(str(context.get("portrait_observation_work_order_path") or "")),
        "AUTONOMOUS_OBSERVATION_WORK_ORDER_INVALID",
    )
    expected_order_sha = context.get("portrait_observation_work_order_sha256")
    if (
        not isinstance(expected_order_sha, str)
        or hashlib.sha256(order_path.read_bytes()).hexdigest() != expected_order_sha
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_WORK_ORDER_HASH_MISMATCH"
        )
    order = _load(order_path, "AUTONOMOUS_OBSERVATION_WORK_ORDER_INVALID")
    registry_path = _require_file(
        Path(str(planning.get("abi_registry") or "")),
        "AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID",
    )
    try:
        registry = load_observation_abi_registry(
            registry_path, root=_paths(config)["project_root"]
        )
        plan, extensions = build_adaptive_probe_plan(
            observation_work_order=order,
            registry=registry,
            root=_paths(config)["project_root"],
        )
    except AdaptiveObservationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_PLAN_INVALID:{exc}"
        ) from exc
    plan_path = attempt_root / "adaptive-probe-plan.json"
    _write_json_idempotent(plan_path, plan)
    portrait = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=_state_root_from_attempt(attempt_root),
    )
    manufacturing_orders = []
    for task in plan["tasks"]:
        if task.get("task_type") != "manufacture_shadow_probe":
            continue
        try:
            manufacturing_order = build_observation_manufacturing_work_order(
                portrait=portrait,
                adaptive_probe_plan=plan,
                observation_task_id=str(task["task_id"]),
                observation_registry=registry,
                root=_paths(config)["project_root"],
            )
        except ModuleManufacturingError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_OBSERVATION_MANUFACTURING_ORDER_INVALID:{exc}"
            ) from exc
        manufacturing_path = (
            attempt_root
            / "module-manufacturing-work-orders"
            / f"{manufacturing_order['work_order_id']}.json"
        )
        _write_json_idempotent(manufacturing_path, manufacturing_order)
        manufacturing_orders.append(
            {
                "work_order_id": manufacturing_order["work_order_id"],
                "work_order_path": str(manufacturing_path),
                "work_order_sha256": hashlib.sha256(
                    manufacturing_path.read_bytes()
                ).hexdigest(),
                "observation_task_id": task["task_id"],
            }
        )
    extension_paths = []
    for extension in extensions:
        extension_path = (
            attempt_root.parent
            / "interface-extensions"
            / f"{extension['extension_id']}.json"
        )
        _write_json_idempotent(extension_path, extension)
        extension_paths.append(str(extension_path))
    next_state = _observation_next_state(plan)
    payload = {
        "adaptive_probe_plan_id": plan["plan_id"],
        "adaptive_probe_plan_path": str(plan_path),
        "adaptive_probe_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "observation_next_state": next_state,
        "observation_task_types": sorted(
            {str(row["task_type"]) for row in plan["tasks"]}
        ),
        "interface_extension_paths": extension_paths,
        "observation_manufacturing_work_orders": manufacturing_orders,
        "intervention_gpu_authority": False,
    }
    return StageResult(
        state="completed" if plan["state"] == "ready_for_observation" else "blocked",
        outcome=str(plan["state"]),
        payload=payload,
        receipt_path=plan_path,
    )


def execute_observation(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Execute admitted observation ABIs and derive one append-only portrait."""

    execution = config.get("observation_execution")
    if not isinstance(execution, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_REQUIRED"
        )
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    state_root = _state_root_from_attempt(attempt_root)
    if state_root is None:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_STATE_ROOT_INVALID")
    resumed = _resume_observation_completion(
        config=config,
        state_root=state_root,
        completion_path=attempt_root.parent / "completion.json",
    )
    if resumed is not None:
        return resumed
    plan = _load_bound_context_document(
        context,
        path_key="adaptive_probe_plan_path",
        sha256_key="adaptive_probe_plan_sha256",
        id_key="adaptive_probe_plan_id",
        document_id="plan_id",
        code="AUTONOMOUS_ADAPTIVE_PROBE_PLAN",
    )
    try:
        validate_adaptive_probe_plan(plan, root=_paths(config)["project_root"])
    except AdaptiveObservationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_ADAPTIVE_PROBE_PLAN_INVALID:{exc}"
        ) from exc
    observation_order = _load_bound_observation_order(context)
    registry_path = _require_file(
        Path(str(config["observation_planning"]["abi_registry"])),  # type: ignore[index]
        "AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID",
    )
    try:
        registry = load_observation_abi_registry(
            registry_path, root=_paths(config)["project_root"]
        )
    except AdaptiveObservationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID:{exc}"
        ) from exc
    if (
        plan.get("registry_id") != registry.get("registry_id")
        or plan.get("registry_digest") != registry.get("registry_digest")
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_REGISTRY_BINDING_MISMATCH"
        )
    parent = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=state_root,
    )
    if parent.get("portrait_id") != plan.get("portrait_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_PORTRAIT_BINDING_MISMATCH"
        )
    tasks = [
        row
        for row in plan["tasks"]
        if row.get("task_type") in {"reuse_existing_probe", "run_read_only_adapter"}
    ]
    if not tasks or len(tasks) != len(plan["tasks"]):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTABLE_TASKS_INVALID"
        )
    adapters = execution.get("adapters")
    runtime_bindings = execution.get("runtime_bindings")
    if not isinstance(adapters, Mapping) or not isinstance(runtime_bindings, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_CONFIG_INVALID"
        )
    fingerprints = []
    structural_rows = []
    evidence_refs: set[str] = set()
    execution_rows = []
    blockers: list[dict[str, object]] = []
    for task in sorted(tasks, key=lambda row: str(row["task_id"])):
        abi_id = str(task.get("abi_id") or "")
        adapter = adapters.get(abi_id)
        if not isinstance(adapter, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_ADAPTER_BINDING_MISSING:" + abi_id
            )
        bindings = runtime_bindings.get(abi_id, {})
        if not isinstance(bindings, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_RUNTIME_BINDING_INVALID:" + abi_id
            )
        requirement = _probe_requirement_for_task(
            task=task, observation_order=observation_order
        )
        request = _observation_execution_request(
            plan=plan,
            portrait=parent,
            task=task,
            probe_requirement=requirement,
            runtime_bindings=bindings,
        )
        transaction_root = (
            attempt_root.parent / "transactions" / str(task["task_id"])
        )
        lease = None
        try:
            if adapter.get("resource_class") == "diagnostic_gpu":
                lease = GpuLeaseManager(
                    lock_root=Path(str(config["gpu_lock_root"]))
                ).acquire(
                    config["gpu_indices"],  # type: ignore[arg-type]
                    wait_seconds=float(config["gpu_wait_seconds"]),
                )
            manifest = run_observation_adapter(
                request=request,
                adapter=adapter,
                output_root=transaction_root,
                project_root=_paths(config)["project_root"],
                gpu_environment=lease.environment() if lease is not None else None,
                gpu_lease=lease.to_document() if lease is not None else None,
            )
        except ObservationAdapterError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_OBSERVATION_ADAPTER_INVALID:{exc}"
            ) from exc
        finally:
            if lease is not None:
                lease.release()
        execution_rows.append(
            {
                "task_id": task["task_id"],
                "abi_id": abi_id,
                "state": manifest["state"],
                "manifest_path": str(transaction_root / "manifest.json"),
                "manifest_sha256": hashlib.sha256(
                    (transaction_root / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
        if manifest.get("state") != "completed":
            blockers.extend(
                dict(row)
                for row in manifest.get("blockers", [])
                if isinstance(row, Mapping)
            )
            continue
        response_path = _require_file(
            Path(str(manifest.get("response_path") or "")),
            "AUTONOMOUS_OBSERVATION_RESPONSE_INVALID",
        )
        if hashlib.sha256(response_path.read_bytes()).hexdigest() != manifest.get(
            "response_sha256"
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_RESPONSE_HASH_MISMATCH"
            )
        response = _load(response_path, "AUTONOMOUS_OBSERVATION_RESPONSE_INVALID")
        _validate_observation_response(
            task=task,
            requirement=requirement,
            response=response,
        )
        response_ref = _archive_observation_bytes(
            execution,
            response_path.read_bytes(),
            project_root=_paths(config)["project_root"],
        )
        receipt_path = _require_file(
            Path(str(manifest["receipt_path"])),
            "AUTONOMOUS_OBSERVATION_RECEIPT_INVALID",
        )
        receipt_ref = _archive_observation_bytes(
            execution,
            receipt_path.read_bytes(),
            project_root=_paths(config)["project_root"],
        )
        evidence_refs.update((response_ref, receipt_ref))
        if response["observation_kind"] == "probe_fingerprint":
            probe = response["probe_observation"]
            assert isinstance(probe, Mapping)
            payload_ref = _archive_observation_bytes(
                execution,
                _canonical_any_bytes(probe["response_payload"]),
                project_root=_paths(config)["project_root"],
            )
            evidence_refs.add(payload_ref)
            fingerprints.append(
                build_probe_fingerprint_summary(
                    model_capability_id=str(parent["model_capability_id"]),
                    model_family=str(parent["model_family"]),
                    probe_protocol_id=str(probe["probe_protocol_id"]),
                    probe_protocol_version=str(probe["probe_protocol_version"]),
                    diagnostic_role=str(probe["diagnostic_role"]),
                    context_class=str(probe["context_class"]),
                    split=str(probe["split"]),
                    horizons=tuple(probe["horizons"]),
                    dose_values=tuple(probe["dose_values"]),
                    replication_count=int(probe["replication_count"]),
                    response_dimension=int(probe["response_dimension"]),
                    response_summary=str(probe["response_summary"]),
                    response_digest=payload_ref,
                    uncertainty_summary=str(probe["uncertainty_summary"]),
                    evidence_refs=(response_ref, receipt_ref),
                )
            )
        else:
            structural = response["structural_observation"]
            assert isinstance(structural, Mapping)
            structural_rows.append(structural)
    if blockers:
        blocker = {
            "schema_version": 1,
            "artifact_type": "verdiwm-observation-execution-blocker",
            "state": "blocked",
            "plan_id": plan["plan_id"],
            "executions": execution_rows,
            "blockers": blockers,
            "claim_boundary": (
                "No blocked observation result changes the active portrait or grants "
                "intervention, evaluator, verdict, or promotion authority."
            ),
        }
        blocker_path = attempt_root / "observation-execution-blocker.json"
        _write_json_idempotent(blocker_path, blocker)
        return StageResult(
            state="blocked",
            outcome="observation_execution_blocked",
            payload={
                "observation_execution_next_state": "observation_execution_blocked",
                "observation_execution_blocker_path": str(blocker_path),
                "observation_execution_gpu_authority": False,
            },
            receipt_path=blocker_path,
        )
    structural_observation = _merge_structural_observations(structural_rows)
    bundle = {
        "schema_version": 1,
        "artifact_type": "verdiwm-observation-evidence-bundle",
        "plan_id": plan["plan_id"],
        "parent_portrait_id": parent["portrait_id"],
        "executions": execution_rows,
        "evidence_refs": sorted(evidence_refs),
    }
    transition_ref = _archive_observation_bytes(
        execution,
        _canonical_bytes(bundle),
        project_root=_paths(config)["project_root"],
    )
    evidence_refs.add(transition_ref)
    try:
        portrait = update_model_portrait_from_observation(
            parent_portrait=parent,
            transition_ref=transition_ref,
            fingerprints=tuple(fingerprints),
            structural_observation=structural_observation,
            evidence_refs=tuple(sorted(evidence_refs)),
            root=_paths(config)["project_root"],
        )
        transition = build_portrait_transition(
            parent_portrait=parent,
            portrait=portrait,
            embodiment_id="observation-plan:" + str(plan["plan_id"]),
            outcome_state="admitted",
            evidence_refs=tuple(sorted(evidence_refs)),
            root=_paths(config)["project_root"],
        )
    except (ModelPortraitError, CommunityKnowledgeError) as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_PORTRAIT_UPDATE_INVALID:{exc}"
        ) from exc
    transition_root = state_root / "portrait-transitions"
    portrait_path = transition_root / f"{portrait['portrait_id']}.json"
    transition_path = transition_root / f"{transition['transition_id']}.json"
    _write_json_idempotent(portrait_path, portrait)
    _write_json_idempotent(transition_path, transition)
    portable_root = Path(
        str(
            config.get("portable_knowledge_records_root")
            or state_root / "portable-knowledge-records"
        )
    ).expanduser().resolve()
    staging = stage_portable_knowledge_records(
        documents=[portrait, transition, *fingerprints], output_root=portable_root
    )
    payload = {
        "observation_execution_next_state": "pending_portrait",
        "observation_execution_plan_id": plan["plan_id"],
        "observation_execution_plan_path": str(
            context["adaptive_probe_plan_path"]
        ),
        "observation_execution_plan_sha256": str(
            context["adaptive_probe_plan_sha256"]
        ),
        "observation_execution_records": execution_rows,
        "observation_parent_portrait_id": parent["portrait_id"],
        "observation_portrait_id": portrait["portrait_id"],
        "observation_portrait_path": str(portrait_path),
        "observation_portrait_sha256": hashlib.sha256(
            portrait_path.read_bytes()
        ).hexdigest(),
        "observation_transition_id": transition["transition_id"],
        "observation_transition_path": str(transition_path),
        "observation_transition_sha256": hashlib.sha256(
            transition_path.read_bytes()
        ).hexdigest(),
        "portable_observation_staging": staging,
        "observation_execution_gpu_authority": False,
    }
    completion = {
        "schema_version": 1,
        "artifact_type": "verdiwm-observation-execution-completion",
        "state": "completed",
        "plan_id": plan["plan_id"],
        "parent_portrait_id": parent["portrait_id"],
        "portrait_id": portrait["portrait_id"],
        "portrait_path": str(portrait_path),
        "portrait_sha256": hashlib.sha256(portrait_path.read_bytes()).hexdigest(),
        "transition_id": transition["transition_id"],
        "transition_path": str(transition_path),
        "transition_sha256": hashlib.sha256(
            transition_path.read_bytes()
        ).hexdigest(),
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "payload": payload,
    }
    completion_path = attempt_root.parent / "completion.json"
    _write_json_idempotent(completion_path, completion)
    _replace_json_atomic(state_root / "active-portrait.json", portrait)
    return StageResult(
        state="completed",
        outcome="portrait_observation_admitted",
        payload=payload,
        receipt_path=completion_path,
    )


def admit_shadow_probe(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Compile novel observation modules without activating their evidence."""

    execution = config.get("observation_execution")
    context = work.get("context")
    if not isinstance(execution, Mapping) or not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_SHADOW_PROBE_CONFIG_INVALID"
        )
    adapter = execution.get("shadow_llm_adapter")
    if not isinstance(adapter, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_SHADOW_PROBE_ADAPTER_REQUIRED"
        )
    plan = _load_bound_context_document(
        context,
        path_key="adaptive_probe_plan_path",
        sha256_key="adaptive_probe_plan_sha256",
        id_key="adaptive_probe_plan_id",
        document_id="plan_id",
        code="AUTONOMOUS_ADAPTIVE_PROBE_PLAN",
    )
    observation_order = _load_bound_observation_order(context)
    rows = context.get("observation_manufacturing_work_orders")
    if not isinstance(rows, list) or not rows:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_SHADOW_PROBE_WORK_ORDERS_MISSING"
        )
    registry_path = Path(str(config["observation_planning"]["abi_registry"]))  # type: ignore[index]
    manifests = []
    blockers = []
    for row in sorted(
        (value for value in rows if isinstance(value, Mapping)),
        key=lambda value: str(value["work_order_id"]),
    ):
        work_order_path = _require_file(
            Path(str(row.get("work_order_path") or "")),
            "AUTONOMOUS_SHADOW_PROBE_WORK_ORDER_INVALID",
        )
        expected = row.get("work_order_sha256")
        if not isinstance(expected, str) or hashlib.sha256(
            work_order_path.read_bytes()
        ).hexdigest() != expected:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_SHADOW_PROBE_WORK_ORDER_HASH_MISMATCH"
            )
        task_id = str(row.get("observation_task_id") or "")
        task = next(
            (
                value
                for value in plan["tasks"]
                if value.get("task_id") == task_id
            ),
            None,
        )
        if not isinstance(task, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_SHADOW_PROBE_TASK_BINDING_INVALID"
            )
        requirement = _probe_requirement_for_task(
            task=task, observation_order=observation_order
        )
        if requirement is None:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_SHADOW_PROBE_REQUIREMENT_MISSING"
            )
        try:
            manifest = compile_shadow_probe_admission(
                manufacturing_work_order_path=work_order_path,
                adaptive_probe_plan_path=Path(
                    str(context["adaptive_probe_plan_path"])
                ),
                observation_registry_path=registry_path,
                probe_requirement=requirement,
                adapter=adapter,
                output_root=(
                    attempt_root.parent
                    / "transactions"
                    / str(row["work_order_id"])
                ),
                project_root=_paths(config)["project_root"],
            )
        except ShadowProbeAdmissionError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_SHADOW_PROBE_ADMISSION_INVALID:{exc}"
            ) from exc
        manifests.append(manifest)
        blockers.extend(
            dict(value)
            for value in manifest.get("blockers", [])
            if isinstance(value, Mapping)
        )
    admitted = all(
        row.get("state") == "admitted_for_shadow_execution" for row in manifests
    )
    next_state = (
        "shadow_probe_evaluator_binding_required"
        if admitted
        else "shadow_probe_admission_blocked"
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "verdiwm-shadow-probe-admission-summary",
        "state": "completed" if admitted else "blocked",
        "plan_id": plan["plan_id"],
        "admissions": manifests,
        "blockers": blockers,
        "active_portrait_coverage_changed": False,
        "claim_boundary": (
            "Shadow admission never changes active portrait coverage. A separately "
            "admitted protocol and evaluator binding are required before reuse."
        ),
    }
    summary_path = attempt_root / "shadow-probe-admission-summary.json"
    _write_json_idempotent(summary_path, summary)
    return StageResult(
        state="completed" if admitted else "blocked",
        outcome=next_state,
        payload={
            "shadow_probe_admission_next_state": next_state,
            "shadow_probe_admission_summary_path": str(summary_path),
            "shadow_probe_candidate_ids": [
                str(row["candidate_id"])
                for row in manifests
                if row.get("candidate_id") is not None
            ],
            "shadow_probe_active_portrait_coverage": False,
            "shadow_probe_gpu_authority": False,
        },
        receipt_path=summary_path,
    )


def plan_capability_gaps(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Compile one ready portrait into exact reuse, composition, or leaf gaps."""

    planning = config.get("gap_planning")
    if not isinstance(planning, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_PLANNING_REQUIRED")
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    if context.get("portrait_gate_state") != "ready_for_gap_planning":
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_PORTRAIT_NOT_READY")
    goal_binding = context.get("portrait_goal_binding")
    if not isinstance(goal_binding, str):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_GOAL_BINDING_INVALID")
    portrait = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=_state_root_from_attempt(attempt_root),
    )
    idea = _load(
        _require_file(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID"),
        "AUTONOMOUS_IDEA_INVALID",
    )
    requirements = _gap_profile_requirements(
        planning, profile_id=str(work["profile_id"])
    )
    objective = str(idea.get("objective") or "")
    if not objective.strip():
        objective = (
            "Resolve the exact capability closure for transfer profile "
            + str(work["profile_id"])
        )
    try:
        goal_ir = build_goal_ir(
            goal_id=str(config["loop_id"]),
            goal_binding=goal_binding,
            model_family=str(portrait["model_family"]),
            objective=objective,
            requirements=requirements,
            root=_paths(config)["project_root"],
        )
        knowledge_graph = _bound_portable_knowledge_graph(planning)
        materializers = config.get("materializers")
        registered = (
            isinstance(materializers, Mapping)
            and isinstance(materializers.get(str(work["profile_id"])), Mapping)
        )
        result = compile_capability_gap_plan(
            goal_ir=goal_ir,
            portrait=portrait,
            abi_registry_path=Path(str(planning["abi_registry"])),
            expected_registry_digest=str(planning["abi_registry_digest"]),
            maximum_authority_level=str(planning["maximum_authority_level"]),
            admitted_abi_ids=tuple(planning["admitted_abi_ids"]),
            manufacturable_capabilities=(
                tuple(str(row["capability"]) for row in requirements)
                if registered
                else ()
            ),
            available_data_regimes=tuple(planning["available_data_regimes"]),
            kernel_capabilities=tuple(planning["kernel_capabilities"]),
            portable_knowledge_graph=knowledge_graph,
            root=_paths(config)["project_root"],
        )
    except CapabilityGapPlannerError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_GAP_PLAN_INVALID:{exc}"
        ) from exc
    goal_path = attempt_root / "goal-ir.json"
    graph_path = attempt_root / "capability-requirement-graph.json"
    plan_path = attempt_root / "capability-gap-plan.json"
    _write_json_idempotent(goal_path, result["goal_ir"])
    _write_json_idempotent(graph_path, result["requirement_graph"])
    _write_json_idempotent(plan_path, result["gap_plan_receipt"])
    composition_paths = []
    for composition in result["composition_receipts"]:
        path = (
            attempt_root
            / "module-compositions"
            / f"{composition['composition_id']}.json"
        )
        _write_json_idempotent(path, composition)
        composition_paths.append(str(path))
    receipt = result["gap_plan_receipt"]
    next_state = _gap_next_state(receipt)
    payload = {
        "goal_ir_id": goal_ir["goal_ir_id"],
        "goal_ir_path": str(goal_path),
        "goal_ir_sha256": hashlib.sha256(goal_path.read_bytes()).hexdigest(),
        "capability_requirement_graph_id": result["requirement_graph"]["graph_id"],
        "capability_requirement_graph_path": str(graph_path),
        "capability_requirement_graph_sha256": hashlib.sha256(
            graph_path.read_bytes()
        ).hexdigest(),
        "capability_gap_plan_id": receipt["plan_id"],
        "capability_gap_plan_state": receipt["state"],
        "capability_gap_plan_path": str(plan_path),
        "capability_gap_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "capability_gap_next_state": next_state,
        "module_composition_paths": composition_paths,
        "gap_planning_gpu_authority": False,
    }
    return StageResult(
        state=(
            "completed"
            if receipt["state"] in {"ready_for_portfolio", "requires_manufacturing"}
            else "blocked"
        ),
        outcome=str(receipt["state"]),
        payload=payload,
        receipt_path=plan_path,
    )


def plan_experiment_portfolio(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Compile a discriminating, evidence-bound portfolio without GPU authority."""

    planning = config.get("portfolio_planning")
    if not isinstance(planning, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTFOLIO_PLANNING_REQUIRED")
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    portrait = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=_state_root_from_attempt(attempt_root),
    )
    goal_ir = _load_bound_context_document(
        context,
        path_key="goal_ir_path",
        sha256_key="goal_ir_sha256",
        id_key="goal_ir_id",
        document_id="goal_ir_id",
        code="AUTONOMOUS_GOAL_IR",
    )
    graph = _load_bound_context_document(
        context,
        path_key="capability_requirement_graph_path",
        sha256_key="capability_requirement_graph_sha256",
        id_key="capability_requirement_graph_id",
        document_id="graph_id",
        code="AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH",
    )
    gap_plan = _load_bound_gap_plan(
        work,
        project_root=_paths(config)["project_root"],
    )
    hypothesis_batch = _load_bound_hypothesis_batch(
        planning,
        project_root=_paths(config)["project_root"],
    )
    graph_capabilities = {
        str(row["capability"])
        for row in graph.get("nodes", [])
        if isinstance(row, Mapping) and row.get("capability") is not None
    }
    candidates = hypothesis_batch.get("candidates")
    if not isinstance(candidates, list):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_HYPOTHESIS_CANDIDATES_INVALID"
        )
    scoped_candidates = [
        row
        for row in candidates
        if isinstance(row, Mapping)
        and set(str(value) for value in row.get("required_module_capabilities", []))
        <= graph_capabilities
    ]
    if not scoped_candidates:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_PORTFOLIO_NO_HYPOTHESIS_FOR_WORK_ITEM"
        )
    hypothesis_batch = {**hypothesis_batch, "candidates": scoped_candidates}
    try:
        portfolio = compile_experiment_portfolio(
            goal_ir=goal_ir,
            portrait=portrait,
            requirement_graph=graph,
            gap_plan=gap_plan,
            hypothesis_batch=hypothesis_batch,
            policy_id=str(planning["policy_id"]),
            maximum_hypotheses=int(planning["maximum_hypotheses"]),
            max_total_gpu_hours=float(planning["max_total_gpu_hours"]),
            minimum_replications=int(planning["minimum_replications"]),
            baseline_gpu_hours=float(planning["baseline_gpu_hours"]),
            control_cost_fraction=float(planning["control_cost_fraction"]),
            ablation_cost_fraction=float(planning["ablation_cost_fraction"]),
            protected_metrics=tuple(planning["protected_metrics"]),
            heldout_protocol=str(planning["heldout_protocol"]),
            required_artifact_classes=tuple(planning["required_artifact_classes"]),
            root=_paths(config)["project_root"],
        )
    except ExperimentPortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_EXPERIMENT_PORTFOLIO_INVALID:{exc}"
        ) from exc
    portfolio_path = attempt_root / "experiment-portfolio.json"
    _write_json_idempotent(portfolio_path, portfolio)
    next_states = {
        "manufacture_modules": "pending_materialization",
        "resource_admission": "pending_resource_admission",
        "stop_budget": "portfolio_budget_blocked",
    }
    next_state = next_states[str(portfolio["next_action"])]
    payload = {
        "experiment_portfolio_id": portfolio["portfolio_id"],
        "experiment_portfolio_path": str(portfolio_path),
        "experiment_portfolio_sha256": hashlib.sha256(
            portfolio_path.read_bytes()
        ).hexdigest(),
        "experiment_portfolio_state": portfolio["state"],
        "experiment_portfolio_next_state": next_state,
        "portfolio_selected_hypotheses": portfolio["budget"][
            "selected_hypotheses"
        ],
        "portfolio_gpu_authority": False,
    }
    return StageResult(
        state=(
            "completed"
            if portfolio["state"] == "ready_for_resource_admission"
            else "blocked"
        ),
        outcome=str(portfolio["state"]),
        payload=payload,
        receipt_path=portfolio_path,
    )


def admit_screen_resources(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Bind one materialized candidate to the campaign's portfolio GPU partition."""

    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    candidate_id = context.get("candidate_id")
    candidate_catalog = context.get("candidate_catalog_path")
    if not isinstance(candidate_id, str) or not candidate_id or not isinstance(
        candidate_catalog, str
    ):
        blocker = {
            "schema_version": 1,
            "artifact_type": "verdiwm-resource-portfolio-blocker",
            "state": "resource_binding_required",
            "work_id": work["work_id"],
            "reason": "RESOURCE_PORTFOLIO_EXECUTION_BINDING_MISSING",
            "gpu_execution_started": False,
            "claim_boundary": (
                "A reusable capability without an exact candidate embodiment cannot receive "
                "GPU authority. The controller retains the gap instead of substituting a module."
            ),
        }
        blocker_path = attempt_root / "resource-portfolio-blocker.json"
        _write_json_idempotent(blocker_path, blocker)
        return StageResult(
            state="blocked",
            outcome="resource_binding_required",
            payload={
                "resource_portfolio_next_state": "resource_binding_required",
                "resource_portfolio_blocker_path": str(blocker_path),
                "resource_gpu_authority": False,
            },
            receipt_path=blocker_path,
        )
    _require_file(
        Path(candidate_catalog), "AUTONOMOUS_CANDIDATE_CATALOG_INVALID"
    )
    paths = _paths(config)
    gap_plan = _load_bound_gap_plan(work, project_root=paths["project_root"])
    portfolio = _load_bound_portfolio(
        work,
        gap_plan=gap_plan,
        project_root=paths["project_root"],
    )
    if portfolio.get("next_action") not in {
        "resource_admission",
        "manufacture_modules",
    }:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_PORTFOLIO_AUTHORITY_REQUIRED"
        )
    policy, experiment_scale, conversion_scale = _load_resource_policy(
        config,
        project_root=paths["project_root"],
    )
    entry_ids = _manufacturing_portfolio_entry_ids(
        context,
        project_root=paths["project_root"],
    )
    try:
        receipt = build_screen_resource_portfolio_receipt(
            portfolio=portfolio,
            work_id=str(work["work_id"]),
            candidate_id=candidate_id,
            config_digest=str(config["config_digest"]),
            resource_allocation=policy["resource_allocation"],
            experiment_scale_plan=experiment_scale,
            conversion_scale_plan=conversion_scale,
            policy_id=str(policy["policy_id"]),
            portfolio_entry_ids=entry_ids,
            root=paths["project_root"],
        )
    except ResourcePortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_RESOURCE_PORTFOLIO_INVALID:{exc}"
        ) from exc
    receipt_path = attempt_root / "resource-portfolio-screen.json"
    _write_json_idempotent(receipt_path, receipt)
    return StageResult(
        state="completed",
        outcome="ready_for_screen",
        payload={
            "resource_portfolio_id": receipt["receipt_id"],
            "resource_portfolio_path": str(receipt_path),
            "resource_portfolio_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            "resource_portfolio_phase": "screen_admission",
            "resource_portfolio_next_state": "pending_screen",
            "resource_trial_id": receipt["allocation"]["selected_trial_id"],
            "resource_gpu_authority": True,
        },
        receipt_path=receipt_path,
    )


def reallocate_confirm_resources(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Admit confirmation only after exact accepted screen evidence is settled."""

    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    if context.get("screen_accepted") is not True:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CONFIRM_SCREEN_ACCEPTANCE_REQUIRED"
        )
    paths = _paths(config)
    screen_receipt = _load_bound_resource_receipt(
        context,
        phase="screen_admission",
        project_root=paths["project_root"],
    )
    evidence_sha256 = context.get("screen_stage_receipt_sha256")
    if not isinstance(evidence_sha256, str):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_SCREEN_STAGE_RECEIPT_BINDING_REQUIRED"
        )
    screen_evidence = {
        "schema_version": 1,
        "artifact_type": "verdiwm-resource-screen-evidence",
        "state": "settled",
        "work_id": work["work_id"],
        "trial_id": screen_receipt["allocation"]["selected_trial_id"],
        "decision": "accepted",
        "evidence_ref": "sha256:" + evidence_sha256,
    }
    evidence_path = attempt_root / "screen-evidence.json"
    _write_json_idempotent(evidence_path, screen_evidence)
    policy, _, _ = _load_resource_policy(
        config,
        project_root=paths["project_root"],
    )
    requested = _confirm_gpu_count(policy, profile_id=str(work["profile_id"]))
    training_scale = None
    rationale = None
    if requested > 1:
        training_scale = _load_profile_training_scale_plan(
            policy,
            profile_id=str(work["profile_id"]),
        )
        rationales = policy.get("scale_rationales")
        if not isinstance(rationales, Mapping) or not isinstance(
            rationales.get(str(work["profile_id"])), Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_DISTRIBUTED_SCALE_RATIONALE_REQUIRED"
            )
        rationale = rationales[str(work["profile_id"])]
    try:
        receipt = build_confirm_resource_portfolio_receipt(
            screen_receipt=screen_receipt,
            screen_evidence=screen_evidence,
            requested_gpu_count=requested,
            training_scale_plan=training_scale,
            scale_rationale=rationale,
            root=paths["project_root"],
        )
    except ResourcePortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_CONFIRM_RESOURCE_REALLOCATION_INVALID:{exc}"
        ) from exc
    receipt_path = attempt_root / "resource-portfolio-confirm.json"
    _write_json_idempotent(receipt_path, receipt)
    return StageResult(
        state="completed",
        outcome="ready_for_confirm",
        payload={
            "resource_portfolio_id": receipt["receipt_id"],
            "resource_portfolio_path": str(receipt_path),
            "resource_portfolio_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            "resource_portfolio_phase": "confirm_reallocation",
            "resource_portfolio_next_state": "pending_confirm",
            "resource_trial_id": receipt["allocation"]["selected_trial_id"],
            "screen_evidence_path": str(evidence_path),
            "screen_evidence_sha256": hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest(),
            "resource_gpu_authority": True,
        },
        receipt_path=receipt_path,
    )


def materialize(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    materializers = config.get("materializers")
    if not isinstance(materializers, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MATERIALIZER_REGISTRY_INVALID")
    registration = materializers.get(str(work["profile_id"]))
    open_generation = config.get("open_method_generation")
    if not isinstance(registration, Mapping) and isinstance(open_generation, Mapping):
        return _prepare_open_method(
            config,
            work=work,
            attempt_root=attempt_root,
            settings=open_generation,
        )
    gap_plan = None
    portfolio = None
    if isinstance(config.get("gap_planning"), Mapping):
        gap_plan = _load_bound_gap_plan(
            work,
            project_root=_paths(config)["project_root"],
        )
        if gap_plan.get("state") != "requires_manufacturing":
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_MATERIALIZATION_GAP_AUTHORITY_REQUIRED"
            )
        if isinstance(config.get("portfolio_planning"), Mapping):
            portfolio = _load_bound_portfolio(
                work,
                gap_plan=gap_plan,
                project_root=_paths(config)["project_root"],
            )
            if portfolio.get("next_action") != "manufacture_modules":
                raise AutonomousTransferWorkflowError(
                    "AUTONOMOUS_MATERIALIZATION_PORTFOLIO_AUTHORITY_REQUIRED"
                )
    automatic_manifest: Mapping[str, object] | None = None
    manufacturing_order: Mapping[str, object] | None = None
    manufacturing_order_path: Path | None = None
    if not isinstance(registration, Mapping):
        automatic = config.get("automatic_module_generation")
        if not isinstance(automatic, Mapping):
            return _materializer_unavailable(work=work, attempt_root=attempt_root)
        if gap_plan is None or portfolio is None:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_MODULE_MANUFACTURING_EVIDENCE_REQUIRED"
            )
        context = work.get("context")
        if not isinstance(context, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
        goal_ir = _load_bound_context_document(
            context,
            path_key="goal_ir_path",
            sha256_key="goal_ir_sha256",
            id_key="goal_ir_id",
            document_id="goal_ir_id",
            code="AUTONOMOUS_GOAL_IR",
        )
        graph = _load_bound_context_document(
            context,
            path_key="capability_requirement_graph_path",
            sha256_key="capability_requirement_graph_sha256",
            id_key="capability_requirement_graph_id",
            document_id="graph_id",
            code="AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH",
        )
        requests = [
            row
            for row in gap_plan["manufacturing_requests"]
            if isinstance(row, Mapping) and row.get("abi_id") is not None
        ]
        if len(requests) != 1:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_MODULE_MANUFACTURING_REQUEST_AMBIGUOUS"
            )
        request = requests[0]
        composition = _load_manufacturing_composition(
            context,
            graph=graph,
            requirement_id=str(request["requirement_id"]),
        )
        portrait = _load_active_portrait(
            config,
            project_root=_paths(config)["project_root"],
            state_root=_state_root_from_attempt(attempt_root),
        )
        try:
            manufacturing_order = build_intervention_manufacturing_work_order(
                goal_ir=goal_ir,
                portrait=portrait,
                requirement_graph=graph,
                gap_plan=gap_plan,
                portfolio=portfolio,
                manufacturing_request_id=str(request["request_id"]),
                composition_receipt=composition,
                abi_registry_path=Path(str(automatic["abi_registry"])),
                root=_paths(config)["project_root"],
            )
        except ModuleManufacturingError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MODULE_MANUFACTURING_ORDER_INVALID:{exc}"
            ) from exc
        manufacturing_order_path = (
            attempt_root.parent
            / "module-manufacturing-work-orders"
            / f"{manufacturing_order['work_order_id']}.json"
        )
        _write_json_idempotent(manufacturing_order_path, manufacturing_order)
        try:
            automatic_manifest = compile_automatic_module_plan(
                idea_path=Path(str(work["idea_path"])),
                work_order_path=Path(str(work["work_order_path"])),
                assessment_path=Path(str(work["assessment_path"])),
                model_capability_ir_path=Path(str(automatic["model_capability_ir"])),
                abi_registry_path=Path(str(automatic["abi_registry"])),
                adapter=automatic["llm_adapter"],
                runtime_python=_paths(config)["runtime_python"],
                output_root=(
                    attempt_root.parent
                    / "automatic-module-plans"
                    / f"attempt-{_attempt_number(attempt_root):03d}"
                ),
                project_root=_paths(config)["project_root"],
                manufacturing_work_order_path=manufacturing_order_path,
            )
        except AutomaticModulePlanError as exc:
            return _automatic_module_policy_gap(
                work=work, attempt_root=attempt_root, detail=str(exc)
            )
        if automatic_manifest.get("state") != "ready_for_materialization":
            gap_path = Path(str(automatic_manifest["capability_gap_path"]))
            return StageResult(
                state="blocked",
                outcome="automatic_module_capability_gap",
                payload={
                    "capability_gap_path": str(gap_path),
                    "automatic_module_manifest": dict(automatic_manifest),
                },
                receipt_path=gap_path,
            )
        if (
            automatic_manifest.get("abi_id")
            != manufacturing_order["target_abi"]["abi_id"]
            or automatic_manifest.get("manufacturing_work_order_id")
            != manufacturing_order["work_order_id"]
        ):
            return _automatic_module_policy_gap(
                work=work,
                attempt_root=attempt_root,
                detail="AUTOMATIC_MODULE_MANUFACTURING_BINDING_MISMATCH",
            )
        plan_path = _require_file(
            Path(str(automatic_manifest["plan_path"])),
            "AUTONOMOUS_AUTOMATIC_MODULE_PLAN_INVALID",
        )
        plan = _load(plan_path, "AUTONOMOUS_AUTOMATIC_MODULE_PLAN_INVALID")
    else:
        template = _load(
            _require_file(
                Path(str(registration["plan_template"])),
                "AUTONOMOUS_MATERIALIZER_TEMPLATE_INVALID",
            ),
            "AUTONOMOUS_MATERIALIZER_TEMPLATE_INVALID",
        )
        idea_path = _require_file(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID")
        idea = _load(idea_path, "AUTONOMOUS_IDEA_INVALID")
        plan = _bind_plan(template, work=work, idea=idea)
        plan_path = (
            attempt_root.parent
            / "plans"
            / f"attempt-{_attempt_number(attempt_root):03d}.json"
        )
        _write_json_idempotent(plan_path, plan)
    idea_path = _require_file(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID")
    manifest = run_automatic_materialization(
        plan_path=plan_path,
        work_order_path=Path(str(work["work_order_path"])),
        idea_path=idea_path,
        source_root=_paths(config)["ctrl_world_root"],
        output_root=attempt_root,
        project_root=_paths(config)["project_root"],
        manufacturing_work_order_path=manufacturing_order_path,
    )
    state = str(manifest["state"])
    payload = {
        "materialization_root": str(attempt_root),
        "materialization_plan_path": str(plan_path),
        "candidate_id": manifest["candidate_id"],
        "materialization_manifest": manifest,
    }
    if automatic_manifest is not None:
        payload.update(
            {
                "automatic_module_abi_id": automatic_manifest["abi_id"],
                "automatic_module_admission_path": automatic_manifest["admission_path"],
                "automatic_module_admission_sha256": automatic_manifest["admission_sha256"],
                "automatic_module_plan_manifest": dict(automatic_manifest),
                "module_manufacturing_work_order_id": manufacturing_order[
                    "work_order_id"
                ],
                "module_manufacturing_work_order_path": str(
                    manufacturing_order_path
                ),
                "module_manufacturing_work_order_sha256": hashlib.sha256(
                    manufacturing_order_path.read_bytes()
                ).hexdigest(),
                "module_manufacturing_request_id": manufacturing_order[
                    "manufacturing_request"
                ]["request_id"],
            }
        )
    if state == "ready_for_candidate_compilation":
        payload["candidate_catalog_path"] = manifest["candidate_catalog_path"]
        return StageResult(
            state="completed",
            outcome="ready_for_screen",
            payload=payload,
            receipt_path=Path(str(manifest["receipt_path"])),
        )
    payload["capability_gap_path"] = manifest.get("capability_gap_path")
    return StageResult(
        state="blocked",
        outcome="materialization_blocked",
        payload=payload,
        receipt_path=Path(str(manifest["receipt_path"])),
    )


def _prepare_open_method(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
    settings: Mapping[str, object],
) -> StageResult:
    """Ask the bounded LLM for an ABI-free method and compile its overlay."""

    paths = _paths(config)
    root = paths["project_root"]
    idea = _load(
        _require_file(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID"),
        "AUTONOMOUS_IDEA_INVALID",
    )
    assessment = _load(
        _require_file(
            Path(str(work["assessment_path"])), "AUTONOMOUS_ASSESSMENT_INVALID"
        ),
        "AUTONOMOUS_ASSESSMENT_INVALID",
    )
    portrait = _load_active_portrait(
        config,
        project_root=root,
        state_root=_state_root_from_attempt(attempt_root),
    )
    portrait_binding = {
        "portrait_id": portrait["portrait_id"],
        "portrait_digest": hashlib.sha256(_canonical_bytes(portrait)).hexdigest(),
    }
    current_fingerprints = [
        dict(row)
        for row in portrait.get("behavioral_fingerprints", [])
        if isinstance(row, Mapping) and row.get("state") == "current"
    ]
    probe_body = {
        "fingerprint_ids": sorted(str(row["fingerprint_id"]) for row in current_fingerprints),
        "coverage_keys": sorted(str(row["coverage_key"]) for row in current_fingerprints),
    }
    probe_binding = {
        **probe_body,
        "binding_digest": hashlib.sha256(_canonical_bytes(probe_body)).hexdigest(),
    }
    failure_context = sorted(
        {
            str(value)
            for source in (
                idea.get("failure_context", []),
                assessment.get("failure_context", []),
                work.get("context", {}).get("failure_context", [])
                if isinstance(work.get("context"), Mapping)
                else [],
            )
            if isinstance(source, list)
            for value in source
            if str(value).strip()
        }
    )
    request = build_open_method_request(
        source_evidence=_open_source_evidence(assessment),
        target_portrait=portrait,
        probe_fingerprints=current_fingerprints,
        failure_context=failure_context,
    )
    request["input"]["required_bindings"] = {
        "target_portrait_binding": portrait_binding,
        "probe_binding": probe_binding,
    }
    task = run_llm_task(
        request=request,
        adapter=settings["llm_adapter"],
        output_root=attempt_root / "llm-task",
        project_root=root,
    )
    if task.get("state") != "completed":
        return StageResult(
            state="blocked",
            outcome="open_method_generation_blocked",
            payload={
                "materialization_next_state": "pending_replan",
                "open_method_task_manifest": task,
            },
            receipt_path=Path(str(task["receipt_path"])),
        )
    response = _load(
        _require_file(
            Path(str(task["response_path"])), "AUTONOMOUS_OPEN_METHOD_RESPONSE_INVALID"
        ),
        "AUTONOMOUS_OPEN_METHOD_RESPONSE_INVALID",
    )
    proposal = response.get("output")
    if not isinstance(proposal, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_OPEN_METHOD_RESPONSE_INVALID")
    source_revision = portrait.get("source_revision")
    revision_value = (
        str(source_revision.get("value"))
        if isinstance(source_revision, Mapping)
        else str(portrait["model_capability_id"])
    )
    try:
        compilation = compile_open_method_proposal(
            proposal=proposal,
            base_revision={
                "revision": revision_value,
                "source_digest": hashlib.sha256(
                    _canonical_bytes(
                        {
                            "source_revision": source_revision,
                            "model_capability_digest": portrait["model_capability_digest"],
                        }
                    )
                ).hexdigest(),
            },
            output_root=attempt_root / "open-method-compilation",
            project_root=root,
            expected_portrait_binding=portrait_binding,
            expected_probe_binding=probe_binding,
        )
    except OpenMethodPipelineError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OPEN_METHOD_COMPILATION_INVALID:{exc}"
        ) from exc
    compilation_root = attempt_root / "open-method-compilation"
    payload: dict[str, object] = {
        "open_method_task_manifest": task,
        "open_method_compilation": compilation,
        "open_method_compilation_root": str(compilation_root),
        "method_id": compilation["method_id"],
        "overlay_id": compilation["overlay_id"],
        "candidate_execution_id": compilation["execution_id"],
    }
    if compilation["state"] == "ready_for_calibration":
        payload["materialization_next_state"] = "pending_open_method_calibration"
        return StageResult(
            state="completed",
            outcome="open_method_ready_for_calibration",
            payload=payload,
            receipt_path=compilation_root / "manifest.json",
        )
    next_states = {
        "interface_extension_required": "pending_interface_extension",
        "data_regime_missing": "missing_data_regime",
        "architecture_bound": "architecture_bound",
        "unmapped": "pending_replan",
    }
    payload["materialization_next_state"] = next_states.get(
        str(compilation["proposal_state"]), "pending_replan"
    )
    return StageResult(
        state="blocked",
        outcome="open_method_differentiated_gap",
        payload=payload,
        receipt_path=compilation_root / "manifest.json",
    )


def _open_source_evidence(
    assessment: Mapping[str, object],
) -> list[dict[str, object]]:
    source_id = str(assessment.get("source_id") or "unknown-source")
    source_digest = str(assessment.get("source_digest") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        source_digest = hashlib.sha256(_canonical_bytes(dict(assessment))).hexdigest()
    rows = assessment.get("source_evidence")
    evidence = []
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            claim = str(
                row.get("evidence_snippet")
                or row.get("claim")
                or row.get("description")
                or ""
            ).strip()
            if len(claim) < 12:
                continue
            evidence.append(
                {
                    "source_id": source_id,
                    "source_digest": source_digest,
                    "locator": str(
                        row.get("component_id")
                        or row.get("locator")
                        or f"assessment-evidence-{index + 1}"
                    ),
                    "claim": claim,
                }
            )
    if not evidence:
        fallback = str(
            assessment.get("target_intervention")
            or assessment.get("source_title")
            or "Source assessment proposes a target-relevant trainable mechanism."
        )
        if len(fallback) < 12:
            fallback = "Source assessment proposes a target-relevant trainable mechanism."
        evidence.append(
            {
                "source_id": source_id,
                "source_digest": source_digest,
                "locator": "source-assessment",
                "claim": fallback,
            }
        )
    return evidence


def calibrate_open_method(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    """Run the deployment's trusted candidate calibration broker.

    The broker is the only component allowed to execute arbitrary overlay code.
    Without one configured, the candidate remains a durable replanning item.
    """

    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    compilation_root = _require_file(
        Path(str(context.get("open_method_compilation_root") or "")) / "manifest.json",
        "AUTONOMOUS_OPEN_METHOD_COMPILATION_INVALID",
    ).parent
    compilation = _load(
        compilation_root / "manifest.json", "AUTONOMOUS_OPEN_METHOD_COMPILATION_INVALID"
    )
    if compilation.get("state") != "ready_for_calibration":
        raise AutonomousTransferWorkflowError("AUTONOMOUS_OPEN_METHOD_NOT_CALIBRATABLE")
    method = _load(compilation_root / "method-ir.json", "AUTONOMOUS_OPEN_METHOD_IR_INVALID")
    overlay = _load(
        compilation_root / "candidate-overlay.json",
        "AUTONOMOUS_OPEN_METHOD_OVERLAY_INVALID",
    )
    adapter_config = config.get("open_method_generation")
    adapter = (
        adapter_config.get("calibration_adapter")
        if isinstance(adapter_config, Mapping)
        else None
    )
    if not isinstance(adapter, Mapping):
        receipt = attempt_root / "calibration-blocked.json"
        payload = {
            "schema_version": 1,
            "artifact_type": "verdiwm-open-method-calibration",
            "state": "blocked",
            "method_id": method["method_id"],
            "overlay_id": overlay["overlay_id"],
            "blockers": [
                {
                    "code": "CALIBRATION_BACKEND_REQUIRED",
                    "detail": "Configure a trusted sandbox broker before arbitrary candidate execution.",
                }
            ],
            "claim_boundary": "No candidate code was executed and no execution authority was granted.",
        }
        _write_json_idempotent(receipt, payload)
        return StageResult(
            state="blocked",
            outcome="open_method_calibration_backend_required",
            payload={
                "open_method_calibration_next_state": "pending_replan",
                "open_method_calibration_receipt_path": str(receipt),
            },
            receipt_path=receipt,
        )
    request_body = {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task",
        "task_id": "calibration-" + hashlib.sha256(
            _canonical_bytes({"method_id": method["method_id"], "overlay_id": overlay["overlay_id"]})
        ).hexdigest()[:24],
        "task_type": "open_method_generation",
        "prompt_template_digest": hashlib.sha256(
            b"candidate-calibration-broker-v1"
        ).hexdigest(),
        "output_schema": "candidate_calibration",
        "input": {
            "method_ir": method,
            "candidate_overlay": overlay,
            "compilation_root": str(compilation_root),
            "instructions": (
                "Execute only inside the supplied candidate sandbox. Run declared CPU "
                "conformance checks, preserve source/evaluator authority, and return a "
                "schema-valid calibration receipt."
            ),
        },
    }
    task = run_llm_task(
        request=request_body,
        adapter=adapter,
        output_root=attempt_root / "broker-task",
        project_root=_paths(config)["project_root"],
    )
    if task.get("state") != "completed":
        return StageResult(
            state="blocked",
            outcome="open_method_calibration_broker_blocked",
            payload={
                "open_method_calibration_next_state": "pending_replan",
                "open_method_calibration_task_manifest": task,
            },
            receipt_path=Path(str(task["receipt_path"])),
        )
    response = _load(
        _require_file(Path(str(task["response_path"])), "AUTONOMOUS_CALIBRATION_RESPONSE_INVALID"),
        "AUTONOMOUS_CALIBRATION_RESPONSE_INVALID",
    )
    calibration = response.get("output")
    if not isinstance(calibration, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CALIBRATION_RESPONSE_INVALID")
    try:
        validate_document(
            "candidate_calibration", calibration, root=_paths(config)["project_root"]
        )
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_CALIBRATION_RECEIPT_INVALID:{exc}"
        ) from exc
    if (
        calibration.get("method_id") != method.get("method_id")
        or calibration.get("overlay_id") != overlay.get("overlay_id")
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CALIBRATION_BINDING_MISMATCH")
    if calibration.get("state") != "passed":
        return StageResult(
            state="blocked",
            outcome="open_method_calibration_failed",
            payload={
                "open_method_calibration_next_state": "pending_replan",
                "open_method_calibration": dict(calibration),
                "open_method_calibration_task_manifest": task,
            },
            receipt_path=Path(str(task["response_path"])),
        )
    catalog_path = calibration.get("candidate_catalog_path")
    if not isinstance(catalog_path, str) or not catalog_path:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CALIBRATION_CATALOG_MISSING")
    catalog = _require_file(Path(catalog_path), "AUTONOMOUS_OPEN_METHOD_CATALOG_INVALID")
    expected_catalog_sha = calibration.get("candidate_catalog_sha256")
    if not isinstance(expected_catalog_sha, str) or hashlib.sha256(
        catalog.read_bytes()
    ).hexdigest() != expected_catalog_sha:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_OPEN_METHOD_CATALOG_HASH_MISMATCH")
    catalog_payload = _load(catalog, "AUTONOMOUS_OPEN_METHOD_CATALOG_INVALID")
    try:
        validate_document(
            "method_candidate_catalog",
            catalog_payload,
            root=_paths(config)["project_root"],
        )
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OPEN_METHOD_CATALOG_INVALID:{exc}"
        ) from exc
    candidates = catalog_payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(
        candidates[0], Mapping
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OPEN_METHOD_CATALOG_CARDINALITY_INVALID"
        )
    candidate_id = candidates[0].get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_OPEN_METHOD_CANDIDATE_ID_INVALID")
    receipt = attempt_root / "calibration.json"
    _write_json_idempotent(receipt, dict(calibration))
    return StageResult(
        state="completed",
        outcome="open_method_calibration_passed",
        payload={
            "open_method_calibration_next_state": "pending_resource_admission",
            "candidate_catalog_path": str(catalog),
            "candidate_catalog_sha256": expected_catalog_sha,
            "open_method_calibration_receipt_path": str(receipt),
            "candidate_id": candidate_id,
        },
        receipt_path=receipt,
    )


def run_gpu_stage(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    stage: str,
    attempt_root: Path,
) -> StageResult:
    if stage not in {"screen", "confirm"}:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GPU_STAGE_INVALID")
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    resource_receipt = None
    if isinstance(config.get("resource_portfolio"), Mapping):
        resource_receipt = _load_bound_resource_receipt(
            context,
            phase=(
                "screen_admission"
                if stage == "screen"
                else "confirm_reallocation"
            ),
            project_root=_paths(config)["project_root"],
        )
        allocation = resource_receipt["allocation"]
        if allocation.get("stage") != stage:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_RESOURCE_PORTFOLIO_STAGE_MISMATCH"
            )
        requested_gpu_count = int(allocation["requested_gpu_count"])
        if requested_gpu_count < 1 or requested_gpu_count > len(allocation["allowed_gpu_indices"]):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_RESOURCE_GPU_COUNT_INVALID")
    catalog = _require_file(
        Path(str(context.get("candidate_catalog_path") or "")),
        "AUTONOMOUS_CANDIDATE_CATALOG_INVALID",
    )
    assessment = _require_file(
        Path(str(work["assessment_path"])), "AUTONOMOUS_ASSESSMENT_INVALID"
    )
    idea = _load(Path(str(work["idea_path"])), "AUTONOMOUS_IDEA_INVALID")
    paths = _paths(config)
    contract = _load(paths["contract"], "AUTONOMOUS_CONTRACT_INVALID")
    registration = _materializer_registration(config, work=work)
    evaluator = Path(str(registration.get("evaluator") or paths["evaluator"])).expanduser().resolve()
    _require_file(evaluator, "AUTONOMOUS_EVALUATOR_INVALID")
    stage_rows = [
        row
        for row in contract.get("stages", [])
        if isinstance(row, Mapping) and row.get("stage") == stage
    ]
    if len(stage_rows) != 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CONTRACT_STAGE_INVALID")
    campaign_root = attempt_root.parent.parent
    candidate_id = str(context["candidate_id"])
    batch_path = campaign_root / f"{stage}-batch.json"
    compilation_path = campaign_root / f"{stage}-compilation-report.json"
    compile_args = argparse.Namespace(
        catalog=catalog,
        assessment=assessment,
        contract=paths["contract"],
        stage=stage,
        batch_id=f"{candidate_id}-{stage}-v1",
        objective=str(idea["objective"]),
        hypothesis=str(idea["hypothesis"]),
        falsification_criterion=str(idea["falsification_criterion"]),
        selection_reason=(
            "Source classification, target capability checks, isolated materialization, "
            "and immutable candidate compilation all passed without compromise."
            if stage == "screen"
            else "The exact receipt-bound candidate passed screen without parameter or implementation changes."
        ),
        expected_gpu_hours=float(stage_rows[0]["max_gpu_hours_per_candidate"]),
        output=batch_path,
        compilation_report=compilation_path,
    )
    hybrid_campaign.compile_batch(compile_args)
    resource_plan = _training_resource_plan_for_batch(
        batch_path, allowed_gpu_indices=(
            [int(value) for value in resource_receipt["allocation"]["allowed_gpu_indices"]]
            if resource_receipt is not None
            else [int(value) for value in config["gpu_indices"]]
        ),
    )

    lock_root = Path(str(config["gpu_lock_root"])).expanduser().resolve()
    lease_manager = GpuLeaseManager(lock_root=lock_root)
    allowed = (
        [int(value) for value in resource_receipt["allocation"]["allowed_gpu_indices"]]
        if resource_receipt is not None
        else [int(value) for value in config["gpu_indices"]]
    )
    requested_gpu_count = int(
        resource_receipt["allocation"]["requested_gpu_count"]
        if resource_receipt is not None
        else 1
    )
    if requested_gpu_count > 1:
        acquire_many = getattr(lease_manager, "acquire_many", None)
        if not callable(acquire_many):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_DISTRIBUTED_EXECUTOR_NOT_BOUND")
        leases = acquire_many(
            allowed,
            requested_gpu_count,
            wait_seconds=float(config["gpu_wait_seconds"]),
            poll_seconds=min(5.0, max(1.0, float(config["poll_seconds"]))),
        )
    else:
        leases = [
            lease_manager.acquire(
                allowed,
                wait_seconds=float(config["gpu_wait_seconds"]),
                poll_seconds=min(5.0, max(1.0, float(config["poll_seconds"]))),
            )
        ]
    admission_path = (
        campaign_root
        / "gpu-admissions"
        / f"{stage}-attempt-{_attempt_number(attempt_root):03d}.json"
    )
    admission = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-gpu-admission",
        "state": "admitted",
        "work_id": work["work_id"],
        "stage": stage,
        "candidate_id": candidate_id,
        "config_digest": config["config_digest"],
        "allowed_gpu_indices": allowed,
        "global_gpu_limit": int(config["max_parallel_gpu_jobs"]),
        "resource_portfolio_id": (
            resource_receipt["receipt_id"] if resource_receipt is not None else None
        ),
        "resource_portfolio_digest": (
            resource_receipt["receipt_digest"]
            if resource_receipt is not None
            else None
        ),
        "resource_trial_id": (
            resource_receipt["allocation"]["selected_trial_id"]
            if resource_receipt is not None
            else None
        ),
        "lease": {
            "world_size": len(leases),
            "members": [lease.to_document() for lease in leases],
        },
        "training_resource_plan": resource_plan,
        "admitted_at": _utc_now(),
        "claim_boundary": (
            "This receipt grants one physical GPU lease for one frozen stage only; "
            "it grants no scientific or promotion authority."
        ),
    }
    _write_json_idempotent(admission_path, admission)
    baseline = paths["screen_baseline"] if stage == "screen" else paths["confirm_baseline"]
    training_payload: dict[str, object] = {}
    campaign_error: Exception | None = None
    lease_release_error: Exception | None = None
    lease_released = False
    summary: dict[str, object] | None = None
    try:
        training_payload = _ensure_masked_adapter_training(
            config,
            registration=registration,
            batch_path=batch_path,
            campaign_root=campaign_root,
            stage=stage,
            attempt_root=attempt_root,
            contract=contract,
            gpu_indices=[lease.index for lease in leases],
        )
        run_args = argparse.Namespace(
            contract=paths["contract"],
            batch=batch_path,
            baseline=baseline,
            evaluator=evaluator,
            base_evaluator=paths["base_evaluator"],
            runtime_python=paths["runtime_python"],
            ctrl_world_root=paths["ctrl_world_root"],
            dataset_root=paths["dataset_root"],
            data_stat=paths["data_stat"],
            checkpoint=paths["checkpoint"],
            svd_model=paths["svd_model"],
            clip_model=paths["clip_model"],
            output_root=attempt_root,
            gpu_indices=",".join(str(lease.index) for lease in leases),
            activity_probe_seconds=int(config["activity_probe_seconds"]),
            worker_timeout_seconds=float(config["worker_timeout_seconds"]),
            resume=attempt_root.exists(),
            dry_run=False,
        )
        summary = hybrid_campaign.run_campaign(run_args)
    except Exception as exc:
        campaign_error = exc
    finally:
        lease_document = {
            "world_size": len(leases),
            "members": [lease.to_document() for lease in leases],
        }
        try:
            for lease in leases:
                lease.release()
            lease_released = True
        except Exception as exc:
            lease_release_error = exc
        _write_json_idempotent(
            admission_path.with_name(admission_path.stem + "-release.json"),
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-gpu-release",
                "state": "released" if lease_released else "release_failed",
                "work_id": work["work_id"],
                "stage": stage,
                "lease": lease_document,
                "released_at": _utc_now(),
                "error": (
                    f"{type(lease_release_error).__name__}:"
                    f"{str(lease_release_error)[:500]}"
                    if lease_release_error is not None
                    else None
                ),
            },
        )
    runtime_receipt_path = _write_gpu_stage_execution_receipt(
        attempt_root=attempt_root,
        work=work,
        stage=stage,
        candidate_id=candidate_id,
        lease=lease_document,
        admission_path=admission_path,
        resource_receipt=resource_receipt,
        summary=summary,
        error=campaign_error,
        lease_released=lease_released,
        lease_release_error=lease_release_error,
    )
    if campaign_error is not None:
        raise campaign_error
    if lease_release_error is not None:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_GPU_LEASE_RELEASE_FAILED"
        ) from lease_release_error
    assert summary is not None
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CAMPAIGN_SUMMARY_INVALID")
    row = candidates[0]
    accepted = bool(row.get("accepted"))
    payload = {
        f"{stage}_root": str(attempt_root),
        f"{stage}_batch_path": str(batch_path),
        f"{stage}_accepted": accepted,
        f"{stage}_settlement_state": row.get("state"),
        f"{stage}_gpu_admission_path": str(admission_path),
        f"{stage}_gpu_execution_receipt_path": str(runtime_receipt_path),
        f"{stage}_gpu_execution_receipt_sha256": hashlib.sha256(
            runtime_receipt_path.read_bytes()
        ).hexdigest(),
        **training_payload,
        "training_resource_plan": resource_plan,
    }
    return StageResult(
        state="completed",
        outcome=f"{stage}_{'accepted' if accepted else 'rejected'}",
        payload=payload,
        receipt_path=attempt_root / "campaign-summary.json",
    )


def _write_gpu_stage_execution_receipt(
    *,
    attempt_root: Path,
    work: Mapping[str, object],
    stage: str,
    candidate_id: str,
    lease: Mapping[str, object],
    admission_path: Path,
    resource_receipt: Mapping[str, object] | None,
    summary: Mapping[str, object] | None,
    error: Exception | None,
    lease_released: bool,
    lease_release_error: Exception | None,
) -> Path:
    candidate_root = attempt_root / "candidates" / candidate_id
    worker_path = candidate_root / "worker-receipt.json"
    settlement_path = candidate_root / "settlement.json"
    worker = (
        _load(worker_path, "AUTONOMOUS_GPU_WORKER_RECEIPT_INVALID")
        if worker_path.is_file()
        else {}
    )
    artifacts = []
    for path in (
        worker_path,
        settlement_path,
        attempt_root / "campaign-summary.json",
    ):
        if path.is_file():
            artifacts.append(
                {
                    "artifact": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-gpu-stage-execution",
        "state": "failed" if error is not None else "settled",
        "work_id": work["work_id"],
        "candidate_id": candidate_id,
        "stage": stage,
        "resource_portfolio_id": (
            resource_receipt.get("receipt_id") if resource_receipt else None
        ),
        "resource_portfolio_digest": (
            resource_receipt.get("receipt_digest") if resource_receipt else None
        ),
        "resource_trial_id": (
            resource_receipt["allocation"]["selected_trial_id"]
            if resource_receipt
            else None
        ),
        "physical_gpus": lease,
        "physical_gpu": (
            lease["members"][0]
            if int(lease.get("world_size", 1)) == 1
            and isinstance(lease.get("members"), list)
            and lease.get("members")
            else None
        ),
        "activity_observation": worker.get("gpu_observation"),
        "exit_status": {
            "worker_state": worker.get("state"),
            "exit_code": worker.get("exit_code"),
            "timed_out": worker.get("timed_out"),
            "error": (
                f"{type(error).__name__}:{str(error)[:500]}"
                if error is not None
                else None
            ),
        },
        "artifacts": artifacts,
        "summary_state": summary.get("state") if summary else None,
        "cleanup": {
            "gpu_lease_released": lease_released,
            "release_receipt": admission_path.with_name(
                admission_path.stem + "-release.json"
            ).name,
            "release_error": (
                f"{type(lease_release_error).__name__}:"
                f"{str(lease_release_error)[:500]}"
                if lease_release_error is not None
                else None
            ),
            "scratch_state": "retained_pending_content_addressed_archive",
            "cleanup_policy": "after_content_addressed_receipt",
        },
        "recorded_at": _utc_now(),
        "claim_boundary": (
            "This receipt records physical GPU identity, observed activity, exit status, "
            "artifacts, and lease cleanup. Scientific authority remains with frozen verification."
        ),
    }
    path = attempt_root / "gpu-stage-execution.json"
    _write_json_idempotent(path, receipt)
    return path


def _training_resource_plan_for_batch(
    batch_path: Path, *, allowed_gpu_indices: Sequence[int]
) -> dict[str, object]:
    """Record the planner's method/scale decision alongside every GPU stage."""
    batch = _load(batch_path, "AUTONOMOUS_BATCH_INVALID")
    candidates = batch.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    if not isinstance(candidate, Mapping):
        return {"state": "blocked", "reason": "candidate_metadata_missing"}
    method_class = method_class_from_candidate(candidate)
    parameters = candidate.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    hidden = int(parameters.get("hidden_dim", 0) or 0)
    action = int(parameters.get("action_dim", 0) or 0)
    trainable = int(candidate.get("estimated_trainable_parameters", 0) or 0)
    if trainable < 1 and method_class == "adapter_training":
        # Conservative estimate for the generated mask and residual projections.
        trainable = max(1, hidden * max(action, 1) * 8)
    try:
        return plan_training_resources(
            method_class=method_class,
            trainable_parameters=trainable,
            train_examples=int(candidate.get("training_examples", 1) or 1),
            sequence_length=int(candidate.get("sequence_length", 32) or 32),
            batch_size=int(candidate.get("batch_size", 2) or 2),
            planned_steps=int(candidate.get("training_steps", 100) or 100),
            available_gpus=allowed_gpu_indices,
            competing_candidates=1,
        )
    except (TypeError, ValueError, TrainingResourcePlanningError) as exc:
        return {
            "state": "metadata_required",
            "method_class": method_class,
            "reason": f"{type(exc).__name__}:{str(exc)}",
            "requested_gpu_count": 1,
        }


def _ensure_masked_adapter_training(
    config: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    batch_path: Path,
    campaign_root: Path,
    stage: str,
    attempt_root: Path,
    contract: Mapping[str, object],
    gpu_indices: Sequence[int] | None = None,
    gpu_index: int | None = None,
) -> dict[str, object]:
    """Fit and bind a masked adapter before any screen/confirm evaluator runs."""

    if gpu_indices is None:
        if gpu_index is None:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_TRAINING_GPU_BINDING_INVALID")
        gpu_indices = [gpu_index]
    gpu_indices = [int(index) for index in gpu_indices]
    if not gpu_indices or len(set(gpu_indices)) != len(gpu_indices):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_TRAINING_GPU_BINDING_INVALID")
    batch = _load(batch_path, "AUTONOMOUS_BATCH_INVALID")
    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_CANDIDATE_INVALID")
    candidate = candidates[0]
    if candidate.get("candidate_kind") != "materialized_masked_intermediate_action_adapter":
        return {}
    paths = _paths(config)
    # A bound batch is already durable. Re-validate it and never retrain on resume.
    provenance = candidate.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("training_binding") is not None:
        validate_materialized_candidate_batch(batch, contract, root=paths["project_root"])
        binding = provenance["training_binding"]
        assert isinstance(binding, Mapping)
        return {
            "training_receipt_path": binding.get("training_receipt_path"),
            "adapter_state_path": binding.get("adapter_state_path"),
            "training_reused": True,
        }
    trainer_value = registration.get("trainer")
    if not isinstance(trainer_value, str) or not trainer_value:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINER_MISSING")
    trainer = _require_file(Path(trainer_value), "AUTONOMOUS_MASKED_ADAPTER_TRAINER_INVALID")
    training_config = registration.get("training", {})
    if not isinstance(training_config, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_CONFIG_INVALID")
    candidate_id = str(candidate["candidate_id"])
    training_root = (
        campaign_root.parent
        / "adapter-training"
        / candidate_id
        / "training-v1"
    ).resolve()
    receipt_path = training_root / "training-receipt.json"
    if not receipt_path.is_file():
        command = [
            str(paths["runtime_python"]),
            str(trainer),
            "--candidate", str(batch_path.resolve()),
            "--ctrl-world-root", str(paths["ctrl_world_root"]),
            "--dataset-root", str(paths["dataset_root"]),
            "--data-stat", str(paths["data_stat"]),
            "--checkpoint", str(paths["checkpoint"]),
            "--svd-model", str(paths["svd_model"]),
            "--clip-model", str(paths["clip_model"]),
            "--output-root", str(training_root),
        ]
        option_map = (
            ("steps", "--steps"),
            ("batch_size", "--batch-size"),
            ("learning_rate", "--learning-rate"),
            ("max_grad_norm", "--max-grad-norm"),
            ("seed", "--seed"),
            ("num_history", "--num-history"),
            ("num_frames", "--num-frames"),
            ("split_fingerprint", "--training-split-fingerprint"),
            ("device", "--device"),
        )
        for key, flag in option_map:
            if key in training_config:
                command.extend([flag, str(training_config[key])])
        training_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path = training_root / "training.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        environment["PYTHONUNBUFFERED"] = "1"
        if len(gpu_indices) > 1:
            # torchrun owns rank/world-size setup; the trainer writes one receipt
            # from rank zero after all ranks finish.
            command = [
                str(paths["runtime_python"]), "-m", "torch.distributed.run",
                "--standalone", "--nproc_per_node", str(len(gpu_indices)),
                *command[1:],
            ]
        try:
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=str(paths["project_root"]),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(config["worker_timeout_seconds"]),
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_TIMEOUT") from exc
        if completed.returncode != 0:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MASKED_ADAPTER_TRAINING_FAILED:{completed.returncode}"
            )
    if not receipt_path.is_file():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MASKED_ADAPTER_TRAINING_RECEIPT_MISSING")
    updated = bind_training_to_batch(
        batch,
        receipt_path=receipt_path,
        contract=contract,
        root=paths["project_root"],
    )
    _replace_compiled_batch(batch_path, expected_digest=str(batch["batch_digest"]), updated=updated)
    binding = updated["candidates"][0]["provenance"]["training_binding"]
    assert isinstance(binding, Mapping)
    return {
        "training_receipt_path": binding["training_receipt_path"],
        "adapter_state_path": binding["adapter_state_path"],
        "training_reused": False,
    }


def _replace_compiled_batch(
    path: Path, *, expected_digest: str, updated: Mapping[str, object]
) -> None:
    """Allow exactly one controlled transition from pending to trained batch."""

    current = _load(path, "AUTONOMOUS_BATCH_INVALID")
    if current.get("batch_digest") != expected_digest:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_BINDING_RACE")
    raw = _canonical_bytes(updated)
    temporary = path.with_name(f".{path.name}.training-{os.getpid()}.tmp")
    if path.is_symlink() or not path.is_file():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_BATCH_INVALID")
    with temporary.open("xb") as handle:
        handle.write(raw)
    os.replace(temporary, path)


def verify(
    config: Mapping[str, object],
    *,
    work: Mapping[str, object],
    attempt_root: Path,
) -> StageResult:
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    screen_root = _require_directory(
        Path(str(context.get("screen_root") or "")), "AUTONOMOUS_SCREEN_ROOT_INVALID"
    )
    confirm_value = context.get("confirm_root")
    confirm_root = (
        _require_directory(Path(str(confirm_value)), "AUTONOMOUS_CONFIRM_ROOT_INVALID")
        if confirm_value
        else None
    )
    paths = _paths(config)
    manifest = run_materialized_frozen_verifier(
        policy_path=paths["verifier_policy"],
        contract_path=paths["contract"],
        screen_root=screen_root,
        confirm_root=confirm_root,
        output_root=attempt_root,
        project_root=paths["project_root"],
    )
    evidence_path = attempt_root / "verified-evidence.jsonl"
    projection = project_materialized_transfer_evidence(
        verifier_root=attempt_root,
        output_path=evidence_path,
        project_root=paths["project_root"],
    )
    return StageResult(
        state="completed",
        outcome=str(manifest["decision"]),
        payload={
            "verifier_root": str(attempt_root),
            "decision": manifest["decision"],
            "verdict_ref": manifest["verdict_ref"],
            "verifier_ref": projection["verifier_ref"],
            "verified_evidence_path": str(evidence_path),
            "evidence_projection": projection,
        },
        receipt_path=attempt_root / "verification-manifest.json",
    )


def stage_verified_knowledge(
    config: Mapping[str, object],
    *,
    state_root: Path,
    work: Mapping[str, object],
    verification: StageResult,
) -> dict[str, object]:
    """Stage only the portable projection of a successful frozen verification."""

    evidence_value = verification.payload.get("verified_evidence_path")
    if evidence_value is None:
        return {}
    evidence_path = _require_file(
        Path(str(evidence_value)), "AUTONOMOUS_VERIFIED_EVIDENCE_INVALID"
    )
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_VERIFIED_EVIDENCE_INVALID")
    assessment_path = _require_file(
        Path(str(work.get("assessment_path") or "")), "AUTONOMOUS_ASSESSMENT_INVALID"
    )
    assessment = _load(assessment_path, "AUTONOMOUS_ASSESSMENT_INVALID")
    portable_root = Path(
        str(config.get("portable_knowledge_records_root") or state_root / "portable-knowledge-records")
    ).expanduser().resolve()
    report = stage_verified_transfer_knowledge(
        source_assessment=assessment,
        transfer_evidence=rows[0],
        output_root=portable_root,
        project_root=_paths(config)["project_root"],
    )
    return {"portable_knowledge_staging": report}


def stage_portrait_knowledge(
    config: Mapping[str, object], *, state_root: Path
) -> dict[str, object]:
    """Copy the bound path-free portrait into the explicit community staging root."""

    gate = config.get("portrait_gate")
    if not isinstance(gate, Mapping):
        return {}
    portrait = _load_active_portrait(
        config,
        project_root=_paths(config)["project_root"],
        state_root=state_root,
    )
    portable_root = Path(
        str(
            config.get("portable_knowledge_records_root")
            or state_root / "portable-knowledge-records"
        )
    ).expanduser().resolve()
    report = stage_portable_knowledge_records(
        documents=[portrait], output_root=portable_root
    )
    return {"portable_portrait_staging": report}


def stage_gap_knowledge(
    config: Mapping[str, object], *, state_root: Path, planning: StageResult
) -> dict[str, object]:
    """Stage only path-free module-composition receipts emitted by gap planning."""

    values = planning.payload.get("module_composition_paths")
    if not isinstance(values, list) or not values:
        return {}
    documents = [
        _load(
            _require_file(
                Path(str(value)), "AUTONOMOUS_MODULE_COMPOSITION_INVALID"
            ),
            "AUTONOMOUS_MODULE_COMPOSITION_INVALID",
        )
        for value in values
    ]
    portable_root = Path(
        str(
            config.get("portable_knowledge_records_root")
            or state_root / "portable-knowledge-records"
        )
    ).expanduser().resolve()
    report = stage_portable_knowledge_records(
        documents=documents, output_root=portable_root
    )
    return {"portable_module_staging": report}


def run_closed_loop_replan(
    config: Mapping[str, object],
    *,
    state_root: Path,
    work: Mapping[str, object],
    attempt_root: Path,
    snapshot: Mapping[str, object],
) -> StageResult:
    """Archive one terminal item, audit the loop, and select one bounded task."""

    policy = config.get("closed_loop")
    if not isinstance(policy, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_REQUIRED")
    project_root = _paths(config)["project_root"]
    current_portrait = _load_active_portrait(
        config, project_root=project_root, state_root=state_root
    )
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    terminal_outcome = str(
        work.get("terminal_outcome")
        or context.get("decision")
        or context.get("pre_verifier_outcome")
        or context.get("replan_trigger")
        or "terminal"
    )
    try:
        archive, archive_path = archive_work(
            work=work,
            state_root=state_root,
            archive_root=Path(str(policy["archive_root"])),
            terminal_outcome=terminal_outcome,
            root=project_root,
        )
        receipts = load_archive_receipts(
            Path(str(policy["archive_root"])), root=project_root
        )
    except ClosedLoopReplanningError as exc:
        raise AutonomousTransferWorkflowError(str(exc)) from exc

    portrait = current_portrait
    transition_payload: dict[str, object] | None = None
    recovered_transition = _load_candidate_portrait_transition(
        state_root=state_root,
        context=context,
        current_portrait=current_portrait,
        project_root=project_root,
    )
    if recovered_transition is not None:
        portrait, transition_payload = recovered_transition
    elif _has_admitted_intervention(work, context):
        portrait, transition_payload = _derive_active_portrait(
            config=config,
            state_root=state_root,
            work=work,
            context=context,
            parent=current_portrait,
            archive=archive,
            archive_root=Path(str(policy["archive_root"])),
            project_root=project_root,
        )
    if transition_payload is not None:
        portable_root = Path(
            str(
                config.get("portable_knowledge_records_root")
                or state_root / "portable-knowledge-records"
            )
        ).expanduser().resolve()
        stage_portable_knowledge_records(
            documents=[portrait, transition_payload], output_root=portable_root
        )
        _replace_json_atomic(state_root / "active-portrait.json", portrait)

    audit = build_quality_audit(
        snapshot=snapshot,
        portrait=portrait,
        protocol_findings=_finding_list(context.get("protocol_findings")),
        cleanup_findings=_finding_list(context.get("cleanup_findings")),
        non_portability_findings=_finding_list(
            context.get("non_portability_findings")
        ),
        archive_receipts=receipts,
        root=project_root,
    )
    audit_path = attempt_root / "quality-audit.json"
    _write_json_idempotent(audit_path, audit)
    metric_shadow: dict[str, object] | None = None
    metric_settings = config.get("metric_evolution")
    if isinstance(metric_settings, Mapping):
        try:
            metric_shadow = compile_shadow_metric_proposals(
                candidates=tuple(
                    value for value in metric_settings.get("candidate_metrics", [])
                    if isinstance(value, Mapping)
                ),
                protected_metric_ids=tuple(
                    str(value) for value in metric_settings.get("protected_metric_ids", [])
                ),
                output_root=attempt_root / "shadow-metrics",
                root=project_root,
            )
        except ShadowMetricEvolutionError as exc:
            raise AutonomousTransferWorkflowError(str(exc)) from exc
    trigger = _replan_trigger(work, context)
    signals = _replan_signals(
        context,
        trigger=trigger,
        portrait=portrait,
        project_root=project_root,
    )
    portrait_ready = (
        str(
            context.get("portrait_gate_state")
            or current_portrait.get("state")
            or current_portrait.get("readiness_state")
            or ""
        )
        == "ready_for_gap_planning"
    )
    try:
        decision = build_next_task_decision(
            work_id=str(work["work_id"]),
            trigger=trigger,
            signals=signals,
            quality_audit=audit,
            minimum_information_gain=float(policy["minimum_information_gain"]),
            maximum_replans=int(policy["maximum_replans"]),
            stop_on_confirmed_positive=bool(policy["stop_on_confirmed_positive"]),
            available_tasks=tuple(
                task
                for task, enabled in (
                    (
                        "observe_portrait",
                        isinstance(config.get("observation_planning"), Mapping)
                        and not portrait_ready,
                    ),
                    (
                        "discover_intervention",
                        isinstance(config.get("gap_planning"), Mapping),
                    ),
                )
                if enabled
            )
            + ("stop",),
            root=project_root,
        )
    except ClosedLoopReplanningError as exc:
        raise AutonomousTransferWorkflowError(str(exc)) from exc
    decision_path = attempt_root / "next-task.json"
    _write_json_idempotent(decision_path, decision)
    selected = str(decision["selected_task"])
    next_state = "terminal"
    if selected == "observe_portrait" and isinstance(
        config.get("observation_planning"), Mapping
    ) and not portrait_ready:
        next_state = "pending_observation"
    elif selected == "discover_intervention" and isinstance(
        config.get("gap_planning"), Mapping
    ):
        next_state = "pending_gap_planning"
    elif selected != "stop":
        raise AutonomousTransferWorkflowError("AUTONOMOUS_REPLAN_TASK_UNAVAILABLE")
    payload: dict[str, object] = {
        "archive_id": archive["archive_id"],
        "archive_receipt_path": str(archive_path),
        "next_task_decision_id": decision["decision_id"],
        "next_task_decision_path": str(decision_path),
        "next_task": decision["selected_task"],
        "next_task_stop_reason": decision["stop_reason"],
        "quality_audit_id": audit["audit_id"],
        "quality_audit_path": str(audit_path),
        "quality_audit_state": audit["state"],
        "replan_signals": signals,
        "replan_count": int(signals["replan_count"]) + 1,
        "replan_trigger": None,
        "replan_next_state": next_state,
    }
    if transition_payload is not None:
        payload["portrait_transition_id"] = transition_payload["transition_id"]
        payload["portrait_transition_candidate_id"] = context["candidate_id"]
        payload["active_portrait_id"] = portrait["portrait_id"]
    if metric_shadow is not None:
        payload["metric_evolution"] = metric_shadow
    outcome = (
        str(decision["stop_reason"])
        if next_state == "terminal"
        else str(decision["selected_task"])
    )
    return StageResult(
        state="completed",
        outcome=outcome,
        payload=payload,
        receipt_path=decision_path,
    )


def _has_admitted_intervention(
    work: Mapping[str, object], context: Mapping[str, object]
) -> bool:
    candidate_id = context.get("candidate_id")
    if not candidate_id or not context.get("materialization_root"):
        return False
    if context.get("portrait_transition_candidate_id") == candidate_id:
        return False
    outcome = str(
        work.get("terminal_outcome")
        or context.get("decision")
        or context.get("pre_verifier_outcome")
        or ""
    )
    return outcome not in {"operational_failure_unverified", "knowledge_projection_failure"}


def _derive_active_portrait(
    *,
    config: Mapping[str, object],
    state_root: Path,
    work: Mapping[str, object],
    context: Mapping[str, object],
    parent: Mapping[str, object],
    archive: Mapping[str, object],
    archive_root: Path,
    project_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    evidence_refs = [
        str(row["cas_ref"])
        for row in archive.get("artifacts", [])
        if isinstance(row, Mapping) and isinstance(row.get("cas_ref"), str)
    ]
    if not evidence_refs:
        evidence_refs = ["sha256:" + hashlib.sha256(str(work["work_id"]).encode()).hexdigest()]
    outcome_state = _portrait_transition_outcome(context)
    if outcome_state != "admitted":
        for field in ("verifier_ref", "verdict_ref"):
            value = context.get(field)
            if isinstance(value, str):
                evidence_refs.append(value)
        transition_ref = str(context.get("verdict_ref") or "")
    else:
        transition_ref = evidence_refs[0]
    try:
        portrait = derive_model_portrait(
            parent_portrait=parent,
            transition_ref=transition_ref,
            evidence_refs=tuple(evidence_refs),
            root=project_root,
        )
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_DERIVED_PORTRAIT_INVALID:{exc}"
        ) from exc
    try:
        transition = build_portrait_transition(
            parent_portrait=parent,
            portrait=portrait,
            embodiment_id=str(context.get("candidate_id") or work["work_id"]),
            outcome_state=outcome_state,
            evaluator_binding=(
                str(context["verifier_ref"])
                if outcome_state != "admitted" and context.get("verifier_ref")
                else None
            ),
            verdict_ref=(
                str(context["verdict_ref"])
                if outcome_state != "admitted" and context.get("verdict_ref")
                else None
            ),
            evidence_refs=tuple(evidence_refs),
            root=project_root,
        )
    except CommunityKnowledgeError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_PORTRAIT_TRANSITION_INVALID:{exc}"
        ) from exc
    transition_bytes = _canonical_bytes(transition)
    transition_digest = hashlib.sha256(transition_bytes).hexdigest()
    cas_path = (
        archive_root.expanduser().resolve()
        / "cas"
        / "sha256"
        / transition_digest[:2]
        / transition_digest
    )
    _write_bytes_idempotent(cas_path, transition_bytes)
    transition_root = state_root / "portrait-transitions"
    portrait_path = transition_root / f"{portrait['portrait_id']}.json"
    transition_path = transition_root / f"{transition['transition_id']}.json"
    _write_json_idempotent(portrait_path, portrait)
    _write_json_idempotent(transition_path, transition)
    candidate_id = str(context.get("candidate_id") or work["work_id"])
    marker_path = (
        transition_root
        / "by-candidate"
        / (hashlib.sha256(candidate_id.encode("utf-8")).hexdigest() + ".json")
    )
    _write_json_idempotent(
        marker_path,
        {
            "candidate_id": candidate_id,
            "parent_portrait_id": parent["portrait_id"],
            "portrait_id": portrait["portrait_id"],
            "transition_id": transition["transition_id"],
        },
    )
    return portrait, transition


def _load_candidate_portrait_transition(
    *,
    state_root: Path,
    context: Mapping[str, object],
    current_portrait: Mapping[str, object],
    project_root: Path,
) -> tuple[dict[str, object], dict[str, object]] | None:
    candidate_id = context.get("candidate_id")
    if not candidate_id or not context.get("materialization_root"):
        return None
    if context.get("portrait_transition_candidate_id") == candidate_id:
        return None
    marker_path = (
        state_root
        / "portrait-transitions"
        / "by-candidate"
        / (hashlib.sha256(str(candidate_id).encode("utf-8")).hexdigest() + ".json")
    )
    if not marker_path.exists():
        return None
    marker = _load(
        _require_file(marker_path, "AUTONOMOUS_PORTRAIT_TRANSITION_MARKER_INVALID"),
        "AUTONOMOUS_PORTRAIT_TRANSITION_MARKER_INVALID",
    )
    if marker != {
        "candidate_id": str(candidate_id),
        "parent_portrait_id": marker.get("parent_portrait_id"),
        "portrait_id": marker.get("portrait_id"),
        "transition_id": marker.get("transition_id"),
    }:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_PORTRAIT_TRANSITION_MARKER_INVALID"
        )
    transition_root = state_root / "portrait-transitions"
    portrait = _load(
        _require_file(
            transition_root / f"{marker['portrait_id']}.json",
            "AUTONOMOUS_DERIVED_PORTRAIT_INVALID",
        ),
        "AUTONOMOUS_DERIVED_PORTRAIT_INVALID",
    )
    transition = _load(
        _require_file(
            transition_root / f"{marker['transition_id']}.json",
            "AUTONOMOUS_PORTRAIT_TRANSITION_INVALID",
        ),
        "AUTONOMOUS_PORTRAIT_TRANSITION_INVALID",
    )
    try:
        validate_model_portrait(portrait, root=project_root)
        validate_portrait_transition(transition, root=project_root)
    except (ModelPortraitError, CommunityKnowledgeError) as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_PORTRAIT_TRANSITION_MARKER_INVALID:{exc}"
        ) from exc
    if (
        transition.get("embodiment_id") != str(candidate_id)
        or transition.get("portrait_id") != portrait.get("portrait_id")
        or transition.get("parent_portrait_id") != marker.get("parent_portrait_id")
        or current_portrait.get("portrait_id")
        not in {marker.get("parent_portrait_id"), marker.get("portrait_id")}
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_PORTRAIT_TRANSITION_MARKER_CONFLICT"
        )
    return portrait, transition


def _portrait_transition_outcome(context: Mapping[str, object]) -> str:
    outcome = str(
        context.get("decision") or context.get("pre_verifier_outcome") or ""
    )
    if outcome == "confirmed_positive":
        return "target_confirmed"
    if outcome in {"rejected_at_screen", "rejected_at_confirm"}:
        return "verified_negative_boundary"
    return "admitted"


def _finding_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_FINDINGS_INVALID")
    return list(value)


def _replan_signals(
    context: Mapping[str, object],
    *,
    trigger: str,
    portrait: Mapping[str, object],
    project_root: Path,
) -> dict[str, object]:
    evidence = _replan_evidence(context, project_root=project_root)
    outcome = str(evidence.get("outcome") or trigger)
    deltas = evidence.get("metric_deltas")
    delta_values = [
        abs(float(value))
        for value in deltas.values()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    ] if isinstance(deltas, Mapping) else []
    blockers = evidence.get("blockers")
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    negative = outcome in {"rejected_at_screen", "rejected_at_confirm"}
    operational = outcome in {"operational_failure", "operational_failure_unverified"}
    defaults = {
        "residual": min(1.0, max(delta_values, default=0.0)),
        "counterexample": 1.0 if negative else 0.0,
        "uncertainty": (
            1.0
            if operational
            else min(1.0, blocker_count / 3.0)
            if blocker_count
            else 0.5
            if negative
            else 0.0
        ),
        "information_gain": (
            0.0
            if operational
            else max(
                min(1.0, max(delta_values, default=0.0)),
                1.0 if negative else 0.0,
            )
        ),
    }
    signals: dict[str, object] = {
        name: context.get(name) if context.get(name) is not None else default
        for name, default in defaults.items()
    }
    coverage = portrait.get("coverage")
    portrait_stale = isinstance(coverage, Mapping) and bool(
        coverage.get("stale_fingerprint_ids")
        or coverage.get("conflicts")
        or coverage.get("unknown_operational_metrics")
    )
    signals["stale_portrait"] = bool(context.get("stale_portrait")) or portrait_stale
    signals["replan_count"] = int(context.get("replan_count") or 0)
    return signals


def _replan_evidence(
    context: Mapping[str, object], *, project_root: Path
) -> dict[str, object]:
    value = context.get("verified_evidence_path")
    if value is None:
        return {}
    path = Path(str(value)).expanduser()
    if path.is_symlink():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_REPLAN_EVIDENCE_INVALID")
    resolved = path.resolve()
    if not resolved.is_file():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_REPLAN_EVIDENCE_INVALID")
    try:
        rows = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_REPLAN_EVIDENCE_INVALID"
        ) from exc
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_REPLAN_EVIDENCE_INVALID")
    try:
        validate_document("materialized_transfer_evidence", rows[0], root=project_root)
    except ContractValidationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_REPLAN_EVIDENCE_INVALID:{exc}"
        ) from exc
    return rows[0]


def _replan_trigger(work: Mapping[str, object], context: Mapping[str, object]) -> str:
    return str(
        context.get("replan_trigger")
        or context.get("decision")
        or context.get("pre_verifier_outcome")
        or work.get("terminal_outcome")
        or "operational_failure"
    )


def rebuild_knowledge_graph(
    config: Mapping[str, object], *, state_root: Path
) -> StageResult:
    graph_root = Path(str(config["knowledge_graph_root"])).expanduser().resolve()
    if graph_root == state_root or state_root in graph_root.parents:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GRAPH_OUTPUT_OVERLAPS_STATE")
    report = write_evidence_graph(input_root=state_root, output_root=graph_root)
    portable_records_root = Path(
        str(config.get("portable_knowledge_records_root") or state_root / "portable-knowledge-records")
    ).expanduser().resolve()
    portable_graph_root = Path(
        str(
            config.get("portable_knowledge_root")
            or graph_root.with_name(graph_root.name + "-portable")
        )
    ).expanduser().resolve()
    if (
        portable_graph_root == state_root
        or state_root in portable_graph_root.parents
        or portable_graph_root == graph_root
        or graph_root in portable_graph_root.parents
        or portable_graph_root in graph_root.parents
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_GRAPH_OUTPUT_INVALID")
    portable_documents = _load_portable_knowledge_documents(portable_records_root)
    portable_report = write_portable_knowledge_graph(
        documents=portable_documents,
        output_root=portable_graph_root,
    )
    return StageResult(
        state="completed",
        outcome="knowledge_graph_updated",
        payload={
            "knowledge_graph_root": str(graph_root),
            "knowledge_graph_node_count": report["node_count"],
            "knowledge_graph_edge_count": report["edge_count"],
            "knowledge_graph_source_count": report["source_count"],
            "portable_knowledge_graph_root": str(portable_graph_root),
            "portable_knowledge_document_count": portable_report["document_count"],
            "portable_knowledge_node_count": portable_report["node_count"],
            "portable_knowledge_edge_count": portable_report["edge_count"],
            "portable_knowledge_quality_audit_id": portable_report[
                "quality_audit_id"
            ],
            "portable_knowledge_quality_audit_state": portable_report[
                "quality_audit_state"
            ],
        },
        receipt_path=graph_root / "manifest.json",
    )


def import_evidence(config: Mapping[str, object], *, state_root: Path) -> list[dict[str, object]]:
    imported = []
    destination = state_root / "imports"
    for raw_root in config.get("existing_evidence_roots", []):
        root = _require_directory(Path(str(raw_root)), "AUTONOMOUS_EVIDENCE_ROOT_INVALID")
        source = _require_file(
            root / "verified-evidence.jsonl", "AUTONOMOUS_EXISTING_EVIDENCE_INVALID"
        )
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise AutonomousTransferWorkflowError("AUTONOMOUS_EXISTING_EVIDENCE_INVALID")
            name = f"{record['source_digest']}-{record['assessment_digest']}-{index}.jsonl"
            local = destination / name
            _write_bytes_idempotent(local, _canonical_bytes(record))
            imported.append({"source_path": source, "local_path": local, "record": record})
    return imported


def _bind_plan(
    template: Mapping[str, object],
    *,
    work: Mapping[str, object],
    idea: Mapping[str, object],
) -> dict[str, object]:
    plan = copy.deepcopy(dict(template))
    base_candidate = str(plan["candidate_id"])
    suffix = str(work["source_digest"])[:12]
    candidate_id = f"{base_candidate[:110]}-{suffix}"
    plan["idea_id"] = idea["idea_id"]
    plan["candidate_id"] = candidate_id
    plan["plan_id"] = f"{str(plan['plan_id'])[:110]}-{suffix}"
    candidate_template = plan.get("candidate_template")
    if not isinstance(candidate_template, dict):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_CANDIDATE_TEMPLATE_INVALID")
    candidate_template["candidate_id"] = candidate_id
    plan["plan_digest"] = ""
    plan["plan_digest"] = materialization_plan_digest(plan)
    return plan


def _validate_paths(config: Mapping[str, object], *, project_root: Path) -> None:
    paths = _paths(config)
    if paths["project_root"] != project_root.resolve():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PROJECT_ROOT_MISMATCH")
    for name in (
        "ctrl_world_root",
        "dataset_root",
        "svd_model",
        "clip_model",
    ):
        _require_directory(paths[name], f"AUTONOMOUS_PATH_INVALID:{name}")
    for name in (
        "data_stat",
        "checkpoint",
        "runtime_python",
        "intake_config",
        "contract",
        "evaluator",
        "base_evaluator",
        "screen_baseline",
        "confirm_baseline",
        "verifier_policy",
    ):
        _require_file(paths[name], f"AUTONOMOUS_PATH_INVALID:{name}")
    if config.get("source_bundle_path") is not None:
        _require_file(
            Path(str(config["source_bundle_path"])),
            "AUTONOMOUS_SOURCE_BUNDLE_INVALID",
        )
    materializers = config["materializers"]
    assert isinstance(materializers, Mapping)
    for profile, row in materializers.items():
        if not isinstance(row, Mapping):
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MATERIALIZER_INVALID:{profile}"
            )
        _require_file(
            Path(str(row["plan_template"])),
            f"AUTONOMOUS_MATERIALIZER_TEMPLATE_INVALID:{profile}",
        )
        if row.get("evaluator") is not None:
            _require_file(
                Path(str(row["evaluator"])),
                f"AUTONOMOUS_MATERIALIZER_EVALUATOR_INVALID:{profile}",
            )
        if row.get("trainer") is not None:
            _require_file(
                Path(str(row["trainer"])),
                f"AUTONOMOUS_MATERIALIZER_TRAINER_INVALID:{profile}",
            )
        training = row.get("training")
        if training is not None and not isinstance(training, Mapping):
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_MATERIALIZER_TRAINING_INVALID:{profile}"
            )
    automatic = config.get("automatic_module_generation")
    if automatic is not None:
        if not isinstance(automatic, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_CONFIG_INVALID"
            )
        _require_file(
            Path(str(automatic["model_capability_ir"])),
            "AUTONOMOUS_AUTOMATIC_MODULE_CAPABILITY_IR_INVALID",
        )
        _require_file(
            Path(str(automatic["abi_registry"])),
            "AUTONOMOUS_AUTOMATIC_MODULE_ABI_REGISTRY_INVALID",
        )
        if not isinstance(automatic.get("llm_adapter"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_ADAPTER_INVALID"
            )
        if not isinstance(config.get("portrait_gate"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_PORTRAIT_GATE_REQUIRED"
            )
        if not isinstance(config.get("gap_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_AUTOMATIC_MODULE_GAP_PLANNING_REQUIRED"
            )
    open_generation = config.get("open_method_generation")
    if open_generation is not None:
        if not isinstance(open_generation, Mapping) or not isinstance(
            open_generation.get("llm_adapter"), Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OPEN_METHOD_CONFIG_INVALID"
            )
        if not isinstance(config.get("portrait_gate"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OPEN_METHOD_PORTRAIT_GATE_REQUIRED"
            )
    gate = config.get("portrait_gate")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_GATE_INVALID")
        _load_bound_portrait(gate, project_root=project_root)
        if not isinstance(config.get("gap_planning"), Mapping) and not isinstance(
            config.get("open_method_generation"), Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTRAIT_GAP_PLANNING_REQUIRED"
            )
    gap_planning = config.get("gap_planning")
    if gap_planning is not None:
        if not isinstance(gap_planning, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_PLANNING_INVALID")
        registry_path = _require_file(
            Path(str(gap_planning.get("abi_registry") or "")),
            "AUTONOMOUS_GAP_ABI_REGISTRY_INVALID",
        )
        try:
            registry = load_module_abi_registry(registry_path, root=project_root)
        except ModuleCompositionError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_GAP_ABI_REGISTRY_INVALID:{exc}"
            ) from exc
        if registry["registry_digest"] != gap_planning.get("abi_registry_digest"):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_ABI_REGISTRY_DIGEST_MISMATCH"
            )
        profiles = gap_planning.get("profile_requirements")
        assert isinstance(profiles, list)
        profile_ids = [str(row["profile_id"]) for row in profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_PROFILE_DUPLICATE"
            )
        _bound_portable_knowledge_graph(gap_planning)
        if not isinstance(config.get("portfolio_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_GAP_PORTFOLIO_PLANNING_REQUIRED"
            )
    portfolio_planning = config.get("portfolio_planning")
    if portfolio_planning is not None:
        if not isinstance(portfolio_planning, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTFOLIO_PLANNING_INVALID"
            )
        if not isinstance(config.get("gap_planning"), Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_PORTFOLIO_GAP_PLANNING_REQUIRED"
            )
        _load_bound_hypothesis_batch(
            portfolio_planning,
            project_root=project_root,
        )
    observation = config.get("observation_planning")
    if observation is not None:
        if not isinstance(observation, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PLANNING_INVALID"
            )
        registry_path = _require_file(
            Path(str(observation.get("abi_registry") or "")),
            "AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID",
        )
        try:
            observation_registry = load_observation_abi_registry(
                registry_path, root=project_root
            )
        except AdaptiveObservationError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_OBSERVATION_ABI_REGISTRY_INVALID:{exc}"
            ) from exc
        _validate_observation_execution_config(
            config,
            registry=observation_registry,
            project_root=project_root,
        )
    closed_loop = config.get("closed_loop")
    if closed_loop is not None:
        if not isinstance(closed_loop, Mapping):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_INVALID")
        archive_value = closed_loop.get("archive_root")
        if not isinstance(archive_value, str) or not archive_value.strip():
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_ARCHIVE_INVALID")
        archive_root = Path(archive_value).expanduser().resolve()
        if archive_root == project_root or project_root in archive_root.parents:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_CLOSED_LOOP_ARCHIVE_INSIDE_SOURCE")
        active_path = closed_loop.get("active_portrait_path")
        active_hash = closed_loop.get("active_portrait_sha256")
        if (active_path is None) != (active_hash is None):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_BINDING_INVALID")
        if active_path is not None:
            if not isinstance(active_path, str) or not active_path.strip():
                raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_PATH_INVALID")
            active = Path(active_path).expanduser().resolve()
            if active == project_root or project_root in active.parents:
                raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_INSIDE_SOURCE")


def _validate_observation_execution_config(
    config: Mapping[str, object],
    *,
    registry: Mapping[str, object],
    project_root: Path,
) -> None:
    execution = config.get("observation_execution")
    if not isinstance(execution, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_REQUIRED"
        )
    archive_root = Path(str(execution.get("archive_root") or "")).expanduser()
    if archive_root.is_symlink():
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INVALID"
        )
    archive_root = archive_root.resolve()
    if archive_root == project_root or project_root in archive_root.parents:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INSIDE_SOURCE"
        )
    adapters = execution.get("adapters")
    runtime_bindings = execution.get("runtime_bindings")
    shadow_adapter = execution.get("shadow_llm_adapter")
    if (
        not isinstance(adapters, Mapping)
        or not isinstance(runtime_bindings, Mapping)
        or not isinstance(shadow_adapter, Mapping)
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_EXECUTION_CONFIG_INVALID"
        )
    rows = [row for row in registry["abis"] if isinstance(row, Mapping)]
    by_id = {str(row["abi_id"]): row for row in rows}
    admitted = {
        abi_id
        for abi_id, row in by_id.items()
        if row.get("admission_state") == "admitted"
    }
    if set(str(value) for value in adapters) != admitted:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ADAPTER_COVERAGE_INVALID"
        )
    if any(str(value) not in admitted for value in runtime_bindings):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_RUNTIME_BINDING_UNKNOWN"
        )
    for abi_id in sorted(admitted):
        adapter = adapters[abi_id]
        assert isinstance(adapter, Mapping)
        abi = by_id[abi_id]
        if (
            abi.get("execution_mode") == "cpu_only"
            and adapter.get("resource_class") != "cpu_only"
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_READ_ONLY_GPU_FORBIDDEN:" + abi_id
            )
        command = adapter.get("command")
        if not isinstance(command, list) or not any(
            "{request_path}" in str(value) for value in command
        ) or not any("{response_path}" in str(value) for value in command):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_ADAPTER_PLACEHOLDERS_REQUIRED:" + abi_id
            )


def _load_bound_portrait(
    gate: Mapping[str, object], *, project_root: Path
) -> dict[str, object]:
    path = _require_file(
        Path(str(gate.get("model_portrait") or "")),
        "AUTONOMOUS_MODEL_PORTRAIT_INVALID",
    )
    expected = gate.get("model_portrait_sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MODEL_PORTRAIT_HASH_MISMATCH")
    portrait = _load(path, "AUTONOMOUS_MODEL_PORTRAIT_INVALID")
    try:
        validate_model_portrait(portrait, root=project_root)
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_MODEL_PORTRAIT_INVALID:{exc}"
        ) from exc
    return portrait


def _load_active_portrait(
    config: Mapping[str, object], *, project_root: Path, state_root: Path | None
) -> dict[str, object]:
    """Load the immutable onboarding portrait or the durable derived portrait."""

    gate = config.get("portrait_gate")
    if not isinstance(gate, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_GATE_REQUIRED")
    candidate: Path | None = None
    expected: str | None = None
    if state_root is not None:
        active = Path(state_root).expanduser().resolve() / "active-portrait.json"
        if active.is_file() and not active.is_symlink():
            candidate = active
    closed_loop = config.get("closed_loop")
    if candidate is None and isinstance(closed_loop, Mapping):
        configured = closed_loop.get("active_portrait_path")
        if isinstance(configured, str) and configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file() and not path.is_symlink():
                candidate = path
                value = closed_loop.get("active_portrait_sha256")
                expected = value if isinstance(value, str) else None
    if candidate is None:
        return _load_bound_portrait(gate, project_root=project_root)
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if expected is not None and actual != expected:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_HASH_MISMATCH")
    portrait = _load(candidate, "AUTONOMOUS_MODEL_PORTRAIT_INVALID")
    try:
        validate_model_portrait(portrait, root=project_root)
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_MODEL_PORTRAIT_INVALID:{exc}"
        ) from exc
    return portrait


def _resume_observation_completion(
    *,
    config: Mapping[str, object],
    state_root: Path,
    completion_path: Path,
) -> StageResult | None:
    if not completion_path.exists() and not completion_path.is_symlink():
        return None
    completion_file = _require_state_artifact_file(
        completion_path,
        state_root=state_root,
        code="AUTONOMOUS_OBSERVATION_COMPLETION_INVALID",
    )
    completion = _load(completion_file, "AUTONOMOUS_OBSERVATION_COMPLETION_INVALID")
    if (
        completion.get("schema_version") != 1
        or completion.get("artifact_type")
        != "verdiwm-observation-execution-completion"
        or completion.get("state") != "completed"
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_INVALID"
        )
    plan_id = completion.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.startswith("adaptive-probe-plan-"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_INVALID"
        )
    payload = completion.get("payload")
    if not isinstance(payload, dict):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PAYLOAD_INVALID"
        )
    payload_sha256 = completion.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != payload_sha256
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PAYLOAD_HASH_MISMATCH"
        )
    if (
        payload.get("observation_execution_plan_id") != plan_id
        or payload.get("observation_parent_portrait_id")
        != completion.get("parent_portrait_id")
        or payload.get("observation_portrait_id") != completion.get("portrait_id")
        or payload.get("observation_transition_id") != completion.get("transition_id")
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_BINDING_MISMATCH"
        )
    plan_path = _require_file(
        Path(str(payload.get("observation_execution_plan_path") or "")),
        "AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_INVALID",
    )
    expected_plan_sha = payload.get("observation_execution_plan_sha256")
    if (
        not isinstance(expected_plan_sha, str)
        or hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected_plan_sha
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_HASH_MISMATCH"
        )
    plan = _load(plan_path, "AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_INVALID")
    try:
        validate_adaptive_probe_plan(plan, root=_paths(config)["project_root"])
    except AdaptiveObservationError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_INVALID:{exc}"
        ) from exc
    if (
        plan.get("plan_id") != plan_id
        or plan.get("portrait_id") != completion.get("parent_portrait_id")
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PLAN_BINDING_MISMATCH"
        )
    portrait_path = _require_state_artifact_file(
        Path(str(completion.get("portrait_path") or "")),
        state_root=state_root,
        code="AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_INVALID",
    )
    expected_portrait_sha = completion.get("portrait_sha256")
    if (
        not isinstance(expected_portrait_sha, str)
        or hashlib.sha256(portrait_path.read_bytes()).hexdigest()
        != expected_portrait_sha
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_HASH_MISMATCH"
        )
    portrait = _load(
        portrait_path, "AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_INVALID"
    )
    try:
        validate_model_portrait(portrait, root=_paths(config)["project_root"])
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_INVALID:{exc}"
        ) from exc
    if portrait.get("portrait_id") != completion.get("portrait_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_MISMATCH"
        )
    transition_path = _require_state_artifact_file(
        Path(str(completion.get("transition_path") or "")),
        state_root=state_root,
        code="AUTONOMOUS_OBSERVATION_COMPLETION_TRANSITION_INVALID",
    )
    expected_transition_sha = completion.get("transition_sha256")
    if (
        not isinstance(expected_transition_sha, str)
        or hashlib.sha256(transition_path.read_bytes()).hexdigest()
        != expected_transition_sha
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_TRANSITION_HASH_MISMATCH"
        )
    transition = _load(
        transition_path, "AUTONOMOUS_OBSERVATION_COMPLETION_TRANSITION_INVALID"
    )
    try:
        validate_portrait_transition(
            transition, root=_paths(config)["project_root"]
        )
    except CommunityKnowledgeError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_OBSERVATION_COMPLETION_TRANSITION_INVALID:{exc}"
        ) from exc
    if (
        transition.get("transition_id") != completion.get("transition_id")
        or transition.get("parent_portrait_id") != completion.get("parent_portrait_id")
        or transition.get("portrait_id") != completion.get("portrait_id")
        or transition.get("transition_ref") != portrait.get("transition_ref")
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_TRANSITION_BINDING_MISMATCH"
        )
    if transition.get("portrait_digest") != "sha256:" + hashlib.sha256(
        _canonical_bytes(portrait)
    ).hexdigest():
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PORTRAIT_DIGEST_MISMATCH"
        )
    if portrait.get("parent_portrait_id") != transition.get("parent_portrait_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PARENT_BINDING_MISMATCH"
        )
    if not _portrait_digest_available(
        config,
        state_root=state_root,
        portrait_id=str(transition.get("parent_portrait_id")),
        expected_digest=str(transition.get("parent_portrait_digest")),
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_COMPLETION_PARENT_DIGEST_MISMATCH"
        )
    active_path = state_root / "active-portrait.json"
    if active_path.is_file() and not active_path.is_symlink():
        active = _load(active_path, "AUTONOMOUS_ACTIVE_PORTRAIT_INVALID")
        try:
            validate_model_portrait(active, root=_paths(config)["project_root"])
        except ModelPortraitError as exc:
            raise AutonomousTransferWorkflowError(
                f"AUTONOMOUS_ACTIVE_PORTRAIT_INVALID:{exc}"
            ) from exc
        if active.get("portrait_id") not in {
            completion.get("parent_portrait_id"),
            completion.get("portrait_id"),
        }:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_COMPLETION_ACTIVE_PORTRAIT_CONFLICT"
            )
    _replace_json_atomic(active_path, portrait)
    return StageResult(
        state="completed",
        outcome="portrait_observation_admitted",
        payload=payload,
        receipt_path=completion_path,
    )


def _require_state_artifact_file(
    path: Path, *, state_root: Path, code: str
) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(code)
    resolved = raw.resolve()
    state = Path(state_root).expanduser().resolve()
    if resolved == state or state not in resolved.parents or not resolved.is_file():
        raise AutonomousTransferWorkflowError(code)
    return resolved


def _portrait_digest_available(
    config: Mapping[str, object],
    *,
    state_root: Path,
    portrait_id: str,
    expected_digest: str,
) -> bool:
    candidates: list[Path] = [state_root / "active-portrait.json"]
    gate = config.get("portrait_gate")
    if isinstance(gate, Mapping):
        configured = gate.get("model_portrait")
        if isinstance(configured, str):
            candidates.append(Path(configured))
    candidates.append(state_root / "portrait-transitions" / f"{portrait_id}.json")
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            document = _load(candidate, "AUTONOMOUS_PORTRAIT_INVALID")
        except AutonomousTransferWorkflowError:
            continue
        if (
            document.get("portrait_id") == portrait_id
            and expected_digest == "sha256:" + hashlib.sha256(
                _canonical_bytes(document)
            ).hexdigest()
        ):
            return True
    return False


def _load_bound_observation_order(
    context: Mapping[str, object],
) -> dict[str, object]:
    path = _require_file(
        Path(str(context.get("portrait_observation_work_order_path") or "")),
        "AUTONOMOUS_OBSERVATION_WORK_ORDER_INVALID",
    )
    expected = context.get("portrait_observation_work_order_sha256")
    if not isinstance(expected, str) or hashlib.sha256(
        path.read_bytes()
    ).hexdigest() != expected:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_WORK_ORDER_HASH_MISMATCH"
        )
    order = _load(path, "AUTONOMOUS_OBSERVATION_WORK_ORDER_INVALID")
    if order.get("observation_id") != context.get("portrait_observation_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_WORK_ORDER_BINDING_MISMATCH"
        )
    return order


def _probe_requirement_for_task(
    *,
    task: Mapping[str, object],
    observation_order: Mapping[str, object],
) -> Mapping[str, object] | None:
    key = task.get("probe_coverage_key")
    if key is None:
        return None
    requirements = observation_order.get("requirements")
    if not isinstance(requirements, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_REQUIREMENTS_INVALID"
        )
    rows = [
        row
        for row in requirements.get("probe_coverage", [])
        if isinstance(row, Mapping) and row.get("coverage_key") == key
    ]
    if len(rows) != 1:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_PROBE_REQUIREMENT_MISSING"
        )
    return rows[0]


def _observation_execution_request(
    *,
    plan: Mapping[str, object],
    portrait: Mapping[str, object],
    task: Mapping[str, object],
    probe_requirement: Mapping[str, object] | None,
    runtime_bindings: Mapping[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-observation-execution-request",
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "abi_id": task["abi_id"],
        "plan_id": plan["plan_id"],
        "portrait_id": portrait["portrait_id"],
        "execution_authority": task["execution_authority"],
        "task": dict(task),
        "probe_requirement": (
            dict(probe_requirement) if probe_requirement is not None else None
        ),
        "runtime_bindings": dict(runtime_bindings),
        "claim_boundary": (
            "This request grants one admitted observation ABI read-only or shadow-only "
            "execution. It grants no intervention, metric, evaluator, verdict, or "
            "promotion authority."
        ),
    }
    body["request_id"] = (
        "observation-execution-"
        + hashlib.sha256(_canonical_bytes(body)).hexdigest()[:24]
    )
    return body


def _validate_observation_response(
    *,
    task: Mapping[str, object],
    requirement: Mapping[str, object] | None,
    response: Mapping[str, object],
) -> None:
    task_type = str(task["task_type"])
    if task_type == "reuse_existing_probe":
        if response.get("observation_kind") != "probe_fingerprint" or not isinstance(
            requirement, Mapping
        ):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PROBE_RESPONSE_INVALID"
            )
        probe = response.get("probe_observation")
        if not isinstance(probe, Mapping):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PROBE_RESPONSE_INVALID"
            )
        exact_fields = (
            "probe_protocol_id",
            "probe_protocol_version",
            "diagnostic_role",
            "context_class",
            "split",
            "horizons",
            "dose_values",
        )
        if any(probe.get(field) != requirement.get(field) for field in exact_fields):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PROBE_REQUIREMENT_MISMATCH"
            )
        if int(probe["replication_count"]) < int(
            requirement["minimum_replication_count"]
        ) or probe_coverage_key(probe) != requirement.get("coverage_key"):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_PROBE_COVERAGE_INSUFFICIENT"
            )
        return
    if task_type != "run_read_only_adapter" or response.get(
        "observation_kind"
    ) != "structural_surface":
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_STRUCTURAL_RESPONSE_INVALID"
        )
    structural = response.get("structural_observation")
    if not isinstance(structural, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_STRUCTURAL_RESPONSE_INVALID"
        )
    blocker = str(task["blocker"])
    expected_field = None
    identity_field = None
    if blocker.startswith("PORTRAIT_CAPABILITY_UNKNOWN:"):
        expected_field, identity_field = "capabilities", "capability"
    elif blocker.startswith("PORTRAIT_INTERFACE_UNKNOWN:"):
        expected_field, identity_field = "execution_interfaces", "kind"
    elif blocker.startswith("PORTRAIT_HOOK_UNKNOWN:"):
        expected_field, identity_field = "hooks", "hook"
    elif blocker.startswith("PORTRAIT_OPERATIONAL_UNKNOWN:"):
        expected_field, identity_field = "operational_metrics", "metric"
    else:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_STRUCTURAL_BLOCKER_INVALID"
        )
    target = blocker.rsplit(":", 1)[1]
    for field in (
        "capabilities",
        "execution_interfaces",
        "hooks",
        "operational_metrics",
    ):
        rows = structural.get(field)
        if not isinstance(rows, list):
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_STRUCTURAL_RESPONSE_INVALID"
            )
        if field == expected_field:
            if len(rows) != 1 or not isinstance(rows[0], Mapping) or str(
                rows[0].get(identity_field)
            ) != target:
                raise AutonomousTransferWorkflowError(
                    "AUTONOMOUS_OBSERVATION_STRUCTURAL_SCOPE_MISMATCH"
                )
        elif rows:
            raise AutonomousTransferWorkflowError(
                "AUTONOMOUS_OBSERVATION_STRUCTURAL_SCOPE_MISMATCH"
            )


def _merge_structural_observations(
    rows: list[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if not rows:
        return None
    result: dict[str, list[object]] = {
        "capabilities": [],
        "execution_interfaces": [],
        "hooks": [],
        "operational_metrics": [],
    }
    seen: set[bytes] = set()
    for row in rows:
        for field in result:
            values = row.get(field)
            if not isinstance(values, list):
                raise AutonomousTransferWorkflowError(
                    "AUTONOMOUS_OBSERVATION_STRUCTURAL_RESPONSE_INVALID"
                )
            for value in values:
                encoded = _canonical_any_bytes(value)
                if encoded in seen:
                    continue
                seen.add(encoded)
                result[field].append(value)
    return result


def _archive_observation_bytes(
    execution: Mapping[str, object],
    payload: bytes,
    *,
    project_root: Path,
) -> str:
    raw = Path(str(execution.get("archive_root") or "")).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INVALID"
        )
    archive_root = raw.resolve()
    if archive_root == project_root or project_root in archive_root.parents:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_ARCHIVE_INSIDE_SOURCE"
        )
    digest = hashlib.sha256(payload).hexdigest()
    path = archive_root / "cas" / "sha256" / digest[:2] / digest
    _write_bytes_idempotent(path, payload)
    return "sha256:" + digest


def _canonical_any_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_PAYLOAD_INVALID"
        ) from exc


def _state_root_from_attempt(attempt_root: Path) -> Path | None:
    for parent in Path(attempt_root).resolve().parents:
        if (parent / "controller.db").is_file():
            return parent
    return None


def _gap_profile_requirements(
    planning: Mapping[str, object], *, profile_id: str
) -> list[Mapping[str, object]]:
    profiles = planning.get("profile_requirements")
    if not isinstance(profiles, list):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_GAP_PROFILE_REQUIREMENTS_INVALID"
        )
    matches = [
        row
        for row in profiles
        if isinstance(row, Mapping) and row.get("profile_id") == profile_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("requirements"), list):
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_GAP_PROFILE_REQUIREMENTS_MISSING:{profile_id}"
        )
    return [
        row
        for row in matches[0]["requirements"]
        if isinstance(row, Mapping)
    ]


def _bound_portable_knowledge_graph(
    planning: Mapping[str, object],
) -> Path | None:
    value = planning.get("portable_knowledge_graph")
    expected = planning.get("portable_knowledge_graph_sha256")
    if value is None and expected is None:
        return None
    if not isinstance(value, str) or not isinstance(expected, str):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_GAP_KNOWLEDGE_GRAPH_BINDING_INVALID"
        )
    path = _require_file(
        Path(value), "AUTONOMOUS_GAP_KNOWLEDGE_GRAPH_INVALID"
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_GAP_KNOWLEDGE_GRAPH_HASH_MISMATCH"
        )
    return path


def _gap_next_state(receipt: Mapping[str, object]) -> str:
    states = {
        "ready_for_portfolio": "pending_portfolio",
        "requires_manufacturing": "pending_portfolio",
        "requires_interface_extension": "pending_interface_extension",
        "missing_data_regime": "missing_data_regime",
        "architecture_bound": "architecture_bound",
    }
    state = str(receipt.get("state"))
    if state not in states:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_GAP_STATE_INVALID")
    return states[state]


def _load_bound_context_document(
    context: Mapping[str, object],
    *,
    path_key: str,
    sha256_key: str,
    id_key: str,
    document_id: str,
    code: str,
) -> dict[str, object]:
    path = _require_file(
        Path(str(context.get(path_key) or "")),
        f"{code}_INVALID",
    )
    expected_sha256 = context.get(sha256_key)
    if (
        not isinstance(expected_sha256, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise AutonomousTransferWorkflowError(f"{code}_HASH_MISMATCH")
    document = _load(path, f"{code}_INVALID")
    if document.get(document_id) != context.get(id_key):
        raise AutonomousTransferWorkflowError(f"{code}_BINDING_MISMATCH")
    return document


def _load_bound_hypothesis_batch(
    planning: Mapping[str, object], *, project_root: Path
) -> dict[str, object]:
    path = _require_file(
        Path(str(planning.get("hypothesis_batch") or "")),
        "AUTONOMOUS_HYPOTHESIS_BATCH_INVALID",
    )
    expected = planning.get("hypothesis_batch_sha256")
    if (
        not isinstance(expected, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_HYPOTHESIS_BATCH_HASH_MISMATCH"
        )
    batch = _load(path, "AUTONOMOUS_HYPOTHESIS_BATCH_INVALID")
    try:
        validate_hypothesis_batch(batch, root=project_root)
    except ExperimentPortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_HYPOTHESIS_BATCH_INVALID:{exc}"
        ) from exc
    return batch


def _load_bound_gap_plan(
    work: Mapping[str, object], *, project_root: Path
) -> dict[str, object]:
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    path = _require_file(
        Path(str(context.get("capability_gap_plan_path") or "")),
        "AUTONOMOUS_CAPABILITY_GAP_PLAN_INVALID",
    )
    expected = context.get("capability_gap_plan_sha256")
    if (
        not isinstance(expected, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CAPABILITY_GAP_PLAN_HASH_MISMATCH"
        )
    plan = _load(path, "AUTONOMOUS_CAPABILITY_GAP_PLAN_INVALID")
    if plan.get("plan_id") != context.get("capability_gap_plan_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CAPABILITY_GAP_PLAN_BINDING_MISMATCH"
        )
    graph_path = _require_file(
        Path(str(context.get("capability_requirement_graph_path") or "")),
        "AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH_INVALID",
    )
    expected_graph_sha256 = context.get("capability_requirement_graph_sha256")
    if (
        not isinstance(expected_graph_sha256, str)
        or hashlib.sha256(graph_path.read_bytes()).hexdigest()
        != expected_graph_sha256
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH_HASH_MISMATCH"
        )
    graph = _load(
        graph_path,
        "AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH_INVALID",
    )
    try:
        validate_capability_requirement_graph(graph, root=project_root)
        validate_gap_plan_against_requirement_graph(
            plan,
            graph,
            root=project_root,
        )
    except CapabilityGapPlannerError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_CAPABILITY_GAP_PLAN_INVALID:{exc}"
        ) from exc
    if graph.get("graph_id") != context.get("capability_requirement_graph_id"):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CAPABILITY_REQUIREMENT_GRAPH_BINDING_MISMATCH"
        )
    return plan


def _load_bound_portfolio(
    work: Mapping[str, object],
    *,
    gap_plan: Mapping[str, object],
    project_root: Path,
) -> dict[str, object]:
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    portfolio = _load_bound_context_document(
        context,
        path_key="experiment_portfolio_path",
        sha256_key="experiment_portfolio_sha256",
        id_key="experiment_portfolio_id",
        document_id="portfolio_id",
        code="AUTONOMOUS_EXPERIMENT_PORTFOLIO",
    )
    try:
        validate_experiment_portfolio(portfolio, root=project_root)
    except ExperimentPortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_EXPERIMENT_PORTFOLIO_INVALID:{exc}"
        ) from exc
    bindings = portfolio.get("bindings")
    if not isinstance(bindings, Mapping) or (
        bindings.get("gap_plan_id") != gap_plan.get("plan_id")
        or bindings.get("gap_plan_digest") != gap_plan.get("plan_digest")
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_EXPERIMENT_PORTFOLIO_GAP_BINDING_MISMATCH"
        )
    return portfolio


def _load_resource_policy(
    config: Mapping[str, object], *, project_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    policy = config.get("resource_portfolio")
    if not isinstance(policy, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_RESOURCE_PORTFOLIO_REQUIRED")
    allocation = policy.get("resource_allocation")
    if not isinstance(allocation, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_ALLOCATION_INVALID"
        )
    experiment_scale = _load_hashed_resource_document(
        policy,
        path_key="experiment_scale_plan",
        sha256_key="experiment_scale_plan_sha256",
        code="AUTONOMOUS_EXPERIMENT_SCALE_PLAN",
    )
    conversion_scale = _load_hashed_resource_document(
        policy,
        path_key="conversion_scale_plan",
        sha256_key="conversion_scale_plan_sha256",
        code="AUTONOMOUS_CONVERSION_SCALE_PLAN",
    )
    roles = allocation.get("roles")
    if not isinstance(roles, list):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_RESOURCE_ROLES_INVALID")
    candidate_roles = [
        row
        for row in roles
        if isinstance(row, Mapping)
        and row.get("role") == "autonomous_candidate_evaluation"
    ]
    conversion_roles = [
        row
        for row in roles
        if isinstance(row, Mapping) and row.get("role") == "droid_data_preparation"
    ]
    if len(candidate_roles) != 1 or len(conversion_roles) != 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_RESOURCE_ROLES_INVALID")
    candidate_gpus = candidate_roles[0].get("gpu_indices")
    conversion_gpus = conversion_roles[0].get("gpu_indices")
    if (
        not isinstance(candidate_gpus, list)
        or not isinstance(conversion_gpus, list)
        or candidate_gpus != config.get("gpu_indices")
        or candidate_roles[0].get("max_parallel_jobs")
        != config.get("max_parallel_gpu_jobs")
        or set(candidate_gpus) & set(conversion_gpus)
        or len(conversion_gpus) != 2
        or len(candidate_gpus) > 6
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_ROLE_BINDING_MISMATCH"
        )
    if Path(str(policy["experiment_scale_plan"])).resolve() == project_root:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_EXPERIMENT_SCALE_PLAN_INVALID"
        )
    return dict(policy), experiment_scale, conversion_scale


def _load_hashed_resource_document(
    policy: Mapping[str, object],
    *,
    path_key: str,
    sha256_key: str,
    code: str,
) -> dict[str, object]:
    path = _require_file(Path(str(policy.get(path_key) or "")), code + "_INVALID")
    expected = policy.get(sha256_key)
    if (
        not isinstance(expected, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise AutonomousTransferWorkflowError(code + "_HASH_MISMATCH")
    return _load(path, code + "_INVALID")


def _manufacturing_portfolio_entry_ids(
    context: Mapping[str, object], *, project_root: Path
) -> tuple[str, ...]:
    raw_path = context.get("module_manufacturing_work_order_path")
    if raw_path is None:
        return ()
    expected = context.get("module_manufacturing_work_order_sha256")
    if not isinstance(expected, str):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_ORDER_BINDING_INVALID"
        )
    try:
        order = load_module_manufacturing_work_order(
            Path(str(raw_path)),
            expected_sha256=expected,
            root=project_root,
        )
    except ModuleManufacturingError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_MODULE_MANUFACTURING_ORDER_INVALID:{exc}"
        ) from exc
    return tuple(str(value) for value in order["portfolio_entry_ids"])


def _load_bound_resource_receipt(
    context: Mapping[str, object], *, phase: str, project_root: Path
) -> dict[str, object]:
    path = context.get("resource_portfolio_path")
    expected = context.get("resource_portfolio_sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_PORTFOLIO_BINDING_REQUIRED"
        )
    try:
        receipt = load_resource_portfolio_receipt(
            Path(path),
            expected_sha256=expected,
            root=project_root,
        )
    except ResourcePortfolioError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_RESOURCE_PORTFOLIO_INVALID:{exc}"
        ) from exc
    if receipt.get("phase") != phase:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_RESOURCE_PORTFOLIO_PHASE_MISMATCH"
        )
    return receipt


def _confirm_gpu_count(policy: Mapping[str, object], *, profile_id: str) -> int:
    overrides = policy.get("confirm_gpu_count_by_profile")
    if isinstance(overrides, Mapping) and profile_id in overrides:
        value = overrides[profile_id]
    else:
        value = policy["default_confirm_gpu_count"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_CONFIRM_GPU_COUNT_INVALID"
        )
    return value


def _load_profile_training_scale_plan(
    policy: Mapping[str, object], *, profile_id: str
) -> dict[str, object]:
    plans = policy.get("training_scale_plans")
    registration = plans.get(profile_id) if isinstance(plans, Mapping) else None
    if not isinstance(registration, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_DISTRIBUTED_TRAINING_SCALE_REQUIRED"
        )
    path = _require_file(
        Path(str(registration.get("path") or "")),
        "AUTONOMOUS_DISTRIBUTED_TRAINING_SCALE_INVALID",
    )
    expected = registration.get("sha256")
    if (
        not isinstance(expected, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_DISTRIBUTED_TRAINING_SCALE_HASH_MISMATCH"
        )
    return _load(path, "AUTONOMOUS_DISTRIBUTED_TRAINING_SCALE_INVALID")


def _load_manufacturing_composition(
    context: Mapping[str, object],
    *,
    graph: Mapping[str, object],
    requirement_id: str,
) -> dict[str, object]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_GRAPH_INVALID"
        )
    selected = [row for row in nodes if row.get("requirement_id") == requirement_id]
    if len(selected) != 1:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_REQUIREMENT_UNKNOWN"
        )
    resolution = selected[0].get("resolution")
    if not isinstance(resolution, Mapping):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_MISSING"
        )
    composition_id = resolution.get("composition_id")
    if not isinstance(composition_id, str):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_MISSING"
        )
    raw_paths = context.get("module_composition_paths")
    if not isinstance(raw_paths, list):
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_MISSING"
        )
    matches = []
    for raw in raw_paths:
        path = _require_file(
            Path(str(raw)),
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_INVALID",
        )
        payload = _load(
            path,
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_INVALID",
        )
        if payload.get("composition_id") == composition_id:
            matches.append(payload)
    if len(matches) != 1:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_MODULE_MANUFACTURING_COMPOSITION_MISSING"
        )
    return matches[0]


def _observation_next_state(plan: Mapping[str, object]) -> str:
    task_types = {str(row["task_type"]) for row in plan["tasks"]}
    if "architecture_bound" in task_types:
        return "architecture_bound"
    if "missing_data_regime" in task_types:
        return "missing_data_regime"
    if "requires_evaluator_binding" in task_types:
        return "requires_evaluator_binding"
    if "generate_interface_extension" in task_types:
        return "pending_interface_extension"
    if "manufacture_shadow_probe" in task_types:
        return "pending_shadow_probe_admission"
    return "pending_probe_execution"


def _materializer_registration(
    config: Mapping[str, object], *, work: Mapping[str, object]
) -> Mapping[str, object]:
    materializers = config.get("materializers")
    if not isinstance(materializers, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MATERIALIZER_REGISTRY_INVALID")
    registration = materializers.get(str(work["profile_id"]))
    if isinstance(registration, Mapping):
        return registration
    context = work.get("context")
    if not isinstance(context, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_WORK_CONTEXT_INVALID")
    admission_path = context.get("automatic_module_admission_path")
    admission_sha256 = context.get("automatic_module_admission_sha256")
    if not isinstance(admission_path, str) or not isinstance(admission_sha256, str):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_AUTOMATIC_MODULE_ADMISSION_MISSING")
    try:
        admission = load_automatic_module_admission(
            Path(admission_path),
            expected_sha256=admission_sha256,
            project_root=_paths(config)["project_root"],
        )
    except AutomaticModulePlanError as exc:
        raise AutonomousTransferWorkflowError(str(exc)) from exc
    if (
        admission.get("candidate_id") != context.get("candidate_id")
        or admission.get("idea_id") != work.get("idea_id")
        or admission.get("abi_id") != context.get("automatic_module_abi_id")
    ):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_AUTOMATIC_MODULE_BINDING_MISMATCH")
    return {"evaluator": admission["evaluator_path"], "automatic_admission": admission}


def _materializer_unavailable(
    *, work: Mapping[str, object], attempt_root: Path
) -> StageResult:
    gap_path = attempt_root.parent / "materializer-capability-gap.json"
    gap = {
        "schema_version": 1,
        "artifact_type": "verdiwm-materialization-capability-gap",
        "state": "capability_gap",
        "candidate_id": str(work["idea_id"]),
        "idea_id": str(work["idea_id"]),
        "profile_id": str(work["profile_id"]),
        "source_id": str(work["source_id"]),
        "source_digest": str(work["source_digest"]),
        "assessment_digest": str(work["assessment_digest"]),
        "blockers": [{"code": "MATERIALIZER_UNAVAILABLE"}],
        "claim_boundary": (
            "No implementation or GPU authority exists for this mechanism profile. "
            "The controller must not substitute a similar registered method."
        ),
    }
    _write_json_idempotent(gap_path, gap)
    return StageResult(
        state="blocked",
        outcome="materializer_unavailable",
        payload={"capability_gap_path": str(gap_path)},
        receipt_path=gap_path,
    )


def _automatic_module_policy_gap(
    *, work: Mapping[str, object], attempt_root: Path, detail: str
) -> StageResult:
    gap_path = attempt_root.parent / "automatic-module-policy-gap.json"
    gap = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-module-capability-gap",
        "state": "requires_interface_extension",
        "idea_id": str(work["idea_id"]),
        "blockers": [{"code": "AUTOMATIC_MODULE_POLICY_REJECTED", "detail": detail}],
        "side_effects": {
            "source_mutated": False,
            "gpu_execution_started": False,
            "gpu_scheduling_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "The generated module failed a trusted compiler boundary and grants no "
            "materialization, GPU, evaluator, or promotion authority."
        ),
    }
    _write_json_idempotent(gap_path, gap)
    return StageResult(
        state="blocked",
        outcome="automatic_module_policy_rejected",
        payload={"capability_gap_path": str(gap_path)},
        receipt_path=gap_path,
    )


def _paths(config: Mapping[str, object]) -> dict[str, Path]:
    raw = config.get("paths")
    if not isinstance(raw, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PATHS_INVALID")
    return {
        str(name): Path(str(value)).expanduser().resolve() for name, value in raw.items()
    }


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ATTEMPT_ROOT_INVALID") from exc


def _failure_query(value: str) -> str:
    return _failure_queries(value)[0]


def _failure_queries(value: str) -> tuple[str, ...]:
    normalized = value.lower().replace(":", " ").replace("_", " ")
    queries: list[str] = []
    if "action conditioning" in normalized:
        queries.extend(
            [
                "vision language action model action conditioning robustness",
                "video generation action conditioned temporal consistency",
            ]
        )
    elif "trajectory fidelity" in normalized:
        queries.extend(
            [
                "video diffusion trajectory fidelity preservation long horizon",
                "world model rollout consistency motion dynamics",
            ]
        )
    elif "horizon drift" in normalized or "long horizon" in normalized:
        queries.extend(
            [
                "long horizon video world model rollout drift memory",
                "autoregressive video generation temporal consistency self forcing",
            ]
        )
    elif "materializer" in normalized or "capability" in normalized:
        queries.extend(
            [
                "modular multimodal world model adapter typed interface",
                "vision language model world model representation transfer",
            ]
        )
    else:
        focus = normalized[:80].strip() or "world model transfer failure"
        queries.append(f"world model {focus} optimization method")
    return tuple(dict.fromkeys(queries))


def _external_research_requires_network(config: Mapping[str, object]) -> bool:
    policy = config.get("external_research")
    return isinstance(policy, Mapping) and policy.get("network_required") is True


def _external_research_minimum_sources(config: Mapping[str, object]) -> int:
    policy = config.get("external_research")
    if not isinstance(policy, Mapping):
        return 1
    value = policy.get("minimum_successful_sources", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_EXTERNAL_RESEARCH_POLICY_INVALID")
    return value


def _network_readiness_error(
    manifest: Mapping[str, object], *, minimum_successful_sources: int
) -> str | None:
    if manifest.get("retrieval_mode") != "network":
        return "AUTONOMOUS_EXTERNAL_RESEARCH_NETWORK_MODE_REQUIRED"
    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, list):
        return "AUTONOMOUS_EXTERNAL_RESEARCH_RETRIEVAL_RECEIPT_MISSING"
    successful = {
        str(row.get("source"))
        for row in retrieval
        if isinstance(row, Mapping) and row.get("state") == "fetched"
    }
    if len(successful) < minimum_successful_sources:
        return (
            "AUTONOMOUS_EXTERNAL_RESEARCH_SOURCES_UNAVAILABLE:"
            f"{len(successful)}<{minimum_successful_sources}"
        )
    return None


def _load_portable_knowledge_documents(root: Path) -> list[Mapping[str, object]]:
    """Load only explicitly staged portable records, never the local state tree."""

    raw = Path(root).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INVALID")
    resolved = raw.resolve()
    if not resolved.exists():
        return []
    if not resolved.is_dir():
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INVALID")
    documents: list[Mapping[str, object]] = []
    for path in sorted(resolved.rglob("*.json")) + sorted(resolved.rglob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INVALID")
        try:
            payloads = (
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else [json.loads(path.read_text(encoding="utf-8"))]
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INVALID") from exc
        if any(not isinstance(payload, Mapping) for payload in payloads):
            raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTABLE_RECORDS_INVALID")
        documents.extend(payloads)
    return documents


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousTransferWorkflowError(code) from exc
    if not isinstance(payload, dict):
        raise AutonomousTransferWorkflowError(code)
    return payload


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise AutonomousTransferWorkflowError(code)
    return resolved


def _require_directory(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(code)
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise AutonomousTransferWorkflowError(code)
    return resolved


def _write_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_idempotent(path, _canonical_bytes(payload))


def _replace_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_INVALID")
    if path.is_file() and path.read_bytes() == encoded:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
