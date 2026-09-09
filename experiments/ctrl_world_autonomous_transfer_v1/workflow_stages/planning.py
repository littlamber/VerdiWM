"""Planning stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from wmloop.control.capability_gap_planner import CapabilityGapPlannerError, build_goal_ir, compile_capability_gap_plan, validate_capability_requirement_graph, validate_gap_plan_against_requirement_graph
from wmloop.control.experiment_portfolio import ExperimentPortfolioError, compile_experiment_portfolio, validate_experiment_portfolio, validate_hypothesis_batch
from wmloop.control.module_manufacturing import ModuleManufacturingError, load_module_manufacturing_work_order
from wmloop.control.resource_portfolio import ResourcePortfolioError, build_confirm_resource_portfolio_receipt, build_screen_resource_portfolio_receipt, load_resource_portfolio_receipt
from .common import AutonomousTransferWorkflowError, StageResult, _load, _load_active_portrait, _paths, _require_file, _state_root_from_attempt, _write_json_idempotent


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
