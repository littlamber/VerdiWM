"""Observation stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.observation_adapter import ObservationAdapterError, run_observation_adapter
from wmloop.control.adaptive_observation import AdaptiveObservationError, build_adaptive_probe_plan, load_observation_abi_registry, validate_adaptive_probe_plan
from wmloop.control.model_portrait import ModelPortraitError, build_portrait_readiness_receipt, probe_coverage_key, update_model_portrait_from_observation, validate_model_portrait
from wmloop.control.shadow_probe_admission import ShadowProbeAdmissionError, compile_shadow_probe_admission
from wmloop.geometry.community_knowledge import CommunityKnowledgeError, build_portrait_transition, validate_portrait_transition
from wmloop.geometry.portable_transfer_knowledge import build_probe_fingerprint_summary
from wmloop.control.module_manufacturing import ModuleManufacturingError, build_observation_manufacturing_work_order
from wmloop.experiments.portable_knowledge_graph import stage_portable_knowledge_records
from .planning import _load_bound_context_document
from .common import AutonomousTransferWorkflowError, StageResult, _canonical_any_bytes, _canonical_bytes, _load, _load_active_portrait, _paths, _replace_json_atomic, _require_file, _state_root_from_attempt, _write_bytes_idempotent, _write_json_idempotent


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
