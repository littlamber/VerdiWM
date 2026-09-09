"""Materialization stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import copy
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.automatic_materialization import materialization_plan_digest, run_automatic_materialization
from wmloop.execute.llm_task_adapter import run_llm_task
from wmloop.control.automatic_module_plan import AutomaticModulePlanError, compile_automatic_module_plan, load_automatic_module_admission
from wmloop.control.open_method_pipeline import OpenMethodPipelineError, build_open_method_request, compile_open_method_proposal
from wmloop.control.module_manufacturing import ModuleManufacturingError, build_intervention_manufacturing_work_order
from .planning import _load_bound_context_document, _load_bound_gap_plan, _load_bound_portfolio, _load_manufacturing_composition
from .common import AutonomousTransferWorkflowError, StageResult, _attempt_number, _canonical_bytes, _load, _load_active_portrait, _paths, _require_file, _state_root_from_attempt, _write_json_idempotent


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
