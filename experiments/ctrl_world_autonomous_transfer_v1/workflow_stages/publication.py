"""Publication stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate.shadow_metric_evolution import ShadowMetricEvolutionError, compile_shadow_metric_proposals
from wmloop.control.model_portrait import ModelPortraitError, derive_model_portrait, validate_model_portrait
from wmloop.geometry.community_knowledge import CommunityKnowledgeError, build_portrait_transition, validate_portrait_transition
from wmloop.experiments.evidence_graph import write_evidence_graph
from wmloop.experiments.portable_knowledge_graph import stage_portable_knowledge_records, write_portable_knowledge_graph
from wmloop.experiments.verified_transfer_knowledge import stage_verified_transfer_knowledge
from experiments.ctrl_world_autonomous_transfer_v1.replanning import ClosedLoopReplanningError, archive_work, build_next_task_decision, build_quality_audit, load_archive_receipts
from .common import AutonomousTransferWorkflowError, StageResult, _canonical_bytes, _load, _load_active_portrait, _paths, _replace_json_atomic, _require_directory, _require_file, _write_bytes_idempotent, _write_json_idempotent


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
