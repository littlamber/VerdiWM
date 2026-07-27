"""Build a read-only dependency plan for clearing the current M4 launch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class M4UnblockPlanError(RuntimeError):
    """M4 unblock plan generation failed closed."""


def run_m4_unblock_plan(
    *,
    phase_gate_manifest: Path,
    blocker_decision_matrix_manifest: Path,
    output_root: Path,
    checkpoint_recovery_packet_manifest: Path | None = None,
    protocol_decision_packet_manifest: Path | None = None,
    attribution_remediation_manifest: Path | None = None,
    protocol_application_preview_manifest: Path | None = None,
    official_asset_inventory_manifest: Path | None = None,
    checkpoint_self_training_preflight_manifest: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a dependency-ordered M4 unblock plan without launching work."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(Path(cas_storage_root))
    sources = {
        "phase_gate": _load_source(phase_gate_manifest, cas=cas, archive=archive),
        "blocker_decision_matrix": _load_source(blocker_decision_matrix_manifest, cas=cas, archive=archive),
    }
    if checkpoint_recovery_packet_manifest is not None:
        sources["checkpoint_recovery_packet"] = _load_source(checkpoint_recovery_packet_manifest, cas=cas, archive=archive)
    if protocol_decision_packet_manifest is not None:
        sources["protocol_decision_packet"] = _load_source(protocol_decision_packet_manifest, cas=cas, archive=archive)
    if attribution_remediation_manifest is not None:
        sources["attribution_remediation"] = _load_source(attribution_remediation_manifest, cas=cas, archive=archive)
    if protocol_application_preview_manifest is not None:
        sources["protocol_application_preview"] = _load_source(protocol_application_preview_manifest, cas=cas, archive=archive)
    if official_asset_inventory_manifest is not None:
        sources["official_asset_inventory"] = _load_source(official_asset_inventory_manifest, cas=cas, archive=archive)
    if checkpoint_self_training_preflight_manifest is not None:
        sources["checkpoint_self_training_preflight"] = _load_source(
            checkpoint_self_training_preflight_manifest,
            cas=cas,
            archive=archive,
        )

    phase_gate = _report_or_payload(sources, "phase_gate")
    blocker_matrix = _report_or_payload(sources, "blocker_decision_matrix")
    official_asset_inventory = _report_or_payload_or_none(sources, "official_asset_inventory")
    checkpoint_self_training_preflight = _report_or_payload_or_none(sources, "checkpoint_self_training_preflight")
    _validate_core_inputs(phase_gate=phase_gate, blocker_matrix=blocker_matrix)
    _validate_optional_inputs(
        official_asset_inventory=official_asset_inventory,
        checkpoint_self_training_preflight=checkpoint_self_training_preflight,
    )
    phase_blockers = _phase_blockers(phase_gate)
    m4_launch_allowed = phase_gate.get("m4_launch_allowed") is True and phase_gate.get("state") == "ready"
    dependencies = _dependency_steps(
        phase_blockers=phase_blockers,
        blocker_matrix=blocker_matrix,
        checkpoint_recovery=_report_or_payload_or_none(sources, "checkpoint_recovery_packet"),
        protocol_decision=_report_or_payload_or_none(sources, "protocol_decision_packet"),
        attribution_remediation=_report_or_payload_or_none(sources, "attribution_remediation"),
        protocol_application_preview=_report_or_payload_or_none(sources, "protocol_application_preview"),
        official_asset_inventory=official_asset_inventory,
        checkpoint_self_training_preflight=checkpoint_self_training_preflight,
        m4_launch_allowed=m4_launch_allowed,
    )
    official_asset_inventory_summary = _official_asset_inventory_summary(official_asset_inventory)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m4-unblock-dependency-plan",
        "state": "ready_for_m4_launch" if m4_launch_allowed else "awaiting_external_and_human_resolution",
        "m4_launch_allowed": m4_launch_allowed,
        "formal_training_allowed": m4_launch_allowed,
        "phase_gate_state": phase_gate.get("state"),
        "phase_gate_blocker_count": len(phase_blockers),
        "blockers_by_authority": blocker_matrix.get("route_counts", {}),
        "official_asset_inventory": official_asset_inventory_summary,
        "dependency_count": len(dependencies),
        "blocking_dependency_count": sum(1 for item in dependencies if item.get("blocks_m4") is True),
        "dependencies": dependencies,
        "critical_path": _critical_path(dependencies),
        "next_safe_actions": _next_safe_actions(dependencies=dependencies, m4_launch_allowed=m4_launch_allowed),
        "sources": {name: source["summary"] for name, source in sources.items()},
        "limitations": [
            "This plan is read-only and does not approve protocol changes, sign attribution, install checkpoints, or launch GPU work.",
            "Free GPUs are not M4 authorization; formal training remains forbidden until phase_gate reports m4_launch_allowed=true.",
            "Every resolved dependency must be followed by regenerated evidence and a new strict phase gate.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _validate_core_inputs(*, phase_gate: Mapping[str, Any], blocker_matrix: Mapping[str, Any]) -> None:
    if phase_gate.get("artifact_type") != "wmloop-strict-phase-gate":
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_PHASE_GATE_INVALID")
    if blocker_matrix.get("artifact_type") != "wmloop-blocker-decision-matrix":
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_BLOCKER_MATRIX_INVALID")


def _validate_optional_inputs(
    *,
    official_asset_inventory: Mapping[str, Any] | None,
    checkpoint_self_training_preflight: Mapping[str, Any] | None,
) -> None:
    if official_asset_inventory is not None and official_asset_inventory.get("artifact_type") != "acwm-official-asset-inventory":
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_OFFICIAL_ASSET_INVENTORY_INVALID")
    if (
        checkpoint_self_training_preflight is not None
        and checkpoint_self_training_preflight.get("artifact_type") != "acwm-m0-checkpoint-self-training-preflight"
    ):
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_CHECKPOINT_SELF_TRAINING_PREFLIGHT_INVALID")


def _dependency_steps(
    *,
    phase_blockers: Sequence[str],
    blocker_matrix: Mapping[str, Any],
    checkpoint_recovery: Mapping[str, Any] | None,
    protocol_decision: Mapping[str, Any] | None,
    attribution_remediation: Mapping[str, Any] | None,
    protocol_application_preview: Mapping[str, Any] | None,
    official_asset_inventory: Mapping[str, Any] | None,
    checkpoint_self_training_preflight: Mapping[str, Any] | None,
    m4_launch_allowed: bool,
) -> list[dict[str, object]]:
    if m4_launch_allowed:
        return [
            {
                "step_id": "launch_m4_guarded_campaign",
                "status": "ready",
                "authority": "phase_gate",
                "blocks_m4": False,
                "resolves_requirements": [],
                "gpu_policy": "formal_m4_training_allowed_by_phase_gate",
                "next_actions": ["Launch formal M4 trials only through guarded entrypoints and track settled-trial progress."],
            }
        ]
    routes = _routes_by_requirement(blocker_matrix)
    steps: list[dict[str, object]] = []
    if _any_blocker(phase_blockers, ("M0_baseline_reproduction", "M0_checkpoint_source_resolution", "M0_checkpoint_launch_guard_clear")):
        steps.append(
            _checkpoint_step(
                routes=routes,
                checkpoint_recovery=checkpoint_recovery,
                official_asset_inventory=official_asset_inventory,
                checkpoint_self_training_preflight=checkpoint_self_training_preflight,
            )
        )
    if _any_blocker(phase_blockers, ("M1_raw_failure_reports", "M1_original_g1_data_feasibility")):
        steps.append(
            _protocol_step(
                routes=routes,
                protocol_decision=protocol_decision,
                protocol_application_preview=protocol_application_preview,
            )
        )
    if "M1_attribution_review" in phase_blockers:
        steps.append(_attribution_step(routes=routes, attribution_remediation=attribution_remediation))
    if "M3_strict_acceptance" in phase_blockers:
        steps.append(_m3_rerun_step())
    steps.append(_phase_gate_rerun_step())
    return steps


def _checkpoint_step(
    *,
    routes: Mapping[str, Mapping[str, Any]],
    checkpoint_recovery: Mapping[str, Any] | None,
    official_asset_inventory: Mapping[str, Any] | None,
    checkpoint_self_training_preflight: Mapping[str, Any] | None,
) -> dict[str, object]:
    recovery_state = checkpoint_recovery.get("state") if isinstance(checkpoint_recovery, Mapping) else "not_provided"
    environment = checkpoint_recovery.get("environment") if isinstance(checkpoint_recovery, Mapping) else "cloth_move"
    official_summary = _official_asset_inventory_summary(official_asset_inventory)
    source_resolution_summary = _checkpoint_source_resolution_summary(checkpoint_recovery)
    self_training_summary = _checkpoint_self_training_preflight_summary(checkpoint_self_training_preflight)
    next_actions = []
    if isinstance(checkpoint_recovery, Mapping):
        raw_actions = checkpoint_recovery.get("next_actions")
        if isinstance(raw_actions, list):
            next_actions.extend(str(item) for item in raw_actions)
    if _official_current_redownload_is_not_fix(official_summary):
        _append_unique(
            next_actions,
            "Do not redownload the current Hugging Face main checkpoint as the fix; the official current hash still matches the blocked local mismatch.",
        )
    revision_watch = source_resolution_summary.get("revision_watch")
    if isinstance(revision_watch, Mapping) and revision_watch.get("state") == "revision_probe_unreachable":
        _append_unique(
            next_actions,
            "Rerun the checkpoint revision watch when Hugging Face refs/revision metadata is reachable; current revision status is unknown.",
        )
    if isinstance(self_training_summary, Mapping) and self_training_summary.get("state") == "awaiting_human_approval":
        _append_unique(
            next_actions,
            "Review the checkpoint self-training preflight packet before approving any continuation-training branch.",
        )
    if isinstance(self_training_summary, Mapping) and self_training_summary.get("state") == "staged_for_manual_launch_planning":
        _append_unique(
            next_actions,
            "Treat checkpoint self-training as launch-planning staged only; it still must produce a quarantine-validated candidate before M4 can open.",
        )
    if not next_actions:
        next_actions = [
            "Provide a verified 100k-step checkpoint candidate under quarantine.",
            "Run checkpoint quarantine validation and install only after explicit confirmation.",
            "Rerun M0 checkpoint audits, launch guard smoke, and baseline reproduction.",
        ]
    return {
        "step_id": "resolve_checkpoint_source",
        "status": recovery_state,
        "authority": "external_checkpoint_required",
        "blocks_m4": True,
        "resolves_requirements": [
            "M0_baseline_reproduction",
            "M0_checkpoint_source_resolution",
            "M0_checkpoint_launch_guard_clear",
        ],
        "prerequisites": ["verified_external_checkpoint_candidate"],
        "environment": environment,
        "official_asset_inventory": official_summary,
        "source_resolution": source_resolution_summary,
        "self_training_preflight": self_training_summary,
        "current_main_redownload_resolves": not _official_current_redownload_is_not_fix(official_summary),
        "resolution_branches": _checkpoint_resolution_branches(
            environment=str(environment),
            official_asset_inventory=official_summary,
        ),
        "gpu_policy": _route_gpu_policy(routes, "M0_checkpoint_source_resolution", "no_gpu_until_checkpoint_quarantine_passes"),
        "next_actions": next_actions,
    }


def _protocol_step(
    *,
    routes: Mapping[str, Mapping[str, Any]],
    protocol_decision: Mapping[str, Any] | None,
    protocol_application_preview: Mapping[str, Any] | None,
) -> dict[str, object]:
    recommendations = protocol_decision.get("recommendations") if isinstance(protocol_decision, Mapping) else {}
    preview_state = protocol_application_preview.get("state") if isinstance(protocol_application_preview, Mapping) else "not_provided"
    ready_candidates = protocol_application_preview.get("ready_candidate_count") if isinstance(protocol_application_preview, Mapping) else None
    return {
        "step_id": "select_protocol_or_data_path",
        "status": protocol_decision.get("state") if isinstance(protocol_decision, Mapping) else "not_provided",
        "authority": "human_protocol_decision",
        "blocks_m4": True,
        "resolves_requirements": [
            "M1_raw_failure_reports",
            "M1_original_g1_data_feasibility",
            "M3_strict_acceptance",
        ],
        "prerequisites": ["human_protocol_approval"],
        "recommended_options": recommendations if isinstance(recommendations, Mapping) else {},
        "application_preview": {"state": preview_state, "ready_candidate_count": ready_candidates},
        "gpu_policy": _route_gpu_policy(routes, "M1_original_g1_data_feasibility", "no_gpu_until_protocol_path_selected"),
        "next_actions": [
            "Choose a protocol/data path at a human version boundary.",
            "If using a staged candidate, apply it only through protocol_application_apply with explicit human approval.",
            "Regenerate M1 raw failure reports under the approved goal before any M4 launch.",
        ],
    }


def _attribution_step(
    *,
    routes: Mapping[str, Mapping[str, Any]],
    attribution_remediation: Mapping[str, Any] | None,
) -> dict[str, object]:
    action_items = []
    if isinstance(attribution_remediation, Mapping):
        raw_actions = attribution_remediation.get("action_items")
        if isinstance(raw_actions, list):
            action_items.extend(str(item) for item in raw_actions)
    if not action_items:
        action_items = [
            "Resolve attribution review gaps and obtain human signoff on a ready attribution review manifest.",
        ]
    return {
        "step_id": "resolve_attribution_review",
        "status": attribution_remediation.get("state") if isinstance(attribution_remediation, Mapping) else "not_provided",
        "authority": "human_review_required",
        "blocks_m4": True,
        "resolves_requirements": ["M1_attribution_review"],
        "prerequisites": ["human_attribution_resolution", "three_reviewable_failure_reports", "zero_expectation_mismatches"],
        "signoff_allowed": attribution_remediation.get("signoff_allowed") if isinstance(attribution_remediation, Mapping) else None,
        "gpu_policy": _route_gpu_policy(routes, "M1_attribution_review", "no_gpu_needed"),
        "next_actions": action_items,
    }


def _m3_rerun_step() -> dict[str, object]:
    return {
        "step_id": "rerun_m3_acceptance_after_m1",
        "status": "blocked_until_m1_ready",
        "authority": "codex_after_upstream_resolution",
        "blocks_m4": True,
        "resolves_requirements": ["M3_strict_acceptance"],
        "prerequisites": [
            "ready_raw_failure_report_batch",
            "accepted_or_signed_attribution_review",
            "approved_protocol_or_data_path_if_needed",
        ],
        "gpu_policy": "cpu_or_guarded_smoke_until_phase_gate_ready",
        "next_actions": ["Rerun proposal readiness and M3 acceptance audit after M1 evidence is regenerated."],
    }


def _phase_gate_rerun_step() -> dict[str, object]:
    return {
        "step_id": "regenerate_strict_phase_gate",
        "status": "blocked_until_dependencies_resolved",
        "authority": "codex_after_upstream_resolution",
        "blocks_m4": True,
        "resolves_requirements": ["M4_launch"],
        "prerequisites": [
            "checkpoint_source_clear",
            "raw_failure_reports_ready",
            "attribution_review_signed",
            "m3_strict_acceptance_ready",
        ],
        "gpu_policy": "no_formal_m4_training_until_new_gate_ready",
        "next_actions": ["Regenerate the strict phase gate and run m4_launch_guard before any formal M4 entrypoint."],
    }


def _critical_path(dependencies: Sequence[Mapping[str, Any]]) -> list[str]:
    order = [
        "resolve_checkpoint_source",
        "select_protocol_or_data_path",
        "resolve_attribution_review",
        "rerun_m3_acceptance_after_m1",
        "regenerate_strict_phase_gate",
    ]
    present = {str(item.get("step_id")) for item in dependencies}
    return [item for item in order if item in present]


def _next_safe_actions(*, dependencies: Sequence[Mapping[str, Any]], m4_launch_allowed: bool) -> list[str]:
    if m4_launch_allowed:
        return ["M4 gate is open; use guarded campaign entrypoints and keep receipt settlement active."]
    actions = ["Do not launch M4 formal training until a regenerated phase gate reports m4_launch_allowed=true."]
    step_ids = {str(item.get("step_id")) for item in dependencies}
    checkpoint_step = next((item for item in dependencies if item.get("step_id") == "resolve_checkpoint_source"), {})
    if "resolve_checkpoint_source" in step_ids:
        actions.append("Resolve the verified 100k-step cloth_move checkpoint source before rerunning M0/M4 evaluator jobs.")
        if checkpoint_step.get("current_main_redownload_resolves") is False:
            actions.append("Do not spend GPU or operator time on a current-main re-download path; choose a verified historical/publisher fix, approved continuation training, or approved claim/protocol downgrade instead.")
    if "select_protocol_or_data_path" in step_ids:
        actions.append("Get human protocol/data approval before regenerating dataset-length-limited formal failure reports.")
    if "resolve_attribution_review" in step_ids:
        actions.append("Use the attribution remediation packet to resolve mismatch and review-count gaps before signoff.")
    actions.append("After upstream resolutions, rerun M1/M3 evidence, strict phase gate, blocker matrix, backup, and consolidation.")
    return actions


def _official_asset_inventory_summary(official_asset_inventory: Mapping[str, Any] | None) -> dict[str, object] | None:
    if official_asset_inventory is None:
        return None
    return {
        "state": official_asset_inventory.get("state"),
        "checkpoint_mismatch_count": _int_or_none(official_asset_inventory.get("checkpoint_mismatch_count")),
        "remote_checkpoint_candidate_count": _int_or_none(official_asset_inventory.get("remote_checkpoint_candidate_count")),
        "official_current_matches_blocked_local_mismatch_count": _int_or_none(
            official_asset_inventory.get("official_current_matches_blocked_local_mismatch_count")
        ),
        "horizon_limited_environment_count": _int_or_none(official_asset_inventory.get("horizon_limited_environment_count")),
        "m4_launch_allowed": official_asset_inventory.get("m4_launch_allowed") is True,
        "formal_training_allowed": official_asset_inventory.get("formal_training_allowed") is True,
        "active_checkpoint_mutated": official_asset_inventory.get("active_checkpoint_mutated") is True,
        "downloaded_checkpoint_bytes": official_asset_inventory.get("downloaded_checkpoint_bytes") is True,
    }


def _checkpoint_resolution_branches(
    *,
    environment: str,
    official_asset_inventory: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    remote_candidates = int(official_asset_inventory.get("remote_checkpoint_candidate_count") or 0) if official_asset_inventory else 0
    current_main_is_dead_end = _official_current_redownload_is_not_fix(official_asset_inventory)
    return [
        {
            "branch_id": "publisher_fixed_or_historical_checkpoint_candidate",
            "status": "candidate_available" if remote_candidates > 0 else "awaiting_candidate",
            "authority": "external_checkpoint_required",
            "preferred": True,
            "environment": environment,
            "current_main_redownload_resolves": not current_main_is_dead_end,
            "human_approval_required": True,
            "fresh_gpu_exclusivity_audit_required": False,
            "m4_launch_allowed_after_branch": False,
            "required_evidence": [
                "candidate placed under results/quarantine/checkpoints",
                "checkpoint_quarantine validate reports ready_for_manual_install",
                "explicit human confirmation before install",
                "rerun M0 checkpoint audits, launch guard smoke, baseline reproduction, M3 acceptance, and strict phase gate",
            ],
        },
        {
            "branch_id": "continue_or_retrain_cloth_move_to_expected_step",
            "status": "requires_human_approval_and_budget",
            "authority": "human_protocol_and_gpu_budget",
            "preferred": False,
            "environment": environment,
            "current_main_redownload_resolves": False,
            "human_approval_required": True,
            "fresh_gpu_exclusivity_audit_required": True,
            "m4_launch_allowed_after_branch": False,
            "required_evidence": [
                "written approval that self-training is an accepted protocol path",
                "budget accounting under the high-cost GPU policy",
                "fresh GPU exclusivity audit for the selected non-user-occupied GPUs",
                "quarantine or equivalent non-active staging before replacing any active checkpoint",
                "rerun all downstream M0/M1/M3/M4 evidence after the checkpoint is produced",
            ],
        },
        {
            "branch_id": "human_approved_protocol_or_claim_downgrade",
            "status": "requires_human_approval",
            "authority": "human_protocol_decision",
            "preferred": False,
            "environment": environment,
            "current_main_redownload_resolves": False,
            "human_approval_required": True,
            "fresh_gpu_exclusivity_audit_required": False,
            "m4_launch_allowed_after_branch": False,
            "candidate_claims": [
                "7-environment claim excluding cloth_move at common supported horizons such as 16/32",
                "6-environment claim excluding cloth_move and reacher at common supported horizons such as 16/32/48",
            ],
            "required_evidence": [
                "approved protocol/application manifest at a human version boundary",
                "no automatic mutation of active goal config or frozen constitution",
                "regenerated raw failure reports, attribution review, M3 acceptance, and strict phase gate under the approved claim",
            ],
        },
    ]


def _checkpoint_source_resolution_summary(checkpoint_recovery: Mapping[str, Any] | None) -> dict[str, object]:
    if not isinstance(checkpoint_recovery, Mapping):
        return {"state": "not_provided"}
    source_resolution = checkpoint_recovery.get("source_resolution")
    if not isinstance(source_resolution, Mapping):
        return {"state": "not_provided"}
    return {
        "state": source_resolution.get("state"),
        "replacement_candidate_count": source_resolution.get("replacement_candidate_count"),
        "remote_unreachable_count": source_resolution.get("remote_unreachable_count"),
        "remote_watch": _watch_summary(source_resolution.get("remote_watch")),
        "revision_watch": _watch_summary(source_resolution.get("revision_watch")),
    }


def _checkpoint_self_training_preflight_summary(checkpoint_self_training_preflight: Mapping[str, Any] | None) -> dict[str, object]:
    if not isinstance(checkpoint_self_training_preflight, Mapping):
        return {"state": "not_provided"}
    return {
        "state": checkpoint_self_training_preflight.get("state"),
        "environment": checkpoint_self_training_preflight.get("environment"),
        "human_approval_provided": checkpoint_self_training_preflight.get("human_approval_provided") is True,
        "self_training_launch_planning_ready": checkpoint_self_training_preflight.get("self_training_launch_planning_ready")
        is True,
        "packet_grants_training_launch_permission": checkpoint_self_training_preflight.get(
            "packet_grants_training_launch_permission"
        )
        is True,
        "m4_launch_allowed": checkpoint_self_training_preflight.get("m4_launch_allowed") is True,
        "formal_training_allowed": checkpoint_self_training_preflight.get("formal_training_allowed") is True,
        "blocker_count": _int_or_none(checkpoint_self_training_preflight.get("blocker_count")),
        "gpu_preflight": _self_training_gpu_summary(checkpoint_self_training_preflight.get("gpu_preflight")),
        "budget": _self_training_budget_summary(checkpoint_self_training_preflight.get("budget")),
    }


def _self_training_gpu_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"state": "not_provided"}
    return {
        "state": value.get("state"),
        "requested_gpus": value.get("requested_gpus", []),
        "max_age_seconds": value.get("max_age_seconds"),
    }


def _self_training_budget_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"state": "not_provided"}
    return {
        "estimated_gpu_hours": value.get("estimated_gpu_hours"),
        "max_allowed_gpu_hours": value.get("max_allowed_gpu_hours"),
        "high_cost_branch": value.get("high_cost_branch") is True,
        "training_budget_debited": value.get("training_budget_debited") is True,
    }


def _watch_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"state": "not_provided"}
    summary: dict[str, object] = {"state": value.get("state")}
    for key in (
        "candidate_available",
        "candidate_revision_count",
        "scanned_revision_count",
        "downloaded_checkpoint_bytes",
        "active_checkpoint_mutated",
    ):
        if key in value:
            summary[key] = value.get(key)
    return summary


def _official_current_redownload_is_not_fix(official_asset_inventory: Mapping[str, object] | None) -> bool:
    if official_asset_inventory is None:
        return False
    matches = int(official_asset_inventory.get("official_current_matches_blocked_local_mismatch_count") or 0)
    remote_candidates = int(official_asset_inventory.get("remote_checkpoint_candidate_count") or 0)
    return matches > 0 and remote_candidates == 0


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _routes_by_requirement(blocker_matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    routes = blocker_matrix.get("routes")
    if not isinstance(routes, list):
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_BLOCKER_ROUTES_INVALID")
    result: dict[str, Mapping[str, Any]] = {}
    for route in routes:
        if not isinstance(route, Mapping) or not isinstance(route.get("requirement"), str):
            raise M4UnblockPlanError("M4_UNBLOCK_PLAN_BLOCKER_ROUTES_INVALID")
        result[str(route["requirement"])] = route
    return result


def _route_gpu_policy(routes: Mapping[str, Mapping[str, Any]], requirement: str, fallback: str) -> str:
    route = routes.get(requirement)
    if route is None:
        return fallback
    policy = route.get("gpu_policy")
    return str(policy) if isinstance(policy, str) and policy else fallback


def _phase_blockers(phase_gate: Mapping[str, Any]) -> list[str]:
    blockers = phase_gate.get("blockers")
    if not isinstance(blockers, list):
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_PHASE_BLOCKERS_INVALID")
    result: list[str] = []
    for blocker in blockers:
        if not isinstance(blocker, Mapping) or not isinstance(blocker.get("requirement"), str):
            raise M4UnblockPlanError("M4_UNBLOCK_PLAN_PHASE_BLOCKERS_INVALID")
        result.append(str(blocker["requirement"]))
    return result


def _any_blocker(blockers: Sequence[str], names: Sequence[str]) -> bool:
    present = set(blockers)
    return any(name in present for name in names)


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    payload, payload_bytes, payload_path = _read_json(path, "M4_UNBLOCK_PLAN_SOURCE_INVALID")
    report = _load_report_from_manifest(payload)
    manifest_ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(manifest_ref)
    summary: dict[str, object] = {
        "path": str(payload_path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "cas_ref": manifest_ref,
        "artifact_type": payload.get("artifact_type"),
        "state": payload.get("state"),
    }
    source: dict[str, object] = {"payload": payload, "summary": summary}
    if report is not None:
        report_payload, report_bytes, report_path = report
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
        summary["report_path"] = str(report_path)
        summary["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        summary["report_cas_ref"] = report_ref
        source["report"] = report_payload
    return source


def _load_report_from_manifest(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], bytes, Path] | None:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        return None
    return _read_json(Path(report_path), "M4_UNBLOCK_PLAN_REPORT_INVALID")


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise M4UnblockPlanError(f"{error}:{Path(path)}") from exc
    if not isinstance(payload, Mapping):
        raise M4UnblockPlanError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _report_or_payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    try:
        source = sources[name]
    except KeyError as exc:
        raise M4UnblockPlanError(f"M4_UNBLOCK_PLAN_SOURCE_MISSING:{name}") from exc
    return _source_document(source)


def _report_or_payload_or_none(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _source_document(sources[name])


def _source_document(source: Mapping[str, object]) -> Mapping[str, Any]:
    document = source.get("report", source.get("payload"))
    if not isinstance(document, Mapping):
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_SOURCE_DOCUMENT_INVALID")
    return document


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "m4-unblock-plan.json", report_bytes)
        _write_bytes_atomic(temporary / "m4-unblock-plan.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-unblock-dependency-plan-manifest",
            "state": report["state"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "phase_gate_state": report["phase_gate_state"],
            "phase_gate_blocker_count": report["phase_gate_blocker_count"],
            "official_asset_inventory": report["official_asset_inventory"],
            "dependency_count": report["dependency_count"],
            "blocking_dependency_count": report["blocking_dependency_count"],
            "critical_path": report["critical_path"],
            "report_path": str(destination / "m4-unblock-plan.json"),
            "markdown_path": str(destination / "m4-unblock-plan.md"),
            "cas_refs": {"m4_unblock_plan_json": report_ref, "m4_unblock_plan_markdown": markdown_ref},
            "next_safe_actions": report["next_safe_actions"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M4 Unblock Dependency Plan",
        "",
        f"State: `{report['state']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        f"Formal training allowed: `{report['formal_training_allowed']}`",
        f"Phase gate blockers: `{report['phase_gate_blocker_count']}`",
        "",
        "## Critical Path",
        "",
    ]
    lines.extend(f"- `{step}`" for step in report["critical_path"])
    official_inventory = report.get("official_asset_inventory")
    if isinstance(official_inventory, Mapping):
        lines.extend(
            [
                "",
                "## Official Asset Inventory",
                "",
                f"- State: `{official_inventory.get('state')}`",
                f"- Checkpoint mismatches: `{official_inventory.get('checkpoint_mismatch_count')}`",
                f"- Remote checkpoint candidates: `{official_inventory.get('remote_checkpoint_candidate_count')}`",
                "- Current-main redownload is fix: "
                f"`{not _official_current_redownload_is_not_fix(official_inventory)}`",
                f"- Horizon-limited environments: `{official_inventory.get('horizon_limited_environment_count')}`",
            ]
        )
    lines.extend(["", "## Dependencies", "", "| Step | Status | Authority | Blocks M4 | GPU Policy |", "|:--|:--|:--|:--|:--|"])
    for dependency in report["dependencies"]:
        lines.append(
            f"| `{dependency['step_id']}` | `{dependency['status']}` | `{dependency['authority']}` | "
            f"`{dependency['blocks_m4']}` | `{dependency['gpu_policy']}` |"
        )
    source_rows: list[str] = []
    for dependency in report["dependencies"]:
        source_resolution = dependency.get("source_resolution")
        if not isinstance(source_resolution, Mapping):
            continue
        for key in ("remote_watch", "revision_watch"):
            watch = source_resolution.get(key)
            if isinstance(watch, Mapping):
                source_rows.append(
                    f"| `{dependency.get('step_id')}` | `{key}` | `{watch.get('state')}` | "
                    f"`{watch.get('candidate_available', 'n/a')}` | `{watch.get('scanned_revision_count', 'n/a')}` | "
                    f"`{watch.get('candidate_revision_count', 'n/a')}` |"
                )
    if source_rows:
        lines.extend(
            [
                "",
                "## Source Resolution",
                "",
                "| Step | Watch | State | Candidate | Scanned Revisions | Candidate Revisions |",
                "|:--|:--|:--|:--|--:|--:|",
                *source_rows,
            ]
        )
    self_training_rows: list[str] = []
    for dependency in report["dependencies"]:
        self_training = dependency.get("self_training_preflight")
        if not isinstance(self_training, Mapping):
            continue
        gpu = self_training.get("gpu_preflight")
        budget = self_training.get("budget")
        self_training_rows.append(
            f"| `{dependency.get('step_id')}` | `{self_training.get('state')}` | "
            f"`{self_training.get('human_approval_provided')}` | "
            f"`{self_training.get('self_training_launch_planning_ready')}` | "
            f"`{gpu.get('requested_gpus', 'n/a') if isinstance(gpu, Mapping) else 'n/a'}` | "
            f"`{budget.get('estimated_gpu_hours', 'n/a') if isinstance(budget, Mapping) else 'n/a'}` |"
        )
    if self_training_rows:
        lines.extend(
            [
                "",
                "## Checkpoint Self-Training Preflight",
                "",
                "| Step | State | Human Approved | Launch Planning Ready | Requested GPUs | Estimated GPU Hours |",
                "|:--|:--|:--|:--|:--|--:|",
                *self_training_rows,
            ]
        )
    branch_rows: list[str] = []
    for dependency in report["dependencies"]:
        branches = dependency.get("resolution_branches")
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, Mapping):
                branch_rows.append(
                    f"| `{branch.get('branch_id')}` | `{branch.get('status')}` | `{branch.get('authority')}` | "
                    f"`{branch.get('fresh_gpu_exclusivity_audit_required')}` | `{branch.get('m4_launch_allowed_after_branch')}` |"
                )
    if branch_rows:
        lines.extend(
            [
                "",
                "## Resolution Branches",
                "",
                "| Branch | Status | Authority | Needs GPU Audit | Opens M4 Directly |",
                "|:--|:--|:--|:--|:--|",
                *branch_rows,
            ]
        )
    lines.extend(["", "## Next Safe Actions", ""])
    lines.extend(f"- {item}" for item in report["next_safe_actions"])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M4UnblockPlanError("M4_UNBLOCK_PLAN_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate a read-only M4 unblock dependency plan")
    run.add_argument("--phase-gate-manifest", type=Path, required=True)
    run.add_argument("--blocker-decision-matrix-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-recovery-packet-manifest", type=Path)
    run.add_argument("--protocol-decision-packet-manifest", type=Path)
    run.add_argument("--attribution-remediation-manifest", type=Path)
    run.add_argument("--protocol-application-preview-manifest", type=Path)
    run.add_argument("--official-asset-inventory-manifest", type=Path)
    run.add_argument("--checkpoint-self-training-preflight-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_m4_unblock_plan(
            phase_gate_manifest=args.phase_gate_manifest,
            blocker_decision_matrix_manifest=args.blocker_decision_matrix_manifest,
            checkpoint_recovery_packet_manifest=args.checkpoint_recovery_packet_manifest,
            protocol_decision_packet_manifest=args.protocol_decision_packet_manifest,
            attribution_remediation_manifest=args.attribution_remediation_manifest,
            protocol_application_preview_manifest=args.protocol_application_preview_manifest,
            official_asset_inventory_manifest=args.official_asset_inventory_manifest,
            checkpoint_self_training_preflight_manifest=args.checkpoint_self_training_preflight_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
