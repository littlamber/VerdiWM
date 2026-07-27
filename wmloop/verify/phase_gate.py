"""Strict phase gate for starting M4-scale campaign work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.vendor import verify_vendor_checkout


class PhaseGateError(RuntimeError):
    """A phase-gate audit failed closed."""


def run_phase_gate(
    *,
    repo_root: Path,
    archive_db: Path,
    baseline_reproduction_manifest: Path,
    raw_failure_batch_manifest: Path,
    attribution_review_manifest: Path,
    m3_acceptance_manifest: Path,
    output_root: Path,
    data_extension_audit_manifest: Path | None = None,
    attribution_signoff_manifest: Path | None = None,
    attribution_remediation_manifest: Path | None = None,
    constitutional_audit_manifest: Path | None = None,
    checkpoint_source_manifest: Path | None = None,
    checkpoint_candidate_inventory_manifest: Path | None = None,
    checkpoint_recovery_packet_manifest: Path | None = None,
    checkpoint_launch_guard_manifest: Path | None = None,
    checkpoint_claim_downgrade_manifest: Path | None = None,
    protocol_application_receipt_manifest: Path | None = None,
    cas_root: Path | None = None,
    required_baselines: int = 8,
    required_m1_reports: int = 8,
    required_attribution_reviews: int = 3,
    expected_primitive_count: int = 13,
    m4_settled_trial_target: int = 150,
) -> dict[str, object]:
    """Write a machine-readable M4 launch gate without launching training."""

    if (
        required_baselines < 1
        or required_m1_reports < 1
        or required_attribution_reviews < 1
        or expected_primitive_count < 1
        or m4_settled_trial_target < 1
    ):
        raise PhaseGateError("PHASE_GATE_REQUIREMENTS_INVALID")
    repo = Path(repo_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PhaseGateError("PHASE_GATE_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db)
    cas_storage_root = cas_root if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(Path(cas_storage_root))
    sources = {
        "baseline_reproduction": _load_source(baseline_reproduction_manifest, cas=cas, archive=archive),
        "raw_failure_batch": _load_source(raw_failure_batch_manifest, cas=cas, archive=archive),
        "attribution_review": _load_source(attribution_review_manifest, cas=cas, archive=archive),
        "m3_acceptance": _load_source(m3_acceptance_manifest, cas=cas, archive=archive),
    }
    if data_extension_audit_manifest is not None:
        sources["data_extension_audit"] = _load_source(data_extension_audit_manifest, cas=cas, archive=archive)
    if attribution_signoff_manifest is not None:
        sources["attribution_signoff"] = _load_source(attribution_signoff_manifest, cas=cas, archive=archive)
    if attribution_remediation_manifest is not None:
        sources["attribution_remediation"] = _load_source(attribution_remediation_manifest, cas=cas, archive=archive)
    if constitutional_audit_manifest is not None:
        sources["constitutional_audit"] = _load_source(constitutional_audit_manifest, cas=cas, archive=archive)
    if checkpoint_source_manifest is not None:
        sources["checkpoint_source"] = _load_source(checkpoint_source_manifest, cas=cas, archive=archive)
    if checkpoint_candidate_inventory_manifest is not None:
        sources["checkpoint_candidate_inventory"] = _load_source(
            checkpoint_candidate_inventory_manifest,
            cas=cas,
            archive=archive,
        )
    if checkpoint_recovery_packet_manifest is not None:
        sources["checkpoint_recovery_packet"] = _load_source(
            checkpoint_recovery_packet_manifest,
            cas=cas,
            archive=archive,
        )
    if checkpoint_launch_guard_manifest is not None:
        sources["checkpoint_launch_guard"] = _load_source(checkpoint_launch_guard_manifest, cas=cas, archive=archive)
    if checkpoint_claim_downgrade_manifest is not None:
        sources["checkpoint_claim_downgrade"] = _load_source(
            checkpoint_claim_downgrade_manifest,
            cas=cas,
            archive=archive,
        )
    if protocol_application_receipt_manifest is not None:
        sources["protocol_application_receipt"] = _load_source(
            protocol_application_receipt_manifest,
            cas=cas,
            archive=archive,
        )

    archive_statistics = archive.archive_statistics()
    infrastructure = {
        "vendor_freeze": _vendor_gate(repo),
        "primitive_registry": _registry_gate(repo, expected_primitive_count=expected_primitive_count),
    }
    requirements = _requirements(
        sources=sources,
        archive_statistics=archive_statistics,
        infrastructure=infrastructure,
        required_baselines=required_baselines,
        required_m1_reports=required_m1_reports,
        required_attribution_reviews=required_attribution_reviews,
    )
    m4_launch_allowed = all(item["passed"] is True for item in requirements.values())
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-strict-phase-gate",
        "state": "ready" if m4_launch_allowed else "blocked",
        "phase": "M4_launch",
        "m4_launch_allowed": m4_launch_allowed,
        "requirements": requirements,
        "blockers": _blockers(requirements),
        "archive_statistics": archive_statistics,
        "m4_settled_trial_target": m4_settled_trial_target,
        "m4_settled_trial_progress": {
            "settled_trials": archive_statistics.get("settled_trials", 0),
            "target": m4_settled_trial_target,
            "remaining": max(0, m4_settled_trial_target - archive_statistics.get("settled_trials", 0)),
        },
        "infrastructure": infrastructure,
        "sources": {name: source["summary"] for name, source in sources.items()},
        "checkpoint_claim_policy": _checkpoint_claim_policy_summary(sources=sources, requirements=requirements),
        "next_actions": _next_actions(requirements=requirements, sources=sources),
        "limitations": [
            "This gate is read-only and does not launch training, evaluation, or protocol changes.",
            "M4 launch remains blocked until every strict prerequisite passes under the active protocol.",
            *_checkpoint_claim_limitations(sources=sources, requirements=requirements),
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _requirements(
    *,
    sources: Mapping[str, Mapping[str, object]],
    archive_statistics: Mapping[str, int],
    infrastructure: Mapping[str, Mapping[str, object]],
    required_baselines: int,
    required_m1_reports: int,
    required_attribution_reviews: int,
) -> dict[str, dict[str, object]]:
    baseline = _payload(sources, "baseline_reproduction")
    raw = _payload(sources, "raw_failure_batch")
    attribution = _payload(sources, "attribution_review")
    m3 = _payload(sources, "m3_acceptance")
    data_extension = _payload_or_none(sources, "data_extension_audit")
    attribution_signoff = _payload_or_none(sources, "attribution_signoff")
    attribution_remediation = _payload_or_none(sources, "attribution_remediation")
    constitutional_audit = _payload_or_none(sources, "constitutional_audit")
    protocol_application = _payload_or_none(sources, "protocol_application_receipt")
    checkpoint_source = _payload_or_none(sources, "checkpoint_source")
    checkpoint_launch_guard = _payload_or_none(sources, "checkpoint_launch_guard")
    checkpoint_claim_downgrade = _payload_or_none(sources, "checkpoint_claim_downgrade")
    checkpoint_claim_clear = _checkpoint_claim_downgrade_passes(
        checkpoint_claim_downgrade=checkpoint_claim_downgrade,
        baseline=baseline,
        checkpoint_source=checkpoint_source,
        checkpoint_launch_guard=checkpoint_launch_guard,
        sources=sources,
    )
    baseline_reproduction_passed = (
        baseline.get("strict_m0_t03_pass") is True and baseline.get("state") == "ready"
    ) or _checkpoint_claim_downgrade_resolves_baseline(
        checkpoint_claim_downgrade=checkpoint_claim_downgrade,
        baseline=baseline,
        checkpoint_claim_clear=checkpoint_claim_clear,
    )
    requirements = {
        "M0_generation_zero_archive": {
            "passed": archive_statistics.get("baselines", 0) >= required_baselines,
            "expected": f"At least {required_baselines} generation-zero baselines are present in the authoritative archive.",
            "observed": {"baselines": archive_statistics.get("baselines", 0)},
        },
        "M0_baseline_reproduction": {
            "passed": baseline_reproduction_passed,
            "expected": (
                "Official-code baseline reproduction includes evaluable PSNR +/-0.5dB comparison, "
                "or a human-approved checkpoint claim downgrade accepts the official-current warning for paired-delta M4 trials."
            ),
            "observed": {
                "state": baseline.get("state"),
                "strict_m0_t03_pass": baseline.get("strict_m0_t03_pass"),
                "warnings": baseline.get("warnings", []),
                "checkpoint_claim_downgrade": _checkpoint_claim_downgrade_observed(
                    checkpoint_claim_downgrade=checkpoint_claim_downgrade,
                    sources=sources,
                    checkpoint_claim_clear=checkpoint_claim_clear,
                ),
            },
        },
        "M1_raw_failure_reports": {
            "passed": raw.get("state") == "ready" and _int(raw.get("report_count")) >= required_m1_reports,
            "expected": f"{required_m1_reports} real raw failure_report records pass schema under the active protocol.",
            "observed": {
                "state": raw.get("state"),
                "report_count": raw.get("report_count"),
                "blocked_count": raw.get("blocked_count"),
                "blocked_records": raw.get("blocked_records", []),
            },
        },
        "M1_attribution_review": {
            "passed": _attribution_review_passes(
                attribution=attribution,
                attribution_signoff=attribution_signoff,
                attribution_review_summary=sources["attribution_review"]["summary"],
                required_attribution_reviews=required_attribution_reviews,
            ),
            "expected": f"At least {required_attribution_reviews} environments are human-reviewed with no attribution mismatch and either the review manifest is signed or a matching signoff receipt is provided.",
            "observed": _attribution_review_observed(
                attribution=attribution,
                attribution_signoff=attribution_signoff,
                attribution_remediation=attribution_remediation,
                attribution_review_summary=sources["attribution_review"]["summary"],
            ),
        },
        "M1_original_g1_data_feasibility": {
            "passed": _data_or_protocol_application_resolves_path(
                data_extension=data_extension,
                protocol_application=protocol_application,
                raw_failure_batch=raw,
            ),
            "expected": "Either active G1 data supports the frozen 16/32/48/64 protocol, or a human-approved protocol application receipt resolves it as a new version boundary.",
            "observed": _data_protocol_observed(
                data_extension=data_extension,
                protocol_application=protocol_application,
                raw_failure_batch=raw,
            ),
        },
        "M2_vendor_and_registry_freeze": {
            "passed": infrastructure["vendor_freeze"].get("passed") is True
            and infrastructure["primitive_registry"].get("passed") is True,
            "expected": "ACWM vendor checkout is frozen/clean and the v1.0 registry exposes exactly the expected primitive set.",
            "observed": infrastructure,
        },
        "M2_constitutional_layer_freeze": {
            "passed": _constitutional_audit_passes(constitutional_audit),
            "expected": "The §6.6 constitutional layer has a ready audit covering all five frozen components and verdict/diagnostic probe separation.",
            "observed": _constitutional_audit_observed(constitutional_audit),
        },
        "M3_strict_acceptance": {
            "passed": m3.get("strict_m3_pass") is True and m3.get("state") == "ready",
            "expected": "T3.1/T3.2/T3.3 strict M3 acceptance all pass.",
            "observed": _m3_acceptance_observed(m3),
        },
    }
    checkpoint_source_clear = (
        checkpoint_source is not None
        and checkpoint_source.get("state") == "ready"
        and _int(checkpoint_source.get("mismatch_count")) == 0
    )
    checkpoint_source_or_claim_clear = checkpoint_source_clear or checkpoint_claim_clear
    requirements["M0_checkpoint_source_resolution"] = {
        "passed": checkpoint_source_or_claim_clear,
        "expected": (
            "No known checkpoint source mismatch remains, a verified replacement has been installed and re-audited, "
            "or a human-approved claim downgrade accepts an official-current warning without allowing 100k reproduction claims."
        ),
        "observed": _checkpoint_source_observed(
            checkpoint_source,
            _payload_or_none(sources, "checkpoint_candidate_inventory"),
            _payload_or_none(sources, "checkpoint_recovery_packet"),
            checkpoint_claim_downgrade=checkpoint_claim_downgrade,
            checkpoint_claim_clear=checkpoint_claim_clear,
        ),
    }
    requirements["M0_checkpoint_launch_guard_clear"] = {
        "passed": checkpoint_source_or_claim_clear,
        "expected": (
            "Strict M0 launch is not blocked by an unaccepted checkpoint-step mismatch guard; "
            "accepted official-current warnings must remain claim-scoped."
        ),
        "observed": _checkpoint_launch_guard_observed(
            checkpoint_launch_guard,
            checkpoint_source_clear=checkpoint_source_or_claim_clear,
            checkpoint_claim_downgrade=checkpoint_claim_downgrade,
            checkpoint_claim_clear=checkpoint_claim_clear,
        ),
    }
    return requirements


def _data_or_protocol_application_resolves_path(
    *,
    data_extension: Mapping[str, Any] | None,
    protocol_application: Mapping[str, Any] | None,
    raw_failure_batch: Mapping[str, Any],
) -> bool:
    return _data_extension_allows_original_g1(data_extension) or _protocol_application_resolves_path(
        protocol_application=protocol_application,
        raw_failure_batch=raw_failure_batch,
    )


def _constitutional_audit_passes(constitutional_audit: Mapping[str, Any] | None) -> bool:
    return (
        constitutional_audit is not None
        and constitutional_audit.get("artifact_type") == "wmloop-constitutional-audit-manifest"
        and constitutional_audit.get("state") == "ready"
        and _int(constitutional_audit.get("ready_surface_count")) == 5
        and _int(constitutional_audit.get("surface_count")) == 5
        and _int(constitutional_audit.get("blocker_count")) == 0
        and constitutional_audit.get("active_constitution_mutated") is False
        and constitutional_audit.get("m4_launch_allowed") is False
        and constitutional_audit.get("formal_training_allowed") is False
    )


def _constitutional_audit_observed(constitutional_audit: Mapping[str, Any] | None) -> dict[str, object]:
    if constitutional_audit is None:
        return {"state": "not_provided"}
    return {
        "state": constitutional_audit.get("state"),
        "entry_count": constitutional_audit.get("entry_count"),
        "ready_entry_count": constitutional_audit.get("ready_entry_count"),
        "surface_count": constitutional_audit.get("surface_count"),
        "ready_surface_count": constitutional_audit.get("ready_surface_count"),
        "blocker_count": constitutional_audit.get("blocker_count"),
        "verdict_probe_ids": constitutional_audit.get("verdict_probe_ids", []),
        "diagnostic_probe_ids": constitutional_audit.get("diagnostic_probe_ids", []),
        "active_constitution_mutated": constitutional_audit.get("active_constitution_mutated"),
        "m4_launch_allowed": constitutional_audit.get("m4_launch_allowed"),
        "formal_training_allowed": constitutional_audit.get("formal_training_allowed"),
    }


def _attribution_review_passes(
    *,
    attribution: Mapping[str, Any],
    attribution_signoff: Mapping[str, Any] | None,
    attribution_review_summary: Mapping[str, Any],
    required_attribution_reviews: int,
) -> bool:
    manifest_signed = (
        attribution.get("manual_pass") is True
        and attribution.get("human_signed_off") is True
        and _int(attribution.get("reviewed_count")) >= required_attribution_reviews
        and _int(attribution.get("expectation_mismatch_count")) == 0
    )
    return manifest_signed or _attribution_signoff_passes(
        attribution=attribution,
        attribution_signoff=attribution_signoff,
        attribution_review_summary=attribution_review_summary,
        required_attribution_reviews=required_attribution_reviews,
    )


def _attribution_signoff_passes(
    *,
    attribution: Mapping[str, Any],
    attribution_signoff: Mapping[str, Any] | None,
    attribution_review_summary: Mapping[str, Any],
    required_attribution_reviews: int,
) -> bool:
    if attribution_signoff is None:
        return False
    source = attribution_signoff.get("source_attribution_review")
    if not isinstance(source, Mapping):
        return False
    source_hash = source.get("manifest_sha256")
    current_hash = attribution_review_summary.get("sha256")
    return (
        attribution_signoff.get("artifact_type") == "wmloop-m1-attribution-signoff-receipt-manifest"
        and attribution_signoff.get("state") == "signed"
        and attribution_signoff.get("manual_pass") is True
        and attribution_signoff.get("human_signed_off") is True
        and attribution.get("state") == "ready_for_human_signoff"
        and attribution.get("manual_pass") is False
        and attribution.get("human_signed_off") is False
        and _int(attribution.get("reviewed_count")) >= required_attribution_reviews
        and _int(attribution.get("expectation_mismatch_count")) == 0
        and _int(attribution_signoff.get("reviewed_count")) == _int(attribution.get("reviewed_count"))
        and _int(attribution_signoff.get("required_review_count")) == _int(attribution.get("required_review_count"))
        and _int(attribution_signoff.get("expectation_mismatch_count")) == _int(attribution.get("expectation_mismatch_count"))
        and source.get("state") == attribution.get("state")
        and source.get("reviewed_count") == attribution.get("reviewed_count")
        and source.get("required_review_count") == attribution.get("required_review_count")
        and source.get("expectation_mismatch_count") == attribution.get("expectation_mismatch_count")
        and isinstance(source_hash, str)
        and source_hash == current_hash
    )


def _attribution_review_observed(
    *,
    attribution: Mapping[str, Any],
    attribution_signoff: Mapping[str, Any] | None,
    attribution_remediation: Mapping[str, Any] | None,
    attribution_review_summary: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "state": attribution.get("state"),
        "manual_pass": attribution.get("manual_pass"),
        "human_signed_off": attribution.get("human_signed_off"),
        "reviewed_count": attribution.get("reviewed_count"),
        "expectation_mismatch_count": attribution.get("expectation_mismatch_count"),
        "signoff": _attribution_signoff_observed(
            attribution_signoff=attribution_signoff,
            attribution_review_summary=attribution_review_summary,
        ),
        "remediation": _attribution_remediation_observed(
            attribution_remediation=attribution_remediation,
            attribution_review_summary=attribution_review_summary,
        ),
    }


def _attribution_signoff_observed(
    *,
    attribution_signoff: Mapping[str, Any] | None,
    attribution_review_summary: Mapping[str, Any],
) -> dict[str, object]:
    if attribution_signoff is None:
        return {"state": "not_provided"}
    source = attribution_signoff.get("source_attribution_review")
    source_hash = source.get("manifest_sha256") if isinstance(source, Mapping) else None
    current_hash = attribution_review_summary.get("sha256")
    return {
        "state": attribution_signoff.get("state"),
        "manual_pass": attribution_signoff.get("manual_pass"),
        "human_signed_off": attribution_signoff.get("human_signed_off"),
        "reviewer": attribution_signoff.get("reviewer"),
        "reviewed_count": attribution_signoff.get("reviewed_count"),
        "required_review_count": attribution_signoff.get("required_review_count"),
        "expectation_mismatch_count": attribution_signoff.get("expectation_mismatch_count"),
        "source_manifest_sha256": source_hash,
        "current_manifest_sha256": current_hash,
        "source_hash_matches_current": isinstance(source_hash, str) and source_hash == current_hash,
    }


def _attribution_remediation_observed(
    *,
    attribution_remediation: Mapping[str, Any] | None,
    attribution_review_summary: Mapping[str, Any],
) -> dict[str, object]:
    if attribution_remediation is None:
        return {"state": "not_provided"}
    source = attribution_remediation.get("source_attribution_review")
    source_hash = source.get("manifest_sha256") if isinstance(source, Mapping) else None
    current_hash = attribution_review_summary.get("sha256")
    return {
        "state": attribution_remediation.get("state"),
        "signoff_allowed": attribution_remediation.get("signoff_allowed"),
        "human_resolution_required": attribution_remediation.get("human_resolution_required"),
        "m4_launch_allowed": attribution_remediation.get("m4_launch_allowed"),
        "formal_training_allowed": attribution_remediation.get("formal_training_allowed"),
        "blocker_count": attribution_remediation.get("blocker_count"),
        "action_items": attribution_remediation.get("action_items", []),
        "source_manifest_sha256": source_hash,
        "current_manifest_sha256": current_hash,
        "source_hash_matches_current": isinstance(source_hash, str) and source_hash == current_hash,
    }


def _data_extension_allows_original_g1(data_extension: Mapping[str, Any] | None) -> bool:
    if data_extension is None:
        return False
    if data_extension.get("active_protocol_changed") is True:
        return False
    return data_extension.get("preserve_original_g1_feasible_with_current_data") is True


def _protocol_application_resolves_path(
    *,
    protocol_application: Mapping[str, Any] | None,
    raw_failure_batch: Mapping[str, Any],
) -> bool:
    if protocol_application is None:
        return False
    target = protocol_application.get("target_goal_config")
    if not isinstance(target, Mapping):
        return False
    target_goal_id = target.get("goal_id")
    return (
        protocol_application.get("artifact_type") == "wmloop-protocol-application-receipt-manifest"
        and protocol_application.get("state") == "applied"
        and protocol_application.get("active_protocol_changed") is True
        and protocol_application.get("human_approval_confirmed") is True
        and protocol_application.get("active_goal_config_mutated") is False
        and protocol_application.get("candidate_goal_paths_mutated") is False
        and protocol_application.get("m4_launch_allowed") is False
        and target.get("created") is True
        and isinstance(target_goal_id, str)
        and target_goal_id
        and raw_failure_batch.get("goal_id") == target_goal_id
    )


def _data_protocol_observed(
    *,
    data_extension: Mapping[str, Any] | None,
    protocol_application: Mapping[str, Any] | None,
    raw_failure_batch: Mapping[str, Any],
) -> dict[str, object]:
    observed = _data_extension_observed(data_extension)
    observed["protocol_application"] = _protocol_application_observed(
        protocol_application=protocol_application,
        raw_failure_batch=raw_failure_batch,
    )
    return observed


def _data_extension_observed(data_extension: Mapping[str, Any] | None) -> dict[str, object]:
    if data_extension is None:
        return {"state": "not_provided"}
    return {
        "state": data_extension.get("state"),
        "active_protocol_changed": data_extension.get("active_protocol_changed"),
        "human_approval_required": data_extension.get("human_approval_required"),
        "preserve_original_g1_feasible_with_current_data": data_extension.get(
            "preserve_original_g1_feasible_with_current_data"
        ),
        "fill_opportunity_count": data_extension.get("fill_opportunity_count"),
        "unfilled_blocked_environments": data_extension.get("unfilled_blocked_environments", []),
    }


def _protocol_application_observed(
    *,
    protocol_application: Mapping[str, Any] | None,
    raw_failure_batch: Mapping[str, Any],
) -> dict[str, object]:
    if protocol_application is None:
        return {"state": "not_provided"}
    target = protocol_application.get("target_goal_config")
    target_goal_id = target.get("goal_id") if isinstance(target, Mapping) else None
    raw_goal_id = raw_failure_batch.get("goal_id")
    return {
        "state": protocol_application.get("state"),
        "active_protocol_changed": protocol_application.get("active_protocol_changed"),
        "human_approval_confirmed": protocol_application.get("human_approval_confirmed"),
        "active_goal_config_mutated": protocol_application.get("active_goal_config_mutated"),
        "candidate_goal_paths_mutated": protocol_application.get("candidate_goal_paths_mutated"),
        "selected_option": protocol_application.get("selected_option"),
        "m4_launch_allowed": protocol_application.get("m4_launch_allowed"),
        "target_goal_id": target_goal_id,
        "target_created": target.get("created") if isinstance(target, Mapping) else None,
        "raw_failure_batch_goal_id": raw_goal_id,
        "goal_id_matches_raw_failure_batch": isinstance(target_goal_id, str) and raw_goal_id == target_goal_id,
    }


def _checkpoint_claim_downgrade_passes(
    *,
    checkpoint_claim_downgrade: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    checkpoint_source: Mapping[str, Any] | None,
    checkpoint_launch_guard: Mapping[str, Any] | None,
    sources: Mapping[str, Mapping[str, object]],
) -> bool:
    if checkpoint_claim_downgrade is None or checkpoint_source is None:
        return False
    policy = checkpoint_claim_downgrade.get("checkpoint_policy")
    boundary = checkpoint_claim_downgrade.get("claim_boundary")
    warned_envs = checkpoint_claim_downgrade.get("warned_envs")
    if (
        checkpoint_claim_downgrade.get("artifact_type") != "wmloop-m4-checkpoint-claim-downgrade-manifest"
        or checkpoint_claim_downgrade.get("state") != "ready"
        or checkpoint_claim_downgrade.get("phase") != "M4_launch"
        or checkpoint_claim_downgrade.get("m4_launch_claim_downgrade_allowed") is not True
        or checkpoint_claim_downgrade.get("human_approval_confirmed") is not True
        or not isinstance(policy, Mapping)
        or policy.get("allow_official_current_checkpoint_warning") is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("paired_delta_m4_trials_allowed") is not True
        or boundary.get("official_100k_reproduction_claim_disallowed_for_warned_envs") is not True
        or boundary.get("source_checkpoint_mutation_allowed") is not False
        or boundary.get("evaluator_or_protocol_mutation_allowed") is not False
        or not isinstance(warned_envs, list)
        or not warned_envs
    ):
        return False
    if baseline.get("state") != "checkpoint_step_mismatch" or baseline.get("strict_m0_t03_pass") is not False:
        return False
    if checkpoint_source.get("state") != "remote_current_mismatch":
        return False
    if _int(checkpoint_source.get("mismatch_count")) != len([env for env in warned_envs if isinstance(env, str)]):
        return False
    if _receipt_source_sha(checkpoint_claim_downgrade, "baseline_reproduction") != _current_source_sha(
        sources,
        "baseline_reproduction",
    ):
        return False
    if _receipt_source_sha(checkpoint_claim_downgrade, "checkpoint_source") != _current_source_sha(
        sources,
        "checkpoint_source",
    ):
        return False
    if checkpoint_launch_guard is not None and _receipt_source_sha(
        checkpoint_claim_downgrade,
        "checkpoint_launch_guard",
    ) != _current_source_sha(sources, "checkpoint_launch_guard"):
        return False
    return True


def _checkpoint_claim_downgrade_resolves_baseline(
    *,
    checkpoint_claim_downgrade: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    checkpoint_claim_clear: bool,
) -> bool:
    return (
        checkpoint_claim_clear
        and checkpoint_claim_downgrade is not None
        and baseline.get("state") == "checkpoint_step_mismatch"
        and baseline.get("strict_m0_t03_pass") is False
    )


def _checkpoint_claim_downgrade_observed(
    *,
    checkpoint_claim_downgrade: Mapping[str, Any] | None,
    sources: Mapping[str, Mapping[str, object]],
    checkpoint_claim_clear: bool,
) -> dict[str, object]:
    if checkpoint_claim_downgrade is None:
        return {"state": "not_provided", "accepted": False}
    return {
        "state": checkpoint_claim_downgrade.get("state"),
        "accepted": checkpoint_claim_clear,
        "human_approval_confirmed": checkpoint_claim_downgrade.get("human_approval_confirmed"),
        "warned_envs": checkpoint_claim_downgrade.get("warned_envs", []),
        "allowed_envs": checkpoint_claim_downgrade.get("allowed_envs", []),
        "checkpoint_policy": checkpoint_claim_downgrade.get("checkpoint_policy", {}),
        "claim_boundary": checkpoint_claim_downgrade.get("claim_boundary", {}),
        "source_hashes": {
            "baseline_reproduction": {
                "receipt": _receipt_source_sha(checkpoint_claim_downgrade, "baseline_reproduction"),
                "current": _current_source_sha(sources, "baseline_reproduction"),
            },
            "checkpoint_source": {
                "receipt": _receipt_source_sha(checkpoint_claim_downgrade, "checkpoint_source"),
                "current": _current_source_sha(sources, "checkpoint_source"),
            },
            "checkpoint_launch_guard": {
                "receipt": _receipt_source_sha(checkpoint_claim_downgrade, "checkpoint_launch_guard"),
                "current": _current_source_sha(sources, "checkpoint_launch_guard"),
            },
        },
    }


def _checkpoint_claim_policy_summary(
    *,
    sources: Mapping[str, Mapping[str, object]],
    requirements: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    claim = _payload_or_none(sources, "checkpoint_claim_downgrade")
    if claim is None:
        return {
            "active": False,
            "allow_official_current_checkpoint_warning": False,
            "claim_boundary": "Every included environment must pass the strict checkpoint-step audit.",
        }
    accepted = (
        requirements.get("M0_checkpoint_source_resolution", {}).get("passed") is True
        and requirements.get("M0_checkpoint_launch_guard_clear", {}).get("passed") is True
        and requirements.get("M0_baseline_reproduction", {}).get("passed") is True
    )
    return {
        "active": accepted,
        "state": claim.get("state"),
        "warned_envs": claim.get("warned_envs", []),
        "allowed_envs": claim.get("allowed_envs", []),
        "checkpoint_policy": claim.get("checkpoint_policy", {}),
        "claim_boundary": claim.get("claim_boundary", {}),
    }


def _checkpoint_claim_limitations(
    *,
    sources: Mapping[str, Mapping[str, object]],
    requirements: Mapping[str, Mapping[str, object]],
) -> list[str]:
    summary = _checkpoint_claim_policy_summary(sources=sources, requirements=requirements)
    if summary.get("active") is not True:
        return []
    return [
        "M4 launch is open under a checkpoint claim downgrade: official-current warned checkpoints are valid only for paired-delta trials.",
        "Do not claim official 100k reproduction for warned environments unless a verified replacement checkpoint is later installed and re-audited.",
    ]


def _receipt_source_sha(claim: Mapping[str, Any], name: str) -> str | None:
    sources = claim.get("sources")
    source = sources.get(name) if isinstance(sources, Mapping) else None
    value = source.get("sha256") if isinstance(source, Mapping) else None
    return value if isinstance(value, str) and value else None


def _current_source_sha(sources: Mapping[str, Mapping[str, object]], name: str) -> str | None:
    source = sources.get(name)
    summary = source.get("summary") if isinstance(source, Mapping) else None
    value = summary.get("sha256") if isinstance(summary, Mapping) else None
    return value if isinstance(value, str) and value else None


def _checkpoint_source_observed(
    checkpoint_source: Mapping[str, Any] | None,
    checkpoint_candidate_inventory: Mapping[str, Any] | None,
    checkpoint_recovery_packet: Mapping[str, Any] | None,
    *,
    checkpoint_claim_downgrade: Mapping[str, Any] | None = None,
    checkpoint_claim_clear: bool = False,
) -> dict[str, object]:
    if checkpoint_source is None:
        observed: dict[str, object] = {"state": "not_provided"}
    else:
        observed = {
            "state": checkpoint_source.get("state"),
            "mismatch_count": checkpoint_source.get("mismatch_count"),
            "replacement_candidate_count": checkpoint_source.get("replacement_candidate_count"),
            "remote_unreachable_count": checkpoint_source.get("remote_unreachable_count"),
        }
    observed["local_candidate_inventory"] = _checkpoint_candidate_inventory_observed(checkpoint_candidate_inventory)
    observed["recovery_packet"] = _checkpoint_recovery_packet_observed(checkpoint_recovery_packet)
    observed["checkpoint_claim_downgrade"] = {
        "state": checkpoint_claim_downgrade.get("state") if checkpoint_claim_downgrade is not None else "not_provided",
        "accepted": checkpoint_claim_clear,
        "warned_envs": checkpoint_claim_downgrade.get("warned_envs", []) if checkpoint_claim_downgrade is not None else [],
    }
    return observed


def _checkpoint_candidate_inventory_observed(inventory: Mapping[str, Any] | None) -> dict[str, object]:
    if inventory is None:
        return {"state": "not_provided"}
    return {
        "state": inventory.get("state"),
        "environment": inventory.get("environment"),
        "candidate_count": inventory.get("candidate_count"),
        "rejected_count": inventory.get("rejected_count"),
        "active_checkpoint_mutated": inventory.get("active_checkpoint_mutated"),
    }


def _checkpoint_recovery_packet_observed(packet: Mapping[str, Any] | None) -> dict[str, object]:
    if packet is None:
        return {"state": "not_provided"}
    return {
        "state": packet.get("state"),
        "environment": packet.get("environment"),
        "mismatch_count": packet.get("mismatch_count"),
        "m4_launch_allowed": packet.get("m4_launch_allowed"),
        "active_checkpoint_mutated": packet.get("active_checkpoint_mutated"),
        "checkpoint_blockers_active": packet.get("checkpoint_blockers_active", []),
    }


def _checkpoint_launch_guard_observed(
    checkpoint_launch_guard: Mapping[str, Any] | None,
    *,
    checkpoint_source_clear: bool,
    checkpoint_claim_downgrade: Mapping[str, Any] | None = None,
    checkpoint_claim_clear: bool = False,
) -> dict[str, object]:
    observed: dict[str, object] = {"checkpoint_source_clear": checkpoint_source_clear}
    observed["checkpoint_claim_downgrade"] = {
        "state": checkpoint_claim_downgrade.get("state") if checkpoint_claim_downgrade is not None else "not_provided",
        "accepted": checkpoint_claim_clear,
        "warned_envs": checkpoint_claim_downgrade.get("warned_envs", []) if checkpoint_claim_downgrade is not None else [],
    }
    if checkpoint_launch_guard is None:
        observed["state"] = "not_provided"
        return observed
    observed.update(
        {
            "state": checkpoint_launch_guard.get("state"),
            "strict_launch_guard_pass": checkpoint_launch_guard.get("strict_launch_guard_pass"),
            "observed_error": checkpoint_launch_guard.get("observed_error"),
            "materialized_launch_plan": checkpoint_launch_guard.get("materialized_launch_plan"),
            "gpu_execution_started": checkpoint_launch_guard.get("gpu_execution_started"),
        }
    )
    return observed


def _m3_acceptance_observed(m3: Mapping[str, Any]) -> dict[str, object]:
    observed: dict[str, object] = {
        "state": m3.get("state"),
        "strict_m3_pass": m3.get("strict_m3_pass"),
        "blockers": m3.get("blockers", []),
    }
    requirements = m3.get("requirements")
    if isinstance(requirements, Mapping):
        observed["requirements"] = {
            str(name): _m3_requirement_summary(requirement)
            for name, requirement in requirements.items()
            if isinstance(requirement, Mapping)
        }
    return observed


def _m3_requirement_summary(requirement: Mapping[str, Any]) -> dict[str, object]:
    return {
        "passed": requirement.get("passed"),
        "observed": requirement.get("observed", {}),
    }


def _vendor_gate(repo: Path) -> dict[str, object]:
    try:
        revision = verify_vendor_checkout(repo)
    except Exception as exc:  # pragma: no cover - exercised through tests with patching
        return {"passed": False, "error": type(exc).__name__, "message": str(exc)}
    return {"passed": True, "source_revision": revision}


def _registry_gate(repo: Path, *, expected_primitive_count: int) -> dict[str, object]:
    try:
        registry = PrimitiveRegistry.from_root(repo)
        names = registry.names()
        digest = registry.digest()
    except Exception as exc:  # pragma: no cover - fail-closed safety path
        return {"passed": False, "error": type(exc).__name__, "message": str(exc)}
    return {
        "passed": len(names) == expected_primitive_count,
        "primitive_count": len(names),
        "expected_primitive_count": expected_primitive_count,
        "registry_digest": digest,
        "primitive_names": list(names),
    }


def _blockers(requirements: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {"requirement": name, "expected": item["expected"], "observed": item["observed"]}
        for name, item in requirements.items()
        if item.get("passed") is not True
    ]


def _next_actions(*, requirements: Mapping[str, Mapping[str, object]], sources: Mapping[str, Mapping[str, object]]) -> list[str]:
    actions: list[str] = []
    if requirements["M0_baseline_reproduction"]["passed"] is not True:
        observed = requirements["M0_baseline_reproduction"]["observed"]
        state = observed.get("state") if isinstance(observed, Mapping) else None
        if state == "reference_unavailable":
            actions.append("Provide an official per-environment PSNR reference and regenerate M0 baseline reproduction, or keep T0.3 marked non-strict.")
        elif state == "delta_out_of_tolerance":
            actions.append("Investigate the official PSNR deltas outside +/-0.5dB, rerun/fix M0 if the gap is procedural, or keep T0.3 marked non-strict.")
        elif state == "reference_incomplete":
            actions.append("Complete the official PSNR reference coverage and regenerate M0 baseline reproduction, or keep T0.3 marked non-strict.")
        elif state == "checkpoint_step_mismatch":
            actions.append("Replace mismatched baseline checkpoints with corrected 100k-step files from a verified source, rerun M0 baseline evaluation/reproduction, or keep T0.3 marked non-strict.")
        else:
            actions.append("Resolve the M0 baseline reproduction blocker before launching M4.")
    if requirements.get("M0_checkpoint_source_resolution", {}).get("passed") is not True and "M0_checkpoint_source_resolution" in requirements:
        checkpoint_source = _payload_or_none(sources, "checkpoint_source")
        checkpoint_source_state = checkpoint_source.get("state") if isinstance(checkpoint_source, Mapping) else None
        candidate_inventory = _payload_or_none(sources, "checkpoint_candidate_inventory")
        candidate_inventory_state = candidate_inventory.get("state") if isinstance(candidate_inventory, Mapping) else None
        recovery_packet = _payload_or_none(sources, "checkpoint_recovery_packet")
        recovery_packet_state = recovery_packet.get("state") if isinstance(recovery_packet, Mapping) else None
        if checkpoint_source_state == "remote_current_mismatch":
            actions.append(
                "Do not redownload the current HF main cloth_move checkpoint as a fix; current remote metadata matches the locally mismatched file. Find an alternate verified 100k-step source or publisher-fixed revision, then validate it in quarantine before install."
            )
        elif checkpoint_source_state == "replacement_candidate_found":
            actions.append(
                "Download the differing remote cloth_move checkpoint into quarantine, validate expected step 100000, install only after explicit confirmation, and rerun M0 audits."
            )
        else:
            actions.append(
                "Resolve checkpoint source by placing a verified 100k-step cloth_move candidate in quarantine, validating it, installing only after explicit confirmation, and rerunning M0 audits."
            )
        if candidate_inventory_state == "no_candidates_found":
            actions.append(
                "Local checkpoint candidate inventory found no usable cloth_move replacement candidate; place a verified alternate 100k-step file under quarantine and rerun the inventory."
            )
        elif candidate_inventory_state == "candidate_found":
            actions.append("Validate every local checkpoint candidate with checkpoint_quarantine before considering install.")
        if recovery_packet_state == "awaiting_verified_external_candidate":
            actions.append("Checkpoint recovery packet is awaiting a verified external cloth_move 100k candidate; M4 remains blocked until quarantine validation and re-audit.")
    if requirements.get("M0_checkpoint_launch_guard_clear", {}).get("passed") is not True and "M0_checkpoint_launch_guard_clear" in requirements:
        actions.append("Strict M0 launch guard currently blocks on checkpoint-step mismatch; do not launch baseline or M4 evaluator jobs until re-audited.")
    if requirements["M1_raw_failure_reports"]["passed"] is not True:
        actions.append("Do not launch M4: M1 still lacks 8 ready raw failure_report records under the active protocol.")
    if requirements["M2_constitutional_layer_freeze"]["passed"] is not True:
        actions.append("Run the §6.6 constitutional audit and keep M4 blocked unless all five frozen components and probe-role separation are ready.")
    if requirements["M1_attribution_review"]["passed"] is not True:
        observed = requirements["M1_attribution_review"]["observed"]
        signoff = observed.get("signoff") if isinstance(observed, Mapping) else None
        remediation = observed.get("remediation") if isinstance(observed, Mapping) else None
        if isinstance(signoff, Mapping) and signoff.get("state") == "signed":
            actions.append("Regenerate attribution signoff from the current ready review manifest; the supplied signoff does not match current evidence.")
        elif isinstance(remediation, Mapping) and remediation.get("state") != "not_provided":
            if remediation.get("source_hash_matches_current") is not True:
                actions.append("Regenerate the attribution remediation packet from the current attribution review before using it for a human decision.")
            elif remediation.get("signoff_allowed") is False:
                actions.append("Use the current attribution remediation packet to resolve the attribution mismatch, third-reviewable-environment gap, and protocol-limited environments before signoff.")
        else:
            actions.append("Resolve attribution review by adding a third reviewable environment under the chosen protocol and obtaining human sign-off.")
    if requirements["M1_original_g1_data_feasibility"]["passed"] is not True:
        protocol_application = _payload_or_none(sources, "protocol_application_receipt")
        if isinstance(protocol_application, Mapping) and protocol_application.get("state") == "applied":
            actions.append("Rerun raw failure reports under the promoted goal config before spending GPU time on M4-scale trials.")
        else:
            actions.append("Choose a human protocol/data path before spending GPU time on M4-scale trials.")
    if requirements["M3_strict_acceptance"]["passed"] is not True:
        actions.append("Rerun M3 proposal readiness and acceptance audit only after M1 provides enough real failure_report inputs.")
    if not actions:
        actions.append("M4 launch gate is open; start formal multi-environment settled trials and monitor progress to 150 settled trials.")
    if "data_extension_audit" in sources:
        observed = requirements["M1_original_g1_data_feasibility"]["observed"]
        unfilled = observed.get("unfilled_blocked_environments") if isinstance(observed, Mapping) else None
        if unfilled:
            actions.append(f"Current data-extension audit still has unfilled blocked environments: {','.join(str(item) for item in unfilled)}.")
    return actions


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseGateError(f"PHASE_GATE_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise PhaseGateError(f"PHASE_GATE_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "cas_ref": ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    try:
        payload = sources[name]["payload"]
    except KeyError as exc:
        raise PhaseGateError(f"PHASE_GATE_REQUIRED_SOURCE_MISSING:{name}") from exc
    if not isinstance(payload, Mapping):
        raise PhaseGateError(f"PHASE_GATE_SOURCE_INVALID:{name}")
    return payload


def _payload_or_none(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _payload(sources, name)


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "phase-gate.json", report_bytes)
        _write_bytes_atomic(temporary / "phase-gate.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-strict-phase-gate-manifest",
            "state": report["state"],
            "phase": report["phase"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "blockers": report["blockers"],
            "report_path": str(destination / "phase-gate.json"),
            "markdown_path": str(destination / "phase-gate.md"),
            "cas_refs": {"phase_gate_json": report_ref, "phase_gate_markdown": markdown_ref},
            "next_actions": report["next_actions"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict Phase Gate",
        "",
        f"Phase: `{report['phase']}`",
        f"State: `{report['state']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Requirements",
        "",
        "| Requirement | Pass | Expected | Observed |",
        "|:--|:--|:--|:--|",
    ]
    for name, item in report["requirements"].items():
        lines.append(
            f"| {name} | {item['passed']} | {item['expected']} | `{_short_json(item['observed'])}` |"
        )
    lines.extend(
        [
            "",
            "## M4 Progress",
            "",
            f"Settled trials: `{report['m4_settled_trial_progress']['settled_trials']}`",
            f"Target: `{report['m4_settled_trial_progress']['target']}`",
            f"Remaining: `{report['m4_settled_trial_progress']['remaining']}`",
            "",
            "## Next Actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- `{item['requirement']}`" for item in report["blockers"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _short_json(value: object, *, limit: int = 420) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PhaseGateError("PHASE_GATE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit whether M4 launch is allowed")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--baseline-reproduction-manifest", type=Path, required=True)
    run.add_argument("--raw-failure-batch-manifest", type=Path, required=True)
    run.add_argument("--attribution-review-manifest", type=Path, required=True)
    run.add_argument("--m3-acceptance-manifest", type=Path, required=True)
    run.add_argument("--data-extension-audit-manifest", type=Path)
    run.add_argument("--attribution-signoff-manifest", type=Path)
    run.add_argument("--attribution-remediation-manifest", type=Path)
    run.add_argument("--constitutional-audit-manifest", type=Path)
    run.add_argument("--checkpoint-source-manifest", type=Path)
    run.add_argument("--checkpoint-candidate-inventory-manifest", type=Path)
    run.add_argument("--checkpoint-recovery-packet-manifest", type=Path)
    run.add_argument("--checkpoint-launch-guard-manifest", type=Path)
    run.add_argument("--checkpoint-claim-downgrade-manifest", type=Path)
    run.add_argument("--protocol-application-receipt-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--required-baselines", type=int, default=8)
    run.add_argument("--required-m1-reports", type=int, default=8)
    run.add_argument("--required-attribution-reviews", type=int, default=3)
    run.add_argument("--expected-primitive-count", type=int, default=13)
    run.add_argument("--m4-settled-trial-target", type=int, default=150)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_phase_gate(
            repo_root=args.repo_root,
            archive_db=args.archive_db,
            baseline_reproduction_manifest=args.baseline_reproduction_manifest,
            raw_failure_batch_manifest=args.raw_failure_batch_manifest,
            attribution_review_manifest=args.attribution_review_manifest,
            m3_acceptance_manifest=args.m3_acceptance_manifest,
            data_extension_audit_manifest=args.data_extension_audit_manifest,
            attribution_signoff_manifest=args.attribution_signoff_manifest,
            attribution_remediation_manifest=args.attribution_remediation_manifest,
            constitutional_audit_manifest=args.constitutional_audit_manifest,
            checkpoint_source_manifest=args.checkpoint_source_manifest,
            checkpoint_candidate_inventory_manifest=args.checkpoint_candidate_inventory_manifest,
            checkpoint_recovery_packet_manifest=args.checkpoint_recovery_packet_manifest,
            checkpoint_launch_guard_manifest=args.checkpoint_launch_guard_manifest,
            checkpoint_claim_downgrade_manifest=args.checkpoint_claim_downgrade_manifest,
            protocol_application_receipt_manifest=args.protocol_application_receipt_manifest,
            output_root=args.output_root,
            cas_root=args.cas_root,
            required_baselines=args.required_baselines,
            required_m1_reports=args.required_m1_reports,
            required_attribution_reviews=args.required_attribution_reviews,
            expected_primitive_count=args.expected_primitive_count,
            m4_settled_trial_target=args.m4_settled_trial_target,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    raise PhaseGateError("PHASE_GATE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
