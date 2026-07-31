"""Prepare and settle protocol-matched ACWM CPBE canary campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.diagnose.probes.source_sign_margin import (
    SourceSignMarginError,
    project_source_sign_margin,
)


class ACWMCPBECanaryError(ValueError):
    """A CPBE canary contract or runtime evidence bundle is invalid."""


def prepare_acwm_cpbe_canary_bundle(
    *,
    plan_path: Path,
    base_campaign_path: Path,
    request_path: Path,
    provenance_path: Path,
    descriptor_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Compile selected Probe DSL work orders into immutable ACWM campaigns."""

    plan_file = Path(plan_path).resolve(strict=True)
    plan = _object(plan_file)
    base_file = Path(base_campaign_path).resolve(strict=True)
    base = _object(base_file)
    request_file = Path(request_path).resolve(strict=True)
    request = _object(request_file)
    provenance_file = Path(provenance_path).resolve(strict=True)
    provenance = _object(provenance_file)
    if plan.get("artifact_type") != "verdiwm-cpbe-plan" or plan.get("state") != "ready":
        raise ACWMCPBECanaryError("ACWM_CPBE_PLAN_NOT_READY")
    if base.get("artifact_type") != "verdiwm-acwm-fingerprint-campaign":
        raise ACWMCPBECanaryError("ACWM_CPBE_BASE_CAMPAIGN_INVALID")
    if request.get("experiment_id") != plan.get("experiment_id"):
        raise ACWMCPBECanaryError("ACWM_CPBE_REQUEST_PLAN_MISMATCH")
    if provenance.get("experiment_id") != plan.get("experiment_id"):
        raise ACWMCPBECanaryError("ACWM_CPBE_PROVENANCE_PLAN_MISMATCH")

    selected = _mapping_list(plan.get("selected_work_orders"), "ACWM_CPBE_WORK_ORDERS_INVALID")
    if not selected:
        raise ACWMCPBECanaryError("ACWM_CPBE_WORK_ORDERS_EMPTY")
    reference_probe_id = _text(
        _mapping(base.get("probe"), "ACWM_CPBE_BASE_PROBE_INVALID"), "probe_id"
    )
    files: dict[str, bytes] = {}
    campaigns: list[dict[str, object]] = []
    descriptor_dir = Path(descriptor_root).resolve(strict=True)
    for order in selected:
        probe_id = _text(order, "probe_id")
        program = _mapping(order.get("program"), "ACWM_CPBE_PROGRAM_INVALID")
        if program.get("parent_probe_ids") != [reference_probe_id]:
            raise ACWMCPBECanaryError(
                f"ACWM_CPBE_REFERENCE_PARENT_MISMATCH:{probe_id}:{reference_probe_id}"
            )
        descriptor_path = descriptor_dir / f"{probe_id}.json"
        descriptor = _object(descriptor_path)
        if (
            descriptor.get("probe_id") != probe_id
            or descriptor.get("program") != program
            or descriptor.get("role") != "diagnostic"
            or descriptor.get("verdict_exposure_allowed") is not False
        ):
            raise ACWMCPBECanaryError(f"ACWM_CPBE_DESCRIPTOR_MISMATCH:{probe_id}")
        campaign = _campaign_from_program(base=base, program=program)
        relative = f"campaigns/{probe_id}.json"
        payload = canonical_json(campaign)
        files[relative] = payload
        campaigns.append(
            {
                "probe_id": probe_id,
                "campaign_id": campaign["campaign_id"],
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    collision_spec = _collision_spec(
        plan=plan,
        request=request,
        provenance=provenance,
        base_campaign=base,
        campaigns=campaigns,
    )
    files["collision-spec.json"] = canonical_json(collision_spec)
    files["source/plan.json"] = plan_file.read_bytes()
    files["source/request.json"] = request_file.read_bytes()
    files["source/provenance.json"] = provenance_file.read_bytes()
    files["source/base-campaign.json"] = base_file.read_bytes()
    return write_bundle(
        output_root=output_root,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-acwm-cpbe-canary-preparation-manifest",
            "state": "ready",
            "experiment_id": plan["experiment_id"],
            "probe_count": len(campaigns),
            "campaigns": campaigns,
            "collision_spec_path": "collision-spec.json",
            "claim_boundary": (
                "This bundle freezes diagnostic canary execution only. It contains no runtime, "
                "collision-resolution, selector-gain, repair-quality, or transfer evidence."
            ),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def evaluate_acwm_cpbe_canary(
    *,
    collision_spec_path: Path,
    candidate_campaign_path: Path,
    reference_campaign_root: Path,
    candidate_campaign_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Reduce paired response charts into a hash-bound CPBE canary receipt."""

    spec_file = Path(collision_spec_path).resolve(strict=True)
    spec = _object(spec_file)
    campaign_file = Path(candidate_campaign_path).resolve(strict=True)
    campaign = _object(campaign_file)
    probe_id = _text(campaign.get("probe", {}), "probe_id")
    campaign_entry = next(
        (row for row in spec.get("candidate_campaigns", []) if row.get("probe_id") == probe_id),
        None,
    )
    if not isinstance(campaign_entry, Mapping):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_CAMPAIGN_NOT_PREREGISTERED:{probe_id}")
    if hashlib.sha256(campaign_file.read_bytes()).hexdigest() != campaign_entry.get("sha256"):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_CAMPAIGN_HASH_MISMATCH:{probe_id}")

    environments = [_text(row, "environment") for row in _mapping_list(spec.get("labels"), "ACWM_CPBE_LABELS_INVALID")]
    labels = {str(row["environment"]): int(row["sign"]) for row in spec["labels"]}
    target = _text(spec, "target_environment")
    reference_root = Path(reference_campaign_root).resolve(strict=True)
    candidate_root = Path(candidate_campaign_root).resolve(strict=True)
    thresholds = _mapping(spec.get("thresholds"), "ACWM_CPBE_THRESHOLDS_INVALID")
    maximum_locality = _finite(thresholds.get("maximum_locality_residual"), "LOCALITY_THRESHOLD")
    minimum_cosine = _finite(thresholds.get("minimum_redundancy_cosine"), "COSINE_THRESHOLD")
    maximum_relative_l2 = _finite(
        thresholds.get("maximum_redundancy_relative_l2"), "RELATIVE_L2_THRESHOLD"
    )
    minimum_separation = _finite(
        thresholds.get("minimum_collision_separation_gain"), "SEPARATION_THRESHOLD"
    )

    reference_vectors: dict[str, tuple[float, ...]] = {}
    candidate_vectors: dict[str, tuple[float, ...]] = {}
    localities: dict[str, float] = {}
    evidence_payloads: dict[str, bytes] = {
        "evidence/collision-spec.json": spec_file.read_bytes(),
        "evidence/candidate-campaign.json": campaign_file.read_bytes(),
    }
    for environment in environments:
        reference_manifest_path = reference_root / "environments" / environment / "manifest.json"
        candidate_manifest_path = candidate_root / "environments" / environment / "manifest.json"
        reference_chart_path = reference_root / "environments" / environment / "response-chart.json"
        candidate_chart_path = candidate_root / "environments" / environment / "response-chart.json"
        reference_manifest = _object(reference_manifest_path)
        candidate_manifest = _object(candidate_manifest_path)
        _assert_matched_contract(reference_manifest, candidate_manifest, environment=environment)
        reference_chart = _object(reference_chart_path)
        candidate_chart = _object(candidate_chart_path)
        reference_probe = _single_probe(reference_chart, environment)
        candidate_probe = _single_probe(candidate_chart, environment)
        if candidate_probe != probe_id or reference_probe != spec.get("reference_probe_id"):
            raise ACWMCPBECanaryError(f"ACWM_CPBE_PROBE_ID_MISMATCH:{environment}")
        if reference_chart.get("outcome_names") != candidate_chart.get("outcome_names"):
            raise ACWMCPBECanaryError(f"ACWM_CPBE_OUTCOME_FRAME_MISMATCH:{environment}")
        reference = _vector(reference_chart.get("response_coordinate"), environment)
        candidate = _vector(candidate_chart.get("response_coordinate"), environment)
        if len(reference) != len(candidate):
            raise ACWMCPBECanaryError(f"ACWM_CPBE_RESPONSE_DIMENSION_MISMATCH:{environment}")
        locality = _finite(
            _mapping(candidate_chart.get("locality_residuals"), "ACWM_CPBE_LOCALITY_INVALID").get(probe_id),
            f"LOCALITY:{environment}",
        )
        reference_vectors[environment] = reference
        candidate_vectors[environment] = candidate
        localities[environment] = locality
        for role, path in (
            ("reference-manifest", reference_manifest_path),
            ("reference-chart", reference_chart_path),
            ("candidate-manifest", candidate_manifest_path),
            ("candidate-chart", candidate_chart_path),
        ):
            evidence_payloads[f"evidence/{environment}/{role}.json"] = path.read_bytes()

    aggregation = _text(_mapping(campaign.get("probe"), "ACWM_CPBE_PROBE_INVALID"), "aggregation")
    candidate_vectors, aggregation_audit = _aggregate_response_vectors(
        candidate_vectors,
        aggregation=aggregation,
        labels=labels,
        target=target,
    )
    comparisons: list[dict[str, object]] = []
    for environment in environments:
        cosine, relative_l2 = _cosine_and_relative_l2(
            reference_vectors[environment], candidate_vectors[environment], environment
        )
        locality = localities[environment]
        local = locality <= maximum_locality
        redundant = local and cosine >= minimum_cosine and relative_l2 <= maximum_relative_l2
        comparisons.append(
            {
                "environment": environment,
                "label_sign": labels[environment],
                "cosine_similarity": cosine,
                "relative_l2": relative_l2,
                "candidate_locality_residual": locality,
                "candidate_locality_pass": local,
                "redundant": redundant,
            }
        )

    reference_margin = _target_source_margin(reference_vectors, labels=labels, target=target)
    candidate_margin = _target_source_margin(candidate_vectors, labels=labels, target=target)
    separation_gain = candidate_margin - reference_margin
    most_redundant = max(
        comparisons,
        key=lambda row: (float(row["cosine_similarity"]), -float(row["relative_l2"])),
    )
    maximum_observed_locality = max(float(row["candidate_locality_residual"]) for row in comparisons)
    locality_passed = all(bool(row["candidate_locality_pass"]) for row in comparisons)
    nonredundant = any(
        bool(row["candidate_locality_pass"]) and not bool(row["redundant"])
        for row in comparisons
    )
    collision_separated = separation_gain > minimum_separation
    passed = locality_passed and nonredundant and collision_separated
    metrics = {
        "locality_residual": maximum_observed_locality,
        "redundancy_cosine": float(most_redundant["cosine_similarity"]),
        "redundancy_relative_l2": float(most_redundant["relative_l2"]),
        "collision_separation": separation_gain,
        "reference_target_source_margin": reference_margin,
        "candidate_target_source_margin": candidate_margin,
        "aggregation": aggregation,
        "locality_passed": locality_passed,
        "nonredundant": nonredundant,
        "collision_separated": collision_separated,
    }
    evidence_artifacts = [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(evidence_payloads.items())
    ]
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-stage-receipt",
        "probe_id": probe_id,
        "stage": "canary",
        "passed": passed,
        "metrics": metrics,
        "evidence_refs": [row["path"] for row in evidence_artifacts],
        "evidence_artifacts": evidence_artifacts,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-cpbe-canary-report",
        "state": "ready",
        "decision": "expand_to_eight_environments" if passed else "eliminate_at_canary",
        "probe_id": probe_id,
        "target_environment": target,
        "reference_probe_id": spec["reference_probe_id"],
        "comparisons": comparisons,
        "aggregation_audit": aggregation_audit,
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "claim_boundary": (
            "This diagnostic canary controls probe expansion only. Passing does not establish "
            "repair quality, selector regret reduction, accepted coverage gain, or transfer."
        ),
    }
    return write_bundle(
        output_root=output_root,
        files={
            **evidence_payloads,
            "canary-report.json": canonical_json(report),
            "cpbe-stage-receipt.json": canonical_json(receipt),
        },
        manifest_fields={
            "artifact_type": "verdiwm-acwm-cpbe-canary-manifest",
            "state": "ready",
            "probe_id": probe_id,
            "decision": report["decision"],
            "passed": passed,
            "report_path": "canary-report.json",
            "receipt_path": "cpbe-stage-receipt.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def publish_acwm_cpbe_expanded_receipt(
    *,
    collision_spec_path: Path,
    probe_id: str,
    baseline_replay_root: Path,
    candidate_replay_root: Path,
    baseline_collision_root: Path,
    candidate_collision_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Settle expanded selector evidence without changing labels or certificates."""

    spec_file = Path(collision_spec_path).resolve(strict=True)
    spec = _object(spec_file)
    target = _text(spec, "target_environment")
    primitive = _text(spec, "primitive")
    target_label = next(
        (int(row["sign"]) for row in spec["labels"] if row["environment"] == target),
        None,
    )
    if target_label not in {-1, 1}:
        raise ACWMCPBECanaryError("ACWM_CPBE_TARGET_LABEL_INVALID")
    baseline_root = Path(baseline_replay_root).resolve(strict=True)
    candidate_root = Path(candidate_replay_root).resolve(strict=True)
    baseline_collision = Path(baseline_collision_root).resolve(strict=True)
    candidate_collision = Path(candidate_collision_root).resolve(strict=True)
    baseline_candidates = baseline_root / "tables/candidates.csv"
    candidate_candidates = candidate_root / "tables/candidates.csv"
    baseline_probability, baseline_rank = _selector_candidate_state(
        baseline_candidates, target=target, primitive=primitive
    )
    candidate_probability, candidate_rank = _selector_candidate_state(
        candidate_candidates, target=target, primitive=primitive
    )
    baseline_error = baseline_probability if target_label < 0 else 1.0 - baseline_probability
    candidate_error = candidate_probability if target_label < 0 else 1.0 - candidate_probability
    regret_reduction = baseline_error - candidate_error
    baseline_collision_report = _object(baseline_collision / "collision-label-evaluation.json")
    candidate_collision_report = _object(candidate_collision / "collision-label-evaluation.json")
    baseline_coverage = _finite(
        baseline_collision_report.get("accepted_coverage"), "BASELINE_ACCEPTED_COVERAGE"
    )
    candidate_coverage = _finite(
        candidate_collision_report.get("accepted_coverage"), "CANDIDATE_ACCEPTED_COVERAGE"
    )
    coverage_gain = candidate_coverage - baseline_coverage
    thresholds = _mapping(spec.get("thresholds"), "ACWM_CPBE_THRESHOLDS_INVALID")
    minimum_regret = _finite(
        thresholds.get("minimum_regret_reduction", 0.0), "REGRET_THRESHOLD"
    )
    minimum_coverage = _finite(
        thresholds.get("minimum_coverage_gain", 0.0), "COVERAGE_THRESHOLD"
    )
    passed = regret_reduction > minimum_regret or coverage_gain > minimum_coverage
    source_paths = {
        "evidence/collision-spec.json": spec_file,
        "evidence/baseline-selector-replay.json": baseline_root / "selector-replay.json",
        "evidence/baseline-candidates.csv": baseline_candidates,
        "evidence/candidate-selector-replay.json": candidate_root / "selector-replay.json",
        "evidence/candidate-candidates.csv": candidate_candidates,
        "evidence/baseline-collision-evaluation.json": baseline_collision
        / "collision-label-evaluation.json",
        "evidence/candidate-collision-evaluation.json": candidate_collision
        / "collision-label-evaluation.json",
        "evidence/candidate-affinity.json": candidate_root / "input-primitive-probe-affinity.json",
    }
    evidence_payloads = {path: source.resolve(strict=True).read_bytes() for path, source in source_paths.items()}
    evidence_artifacts = [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(evidence_payloads.items())
    ]
    metrics = {
        "regret_reduction": regret_reduction,
        "coverage_gain": coverage_gain,
        "baseline_source_positive_probability": baseline_probability,
        "candidate_source_positive_probability": candidate_probability,
        "baseline_target_primitive_rank": baseline_rank,
        "candidate_target_primitive_rank": candidate_rank,
        "target_primitive_demoted": candidate_rank > baseline_rank,
        "target_label_sign": target_label,
        "baseline_accepted_coverage": baseline_coverage,
        "candidate_accepted_coverage": candidate_coverage,
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-stage-receipt",
        "probe_id": probe_id,
        "stage": "expanded",
        "passed": passed,
        "metrics": metrics,
        "evidence_refs": [row["path"] for row in evidence_artifacts],
        "evidence_artifacts": evidence_artifacts,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-cpbe-expanded-report",
        "state": "ready",
        "decision": "admit_probe" if passed else "eliminate_no_selector_gain",
        "probe_id": probe_id,
        "target_environment": target,
        "primitive": primitive,
        "metrics": metrics,
        "thresholds": {
            "minimum_regret_reduction": minimum_regret,
            "minimum_coverage_gain": minimum_coverage,
        },
        "claim_boundary": (
            "Expanded diagnostic settlement only. A failed receipt records no selector or accepted "
            "coverage gain and cannot be reported as repair-quality or transfer evidence."
        ),
    }
    return write_bundle(
        output_root=output_root,
        files={
            **evidence_payloads,
            "expanded-report.json": canonical_json(report),
            "cpbe-stage-receipt.json": canonical_json(receipt),
        },
        manifest_fields={
            "artifact_type": "verdiwm-acwm-cpbe-expanded-manifest",
            "state": "ready",
            "probe_id": probe_id,
            "decision": report["decision"],
            "passed": passed,
            "report_path": "expanded-report.json",
            "receipt_path": "cpbe-stage-receipt.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def assemble_acwm_cpbe_receipt_ledger(
    *,
    plan_path: Path,
    static_offline_root: Path,
    stage_roots: Sequence[Path],
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Rebase hash-verified stage receipts into one live-settlement bundle."""

    plan_file = Path(plan_path).resolve(strict=True)
    plan = _object(plan_file)
    selected = [str(row["probe_id"]) for row in plan["selected_work_orders"]]
    static_root = Path(static_offline_root).resolve(strict=True)
    sources: list[tuple[dict[str, Any], Path]] = []
    for line in (static_root / "cpbe-stage-receipts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_INVALID")
            sources.append((value, static_root))
    for root in stage_roots:
        stage_root = Path(root).resolve(strict=True)
        sources.append((_object(stage_root / "cpbe-stage-receipt.json"), stage_root))
    indexed: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for receipt, root in sources:
        key = (_text(receipt, "probe_id"), _text(receipt, "stage"))
        if key in indexed or key[0] not in selected:
            raise ACWMCPBECanaryError(f"ACWM_CPBE_RECEIPT_FRAME_INVALID:{key[0]}:{key[1]}")
        indexed[key] = (receipt, root)
    files: dict[str, bytes] = {"source/plan.json": plan_file.read_bytes()}
    rewritten: list[dict[str, object]] = []
    stage_order = ("static", "offline", "canary", "expanded")
    for probe_id in selected:
        for stage in stage_order:
            source = indexed.get((probe_id, stage))
            if source is None:
                continue
            receipt, root = source
            artifacts = receipt.get("evidence_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACTS_MISSING")
            rebased_artifacts = []
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_INVALID")
                raw = artifact.get("path")
                if not isinstance(raw, str) or not raw:
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_INVALID")
                relative = Path(raw)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_PATH_INVALID")
                source_path = (root / relative).resolve(strict=True)
                try:
                    source_path.relative_to(root)
                except ValueError as exc:
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_PATH_INVALID") from exc
                payload = source_path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if artifact.get("size_bytes") != len(payload) or artifact.get("sha256") != digest:
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_HASH_MISMATCH")
                destination = f"evidence/{probe_id}/{stage}/{raw}"
                if destination in files and files[destination] != payload:
                    raise ACWMCPBECanaryError("ACWM_CPBE_RECEIPT_ARTIFACT_COLLISION")
                files[destination] = payload
                rebased_artifacts.append(
                    {"path": destination, "sha256": digest, "size_bytes": len(payload)}
                )
            rewritten.append(
                {
                    **receipt,
                    "evidence_refs": [row["path"] for row in rebased_artifacts],
                    "evidence_artifacts": rebased_artifacts,
                }
            )
    receipt_payload = b"".join(canonical_json(receipt) for receipt in rewritten)
    files["cpbe-stage-receipts.jsonl"] = receipt_payload
    return write_bundle(
        output_root=output_root,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-acwm-cpbe-receipt-ledger-manifest",
            "state": "ready",
            "experiment_id": plan["experiment_id"],
            "receipt_count": len(rewritten),
            "probe_count": len(selected),
            "receipt_path": "cpbe-stage-receipts.jsonl",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _campaign_from_program(
    *, base: Mapping[str, Any], program: Mapping[str, Any]
) -> dict[str, object]:
    campaign = deepcopy(dict(base))
    probe_id = _text(program, "probe_id")
    doses = list(program.get("dose_schedule", ()))
    if doses != list(base.get("probe", {}).get("doses", ())):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_DOSE_FRAME_MISMATCH:{probe_id}")
    campaign["campaign_id"] = f"acwm_phys_{probe_id}_canary_v1"
    campaign["claim_scope"] = (
        f"Diagnostic-only CPBE canary for {probe_id}; runtime locality, protocol-matched "
        "nonredundancy, and source-sign collision separation only."
    )
    campaign["probe"] = {
        "probe_id": probe_id,
        "kind": "probe_path",
        "hook_type": program["hook_type"],
        "transformation": (
            f"{program['contrast_operator']} over {program['temporal_basis']} from "
            f"{program['signal_source']} on {program['spatial_mask']}"
        ),
        "scope": "inference_only",
        "dose_unit": f"cpbe_{program['temporal_basis']}_{program['contrast_operator']}",
        "doses": doses,
        "schedule": program["temporal_basis"],
        "generation_mode": "autoregressive",
        "required_capabilities": list(program["required_capabilities"]),
        "preconditions": [
            "DiT action embedder is exposed",
            "time-aligned action sequences are exposed",
            "paired random seed control is available",
        ],
        "invariants": list(program["invariants"]),
        "prediction": program["rationale"],
        "inference_only": True,
        "reversible": program["reversible"],
        "signal_source": program["signal_source"],
        "spatial_mask": program["spatial_mask"],
        "temporal_basis": program["temporal_basis"],
        "contrast_operator": program["contrast_operator"],
        "aggregation": program["aggregation"],
        "diagnostic_only": program["diagnostic_only"],
        "origin": program["origin"],
        "parent_probe_ids": list(program["parent_probe_ids"]),
    }
    campaign["formal_claim_requires"] = (
        "separate expanded-stage selector replay and certificate settlement; canary evidence alone "
        "does not license a model-quality or transfer claim"
    )
    return campaign


def _collision_spec(
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    provenance: Mapping[str, Any],
    base_campaign: Mapping[str, Any],
    campaigns: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    context = _mapping(request.get("context"), "ACWM_CPBE_CONTEXT_INVALID")
    facts = _mapping(provenance.get("collision_facts"), "ACWM_CPBE_COLLISION_FACTS_INVALID")
    target = _text(context, "target_id")
    target_positive = facts.get("target_positive")
    if not isinstance(target_positive, bool):
        raise ACWMCPBECanaryError("ACWM_CPBE_TARGET_LABEL_INVALID")
    source_signs = _mapping(facts.get("source_effect_signs"), "ACWM_CPBE_SOURCE_SIGNS_INVALID")
    labels = [{"environment": target, "role": "target", "sign": 1 if target_positive else -1}]
    for environment, sign in sorted(source_signs.items()):
        if sign not in {"positive", "negative"}:
            raise ACWMCPBECanaryError(f"ACWM_CPBE_SOURCE_SIGN_INVALID:{environment}")
        labels.append(
            {
                "environment": environment,
                "role": "source",
                "sign": 1 if sign == "positive" else -1,
            }
        )
    environment_frame = set(base_campaign.get("environments", {}))
    if any(row["environment"] not in environment_frame for row in labels):
        raise ACWMCPBECanaryError("ACWM_CPBE_COLLISION_ENVIRONMENT_UNKNOWN")
    if not any(row["role"] == "source" and row["sign"] > 0 for row in labels):
        raise ACWMCPBECanaryError("ACWM_CPBE_POSITIVE_SOURCE_MISSING")
    if not any(row["role"] == "source" and row["sign"] < 0 for row in labels):
        raise ACWMCPBECanaryError("ACWM_CPBE_NEGATIVE_SOURCE_MISSING")
    thresholds = _mapping(
        plan.get("successive_halving_thresholds"), "ACWM_CPBE_THRESHOLDS_INVALID"
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-cpbe-collision-specification",
        "experiment_id": plan["experiment_id"],
        "collision_id": context["collision_id"],
        "primitive": context["primitive"],
        "target_environment": target,
        "reference_probe_id": base_campaign["probe"]["probe_id"],
        "protocol": "pilot",
        "labels": labels,
        "candidate_campaigns": list(campaigns),
        "thresholds": {
            "maximum_locality_residual": thresholds["maximum_locality_residual"],
            "minimum_redundancy_cosine": thresholds["maximum_redundancy_cosine"],
            "maximum_redundancy_relative_l2": thresholds[
                "maximum_redundancy_relative_l2"
            ],
            "minimum_collision_separation_gain": thresholds[
                "minimum_collision_separation"
            ],
            "minimum_regret_reduction": thresholds["minimum_regret_reduction"],
            "minimum_coverage_gain": thresholds["minimum_coverage_gain"],
        },
        "separation_metric": (
            "gain over the parent in scale-invariant target-to-positive versus "
            "target-to-negative source cosine-distance margin"
        ),
        "claim_boundary": (
            "Frozen diagnostic collision frame. Source signs and target sign come from settled "
            "target-local official gates; the canary cannot change these labels."
        ),
    }


def _assert_matched_contract(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, environment: str
) -> None:
    fields = (
        "environment",
        "protocol",
        "checkpoint_sha256",
        "config_sha256",
        "seeds",
        "doses",
        "measurement_count",
    )
    mismatch = [field for field in fields if reference.get(field) != candidate.get(field)]
    if mismatch:
        raise ACWMCPBECanaryError(
            f"ACWM_CPBE_MEASUREMENT_CONTRACT_MISMATCH:{environment}:{','.join(mismatch)}"
        )


def _target_source_margin(
    vectors: Mapping[str, tuple[float, ...]], *, labels: Mapping[str, int], target: str
) -> float:
    target_vector = _unit(vectors[target], target)
    positive_distances: list[float] = []
    negative_distances: list[float] = []
    for environment, sign in labels.items():
        if environment == target:
            continue
        distance = 1.0 - sum(
            left * right for left, right in zip(target_vector, _unit(vectors[environment], environment), strict=True)
        )
        (positive_distances if sign > 0 else negative_distances).append(distance)
    if not positive_distances or not negative_distances:
        raise ACWMCPBECanaryError("ACWM_CPBE_SOURCE_SIGN_CLASS_MISSING")
    return sum(positive_distances) / len(positive_distances) - sum(negative_distances) / len(
        negative_distances
    )


def _aggregate_response_vectors(
    vectors: Mapping[str, tuple[float, ...]],
    *,
    aggregation: str,
    labels: Mapping[str, int],
    target: str,
) -> tuple[dict[str, tuple[float, ...]], dict[str, object]]:
    if aggregation == "goal_outcome_vector":
        return dict(vectors), {
            "aggregation": aggregation,
            "fit_environments": [],
            "target_label_used_for_fit": False,
        }
    if aggregation != "source_sign_margin":
        raise ACWMCPBECanaryError(f"ACWM_CPBE_AGGREGATION_UNSUPPORTED:{aggregation}")

    source_signs = {name: sign for name, sign in labels.items() if name != target}
    try:
        return project_source_sign_margin(vectors, source_signs=source_signs, target=target)
    except SourceSignMarginError as exc:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_{exc}") from exc


def _cosine_and_relative_l2(
    left: tuple[float, ...], right: tuple[float, ...], environment: str
) -> tuple[float, float]:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_ZERO_RESPONSE:{environment}")
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    relative_l2 = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    ) / left_norm
    return cosine, relative_l2


def _unit(value: tuple[float, ...], environment: str) -> tuple[float, ...]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-12:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_ZERO_RESPONSE:{environment}")
    return tuple(item / norm for item in value)


def _single_probe(chart: Mapping[str, Any], environment: str) -> str:
    names = chart.get("intervention_names")
    residuals = chart.get("locality_residuals")
    if (
        not isinstance(names, list)
        or len(names) != 1
        or not isinstance(names[0], str)
        or not isinstance(residuals, Mapping)
        or names[0] not in residuals
    ):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_CHART_PROBE_INVALID:{environment}")
    return names[0]


def _vector(value: object, environment: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_RESPONSE_INVALID:{environment}")
    result = tuple(_finite(item, f"RESPONSE:{environment}") for item in value)
    return result


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_JSON_INVALID:{path}")
    return value


def _mapping(value: object, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ACWMCPBECanaryError(error)
    return value


def _mapping_list(value: object, error: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ACWMCPBECanaryError(error)
    return list(value)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ACWMCPBECanaryError(f"ACWM_CPBE_TEXT_INVALID:{key}")
    return item


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ACWMCPBECanaryError(f"ACWM_CPBE_NUMBER_INVALID:{name}")
    return float(value)


def _selector_candidate_state(
    path: Path, *, target: str, primitive: str
) -> tuple[float, int]:
    values: set[tuple[float, int]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("target_environment") == target
                and row.get("selector") == "irg"
                and row.get("primitive") == primitive
            ):
                probability = _finite(
                    float(row["source_positive_probability"]), "SOURCE_PROBABILITY"
                )
                try:
                    rank = int(row["rank"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ACWMCPBECanaryError("ACWM_CPBE_SELECTOR_RANK_INVALID") from exc
                if rank <= 0:
                    raise ACWMCPBECanaryError("ACWM_CPBE_SELECTOR_RANK_INVALID")
                values.add((probability, rank))
    if len(values) != 1:
        raise ACWMCPBECanaryError("ACWM_CPBE_SELECTOR_CANDIDATE_STATE_INVALID")
    return next(iter(values))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--base-campaign", type=Path, required=True)
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--provenance", type=Path, required=True)
    prepare.add_argument("--descriptor-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--archive-db", type=Path)
    prepare.add_argument("--cas-root", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--collision-spec", type=Path, required=True)
    evaluate.add_argument("--candidate-campaign", type=Path, required=True)
    evaluate.add_argument("--reference-campaign-root", type=Path, required=True)
    evaluate.add_argument("--candidate-campaign-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--archive-db", type=Path)
    evaluate.add_argument("--cas-root", type=Path)
    expanded = subparsers.add_parser("expanded")
    expanded.add_argument("--collision-spec", type=Path, required=True)
    expanded.add_argument("--probe-id", required=True)
    expanded.add_argument("--baseline-replay-root", type=Path, required=True)
    expanded.add_argument("--candidate-replay-root", type=Path, required=True)
    expanded.add_argument("--baseline-collision-root", type=Path, required=True)
    expanded.add_argument("--candidate-collision-root", type=Path, required=True)
    expanded.add_argument("--output-root", type=Path, required=True)
    expanded.add_argument("--archive-db", type=Path)
    expanded.add_argument("--cas-root", type=Path)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--plan", type=Path, required=True)
    assemble.add_argument("--static-offline-root", type=Path, required=True)
    assemble.add_argument("--stage-root", type=Path, action="append", required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--archive-db", type=Path)
    assemble.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        manifest = prepare_acwm_cpbe_canary_bundle(
            plan_path=args.plan,
            base_campaign_path=args.base_campaign,
            request_path=args.request,
            provenance_path=args.provenance,
            descriptor_root=args.descriptor_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    elif args.command == "evaluate":
        manifest = evaluate_acwm_cpbe_canary(
            collision_spec_path=args.collision_spec,
            candidate_campaign_path=args.candidate_campaign,
            reference_campaign_root=args.reference_campaign_root,
            candidate_campaign_root=args.candidate_campaign_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    elif args.command == "expanded":
        manifest = publish_acwm_cpbe_expanded_receipt(
            collision_spec_path=args.collision_spec,
            probe_id=args.probe_id,
            baseline_replay_root=args.baseline_replay_root,
            candidate_replay_root=args.candidate_replay_root,
            baseline_collision_root=args.baseline_collision_root,
            candidate_collision_root=args.candidate_collision_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    else:
        manifest = assemble_acwm_cpbe_receipt_ledger(
            plan_path=args.plan,
            static_offline_root=args.static_offline_root,
            stage_roots=args.stage_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
