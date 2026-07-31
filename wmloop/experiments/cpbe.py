"""Counterexample-guided probe basis expansion (CPBE).

CPBE keeps language models in the hypothesis-generation role.  Probe
selection is deterministic and evidence-driven: candidates are expressed in a
small diagnostic DSL, scored with a context-weighted surrogate, and promoted
through a fail-closed successive-halving protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments._artifacts import canonical_json, write_bundle


class CPBEError(ValueError):
    """A CPBE request, history record, or stage receipt is invalid."""


_DSL_FIELDS = (
    "signal_source",
    "hook_type",
    "spatial_mask",
    "temporal_basis",
    "contrast_operator",
    "aggregation",
)
_ORIGINS = {"residual", "mutation", "retrieval", "llm"}
_STAGES = ("static", "offline", "canary", "expanded")
_EVIDENCE_CLASSES = {"synthetic_fixture", "historical_replay", "live"}


@dataclass(frozen=True)
class ProbeProgram:
    """A diagnostic probe represented as a capability-checkable program."""

    probe_id: str
    signal_source: str
    hook_type: str
    spatial_mask: str
    temporal_basis: str
    contrast_operator: str
    dose_schedule: tuple[float, ...]
    aggregation: str
    invariants: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    estimated_gpu_hours: float
    origin: str
    parent_probe_ids: tuple[str, ...] = ()
    rationale: str = ""
    diagnostic_only: bool = True
    reversible: bool = True

    def __post_init__(self) -> None:
        text_fields = (
            self.probe_id,
            self.signal_source,
            self.hook_type,
            self.spatial_mask,
            self.temporal_basis,
            self.contrast_operator,
            self.aggregation,
        )
        if any(not value for value in text_fields) or self.origin not in _ORIGINS:
            raise CPBEError("CPBE_PROBE_IDENTITY_INVALID")
        if not self.diagnostic_only or not self.reversible:
            raise CPBEError("CPBE_PROBE_SCOPE_INVALID")
        if not self.invariants or len(set(self.invariants)) != len(self.invariants):
            raise CPBEError("CPBE_PROBE_INVARIANTS_INVALID")
        if not self.dose_schedule or any(not math.isfinite(value) for value in self.dose_schedule):
            raise CPBEError("CPBE_PROBE_DOSE_INVALID")
        if 0.0 not in self.dose_schedule or not any(value != 0.0 for value in self.dose_schedule):
            raise CPBEError("CPBE_PROBE_ZERO_AND_NONZERO_DOSE_REQUIRED")
        if len(set(self.dose_schedule)) != len(self.dose_schedule):
            raise CPBEError("CPBE_PROBE_DOSE_DUPLICATE")
        if not math.isfinite(self.estimated_gpu_hours) or self.estimated_gpu_hours <= 0.0:
            raise CPBEError("CPBE_PROBE_COST_INVALID")

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.signal_source,
            self.hook_type,
            self.spatial_mask,
            self.temporal_basis,
            self.contrast_operator,
            self.dose_schedule,
            self.aggregation,
            self.invariants,
            self.required_capabilities,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "dose_schedule": list(self.dose_schedule),
            "invariants": list(self.invariants),
            "required_capabilities": list(self.required_capabilities),
            "parent_probe_ids": list(self.parent_probe_ids),
        }


@dataclass(frozen=True)
class CollisionContext:
    collision_id: str
    target_id: str
    backbone_family: str
    capability_class: str
    failure_signature: str
    primitive: str
    available_hooks: tuple[str, ...]
    capabilities: tuple[str, ...]
    unexplained_residual: Mapping[str, float]
    residual_evidence_refs: tuple[str, ...]
    max_canaries: int

    def __post_init__(self) -> None:
        identity = (
            self.collision_id,
            self.target_id,
            self.backbone_family,
            self.capability_class,
            self.failure_signature,
            self.primitive,
        )
        if any(not value for value in identity) or not self.available_hooks or self.max_canaries < 1:
            raise CPBEError("CPBE_CONTEXT_INVALID")
        if not self.unexplained_residual:
            raise CPBEError("CPBE_CONTEXT_RESIDUAL_MISSING")
        if not self.residual_evidence_refs:
            raise CPBEError("CPBE_CONTEXT_RESIDUAL_EVIDENCE_MISSING")
        if set(self.unexplained_residual) - set(_DSL_FIELDS):
            raise CPBEError("CPBE_CONTEXT_RESIDUAL_AXIS_INVALID")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.unexplained_residual.values()):
            raise CPBEError("CPBE_CONTEXT_RESIDUAL_INVALID")
        if sum(float(value) for value in self.unexplained_residual.values()) <= 0.0:
            raise CPBEError("CPBE_CONTEXT_RESIDUAL_EMPTY")


@dataclass(frozen=True)
class HistoricalProbeTrial:
    trial_id: str
    evidence_class: str
    backbone_family: str
    capability_class: str
    failure_signature: str
    primitive: str
    probe: ProbeProgram
    locality_pass: bool | None
    nonredundant: bool | None
    collision_resolved: bool | None
    regret_reduction: float | None
    coverage_gain: float | None
    gpu_hours: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        measured_numbers = (
            value for value in (self.regret_reduction, self.coverage_gain) if value is not None
        )
        if (
            not self.trial_id
            or self.evidence_class not in _EVIDENCE_CLASSES
            or not self.evidence_refs
            or any(not math.isfinite(value) for value in measured_numbers)
            or not math.isfinite(self.gpu_hours)
        ):
            raise CPBEError("CPBE_HISTORY_INVALID")
        if self.gpu_hours <= 0.0:
            raise CPBEError("CPBE_HISTORY_COST_INVALID")


@dataclass(frozen=True)
class AcquisitionPolicy:
    confidence_z: float = 1.0
    coverage_weight: float = 0.5
    collision_weight: float = 0.5
    residual_weight: float = 0.5
    exploration_weight: float = 0.15
    nonlocal_penalty: float = 0.5
    redundancy_penalty: float = 0.25
    nonredundancy_penalty: float = 0.25
    complexity_penalty: float = 0.15

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0.0 for value in asdict(self).values()):
            raise CPBEError("CPBE_POLICY_INVALID")


def build_cpbe_plan(
    *,
    request: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Build one deterministic, capability-filtered CPBE canary plan."""

    if request.get("artifact_type") != "verdiwm-cpbe-request":
        raise CPBEError("CPBE_REQUEST_INVALID")
    experiment_id = _required_string(request, "experiment_id")
    evidence_class = _required_string(request, "evidence_class")
    if evidence_class not in _EVIDENCE_CLASSES:
        raise CPBEError("CPBE_EVIDENCE_CLASS_INVALID")
    context = _parse_context(_mapping(request, "context"))
    current = tuple(_parse_probe(item, default_origin="retrieval") for item in _list(request, "current_probes"))
    if not current:
        raise CPBEError("CPBE_CURRENT_BASIS_REQUIRED")
    grammar = _parse_grammar(_mapping(request, "grammar"))
    retrieval = tuple(
        _parse_probe(item, default_origin="retrieval") for item in _list(request, "retrieval_candidates", optional=True)
    )
    llm = tuple(_parse_probe(item, default_origin="llm") for item in _list(request, "llm_candidates", optional=True))
    for program in (*current, *retrieval, *llm):
        _validate_program_against_grammar(program, grammar)
    all_trials = tuple(_parse_history(item) for item in history)
    trials = tuple(
        trial
        for trial in all_trials
        if evidence_class == "synthetic_fixture" or trial.evidence_class != "synthetic_fixture"
    )
    policy = _parse_policy(request.get("acquisition_policy"))
    halving_thresholds = _parse_halving_thresholds(request.get("successive_halving_thresholds"))

    residual = _generate_residual_candidates(context=context, current=current, grammar=grammar)
    mutations = _generate_mutation_candidates(current=current, grammar=grammar)
    candidates = _deduplicate_candidates((*residual, *mutations, *retrieval, *llm))
    historical_probe_ids = {trial.probe.probe_id for trial in trials}
    candidates = tuple(
        candidate for candidate in candidates if candidate.probe_id not in historical_probe_ids
    )
    if not candidates:
        raise CPBEError("CPBE_NO_CANDIDATES")

    ranking = [
        _score_candidate(candidate, context=context, current=current, history=trials, policy=policy)
        for candidate in candidates
    ]
    ranking.sort(key=lambda row: (-float(row["acquisition_score"]), str(row["probe_id"])))
    compatible = [row for row in ranking if not row["blockers"]]
    selected = compatible[: context.max_canaries]
    selected_ids = {str(row["probe_id"]) for row in selected}
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
        row["selected_for_canary"] = row["probe_id"] in selected_ids

    programs = {candidate.probe_id: candidate for candidate in candidates}
    work_orders = [
        _work_order(
            experiment_id=experiment_id,
            context=context,
            program=programs[str(row["probe_id"])],
            ranking_row=row,
        )
        for row in selected
    ]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-plan",
        "state": "ready" if selected else "blocked",
        "experiment_id": experiment_id,
        "evidence_class": evidence_class,
        "algorithm": "counterexample_guided_probe_basis_expansion_v1",
        "context": _context_dict(context),
        "candidate_generation": {
            "residual_count": len(residual),
            "mutation_count": len(mutations),
            "retrieval_count": len(retrieval),
            "llm_count": len(llm),
            "deduplicated_count": len(candidates),
            "selected_count": len(selected),
            "history_trial_count": len(all_trials),
            "history_trial_count_used": len(trials),
            "synthetic_history_excluded": len(all_trials) - len(trials),
            "historical_probe_ids_excluded": sorted(historical_probe_ids),
        },
        "acquisition_policy": asdict(policy),
        "successive_halving_thresholds": halving_thresholds,
        "ranking": ranking,
        "selected_work_orders": work_orders,
        "claim_boundary": (
            ("Synthetic algorithm fixture only. " if evidence_class == "synthetic_fixture" else "")
            + "A CPBE plan selects diagnostic canaries only. It cannot alter frozen verdict evidence, "
            "admit a probe, establish repair quality, or license transfer before stage settlement."
        ),
    }


def publish_cpbe_plan(
    *,
    request_path: Path,
    history_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    request_bytes = Path(request_path).read_bytes()
    history_bytes = Path(history_path).read_bytes()
    request = _load_json_object(request_path)
    try:
        validate_document("cpbe_request", request, root=Path(__file__).resolve().parents[2])
    except ContractValidationError as exc:
        raise CPBEError(f"CPBE_REQUEST_CONTRACT_INVALID:{exc}") from exc
    history = _load_jsonl(history_path)
    for index, trial in enumerate(history, start=1):
        try:
            validate_document("cpbe_history_trial", trial, root=Path(__file__).resolve().parents[2])
        except ContractValidationError as exc:
            raise CPBEError(f"CPBE_HISTORY_CONTRACT_INVALID:{index}:{exc}") from exc
    report = build_cpbe_plan(request=request, history=history)
    files: dict[str, bytes] = {
        "cpbe-plan.json": canonical_json(report),
        "inputs/cpbe-request.json": request_bytes,
        "inputs/probe-trials.jsonl": history_bytes,
        "tables/candidate-ranking.csv": _ranking_csv(report["ranking"]),
        "README.md": _plan_markdown(report).encode("utf-8"),
    }
    for work_order in report["selected_work_orders"]:
        files[f"work-orders/{work_order['probe_id']}.json"] = canonical_json(work_order)
    return write_bundle(
        output_root=output_root,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-cpbe-plan-manifest",
            "state": report["state"],
            "experiment_id": report["experiment_id"],
            "selected_count": report["candidate_generation"]["selected_count"],
            "probe_work_order_paths": {
                str(work_order["probe_id"]): f"work-orders/{work_order['probe_id']}.json"
                for work_order in report["selected_work_orders"]
            },
            "input_sha256": {
                "request": hashlib.sha256(request_bytes).hexdigest(),
                "history": hashlib.sha256(history_bytes).hexdigest(),
            },
            "report_path": "cpbe-plan.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def settle_cpbe_plan(
    *,
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Apply the frozen successive-halving state machine to stage receipts."""

    if plan.get("artifact_type") != "verdiwm-cpbe-plan" or plan.get("state") not in {"ready", "blocked"}:
        raise CPBEError("CPBE_PLAN_INVALID")
    thresholds = {
        "maximum_locality_residual": 0.5,
        "maximum_redundancy_cosine": 0.999,
        "maximum_redundancy_relative_l2": 0.1,
        "minimum_collision_separation": 0.0,
        "minimum_regret_reduction": 0.0,
        "minimum_coverage_gain": 0.0,
    }
    raw_thresholds = plan.get("successive_halving_thresholds")
    if isinstance(raw_thresholds, Mapping):
        for key in thresholds:
            if key in raw_thresholds:
                value = raw_thresholds[key]
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise CPBEError("CPBE_SETTLEMENT_THRESHOLD_INVALID")
                thresholds[key] = float(value)

    selected = {
        str(item["probe_id"]): item
        for item in _mapping_sequence(plan.get("selected_work_orders"), "CPBE_PLAN_WORK_ORDERS_INVALID")
    }
    by_probe: dict[str, dict[str, Mapping[str, Any]]] = {probe_id: {} for probe_id in selected}
    for receipt in receipts:
        if receipt.get("artifact_type") != "verdiwm-cpbe-stage-receipt":
            raise CPBEError("CPBE_STAGE_RECEIPT_TYPE_INVALID")
        try:
            validate_document("cpbe_stage_receipt", receipt, root=Path(__file__).resolve().parents[2])
        except ContractValidationError as exc:
            raise CPBEError(f"CPBE_STAGE_RECEIPT_CONTRACT_INVALID:{exc}") from exc
        probe_id = _required_string(receipt, "probe_id")
        stage = _required_string(receipt, "stage")
        if probe_id not in selected or stage not in _STAGES:
            raise CPBEError("CPBE_STAGE_RECEIPT_SCOPE_INVALID")
        if stage in by_probe[probe_id]:
            raise CPBEError("CPBE_STAGE_RECEIPT_DUPLICATE")
        if not isinstance(receipt.get("passed"), bool):
            raise CPBEError("CPBE_STAGE_RECEIPT_DECISION_INVALID")
        evidence_refs = receipt.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs or not all(
            isinstance(value, str) and value for value in evidence_refs
        ):
            raise CPBEError("CPBE_STAGE_RECEIPT_EVIDENCE_MISSING")
        metrics = receipt.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise CPBEError("CPBE_STAGE_RECEIPT_METRICS_INVALID")
        by_probe[probe_id][stage] = receipt

    settlements = [
        _settle_candidate(probe_id=probe_id, receipts=by_probe[probe_id], thresholds=thresholds)
        for probe_id in sorted(selected)
    ]
    admitted = sum(item["state"] == "settled_admitted" for item in settlements)
    evidence_class = str(plan.get("evidence_class", "historical_replay"))
    if evidence_class not in _EVIDENCE_CLASSES:
        raise CPBEError("CPBE_EVIDENCE_CLASS_INVALID")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-settlement",
        "state": "settled" if all(item["terminal"] for item in settlements) else "partial",
        "experiment_id": plan["experiment_id"],
        "evidence_class": evidence_class,
        "successive_halving_thresholds": thresholds,
        "candidate_count": len(settlements),
        "admitted_count": admitted,
        "candidates": settlements,
        "claim_boundary": (
            ("Synthetic algorithm fixture only. " if evidence_class == "synthetic_fixture" else "")
            + "Probe admission only updates the diagnostic basis. It does not establish primitive benefit "
            "or cross-backbone transfer; those require independent frozen selector and effect confirmation."
        ),
    }


def publish_cpbe_settlement(
    *,
    plan_path: Path,
    receipts_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    plan_bytes = Path(plan_path).read_bytes()
    receipt_bytes = Path(receipts_path).read_bytes()
    plan = _load_json_object(plan_path)
    receipts = _load_jsonl(receipts_path)
    _verify_live_receipt_artifacts(
        plan=plan,
        receipts=receipts,
        receipt_root=Path(receipts_path).resolve().parent,
    )
    report = settle_cpbe_plan(plan=plan, receipts=receipts)
    return write_bundle(
        output_root=output_root,
        files={
            "cpbe-settlement.json": canonical_json(report),
            "inputs/cpbe-plan.json": plan_bytes,
            "inputs/cpbe-stage-receipts.jsonl": receipt_bytes,
            "tables/candidate-settlement.csv": _settlement_csv(report["candidates"]),
            "README.md": _settlement_markdown(report).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cpbe-settlement-manifest",
            "state": report["state"],
            "experiment_id": report["experiment_id"],
            "candidate_count": report["candidate_count"],
            "admitted_count": report["admitted_count"],
            "input_sha256": {
                "plan": hashlib.sha256(plan_bytes).hexdigest(),
                "receipts": hashlib.sha256(receipt_bytes).hexdigest(),
            },
            "report_path": "cpbe-settlement.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _verify_live_receipt_artifacts(
    *,
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    receipt_root: Path,
) -> None:
    """Fail closed when a non-fixture receipt is not bound to local evidence bytes."""

    if plan.get("evidence_class") == "synthetic_fixture":
        return
    receipt_root = receipt_root.resolve()
    for receipt in receipts:
        artifacts = receipt.get("evidence_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACTS_REQUIRED")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_INVALID")
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_INVALID")
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_PATH_INVALID")
            candidate = receipt_root / relative
            if candidate.is_symlink():
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_PATH_INVALID")
            try:
                path = candidate.resolve(strict=True)
                path.relative_to(receipt_root)
            except (OSError, ValueError) as exc:
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_PATH_INVALID") from exc
            if not path.is_file():
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_PATH_INVALID")
            payload = path.read_bytes()
            if artifact.get("size_bytes") != len(payload):
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_SIZE_MISMATCH")
            if artifact.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise CPBEError("CPBE_LIVE_RECEIPT_ARTIFACT_SHA256_MISMATCH")


def _generate_residual_candidates(
    *,
    context: CollisionContext,
    current: Sequence[ProbeProgram],
    grammar: Mapping[str, tuple[str, ...]],
) -> tuple[ProbeProgram, ...]:
    candidates: list[ProbeProgram] = []
    maximum = max(float(value) for value in context.unexplained_residual.values())
    axes = sorted(
        (
            key
            for key, value in context.unexplained_residual.items()
            if float(value) >= 0.5 * maximum
        ),
        key=lambda key: (-context.unexplained_residual[key], key),
    )
    for axis in axes:
        for parent in current:
            for value in grammar[axis]:
                if value == getattr(parent, axis):
                    continue
                candidates.append(_mutate(parent, axis=axis, value=value, origin="residual", context=context))
    return tuple(candidates)


def _generate_mutation_candidates(
    *,
    current: Sequence[ProbeProgram],
    grammar: Mapping[str, tuple[str, ...]],
) -> tuple[ProbeProgram, ...]:
    candidates: list[ProbeProgram] = []
    for parent in current:
        for axis in _DSL_FIELDS:
            for value in grammar[axis]:
                if value != getattr(parent, axis):
                    candidates.append(_mutate(parent, axis=axis, value=value, origin="mutation"))
    return tuple(candidates)


def _mutate(
    parent: ProbeProgram,
    *,
    axis: str,
    value: str,
    origin: str,
    context: CollisionContext | None = None,
) -> ProbeProgram:
    semantic = f"{parent.probe_id}|{axis}|{value}|{origin}"
    suffix = hashlib.sha256(semantic.encode("utf-8")).hexdigest()[:10]
    rationale = f"Change {axis} from {getattr(parent, axis)} to {value}."
    if context is not None:
        rationale += f" Residual weight={context.unexplained_residual[axis]:.6f}."
    return replace(
        parent,
        probe_id=f"cpbe_{origin}_{suffix}",
        origin=origin,
        parent_probe_ids=(parent.probe_id,),
        rationale=rationale,
        **{axis: value},
    )


def _deduplicate_candidates(candidates: Iterable[ProbeProgram]) -> tuple[ProbeProgram, ...]:
    priority = {"residual": 0, "retrieval": 1, "mutation": 2, "llm": 3}
    by_semantics: dict[tuple[object, ...], ProbeProgram] = {}
    for candidate in candidates:
        existing = by_semantics.get(candidate.semantic_key)
        if existing is None or (priority[candidate.origin], candidate.probe_id) < (
            priority[existing.origin],
            existing.probe_id,
        ):
            by_semantics[candidate.semantic_key] = candidate
    return tuple(sorted(by_semantics.values(), key=lambda item: item.probe_id))


def _score_candidate(
    candidate: ProbeProgram,
    *,
    context: CollisionContext,
    current: Sequence[ProbeProgram],
    history: Sequence[HistoricalProbeTrial],
    policy: AcquisitionPolicy,
) -> dict[str, object]:
    blockers: list[str] = []
    if candidate.hook_type not in context.available_hooks:
        blockers.append(f"hook_unavailable:{candidate.hook_type}")
    missing = sorted(set(candidate.required_capabilities) - set(context.capabilities))
    blockers.extend(f"capability_missing:{value}" for value in missing)

    weighted = [(trial, _trial_similarity(candidate, context, trial)) for trial in history]
    weighted = [(trial, weight) for trial, weight in weighted if weight > 0.0]
    weighted.sort(key=lambda item: (-item[1], item[0].trial_id))
    total_weight = sum(weight for _, weight in weighted)
    p_local = _beta_probability(weighted, lambda trial: trial.locality_pass)
    p_nonredundant = _beta_probability(weighted, lambda trial: trial.nonredundant)
    p_resolve = _beta_probability(weighted, lambda trial: trial.collision_resolved)
    regret_mean, regret_se = _weighted_mean_se(weighted, lambda trial: trial.regret_reduction)
    coverage_mean, coverage_se = _weighted_mean_se(weighted, lambda trial: trial.coverage_gain)
    utility_mean = regret_mean + policy.coverage_weight * coverage_mean + policy.collision_weight * p_resolve
    utility_se = math.sqrt(regret_se**2 + (policy.coverage_weight * coverage_se) ** 2)
    utility_lcb = utility_mean - policy.confidence_z * utility_se
    residual_alignment = _residual_alignment(candidate, current=current, context=context)
    structural_redundancy = max(_program_similarity(candidate, parent) for parent in current)
    edit_count = min(_program_edit_count(candidate, parent) for parent in current)
    uncertainty = 1.0 / math.sqrt(1.0 + total_weight)
    acquisition = (
        utility_lcb
        + policy.residual_weight * residual_alignment
        + policy.exploration_weight * uncertainty
    ) / candidate.estimated_gpu_hours
    acquisition -= policy.nonlocal_penalty * (1.0 - p_local)
    acquisition -= policy.redundancy_penalty * structural_redundancy
    acquisition -= policy.nonredundancy_penalty * (1.0 - p_nonredundant)
    complexity_penalty = (
        policy.complexity_penalty * max(edit_count - 1, 0) / candidate.estimated_gpu_hours
    )
    acquisition -= complexity_penalty
    return {
        "probe_id": candidate.probe_id,
        "origin": candidate.origin,
        "program": candidate.to_dict(),
        "history_effective_weight": total_weight,
        "history_trial_ids": [trial.trial_id for trial, _ in weighted[:8]],
        "history_evidence_refs": sorted(
            {reference for trial, _ in weighted[:8] for reference in trial.evidence_refs}
        ),
        "p_local": p_local,
        "p_nonredundant": p_nonredundant,
        "p_collision_resolved": p_resolve,
        "expected_regret_reduction": regret_mean,
        "expected_coverage_gain": coverage_mean,
        "utility_lcb": utility_lcb,
        "residual_alignment": residual_alignment,
        "structural_redundancy": structural_redundancy,
        "edit_count": edit_count,
        "complexity_penalty": complexity_penalty,
        "uncertainty": uncertainty,
        "estimated_gpu_hours": candidate.estimated_gpu_hours,
        "acquisition_score": acquisition,
        "blockers": blockers,
    }


def _trial_similarity(
    candidate: ProbeProgram,
    context: CollisionContext,
    trial: HistoricalProbeTrial,
) -> float:
    context_score = (
        float(context.backbone_family == trial.backbone_family)
        + float(context.capability_class == trial.capability_class)
        + float(context.failure_signature == trial.failure_signature)
        + float(context.primitive == trial.primitive)
    ) / 4.0
    return _program_similarity(candidate, trial.probe) * (0.25 + 0.75 * context_score)


def _program_similarity(left: ProbeProgram, right: ProbeProgram) -> float:
    compared = [left.hook_type == right.hook_type]
    compared.extend(getattr(left, field) == getattr(right, field) for field in _DSL_FIELDS if field != "hook_type")
    return sum(compared) / len(compared)


def _program_edit_count(left: ProbeProgram, right: ProbeProgram) -> int:
    return sum(getattr(left, field) != getattr(right, field) for field in _DSL_FIELDS)


def _beta_probability(
    weighted: Sequence[tuple[HistoricalProbeTrial, float]],
    outcome: Any,
) -> float:
    observed = [(trial, weight, outcome(trial)) for trial, weight in weighted]
    observed = [(trial, weight, value) for trial, weight, value in observed if value is not None]
    successes = sum(weight for _, weight, value in observed if value is True)
    total = sum(weight for _, weight, _ in observed)
    return (1.0 + successes) / (2.0 + total)


def _weighted_mean_se(
    weighted: Sequence[tuple[HistoricalProbeTrial, float]],
    value: Any,
) -> tuple[float, float]:
    observed = [(trial, weight, value(trial)) for trial, weight in weighted]
    observed = [(trial, weight, item) for trial, weight, item in observed if item is not None]
    total = sum(weight for _, weight, _ in observed)
    if total <= 0.0:
        return 0.0, 0.5
    mean = sum(weight * float(item) for _, weight, item in observed) / total
    variance = sum(weight * (float(item) - mean) ** 2 for _, weight, item in observed) / total
    effective_n = total**2 / max(sum(weight**2 for _, weight, _ in observed), 1e-12)
    return mean, math.sqrt(variance / max(effective_n, 1.0) + 0.25 / (1.0 + total))


def _residual_alignment(
    candidate: ProbeProgram,
    *,
    current: Sequence[ProbeProgram],
    context: CollisionContext,
) -> float:
    closest = max(current, key=lambda parent: _program_similarity(candidate, parent))
    total = sum(float(value) for value in context.unexplained_residual.values())
    changed = sum(
        float(context.unexplained_residual.get(field, 0.0))
        for field in _DSL_FIELDS
        if getattr(candidate, field) != getattr(closest, field)
    )
    return changed / total


def _work_order(
    *,
    experiment_id: str,
    context: CollisionContext,
    program: ProbeProgram,
    ranking_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-probe-work-order",
        "work_order_id": f"{experiment_id}:{program.probe_id}",
        "probe_id": program.probe_id,
        "role": "diagnostic",
        "verdict_exposure_allowed": False,
        "collision_id": context.collision_id,
        "environment": context.target_id,
        "signature": context.failure_signature,
        "priority": "P0_collision_basis_gap",
        "signal_contract": (
            "Measure the frozen Probe DSL program as diagnostic-only evidence for collision separation; "
            "never expose its output to verdict evidence."
        ),
        "failure_signature": context.failure_signature,
        "primitive": context.primitive,
        "program": program.to_dict(),
        "selection_evidence": {
            key: ranking_row[key]
            for key in (
                "rank",
                "acquisition_score",
                "utility_lcb",
                "residual_alignment",
                "p_local",
                "p_nonredundant",
                "p_collision_resolved",
                "estimated_gpu_hours",
                "history_trial_ids",
                "history_evidence_refs",
            )
        },
        "required_stages": list(_STAGES),
        "admission_gates": [
            "schema_valid_diagnostic_probe_output",
            "offline_fixture_test_passed",
            "runtime_smoke_on_dev_split",
            "locality_and_nonredundancy_canary_passed",
            "selector_regret_or_coverage_gain_observed",
            "no_verdict_evidence_exposure",
        ],
        "allowed_mutation_paths": [
            f"wmloop/diagnose/probes/{program.probe_id}.py",
            f"tests/test_{program.probe_id}.py",
            "configs/probes/staging/",
        ],
        "forbidden_surfaces": [
            "configs/goal/",
            "configs/constitution/",
            "frozen evaluator code",
            "verdict_evidence",
        ],
        "claim_boundary": (
            "Selection creates a diagnostic canary work order only; "
            "no effect or transfer claim is licensed."
        ),
    }


def _settle_candidate(
    *,
    probe_id: str,
    receipts: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    supplied_indices = sorted(_STAGES.index(stage) for stage in receipts)
    if supplied_indices and supplied_indices != list(range(supplied_indices[-1] + 1)):
        raise CPBEError("CPBE_STAGE_RECEIPT_OUT_OF_ORDER")
    completed: list[str] = []
    blockers: list[str] = []
    state = "planned"
    terminal = False
    for stage in _STAGES:
        receipt = receipts.get(stage)
        if receipt is None:
            break
        if len(completed) != _STAGES.index(stage):
            raise CPBEError("CPBE_STAGE_RECEIPT_OUT_OF_ORDER")
        passed = bool(receipt["passed"])
        metrics = receipt.get("metrics", {})
        stage_blockers = _stage_metric_blockers(stage=stage, metrics=metrics, thresholds=thresholds)
        if not passed:
            stage_blockers.append(f"{stage}_receipt_failed")
        if stage_blockers:
            blockers.extend(stage_blockers)
            state = f"eliminated_{stage}"
            terminal = True
            break
        completed.append(stage)
        state = {
            "static": "offline_ready",
            "offline": "canary_ready",
            "canary": "expansion_ready",
            "expanded": "settled_admitted",
        }[stage]
    if state == "settled_admitted":
        terminal = True
    return {
        "probe_id": probe_id,
        "state": state,
        "terminal": terminal,
        "completed_stages": completed,
        "blockers": blockers,
        "evidence_refs": [
            ref
            for stage in _STAGES
            for ref in receipts.get(stage, {}).get("evidence_refs", [])
            if isinstance(ref, str)
        ],
    }


def _stage_metric_blockers(
    *,
    stage: str,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    if stage in {"static", "offline"}:
        return []
    blockers: list[str] = []
    if stage == "canary":
        locality = _metric(metrics, "locality_residual")
        redundancy = _metric(metrics, "redundancy_cosine")
        relative_l2 = metrics.get("redundancy_relative_l2")
        nonredundant = metrics.get("nonredundant")
        separation = _metric(metrics, "collision_separation")
        if locality > thresholds["maximum_locality_residual"]:
            blockers.append("locality_residual_exceeded")
        if nonredundant is not None:
            if not isinstance(nonredundant, bool):
                raise CPBEError("CPBE_STAGE_METRIC_INVALID:nonredundant")
            if not nonredundant:
                blockers.append("empirically_redundant")
        elif relative_l2 is None:
            # Backward compatibility for historical one-dimensional receipts.
            if abs(redundancy) > thresholds["maximum_redundancy_cosine"]:
                blockers.append("empirically_redundant")
        else:
            parsed_relative_l2 = _metric(metrics, "redundancy_relative_l2")
            if (
                redundancy >= thresholds["maximum_redundancy_cosine"]
                and parsed_relative_l2 <= thresholds["maximum_redundancy_relative_l2"]
            ):
                blockers.append("empirically_redundant")
        if separation <= thresholds["minimum_collision_separation"]:
            blockers.append("collision_not_separated")
    if stage == "expanded":
        regret = _metric(metrics, "regret_reduction")
        coverage = _metric(metrics, "coverage_gain")
        if regret <= thresholds["minimum_regret_reduction"] and coverage <= thresholds["minimum_coverage_gain"]:
            blockers.append("no_selector_or_coverage_gain")
    return blockers


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CPBEError(f"CPBE_STAGE_METRIC_MISSING:{name}")
    return float(value)


def _parse_context(value: Mapping[str, Any]) -> CollisionContext:
    residual = _mapping(value, "unexplained_residual")
    return CollisionContext(
        collision_id=_required_string(value, "collision_id"),
        target_id=_required_string(value, "target_id"),
        backbone_family=_required_string(value, "backbone_family"),
        capability_class=_required_string(value, "capability_class"),
        failure_signature=_required_string(value, "failure_signature"),
        primitive=_required_string(value, "primitive"),
        available_hooks=_string_tuple(value.get("available_hooks"), "CPBE_CONTEXT_HOOKS_INVALID"),
        capabilities=_string_tuple(
            value.get("capabilities", []),
            "CPBE_CONTEXT_CAPABILITIES_INVALID",
            allow_empty=True,
        ),
        unexplained_residual={str(key): float(item) for key, item in residual.items()},
        residual_evidence_refs=_string_tuple(
            value.get("residual_evidence_refs"),
            "CPBE_CONTEXT_RESIDUAL_EVIDENCE_MISSING",
        ),
        max_canaries=_required_int(value, "max_canaries"),
    )


def _parse_probe(value: object, *, default_origin: str) -> ProbeProgram:
    if not isinstance(value, Mapping):
        raise CPBEError("CPBE_PROBE_INVALID")
    origin = value.get("origin", default_origin)
    return ProbeProgram(
        probe_id=_required_string(value, "probe_id"),
        signal_source=_required_string(value, "signal_source"),
        hook_type=_required_string(value, "hook_type"),
        spatial_mask=_required_string(value, "spatial_mask"),
        temporal_basis=_required_string(value, "temporal_basis"),
        contrast_operator=_required_string(value, "contrast_operator"),
        dose_schedule=_number_tuple(value.get("dose_schedule"), "CPBE_PROBE_DOSE_INVALID"),
        aggregation=_required_string(value, "aggregation"),
        invariants=_string_tuple(value.get("invariants"), "CPBE_PROBE_INVARIANTS_INVALID"),
        required_capabilities=_string_tuple(
            value.get("required_capabilities", []),
            "CPBE_PROBE_CAPABILITIES_INVALID",
            allow_empty=True,
        ),
        estimated_gpu_hours=_required_number(value, "estimated_gpu_hours"),
        origin=str(origin),
        parent_probe_ids=_string_tuple(
            value.get("parent_probe_ids", []),
            "CPBE_PROBE_PARENTS_INVALID",
            allow_empty=True,
        ),
        rationale=str(value.get("rationale", "")),
        diagnostic_only=value.get("diagnostic_only", True) is True,
        reversible=value.get("reversible", True) is True,
    )


def _parse_history(value: Mapping[str, Any]) -> HistoricalProbeTrial:
    context = _mapping(value, "context")
    outcomes = _mapping(value, "outcomes")
    return HistoricalProbeTrial(
        trial_id=_required_string(value, "trial_id"),
        evidence_class=_required_string(value, "evidence_class"),
        backbone_family=_required_string(context, "backbone_family"),
        capability_class=_required_string(context, "capability_class"),
        failure_signature=_required_string(context, "failure_signature"),
        primitive=_required_string(context, "primitive"),
        probe=_parse_probe(value.get("probe"), default_origin="retrieval"),
        locality_pass=_optional_bool(outcomes, "locality_pass"),
        nonredundant=_optional_bool(outcomes, "nonredundant"),
        collision_resolved=_optional_bool(outcomes, "collision_resolved"),
        regret_reduction=_optional_number(outcomes, "regret_reduction"),
        coverage_gain=_optional_number(outcomes, "coverage_gain"),
        gpu_hours=_required_number(outcomes, "gpu_hours"),
        evidence_refs=_string_tuple(
            value.get("evidence_refs"),
            "CPBE_HISTORY_EVIDENCE_MISSING",
        ),
    )


def _parse_grammar(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    missing = [field for field in _DSL_FIELDS if field not in value]
    if missing:
        raise CPBEError("CPBE_GRAMMAR_MISSING:" + ",".join(missing))
    extras = set(value) - set(_DSL_FIELDS)
    if extras:
        raise CPBEError("CPBE_GRAMMAR_UNKNOWN:" + ",".join(sorted(extras)))
    return {
        field: _string_tuple(value[field], f"CPBE_GRAMMAR_INVALID:{field}")
        for field in _DSL_FIELDS
    }


def _validate_program_against_grammar(
    program: ProbeProgram,
    grammar: Mapping[str, tuple[str, ...]],
) -> None:
    outside = [field for field in _DSL_FIELDS if getattr(program, field) not in grammar[field]]
    if outside:
        raise CPBEError(f"CPBE_PROBE_OUTSIDE_GRAMMAR:{program.probe_id}:" + ",".join(outside))


def _parse_policy(value: object) -> AcquisitionPolicy:
    if value is None:
        return AcquisitionPolicy()
    if not isinstance(value, Mapping):
        raise CPBEError("CPBE_POLICY_INVALID")
    allowed = set(asdict(AcquisitionPolicy()))
    extras = set(value) - allowed
    if extras:
        raise CPBEError("CPBE_POLICY_UNKNOWN:" + ",".join(sorted(extras)))
    defaults = asdict(AcquisitionPolicy())
    defaults.update({key: float(item) for key, item in value.items()})
    return AcquisitionPolicy(**defaults)


def _parse_halving_thresholds(value: object) -> dict[str, float]:
    defaults = {
        "maximum_locality_residual": 0.5,
        "maximum_redundancy_cosine": 0.999,
        "maximum_redundancy_relative_l2": 0.1,
        "minimum_collision_separation": 0.0,
        "minimum_regret_reduction": 0.0,
        "minimum_coverage_gain": 0.0,
    }
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise CPBEError("CPBE_SETTLEMENT_THRESHOLD_INVALID")
    extras = set(value) - set(defaults)
    if extras:
        raise CPBEError("CPBE_SETTLEMENT_THRESHOLD_UNKNOWN:" + ",".join(sorted(extras)))
    for key, item in value.items():
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise CPBEError("CPBE_SETTLEMENT_THRESHOLD_INVALID")
        defaults[key] = float(item)
    if (
        defaults["maximum_locality_residual"] < 0.0
        or not 0.0 <= defaults["maximum_redundancy_cosine"] <= 1.0
        or defaults["maximum_redundancy_relative_l2"] < 0.0
    ):
        raise CPBEError("CPBE_SETTLEMENT_THRESHOLD_INVALID")
    return defaults


def _context_dict(context: CollisionContext) -> dict[str, object]:
    return {
        **asdict(context),
        "available_hooks": list(context.available_hooks),
        "capabilities": list(context.capabilities),
        "unexplained_residual": dict(context.unexplained_residual),
        "residual_evidence_refs": list(context.residual_evidence_refs),
    }


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise CPBEError(f"CPBE_STRING_REQUIRED:{key}")
    return item


def _required_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
        raise CPBEError(f"CPBE_NUMBER_REQUIRED:{key}")
    return float(item)


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise CPBEError(f"CPBE_INTEGER_REQUIRED:{key}")
    return item


def _required_bool(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise CPBEError(f"CPBE_BOOLEAN_REQUIRED:{key}")
    return item


def _optional_bool(value: Mapping[str, Any], key: str) -> bool | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, bool):
        raise CPBEError(f"CPBE_BOOLEAN_REQUIRED:{key}")
    return item


def _optional_number(value: Mapping[str, Any], key: str) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
        raise CPBEError(f"CPBE_NUMBER_REQUIRED:{key}")
    return float(item)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise CPBEError(f"CPBE_MAPPING_REQUIRED:{key}")
    return item


def _list(value: Mapping[str, Any], key: str, *, optional: bool = False) -> list[Any]:
    item = value.get(key, [] if optional else None)
    if not isinstance(item, list):
        raise CPBEError(f"CPBE_LIST_REQUIRED:{key}")
    return item


def _string_tuple(value: object, code: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CPBEError(code)
    if not all(isinstance(item, str) and item for item in value) or len(set(value)) != len(value):
        raise CPBEError(code)
    return tuple(value)


def _number_tuple(value: object, code: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value
    ):
        raise CPBEError(code)
    return tuple(float(item) for item in value)


def _mapping_sequence(value: object, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CPBEError(code)
    return list(value)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CPBEError(f"CPBE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CPBEError(f"CPBE_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CPBEError(f"CPBE_JSONL_INVALID:{path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CPBEError(f"CPBE_JSONL_INVALID:{path}:{index}") from exc
        if not isinstance(payload, Mapping):
            raise CPBEError(f"CPBE_JSONL_OBJECT_REQUIRED:{path}:{index}")
        rows.append(payload)
    return rows


def _ranking_csv(rows: object) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "rank",
        "probe_id",
        "origin",
        "selected_for_canary",
        "acquisition_score",
        "utility_lcb",
        "residual_alignment",
        "p_local",
        "p_nonredundant",
        "p_collision_resolved",
        "estimated_gpu_hours",
        "blockers",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in _mapping_sequence(rows, "CPBE_RANKING_INVALID"):
        writer.writerow({**{field: row.get(field, "") for field in fields}, "blockers": ";".join(row["blockers"])})
    return output.getvalue().encode("utf-8")


def _settlement_csv(rows: object) -> bytes:
    output = io.StringIO(newline="")
    fields = ["probe_id", "state", "terminal", "completed_stages", "blockers", "evidence_refs"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in _mapping_sequence(rows, "CPBE_SETTLEMENT_INVALID"):
        writer.writerow(
            {
                "probe_id": row["probe_id"],
                "state": row["state"],
                "terminal": row["terminal"],
                "completed_stages": ";".join(row["completed_stages"]),
                "blockers": ";".join(row["blockers"]),
                "evidence_refs": ";".join(row["evidence_refs"]),
            }
        )
    return output.getvalue().encode("utf-8")


def _plan_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CPBE Plan",
        "",
        f"State: `{report['state']}`",
        f"Evidence class: `{report['evidence_class']}`",
        f"Experiment: `{report['experiment_id']}`",
        f"Evidence class: `{report['evidence_class']}`",
        f"Collision: `{report['context']['collision_id']}`",
        "",
        "| Rank | Probe | Origin | Score | Selected | Blockers |",
        "|---:|:--|:--|---:|:--:|:--|",
    ]
    for row in report["ranking"]:
        lines.append(
            f"| {row['rank']} | `{row['probe_id']}` | {row['origin']} | "
            f"{float(row['acquisition_score']):.6f} | {row['selected_for_canary']} | "
            f"{', '.join(row['blockers']) or '-'} |"
        )
    lines.extend(["", "## Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _settlement_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CPBE Settlement",
        "",
        f"State: `{report['state']}`",
        f"Admitted: `{report['admitted_count']}/{report['candidate_count']}`",
        "",
        "| Probe | State | Completed stages | Blockers |",
        "|:--|:--|:--|:--|",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| `{row['probe_id']}` | {row['state']} | {', '.join(row['completed_stages']) or '-'} | "
            f"{', '.join(row['blockers']) or '-'} |"
        )
    lines.extend(["", "## Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="generate a CPBE candidate and canary plan")
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument("--history", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--archive-db", type=Path)
    plan.add_argument("--cas-root", type=Path)
    settle = commands.add_parser("settle", help="settle CPBE stage receipts")
    settle.add_argument("--plan", type=Path, required=True)
    settle.add_argument("--receipts", type=Path, required=True)
    settle.add_argument("--output-root", type=Path, required=True)
    settle.add_argument("--archive-db", type=Path)
    settle.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "plan":
        manifest = publish_cpbe_plan(
            request_path=args.request,
            history_path=args.history,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    else:
        manifest = publish_cpbe_settlement(
            plan_path=args.plan,
            receipts_path=args.receipts,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
