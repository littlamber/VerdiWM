"""Compile evidence-bound, discriminating experiment portfolios."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.capability_gap_planner import (
    CapabilityGapPlannerError,
    validate_capability_requirement_graph,
    validate_gap_plan_against_requirement_graph,
    validate_goal_ir,
)
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.portable_transfer_knowledge import (
    build_mechanism_contract,
    validate_mechanism_contract,
)
from wmloop.geometry.mechanism_relations import validate_mechanism_relation
from wmloop.geometry.types import GeometryValidationError


class ExperimentPortfolioError(RuntimeError):
    """A hypothesis batch or compiled portfolio failed closed."""


_RANKING_WEIGHTS = {
    "information_gain_weight": 0.50,
    "uncertainty_weight": 0.20,
    "competition_weight": 0.15,
    "diversity_weight": 0.15,
    "cost_weight": 0.10,
}
_ENTRY_ROLES = {"baseline", "mechanism_test", "negative_control", "ablation"}


def build_relation_hypothesis_batch(
    *,
    relation: Mapping[str, object],
    source_mechanism: Mapping[str, object],
    target_mechanism: Mapping[str, object],
    expected_portrait_changes: Sequence[str],
    required_module_capabilities: Sequence[str],
    information_gain: float,
    uncertainty: float,
    estimated_screen_gpu_hours: float,
    selection_reason: str = "relation_candidate_from_settled_evidence",
) -> dict[str, object]:
    """Turn a pairwise relation into the existing portfolio input contract.

    The resulting mechanism is the A+B composition. Its two required
    component-removal ablations become the A-only and B-only counterfactuals in
    ``_portfolio_entries``. No relation is promoted by this adapter; the
    resulting batch still requires normal resource admission and verification.
    """

    validate_mechanism_relation(relation)
    validate_mechanism_contract(source_mechanism)
    validate_mechanism_contract(target_mechanism)
    if not expected_portrait_changes or any(not str(value).strip() for value in expected_portrait_changes):
        raise ExperimentPortfolioError("RELATION_EXPECTED_CHANGES_INVALID")
    if any(not str(value).strip() for value in required_module_capabilities):
        raise ExperimentPortfolioError("RELATION_MODULE_CAPABILITIES_INVALID")
    source_id = str(relation["source_mechanism_id"])
    target_id = str(relation["target_mechanism_id"])
    if source_id != str(source_mechanism["mechanism_id"]) or target_id != str(target_mechanism["mechanism_id"]):
        raise ExperimentPortfolioError("RELATION_MECHANISM_BINDING_MISMATCH")
    source_caps = [str(value) for value in source_mechanism["required_capabilities"]]
    target_caps = [str(value) for value in target_mechanism["required_capabilities"]]
    source_interfaces = [str(value) for value in source_mechanism["target_interface_requirements"]]
    target_interfaces = [str(value) for value in target_mechanism["target_interface_requirements"]]
    relation_type = str(relation["relation_type"])
    combined = build_mechanism_contract(
        causal_claim=(
            f"The {relation_type} composition of {source_id} and {target_id} "
            "improves the declared outcome beyond the single-mechanism counterfactuals."
        ),
        intervention_semantics=(
            f"compose[{relation['composition_operator']}]({source_id},{target_id})"
        ),
        required_capabilities=sorted(set(source_caps + target_caps)),
        optional_capabilities=sorted(
            set(str(value) for value in source_mechanism["optional_capabilities"])
            | set(str(value) for value in target_mechanism["optional_capabilities"])
        ),
        target_interface_requirements=sorted(set(source_interfaces + target_interfaces)),
        prohibited_substitutions=sorted(
            set(str(value) for value in source_mechanism["prohibited_substitutions"])
            | set(str(value) for value in target_mechanism["prohibited_substitutions"])
        ),
        required_ablations=sorted(
            {f"remove:{source_id}", f"remove:{target_id}"}
            | set(str(value) for value in relation["required_ablations"])
        ),
        falsification_criterion=(
            "The composition is falsified when the combined effect does not exceed "
            "the additive single-mechanism contrast, or when a protected metric regresses."
        ),
        known_anti_conditions=sorted(
            set(str(value) for value in source_mechanism["known_anti_conditions"])
            | set(str(value) for value in target_mechanism["known_anti_conditions"])
            | set(str(value) for value in relation["anti_conditions"])
        ),
        source_evidence_refs=sorted(
            set(str(value) for value in source_mechanism["source_evidence_refs"])
            | set(str(value) for value in target_mechanism["source_evidence_refs"])
            | set(str(value) for value in relation["evidence_refs"])
        ),
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-experiment-hypothesis-batch",
        "candidates": [{
            "mechanism_contract": combined,
            "expected_portrait_changes": sorted(set(str(value) for value in expected_portrait_changes)),
            "structural_conditions": sorted(set(str(value) for value in relation["condition_set"])),
            "behavioral_conditions": [],
            "discriminates_from": [],
            "required_module_capabilities": sorted(set(str(value) for value in required_module_capabilities)),
            "information_gain": float(information_gain),
            "uncertainty": float(uncertainty),
            "estimated_screen_gpu_hours": float(estimated_screen_gpu_hours),
            "selection_reason": selection_reason,
        }],
    }


def compile_relation_experiment_portfolio(
    *,
    relation: Mapping[str, object],
    source_mechanism: Mapping[str, object],
    target_mechanism: Mapping[str, object],
    expected_portrait_changes: Sequence[str],
    required_module_capabilities: Sequence[str],
    information_gain: float,
    uncertainty: float,
    estimated_screen_gpu_hours: float,
    portfolio_arguments: Mapping[str, object],
) -> dict[str, object]:
    """Compile a relation candidate through the normal portfolio pipeline.

    ``portfolio_arguments`` contains the regular goal, portrait, requirement,
    policy, and evaluator inputs accepted by :func:`compile_experiment_portfolio`.
    Keeping this as a thin adapter guarantees relation experiments inherit the
    same budget, admission, and frozen-evaluator policy as ordinary hypotheses.
    """

    batch = build_relation_hypothesis_batch(
        relation=relation,
        source_mechanism=source_mechanism,
        target_mechanism=target_mechanism,
        expected_portrait_changes=expected_portrait_changes,
        required_module_capabilities=required_module_capabilities,
        information_gain=information_gain,
        uncertainty=uncertainty,
        estimated_screen_gpu_hours=estimated_screen_gpu_hours,
    )
    arguments = dict(portfolio_arguments)
    arguments["hypothesis_batch"] = batch
    return compile_experiment_portfolio(**arguments)  # type: ignore[arg-type]


def compile_experiment_portfolio(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    requirement_graph: Mapping[str, object],
    gap_plan: Mapping[str, object],
    hypothesis_batch: Mapping[str, object],
    policy_id: str,
    maximum_hypotheses: int,
    max_total_gpu_hours: float,
    minimum_replications: int,
    baseline_gpu_hours: float,
    control_cost_fraction: float,
    ablation_cost_fraction: float,
    protected_metrics: Sequence[str],
    heldout_protocol: str,
    required_artifact_classes: Sequence[str],
    root: Path | None = None,
) -> dict[str, object]:
    """Compile LLM-shaped mechanism drafts under Kernel-owned authority."""

    _validate_inputs(
        goal_ir=goal_ir,
        portrait=portrait,
        requirement_graph=requirement_graph,
        gap_plan=gap_plan,
        hypothesis_batch=hypothesis_batch,
        root=root,
    )
    policy = _normalize_policy(
        policy_id=policy_id,
        maximum_hypotheses=maximum_hypotheses,
        max_total_gpu_hours=max_total_gpu_hours,
        minimum_replications=minimum_replications,
        baseline_gpu_hours=baseline_gpu_hours,
        control_cost_fraction=control_cost_fraction,
        ablation_cost_fraction=ablation_cost_fraction,
        protected_metrics=protected_metrics,
        heldout_protocol=heldout_protocol,
        required_artifact_classes=required_artifact_classes,
    )
    candidates = _normalize_candidates(
        hypothesis_batch,
        requirement_graph=requirement_graph,
    )
    evaluator = _frozen_evaluator(portrait, heldout_protocol=heldout_protocol)
    ranked = _rank_and_select(candidates, policy=policy)
    selected = [row for row in ranked if row["selected"]]
    entries = _portfolio_entries(
        selected,
        goal_ir=goal_ir,
        evaluator=evaluator,
        policy=policy,
    )
    estimated_total = round(
        sum(float(row["cost"]["total_gpu_hours"]) for row in entries),
        8,
    )
    state = "ready_for_resource_admission" if selected else "blocked_budget"
    if state == "blocked_budget":
        next_action = "stop_budget"
    elif gap_plan["state"] == "requires_manufacturing":
        next_action = "manufacture_modules"
    else:
        next_action = "resource_admission"
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-experiment-portfolio",
        "state": state,
        "bindings": {
            "goal_ir_id": goal_ir["goal_ir_id"],
            "goal_ir_digest": _digest(goal_ir),
            "portrait_id": portrait["portrait_id"],
            "portrait_digest": _digest(portrait),
            "requirement_graph_id": requirement_graph["graph_id"],
            "requirement_graph_digest": requirement_graph["graph_digest"],
            "gap_plan_id": gap_plan["plan_id"],
            "gap_plan_digest": gap_plan["plan_digest"],
        },
        "hypothesis_batch_digest": _digest(hypothesis_batch),
        "ranking_policy": {
            "policy_id": policy["policy_id"],
            **_RANKING_WEIGHTS,
        },
        "ranked_candidates": [_ranking_projection(row) for row in ranked],
        "entries": entries,
        "budget": {
            "maximum_hypotheses": policy["maximum_hypotheses"],
            "selected_hypotheses": len(selected),
            "max_total_gpu_hours": policy["max_total_gpu_hours"],
            "estimated_total_gpu_hours": estimated_total,
            "minimum_replications": policy["minimum_replications"],
        },
        "next_action": next_action,
        "authority": {
            "module_manufacturing_authority": False,
            "gpu_authority": False,
            "evaluator_authority": False,
            "promotion_authority": False,
        },
        "side_effects": {
            "source_mutated": False,
            "gpu_execution_started": False,
            "evaluator_modified": False,
        },
        "claim_boundary": (
            "This portfolio ranks bounded experiments and reserves no resources. "
            "Only later admission may authorize module manufacture or GPU execution."
        ),
    }
    portfolio: dict[str, object] = {
        **body,
        "portfolio_id": _stable_id("experiment-portfolio", body),
    }
    portfolio["portfolio_digest"] = _digest(portfolio)
    validate_experiment_portfolio(portfolio, root=root)
    return portfolio


def validate_hypothesis_batch(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate the only LLM-shaped input accepted by the portfolio compiler."""

    try:
        reject_runtime_bindings(document)
        validate_document("experiment_hypothesis_batch", document, root=root)
    except (GeometryValidationError, ContractValidationError) as exc:
        raise ExperimentPortfolioError(f"HYPOTHESIS_BATCH_INVALID:{exc}") from exc
    candidates = document.get("candidates")
    assert isinstance(candidates, list)
    mechanism_ids = []
    for row in candidates:
        assert isinstance(row, Mapping)
        mechanism = row["mechanism_contract"]
        assert isinstance(mechanism, Mapping)
        try:
            validate_mechanism_contract(mechanism)
        except GeometryValidationError as exc:
            raise ExperimentPortfolioError(
                f"HYPOTHESIS_MECHANISM_INVALID:{exc}"
            ) from exc
        mechanism_ids.append(str(mechanism["mechanism_id"]))
        for field in (
            "expected_portrait_changes",
            "structural_conditions",
            "behavioral_conditions",
            "discriminates_from",
            "required_module_capabilities",
        ):
            _unique_strings(row[field], f"HYPOTHESIS_{field.upper()}_INVALID")
    if len(mechanism_ids) != len(set(mechanism_ids)):
        raise ExperimentPortfolioError("HYPOTHESIS_MECHANISM_DUPLICATE")
    known = set(mechanism_ids)
    for row in candidates:
        mechanism = row["mechanism_contract"]
        assert isinstance(mechanism, Mapping)
        mechanism_id = str(mechanism["mechanism_id"])
        discriminates = set(str(value) for value in row["discriminates_from"])
        if mechanism_id in discriminates:
            raise ExperimentPortfolioError("HYPOTHESIS_SELF_DISCRIMINATION")
        unknown = sorted(discriminates - known)
        if unknown:
            raise ExperimentPortfolioError(
                f"HYPOTHESIS_DISCRIMINATION_UNKNOWN:{unknown[0]}"
            )


def validate_experiment_portfolio(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        reject_runtime_bindings(document)
        validate_document("experiment_portfolio", document, root=root)
    except (GeometryValidationError, ContractValidationError) as exc:
        raise ExperimentPortfolioError(f"EXPERIMENT_PORTFOLIO_INVALID:{exc}") from exc
    body = dict(document)
    received_digest = body.pop("portfolio_digest", None)
    if received_digest != _digest(body):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_DIGEST_MISMATCH")
    identity = dict(body)
    received_id = identity.pop("portfolio_id", None)
    if received_id != _stable_id("experiment-portfolio", identity):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_ID_MISMATCH")
    ranked = document["ranked_candidates"]
    entries = document["entries"]
    budget = document["budget"]
    assert isinstance(ranked, list) and isinstance(entries, list)
    assert isinstance(budget, Mapping)
    candidate_ids = [str(row["candidate_id"]) for row in ranked]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_CANDIDATE_DUPLICATE")
    if [int(row["rank"]) for row in ranked] != list(range(1, len(ranked) + 1)):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_RANK_INVALID")
    selected = {str(row["candidate_id"]) for row in ranked if row["selected"]}
    entry_ids = [str(row["entry_id"]) for row in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_ENTRY_DUPLICATE")
    known_entries = set(entry_ids)
    roles_by_candidate: dict[str, set[str]] = {}
    baseline_count = 0
    for row in entries:
        role = str(row["role"])
        if role not in _ENTRY_ROLES:
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_ROLE_INVALID")
        if any(str(value) not in known_entries for value in row["dependencies"]):
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_DEPENDENCY_UNKNOWN")
        replication = row["replication"]
        cost = row["cost"]
        assert isinstance(replication, Mapping) and isinstance(cost, Mapping)
        if int(replication["count"]) != len(replication["seeds"]):
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_REPLICATION_INVALID")
        expected_cost = round(
            float(cost["per_replication_gpu_hours"]) * int(replication["count"]),
            8,
        )
        if not math.isclose(
            float(cost["total_gpu_hours"]), expected_cost, abs_tol=1e-8
        ):
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_COST_INVALID")
        candidate_id = row.get("candidate_id")
        if role == "baseline":
            baseline_count += 1
            if candidate_id is not None or row.get("mechanism_id") is not None:
                raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_BASELINE_INVALID")
        else:
            if str(candidate_id) not in selected:
                raise ExperimentPortfolioError(
                    "EXPERIMENT_PORTFOLIO_UNSELECTED_ENTRY"
                )
            roles_by_candidate.setdefault(str(candidate_id), set()).add(role)
    if selected:
        if baseline_count != 1:
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_BASELINE_REQUIRED")
        for candidate_id in selected:
            roles = roles_by_candidate.get(candidate_id, set())
            if not {"mechanism_test", "negative_control", "ablation"}.issubset(roles):
                raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_CONTROL_REQUIRED")
    elif entries or baseline_count:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_BLOCKED_ENTRIES_INVALID")
    estimated_total = round(
        sum(float(row["cost"]["total_gpu_hours"]) for row in entries),
        8,
    )
    if not math.isclose(
        float(budget["estimated_total_gpu_hours"]), estimated_total, abs_tol=1e-8
    ):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_BUDGET_INVALID")
    expected_state = "ready_for_resource_admission" if selected else "blocked_budget"
    if document["state"] != expected_state:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_STATE_INVALID")
    if int(budget["selected_hypotheses"]) != len(selected):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_SELECTION_COUNT_INVALID")
    if estimated_total > float(budget["max_total_gpu_hours"]) + 1e-8:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_BUDGET_EXCEEDED")
    if not selected and document["next_action"] != "stop_budget":
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_NEXT_ACTION_INVALID")


def _validate_inputs(
    *,
    goal_ir: Mapping[str, object],
    portrait: Mapping[str, object],
    requirement_graph: Mapping[str, object],
    gap_plan: Mapping[str, object],
    hypothesis_batch: Mapping[str, object],
    root: Path | None,
) -> None:
    try:
        validate_goal_ir(goal_ir, root=root)
        validate_model_portrait(portrait, root=root)
        validate_capability_requirement_graph(requirement_graph, root=root)
        validate_gap_plan_against_requirement_graph(
            gap_plan,
            requirement_graph,
            root=root,
        )
    except (CapabilityGapPlannerError, ModelPortraitError) as exc:
        raise ExperimentPortfolioError(f"EXPERIMENT_PORTFOLIO_BINDING_INVALID:{exc}") from exc
    validate_hypothesis_batch(hypothesis_batch, root=root)
    if (
        requirement_graph["goal_ir_id"] != goal_ir["goal_ir_id"]
        or requirement_graph["goal_ir_digest"] != _digest(goal_ir)
    ):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_GOAL_BINDING_MISMATCH")
    if (
        requirement_graph["portrait_id"] != portrait["portrait_id"]
        or requirement_graph["portrait_digest"] != _digest(portrait)
    ):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_PORTRAIT_BINDING_MISMATCH")
    if gap_plan["state"] not in {"ready_for_portfolio", "requires_manufacturing"}:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_GAP_STATE_INVALID")


def _normalize_policy(
    *,
    policy_id: str,
    maximum_hypotheses: int,
    max_total_gpu_hours: float,
    minimum_replications: int,
    baseline_gpu_hours: float,
    control_cost_fraction: float,
    ablation_cost_fraction: float,
    protected_metrics: Sequence[str],
    heldout_protocol: str,
    required_artifact_classes: Sequence[str],
) -> dict[str, object]:
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_POLICY_ID_INVALID")
    if not isinstance(maximum_hypotheses, int) or isinstance(maximum_hypotheses, bool):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_MAXIMUM_INVALID")
    if not isinstance(minimum_replications, int) or isinstance(minimum_replications, bool):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_REPLICATION_INVALID")
    if maximum_hypotheses < 1 or minimum_replications < 1:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_POLICY_INVALID")
    numeric = (
        max_total_gpu_hours,
        baseline_gpu_hours,
        control_cost_fraction,
        ablation_cost_fraction,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_POLICY_INVALID")
    if (
        max_total_gpu_hours <= 0
        or baseline_gpu_hours < 0
        or not 0 < control_cost_fraction <= 1
        or not 0 < ablation_cost_fraction <= 1
    ):
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_POLICY_INVALID")
    return {
        "policy_id": policy_id.strip(),
        "maximum_hypotheses": maximum_hypotheses,
        "max_total_gpu_hours": float(max_total_gpu_hours),
        "minimum_replications": minimum_replications,
        "baseline_gpu_hours": float(baseline_gpu_hours),
        "control_cost_fraction": float(control_cost_fraction),
        "ablation_cost_fraction": float(ablation_cost_fraction),
        "protected_metrics": _unique_strings(
            protected_metrics,
            "EXPERIMENT_PORTFOLIO_METRIC_INVALID",
            nonempty=True,
        ),
        "heldout_protocol": _required_text(
            heldout_protocol,
            "EXPERIMENT_PORTFOLIO_HELDOUT_INVALID",
        ),
        "required_artifact_classes": _unique_strings(
            required_artifact_classes,
            "EXPERIMENT_PORTFOLIO_ARTIFACT_INVALID",
            nonempty=True,
        ),
    }


def _normalize_candidates(
    batch: Mapping[str, object],
    *,
    requirement_graph: Mapping[str, object],
) -> list[dict[str, object]]:
    graph_capabilities = {str(row["capability"]) for row in requirement_graph["nodes"]}
    rows = []
    signatures: set[tuple[object, ...]] = set()
    for raw in batch["candidates"]:
        mechanism = dict(raw["mechanism_contract"])
        expected_changes = sorted(str(value) for value in raw["expected_portrait_changes"])
        required_modules = sorted(
            str(value) for value in raw["required_module_capabilities"]
        )
        unknown_modules = sorted(set(required_modules) - graph_capabilities)
        if unknown_modules:
            raise ExperimentPortfolioError(
                f"EXPERIMENT_PORTFOLIO_MODULE_CAPABILITY_UNKNOWN:{unknown_modules[0]}"
            )
        signature = (
            mechanism["intervention_semantics"],
            tuple(expected_changes),
            tuple(required_modules),
        )
        if signature in signatures:
            raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_COSMETIC_VARIANT")
        signatures.add(signature)
        identity: dict[str, object] = {
            "mechanism_id": mechanism["mechanism_id"],
            "expected_portrait_changes": expected_changes,
            "structural_conditions": sorted(
                str(value) for value in raw["structural_conditions"]
            ),
            "behavioral_conditions": sorted(
                str(value) for value in raw["behavioral_conditions"]
            ),
            "discriminates_from": sorted(
                str(value) for value in raw["discriminates_from"]
            ),
            "required_module_capabilities": required_modules,
        }
        candidate_digest = _digest(identity)
        rows.append(
            {
                **identity,
                "candidate_id": "experiment-candidate-" + candidate_digest[:24],
                "candidate_digest": candidate_digest,
                "mechanism": mechanism,
                "information_gain": float(raw["information_gain"]),
                "uncertainty": float(raw["uncertainty"]),
                "estimated_screen_gpu_hours": float(
                    raw["estimated_screen_gpu_hours"]
                ),
                "selection_reason": str(raw["selection_reason"]),
            }
        )
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _rank_and_select(
    candidates: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
) -> list[dict[str, object]]:
    remaining = [dict(row) for row in candidates]
    ranked: list[dict[str, object]] = []
    seen_changes: set[str] = set()
    spent = 0.0
    selected_count = 0
    baseline_cost = float(policy["baseline_gpu_hours"]) * int(
        policy["minimum_replications"]
    )
    maximum_cost = float(policy["max_total_gpu_hours"])
    while remaining:
        scored = []
        for row in remaining:
            expected_changes = set(str(value) for value in row["expected_portrait_changes"])
            diversity = 1.0 if expected_changes - seen_changes else 0.0
            competition = len(row["discriminates_from"]) / max(1, len(candidates) - 1)
            competition = min(1.0, float(competition))
            cost = min(
                1.0,
                float(row["estimated_screen_gpu_hours"]) / maximum_cost,
            )
            components = {
                "information_gain": float(row["information_gain"]),
                "uncertainty": float(row["uncertainty"]),
                "competition": competition,
                "diversity": diversity,
                "cost": cost,
            }
            score = (
                _RANKING_WEIGHTS["information_gain_weight"]
                * components["information_gain"]
                + _RANKING_WEIGHTS["uncertainty_weight"] * components["uncertainty"]
                + _RANKING_WEIGHTS["competition_weight"] * components["competition"]
                + _RANKING_WEIGHTS["diversity_weight"] * components["diversity"]
                - _RANKING_WEIGHTS["cost_weight"] * components["cost"]
            )
            scored.append((round(score, 8), str(row["candidate_id"]), row, components))
        score, _, chosen, components = max(scored, key=lambda value: (value[0], value[1]))
        remaining.remove(chosen)
        bundle_cost = _candidate_bundle_cost(chosen, policy=policy)
        marginal = bundle_cost + (baseline_cost if selected_count == 0 else 0.0)
        selected = (
            selected_count < int(policy["maximum_hypotheses"])
            and spent + marginal <= maximum_cost + 1e-8
        )
        if selected:
            spent += marginal
            selected_count += 1
            selection_reason = (
                f"selected_by_{policy['policy_id']}:score={score:.8f};"
                f"draft={chosen['selection_reason']}"
            )
        elif selected_count >= int(policy["maximum_hypotheses"]):
            selection_reason = "deferred_maximum_hypotheses"
        else:
            selection_reason = "deferred_budget"
        ranked.append(
            {
                **chosen,
                "rank": len(ranked) + 1,
                "selected": selected,
                "score": score,
                "score_components": components,
                "estimated_bundle_gpu_hours": round(bundle_cost, 8),
                "selection_reason": selection_reason,
            }
        )
        seen_changes.update(str(value) for value in chosen["expected_portrait_changes"])
    return ranked


def _candidate_bundle_cost(
    candidate: Mapping[str, object], *, policy: Mapping[str, object]
) -> float:
    replication = int(policy["minimum_replications"])
    screen = float(candidate["estimated_screen_gpu_hours"])
    mechanism = candidate["mechanism"]
    assert isinstance(mechanism, Mapping)
    ablation_count = len(mechanism["required_ablations"])
    per_replication = screen * (
        1.0
        + float(policy["control_cost_fraction"])
        + float(policy["ablation_cost_fraction"]) * ablation_count
    )
    return round(per_replication * replication, 8)


def _portfolio_entries(
    selected: Sequence[Mapping[str, object]],
    *,
    goal_ir: Mapping[str, object],
    evaluator: Mapping[str, object],
    policy: Mapping[str, object],
) -> list[dict[str, object]]:
    if not selected:
        return []
    baseline = _entry(
        candidate=None,
        role="baseline",
        hypothesis="The frozen unmodified model defines the comparison baseline.",
        falsification=(
            "The baseline receipt is invalid if protected metrics cannot be reproduced "
            "under the frozen held-out evaluator."
        ),
        expected_changes=(),
        structural_conditions=(),
        behavioral_conditions=(),
        intervention={"operator": "none", "semantic_target": "unmodified_model"},
        dependencies=(),
        required_modules=(),
        per_replication_gpu_hours=float(policy["baseline_gpu_hours"]),
        evaluator=evaluator,
        policy=policy,
        unresolved_risks=("baseline_instability",),
        selection_reason="shared_baseline_required_for_all_selected_mechanisms",
    )
    entries = [baseline]
    baseline_id = str(baseline["entry_id"])
    for candidate in selected:
        mechanism = candidate["mechanism"]
        assert isinstance(mechanism, Mapping)
        common = {
            "candidate": candidate,
            "expected_changes": candidate["expected_portrait_changes"],
            "structural_conditions": candidate["structural_conditions"],
            "behavioral_conditions": candidate["behavioral_conditions"],
            "dependencies": (baseline_id,),
            "required_modules": candidate["required_module_capabilities"],
            "evaluator": evaluator,
            "policy": policy,
        }
        screen = float(candidate["estimated_screen_gpu_hours"])
        entries.append(
            _entry(
                **common,
                role="mechanism_test",
                hypothesis=str(mechanism["causal_claim"]),
                falsification=str(mechanism["falsification_criterion"]),
                intervention={
                    "operator": "mechanism_intervention",
                    "semantic_target": mechanism["intervention_semantics"],
                },
                per_replication_gpu_hours=screen,
                unresolved_risks=mechanism["known_anti_conditions"],
                selection_reason=str(candidate["selection_reason"]),
            )
        )
        entries.append(
            _entry(
                **common,
                role="negative_control",
                hypothesis=(
                    "A no-op control must not reproduce the declared portrait change for "
                    + str(mechanism["mechanism_id"])
                    + "."
                ),
                falsification=(
                    "The mechanism is not discriminated if the no-op control matches the "
                    "mechanism effect on protected metrics."
                ),
                intervention={
                    "operator": "no_op_control",
                    "semantic_target": mechanism["intervention_semantics"],
                },
                per_replication_gpu_hours=(
                    screen * float(policy["control_cost_fraction"])
                ),
                unresolved_risks=("control_leakage",),
                selection_reason="automatic_negative_control",
            )
        )
        for ablation in mechanism["required_ablations"]:
            entries.append(
                _entry(
                    **common,
                    role="ablation",
                    hypothesis=(
                        "Removing the declared component should eliminate or reduce the "
                        "mechanism-specific portrait change."
                    ),
                    falsification=(
                        "The causal claim is not isolated if the ablation preserves the "
                        "full protected-metric effect."
                    ),
                    intervention={
                        "operator": "remove_component",
                        "semantic_target": str(ablation),
                    },
                    per_replication_gpu_hours=(
                        screen * float(policy["ablation_cost_fraction"])
                    ),
                    unresolved_risks=("ablation_non_specificity",),
                    selection_reason="automatic_discriminating_ablation",
                )
            )
    entries.sort(key=lambda row: str(row["entry_id"]))
    return entries


def _entry(
    *,
    candidate: Mapping[str, object] | None,
    role: str,
    hypothesis: str,
    falsification: str,
    expected_changes: Sequence[object],
    structural_conditions: Sequence[object],
    behavioral_conditions: Sequence[object],
    intervention: Mapping[str, object],
    dependencies: Sequence[str],
    required_modules: Sequence[object],
    per_replication_gpu_hours: float,
    evaluator: Mapping[str, object],
    policy: Mapping[str, object],
    unresolved_risks: Sequence[object],
    selection_reason: str,
) -> dict[str, object]:
    replication_count = int(policy["minimum_replications"])
    identity: dict[str, object] = {
        "candidate_id": candidate["candidate_id"] if candidate is not None else None,
        "mechanism_id": (
            candidate["mechanism_id"] if candidate is not None else None
        ),
        "role": role,
        "hypothesis": hypothesis,
        "falsification_criterion": falsification,
        "expected_portrait_changes": sorted(str(value) for value in expected_changes),
        "applicability": {
            "structural_conditions": sorted(
                str(value) for value in structural_conditions
            ),
            "behavioral_conditions": sorted(
                str(value) for value in behavioral_conditions
            ),
        },
        "intervention": dict(intervention),
        "dependencies": sorted(str(value) for value in dependencies),
        "required_module_capabilities": sorted(
            str(value) for value in required_modules
        ),
        "replication_count": replication_count,
    }
    entry_id = _stable_id("portfolio-entry", identity)
    seeds = [
        int(_digest({"entry_id": entry_id, "replication": index})[:8], 16)
        for index in range(replication_count)
    ]
    return {
        "entry_id": entry_id,
        "candidate_id": identity["candidate_id"],
        "mechanism_id": identity["mechanism_id"],
        "role": role,
        "hypothesis": hypothesis,
        "falsification_criterion": falsification,
        "expected_portrait_changes": identity["expected_portrait_changes"],
        "applicability": identity["applicability"],
        "intervention": dict(intervention),
        "dependencies": identity["dependencies"],
        "required_module_capabilities": identity["required_module_capabilities"],
        "replication": {"count": replication_count, "seeds": seeds},
        "cost": {
            "per_replication_gpu_hours": round(per_replication_gpu_hours, 8),
            "total_gpu_hours": round(
                per_replication_gpu_hours * replication_count,
                8,
            ),
        },
        "protected_metrics": list(policy["protected_metrics"]),
        "evaluator": dict(evaluator),
        "artifact_policy": {
            "required_artifact_classes": list(
                policy["required_artifact_classes"]
            ),
            "archive_policy": "archive_all_terminal",
            "cleanup_policy": "after_content_addressed_receipt",
        },
        "unresolved_risks": sorted(str(value) for value in unresolved_risks),
        "selection_reason": selection_reason,
    }


def _frozen_evaluator(
    portrait: Mapping[str, object], *, heldout_protocol: str
) -> dict[str, object]:
    structural = portrait["structural_profile"]
    assert isinstance(structural, Mapping)
    evaluator = structural["evaluator"]
    assert isinstance(evaluator, Mapping)
    if evaluator.get("state") != "ready":
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_EVALUATOR_NOT_READY")
    result = {
        "evaluator_id": _required_text(
            evaluator.get("evaluator_id"),
            "EXPERIMENT_PORTFOLIO_EVALUATOR_INVALID",
        ),
        "contract_digest": _required_text(
            evaluator.get("contract_digest"),
            "EXPERIMENT_PORTFOLIO_EVALUATOR_INVALID",
        ),
        "verifier": _required_text(
            evaluator.get("verifier"),
            "EXPERIMENT_PORTFOLIO_EVALUATOR_INVALID",
        ),
        "heldout_protocol": heldout_protocol,
    }
    return result


def _ranking_projection(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"],
        "candidate_digest": row["candidate_digest"],
        "mechanism_id": row["mechanism_id"],
        "rank": row["rank"],
        "selected": row["selected"],
        "score": row["score"],
        "score_components": dict(row["score_components"]),
        "estimated_bundle_gpu_hours": row["estimated_bundle_gpu_hours"],
        "selection_reason": row["selection_reason"],
    }


def _unique_strings(
    values: object,
    code: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ExperimentPortfolioError(code)
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ExperimentPortfolioError(code)
        result.append(value.strip())
    if nonempty and not result:
        raise ExperimentPortfolioError(code)
    if len(result) != len(set(result)):
        raise ExperimentPortfolioError(code)
    return sorted(result)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentPortfolioError(code)
    return value.strip()


def _stable_id(prefix: str, value: Mapping[str, object]) -> str:
    return f"{prefix}-{_digest(value)[:24]}"


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentPortfolioError("EXPERIMENT_PORTFOLIO_CANONICAL_INVALID") from exc
    return hashlib.sha256(encoded).hexdigest()
