"""Compile user-facing research modes into bounded execution policy.

The mode is a routing policy, not scientific authority. It selects candidate
sources and stage ordering while preserving the portrait-readiness gate and
the shared frozen verifier for every candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class ResearchModeError(ValueError):
    """A research mode or its evidence prerequisites are invalid."""


_ALIASES = {
    "quick-start": "quick_start",
    "quick_start": "quick_start",
    "causal-discovery": "causal_discovery",
    "causal_discovery": "causal_discovery",
    "hybrid": "hybrid",
}

_MODE_DEFINITIONS: dict[str, dict[str, object]] = {
    "quick_start": {
        "label": "Quick start",
        "description": "用本地证据和联网论文完成冷启动，再在目标模型上逐个验证迁移候选。",
        "candidate_sources": ["local_verified_experience", "online_literature"],
        "budget_allocation": {
            "baseline_and_portrait": 0.20,
            "source_grounded_candidates": 0.45,
            "screen": 0.20,
            "confirm_reserve": 0.15,
        },
    },
    "causal_discovery": {
        "label": "Causal discovery",
        "description": "根据目标模型响应几何与反例扩展诊断基底，再构造可验证的干预。",
        "candidate_sources": ["target_response_geometry", "counterexample_guided_basis"],
        "budget_allocation": {
            "baseline_and_portrait": 0.20,
            "causal_probe_expansion": 0.35,
            "intervention_construction": 0.25,
            "confirm_reserve": 0.20,
        },
    },
    "hybrid": {
        "label": "Hybrid",
        "description": "先用外部知识启动，同时让目标证据逐步启用因果发现与联合干预合成。",
        "candidate_sources": [
            "local_verified_experience",
            "online_literature",
            "target_response_geometry",
            "counterexample_guided_basis",
        ],
        "budget_allocation": {
            "baseline_and_portrait": 0.20,
            "source_grounded_candidates": 0.25,
            "causal_probe_expansion": 0.20,
            "joint_screen": 0.20,
            "confirm_reserve": 0.15,
        },
    },
}


def research_mode_catalog() -> list[dict[str, object]]:
    """Return stable user-facing mode metadata without execution bindings."""

    return [
        {
            "mode": mode,
            "label": definition["label"],
            "description": definition["description"],
            "candidate_sources": list(definition["candidate_sources"]),
        }
        for mode, definition in _MODE_DEFINITIONS.items()
    ]


def normalize_research_mode(value: object) -> str:
    if not isinstance(value, str) or value not in _ALIASES:
        raise ResearchModeError("RESEARCH_MODE_INVALID")
    return _ALIASES[value]


def compile_research_mode_plan(
    *,
    mode: object,
    goal: str,
    execution: Mapping[str, object],
    evidence_context: Mapping[str, object] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Compile one deterministic mode plan against current target evidence."""

    normalized = normalize_research_mode(mode)
    if not isinstance(goal, str) or not goal.strip():
        raise ResearchModeError("RESEARCH_MODE_GOAL_REQUIRED")
    if execution.get("kind") not in {"pipeline", "evolution"}:
        raise ResearchModeError("RESEARCH_MODE_EXECUTION_KIND_UNSUPPORTED")
    context = _evidence_context(evidence_context)
    has_probe = _nonempty(execution.get("probe_contract"))
    has_cpbe_request = _nonempty(execution.get("cpbe_request"))
    has_cpbe_history = _nonempty(execution.get("cpbe_history"))
    if has_cpbe_request != has_cpbe_history:
        raise ResearchModeError("RESEARCH_MODE_CPBE_INPUT_PAIR_REQUIRED")
    has_cpbe = has_cpbe_request and has_cpbe_history
    causal_ready = has_probe and has_cpbe

    blockers: list[str] = []
    if normalized == "causal_discovery":
        if not has_probe:
            blockers.append("probe_contract_required")
        if not has_cpbe_request:
            blockers.append("cpbe_request_required")
        if not has_cpbe_history:
            blockers.append("cpbe_history_required")

    source_states = []
    for source in _MODE_DEFINITIONS[normalized]["candidate_sources"]:
        causal_source = source in {
            "target_response_geometry",
            "counterexample_guided_basis",
        }
        source_states.append(
            {
                "source": source,
                "state": "active" if not causal_source or causal_ready else "deferred",
                "reason": (
                    None
                    if not causal_source or causal_ready
                    else "Causal sources require a diagnostic probe contract and frozen CPBE request/history."
                ),
            }
        )

    stages = _stages(
        mode=normalized,
        context=context,
        causal_ready=causal_ready,
        blocked=bool(blockers),
    )
    state = "blocked" if blockers else (
        "ready_with_deferred_discovery"
        if normalized == "hybrid" and not causal_ready
        else "ready"
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-research-mode-plan",
        "mode": normalized,
        "label": _MODE_DEFINITIONS[normalized]["label"],
        "goal": goal.strip(),
        "state": state,
        "candidate_sources": source_states,
        "stages": stages,
        "budget_allocation": dict(
            _MODE_DEFINITIONS[normalized]["budget_allocation"]
        ),
        "evidence_context": context,
        "blockers": blockers,
        "transition_policy": {
            "portrait_gate": "ready_for_gap_planning",
            "causal_activation": "probe_contract_and_frozen_cpbe_inputs",
            "candidate_admission": "shared_screen_then_independent_confirm",
            "promotion_authority": "frozen_verifier_only",
        },
        "claim_boundary": (
            "The mode selects candidate sources and stage order only. Literature is a prior, "
            "diagnostic probes are not repairs, and only an independent frozen confirmation "
            "verdict can establish a target-model improvement."
        ),
    }
    body["plan_id"] = "research-mode-plan-" + _digest(body)[:24]
    try:
        validate_document("research_mode_plan", body, root=root)
    except ContractValidationError as exc:
        raise ResearchModeError(f"RESEARCH_MODE_PLAN_INVALID:{exc}") from exc
    return body


def apply_research_mode_to_execution(
    execution: Mapping[str, object],
    *,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Bind a validated mode plan to an execution without granting authority."""

    updated = dict(execution)
    updated["research_mode"] = str(plan["mode"])
    updated["research_mode_plan_id"] = str(plan["plan_id"])
    if plan["mode"] in {"quick_start", "hybrid"} and not _nonempty(
        updated.get("literature_query")
    ):
        updated["literature_query"] = str(plan["goal"])
    return updated


def _evidence_context(value: Mapping[str, object] | None) -> dict[str, object]:
    supplied = dict(value or {})
    allowed = {
        "portrait_readiness",
        "baseline_replicates",
        "admitted_probe_count",
        "stable_collision_count",
        "unexplained_residual_count",
    }
    extras = sorted(set(supplied) - allowed)
    if extras:
        raise ResearchModeError("RESEARCH_MODE_EVIDENCE_UNKNOWN:" + ",".join(extras))
    readiness = supplied.get("portrait_readiness", "not_built")
    if readiness not in {
        "not_built",
        "requires_static_onboarding",
        "requires_probe_coverage",
        "ready_for_gap_planning",
        "conflicting_evidence",
        "stale_portrait",
    }:
        raise ResearchModeError("RESEARCH_MODE_PORTRAIT_STATE_INVALID")
    context: dict[str, object] = {"portrait_readiness": readiness}
    for field in (
        "baseline_replicates",
        "admitted_probe_count",
        "stable_collision_count",
        "unexplained_residual_count",
    ):
        item = supplied.get(field, 0)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ResearchModeError(f"RESEARCH_MODE_EVIDENCE_COUNT_INVALID:{field}")
        context[field] = item
    return context


def _stages(
    *,
    mode: str,
    context: Mapping[str, object],
    causal_ready: bool,
    blocked: bool,
) -> list[dict[str, object]]:
    portrait_ready = context["portrait_readiness"] == "ready_for_gap_planning"
    definitions = [
        ("onboarding", "active", "Read-only model and evaluator conformance"),
        ("baseline_and_portrait", "pending", "Target baseline and behavioral portrait"),
    ]
    if mode in {"quick_start", "hybrid"}:
        definitions.append(
            ("external_knowledge_bootstrap", "pending", "Local evidence and online literature")
        )
    if mode in {"causal_discovery", "hybrid"}:
        definitions.append(
            (
                "causal_basis_expansion",
                "pending" if causal_ready else ("blocked" if blocked else "deferred"),
                "IRG collision and CPBE diagnostic basis expansion",
            )
        )
    definitions.extend(
        [
            (
                "joint_hypothesis_synthesis",
                "pending" if portrait_ready else "gated",
                "Mechanism hypotheses become bounded intervention work orders",
            ),
            ("screen", "gated", "Cheap target-model falsification"),
            ("independent_confirm", "gated", "Held-out multi-replication confirmation"),
            ("knowledge_deposition", "gated", "Verified evidence and negative boundaries"),
        ]
    )
    return [
        {"stage": stage, "state": state, "purpose": purpose, "order": index}
        for index, (stage, state, purpose) in enumerate(definitions, start=1)
    ]


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
