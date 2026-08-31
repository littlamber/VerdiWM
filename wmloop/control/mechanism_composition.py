"""Compile reusable mechanisms into executable four-cell compositions.

This module is the bridge between semantic method memory and the generic
primitive renderer/executor.  A mechanism embodiment binds once to a registered
primitive and its parameters.  Every later composition and ablation is derived
from that binding; callers do not provide a bespoke runner for A or B.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any

from wmloop.geometry.mechanism_relations import settle_mechanism_relation
from wmloop.geometry.memory import EffectRecord
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveValidationError


class MechanismCompositionError(ValueError):
    """A semantic mechanism could not be compiled into executable cells."""


_CELL_KINDS = ("baseline", "source_only", "target_only", "combined")


def discover_mechanism_compositions(
    *,
    registry: PrimitiveRegistry,
    effect_records: Sequence[EffectRecord],
    executable_bindings: Sequence[Mapping[str, object]],
    existing_relations: Sequence[object] = (),
    maximum_candidates: int = 8,
) -> tuple[dict[str, object], ...]:
    """Discover compatible A+B candidates from settled method experience.

    Discovery is deterministic and evidence-first: only confirmed effects are
    considered, both mechanisms must have executable primitive bindings, their
    contexts must be comparable, and the frozen registry must admit the pair.
    """

    if maximum_candidates < 1:
        raise MechanismCompositionError("MECHANISM_DISCOVERY_LIMIT_INVALID")
    bindings = [_normalize_binding(registry, row) for row in executable_bindings]
    by_primitive: dict[str, list[EffectRecord]] = {}
    for record in effect_records:
        if not isinstance(record, EffectRecord):
            raise MechanismCompositionError("MECHANISM_DISCOVERY_EFFECT_INVALID")
        if record.status == "confirmed" and all(record.validity_gates.values()):
            by_primitive.setdefault(record.primitive, []).append(record)
    known_pairs = set()
    for relation in existing_relations:
        source_id = getattr(relation, "source_mechanism_id", None)
        target_id = getattr(relation, "target_mechanism_id", None)
        if isinstance(relation, Mapping):
            source_id = relation.get("source_mechanism_id")
            target_id = relation.get("target_mechanism_id")
        if source_id and target_id:
            known_pairs.add(frozenset((str(source_id), str(target_id))))
    candidates = []
    for source, target in combinations(bindings, 2):
        pair = frozenset((str(source["mechanism_id"]), str(target["mechanism_id"])))
        if pair in known_pairs:
            continue
        source_effects = by_primitive.get(str(source["primitive"]), [])
        target_effects = by_primitive.get(str(target["primitive"]), [])
        comparable = [
            (left, right)
            for left in source_effects
            for right in target_effects
            if _comparable_effect_context(left, right)
        ]
        if not comparable:
            continue
        try:
            plan = compile_mechanism_composition(registry=registry, source=source, target=target)
        except MechanismCompositionError:
            continue
        left, right = max(comparable, key=lambda pair: pair[0].lower_bound + pair[1].lower_bound)
        source_manifest = registry.manifest(str(source["primitive"]))
        target_manifest = registry.manifest(str(target["primitive"]))
        family_diversity = source_manifest.family != target_manifest.family
        hook_diversity = set(source_manifest.hooks) != set(target_manifest.hooks)
        score = (
            max(0.0, left.lower_bound)
            + max(0.0, right.lower_bound)
            + (0.25 if family_diversity else 0.0)
            + (0.15 if hook_diversity else 0.0)
            - left.standard_error
            - right.standard_error
        )
        candidates.append({
            "candidate_id": "composition-candidate-" + _digest({"plan": plan["composition_id"], "evidence": [left.record_id, right.record_id]})[:24],
            "source_mechanism_id": source["mechanism_id"],
            "target_mechanism_id": target["mechanism_id"],
            "score": round(score, 8),
            "evidence_record_ids": [left.record_id, right.record_id],
            "rationale": {
                "comparable_context": True,
                "registry_compatible": True,
                "family_diversity": family_diversity,
                "hook_diversity": hook_diversity,
            },
            "plan": plan,
            "claim_boundary": "Ranking-only composition candidate; four-cell execution and verifier settlement are required.",
        })
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    return tuple(candidates[:maximum_candidates])


def discover_from_memory(
    *,
    registry: PrimitiveRegistry,
    effect_records: Sequence[EffectRecord],
    embodiments: Sequence[Mapping[str, object]],
    existing_relations: Sequence[object] = (),
    maximum_candidates: int = 8,
) -> tuple[dict[str, object], ...]:
    """Discover compositions directly from deposited method embodiments."""

    bindings = [binding_from_embodiment(registry=registry, embodiment=embodiment) for embodiment in embodiments]
    return discover_mechanism_compositions(
        registry=registry,
        effect_records=effect_records,
        executable_bindings=bindings,
        existing_relations=existing_relations,
        maximum_candidates=maximum_candidates,
    )


def bind_executable_mechanism(
    *,
    registry: PrimitiveRegistry,
    mechanism_id: str,
    primitive: str,
    params: Mapping[str, Any],
    implementation_revision: str,
) -> dict[str, object]:
    """Bind one semantic mechanism to the frozen primitive registry."""

    if not mechanism_id or not implementation_revision:
        raise MechanismCompositionError("MECHANISM_EXECUTABLE_BINDING_INVALID")
    try:
        selection = registry.validate_selection(primitive, params)
    except PrimitiveValidationError as exc:
        raise MechanismCompositionError(f"MECHANISM_EXECUTABLE_BINDING_INVALID:{exc}") from exc
    return {
        "mechanism_id": mechanism_id,
        "primitive": selection.name,
        "params": dict(selection.params),
        "implementation_revision": implementation_revision,
        "hooks": list(selection.hooks),
        "registry_digest": registry.digest(),
    }


def binding_from_embodiment(
    *,
    registry: PrimitiveRegistry,
    embodiment: Mapping[str, object],
) -> dict[str, object]:
    """Extract and validate the executable binding carried by an embodiment."""

    executable = embodiment.get("executable_binding")
    if not isinstance(executable, Mapping):
        raise MechanismCompositionError("MECHANISM_EMBODIMENT_EXECUTABLE_BINDING_MISSING")
    return bind_executable_mechanism(
        registry=registry,
        mechanism_id=str(embodiment.get("mechanism_id") or ""),
        primitive=str(executable.get("primitive") or ""),
        params=executable.get("params") if isinstance(executable.get("params"), Mapping) else {},
        implementation_revision=str(executable.get("implementation_revision") or ""),
    )


def compile_mechanism_composition(
    *,
    registry: PrimitiveRegistry,
    source: Mapping[str, object],
    target: Mapping[str, object],
    composition_operator: str = "parallel",
    relation_id: str | None = None,
    condition_set: Sequence[str] = (),
    anti_conditions: Sequence[str] = (),
) -> dict[str, object]:
    """Generate baseline, A-only, B-only and A+B executable cells."""

    if source.get("mechanism_id") == target.get("mechanism_id"):
        raise MechanismCompositionError("MECHANISM_COMPOSITION_SELF_REFERENCE")
    if composition_operator not in {"parallel", "sequential", "gated", "conditional"}:
        raise MechanismCompositionError("MECHANISM_COMPOSITION_OPERATOR_INVALID")
    source_row = _normalize_binding(registry, source)
    target_row = _normalize_binding(registry, target)
    try:
        registry.validate_combination(
            [(str(source_row["primitive"]), source_row["params"]), (str(target_row["primitive"]), target_row["params"])]
        )
    except PrimitiveValidationError as exc:
        raise MechanismCompositionError(f"MECHANISM_COMPOSITION_INVALID:{exc}") from exc
    identity = {
        "source": source_row,
        "target": target_row,
        "composition_operator": composition_operator,
        "relation_id": relation_id,
        "condition_set": sorted(set(str(value) for value in condition_set)),
        "anti_conditions": sorted(set(str(value) for value in anti_conditions)),
    }
    composition_id = "composition-plan-" + _digest(identity)[:24]
    cells = []
    for kind, interventions in (
        ("baseline", []),
        ("source_only", [source_row]),
        ("target_only", [target_row]),
        ("combined", [source_row, target_row]),
    ):
        cell_identity = {"composition_id": composition_id, "kind": kind, "interventions": interventions}
        cells.append({
            "cell_id": "composition-cell-" + _digest(cell_identity)[:24],
            "kind": kind,
            "interventions": interventions,
            "execution_protocol": "verdiwm-unified-primitive-executor-v1",
        })
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-mechanism-composition-plan",
        "composition_id": composition_id,
        "relation_id": relation_id,
        "composition_operator": composition_operator,
        "source": source_row,
        "target": target_row,
        "condition_set": identity["condition_set"],
        "anti_conditions": identity["anti_conditions"],
        "cells": cells,
        "registry_digest": registry.digest(),
        "claim_boundary": "Executable experiment plan only; no mechanism relation is promoted until all four settled effects pass the verifier.",
    }
    validate_composition_plan(document, registry=registry)
    return document


def validate_composition_plan(
    plan: Mapping[str, object], *, registry: PrimitiveRegistry | None = None
) -> None:
    """Fail closed on malformed or stale generated composition plans."""

    if plan.get("artifact_type") != "verdiwm-mechanism-composition-plan":
        raise MechanismCompositionError("MECHANISM_COMPOSITION_ARTIFACT_INVALID")
    cells = plan.get("cells")
    if not isinstance(cells, list) or tuple(row.get("kind") for row in cells if isinstance(row, Mapping)) != _CELL_KINDS:
        raise MechanismCompositionError("MECHANISM_COMPOSITION_CELL_LADDER_INVALID")
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str) or not isinstance(cell.get("interventions"), list):
            raise MechanismCompositionError("MECHANISM_COMPOSITION_CELL_INVALID")
        for intervention in cell["interventions"]:
            if not isinstance(intervention, Mapping) or not isinstance(intervention.get("primitive"), str) or not isinstance(intervention.get("params"), Mapping):
                raise MechanismCompositionError("MECHANISM_COMPOSITION_INTERVENTION_INVALID")
    if registry is not None:
        source = plan.get("source")
        target = plan.get("target")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise MechanismCompositionError("MECHANISM_COMPOSITION_BINDING_INVALID")
        _normalize_binding(registry, source)
        _normalize_binding(registry, target)
        registry.validate_combination(
            [(str(source["primitive"]), source["params"]), (str(target["primitive"]), target["params"])]
        )


def execute_mechanism_composition(
    *,
    plan: Mapping[str, object],
    executor: Callable[[Mapping[str, object]], EffectRecord],
    source_mechanism_id: str | None = None,
    target_mechanism_id: str | None = None,
    required_ablations: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    synergy_threshold: float = 0.0,
) -> dict[str, object]:
    """Run all cells through one generic executor and settle the relation."""

    validate_composition_plan(plan)
    cells = plan["cells"]
    assert isinstance(cells, list)
    source_id = source_mechanism_id or str(plan["source"]["mechanism_id"])
    target_id = target_mechanism_id or str(plan["target"]["mechanism_id"])
    records: dict[str, EffectRecord] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise MechanismCompositionError("MECHANISM_COMPOSITION_CELL_INVALID")
        result = executor(cell)
        if not isinstance(result, EffectRecord):
            raise MechanismCompositionError("MECHANISM_COMPOSITION_EXECUTOR_RESULT_INVALID")
        records[str(cell["kind"])] = result
    relation = settle_mechanism_relation(
        baseline=records["baseline"],
        source=records["source_only"],
        target=records["target_only"],
        combined=records["combined"],
        source_mechanism_id=source_id,
        target_mechanism_id=target_id,
        composition_operator=str(plan["composition_operator"]),
        required_ablations=required_ablations or (f"remove:{source_id}", f"remove:{target_id}"),
        evidence_refs=evidence_refs,
        condition_set=tuple(str(value) for value in plan.get("condition_set", [])),
        anti_conditions=tuple(str(value) for value in plan.get("anti_conditions", [])),
        synergy_threshold=synergy_threshold,
    )
    return {"composition_id": plan["composition_id"], "cells": records, "relation": relation}


def write_composition_plan(path: Path, plan: Mapping[str, object]) -> Path:
    """Write a deterministic plan consumable by the unified executor."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def _normalize_binding(registry: PrimitiveRegistry, binding: Mapping[str, object]) -> dict[str, object]:
    try:
        mechanism_id = str(binding["mechanism_id"])
        primitive = str(binding["primitive"])
        params = binding["params"]
        revision = str(binding.get("implementation_revision") or "unknown")
    except (KeyError, TypeError) as exc:
        raise MechanismCompositionError("MECHANISM_EXECUTABLE_BINDING_INVALID") from exc
    if not isinstance(params, Mapping):
        raise MechanismCompositionError("MECHANISM_EXECUTABLE_PARAMS_INVALID")
    return bind_executable_mechanism(
        registry=registry,
        mechanism_id=mechanism_id,
        primitive=primitive,
        params=params,
        implementation_revision=revision,
    )


def _comparable_effect_context(left: EffectRecord, right: EffectRecord) -> bool:
    fields = ("backbone_family", "capability_class", "goal_schema", "outcome_schema", "data_regime", "horizons")
    return all(getattr(left.context, field) == getattr(right.context, field) for field in fields)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
