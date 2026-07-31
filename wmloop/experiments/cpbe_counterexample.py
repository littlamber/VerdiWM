"""Turn a failed CPBE expansion into measured history and a new search round."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments._artifacts import canonical_json, write_bundle


class CPBECounterexampleError(ValueError):
    """The prior round is incomplete or cannot support causal axis credit."""


_DSL_FIELDS = (
    "signal_source",
    "hook_type",
    "spatial_mask",
    "temporal_basis",
    "contrast_operator",
    "aggregation",
)


def build_counterexample_round(
    *,
    request: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    settlement: Mapping[str, Any],
    atlas: Mapping[str, Any],
    probe_id: str,
    next_experiment_id: str,
    gpu_hours: float,
    evidence_refs: Sequence[str],
) -> tuple[dict[str, object], dict[str, object], list[Mapping[str, Any]], dict[str, object]]:
    """Compile one terminal expanded failure into the next deterministic round."""

    if request.get("artifact_type") != "verdiwm-cpbe-request" or request.get("evidence_class") != "live":
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_REQUEST_INVALID")
    if plan.get("artifact_type") != "verdiwm-cpbe-plan" or plan.get("state") != "ready":
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_PLAN_INVALID")
    if settlement.get("artifact_type") != "verdiwm-cpbe-settlement" or settlement.get("state") != "settled":
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_SETTLEMENT_INVALID")
    if not next_experiment_id or not math.isfinite(gpu_hours) or gpu_hours <= 0.0:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_ID_OR_COST_INVALID")

    work_order = _unique_by_probe(plan.get("selected_work_orders"), probe_id, "CPBE_COUNTEREXAMPLE_WORK_ORDER")
    candidate = _unique_by_probe(settlement.get("candidates"), probe_id, "CPBE_COUNTEREXAMPLE_CANDIDATE")
    if candidate.get("state") != "eliminated_expanded" or not candidate.get("terminal"):
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_NOT_EXPANDED_FAILURE")
    stage_receipts = {
        str(receipt.get("stage")): receipt
        for receipt in receipts
        if receipt.get("probe_id") == probe_id
    }
    if set(stage_receipts) != {"static", "offline", "canary", "expanded"}:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_RECEIPTS_INCOMPLETE")
    canary = stage_receipts["canary"]
    expanded = stage_receipts["expanded"]
    if canary.get("passed") is not True or expanded.get("passed") is not False:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_STAGE_PATTERN_INVALID")

    program = _mapping(work_order, "program")
    parent = _unique_parent(request=request, program=program)
    edited_axes = [field for field in _DSL_FIELDS if parent.get(field) != program.get(field)]
    if len(edited_axes) != 1:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_SINGLE_AXIS_REQUIRED")

    canary_metrics = _mapping(canary, "metrics")
    expanded_metrics = _mapping(expanded, "metrics")
    nonredundant = _boolean(canary_metrics, "nonredundant")
    if not nonredundant or _number(canary_metrics, "collision_separation") <= 0.0:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_CANARY_NOT_INFORMATIVE")
    regret = _number(expanded_metrics, "regret_reduction")
    coverage = _number(expanded_metrics, "coverage_gain")
    if regret > 0.0 or coverage > 0.0:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_EXPANDED_GAIN_PRESENT")

    environments = _mapping(atlas, "environments")
    expected = int(atlas.get("environment_count", -1))
    complete = int(atlas.get("measurement_complete_count", -1))
    if atlas.get("state") != "ready" or expected <= 0 or complete != expected or len(environments) != expected:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_ATLAS_INCOMPLETE")
    locality_failures = sorted(
        name
        for name, row in environments.items()
        if isinstance(row, Mapping) and row.get("locality_state") != "passed"
    )

    refs = [ref for ref in evidence_refs if isinstance(ref, str) and ref]
    if len(refs) < 4 or len(refs) != len(set(refs)):
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_EVIDENCE_INVALID")
    trial = {
        "trial_id": f"{probe_id}__expanded_failure",
        "evidence_class": "live",
        "context": {
            "backbone_family": _text(_mapping(request, "context"), "backbone_family"),
            "capability_class": _text(_mapping(request, "context"), "capability_class"),
            "failure_signature": _text(_mapping(request, "context"), "failure_signature"),
            "primitive": _text(_mapping(request, "context"), "primitive"),
        },
        "probe": dict(program),
        "outcomes": {
            "locality_pass": not locality_failures,
            "nonredundant": nonredundant,
            "collision_resolved": False,
            "regret_reduction": regret,
            "coverage_gain": coverage,
            "gpu_hours": gpu_hours,
        },
        "evidence_refs": refs,
    }

    prior_residual = _mapping(_mapping(request, "context"), "unexplained_residual")
    next_residual, credit = _expanded_failure_credit(
        residual=prior_residual,
        edited_axis=edited_axes[0],
        locality_failed=bool(locality_failures),
    )
    next_current = [dict(parent)]
    prior_max_canaries = int(_mapping(request, "context")["max_canaries"])
    next_context = {
        **_mapping(request, "context"),
        "unexplained_residual": next_residual,
        "residual_evidence_refs": refs,
        "max_canaries": 1,
    }
    next_grammar, filtered_grammar_values = _capability_filter_grammar(
        grammar=_mapping(request, "grammar"),
        capabilities=next_context.get("capabilities", []),
    )
    retrieval, removed_retrieval = _single_axis_candidates(
        request.get("retrieval_candidates", []), current=next_current
    )
    llm, removed_llm = _single_axis_candidates(
        request.get("llm_candidates", []), current=next_current
    )
    next_request = {
        **request,
        "experiment_id": next_experiment_id,
        "context": next_context,
        "current_probes": next_current,
        "grammar": next_grammar,
        "retrieval_candidates": retrieval,
        "llm_candidates": llm,
    }
    next_history = [*history, trial]
    _validate_outputs(request=next_request, history=next_history)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-counterexample-update",
        "state": "ready",
        "prior_experiment_id": request["experiment_id"],
        "next_experiment_id": next_experiment_id,
        "probe_id": probe_id,
        "terminal_state": candidate["state"],
        "edited_axis": edited_axes[0],
        "measured_outcomes": trial["outcomes"],
        "locality_failure_environments": locality_failures,
        "axis_credit_update": credit,
        "next_unexplained_residual": next_residual,
        "single_axis_filter": {
            "removed_retrieval_candidates": removed_retrieval,
            "removed_llm_candidates": removed_llm,
        },
        "capability_filtered_grammar_values": filtered_grammar_values,
        "successive_halving_width_policy": {
            "policy_id": "expanded_failure_direct_parent_single_canary_v1",
            "prior_max_canaries": prior_max_canaries,
            "next_max_canaries": next_context["max_canaries"],
            "reference_parent_probe_id": parent["probe_id"],
            "excluded_current_probe_ids": sorted(
                row["probe_id"]
                for row in _mapping_sequence(
                    request.get("current_probes"), "CPBE_COUNTEREXAMPLE_CURRENT_INVALID"
                )
                if row.get("probe_id") != parent.get("probe_id")
            ),
        },
        "claim_boundary": (
            "This update records a negative diagnostic search result and reprioritizes single-axis "
            "probe hypotheses. It does not admit a probe, improve a world model, or establish transfer."
        ),
    }
    return next_request, trial, next_history, report


def publish_counterexample_round(
    *,
    request_path: Path,
    history_path: Path,
    plan_path: Path,
    receipts_path: Path,
    settlement_path: Path,
    atlas_path: Path,
    canary_manifest_root: Path,
    expanded_manifest_root: Path,
    probe_id: str,
    next_experiment_id: str,
    output_root: Path,
) -> dict[str, object]:
    source_paths = [
        Path(plan_path),
        Path(receipts_path),
        Path(settlement_path),
        Path(atlas_path),
    ]
    atlas = _load_object(atlas_path)
    gpu_hours, runtime_refs = _measured_gpu_hours(
        atlas=atlas,
        probe_id=probe_id,
        canary_manifest_root=canary_manifest_root,
        expanded_manifest_root=expanded_manifest_root,
    )
    refs = [path.as_posix() for path in source_paths] + runtime_refs
    request, trial, history, report = build_counterexample_round(
        request=_load_object(request_path),
        history=_load_jsonl(history_path),
        plan=_load_object(plan_path),
        receipts=_load_jsonl(receipts_path),
        settlement=_load_object(settlement_path),
        atlas=atlas,
        probe_id=probe_id,
        next_experiment_id=next_experiment_id,
        gpu_hours=gpu_hours,
        evidence_refs=refs,
    )
    source_refs = {
        path.as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in [*source_paths, *map(Path, runtime_refs)]
    }
    return write_bundle(
        output_root=output_root,
        files={
            "counterexample-update.json": canonical_json({**report, "source_refs": source_refs}),
            "inputs/cpbe-request.json": canonical_json(request),
            "inputs/probe-trials.jsonl": b"".join(canonical_json(row) for row in history),
            "inputs/appended-trial.json": canonical_json(trial),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cpbe-counterexample-update-manifest",
            "state": "ready",
            "prior_experiment_id": report["prior_experiment_id"],
            "next_experiment_id": next_experiment_id,
            "probe_id": probe_id,
            "history_trial_count": len(history),
            "request_path": "inputs/cpbe-request.json",
            "history_path": "inputs/probe-trials.jsonl",
            "report_path": "counterexample-update.json",
        },
    )


def _expanded_failure_credit(
    *, residual: Mapping[str, Any], edited_axis: str, locality_failed: bool
) -> tuple[dict[str, float], dict[str, object]]:
    weights = {field: _number(residual, field) for field in _DSL_FIELDS}
    before = dict(weights)
    weights[edited_axis] *= 0.10
    weights["aggregation"] += 0.35
    weights["signal_source"] += 0.20
    weights["contrast_operator"] *= 0.25
    if locality_failed:
        weights["spatial_mask"] += 0.25
    total = sum(weights.values())
    normalized = {field: weights[field] / total for field in _DSL_FIELDS}
    return normalized, {
        "policy_id": "expanded_counterexample_axis_credit_v1",
        "before": before,
        "edited_axis_discount": {"axis": edited_axis, "multiplier": 0.10},
        "objective_uplift": {"aggregation": 0.35, "signal_source": 0.20},
        "canary_separation_discount": {"contrast_operator": 0.25},
        "cross_environment_locality_uplift": {"spatial_mask": 0.25 if locality_failed else 0.0},
    }


def _single_axis_candidates(
    candidates: object, *, current: object
) -> tuple[list[Mapping[str, Any]], list[str]]:
    candidate_rows = _mapping_sequence(candidates, "CPBE_COUNTEREXAMPLE_CANDIDATES_INVALID")
    parents = _mapping_sequence(current, "CPBE_COUNTEREXAMPLE_CURRENT_INVALID")
    kept: list[Mapping[str, Any]] = []
    removed: list[str] = []
    for candidate in candidate_rows:
        minimum = min(
            sum(candidate.get(field) != parent.get(field) for field in _DSL_FIELDS)
            for parent in parents
        )
        if minimum == 1:
            kept.append(candidate)
        else:
            removed.append(_text(candidate, "probe_id"))
    return kept, removed


def _capability_filter_grammar(
    *, grammar: Mapping[str, Any], capabilities: object
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not isinstance(capabilities, list) or not all(isinstance(value, str) for value in capabilities):
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_CAPABILITIES_INVALID")
    available = set(capabilities)
    requirements = {
        ("aggregation", "horizon_weighted_goal_outcome"): "horizon_indexed_goal_outcomes",
    }
    filtered: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    for field in _DSL_FIELDS:
        values = grammar.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_GRAMMAR_INVALID:{field}")
        kept_values = []
        removed_values = []
        for value in values:
            required = requirements.get((field, value))
            (removed_values if required and required not in available else kept_values).append(value)
        if not kept_values:
            raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_GRAMMAR_EMPTY:{field}")
        filtered[field] = kept_values
        if removed_values:
            removed[field] = removed_values
    return filtered, removed


def _measured_gpu_hours(
    *,
    atlas: Mapping[str, Any],
    probe_id: str,
    canary_manifest_root: Path,
    expanded_manifest_root: Path,
) -> tuple[float, list[str]]:
    environments = _mapping(atlas, "environments")
    canary_names = ("cloth_move", "pour_water", "push_cube", "push_sand")
    paths = [Path(canary_manifest_root) / name / "manifest.json" for name in canary_names]
    paths.extend(Path(expanded_manifest_root) / name / "manifest.json" for name in sorted(environments))
    elapsed = 0.0
    for path in paths:
        manifest = _load_object(path)
        if manifest.get("state") != "ready" or manifest.get("probe_id") != probe_id:
            raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_RUNTIME_MANIFEST_INVALID:{path}")
        elapsed += _number(manifest, "elapsed_seconds")
    return elapsed / 3600.0, [path.as_posix() for path in paths]


def _unique_parent(*, request: Mapping[str, Any], program: Mapping[str, Any]) -> Mapping[str, Any]:
    parent_ids = program.get("parent_probe_ids")
    if not isinstance(parent_ids, list) or len(parent_ids) != 1:
        raise CPBECounterexampleError("CPBE_COUNTEREXAMPLE_PARENT_INVALID")
    return _unique_by_probe(request.get("current_probes"), parent_ids[0], "CPBE_COUNTEREXAMPLE_PARENT")


def _unique_by_probe(rows: object, probe_id: str, code: str) -> Mapping[str, Any]:
    matches = [row for row in _mapping_sequence(rows, code) if row.get("probe_id") == probe_id]
    if len(matches) != 1:
        raise CPBECounterexampleError(f"{code}_NOT_UNIQUE")
    return matches[0]


def _validate_outputs(*, request: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        validate_document("cpbe_request", request, root=root)
        for row in history:
            validate_document("cpbe_history_trial", row, root=root)
    except ContractValidationError as exc:
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_CONTRACT_INVALID:{exc}") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_JSON_INVALID:{path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_JSONL_INVALID:{path}:{index}")
        rows.append(value)
    return rows


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_MAPPING_INVALID:{key}")
    return item


def _mapping_sequence(value: object, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CPBECounterexampleError(code)
    return list(value)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_TEXT_INVALID:{key}")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_BOOL_INVALID:{key}")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
        raise CPBECounterexampleError(f"CPBE_COUNTEREXAMPLE_NUMBER_INVALID:{key}")
    return float(item)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--settlement", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--canary-manifest-root", type=Path, required=True)
    parser.add_argument("--expanded-manifest-root", type=Path, required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--next-experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = publish_counterexample_round(
        request_path=args.request,
        history_path=args.history,
        plan_path=args.plan,
        receipts_path=args.receipts,
        settlement_path=args.settlement,
        atlas_path=args.atlas,
        canary_manifest_root=args.canary_manifest_root,
        expanded_manifest_root=args.expanded_manifest_root,
        probe_id=args.probe_id,
        next_experiment_id=args.next_experiment_id,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
