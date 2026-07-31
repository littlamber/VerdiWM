"""Convert frozen ACWM collision evidence into a live CPBE planning bundle."""

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


class ACWMCPBEBootstrapError(ValueError):
    """ACWM evidence could not be converted without inventing missing facts."""


def build_acwm_cpbe_bootstrap(
    *,
    request_template: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
    redundancy_report: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build a live CPBE request and one measured historical trial.

    The template owns the typed Probe DSL and grammar. Frozen experiment
    receipts own every empirical field. The adapter never infers a positive
    collision, regret, or coverage outcome from an incomplete replay.
    """

    template = dict(request_template)
    if template.get("artifact_type") != "verdiwm-cpbe-request":
        raise ACWMCPBEBootstrapError("ACWM_CPBE_TEMPLATE_INVALID")
    if template.get("evidence_class") != "live":
        raise ACWMCPBEBootstrapError("ACWM_CPBE_TEMPLATE_NOT_LIVE")
    context = _mapping(template, "context")
    target = _text(context, "target_id")
    primitive = _text(context, "primitive")
    programs = _mapping_sequence(template.get("current_probes"), "ACWM_CPBE_CURRENT_PROBES_INVALID")
    if len(programs) != 1:
        raise ACWMCPBEBootstrapError("ACWM_CPBE_ONE_CURRENT_PROBE_REQUIRED")
    program = dict(programs[0])
    probe_id = _text(program, "probe_id")

    work_order = _unique_work_order(selector_replay, target=target, primitive=primitive)
    comparison = _unique_comparison(redundancy_report, target=target, probe_id=probe_id)
    _validate_environment_manifest(environment_manifest, target=target, probe_id=probe_id)
    refs = tuple(value for value in evidence_refs if isinstance(value, str) and value)
    if len(refs) < 3 or len(set(refs)) != len(refs):
        raise ACWMCPBEBootstrapError("ACWM_CPBE_EVIDENCE_REFS_INVALID")

    locality_pass = _boolean(comparison, "candidate_locality_pass")
    redundant = _boolean(comparison, "redundant")
    locality_residual = _finite_number(comparison, "candidate_locality_residual", minimum=0.0)
    redundancy_cosine = _finite_number(comparison, "cosine_similarity")
    elapsed_seconds = _finite_number(environment_manifest, "elapsed_seconds", exclusive_minimum=0.0)
    unresolved = _work_order_still_open(selector_replay, target=target, primitive=primitive)
    if not unresolved:
        raise ACWMCPBEBootstrapError("ACWM_CPBE_COLLISION_ALREADY_RESOLVED")

    residual = _counterexample_residual(
        locality_pass=locality_pass,
        redundant=redundant,
        locality_residual=locality_residual,
    )
    filled_context = {
        **context,
        "failure_signature": _text(work_order, "reason"),
        "unexplained_residual": residual,
        "residual_evidence_refs": list(refs),
    }
    request = {**template, "context": filled_context}
    trial = {
        "trial_id": f"{probe_id}__{target}__measured_counterexample",
        "evidence_class": "live",
        "context": {
            "backbone_family": _text(filled_context, "backbone_family"),
            "capability_class": _text(filled_context, "capability_class"),
            "failure_signature": _text(work_order, "reason"),
            "primitive": primitive,
        },
        "probe": program,
        "outcomes": {
            "locality_pass": locality_pass,
            "nonredundant": not redundant,
            "collision_resolved": None,
            "regret_reduction": None,
            "coverage_gain": None,
            "gpu_hours": elapsed_seconds / 3600.0,
        },
        "evidence_refs": list(refs),
    }
    _validate_generated(request=request, trial=trial)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-cpbe-bootstrap",
        "state": "ready",
        "experiment_id": request["experiment_id"],
        "target_environment": target,
        "primitive": primitive,
        "current_probe_id": probe_id,
        "open_work_order": dict(work_order),
        "measured_counterexample": {
            "locality_pass": locality_pass,
            "locality_residual": locality_residual,
            "nonredundant": not redundant,
            "redundancy_cosine": redundancy_cosine,
            "gpu_hours": elapsed_seconds / 3600.0,
        },
        "residual_policy": {
            "policy_id": "acwm_counterexample_axis_attribution_v1",
            "weights": residual,
            "rule": _residual_rule(locality_pass=locality_pass, redundant=redundant),
        },
        "source_refs": [
            {"path": ref, "sha256": _sha256(Path(ref)), "size_bytes": Path(ref).stat().st_size}
            for ref in refs
        ],
        "claim_boundary": (
            "This adapter creates a live diagnostic search request from frozen ACWM evidence. "
            "Unobserved collision, regret, and coverage outcomes remain null. The historical trial "
            "is not a positive probe, repair-quality result, accepted-coverage gain, or transfer result."
        ),
    }
    return request, trial, report


def publish_acwm_cpbe_bootstrap(
    *,
    template_path: Path,
    selector_replay_path: Path,
    redundancy_report_path: Path,
    environment_manifest_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    paths = (
        Path(selector_replay_path),
        Path(redundancy_report_path),
        Path(environment_manifest_path),
    )
    request, trial, report = build_acwm_cpbe_bootstrap(
        request_template=_load_object(template_path),
        selector_replay=_load_object(selector_replay_path),
        redundancy_report=_load_object(redundancy_report_path),
        environment_manifest=_load_object(environment_manifest_path),
        evidence_refs=[path.as_posix() for path in paths],
    )
    return write_bundle(
        output_root=output_root,
        files={
            "acwm-cpbe-bootstrap.json": canonical_json(report),
            "inputs/cpbe-request.json": canonical_json(request),
            "inputs/probe-trials.jsonl": canonical_json(trial),
        },
        manifest_fields={
            "artifact_type": "verdiwm-acwm-cpbe-bootstrap-manifest",
            "state": "ready",
            "experiment_id": report["experiment_id"],
            "target_environment": report["target_environment"],
            "current_probe_id": report["current_probe_id"],
            "report_path": "acwm-cpbe-bootstrap.json",
            "request_path": "inputs/cpbe-request.json",
            "history_path": "inputs/probe-trials.jsonl",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _counterexample_residual(
    *, locality_pass: bool, redundant: bool, locality_residual: float
) -> dict[str, float]:
    if not locality_pass:
        weights = {
            "signal_source": 0.10,
            "hook_type": 0.02,
            "spatial_mask": 0.35,
            "temporal_basis": 0.20,
            "contrast_operator": 0.08,
            "aggregation": 0.25,
        }
    elif redundant:
        weights = {
            "signal_source": 0.15,
            "hook_type": 0.05,
            "spatial_mask": 0.10,
            "temporal_basis": 0.25,
            "contrast_operator": 0.35,
            "aggregation": 0.10,
        }
    else:
        weights = {
            "signal_source": 0.20,
            "hook_type": 0.05,
            "spatial_mask": 0.20,
            "temporal_basis": 0.25,
            "contrast_operator": 0.15,
            "aggregation": 0.15,
        }
    severity = 1.0 + min(locality_residual, 10.0) / 10.0
    scaled = {key: value * severity for key, value in weights.items()}
    total = sum(scaled.values())
    return {key: value / total for key, value in scaled.items()}


def _residual_rule(*, locality_pass: bool, redundant: bool) -> str:
    if not locality_pass:
        return "nonlocal_response_prioritizes_spatial_mask_then_aggregation_and_temporal_basis"
    if redundant:
        return "redundant_response_prioritizes_contrast_operator_then_temporal_basis"
    return "local_nonredundant_but_unresolved_prioritizes_temporal_basis_and_signal_source"


def _unique_work_order(
    replay: Mapping[str, Any], *, target: str, primitive: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in _mapping_sequence(
            replay.get("probe_evolution_work_orders"), "ACWM_CPBE_WORK_ORDERS_INVALID"
        )
        if item.get("target_environment") == target and item.get("primitive") == primitive
    ]
    if len(matches) != 1:
        raise ACWMCPBEBootstrapError("ACWM_CPBE_WORK_ORDER_NOT_UNIQUE")
    return matches[0]


def _unique_comparison(
    report: Mapping[str, Any], *, target: str, probe_id: str
) -> Mapping[str, Any]:
    if report.get("candidate_probe_id") != probe_id:
        raise ACWMCPBEBootstrapError("ACWM_CPBE_REDUNDANCY_PROBE_MISMATCH")
    matches = [
        item
        for item in _mapping_sequence(report.get("comparisons"), "ACWM_CPBE_COMPARISONS_INVALID")
        if item.get("environment") == target
    ]
    if len(matches) != 1:
        raise ACWMCPBEBootstrapError("ACWM_CPBE_COMPARISON_NOT_UNIQUE")
    return matches[0]


def _validate_environment_manifest(
    manifest: Mapping[str, Any], *, target: str, probe_id: str
) -> None:
    if (
        manifest.get("artifact_type") != "verdiwm-acwm-fingerprint-environment-manifest"
        or manifest.get("state") != "ready"
        or manifest.get("environment") != target
        or manifest.get("probe_id") != probe_id
    ):
        raise ACWMCPBEBootstrapError("ACWM_CPBE_ENVIRONMENT_MANIFEST_MISMATCH")


def _work_order_still_open(replay: Mapping[str, Any], *, target: str, primitive: str) -> bool:
    return any(
        item.get("target_environment") == target and item.get("primitive") == primitive
        for item in _mapping_sequence(
            replay.get("probe_evolution_work_orders"), "ACWM_CPBE_WORK_ORDERS_INVALID"
        )
    )


def _validate_generated(*, request: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        validate_document("cpbe_request", request, root=root)
        validate_document("cpbe_history_trial", trial, root=root)
    except ContractValidationError as exc:
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_GENERATED_CONTRACT_INVALID:{exc}") from exc


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_MAPPING_INVALID:{key}")
    return item


def _mapping_sequence(value: object, error: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ACWMCPBEBootstrapError(error)
    return list(value)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_TEXT_INVALID:{key}")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_BOOL_INVALID:{key}")
    return item


def _finite_number(
    value: Mapping[str, Any], key: str, *, minimum: float | None = None, exclusive_minimum: float | None = None
) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_NUMBER_INVALID:{key}")
    number = float(item)
    if minimum is not None and number < minimum:
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_NUMBER_INVALID:{key}")
    if exclusive_minimum is not None and number <= exclusive_minimum:
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_NUMBER_INVALID:{key}")
    return number


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_INPUT_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise ACWMCPBEBootstrapError(f"ACWM_CPBE_INPUT_INVALID:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--selector-replay", type=Path, required=True)
    parser.add_argument("--redundancy-report", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = publish_acwm_cpbe_bootstrap(
        template_path=args.template,
        selector_replay_path=args.selector_replay,
        redundancy_report_path=args.redundancy_report,
        environment_manifest_path=args.environment_manifest,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
