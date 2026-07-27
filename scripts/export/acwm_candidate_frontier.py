#!/usr/bin/env python3
"""Audit the live ACWM candidate frontier and emit retrieval work orders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from scripts.export.acwm_8env_live_coverage import build_live_coverage
from scripts.export.acwm_autoloop_queue import ROOT, _runtime_parameters
from scripts.export.acwm_autoloop_replenisher import (
    _PARAMETER_VARIANTS,
    _attempted_parameter_signatures,
    _attempted_runtime_only_signatures,
    _candidate_experience_routing,
    _diagnosed_failure_names,
    _diagnosis_matched_primitive_names,
    _load_horizon_experience_entries,
    _load_json_object,
    _load_registry_if_present,
    _materialization_records,
    _official_training_gate_failure_counts,
    _officially_positive_training_primitives,
    _primitive_matches_failures,
    _signature,
    _training_screen_negative_counts,
)
from wmloop.execute.acwm_primitive_routes import (
    QUALITY_SCREEN_PRIMITIVES,
    RUNTIME_ONLY_PRIMITIVES,
    TRAINING_QUALITY_SCREEN_PRIMITIVES,
    primitive_execution_role,
)


class AcwmCandidateFrontierError(RuntimeError):
    """The frontier audit inputs or output contract were invalid."""


def build_candidate_frontier(
    *,
    staging_plan: Path,
    materialization_gate: Path,
    report_root: Path,
    output_root: Path,
    repo_root: Path = ROOT,
    quality_discovery_only: bool = True,
) -> dict[str, object]:
    """Build one durable environment-by-environment exploration frontier."""

    plan_path = Path(staging_plan).resolve(strict=True)
    gate_path = _latest_materialization_gate(
        configured=Path(materialization_gate).resolve(strict=True),
        report_root=Path(report_root).resolve(),
    )
    plan = _load_json_object(plan_path)
    gate = _load_json_object(gate_path)
    raw_records = plan.get("environment_records")
    if not isinstance(raw_records, list):
        raise AcwmCandidateFrontierError("ACWM_FRONTIER_PLAN_RECORDS_INVALID")

    registry = _load_registry_if_present(Path(repo_root))
    attempted = _attempted_parameter_signatures(Path(report_root))
    attempted_runtime = _attempted_runtime_only_signatures(Path(report_root))
    gate_failures = _official_training_gate_failure_counts(Path(report_root))
    screen_negatives = _training_screen_negative_counts(Path(report_root))
    transfer_primitives = _officially_positive_training_primitives(Path(report_root))
    experience_entries = _load_horizon_experience_entries(Path(report_root))
    materialization_records = _materialization_records(gate)
    work_order_paths = gate.get("work_order_paths")
    if not isinstance(work_order_paths, Mapping):
        work_order_paths = {}
    quality_values = gate.get("quality_screen_ready_primitives")
    if not isinstance(quality_values, list):
        quality_values = gate.get("closed_loop_ready_primitives", [])
    quality_ready = {
        str(value) for value in quality_values if isinstance(value, str)
    } & QUALITY_SCREEN_PRIMITIVES
    training_ready = quality_ready & TRAINING_QUALITY_SCREEN_PRIMITIVES

    coverage = build_live_coverage(report_root=Path(report_root))
    coverage_rows = {
        str(row["environment"]): row
        for row in coverage.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("environment"), str)
    }
    failure_manifest = _current_failure_manifest(Path(report_root))

    environment_rows: list[dict[str, object]] = []
    retrieval_orders: list[dict[str, object]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            continue
        environment = str(raw_record.get("environment") or "")
        recommended_raw = raw_record.get("recommended_existing_primitives")
        if not environment or not isinstance(recommended_raw, list):
            continue
        target_failures = _diagnosed_failure_names(
            environment=environment,
            registry=registry,
            report_root=Path(report_root),
            failure_manifest=failure_manifest,
        )
        recommended = [str(value) for value in recommended_raw if isinstance(value, str)]
        for primitive in _diagnosis_matched_primitive_names(
            environment=environment,
            registry=registry,
            report_root=Path(report_root),
            failure_manifest=failure_manifest,
        ):
            if primitive not in recommended:
                recommended.append(primitive)
        for primitive in transfer_primitives:
            if primitive not in recommended:
                recommended.append(primitive)

        attempted_rows: list[dict[str, object]] = []
        open_rows: list[dict[str, object]] = []
        blocked_rows: list[dict[str, object]] = []
        unavailable_rows: list[dict[str, object]] = []
        penalties: list[dict[str, object]] = []
        for primitive in recommended:
            if primitive not in transfer_primitives and not _primitive_matches_failures(
                primitive=primitive,
                failures=target_failures,
                registry=registry,
            ):
                continue
            variants = _PARAMETER_VARIANTS.get(primitive, ({},))
            official_failures = gate_failures.get((environment, primitive), 0)
            negative_screens = screen_negatives.get((environment, primitive), 0)
            penalty = 2 * official_failures + negative_screens
            experience_routing = _candidate_experience_routing(
                environment=environment,
                primitive=primitive,
                target_failures=target_failures,
                entries=experience_entries,
            )
            penalties.append(
                {
                    "primitive": primitive,
                    "official_gate_failure_count": official_failures,
                    "screen_negative_count": negative_screens,
                    "failure_evidence_penalty": penalty,
                }
            )
            if primitive not in quality_ready:
                record = materialization_records.get(primitive, {})
                blocked = {
                    "primitive": primitive,
                    "reason": "not_materialized_for_quality_screen",
                    "execution_role": primitive_execution_role(primitive),
                    "admission_state": str(record.get("admission_state") or "unknown"),
                    "work_order_path": str(work_order_paths.get(primitive) or ""),
                }
                if blocked["work_order_path"] or blocked["admission_state"] not in {
                    "closed_loop_runtime_ready",
                    "runtime_hook_ready",
                }:
                    blocked_rows.append(blocked)
                else:
                    blocked["reason"] = "quality_role_not_admitted"
                    unavailable_rows.append(blocked)
                continue
            if primitive in RUNTIME_ONLY_PRIMITIVES:
                for parameters in variants:
                    runtime_parameters = _runtime_parameters(parameters)
                    signature = _signature(runtime_parameters)
                    item = {
                        "primitive": primitive,
                        "mode": "runtime_only",
                        "parameters": dict(parameters),
                        "signature": signature,
                        "failure_evidence_penalty": penalty,
                        "experience_routing": experience_routing,
                    }
                    if (environment, primitive, signature) in attempted_runtime:
                        attempted_rows.append(item)
                    else:
                        open_rows.append(item)
                continue
            if primitive not in training_ready:
                continue
            for parameters in variants:
                runtime_parameters = _runtime_parameters(parameters)
                signature = _signature(runtime_parameters)
                item = {
                    "primitive": primitive,
                    "parameters": dict(parameters),
                    "signature": signature,
                    "failure_evidence_penalty": penalty,
                    "experience_routing": experience_routing,
                }
                if (environment, primitive, signature) in attempted:
                    attempted_rows.append(item)
                else:
                    open_rows.append(item)

        coverage_row = coverage_rows.get(environment, {})
        is_metric_confirmed = coverage_row.get("formally_confirmed") is True
        is_confirmed = coverage_row.get("claim_ready") is True
        event_semantic_required = coverage_row.get("event_semantic_required") is True
        event_semantic_validated = coverage_row.get("event_semantic_validated") is True
        is_in_flight = coverage_row.get("screen_running") is True
        is_pending = (
            coverage_row.get("confirmation_pending") is True
            or coverage_row.get("internal_positive_pending_official") is True
        )
        requires_new = not is_confirmed and not is_in_flight and not is_pending and not open_rows
        if is_confirmed:
            next_action = "confirmed_no_exploration_required"
        elif is_in_flight or is_pending:
            next_action = "await_in_flight_evidence"
        elif open_rows:
            next_action = "continue_existing_candidate_frontier"
        elif blocked_rows:
            next_action = "materialize_existing_diagnosis_matched_primitive"
        else:
            next_action = "retrieve_and_stage_new_primitive"

        row = {
            "environment": environment,
            "metric_positive": is_metric_confirmed,
            "formal_positive": is_confirmed,
            "event_semantic_required": event_semantic_required,
            "event_semantic_validated": event_semantic_validated,
            "in_flight": is_in_flight,
            "pending_confirmation": is_pending,
            "target_failure_signatures": list(target_failures),
            "open_candidates": sorted(
                open_rows,
                key=lambda item: (
                    int(
                        (item.get("experience_routing") or {}).get("rank_band", 3)
                        if isinstance(item.get("experience_routing"), Mapping)
                        else 3
                    ),
                    int(item.get("failure_evidence_penalty", 0)),
                    str(item.get("primitive") or ""),
                    str(item.get("signature") or ""),
                ),
            ),
            "attempted_signatures": attempted_rows,
            "failed_evidence_penalties": sorted(
                penalties,
                key=lambda item: (
                    -int(item["failure_evidence_penalty"]),
                    str(item["primitive"]),
                ),
            ),
            "blocked_materialization": blocked_rows,
            "unavailable_candidates": unavailable_rows,
            "requires_new_primitive_or_materialization": requires_new,
            "next_action": next_action,
        }
        environment_rows.append(row)
        if requires_new:
            retrieval_orders.append(_retrieval_work_order(row))

    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-candidate-frontier",
        "state": "ready",
        "created_at": _utc_now(),
        "staging_plan": str(plan_path),
        "materialization_gate": str(gate_path),
        "failure_manifest": str(failure_manifest),
        "quality_discovery_only": quality_discovery_only,
        "summary": {
            "environment_count": len(environment_rows),
            "formal_positive_count": sum(bool(row["formal_positive"]) for row in environment_rows),
            "metric_positive_count": sum(bool(row["metric_positive"]) for row in environment_rows),
            "in_flight_or_pending_count": sum(
                bool(row["in_flight"] or row["pending_confirmation"]) for row in environment_rows
            ),
            "open_candidate_count": sum(len(row["open_candidates"]) for row in environment_rows),
            "experience_ranked_open_candidate_count": sum(
                1
                for row in environment_rows
                for candidate in row["open_candidates"]
                if isinstance(candidate, Mapping)
                and isinstance(candidate.get("experience_routing"), Mapping)
                and int(candidate["experience_routing"].get("matched_profile_count", 0)) > 0
            ),
            "experience_profile_count": len(experience_entries),
            "blocked_materialization_count": sum(
                len(row["blocked_materialization"]) for row in environment_rows
            ),
            "unavailable_candidate_count": sum(
                len(row["unavailable_candidates"]) for row in environment_rows
            ),
            "requires_new_primitive_or_materialization_count": len(retrieval_orders),
        },
        "environments": environment_rows,
        "retrieval_work_orders": retrieval_orders,
        "claim_boundary": (
            "This frontier controls exploration routing only. A candidate becomes a positive result only after "
            "the frozen official gate and the current 800/1000 checkpoint confirmation contract pass; "
            "environments with a semantic event contract additionally require that event gate to pass."
        ),
    }
    return _write_bundle(Path(output_root).resolve(), payload)


def _latest_materialization_gate(*, configured: Path, report_root: Path) -> Path:
    candidates = [configured]
    candidates.extend(report_root.glob("primitive-materialization-gate-dynamic-r*/manifest.json"))
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        raise AcwmCandidateFrontierError("ACWM_FRONTIER_MATERIALIZATION_GATE_MISSING")
    return max(existing, key=lambda path: (path.stat().st_mtime_ns, path.parent.name))


def _current_failure_manifest(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("m1-raw-failure-reports-ladder-r*/manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
        reverse=True,
    )
    if not candidates:
        raise AcwmCandidateFrontierError("ACWM_FRONTIER_FAILURE_MANIFEST_MISSING")
    return candidates[0].resolve()


def _retrieval_work_order(environment: Mapping[str, object]) -> dict[str, object]:
    failures = [str(value) for value in environment.get("target_failure_signatures", [])]
    attempted = environment.get("attempted_signatures", [])
    attempted_primitives = sorted(
        {
            str(item.get("primitive"))
            for item in attempted
            if isinstance(item, Mapping) and item.get("primitive")
        }
    )
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-retrieval-work-order",
        "state": "ready",
        "environment": environment["environment"],
        "target_failure_signatures": failures,
        "attempted_primitives": attempted_primitives,
        "query_terms": _query_terms(failures),
        "required_candidate_fields": [
            "mechanism_hypothesis",
            "target_failure_signatures",
            "proposed_manifest",
            "source_arxiv_ids",
            "expected_runtime_hook",
            "falsification_test",
        ],
        "admission_gates": [
            "literature_source_verified",
            "primitive_schema_valid",
            "configuration_intent_matches_runtime_hook",
            "clean_diff_and_tests",
            "runtime_materialization_receipt",
            "512_screen_then_official_gate",
        ],
        "forbidden_compromises": [
            "sidecar_only_implementation_presented_as_runtime_effect",
            "substituting_an_easier_primitive_for_the_retrieved_mechanism",
            "changing_frozen_evaluator_or_heldout_protocol",
            "ground_truth_injection_into_candidate_rollout",
        ],
    }


def _query_terms(failures: Sequence[str]) -> list[str]:
    terms = ["action-conditioned world model long-horizon consistency"]
    terms.extend(f"world model {failure.replace('_', ' ')}" for failure in failures)
    return list(dict.fromkeys(terms))


def _write_bundle(destination: Path, payload: dict[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmCandidateFrontierError("ACWM_FRONTIER_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "candidate-frontier.json", payload)
        (temporary / "candidate-frontier.md").write_text(_markdown(payload), encoding="utf-8")
        _write_csv(temporary / "candidate-frontier.csv", payload)
        orders_root = temporary / "retrieval-work-orders"
        for order in payload["retrieval_work_orders"]:
            assert isinstance(order, Mapping)
            orders_root.mkdir(mode=0o700, exist_ok=True)
            _write_json(orders_root / f"{order['environment']}.json", order)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-candidate-frontier-manifest",
            "state": "ready",
            "report_path": str(destination / "candidate-frontier.json"),
            "markdown_path": str(destination / "candidate-frontier.md"),
            "csv_path": str(destination / "candidate-frontier.csv"),
            "retrieval_work_order_count": len(payload["retrieval_work_orders"]),
            "summary": payload["summary"],
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_csv(path: Path, payload: Mapping[str, object]) -> None:
    fields = [
        "environment",
        "metric_positive",
        "formal_positive",
        "event_semantic_required",
        "event_semantic_validated",
        "in_flight",
        "pending_confirmation",
        "open_candidate_count",
        "attempted_signature_count",
        "blocked_materialization_count",
        "requires_new_primitive_or_materialization",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["environments"]:
            assert isinstance(row, Mapping)
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "open_candidate_count": len(row["open_candidates"]),
                    "attempted_signature_count": len(row["attempted_signatures"]),
                    "blocked_materialization_count": len(row["blocked_materialization"]),
                }
            )


def _markdown(payload: Mapping[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    lines = [
        "# ACWM Candidate Frontier",
        "",
        f"- Formal positives: {summary['formal_positive_count']}/{summary['environment_count']}",
        f"- Open candidates: {summary['open_candidate_count']}",
        f"- Retrieval/materialization required: {summary['requires_new_primitive_or_materialization_count']}",
        "",
        "| Environment | Positive | In flight | Open | Attempted | Blocked | Next action |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["environments"]:
        assert isinstance(row, Mapping)
        lines.append(
            "| {environment} | {positive} | {flight} | {open_count} | {attempted} | {blocked} | {action} |".format(
                environment=row["environment"],
                positive="yes" if row["formal_positive"] else "no",
                flight="yes" if row["in_flight"] or row["pending_confirmation"] else "no",
                open_count=len(row["open_candidates"]),
                attempted=len(row["attempted_signatures"]),
                blocked=len(row["blocked_materialization"]),
                action=row["next_action"],
            )
        )
    lines.extend(["", str(payload["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--staging-plan", type=Path, required=True)
    parser.add_argument("--materialization-gate", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--include-unconfirmed-confirmation", action="store_true")
    args = parser.parse_args(argv)
    result = build_candidate_frontier(
        staging_plan=args.staging_plan,
        materialization_gate=args.materialization_gate,
        report_root=args.report_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
        quality_discovery_only=not args.include_unconfirmed_confirmation,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
