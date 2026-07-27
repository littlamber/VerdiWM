"""Build a portable failure-signature bank from closed-loop evidence exports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document


class FailureSignatureBankError(RuntimeError):
    """Failure-signature bank export could not be produced safely."""


_P0_STATUSES = {"uplift_gap", "rejected_candidate_needs_new_diagnosis"}
_REJECTED_STATUSES = {"rejected_candidate_needs_new_diagnosis"}

_SIGNATURE_TO_FAILURE_FAMILY = {
    "action_binding": "action_binding",
    "cloth_identity_drift": "appearance_drift",
    "contact_event": "ood_physics",
    "contact_instability": "ood_physics",
    "container_boundary_leak": "ood_physics",
    "deformable_contact": "ood_physics",
    "deformable_memory": "train_infer_mismatch",
    "endpoint_control": "action_binding",
    "endpoint_path": "action_binding",
    "free_surface": "ood_physics",
    "fluid_volume_transport": "ood_physics",
    "granular_frontier": "ood_physics",
    "inverse_dynamics_confidence": "action_binding",
    "kinematic_action_binding": "action_binding",
    "mass_redistribution": "ood_physics",
    "object_identity": "appearance_drift",
    "particle_boundary": "ood_physics",
    "rigid_pose_slip": "ood_physics",
    "smooth_motion_prior": "action_binding",
    "support_relation": "ood_physics",
    "surface_fold": "appearance_drift",
    "target_conditioning": "action_binding",
    "target_path": "action_binding",
    "topology_change": "ood_physics",
}


def run_failure_signature_bank(
    *,
    repo_root: Path,
    uplift_plan: Path,
    mechanism_cards: Path,
    output_root: Path,
    gap_staging_plan: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only bank of failure signatures and AutoProbe work orders."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_OUTPUT_EXISTS")
    uplift_path = Path(uplift_plan).resolve(strict=True)
    cards_path = Path(mechanism_cards).resolve(strict=True)
    gap_path = Path(gap_staging_plan).resolve(strict=True) if gap_staging_plan is not None else None

    uplift = _load_json_object(uplift_path, "FAILURE_SIGNATURE_BANK_UPLIFT_PLAN_INVALID")
    if uplift.get("artifact_type") != "wmloop-acwm-8env-uplift-transfer-plan":
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_UPLIFT_PLAN_INVALID")
    gap = _load_gap_plan(gap_path)
    cards = _load_mechanism_cards(cards_path)

    records = [
        _environment_record(row, gap=gap.get(str(row.get("environment")), {}))
        for row in _rows(uplift, "environment_rows")
    ]
    probe_orders = [
        _probe_work_order(record, signature)
        for record in records
        for signature in record["failure_signatures"]
        if signature["priority"].startswith("P0")
    ]
    primitive_routing = [
        route
        for record in records
        for route in _primitive_routes(record=record, gap=gap.get(str(record["environment"]), {}), cards=cards)
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-failure-signature-bank",
        "state": "ready",
        "source_files": {
            "uplift_plan": str(uplift_path),
            "gap_staging_plan": str(gap_path) if gap_path is not None else None,
            "mechanism_cards": str(cards_path),
        },
        "summary": {
            "environment_count": len(records),
            "failure_signature_count": sum(len(record["failure_signatures"]) for record in records),
            "p0_probe_order_count": len(probe_orders),
            "primitive_routing_count": len(primitive_routing),
            "positive_exemplar_count": sum(1 for record in records if record["current_status"] == "stable_positive_exemplar"),
            "uplift_gap_count": sum(1 for record in records if record["current_status"] == "uplift_gap"),
            "rejected_candidate_count": sum(1 for record in records if record["current_status"] in _REJECTED_STATUSES),
        },
        "records": records,
        "diagnostic_probe_work_orders": probe_orders,
        "primitive_routing": primitive_routing,
        "transfer_contract": {
            "claim_boundary": (
                "This bank transfers failure mechanisms and routing priors, not ACWM-Phys verdict metrics. "
                "A new backbone still needs its own goal_spec, held-out split, probe registry, evaluator adapter, "
                "hook adapter, primitive admission, and frozen verifier."
            ),
            "portable_inputs": [
                "base_model_or_checkpoint",
                "dataset_or_rollout_source",
                "target_metrics_or_evaluator",
                "budget",
            ],
            "portable_outputs": [
                "failure_signature_candidates",
                "diagnostic_probe_work_orders",
                "primitive_routing_priors",
                "mechanism_transfer_preconditions",
                "anti_conditions_for_failed_routes",
            ],
            "not_allowed": [
                "mutate verdict probes during an active campaign",
                "promote diagnostic probes to verdict role without a version boundary",
                "claim transfer on a new backbone before its evaluator and held-out split are frozen",
                "promote primitive code without materialization and runtime admission gates",
            ],
        },
        "side_effects": {
            "verdict_probe_mutated": False,
            "goal_config_mutated": False,
            "primitive_registry_mutated": False,
            "gpu_execution_started": False,
            "formal_verdict_mutated": False,
        },
        "limitations": [
            "This export is a planning and routing artifact; it does not run evaluation or training.",
            "Diagnostic probe work orders can improve proposal sorting, but they cannot enter verdict evidence in the same frozen campaign.",
            "A positive route remains environment- and protocol-scoped until confirmed by canary and formal verdict runs.",
        ],
    }
    try:
        validate_document("failure_signature_bank", report, root=root)
    except ContractValidationError as exc:
        raise FailureSignatureBankError(f"FAILURE_SIGNATURE_BANK_CONTRACT_INVALID:{exc}") from exc
    return _write_bundle(
        report=report,
        output_root=destination,
        archive_db=archive_db,
        cas_root=cas_root,
        source_paths=[path for path in (uplift_path, gap_path, cards_path) if path is not None],
    )


def _load_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureSignatureBankError(code) from exc
    if not isinstance(payload, Mapping):
        raise FailureSignatureBankError(code)
    return payload


def _load_gap_plan(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    gap = _load_json_object(path, "FAILURE_SIGNATURE_BANK_GAP_PLAN_INVALID")
    if gap.get("artifact_type") != "wmloop-acwm-gap-driven-staging-plan":
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_GAP_PLAN_INVALID")
    return {str(row["environment"]): row for row in _rows(gap, "environment_records")}


def _load_mechanism_cards(path: Path) -> dict[str, Mapping[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_MECHANISM_CARDS_INVALID") from exc
    if not rows or any(not row.get("primitive") for row in rows):
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_MECHANISM_CARDS_INVALID")
    return {str(row["primitive"]): row for row in rows}


def _rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise FailureSignatureBankError(f"FAILURE_SIGNATURE_BANK_ROWS_INVALID:{key}")
    return rows


def _environment_record(row: Mapping[str, Any], *, gap: Mapping[str, Any]) -> dict[str, object]:
    environment = _string(row, "environment")
    status = _string(row, "current_status")
    candidates = _strings(gap.get("recommended_existing_primitives"))
    signatures = [
        _signature_record(
            environment=environment,
            regime=_string(row, "regime"),
            status=status,
            signature=signature,
            candidate_primitives=candidates,
        )
        for signature in _strings(row.get("failure_signatures_to_probe"))
    ]
    return {
        "environment": environment,
        "regime": _string(row, "regime"),
        "current_status": status,
        "current_best_primitive": _string(row, "current_best_primitive"),
        "evidence_level": _string(gap, "evidence_level", default=status),
        "trial_count": _int_or_zero(row.get("trial_count")),
        "positive_trial_count": _int_or_zero(row.get("positive_trial_count")),
        "positive_rate": _float_or_none(row.get("positive_rate")),
        "mean_delta": _float_or_none(row.get("mean_delta")),
        "max_delta": _float_or_none(row.get("max_delta")),
        "failure_signatures": signatures,
        "next_action": _string(row, "next_action"),
    }


def _signature_record(
    *,
    environment: str,
    regime: str,
    status: str,
    signature: str,
    candidate_primitives: Sequence[str],
) -> dict[str, object]:
    return {
        "signature": signature,
        "wm_dx_failure_family": _SIGNATURE_TO_FAILURE_FAMILY.get(signature, "mixed"),
        "diagnostic_probe_id": _probe_id(environment, signature),
        "probe_role": "diagnostic",
        "priority": _priority(status),
        "verdict_exposure_allowed": False,
        "candidate_primitives": list(candidate_primitives),
        "transfer_tags": [regime, signature, _SIGNATURE_TO_FAILURE_FAMILY.get(signature, "mixed")],
    }


def _probe_work_order(record: Mapping[str, object], signature: Mapping[str, object]) -> dict[str, object]:
    probe_id = str(signature["diagnostic_probe_id"])
    environment = str(record["environment"])
    signature_name = str(signature["signature"])
    return {
        "work_order_id": f"{probe_id}_work_order",
        "environment": environment,
        "signature": signature_name,
        "probe_id": probe_id,
        "role": "diagnostic",
        "priority": str(signature["priority"]),
        "signal_contract": _signal_contract(signature_name),
        "allowed_mutation_paths": [
            f"wmloop/diagnose/probes/{probe_id}.py",
            f"tests/test_{probe_id}.py",
            "configs/probes/staging/",
            "results/reports/",
        ],
        "forbidden_surfaces": [
            "configs/goal/",
            "configs/constitution/",
            "configs/eval_frozen.sha256",
            "configs/registry_frozen.sha256",
            "verdict_evidence",
            "frozen evaluator code",
        ],
        "admission_gates": [
            "schema_valid_diagnostic_probe_output",
            "offline_fixture_test_passed",
            "runtime_smoke_on_dev_split",
            "no_verdict_evidence_exposure",
            "human_approved_version_boundary_before_verdict_role",
        ],
        "verdict_exposure_allowed": False,
    }


def _primitive_routes(
    *,
    record: Mapping[str, object],
    gap: Mapping[str, Any],
    cards: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    primitives = _strings(gap.get("recommended_existing_primitives"))
    if not primitives and isinstance(record.get("current_best_primitive"), str):
        primitives = [str(record["current_best_primitive"])]
    routes = []
    for primitive in primitives:
        card = cards.get(primitive, {})
        routes.append(
            {
                "environment": str(record["environment"]),
                "primitive": primitive,
                "mechanism_family": _card_field(card, "mechanism_family"),
                "layer": _card_field(card, "layer"),
                "current_status": str(record["current_status"]),
                "evidence_level": str(record["evidence_level"]),
                "routing_decision": _routing_decision(record=record, primitive=primitive),
                "target_failures": _split_csv_field(_card_field(card, "targets_failures")),
                "transfer_preconditions": _card_field(card, "transfer_preconditions"),
                "known_anti_conditions": _card_field(card, "known_anti_conditions"),
                "next_gate": _next_gate(record=record, primitive=primitive),
            }
        )
    return routes


def _routing_decision(*, record: Mapping[str, object], primitive: str) -> str:
    status = str(record["current_status"])
    if status == "stable_positive_exemplar":
        return "retain_as_source_exemplar"
    if status in _REJECTED_STATUSES and primitive == str(record["current_best_primitive"]):
        return "reject_or_demote_until_new_diagnosis"
    if status in _P0_STATUSES:
        return "stage_canary_after_diagnostic_probe"
    return "monitor_only"


def _next_gate(*, record: Mapping[str, object], primitive: str) -> str:
    if _routing_decision(record=record, primitive=primitive) == "retain_as_source_exemplar":
        return "export_visual_and_formal_replication_evidence"
    if _routing_decision(record=record, primitive=primitive) == "reject_or_demote_until_new_diagnosis":
        return "do_not_promote; require new diagnostic signature before retry"
    return "diagnostic_probe_output -> proposal_readiness -> canary -> formal_verdict"


def _priority(status: str) -> str:
    if status == "rejected_candidate_needs_new_diagnosis":
        return "P0_rediagnose"
    if status == "uplift_gap":
        return "P0_uplift_gap"
    if status == "stable_positive_exemplar":
        return "P2_retain_exemplar"
    return "P1_monitor"


def _signal_contract(signature: str) -> str:
    family = _SIGNATURE_TO_FAILURE_FAMILY.get(signature, "mixed")
    return (
        f"Measure `{signature}` as diagnostic-only evidence for `{family}` routing. "
        "The probe output may sort hypotheses and primitive candidates, but it must not enter frozen verdict evidence."
    )


def _probe_id(environment: str, signature: str) -> str:
    return f"acwm_{_safe_name(environment)}_{_safe_name(signature)}_diagnostic_v1"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.strip().lower()).strip("_")


def _card_field(card: Mapping[str, str], key: str) -> str:
    value = card.get(key)
    if isinstance(value, str) and value:
        return value
    return "mechanism_card_missing"


def _split_csv_field(value: str) -> list[str]:
    if value == "mechanism_card_missing":
        return ["mechanism_card_missing"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _string(document: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = document.get(key, default)
    if not isinstance(value, str) or not value:
        raise FailureSignatureBankError(f"FAILURE_SIGNATURE_BANK_FIELD_INVALID:{key}")
    return value


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _write_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
    source_paths: Sequence[Path],
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    probe_orders = report.get("diagnostic_probe_work_orders")
    primitive_routing = report.get("primitive_routing")
    if not isinstance(probe_orders, list) or not isinstance(primitive_routing, list):
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_REPORT_INVALID")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "failure-signature-bank.json", report_bytes)
        _write_bytes_atomic(temporary / "failure-signature-bank.md", markdown_bytes)
        tables_dir = temporary / "tables"
        tables_dir.mkdir(mode=0o700)
        _write_probe_order_csv(tables_dir / "diagnostic_probe_work_orders.csv", probe_orders)
        _write_primitive_routing_csv(tables_dir / "primitive_routing.csv", primitive_routing)
        orders_dir = temporary / "diagnostic-probe-work-orders"
        orders_dir.mkdir(mode=0o700)
        work_order_paths: dict[str, str] = {}
        for item in probe_orders:
            if not isinstance(item, Mapping) or not isinstance(item.get("probe_id"), str):
                raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_PROBE_ORDER_INVALID")
            filename = f"{item['probe_id']}.json"
            _write_bytes_atomic(orders_dir / filename, _canonical_json_bytes(item))
            work_order_paths[str(item["probe_id"])] = str(destination / "diagnostic-probe-work-orders" / filename)

        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = Path(cas_root).resolve() if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(root)
            for name, payload, media_type in (
                ("failure_signature_bank_json", report_bytes, "application/json"),
                ("failure_signature_bank_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
            for index, path in enumerate(source_paths, start=1):
                ref = cas.put_bytes(path.read_bytes(), media_type=_media_type(path)).uri
                cas_refs[f"source_{index}_{path.stem}"] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)

        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-failure-signature-bank-manifest",
            "state": report["state"],
            "summary": report["summary"],
            "report_path": str(destination / "failure-signature-bank.json"),
            "markdown_path": str(destination / "failure-signature-bank.md"),
            "probe_work_order_paths": work_order_paths,
            "tables": {
                "diagnostic_probe_work_orders": str(destination / "tables" / "diagnostic_probe_work_orders.csv"),
                "primitive_routing": str(destination / "tables" / "primitive_routing.csv"),
            },
            "cas_refs": cas_refs,
            "side_effects": report["side_effects"],
            "limitations": report["limitations"],
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


def _write_probe_order_csv(path: Path, rows: Sequence[object]) -> None:
    fields = [
        "work_order_id",
        "environment",
        "signature",
        "probe_id",
        "role",
        "priority",
        "signal_contract",
        "verdict_exposure_allowed",
    ]
    _write_csv(path, rows, fields)


def _write_primitive_routing_csv(path: Path, rows: Sequence[object]) -> None:
    fields = [
        "environment",
        "primitive",
        "mechanism_family",
        "layer",
        "current_status",
        "evidence_level",
        "routing_decision",
        "target_failures",
        "transfer_preconditions",
        "known_anti_conditions",
        "next_gate",
    ]
    _write_csv(path, rows, fields)


def _write_csv(path: Path, rows: Sequence[object], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_CSV_ROW_INVALID")
            item = {}
            for field in fields:
                value = row.get(field, "")
                item[field] = ",".join(value) if isinstance(value, list) else value
            writer.writerow(item)


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Failure Signature Bank",
        "",
        f"State: `{report['state']}`",
        f"Environments: `{report['summary']['environment_count']}`",
        f"Failure signatures: `{report['summary']['failure_signature_count']}`",
        f"P0 diagnostic probe work orders: `{report['summary']['p0_probe_order_count']}`",
        "",
        "## Claim Boundary",
        "",
        str(report["transfer_contract"]["claim_boundary"]),
        "",
        "## Environments",
        "",
        "| Env | Regime | Status | Best primitive | Evidence | Signatures | Next action |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        signatures = ", ".join(item["signature"] for item in record["failure_signatures"])
        lines.append(
            "| {environment} | {regime} | `{current_status}` | `{current_best_primitive}` | `{evidence_level}` | {signatures} | {next_action} |".format(
                signatures=signatures,
                **record,
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Probe Work Orders",
            "",
            "| Probe | Env | Signature | Priority | Verdict exposure |",
            "|:--|:--|:--|:--|:--|",
        ]
    )
    for order in report["diagnostic_probe_work_orders"]:
        lines.append(
            f"| `{order['probe_id']}` | `{order['environment']}` | `{order['signature']}` | `{order['priority']}` | `{order['verdict_exposure_allowed']}` |"
        )
    lines.extend(["", "## Primitive Routing", "", "| Env | Primitive | Decision | Next gate |", "|:--|:--|:--|:--|"])
    for route in report["primitive_routing"]:
        lines.append(
            f"| `{route['environment']}` | `{route['primitive']}` | `{route['routing_decision']}` | {route['next_gate']} |"
        )
    lines.extend(["", "## Side Effects", ""])
    for key, value in report["side_effects"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    return "text/plain"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FailureSignatureBankError("FAILURE_SIGNATURE_BANK_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="export failure-signature bank and AutoProbe work orders")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--uplift-plan", type=Path, required=True)
    run.add_argument("--gap-staging-plan", type=Path)
    run.add_argument("--mechanism-cards", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        manifest = run_failure_signature_bank(
            repo_root=args.repo_root,
            uplift_plan=args.uplift_plan,
            gap_staging_plan=args.gap_staging_plan,
            mechanism_cards=args.mechanism_cards,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
