"""Audit primitive materialization before closed-loop campaigns use it.

The frozen registry lists the bounded action space.  This gate separates that
control-plane fact from the harder execution-plane fact: whether a primitive is
actually wired into ACWM training/evaluation code and backed by runtime
evidence.  Sidecar-only primitives are turned into explicit agent work orders
instead of being silently routed into formal experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.acwm_primitive_routes import (
    DIAGNOSTIC_ROUTING_PRIMITIVES,
    QUALITY_SCREEN_PRIMITIVES,
    primitive_execution_role,
)
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.vendor import verify_vendor_checkout


class PrimitiveMaterializationGateError(RuntimeError):
    """Primitive materialization evidence is incomplete or inconsistent."""


CLOSED_LOOP_READY = "closed_loop_runtime_ready"
HOOK_ONLY_READY = "hook_only_runtime_ready"
TEMPLATE_PRESENT = "runtime_hook_template_present"
SIDECAR_ONLY = "sidecar_only"
MISSING = "materialization_missing"


def run_primitive_materialization_gate(
    *,
    repo_root: Path,
    primitive_apply_manifest: Path,
    output_root: Path,
    evidence_manifests: Sequence[Path] = (),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    min_closed_loop_ready_for_t44: int = 3,
) -> dict[str, object]:
    """Write a materialization gate report and per-primitive work orders."""

    if min_closed_loop_ready_for_t44 < 1:
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_MIN_READY_INVALID")
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_OUTPUT_EXISTS")
    source_revision = verify_vendor_checkout(root)
    registry = PrimitiveRegistry.from_root(root)
    apply_report = _load_apply_report(primitive_apply_manifest)
    apply_records = _apply_records_by_primitive(apply_report, registry=registry)
    evidence = _collect_evidence(evidence_manifests)
    records = [
        _primitive_record(
            registry=registry,
            primitive=name,
            apply_record=apply_records.get(name),
            evidence=evidence.get(name, ()),
        )
        for name in registry.names()
    ]
    work_orders = [
        _work_order(
            source_revision=source_revision,
            registry=registry,
            record=record,
        )
        for record in records
        if record["closed_loop_eligible"] is not True
    ]
    closed_loop_ready = [record for record in records if record["admission_state"] == CLOSED_LOOP_READY]
    quality_screen_ready = [
        record for record in closed_loop_ready if str(record["primitive"]) in QUALITY_SCREEN_PRIMITIVES
    ]
    diagnostic_routing_ready = [
        record for record in closed_loop_ready if str(record["primitive"]) in DIAGNOSTIC_ROUTING_PRIMITIVES
    ]
    sidecar_only = [record for record in records if record["admission_state"] in {SIDECAR_ONLY, MISSING}]
    hook_only = [record for record in records if record["admission_state"] in {HOOK_ONLY_READY, TEMPLATE_PRESENT}]
    blockers: list[dict[str, object]] = []
    if sidecar_only:
        blockers.append(
            {
                "code": "sidecar_only_primitives_present",
                "primitive_count": len(sidecar_only),
                "primitives": [str(record["primitive"]) for record in sidecar_only],
            }
        )
    if hook_only:
        blockers.append(
            {
                "code": "runtime_hook_evidence_incomplete",
                "primitive_count": len(hook_only),
                "primitives": [str(record["primitive"]) for record in hook_only],
            }
        )
    if len(closed_loop_ready) < min_closed_loop_ready_for_t44:
        blockers.append(
            {
                "code": "t44_runtime_ready_primitive_count_below_minimum",
                "observed": len(closed_loop_ready),
                "minimum": min_closed_loop_ready_for_t44,
            }
        )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-materialization-gate",
        "state": "ready" if not blockers else "blocked",
        "source_revision": source_revision,
        "registry_digest": registry.digest(),
        "primitive_count": len(records),
        "closed_loop_ready_count": len(closed_loop_ready),
        "quality_screen_ready_count": len(quality_screen_ready),
        "diagnostic_routing_ready_count": len(diagnostic_routing_ready),
        "hook_only_ready_count": len([record for record in records if record["admission_state"] == HOOK_ONLY_READY]),
        "runtime_template_only_count": len([record for record in records if record["admission_state"] == TEMPLATE_PRESENT]),
        "sidecar_only_count": len(sidecar_only),
        "closed_loop_ready_primitives": [str(record["primitive"]) for record in closed_loop_ready],
        "quality_screen_ready_primitives": [str(record["primitive"]) for record in quality_screen_ready],
        "diagnostic_routing_ready_primitives": [str(record["primitive"]) for record in diagnostic_routing_ready],
        "records": records,
        "work_orders": work_orders,
        "blockers": blockers,
        "evidence_manifest_paths": [str(Path(path).resolve()) for path in evidence_manifests],
        "primitive_apply_manifest": str(Path(primitive_apply_manifest).resolve()),
        "limitations": [
            "A primitive in the frozen registry is not automatically execution-ready.",
            "Runtime readiness does not imply that a diagnostic-only primitive can enter a quality screen.",
            "Sidecar-only primitives are excluded from closed-loop evidence until a real ACWM hook passes admission.",
            "Generated work orders are staging inputs for agent code conversion; they do not promote code by themselves.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _load_apply_report(manifest_path: Path) -> Mapping[str, Any]:
    manifest = _load_json_object(manifest_path, "PRIMITIVE_MATERIALIZATION_APPLY_MANIFEST_INVALID")
    if manifest.get("artifact_type") != "wmloop-m2-primitive-apply-all-manifest":
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_MANIFEST_INVALID")
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_REPORT_MISSING")
    report = _load_json_object(Path(report_path), "PRIMITIVE_MATERIALIZATION_APPLY_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-m2-primitive-apply-all-report" or report.get("state") != "ready":
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_REPORT_INVALID")
    return report


def _apply_records_by_primitive(
    apply_report: Mapping[str, Any],
    *,
    registry: PrimitiveRegistry,
) -> dict[str, Mapping[str, Any]]:
    raw_records = apply_report.get("records")
    if not isinstance(raw_records, list):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_RECORDS_INVALID")
    output: dict[str, Mapping[str, Any]] = {}
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_RECORDS_INVALID")
        primitive = item.get("primitive")
        if not isinstance(primitive, str) or primitive not in registry.names() or primitive in output:
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_APPLY_RECORDS_INVALID")
        output[primitive] = item
    return output


def _collect_evidence(evidence_manifests: Sequence[Path]) -> dict[str, tuple[dict[str, object], ...]]:
    by_primitive: dict[str, list[dict[str, object]]] = {}
    for manifest_path in evidence_manifests:
        manifest = _load_json_object(manifest_path, "PRIMITIVE_MATERIALIZATION_EVIDENCE_INVALID")
        artifact_type = manifest.get("artifact_type")
        if artifact_type == "wmloop-m2-medium-primitive-runtime-smoke-manifest":
            evidence = _runtime_smoke_evidence(manifest, manifest_path=manifest_path)
            by_primitive.setdefault(str(evidence["primitive"]), []).append(evidence)
        elif artifact_type == "wmloop-m3-training-eval-smoke-manifest":
            for evidence in _training_eval_manifest_evidence(manifest, manifest_path=manifest_path):
                by_primitive.setdefault(str(evidence["primitive"]), []).append(evidence)
        elif artifact_type == "wmloop-training-eval-limited-campaign-manifest":
            for evidence in _campaign_evidence(manifest, manifest_path=manifest_path):
                by_primitive.setdefault(str(evidence["primitive"]), []).append(evidence)
        elif artifact_type == "wmloop-budget-overrun-salvage-report":
            evidence = _budget_overrun_salvage_evidence(manifest, manifest_path=manifest_path)
            by_primitive.setdefault(str(evidence["primitive"]), []).append(evidence)
        else:
            raise PrimitiveMaterializationGateError(f"PRIMITIVE_MATERIALIZATION_EVIDENCE_UNSUPPORTED:{artifact_type}")
    return {primitive: tuple(items) for primitive, items in by_primitive.items()}


def _runtime_smoke_evidence(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, object]:
    primitive = manifest.get("primitive")
    report_path = manifest.get("report_path")
    if not isinstance(primitive, str) or not isinstance(report_path, str):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_RUNTIME_SMOKE_INVALID")
    report = _load_json_object(Path(report_path), "PRIMITIVE_MATERIALIZATION_RUNTIME_SMOKE_INVALID")
    if report.get("artifact_type") != "wmloop-m2-medium-primitive-runtime-smoke-report":
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_RUNTIME_SMOKE_INVALID")
    state = str(manifest.get("state"))
    run_training = report.get("run_training") is True
    return {
        "primitive": primitive,
        "evidence_type": "primitive_runtime_smoke",
        "state": state,
        "run_training": run_training,
        "closed_loop_ready": state == "ready" and run_training,
        "hook_only_ready": state in {"ready", "hook_only"},
        "manifest_path": str(Path(manifest_path).resolve()),
        "report_path": str(Path(report_path).resolve()),
    }


def _campaign_evidence(manifest: Mapping[str, Any], *, manifest_path: Path) -> list[dict[str, object]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_CAMPAIGN_RECORDS_INVALID")
    evidence: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("state") != "ready":
            continue
        child_path = record.get("manifest_path")
        if not isinstance(child_path, str) or not child_path:
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_CAMPAIGN_CHILD_MISSING")
        child = _load_json_object(Path(child_path), "PRIMITIVE_MATERIALIZATION_CAMPAIGN_CHILD_INVALID")
        if child.get("artifact_type") != "wmloop-m3-training-eval-smoke-manifest":
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_CAMPAIGN_CHILD_INVALID")
        evidence.extend(_training_eval_manifest_evidence(child, manifest_path=Path(child_path), parent_manifest_path=manifest_path))
    return evidence


def _training_eval_manifest_evidence(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    parent_manifest_path: Path | None = None,
) -> list[dict[str, object]]:
    if manifest.get("state") not in {"ready", "checks_failed"}:
        return []
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_REPORT_MISSING")
    report = _load_json_object(Path(report_path), "PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_REPORT_INVALID")
    if report.get("artifact_type") != "wmloop-m3-training-eval-smoke-report":
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_REPORT_INVALID")
    receipt = report.get("receipt")
    if not isinstance(receipt, Mapping):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_RECEIPT_INVALID")
    rendered = receipt.get("rendered_primitives")
    if not isinstance(rendered, list):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_RENDERED_INVALID")
    output: list[dict[str, object]] = []
    for item in rendered:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_TRAINING_EVAL_RENDERED_INVALID")
        output.append(
            {
                "primitive": str(item["name"]),
                "evidence_type": "training_eval_closed_loop",
                "state": manifest.get("state"),
                "closed_loop_ready": manifest.get("state") == "ready" and report.get("state") == "ready",
                "hook_only_ready": False,
                "manifest_path": str(Path(manifest_path).resolve()),
                "parent_manifest_path": str(Path(parent_manifest_path).resolve()) if parent_manifest_path else None,
                "report_path": str(Path(report_path).resolve()),
                "environment": manifest.get("environment"),
                "verdict": manifest.get("verdict"),
            }
        )
    return output


def _budget_overrun_salvage_evidence(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, object]:
    primitive = manifest.get("primitive")
    rows = manifest.get("rows")
    if not isinstance(primitive, str) or not primitive.strip() or not isinstance(rows, list) or not rows:
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_BUDGET_SALVAGE_INVALID")
    ready_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_BUDGET_SALVAGE_INVALID")
        receipt_ref = row.get("receipt_ref")
        if (
            row.get("state") == "ready"
            and isinstance(receipt_ref, str)
            and receipt_ref.startswith("cas://sha256/")
            and row.get("required_horizons_complete") is True
        ):
            ready_rows.append(row)
    if manifest.get("state") != "ready" or not ready_rows:
        return {
            "primitive": primitive,
            "evidence_type": "budget_overrun_salvage_closed_loop",
            "state": manifest.get("state"),
            "closed_loop_ready": False,
            "hook_only_ready": False,
            "manifest_path": str(Path(manifest_path).resolve()),
            "decision": manifest.get("decision"),
            "quality_signal_usable_for_uplift": False,
            "reason": "salvage_report_not_ready_or_no_complete_ready_rows",
        }
    return {
        "primitive": primitive,
        "evidence_type": "budget_overrun_salvage_closed_loop",
        "state": manifest.get("state"),
        "closed_loop_ready": True,
        "hook_only_ready": False,
        "manifest_path": str(Path(manifest_path).resolve()),
        "decision": manifest.get("decision"),
        "quality_signal_usable_for_uplift": False,
        "quality_evidence_salvaged_from": manifest.get("quality_evidence_salvaged_from"),
        "seed_count": manifest.get("seed_count"),
        "ready_receipt_count": len(ready_rows),
        "positive_seed_count": manifest.get("positive_seed_count"),
        "negative_seed_count": manifest.get("negative_seed_count"),
        "limitations": list(manifest.get("limitations", ())) if isinstance(manifest.get("limitations"), list) else [],
    }


def _primitive_record(
    *,
    registry: PrimitiveRegistry,
    primitive: str,
    apply_record: Mapping[str, Any] | None,
    evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifest = registry.manifest(primitive)
    materialization_state = apply_record.get("materialization_state") if apply_record is not None else None
    if any(item.get("closed_loop_ready") is True for item in evidence):
        admission_state = CLOSED_LOOP_READY
    elif any(item.get("hook_only_ready") is True for item in evidence):
        admission_state = HOOK_ONLY_READY
    elif materialization_state == "acwm_runtime_hook_smoke" and _apply_record_runtime_template_complete(apply_record):
        admission_state = TEMPLATE_PRESENT
    elif materialization_state == "smoke_sidecar_only":
        admission_state = SIDECAR_ONLY
    else:
        admission_state = MISSING
    return {
        "primitive": primitive,
        "layer": manifest.layer,
        "hooks": list(manifest.hooks),
        "targets_failures": list(manifest.targets_failures),
        "cost_class": manifest.cost_class,
        "apply_module": manifest.apply_module,
        "materialization_state": materialization_state,
        "admission_state": admission_state,
        "execution_role": primitive_execution_role(primitive),
        "closed_loop_eligible": admission_state == CLOSED_LOOP_READY,
        "quality_screen_eligible": admission_state == CLOSED_LOOP_READY and primitive in QUALITY_SCREEN_PRIMITIVES,
        "diagnostic_routing_eligible": (
            admission_state == CLOSED_LOOP_READY and primitive in DIAGNOSTIC_ROUTING_PRIMITIVES
        ),
        "evidence": [dict(item) for item in evidence],
        "changed_paths_from_apply": list(apply_record.get("changed_paths_from_apply", ())) if apply_record else [],
        "next_action": _next_action(admission_state),
    }


def _apply_record_runtime_template_complete(apply_record: Mapping[str, Any] | None) -> bool:
    if apply_record is None:
        return False
    hook_paths = apply_record.get("runtime_hook_paths")
    if not isinstance(hook_paths, list) or not hook_paths:
        return False
    if any(not isinstance(path, str) or not path.strip() for path in hook_paths):
        return False
    return _intent_contract_complete(apply_record.get("intent_to_code_contract"))


def _intent_contract_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    for field in ("method_intent", "runtime_behavior", "declared_proxy"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            return False
    not_claimed = value.get("not_claimed")
    return isinstance(not_claimed, list) and bool(not_claimed) and all(
        isinstance(item, str) and item.strip() for item in not_claimed
    )


def _next_action(admission_state: str) -> str:
    if admission_state == CLOSED_LOOP_READY:
        return "eligible_for_closed_loop_campaigns"
    if admission_state == HOOK_ONLY_READY:
        return "run_gpu_training_eval_smoke_before_formal_use"
    if admission_state == TEMPLATE_PRESENT:
        return "run_runtime_smoke_and_training_eval_evidence"
    return "materialize_real_acwm_hook_then_run_admission_gates"


def _work_order(
    *,
    source_revision: str,
    registry: PrimitiveRegistry,
    record: Mapping[str, object],
) -> dict[str, object]:
    primitive = str(record["primitive"])
    manifest = registry.manifest(primitive)
    current_state = str(record["admission_state"])
    gates = [
        "schema_valid",
        "clean_diff_no_frozen_evaluator",
        "primitive_apply_audit_passed",
        "runtime_hook_unit_passed",
        "gpu_training_smoke_passed",
        "human_approved_version_boundary",
    ]
    if current_state in {HOOK_ONLY_READY, TEMPLATE_PRESENT}:
        gates = gates[3:]
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-materialization-work-order",
        "primitive": primitive,
        "current_admission_state": current_state,
        "target_admission_state": CLOSED_LOOP_READY,
        "source_revision": source_revision,
        "registry_digest": registry.digest(),
        "layer": manifest.layer,
        "hooks": list(manifest.hooks),
        "targets_failures": list(manifest.targets_failures),
        "params_schema": manifest.params_schema,
        "apply_module": manifest.apply_module,
        "required_gates": gates,
        "allowed_mutation_paths": _allowed_mutation_paths(primitive),
        "forbidden_paths": ["eval.py", "scripts/eval_all.sh", "results/", "configs/goal/", "runs/m0/protocol/"],
        "suggested_checks": [
            f"uv run pytest -q tests/test_primitive_render.py tests/test_primitive_apply_audit.py -k {primitive}",
            "uv run python -m wmloop.execute.primitive_apply_audit run --repo-root . --output-root <new-output-root>",
            "uv run python -m wmloop.execute.primitive_runtime_smoke run --repo-root . --output-root <new-output-root> --runtime-python <python> --data-root <data-root> --checkpoint-root <ckpt-root>",
        ],
        "promotion_rule": (
            "Promotion only records readiness after all required gates pass; it does not mutate the frozen registry "
            "without a human-approved version boundary."
        ),
    }


def _allowed_mutation_paths(primitive: str) -> list[str]:
    return [
        f"wmloop/primitives/definitions/{primitive}/apply.py",
        f"wmloop/primitives/definitions/{primitive}/templates/",
        f"tests/test_{primitive}.py",
        "tests/test_primitive_render.py",
        "tests/test_primitive_apply_audit.py",
        "tests/test_primitive_runtime_smoke.py",
        "vendor/ACWM-Phys/acwm/wmloop_hooks/",
        "vendor/ACWM-Phys/acwm/trainer/train_dynamics.py",
        "vendor/ACWM-Phys/acwm/dynamics/",
    ]


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    work_orders = report.get("work_orders")
    if not isinstance(work_orders, list):
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_WORK_ORDERS_INVALID")
    cas_refs: dict[str, str] = {}
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "primitive-materialization-gate.json", report_bytes)
        _write_bytes_atomic(temporary / "primitive-materialization-gate.md", markdown_bytes)
        work_order_dir = temporary / "work-orders"
        work_order_dir.mkdir(mode=0o700)
        work_order_paths: dict[str, str] = {}
        for item in work_orders:
            if not isinstance(item, Mapping) or not isinstance(item.get("primitive"), str):
                raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_WORK_ORDERS_INVALID")
            filename = f"{item['primitive']}.json"
            _write_bytes_atomic(work_order_dir / filename, _canonical_json_bytes(item))
            work_order_paths[str(item["primitive"])] = str(destination / "work-orders" / filename)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("primitive_materialization_gate_json", report_bytes, "application/json"),
                ("primitive_materialization_gate_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-primitive-materialization-gate-manifest",
            "state": report["state"],
            "primitive_count": report["primitive_count"],
            "closed_loop_ready_count": report["closed_loop_ready_count"],
            "quality_screen_ready_count": report["quality_screen_ready_count"],
            "diagnostic_routing_ready_count": report["diagnostic_routing_ready_count"],
            "sidecar_only_count": report["sidecar_only_count"],
            "closed_loop_ready_primitives": report["closed_loop_ready_primitives"],
            "quality_screen_ready_primitives": report["quality_screen_ready_primitives"],
            "diagnostic_routing_ready_primitives": report["diagnostic_routing_ready_primitives"],
            "blockers": report["blockers"],
            "report_path": str(destination / "primitive-materialization-gate.json"),
            "markdown_path": str(destination / "primitive-materialization-gate.md"),
            "work_order_paths": work_order_paths,
            "cas_refs": cas_refs,
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Primitive Materialization Gate",
        "",
        f"State: `{report['state']}`",
        f"Closed-loop ready: `{report['closed_loop_ready_count']}/{report['primitive_count']}`",
        f"Quality-screen ready: `{report['quality_screen_ready_count']}/{report['primitive_count']}`",
        f"Diagnostic-routing ready: `{report['diagnostic_routing_ready_count']}/{report['primitive_count']}`",
        "",
        "| Primitive | Layer | Role | Materialization | Admission | Next action |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        lines.append(
            f"| `{record['primitive']}` | `{record['layer']}` | `{record['execution_role']}` | "
            f"`{record.get('materialization_state')}` | "
            f"`{record['admission_state']}` | {record['next_action']} |"
        )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _load_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveMaterializationGateError(code) from exc
    if not isinstance(payload, Mapping):
        raise PrimitiveMaterializationGateError(code)
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PrimitiveMaterializationGateError("PRIMITIVE_MATERIALIZATION_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit primitive materialization readiness")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--primitive-apply-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--evidence-manifest", action="append", type=Path, default=[])
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--min-closed-loop-ready-for-t44", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_primitive_materialization_gate(
            repo_root=args.repo_root,
            primitive_apply_manifest=args.primitive_apply_manifest,
            output_root=args.output_root,
            evidence_manifests=args.evidence_manifest,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            min_closed_loop_ready_for_t44=args.min_closed_loop_ready_for_t44,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
