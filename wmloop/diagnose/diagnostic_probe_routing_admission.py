"""Gate diagnostic probe outputs before they can influence primitive routing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class DiagnosticProbeRoutingAdmissionError(RuntimeError):
    """Diagnostic probe routing admission failed closed."""


RUNTIME_READY_STATES = {"runtime_smoke_passed", "runtime_ready", "routing_ready"}
FIXTURE_ADMITTED_STATES = {"fixture_unit_admitted"}


def run_diagnostic_probe_routing_admission(
    *,
    failure_signature_bank_manifest: Path,
    probe_admission_manifest: Path,
    primitive_materialization_gate_manifest: Path,
    output_root: Path,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Write a control-plane report for diagnostic-probe-driven route authority.

    The gate intentionally separates two stages:
    - fixture/unit admitted probes may produce route previews only.
    - dev-split runtime smoke is required before a route may drive canary launch.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_OUTPUT_EXISTS")

    bank_manifest_path = Path(failure_signature_bank_manifest).resolve(strict=True)
    probe_manifest_path = Path(probe_admission_manifest).resolve(strict=True)
    primitive_manifest_path = Path(primitive_materialization_gate_manifest).resolve(strict=True)
    bank_manifest = _load_json_object(bank_manifest_path, "DIAGNOSTIC_PROBE_ROUTING_BANK_MANIFEST_INVALID")
    bank_report_path = _manifest_report_path(bank_manifest, key="report_path", code="DIAGNOSTIC_PROBE_ROUTING_BANK_MANIFEST_INVALID")
    bank_report = _load_json_object(bank_report_path, "DIAGNOSTIC_PROBE_ROUTING_BANK_REPORT_INVALID")
    probe_manifest = _load_json_object(probe_manifest_path, "DIAGNOSTIC_PROBE_ROUTING_PROBE_MANIFEST_INVALID")
    primitive_manifest = _load_json_object(primitive_manifest_path, "DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_GATE_INVALID")

    _require_artifact(bank_manifest, "wmloop-failure-signature-bank-manifest", "DIAGNOSTIC_PROBE_ROUTING_BANK_MANIFEST_INVALID")
    _require_artifact(bank_report, "wmloop-failure-signature-bank", "DIAGNOSTIC_PROBE_ROUTING_BANK_REPORT_INVALID")
    _require_artifact(probe_manifest, "wmloop-diagnostic-probe-admission-manifest", "DIAGNOSTIC_PROBE_ROUTING_PROBE_MANIFEST_INVALID")
    _require_artifact(primitive_manifest, "wmloop-primitive-materialization-gate", "DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_GATE_INVALID")

    admitted = _admitted_probe_map(probe_manifest)
    signature_rows = _signature_rows_by_environment(bank_report)
    probe_rows = [_probe_row(order, admitted=admitted) for order in _list_of_mappings(bank_report, "diagnostic_probe_work_orders")]
    probe_rows_by_env = _probe_rows_by_environment(probe_rows)
    primitive_rows = [
        _primitive_route_row(
            route,
            primitive_gate=_primitive_record_map(primitive_manifest),
            signatures=signature_rows.get(str(route.get("environment")), []),
            env_probe_rows=probe_rows_by_env.get(str(route.get("environment")), []),
        )
        for route in _list_of_mappings(bank_report, "primitive_routing")
    ]
    summary = {
        "probe_work_order_count": len(probe_rows),
        "admitted_probe_count": sum(row["admission_state"] in (RUNTIME_READY_STATES | FIXTURE_ADMITTED_STATES) for row in probe_rows),
        "runtime_ready_probe_count": sum(row["routing_authority"] == "routing_ready" for row in probe_rows),
        "primitive_route_count": len(primitive_rows),
        "routing_ready_route_count": sum(row["route_gate"] == "canary_after_gpu_free" for row in primitive_rows),
        "preview_only_route_count": sum(row["route_gate"] == "preview_only_runtime_smoke_required" for row in primitive_rows),
        "blocked_route_count": sum(row["route_gate"].startswith("blocked") for row in primitive_rows),
    }
    blockers = _blockers(probe_rows=probe_rows, primitive_rows=primitive_rows)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-routing-admission",
        "state": _state(summary=summary, blockers=blockers),
        "sources": {
            "failure_signature_bank_manifest": str(bank_manifest_path),
            "failure_signature_bank_report": str(bank_report_path),
            "probe_admission_manifest": str(probe_manifest_path),
            "primitive_materialization_gate_manifest": str(primitive_manifest_path),
        },
        "summary": summary,
        "probe_rows": probe_rows,
        "primitive_rows": primitive_rows,
        "blockers": blockers,
        "side_effects": {
            "source_code_mutated": False,
            "goal_config_mutated": False,
            "constitution_mutated": False,
            "probe_registry_mutated": False,
            "verdict_probe_mutated": False,
            "primitive_registry_mutated": False,
            "gpu_execution_started": False,
            "formal_verdict_mutated": False,
        },
        "limitations": [
            "This gate is a CPU/control-plane artifact; it does not run probe adapters, training, or evaluation.",
            "Fixture/unit admitted probes can only produce route previews; dev-split runtime smoke is required before canary launch.",
            "Diagnostic probes remain excluded from verdict_evidence unless promoted through a human-approved version boundary before a new campaign.",
            "Primitive routes still require primitive materialization gates, GPU exclusivity, canary evidence, and frozen formal verdicts.",
        ],
    }
    try:
        validate_document("diagnostic_probe_routing_admission", report, root=root)
    except ContractValidationError as exc:
        raise DiagnosticProbeRoutingAdmissionError(f"DIAGNOSTIC_PROBE_ROUTING_CONTRACT_INVALID:{exc}") from exc
    return _write_bundle(report=report, output_root=destination)


def _admitted_probe_map(probe_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = probe_manifest.get("probes")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_PROBE_MANIFEST_INVALID")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        probe_id = row.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_PROBE_MANIFEST_INVALID")
        out[probe_id] = row
    return out


def _probe_row(order: Mapping[str, Any], *, admitted: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    probe_id = _string(order, "probe_id", "DIAGNOSTIC_PROBE_ROUTING_PROBE_ORDER_INVALID")
    admission = admitted.get(probe_id)
    admission_state = _admission_state(admission)
    runtime_state = _runtime_state(admission)
    verdict_exposure = bool(order.get("verdict_exposure_allowed")) or bool(admission.get("verdict_exposure_allowed")) if admission else bool(order.get("verdict_exposure_allowed"))
    routing_authority = _routing_authority(
        admission_state=admission_state,
        runtime_state=runtime_state,
        verdict_exposure_allowed=verdict_exposure,
    )
    return {
        "probe_id": probe_id,
        "environment": _string(order, "environment", "DIAGNOSTIC_PROBE_ROUTING_PROBE_ORDER_INVALID"),
        "signature": _string(order, "signature", "DIAGNOSTIC_PROBE_ROUTING_PROBE_ORDER_INVALID"),
        "priority": _string(order, "priority", "DIAGNOSTIC_PROBE_ROUTING_PROBE_ORDER_INVALID"),
        "work_order_present": True,
        "admission_state": admission_state,
        "runtime_smoke_on_dev_split": runtime_state,
        "routing_authority": routing_authority,
        "verdict_exposure_allowed": verdict_exposure,
    }


def _primitive_route_row(
    route: Mapping[str, Any],
    *,
    primitive_gate: Mapping[str, Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
    env_probe_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    environment = _string(route, "environment", "DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_ROUTE_INVALID")
    primitive = _string(route, "primitive", "DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_ROUTE_INVALID")
    decision = _string(route, "routing_decision", "DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_ROUTE_INVALID")
    primitive_record = primitive_gate.get(primitive, {})
    primitive_state = str(primitive_record.get("admission_state") or "primitive_gate_record_missing")
    closed_loop_eligible = bool(primitive_record.get("closed_loop_eligible"))
    matched_probe_ids = _matched_probe_ids(route=route, signatures=signatures, env_probe_rows=env_probe_rows)
    env_authority = _environment_probe_authority(env_probe_rows)
    route_gate, reason = _route_gate(
        routing_decision=decision,
        closed_loop_eligible=closed_loop_eligible,
        environment_probe_authority=env_authority,
        matched_probe_ids=matched_probe_ids,
    )
    return {
        "environment": environment,
        "primitive": primitive,
        "routing_decision": decision,
        "primitive_admission_state": primitive_state,
        "closed_loop_eligible": closed_loop_eligible,
        "matched_probe_ids": matched_probe_ids,
        "environment_probe_authority": env_authority,
        "route_gate": route_gate,
        "reason": reason,
    }


def _primitive_record_map(primitive_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = primitive_manifest.get("records")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_PRIMITIVE_GATE_INVALID")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        primitive = row.get("primitive")
        if isinstance(primitive, str) and primitive:
            out[primitive] = row
    return out


def _signature_rows_by_environment(bank_report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in _list_of_mappings(bank_report, "records"):
        env = _string(record, "environment", "DIAGNOSTIC_PROBE_ROUTING_BANK_REPORT_INVALID")
        signatures = record.get("failure_signatures")
        if not isinstance(signatures, list) or any(not isinstance(item, Mapping) for item in signatures):
            raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_BANK_REPORT_INVALID")
        out[env].extend(signatures)
    return dict(out)


def _probe_rows_by_environment(probe_rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    out: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in probe_rows:
        out[str(row["environment"])].append(row)
    return dict(out)


def _matched_probe_ids(
    *,
    route: Mapping[str, Any],
    signatures: Sequence[Mapping[str, Any]],
    env_probe_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    target_failures = route.get("target_failures")
    targets = {str(item) for item in target_failures if isinstance(item, str)} if isinstance(target_failures, list) else set()
    matched = [
        str(signature["diagnostic_probe_id"])
        for signature in signatures
        if isinstance(signature.get("diagnostic_probe_id"), str)
        and (
            str(signature.get("signature")) in targets
            or str(signature.get("wm_dx_failure_family")) in targets
            or not targets
        )
    ]
    if matched:
        return sorted(set(matched))
    return sorted(str(row["probe_id"]) for row in env_probe_rows)


def _environment_probe_authority(env_probe_rows: Sequence[Mapping[str, object]]) -> str:
    if not env_probe_rows:
        return "no_probe_work_orders"
    authorities = {str(row["routing_authority"]) for row in env_probe_rows}
    if authorities == {"routing_ready"}:
        return "routing_ready"
    if "blocked" in authorities:
        return "blocked"
    if "preview_only" in authorities or "routing_ready" in authorities:
        return "preview_only"
    return "blocked"


def _route_gate(
    *,
    routing_decision: str,
    closed_loop_eligible: bool,
    environment_probe_authority: str,
    matched_probe_ids: Sequence[str],
) -> tuple[str, str]:
    if routing_decision == "retain_as_source_exemplar":
        return "export_formal_evidence", "positive exemplar route does not need diagnostic probe authority"
    if routing_decision == "reject_or_demote_until_new_diagnosis":
        return "blocked_by_rejected_prior", "route is explicitly rejected until new diagnostic evidence is available"
    if not closed_loop_eligible:
        return "blocked_by_primitive_materialization", "primitive is not closed-loop eligible under the materialization gate"
    if not matched_probe_ids:
        return "blocked_by_missing_diagnostic_probe", "no diagnostic probe work order covers this environment route"
    if environment_probe_authority == "routing_ready":
        return "canary_after_gpu_free", "diagnostic probes have runtime authority and primitive is closed-loop eligible"
    if environment_probe_authority == "preview_only":
        return "preview_only_runtime_smoke_required", "diagnostic probes passed fixture/unit gates but still need dev-split runtime smoke"
    return "blocked_by_diagnostic_probe_admission", "diagnostic probe admission is missing or unsafe"


def _admission_state(admission: Mapping[str, Any] | None) -> str:
    if admission is None:
        return "not_admitted"
    state = admission.get("admission_state") or admission.get("runtime_status")
    if isinstance(state, str) and state:
        return state
    return "fixture_unit_admitted"


def _runtime_state(admission: Mapping[str, Any] | None) -> str:
    if admission is None:
        return "not_run"
    state = admission.get("runtime_smoke_on_dev_split")
    if isinstance(state, str) and state:
        return state
    return "not_run"


def _routing_authority(
    *,
    admission_state: str,
    runtime_state: str,
    verdict_exposure_allowed: bool,
) -> str:
    if verdict_exposure_allowed:
        return "blocked"
    if admission_state in RUNTIME_READY_STATES or runtime_state == "passed":
        return "routing_ready"
    if admission_state in FIXTURE_ADMITTED_STATES:
        return "preview_only"
    return "blocked"


def _blockers(
    *,
    probe_rows: Sequence[Mapping[str, object]],
    primitive_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers = []
    missing = sorted(str(row["probe_id"]) for row in probe_rows if row["routing_authority"] == "blocked")
    if missing:
        blockers.append({"code": "diagnostic_probe_admission_missing_or_blocked", "probe_count": len(missing), "probes": missing})
    not_closed_loop = sorted(
        {str(row["primitive"]) for row in primitive_rows if row["route_gate"] == "blocked_by_primitive_materialization"}
    )
    if not_closed_loop:
        blockers.append({"code": "routed_primitives_not_closed_loop_eligible", "primitive_count": len(not_closed_loop), "primitives": not_closed_loop})
    return blockers


def _state(*, summary: Mapping[str, int], blockers: Sequence[Mapping[str, object]]) -> str:
    if summary["routing_ready_route_count"] > 0:
        return "routing_ready"
    if summary["preview_only_route_count"] > 0:
        return "preview_only"
    if blockers:
        return "blocked"
    return "preview_only"


def _write_bundle(*, report: Mapping[str, object], output_root: Path) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "diagnostic-probe-routing-admission.json", report_bytes)
        _write_bytes_atomic(temporary / "diagnostic-probe-routing-admission.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-diagnostic-probe-routing-admission-manifest",
            "state": report["state"],
            "summary": report["summary"],
            "report_path": str(destination / "diagnostic-probe-routing-admission.json"),
            "markdown_path": str(destination / "diagnostic-probe-routing-admission.md"),
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnostic Probe Routing Admission",
        "",
        f"State: `{report['state']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Primitive Routes",
            "",
            "| Environment | Primitive | Gate | Probe authority | Reason |",
            "|:--|:--|:--|:--|:--|",
        ]
    )
    for row in report["primitive_rows"]:
        lines.append(
            f"| `{row['environment']}` | `{row['primitive']}` | `{row['route_gate']}` | `{row['environment_probe_authority']}` | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Probe Authority",
            "",
            "| Probe | Env | Signature | Admission | Runtime | Authority |",
            "|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for row in report["probe_rows"]:
        lines.append(
            f"| `{row['probe_id']}` | `{row['environment']}` | `{row['signature']}` | `{row['admission_state']}` | `{row['runtime_smoke_on_dev_split']}` | `{row['routing_authority']}` |"
        )
    lines.extend(["", "## Side Effects", ""])
    for key, value in report["side_effects"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _manifest_report_path(manifest: Mapping[str, Any], *, key: str, code: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise DiagnosticProbeRoutingAdmissionError(code)
    return Path(value).resolve(strict=True)


def _load_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticProbeRoutingAdmissionError(code) from exc
    if not isinstance(payload, Mapping):
        raise DiagnosticProbeRoutingAdmissionError(code)
    return payload


def _require_artifact(payload: Mapping[str, Any], expected: str, code: str) -> None:
    if payload.get("artifact_type") != expected:
        raise DiagnosticProbeRoutingAdmissionError(code)


def _list_of_mappings(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise DiagnosticProbeRoutingAdmissionError(f"DIAGNOSTIC_PROBE_ROUTING_LIST_INVALID:{key}")
    return value


def _string(payload: Mapping[str, Any], key: str, code: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DiagnosticProbeRoutingAdmissionError(code)
    return value


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticProbeRoutingAdmissionError("DIAGNOSTIC_PROBE_ROUTING_OUTPUT_EXISTS")
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
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--failure-signature-bank-manifest", type=Path, required=True)
    parser.add_argument("--probe-admission-manifest", type=Path, required=True)
    parser.add_argument("--primitive-materialization-gate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = run_diagnostic_probe_routing_admission(
        repo_root=args.repo_root,
        failure_signature_bank_manifest=args.failure_signature_bank_manifest,
        probe_admission_manifest=args.probe_admission_manifest,
        primitive_materialization_gate_manifest=args.primitive_materialization_gate_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
