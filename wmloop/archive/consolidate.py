"""One-click M5 consolidation export with fail-closed claim boundaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ConsolidateError(RuntimeError):
    """The consolidation export failed closed."""


def run_consolidation(
    *,
    output_root: Path,
    archive_db: Path,
    cas_root: Path | None = None,
    phase_gate_manifest: Path | None = None,
    m3_acceptance_manifest: Path | None = None,
    surrogate_readiness_manifest: Path | None = None,
    surrogate_benefit_manifest: Path | None = None,
    adversarial_gate_audit_manifest: Path | None = None,
    defensive_outputs_manifest: Path | None = None,
    results_backup_manifest: Path | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    cell_projection_manifest: Path | None = None,
    raw_failure_manifest: Path | None = None,
    proposal_readiness_manifest: Path | None = None,
    manual_proposal_packet_manifest: Path | None = None,
    m4_unblock_plan_manifest: Path | None = None,
) -> dict[str, object]:
    """Export current paper-table readiness without inventing missing results."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ConsolidateError("CONSOLIDATE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    manifests = _load_named_manifests(
        {
            "phase_gate": phase_gate_manifest,
            "m3_acceptance": m3_acceptance_manifest,
            "surrogate_readiness": surrogate_readiness_manifest,
            "surrogate_benefit": surrogate_benefit_manifest,
            "adversarial_gate_audit": adversarial_gate_audit_manifest,
            "defensive_outputs": defensive_outputs_manifest,
            "results_backup": results_backup_manifest,
            "gpu_exclusivity_audit": gpu_exclusivity_audit_manifest,
            "cell_projection": cell_projection_manifest,
            "raw_failure": raw_failure_manifest,
            "proposal_readiness": proposal_readiness_manifest,
            "manual_proposal_packet": manual_proposal_packet_manifest,
            "m4_unblock_plan": m4_unblock_plan_manifest,
        }
    )
    stats = archive.archive_statistics()
    cells = archive.list_cells()
    archive_summary = {
        **stats,
        "cells": len(cells),
        "cell_observations": sum(record.stats.visits for record in cells),
    }
    claim_rows = _claim_status_rows(archive_summary=archive_summary, manifests=manifests)
    export_rows = _m5_export_rows(archive_summary=archive_summary, manifests=manifests)
    blocker_rows = _blocker_rows(manifests)
    archive_rows = [{"metric": key, "value": value} for key, value in archive_summary.items()]
    cost_rows = _cost_rows(Path(archive_db).resolve())
    state = "ready" if all(row["state"] == "ready" for row in export_rows) else "blocked"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m5-consolidation-report",
        "state": state,
        "m5_export_complete": state == "ready",
        "archive_db": str(Path(archive_db).resolve()),
        "archive_summary": archive_summary,
        "source_manifests": {
            name: {
                "path": item["path"],
                "artifact_type": item["payload"].get("artifact_type"),
                "state": item["payload"].get("state"),
            }
            for name, item in manifests.items()
        },
        "claim_status": claim_rows,
        "m5_exports": export_rows,
        "blockers": blocker_rows,
        "cost_summary": cost_rows,
        "limitations": [
            "This export is an auditable status surface; it does not launch training or evaluation.",
            "Blocked rows are exported as blockers, not as paper-ready experimental numbers.",
            "Surrogate outputs are permitted for proposal sorting only and are never verifier inputs.",
        ],
    }
    files = {
        "consolidation_report_json": ("consolidation-report.json", _canonical_json_bytes(report), "application/json"),
        "consolidation_report_markdown": (
            "consolidation-report.md",
            _render_markdown(report).encode("utf-8"),
            "text/markdown",
        ),
        "claim_status_csv": ("tables/claim-status.csv", _csv_bytes(claim_rows), "text/csv"),
        "m5_exports_csv": ("tables/m5-exports.csv", _csv_bytes(export_rows), "text/csv"),
        "blockers_csv": ("tables/blockers.csv", _csv_bytes(blocker_rows), "text/csv"),
        "archive_counts_csv": ("tables/archive-counts.csv", _csv_bytes(archive_rows), "text/csv"),
        "cost_summary_csv": ("tables/cost-summary.csv", _csv_bytes(cost_rows), "text/csv"),
        "m5_exports_latex": ("latex/m5-exports.tex", _latex_table(export_rows, "M5 Export Readiness").encode("utf-8"), "text/x-tex"),
        "claim_status_latex": (
            "latex/claim-status.tex",
            _latex_table(claim_rows, "Claim Status").encode("utf-8"),
            "text/x-tex",
        ),
    }
    try:
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        for key, (relative, payload, media_type) in files.items():
            _write_bytes_atomic(temporary / relative, payload)
            ref = cas.put_bytes(payload, media_type=media_type).uri
            archive.record_artifact_reference(ref)
            cas_refs[key] = ref
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m5-consolidation-manifest",
            "state": state,
            "m5_export_complete": state == "ready",
            "archive_summary": archive_summary,
            "claim_status_count": len(claim_rows),
            "m5_export_count": len(export_rows),
            "ready_export_count": sum(1 for row in export_rows if row["state"] == "ready"),
            "blocker_count": len(blocker_rows),
            "report_path": str(destination / "consolidation-report.json"),
            "markdown_path": str(destination / "consolidation-report.md"),
            "tables_dir": str(destination / "tables"),
            "latex_dir": str(destination / "latex"),
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


def _load_named_manifests(paths: Mapping[str, Path | None]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve(strict=True)
        payload = _load_json_mapping(resolved)
        loaded[name] = {"path": str(resolved), "payload": payload}
    return loaded


def _claim_status_rows(
    *,
    archive_summary: Mapping[str, int],
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    phase = _payload(manifests, "phase_gate")
    m3 = _payload(manifests, "m3_acceptance")
    surrogate = _payload(manifests, "surrogate_readiness")
    surrogate_benefit = _payload(manifests, "surrogate_benefit")
    adversarial = _payload(manifests, "adversarial_gate_audit")
    defensive = _payload(manifests, "defensive_outputs")
    backup = _payload(manifests, "results_backup")
    gpu = _payload(manifests, "gpu_exclusivity_audit")
    raw = _payload(manifests, "raw_failure")
    proposal = _payload(manifests, "proposal_readiness")
    manual = _payload(manifests, "manual_proposal_packet")
    unblock = _payload(manifests, "m4_unblock_plan")
    return [
        _row("M0_generation_zero_archive", "ready" if archive_summary.get("baselines") == 8 else "blocked", f"baselines={archive_summary.get('baselines', 0)}"),
        _row("M1_raw_failure_reports", _ready_if(raw.get("state") == "ready"), f"state={raw.get('state', 'missing')}; reports={raw.get('report_count', 'n/a')}/8"),
        _row("M3_strict_acceptance", _ready_if(m3.get("strict_m3_pass") is True), f"state={m3.get('state', 'missing')}; strict_m3_pass={m3.get('strict_m3_pass', False)}"),
        _row("M4_launch_gate", _ready_if(phase.get("m4_launch_allowed") is True), f"state={phase.get('state', 'missing')}; m4_launch_allowed={phase.get('m4_launch_allowed', False)}"),
        _row("GPU_exclusivity_preflight", _ready_if(gpu.get("gpu_exclusivity_ready") is True), f"state={gpu.get('state', 'missing')}; requested={gpu.get('requested_gpu_count', 'n/a')}; ready={gpu.get('ready_requested_gpu_count', 'n/a')}; blocked={gpu.get('blocked_requested_gpu_count', 'n/a')}"),
        _row("M4_unblock_dependency_plan", _ready_if(_unblock_plan_exportable(unblock)), f"state={unblock.get('state', 'missing')}; dependencies={unblock.get('dependency_count', 'n/a')}; formal_training_allowed={unblock.get('formal_training_allowed', False)}"),
        _row("M4_settled_trial_target", _ready_if(int(archive_summary.get("settled_trials", 0)) >= 150), f"settled_trials={archive_summary.get('settled_trials', 0)}/150"),
        _row("M5_surrogate_readiness", _ready_if(surrogate.get("surrogate_training_allowed") is True), f"state={surrogate.get('state', 'missing')}; training_allowed={surrogate.get('surrogate_training_allowed', False)}"),
        _row("T5_2_surrogate_benefit", _ready_if(_surrogate_benefit_ready(surrogate_benefit)), _surrogate_benefit_evidence(surrogate_benefit)),
        _row("T5_1_adversarial_gate_audit", _ready_if(_adversarial_ready(adversarial)), f"state={adversarial.get('state', 'missing')}; intercepted={adversarial.get('intercepted_required_attack_count', 'n/a')}/{adversarial.get('required_attack_count', 'n/a')}; rate={adversarial.get('interception_rate', 'n/a')}"),
        _row("T5_5_defensive_outputs", _ready_if(_defensive_outputs_ready(defensive)), _defensive_outputs_evidence(defensive)),
        _row("T3_1_proposal_readiness", _ready_if(proposal.get("strict_t3_1_pass") is True), f"state={proposal.get('state', 'missing')}; legal={proposal.get('legal_proposal_count', 'n/a')}/{proposal.get('strict_required_reports', 'n/a')}"),
        _row("T4_2_manual_proposal_preflight", _ready_if(_manual_proposal_preflight_exportable(manual)), f"state={manual.get('state', 'missing')}; validation={manual.get('proposal_validation_passed', False)}; executor_ready={manual.get('manual_proposal_executor_ready', False)}; m4_gate={manual.get('m4_launch_allowed_by_phase_gate', False)}"),
        _row("daily_results_backup", _ready_if(backup.get("state") == "ready" and backup.get("executed") is True), f"state={backup.get('state', 'missing')}; executed={backup.get('executed', False)}"),
    ]


def _m5_export_rows(
    *,
    archive_summary: Mapping[str, int],
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    phase = _payload(manifests, "phase_gate")
    m3 = _payload(manifests, "m3_acceptance")
    surrogate = _payload(manifests, "surrogate_readiness")
    surrogate_benefit = _payload(manifests, "surrogate_benefit")
    adversarial = _payload(manifests, "adversarial_gate_audit")
    defensive = _payload(manifests, "defensive_outputs")
    backup = _payload(manifests, "results_backup")
    gpu = _payload(manifests, "gpu_exclusivity_audit")
    cell = _payload(manifests, "cell_projection")
    manual = _payload(manifests, "manual_proposal_packet")
    unblock = _payload(manifests, "m4_unblock_plan")
    settled = int(archive_summary.get("settled_trials", 0))
    cells = int(archive_summary.get("cells", 0))
    observations = int(archive_summary.get("cell_observations", 0))
    return [
        _export_row("best_recipe", _ready_if(phase.get("m4_launch_allowed") is True and settled >= 150), f"m4_launch_allowed={phase.get('m4_launch_allowed', False)}; settled_trials={settled}/150"),
        _export_row("failure_intervention_matrix", _ready_if(cells > 0 and observations >= 100), f"archive_cells={cells}; cell_observations={observations}/100"),
        _export_row("transfer_matrix_8x8", _ready_if(settled >= 150 and m3.get("strict_m3_pass") is True), f"settled_trials={settled}/150; strict_m3_pass={m3.get('strict_m3_pass', False)}"),
        _export_row("prediction_measured_calibration", _ready_if(surrogate.get("surrogate_training_allowed") is True), f"surrogate_state={surrogate.get('state', 'missing')}; training_allowed={surrogate.get('surrogate_training_allowed', False)}"),
        _export_row("surrogate_sorting_benefit", _ready_if(_surrogate_benefit_ready(surrogate_benefit)), _surrogate_benefit_evidence(surrogate_benefit)),
        _export_row("adversarial_gate_interception", _ready_if(_adversarial_ready(adversarial)), f"intercepted={adversarial.get('intercepted_required_attack_count', 'n/a')}/{adversarial.get('required_attack_count', 'n/a')}; rate={adversarial.get('interception_rate', 'n/a')}"),
        _export_row("defensive_taxonomy_and_tau_sensitivity", _ready_if(_defensive_outputs_ready(defensive)), _defensive_outputs_evidence(defensive)),
        _export_row("cost_benefit", _ready_if(settled >= 150), f"settled_trials={settled}/150"),
        _export_row("cell_projection_evidence", _ready_if(cell.get("state") == "ready"), f"state={cell.get('state', 'missing')}; projected_cell_count={cell.get('projected_cell_count', 'n/a')}"),
        _export_row("daily_backup_evidence", _ready_if(backup.get("state") == "ready" and backup.get("executed") is True), f"state={backup.get('state', 'missing')}; executed={backup.get('executed', False)}"),
        _export_row("gpu_exclusivity_audit_evidence", _ready_if(_gpu_exclusivity_exportable(gpu)), f"state={gpu.get('state', 'missing')}; ready={gpu.get('ready_requested_gpu_count', 'n/a')}/{gpu.get('requested_gpu_count', 'n/a')}; blocked={gpu.get('blocked_requested_gpu_count', 'n/a')}; no_gpu_start={_gpu_audit_no_gpu_start(gpu)}; no_kill={_gpu_audit_no_kill(gpu)}"),
        _export_row("t4_2_manual_proposal_preflight", _ready_if(_manual_proposal_preflight_exportable(manual)), f"state={manual.get('state', 'missing')}; proposal={manual.get('proposal_id', 'n/a')}; validation={manual.get('proposal_validation_passed', False)}; executor_ready={manual.get('manual_proposal_executor_ready', False)}; no_gpu={_manual_no_gpu(manual)}; no_budget_debit={_manual_no_budget_debit(manual)}; m4_gate={manual.get('m4_launch_allowed_by_phase_gate', False)}"),
        _export_row("m4_unblock_dependency_plan", _ready_if(_unblock_plan_exportable(unblock)), f"state={unblock.get('state', 'missing')}; dependencies={unblock.get('dependency_count', 'n/a')}; critical_path={len(unblock.get('critical_path', [])) if isinstance(unblock.get('critical_path'), list) else 'n/a'}"),
    ]


def _blocker_rows(manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, item in manifests.items():
        payload = item["payload"]
        if _exportable_nonblocking_state(name, payload):
            continue
        blockers = payload.get("blockers")
        if isinstance(blockers, list) and blockers:
            for index, blocker in enumerate(blockers, start=1):
                rows.append(
                    {
                        "source": name,
                        "index": index,
                        "requirement": _blocker_field(blocker, "requirement"),
                        "reason": _blocker_field(blocker, "reason")
                        or _blocker_field(blocker, "code")
                        or _blocker_field(blocker, "expected"),
                        "observed": _compact_json(blocker.get("observed") if isinstance(blocker, Mapping) else blocker),
                    }
                )
        elif payload.get("state") not in ("ready", None) and not _exportable_nonblocking_state(name, payload):
            rows.append(
                {
                    "source": name,
                    "index": 1,
                    "requirement": "",
                    "reason": f"state={payload.get('state')}",
                    "observed": "",
                }
            )
    return rows


def _exportable_nonblocking_state(name: str, payload: Mapping[str, Any]) -> bool:
    if name == "manual_proposal_packet":
        return _manual_proposal_preflight_exportable(payload)
    if name == "m4_unblock_plan":
        return _unblock_plan_exportable(payload)
    return False


def _cost_rows(archive_db: Path) -> list[dict[str, object]]:
    import sqlite3

    rows: list[dict[str, object]] = []
    connection = sqlite3.connect(archive_db)
    connection.row_factory = sqlite3.Row
    try:
        total = 0.0
        count = 0
        for row in connection.execute("SELECT trial_id, cost_json FROM trials ORDER BY trial_id").fetchall():
            payload = json.loads(str(row["cost_json"]))
            gpu_hours = float(payload.get("gpu_hours", 0.0))
            total += gpu_hours
            count += 1
        rows.append({"metric": "settled_trial_count", "value": count})
        rows.append({"metric": "total_gpu_hours", "value": total})
        rows.append({"metric": "mean_gpu_hours_per_settled_trial", "value": total / count if count else 0.0})
    finally:
        connection.close()
    return rows


def _payload(manifests: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    item = manifests.get(name)
    if not item:
        return {}
    payload = item.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _row(requirement: str, state: str, evidence: str) -> dict[str, object]:
    return {"requirement": requirement, "state": state, "evidence": evidence}


def _export_row(export_id: str, state: str, evidence: str) -> dict[str, object]:
    return {"export_id": export_id, "state": state, "evidence": evidence}


def _ready_if(condition: bool) -> str:
    return "ready" if condition else "blocked"


def _surrogate_benefit_ready(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("artifact_type") == "wmloop-surrogate-benefit-manifest"
        and payload.get("state") == "ready"
        and payload.get("surrogate_benefit_measured") is True
        and payload.get("prediction_values_enter_reward_or_verdict") is False
        and isinstance(payload.get("benefit_summary"), Mapping)
    )


def _surrogate_benefit_evidence(payload: Mapping[str, Any]) -> str:
    summary = payload.get("benefit_summary")
    if not isinstance(summary, Mapping):
        return f"state={payload.get('state', 'missing')}; benefit_summary=missing"
    return (
        f"state={payload.get('state', 'missing')}; "
        f"first_positive_savings={summary.get('sorting_savings_factor_first_positive', 'n/a')}; "
        f"first_positive_rank={summary.get('surrogate_rank_first_positive', 'n/a')}; "
        f"best_cell_rank={summary.get('surrogate_rank_best_cell', 'n/a')}; "
        f"pairwise_concordance={summary.get('pairwise_concordance_rate', 'n/a')}"
    )


def _defensive_outputs_ready(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("artifact_type") == "wmloop-t5-5-defensive-outputs-manifest"
        and payload.get("state") == "ready"
        and payload.get("defensive_outputs_complete") is True
        and int(payload.get("taxonomy_row_count", 0)) >= 8
        and int(payload.get("sensitivity_row_count", 0)) >= 3
        and int(payload.get("blocker_count", 0)) == 0
    )


def _defensive_outputs_evidence(payload: Mapping[str, Any]) -> str:
    return (
        f"state={payload.get('state', 'missing')}; "
        f"taxonomy_rows={payload.get('taxonomy_row_count', 'n/a')}; "
        f"tau_sweep_rows={payload.get('sensitivity_row_count', 'n/a')}; "
        f"base_tau_af={payload.get('base_tau_af', 'n/a')}; "
        f"blockers={payload.get('blocker_count', 'n/a')}"
    )


def _adversarial_ready(payload: Mapping[str, Any]) -> bool:
    required = payload.get("required_attack_count")
    intercepted = payload.get("intercepted_required_attack_count")
    return (
        payload.get("state") == "ready"
        and isinstance(required, int)
        and isinstance(intercepted, int)
        and required >= 3
        and intercepted == required
        and payload.get("interception_rate") == 1.0
    )


def _unblock_plan_exportable(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("artifact_type") == "wmloop-m4-unblock-dependency-plan-manifest"
        and payload.get("state") in {"awaiting_external_and_human_resolution", "ready_for_m4_launch"}
        and isinstance(payload.get("dependency_count"), int)
        and isinstance(payload.get("critical_path"), list)
        and payload.get("formal_training_allowed") is payload.get("m4_launch_allowed")
    )


def _manual_proposal_preflight_exportable(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("artifact_type") == "wmloop-t4-2-manual-proposal-packet-manifest"
        and payload.get("state") in {"staged_ready", "staged_blocked"}
        and payload.get("proposal_validation_passed") is True
        and payload.get("packet_grants_m4_launch_permission") is False
        and _manual_no_gpu(payload)
        and _manual_no_budget_debit(payload)
    )


def _manual_no_gpu(payload: Mapping[str, Any]) -> bool:
    side_effects = payload.get("side_effects")
    if isinstance(side_effects, Mapping):
        return side_effects.get("gpu_execution_started") is False
    return payload.get("manual_proposal_executor_ready") is False


def _manual_no_budget_debit(payload: Mapping[str, Any]) -> bool:
    side_effects = payload.get("side_effects")
    if isinstance(side_effects, Mapping):
        return side_effects.get("automatic_search_budget_debited") is False
    return payload.get("manual_proposal_executor_ready") is False


def _gpu_exclusivity_exportable(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("artifact_type") == "wmloop-gpu-exclusivity-audit-manifest"
        and payload.get("state") in {"ready", "blocked"}
        and isinstance(payload.get("requested_gpu_count"), int)
        and isinstance(payload.get("ready_requested_gpu_count"), int)
        and isinstance(payload.get("blocked_requested_gpu_count"), int)
        and _gpu_audit_no_gpu_start(payload)
        and _gpu_audit_no_kill(payload)
        and _gpu_audit_no_active_training_mutation(payload)
        and payload.get("m4_launch_allowed") is False
        and payload.get("formal_training_allowed") is False
    )


def _gpu_audit_no_gpu_start(payload: Mapping[str, Any]) -> bool:
    side_effects = payload.get("side_effects")
    return isinstance(side_effects, Mapping) and side_effects.get("gpu_execution_started") is False


def _gpu_audit_no_kill(payload: Mapping[str, Any]) -> bool:
    side_effects = payload.get("side_effects")
    return isinstance(side_effects, Mapping) and side_effects.get("process_kill_attempted") is False


def _gpu_audit_no_active_training_mutation(payload: Mapping[str, Any]) -> bool:
    side_effects = payload.get("side_effects")
    return isinstance(side_effects, Mapping) and side_effects.get("active_training_mutated") is False


def _blocker_field(blocker: object, field: str) -> str:
    if isinstance(blocker, Mapping):
        value = blocker.get(field)
        return str(value) if value is not None else ""
    return ""


def _compact_json(value: object) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b"\n"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return output.getvalue().encode("utf-8")


def _latex_table(rows: Sequence[Mapping[str, object]], caption: str) -> str:
    if not rows:
        return "% empty table\n"
    columns = [str(key) for key in rows[0].keys()]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(row.get(column, ""))) for column in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", f"% {caption}", ""])
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _render_markdown(report: Mapping[str, object]) -> str:
    archive_summary = report["archive_summary"]
    lines = [
        "# M5 Consolidation Export",
        "",
        f"State: `{report['state']}`",
        f"M5 export complete: `{report['m5_export_complete']}`",
        (
            "Archive: "
            f"`baselines={archive_summary['baselines']}`, "
            f"`settled_trials={archive_summary['settled_trials']}`, "
            f"`artifacts={archive_summary['artifacts']}`, "
            f"`cells={archive_summary['cells']}`"
        ),
        "",
        "## M5 Exports",
        "",
        "| Export | State | Evidence |",
        "|:--|:--|:--|",
    ]
    for row in report["m5_exports"]:
        lines.append(f"| {row['export_id']} | {row['state']} | {row['evidence']} |")
    lines.extend(["", "## Claim Status", "", "| Requirement | State | Evidence |", "|:--|:--|:--|"])
    for row in report["claim_status"]:
        lines.append(f"| {row['requirement']} | {row['state']} | {row['evidence']} |")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsolidateError(f"CONSOLIDATE_MANIFEST_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ConsolidateError(f"CONSOLIDATE_MANIFEST_INVALID:{path}")
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ConsolidateError("CONSOLIDATE_OUTPUT_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="export current M5 consolidation readiness tables")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--phase-gate-manifest", type=Path)
    run.add_argument("--m3-acceptance-manifest", type=Path)
    run.add_argument("--surrogate-readiness-manifest", type=Path)
    run.add_argument("--surrogate-benefit-manifest", type=Path)
    run.add_argument("--adversarial-gate-audit-manifest", type=Path)
    run.add_argument("--defensive-outputs-manifest", type=Path)
    run.add_argument("--results-backup-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--cell-projection-manifest", type=Path)
    run.add_argument("--raw-failure-manifest", type=Path)
    run.add_argument("--proposal-readiness-manifest", type=Path)
    run.add_argument("--manual-proposal-packet-manifest", type=Path)
    run.add_argument("--m4-unblock-plan-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_consolidation(
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            phase_gate_manifest=args.phase_gate_manifest,
            m3_acceptance_manifest=args.m3_acceptance_manifest,
            surrogate_readiness_manifest=args.surrogate_readiness_manifest,
            surrogate_benefit_manifest=args.surrogate_benefit_manifest,
            adversarial_gate_audit_manifest=args.adversarial_gate_audit_manifest,
            defensive_outputs_manifest=args.defensive_outputs_manifest,
            results_backup_manifest=args.results_backup_manifest,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            cell_projection_manifest=args.cell_projection_manifest,
            raw_failure_manifest=args.raw_failure_manifest,
            proposal_readiness_manifest=args.proposal_readiness_manifest,
            manual_proposal_packet_manifest=args.manual_proposal_packet_manifest,
            m4_unblock_plan_manifest=args.m4_unblock_plan_manifest,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise ConsolidateError("CONSOLIDATE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
