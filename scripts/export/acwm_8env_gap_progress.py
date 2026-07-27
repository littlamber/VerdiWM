#!/usr/bin/env python3
"""Export the current 8-environment ACWM gap-routing progress matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES


DEFAULT_OUT = ROOT / "results/reports/acwm-8env-gap-progress-r1"
DEFAULT_SIGNATURE_BANK = ROOT / "results/reports/failure-signature-bank-r1/failure-signature-bank.json"
DEFAULT_MATERIALIZATION_GATE = ROOT / "results/reports/primitive-materialization-gate-r8/primitive-materialization-gate.json"
DEFAULT_DIRECT_CAMPAIGNS = ROOT / "results/reports/acwm-gap-execution-queue-1w-r3/tables/direct-campaigns.csv"
DEFAULT_MATERIALIZATION_ORDERS = ROOT / "results/reports/acwm-gap-execution-queue-1w-r3/tables/materialization-orders.csv"
DEFAULT_ARCHIVE = ROOT / "results/archive.db"


class Acwm8EnvGapProgressError(RuntimeError):
    """Progress export failed closed."""


def run_export(
    *,
    output_root: Path = DEFAULT_OUT,
    signature_bank: Path = DEFAULT_SIGNATURE_BANK,
    materialization_gate: Path = DEFAULT_MATERIALIZATION_GATE,
    direct_campaigns: Path = DEFAULT_DIRECT_CAMPAIGNS,
    materialization_orders: Path = DEFAULT_MATERIALIZATION_ORDERS,
    archive_db: Path = DEFAULT_ARCHIVE,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Acwm8EnvGapProgressError("ACWM_8ENV_GAP_PROGRESS_OUTPUT_EXISTS")
    envs = [spec.environment for spec in CANONICAL_ACWM_ENVIRONMENTS]
    routes = _routes_by_env(_read_json(signature_bank))
    materialization = _read_json(materialization_gate)
    closed_loop_ready = set(_string_list(materialization.get("closed_loop_ready_primitives")))
    sidecar_only = set(_string_list(materialization.get("sidecar_only_primitives")))
    direct_by_env_all = _csv_rows_by_env(direct_campaigns)
    direct_by_env = {
        env: [row for row in rows if str(row.get("primitive") or "") not in INVALIDATED_QUALITY_PRIMITIVES]
        for env, rows in direct_by_env_all.items()
    }
    invalidated_direct_by_env = {
        env: [row for row in rows if str(row.get("primitive") or "") in INVALIDATED_QUALITY_PRIMITIVES]
        for env, rows in direct_by_env_all.items()
    }
    orders_by_env = _csv_rows_by_env(materialization_orders)
    cell_rows = _archive_cells(archive_db)
    rows = []
    for env in envs:
        cells = sorted(cell_rows.get(env, []), key=lambda item: float(item["mean_verified_gain"]), reverse=True)
        best = cells[0] if cells else {}
        direct = direct_by_env.get(env, [])
        orders = orders_by_env.get(env, [])
        route_rows = routes.get(env, [])
        positive_cells = [cell for cell in cells if float(cell["mean_verified_gain"]) > 0.0]
        invalidated_positive_cells = [
            cell
            for cell in positive_cells
            if str(cell.get("primitive_family") or "") in INVALIDATED_QUALITY_PRIMITIVES
        ]
        claimable_positive_cells = [cell for cell in positive_cells if cell not in invalidated_positive_cells]
        rows.append(
            {
                "environment": env,
                "routing_count": len(route_rows),
                "routed_primitives": ";".join(f"{r['primitive']}[{','.join(r['target_failures'])}]" for r in route_rows),
                "closed_loop_ready_routed_primitives": ";".join(
                    sorted({str(r["primitive"]) for r in route_rows if str(r["primitive"]) in closed_loop_ready})
                ),
                "sidecar_only_routed_primitives": ";".join(
                    sorted({str(r["primitive"]) for r in route_rows if str(r["primitive"]) in sidecar_only})
                ),
                "best_primitive": best.get("primitive_family", ""),
                "best_mean_verified_gain": best.get("mean_verified_gain", ""),
                "best_settled_trial_count": best.get("settled_trial_count", ""),
                "positive_cell_count": len(positive_cells),
                "claimable_positive_cell_count": len(claimable_positive_cells),
                "invalidated_positive_cell_count": len(invalidated_positive_cells),
                "direct_campaigns": ";".join(
                    f"{r.get('queue_type')}:{r.get('primitive')}:s{r.get('seed')}:gpu{r.get('assigned_gpu')}:steps{r.get('train_steps')}"
                    for r in direct
                ),
                "direct_campaign_count": len(direct),
                "invalidated_direct_campaign_count": len(invalidated_direct_by_env.get(env, [])),
                "invalidated_direct_campaigns": ";".join(
                    f"{r.get('queue_type')}:{r.get('primitive')}:s{r.get('seed')}:gpu{r.get('assigned_gpu')}:steps{r.get('train_steps')}"
                    for r in invalidated_direct_by_env.get(env, [])
                ),
                "materialization_orders": ";".join(
                    f"{r.get('priority')}:{r.get('order_type')}:{r.get('target_name')}" for r in orders
                ),
                "materialization_order_count": len(orders),
                "progress_state": _progress_state(
                    env=env,
                    positive_cells=claimable_positive_cells,
                    invalidated_positive_cells=invalidated_positive_cells,
                    direct=direct,
                    orders=orders,
                ),
                "next_action": _next_action(
                    env=env,
                    direct=direct,
                    orders=orders,
                    positive_cells=claimable_positive_cells,
                    invalidated_positive_cells=invalidated_positive_cells,
                ),
            }
        )
    summary = _summary(rows=rows, materialization=materialization)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-8env-gap-progress",
        "state": "ready",
        "created_at": _utc_now(),
        "summary": summary,
        "rows": rows,
        "sources": {
            "signature_bank": str(Path(signature_bank).resolve()),
            "materialization_gate": str(Path(materialization_gate).resolve()),
            "direct_campaigns": str(Path(direct_campaigns).resolve()),
            "materialization_orders": str(Path(materialization_orders).resolve()),
            "archive_db": str(Path(archive_db).resolve()),
        },
    }
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "acwm-8env-gap-progress.json", report)
    _write_csv(destination / "acwm-8env-gap-progress.csv", rows)
    (destination / "acwm-8env-gap-progress.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-8env-gap-progress-manifest",
            "state": "ready",
            "summary": summary,
            "report_path": str(destination / "acwm-8env-gap-progress.json"),
            "csv_path": str(destination / "acwm-8env-gap-progress.csv"),
            "markdown_path": str(destination / "acwm-8env-gap-progress.md"),
        },
    )
    return report


def _routes_by_env(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = payload.get("primitive_routing")
    if not isinstance(rows, list):
        raise Acwm8EnvGapProgressError("ACWM_8ENV_GAP_PROGRESS_SIGNATURE_BANK_INVALID")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        env = row.get("environment")
        primitive = row.get("primitive")
        failures = row.get("target_failures")
        if isinstance(env, str) and isinstance(primitive, str) and isinstance(failures, list):
            out[env].append(
                {
                    "primitive": primitive,
                    "routing_decision": row.get("routing_decision", ""),
                    "target_failures": [str(item) for item in failures],
                }
            )
    return out


def _csv_rows_by_env(path: Path) -> dict[str, list[dict[str, str]]]:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise Acwm8EnvGapProgressError(f"ACWM_8ENV_GAP_PROGRESS_CSV_MISSING:{path}")
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            env = row.get("environment")
            if env:
                out[env].append(dict(row))
    return out


def _archive_cells(path: Path) -> dict[str, list[dict[str, object]]]:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise Acwm8EnvGapProgressError(f"ACWM_8ENV_GAP_PROGRESS_ARCHIVE_MISSING:{path}")
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(
            "select environment, primitive_family, settled_trial_count, mean_verified_gain "
            "from cells order by environment, mean_verified_gain desc"
        ):
            out[str(row["environment"])].append(dict(row))
    finally:
        con.close()
    return out


def _progress_state(
    *,
    env: str,
    positive_cells: list[dict[str, object]],
    invalidated_positive_cells: list[dict[str, object]],
    direct: list[dict[str, str]],
    orders: list[dict[str, str]],
) -> str:
    if direct:
        return "queued_for_canary_or_confirmation"
    if positive_cells:
        return "positive_signal_needs_replication"
    if invalidated_positive_cells:
        return "invalidated_positive_signal_audit_only"
    if orders:
        return "diagnostic_or_primitive_materialization_needed"
    return "no_actionable_route_recorded"


def _next_action(
    *,
    env: str,
    direct: list[dict[str, str]],
    orders: list[dict[str, str]],
    positive_cells: list[dict[str, object]],
    invalidated_positive_cells: list[dict[str, object]],
) -> str:
    if direct:
        first = sorted(direct, key=lambda row: int(row.get("ordinal") or 999))[0]
        return f"Run queued {first.get('queue_type')} for {first.get('primitive')} after GPUs free."
    p0 = [row for row in orders if row.get("priority") == "P0"]
    if p0:
        return "Materialize diagnostic probes: " + ", ".join(row.get("target_name", "") for row in p0)
    if positive_cells:
        return "Replicate the positive signal under fixed protocol before claiming an improvement."
    if invalidated_positive_cells:
        return "Retain the invalidated metric signal for audit, but diagnose and stage a valid replacement primitive."
    return "Materialize routed primitives and rerun gap diagnosis."


def _summary(*, rows: list[dict[str, object]], materialization: dict[str, Any]) -> dict[str, object]:
    states = defaultdict(int)
    for row in rows:
        states[str(row["progress_state"])] += 1
    return {
        "environment_count": len(rows),
        "closed_loop_ready_primitive_count": materialization.get("closed_loop_ready_count"),
        "sidecar_only_primitive_count": materialization.get("sidecar_only_count"),
        "direct_campaign_count": sum(int(row["direct_campaign_count"]) for row in rows),
        "materialization_order_count": sum(int(row["materialization_order_count"]) for row in rows),
        "positive_environment_count": sum(1 for row in rows if int(row["positive_cell_count"]) > 0),
        "progress_state_counts": dict(sorted(states.items())),
    }


def _render_markdown(report: dict[str, object]) -> str:
    rows = report["rows"]
    if not isinstance(rows, list):
        raise Acwm8EnvGapProgressError("ACWM_8ENV_GAP_PROGRESS_REPORT_INVALID")
    lines = [
        "# ACWM 8-Environment Gap Progress",
        "",
        f"State: `{report['state']}`",
        "",
        "## Summary",
        "",
    ]
    summary = report["summary"]
    if isinstance(summary, dict):
        for key, value in summary.items():
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Environment | State | Best cell | Ready routed primitives | Direct queue | Next action |",
            "|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for row in rows:
        best = row["best_primitive"]
        gain = row["best_mean_verified_gain"]
        trials = row["best_settled_trial_count"]
        best_cell = f"{best} / gain={gain} / n={trials}" if best else "none"
        lines.append(
            "| {environment} | {progress_state} | {best_cell} | {ready} | {direct} | {next_action} |".format(
                environment=row["environment"],
                progress_state=row["progress_state"],
                best_cell=best_cell,
                ready=row["closed_loop_ready_routed_primitives"] or "none",
                direct=row["direct_campaigns"] or "none",
                next_action=row["next_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise Acwm8EnvGapProgressError(f"ACWM_8ENV_GAP_PROGRESS_JSON_MISSING:{path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise Acwm8EnvGapProgressError("ACWM_8ENV_GAP_PROGRESS_JSON_INVALID")
    return payload


def _write_json(path: Path, payload: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--signature-bank", type=Path, default=DEFAULT_SIGNATURE_BANK)
    parser.add_argument("--materialization-gate", type=Path, default=DEFAULT_MATERIALIZATION_GATE)
    parser.add_argument("--direct-campaigns", type=Path, default=DEFAULT_DIRECT_CAMPAIGNS)
    parser.add_argument("--materialization-orders", type=Path, default=DEFAULT_MATERIALIZATION_ORDERS)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    report = run_export(
        output_root=args.output_root,
        signature_bank=args.signature_bank,
        materialization_gate=args.materialization_gate,
        direct_campaigns=args.direct_campaigns,
        materialization_orders=args.materialization_orders,
        archive_db=args.archive_db,
    )
    print(json.dumps({"state": report["state"], "output_root": str(Path(args.output_root).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
