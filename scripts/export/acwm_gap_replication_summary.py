#!/usr/bin/env python3
"""Summarize cross-campaign replication evidence without mutating verdicts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def safe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def manifest_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    metric = payload.get("primary_metric")
    delta = None
    if isinstance(metric, str):
        delta = safe_float((payload.get("delta_m_ver") or {}).get(metric))
    action_gate = payload.get("action_following_gate") or {}
    return {
        "environment": payload.get("environment") or path.parent.name,
        "proposal_id": payload.get("proposal_id"),
        "seed": payload.get("seed"),
        "state": payload.get("state"),
        "verdict": payload.get("verdict"),
        "primary_metric": metric,
        "delta": delta,
        "action_following_pass": bool(action_gate.get("pass")) if isinstance(action_gate, dict) else False,
        "action_following_observed": action_gate.get("observed") if isinstance(action_gate, dict) else None,
        "manifest_path": str(path),
        "report_path": payload.get("report_path"),
    }


def summarize_case(*, environment: str, primitive: str, manifests: list[Path], min_replications: int) -> dict[str, Any]:
    rows = [manifest_row(path) for path in manifests]
    rows = [row for row in rows if row["environment"] == environment and primitive in str(row.get("proposal_id") or "")]
    deltas = [row["delta"] for row in rows if isinstance(row["delta"], float)]
    ready_count = sum(1 for row in rows if row["state"] == "ready")
    positive_count = sum(1 for value in deltas if value > 0.0)
    action_pass_count = sum(1 for row in rows if row["action_following_pass"])
    mean_delta = statistics.fmean(deltas) if deltas else None
    pstdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None
    replication_threshold = mean_delta * 0.5 if isinstance(mean_delta, float) and mean_delta > 0.0 else None
    stable_positive = (
        len(deltas) >= min_replications
        and ready_count == len(rows)
        and positive_count == len(deltas)
        and action_pass_count == len(rows)
        and isinstance(mean_delta, float)
        and isinstance(pstdev_delta, float)
        and mean_delta > 0.0
        and pstdev_delta <= mean_delta * 0.5
    )
    return {
        "environment": environment,
        "primitive": primitive,
        "state": "ready" if rows else "empty",
        "trial_count": len(rows),
        "ready_trial_count": ready_count,
        "delta_count": len(deltas),
        "positive_trial_count": positive_count,
        "action_following_pass_count": action_pass_count,
        "min_replications": min_replications,
        "mean_delta": mean_delta,
        "pstdev_delta": pstdev_delta,
        "replication_threshold_pstdev_max": replication_threshold,
        "cross_campaign_stable_positive": stable_positive,
        "formal_verdict_mutated": False,
        "claim_boundary": (
            "Read-only cross-campaign stability summary. It does not rewrite individual frozen verdicts; "
            "single-campaign manifests can remain INCONCLUSIVE because their internal G4 replication gate is pending."
        ),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "proposal_id",
        "seed",
        "state",
        "verdict",
        "primary_metric",
        "delta",
        "action_following_pass",
        "action_following_observed",
        "manifest_path",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# ACWM Gap Replication Summary: {summary['environment']} + {summary['primitive']}",
        "",
        f"State: `{summary['state']}`",
        f"Cross-campaign stable positive: `{summary['cross_campaign_stable_positive']}`",
        f"Trial count: `{summary['trial_count']}`",
        f"Ready trial count: `{summary['ready_trial_count']}`",
        f"Positive trial count: `{summary['positive_trial_count']}`",
        f"Mean delta: `{summary['mean_delta']}`",
        f"Population std delta: `{summary['pstdev_delta']}`",
        f"Replication threshold pstdev <= mean * 0.5: `{summary['replication_threshold_pstdev_max']}`",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
        "| Seed | State | Verdict | Delta | Action gate | Manifest |",
        "|:--|:--|:--|--:|:--|:--|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {seed} | `{state}` | `{verdict}` | `{delta}` | `{action}` | `{manifest}` |".format(
                seed=row.get("seed"),
                state=row.get("state"),
                verdict=row.get("verdict"),
                delta=row.get("delta"),
                action=row.get("action_following_pass"),
                manifest=row.get("manifest_path"),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--manifest-glob", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-replications", type=int, default=3)
    args = parser.parse_args()

    manifests: list[Path] = []
    for pattern in args.manifest_glob:
        manifests.extend(sorted(Path().glob(pattern)))
    summary = summarize_case(
        environment=args.environment,
        primitive=args.primitive,
        manifests=manifests,
        min_replications=args.min_replications,
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit("ACWM_GAP_REPLICATION_SUMMARY_OUTPUT_EXISTS")
    write_json(output_root / "replication-summary.json", summary)
    write_csv(output_root / "tables" / "trial_rows.csv", summary["rows"])
    write_markdown(output_root / "replication-summary.md", summary)
    print(json.dumps({"state": summary["state"], "output_root": str(output_root)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
