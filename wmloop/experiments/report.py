"""Aggregate settled LOBO receipts into claim-safe paper tables."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.ledger import load_settled_receipts
from wmloop.experiments.lobo import build_lobo_plan
from wmloop.experiments.spec import load_experiment_spec


def build_experiment_report(
    *,
    spec: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    plan = build_lobo_plan(spec)
    trials = list(plan["trials"])
    by_trial_stage = {(str(item["trial_id"]), str(item["stage"])): item for item in receipts}
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[(str(trial["fold_id"]), str(trial["arm"]), str(trial["selector"]))].append(trial)

    summary_rows = [
        _summarize_group(key=key, trials=group_trials, by_trial_stage=by_trial_stage)
        for key, group_trials in sorted(groups.items())
    ]
    cost_rows = _cost_rows(receipts)
    selector_rows = [row for row in summary_rows if row["arm"] == "warm_start"]
    expected_confirm = len(trials)
    observed_confirm = sum(1 for item in receipts if item["stage"] == "confirm")
    formal_positives = sum(1 for item in receipts if item["stage"] == "confirm" and item["outcome"] == "positive")
    missing_confirm = expected_confirm - observed_confirm
    blockers = list(plan["blockers"])
    if missing_confirm:
        blockers.append(
            {
                "code": "confirm_receipts_missing",
                "expected": expected_confirm,
                "observed": observed_confirm,
            }
        )
    if plan["blockers"]:
        state = "blocked"
    elif not missing_confirm:
        state = "ready"
    else:
        state = "partial" if receipts else "empty"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cross-backbone-experiment-report",
        "experiment_id": spec["experiment_id"],
        "state": state,
        "claim_ready": not blockers,
        "metric_contract": dict(spec["metric_contract"]),
        "settled_receipt_count": len(receipts),
        "expected_confirm_count": expected_confirm,
        "observed_confirm_count": observed_confirm,
        "formal_positive_count": formal_positives,
        "summary_rows": summary_rows,
        "selector_rows": selector_rows,
        "cost_rows": cost_rows,
        "blockers": blockers,
        "claim_boundary": {
            "formal_positive_rule": "confirm receipt is settled, outcome is positive, and normalized delta exceeds the frozen threshold",
            "screen_or_gate_positive_is_formal": False,
            "unsettled_receipts_included_in_cost": False,
            "missing_confirm_counts_as_positive": False,
        },
    }


def run_experiment_report(
    *,
    spec_path: Path,
    receipt_paths: Sequence[Path],
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    spec = load_experiment_spec(spec_path)
    receipts = load_settled_receipts(spec=spec, receipt_paths=receipt_paths)
    report = build_experiment_report(spec=spec, receipts=receipts)
    public_receipts = [{key: value for key, value in item.items() if not key.startswith("_")} for item in receipts]
    files = {
        "experiment-report.json": canonical_json(report),
        "experiment-report.md": _render_markdown(report).encode("utf-8"),
        "tables/lobo-summary.csv": _csv_bytes(report["summary_rows"]),
        "tables/lobo-summary.tex": _latex_table(report["summary_rows"]).encode("utf-8"),
        "tables/selector-ablation.csv": _csv_bytes(report["selector_rows"]),
        "tables/cost-summary.csv": _csv_bytes(report["cost_rows"]),
        "input-receipts.json": canonical_json(public_receipts),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-cross-backbone-experiment-report-manifest",
            "state": report["state"],
            "experiment_id": report["experiment_id"],
            "claim_ready": report["claim_ready"],
            "settled_receipt_count": report["settled_receipt_count"],
            "formal_positive_count": report["formal_positive_count"],
            "blocker_count": len(report["blockers"]),
            "report_path": str(destination / "experiment-report.json"),
            "paper_table_path": str(destination / "tables" / "lobo-summary.tex"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _summarize_group(
    *,
    key: tuple[str, str, str],
    trials: Sequence[Mapping[str, Any]],
    by_trial_stage: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, object]:
    fold_id, arm, selector = key
    confirms = [by_trial_stage[(str(trial["trial_id"]), "confirm")] for trial in trials if (str(trial["trial_id"]), "confirm") in by_trial_stage]
    confirms.sort(key=lambda item: int(item["sequence_index"]))
    positives = sum(item["outcome"] == "positive" for item in confirms)
    negatives = sum(item["outcome"] == "negative" for item in confirms)
    abstains = sum(item["outcome"] == "abstain" for item in confirms)
    licensed = positives + negatives
    first_positive = next((index for index, item in enumerate(confirms, start=1) if item["outcome"] == "positive"), None)
    total_gpu_hours = sum(
        float(receipt["gpu_hours"])
        for trial in trials
        for stage in ("screen", "gate", "confirm")
        if (receipt := by_trial_stage.get((str(trial["trial_id"]), stage))) is not None
    )
    expected = len(trials)
    return {
        "fold_id": fold_id,
        "arm": arm,
        "selector": selector,
        "expected_trials": expected,
        "confirmed_trials": len(confirms),
        "formal_positive_count": positives,
        "formal_negative_count": negatives,
        "abstain_count": abstains,
        "trials_to_first_positive": first_positive,
        "transfer_hit_rate": _ratio(positives, licensed),
        "negative_transfer_rate": _ratio(negatives, licensed) if arm == "warm_start" else None,
        "abstention_rate": _ratio(abstains, licensed + abstains),
        "coverage": _ratio(licensed, expected),
        "risk": _ratio(negatives, licensed),
        "total_gpu_hours": round(total_gpu_hours, 9),
    }


def _cost_rows(receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"gpu_hours": 0.0, "receipt_count": 0.0})
    for receipt in receipts:
        key = (str(receipt["arm"]), str(receipt["stage"]))
        totals[key]["gpu_hours"] += float(receipt["gpu_hours"])
        totals[key]["receipt_count"] += 1.0
    return [
        {
            "arm": arm,
            "stage": stage,
            "settled_receipt_count": int(values["receipt_count"]),
            "gpu_hours": round(values["gpu_hours"], 9),
        }
        for (arm, stage), values in sorted(totals.items())
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 9)


def _csv_bytes(rows: object) -> bytes:
    records = list(rows)  # type: ignore[arg-type]
    if not records:
        return b"\n"
    output = io.StringIO(newline="")
    fieldnames = list(records[0])
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def _latex_table(rows: object) -> str:
    records = list(rows)  # type: ignore[arg-type]
    columns = ["fold_id", "arm", "selector", "confirmed_trials", "formal_positive_count", "trials_to_first_positive", "coverage", "risk", "total_gpu_hours"]
    header = ["Fold", "Arm", "Selector", "Confirmed", "Positive", "Trials-to-positive", "Coverage", "Risk", "GPU hours"]
    lines = [
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in records:
        values = [_latex_value(row.get(column)) for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_value(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("_", "\\_")


def _render_markdown(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Cross-Backbone Experiment Report",
            "",
            f"Experiment: `{report['experiment_id']}`",
            f"State: `{report['state']}`",
            f"Claim ready: `{str(report['claim_ready']).lower()}`",
            f"Settled receipts: `{report['settled_receipt_count']}`",
            f"Formal positives: `{report['formal_positive_count']}`",
            "",
            "Only settled confirm receipts contribute formal positive results. Costs include settled receipts only.",
            "",
            "## Blockers",
            "",
            *(
                [f"- `{item['code']}`: `{json.dumps(item, sort_keys=True)}`" for item in report["blockers"]]  # type: ignore[index]
                if report["blockers"]
                else ["- None."]
            ),
        ]
    ) + "\n"


def _receipt_paths(values: Iterable[Path], directories: Iterable[Path]) -> list[Path]:
    paths = [Path(value) for value in values]
    for directory in directories:
        paths.extend(sorted(Path(directory).rglob("*.json")))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--receipt-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_experiment_report(
        spec_path=args.spec,
        receipt_paths=_receipt_paths(args.receipt, args.receipt_dir),
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
