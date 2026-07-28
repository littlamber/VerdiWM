#!/usr/bin/env python3
"""Audit whether ACWM-Phys receipts are sufficient to calibrate IRG charts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


EXPECTED_ENVIRONMENTS = (
    "push_cube",
    "stack_cube",
    "push_rope",
    "cloth_move",
    "push_sand",
    "pour_water",
    "robot_arm",
    "reacher",
)
REQUIRED_HORIZONS = (16, 32, 48, 64)


def audit(*, screen_trials: Path, horizon_metrics: Path, output_root: Path) -> dict[str, object]:
    with screen_trials.open(newline="", encoding="utf-8") as handle:
        screen_rows = list(csv.DictReader(handle))
    with horizon_metrics.open(newline="", encoding="utf-8") as handle:
        horizon_rows = list(csv.DictReader(handle))
    by_environment: dict[str, dict[str, object]] = {}
    horizon_by_campaign: dict[str, set[int]] = defaultdict(set)
    for row in horizon_rows:
        horizon_by_campaign[str(row["campaign_id"])].add(int(row["horizon"]))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in screen_rows:
        grouped[str(row["environment"])].append(row)
    for environment in EXPECTED_ENVIRONMENTS:
        rows = grouped.get(environment, [])
        primitive_campaigns: dict[str, set[str]] = defaultdict(set)
        primitive_seeds: dict[str, set[str]] = defaultdict(set)
        primitive_steps: dict[str, set[int]] = defaultdict(set)
        horizon_coverage: set[int] = set()
        for row in rows:
            campaign = str(row["campaign_id"])
            primitive = str(row["primitive"])
            primitive_campaigns[primitive].add(campaign)
            primitive_seeds[primitive].add(str(row["seed"]))
            primitive_steps[primitive].add(int(row["train_steps"]))
            horizon_coverage.update(horizon_by_campaign.get(campaign, set()))
        repeated = {
            primitive: len(seeds)
            for primitive, seeds in primitive_seeds.items()
            if len(seeds) >= 3
        }
        by_environment[environment] = {
            "screen_trial_count": len(rows),
            "unique_primitive_count": len(primitive_campaigns),
            "unique_primitives": sorted(primitive_campaigns),
            "primitive_seed_counts": {key: len(value) for key, value in sorted(primitive_seeds.items())},
            "primitive_train_steps": {key: sorted(value) for key, value in sorted(primitive_steps.items())},
            "repeated_primitive_seed_counts_ge3": repeated,
            "horizon_coverage": sorted(horizon_coverage),
            "horizon_coverage_missing": [horizon for horizon in REQUIRED_HORIZONS if horizon not in horizon_coverage],
            "has_paired_positive_negative_probe_doses": False,
            "has_three_point_dose_sweep": False,
            "irg_calibration_status": "candidate_response_inventory_only",
        }
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-fingerprint-calibration-audit",
        "claim_boundary": "Existing training trials provide observational response vectors. They do not establish an IRG derivative until the same semantic probe has paired negative, zero, and positive doses under matched repeats.",
        "source_files": {
            "screen_trials": str(screen_trials),
            "horizon_metrics": str(horizon_metrics),
        },
        "source_counts": {
            "screen_trial_count": len(screen_rows),
            "horizon_metric_row_count": len(horizon_rows),
            "environment_count": len({str(row["environment"]) for row in screen_rows}),
        },
        "environments": by_environment,
        "next_calibration_contract": {
            "minimum_per_environment": "one frozen baseline plus one semantic probe family with negative, zero, and positive doses",
            "minimum_repeats_per_dose": 3,
            "required_measurements": ["same target trajectories", "same verdict metrics", "same action protocol", "dose and hook receipt"],
            "do_not_call_existing_trials": "IRG calibrated charts",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    files = {
        "fingerprint-calibration-audit.json": json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        "fingerprint-calibration-audit.md": _markdown(report),
        "tables/environment-fingerprint-coverage.csv": _csv(by_environment),
    }
    hashes: dict[str, str] = {}
    for relative, content in files.items():
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        hashes[relative] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-fingerprint-calibration-audit-manifest",
        "source_screen_trial_count": len(screen_rows),
        "source_horizon_metric_row_count": len(horizon_rows),
        "environment_count": len(by_environment),
        "calibrated_environment_count": 0,
        "files": hashes,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# ACWM-Phys Fingerprint Calibration Audit",
        "",
        str(report["claim_boundary"]),
        "",
        "| Environment | Screen trials | Primitives | Repeated primitive (>=3 seeds) | Horizon coverage | IRG status |",
        "|---|---:|---:|---|---|---|",
    ]
    for environment, row in report["environments"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {environment} | {row['screen_trial_count']} | {row['unique_primitive_count']} | "
            f"{', '.join(row['repeated_primitive_seed_counts_ge3']) or '--'} | "
            f"{', '.join(str(value) for value in row['horizon_coverage']) or '--'} | {row['irg_calibration_status']} |"
        )
    lines.extend(
        [
            "",
            "## Required Next Probe Run",
            "",
            "1. Freeze one target trajectory/evaluator per environment.",
            "2. Apply one semantic, reversible probe at negative, zero, and positive dose.",
            "3. Repeat every dose at least three times with paired seeds and archive hook/dose receipts.",
            "4. Fit the response chart only after all three dose levels and evaluator outputs exist.",
        ]
    )
    return "\n".join(lines) + "\n"


def _csv(by_environment: dict[str, dict[str, object]]) -> str:
    rows: list[dict[str, object]] = []
    for environment, row in by_environment.items():
        rows.append(
            {
                "environment": environment,
                "screen_trial_count": row["screen_trial_count"],
                "unique_primitive_count": row["unique_primitive_count"],
                "repeated_primitive_seed_counts_ge3": ";".join(
                    f"{key}:{value}" for key, value in row["repeated_primitive_seed_counts_ge3"].items()  # type: ignore[union-attr]
                ),
                "horizon_coverage": ";".join(str(value) for value in row["horizon_coverage"]),  # type: ignore[union-attr]
                "horizon_coverage_missing": ";".join(str(value) for value in row["horizon_coverage_missing"]),  # type: ignore[union-attr]
                "irg_calibration_status": row["irg_calibration_status"],
            }
        )
    fields = list(rows[0])
    output: list[str] = []
    buffer = csv.DictWriter(_StringWriter(output), fieldnames=fields, lineterminator="\n")
    buffer.writeheader()
    buffer.writerows(rows)
    return "".join(output)


class _StringWriter:
    def __init__(self, output: list[str]) -> None:
        self.output = output

    def write(self, value: str) -> int:
        self.output.append(value)
        return len(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-trials", type=Path, required=True)
    parser.add_argument("--horizon-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(screen_trials=args.screen_trials, horizon_metrics=args.horizon_metrics, output_root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
