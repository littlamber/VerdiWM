#!/usr/bin/env python3
"""Summarize selected-checkpoint horizon effects across ACWM training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean


class AcwmTrainingSeedHorizonSummaryError(RuntimeError):
    """Training-seed horizon evidence is incomplete or inconsistent."""


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmTrainingSeedHorizonSummaryError(f"TRAINSEED_HORIZON_SUMMARY_JSON_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise AcwmTrainingSeedHorizonSummaryError(f"TRAINSEED_HORIZON_SUMMARY_JSON_INVALID:{path}")
    return payload


def _finite(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcwmTrainingSeedHorizonSummaryError(code)
    result = float(value)
    if not math.isfinite(result):
        raise AcwmTrainingSeedHorizonSummaryError(code)
    return result


def build_training_seed_horizon_summary(
    *, profile_paths: Sequence[Path], stability_manifest: Path, output_root: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_OUTPUT_EXISTS")
    manifest_path = Path(stability_manifest).resolve(strict=True)
    manifest = _json(manifest_path)
    if (
        manifest.get("artifact_type")
        != "verdiwm-acwm-training-seed-checkpoint-stability-summary-manifest"
        or manifest.get("state") != "ready"
    ):
        raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_STABILITY_INVALID")
    stability = _json(Path(str(manifest.get("summary_path") or "")).resolve(strict=True))
    selected = stability.get("selected_checkpoints")
    if not isinstance(selected, list) or not selected:
        raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_STABILITY_INVALID")
    expected = {
        int(row["training_seed"]): row for row in selected if isinstance(row, Mapping)
    }
    profiles = [_json(Path(path).resolve(strict=True)) for path in profile_paths]
    observed = {
        int(profile["training_seed"]): profile
        for profile in profiles
        if profile.get("artifact_type") == "wmloop-acwm-horizon-effect-profile"
        and profile.get("state") == "ready"
        and isinstance(profile.get("training_seed"), int)
    }
    if set(observed) != set(expected) or len(observed) != len(profiles):
        raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_SEED_COVERAGE_MISMATCH")
    environment = str(stability.get("environment") or "")
    primitive = str(stability.get("primitive") or "")
    horizon_sets = {tuple(int(value) for value in profile.get("horizons", [])) for profile in profiles}
    if (
        len(horizon_sets) != 1
        or any(profile.get("environment") != environment or profile.get("primitive") != primitive for profile in profiles)
    ):
        raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_IDENTITY_MISMATCH")
    horizons = list(next(iter(horizon_sets)))
    rows: list[dict[str, object]] = []
    for training_seed in sorted(observed):
        profile = observed[training_seed]
        classification = profile.get("effect_classification")
        effects = profile.get("horizon_effects")
        if not isinstance(classification, Mapping) or not isinstance(effects, list):
            raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_PROFILE_INVALID")
        selected_row = expected[training_seed]
        by_horizon: dict[str, dict[str, object]] = {}
        for raw in effects:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("delta_candidate_minus_baseline"), Mapping):
                raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_PROFILE_INVALID")
            horizon = int(raw["horizon"])
            delta = raw["delta_candidate_minus_baseline"]
            by_horizon[str(horizon)] = {
                "delta_psnr": _finite(delta.get("psnr"), "TRAINSEED_HORIZON_SUMMARY_METRIC_INVALID"),
                "delta_ssim": _finite(delta.get("ssim"), "TRAINSEED_HORIZON_SUMMARY_METRIC_INVALID"),
                "delta_mse": _finite(delta.get("mse"), "TRAINSEED_HORIZON_SUMMARY_METRIC_INVALID"),
                "delta_masked_mse": _finite(delta.get("masked_mse"), "TRAINSEED_HORIZON_SUMMARY_METRIC_INVALID"),
                "strict_quality_pass": raw.get("strict_quality_pass") is True,
            }
        if set(by_horizon) != {str(value) for value in horizons}:
            raise AcwmTrainingSeedHorizonSummaryError("TRAINSEED_HORIZON_SUMMARY_HORIZON_COVERAGE_MISMATCH")
        rows.append(
            {
                "training_seed": training_seed,
                "checkpoint_step": int(selected_row["checkpoint_step"]),
                "checkpoint_sha256": str(selected_row["checkpoint_sha256"]),
                "effect_scope": str(classification.get("effect_scope") or ""),
                "aggregate_max_horizon_pass": classification.get("aggregate_max_horizon_pass") is True,
                "positive_trajectory_rate_at_max_horizon": _finite(
                    classification.get("positive_trajectory_rate_at_max_horizon", 0.0),
                    "TRAINSEED_HORIZON_SUMMARY_RATE_INVALID",
                ),
                "horizon_effects": by_horizon,
            }
        )
    max_pass_count = sum(1 for row in rows if row["aggregate_max_horizon_pass"])
    horizon_summary = []
    for horizon in horizons:
        values = [float(row["horizon_effects"][str(horizon)]["delta_psnr"]) for row in rows]
        strict_count = sum(
            1 for row in rows if row["horizon_effects"][str(horizon)]["strict_quality_pass"]
        )
        horizon_summary.append(
            {
                "horizon": horizon,
                "training_seed_count": len(rows),
                "strict_pass_count": strict_count,
                "strict_pass_rate": strict_count / len(rows),
                "mean_delta_psnr": mean(values),
                "minimum_delta_psnr": min(values),
                "maximum_delta_psnr": max(values),
            }
        )
    verdict = (
        "stable_long_horizon_positive"
        if max_pass_count == len(rows)
        else "training_seed_sensitive_long_horizon_effect"
        if max_pass_count
        else "no_aggregate_long_horizon_positive"
    )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-horizon-stability-summary",
        "state": "ready",
        "environment": environment,
        "primitive": primitive,
        "training_seeds": sorted(observed),
        "horizons": horizons,
        "selected_checkpoint_policy": stability.get("global_selection_reason"),
        "training_seed_records": rows,
        "horizon_summary": horizon_summary,
        "max_horizon_pass_count": max_pass_count,
        "max_horizon_cell_count": len(rows),
        "stability_verdict": verdict,
        "claim_boundary": (
            "Paired autoregressive long-horizon evidence across independent repair-training seeds. "
            "It updates scoped routing priors but does not establish cross-backbone transfer or causal credit."
        ),
    }
    return _write(destination, report)


def _write(destination: Path, report: dict[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700, parents=True)
    try:
        (temporary / "summary.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (temporary / "horizon-summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["horizon", "training_seed_count", "strict_pass_count", "strict_pass_rate",
                      "mean_delta_psnr", "minimum_delta_psnr", "maximum_delta_psnr"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report["horizon_summary"]:
                writer.writerow({field: row[field] for field in fields})
        lines = [
            "# ACWM training-seed horizon stability", "",
            f"Target: `{report['environment']} + {report['primitive']}`",
            f"Verdict: `{report['stability_verdict']}`", "",
            "| Horizon | Strict pass | Mean PSNR delta | Minimum PSNR delta |",
            "|---:|---:|---:|---:|",
        ]
        for row in report["horizon_summary"]:
            lines.append(
                f"| {row['horizon']} | {row['strict_pass_count']}/{row['training_seed_count']} | "
                f"{float(row['mean_delta_psnr']):+.4f} | {float(row['minimum_delta_psnr']):+.4f} |"
            )
        lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
        (temporary / "README.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-training-seed-horizon-stability-summary-manifest",
            "state": "ready",
            "environment": report["environment"],
            "primitive": report["primitive"],
            "stability_verdict": report["stability_verdict"],
            "summary_path": str(destination / "summary.json"),
            "csv_path": str(destination / "horizon-summary.csv"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--stability-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_training_seed_horizon_summary(
        profile_paths=args.profile, stability_manifest=args.stability_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
