#!/usr/bin/env python3
"""Aggregate a complete ACWM training-seed by evaluation-seed gate matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import uuid
from pathlib import Path
from typing import Mapping, Sequence


METRICS = ("psnr", "ssim", "mse", "masked_mse")


class AcwmTrainingSeedSummaryError(RuntimeError):
    """Training-seed replication evidence is incomplete or inconsistent."""


def build_summary(*, manifests: Sequence[Path], output_root: Path) -> dict[str, object]:
    records = [_record(Path(path).resolve(strict=True)) for path in manifests]
    if not records:
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_RECEIPTS_EMPTY")
    identities = {(row["environment"], row["primitive"]) for row in records}
    if len(identities) != 1:
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_IDENTITY_MISMATCH")
    training_seeds = sorted({int(row["training_seed"]) for row in records})
    eval_seeds = sorted({int(row["eval_seed"]) for row in records})
    expected = {(training_seed, eval_seed) for training_seed in training_seeds for eval_seed in eval_seeds}
    observed = {(int(row["training_seed"]), int(row["eval_seed"])) for row in records}
    if len(records) != len(observed) or observed != expected:
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_FACTORIAL_INCOMPLETE")
    if len(training_seeds) < 2 or len(eval_seeds) < 2:
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_FACTORIAL_TOO_SMALL")
    checkpoint_by_seed: dict[int, set[str]] = {}
    for row in records:
        checkpoint_by_seed.setdefault(int(row["training_seed"]), set()).add(str(row["checkpoint_sha256"]))
    if any(len(shas) != 1 for shas in checkpoint_by_seed.values()):
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_CHECKPOINT_UNSTABLE")
    checkpoint_shas = {next(iter(shas)) for shas in checkpoint_by_seed.values()}
    if len(checkpoint_shas) != len(training_seeds):
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_CHECKPOINTS_NOT_DISTINCT")

    records.sort(key=lambda row: (int(row["training_seed"]), int(row["eval_seed"])))
    metric_summary = {metric: _metric_statistics(records, metric, training_seeds, eval_seeds) for metric in METRICS}
    environment, primitive = next(iter(identities))
    pass_count = sum(bool(row["pass"]) for row in records)
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-replication-summary",
        "state": "ready",
        "environment": environment,
        "primitive": primitive,
        "training_seeds": training_seeds,
        "eval_seeds": eval_seeds,
        "factorial_shape": [len(training_seeds), len(eval_seeds)],
        "receipt_count": len(records),
        "distinct_candidate_checkpoint_count": len(checkpoint_shas),
        "official_gate_pass_count": pass_count,
        "official_gate_pass_rate": pass_count / len(records),
        "all_cells_pass": pass_count == len(records),
        "records": records,
        "metric_summary": metric_summary,
        "claim_boundary": (
            "This separates independent repair-fine-tuning seeds from shared evaluation seeds. "
            "It does not claim independent base-model pretraining or cross-backbone transfer."
        ),
    }
    return _write_bundle(Path(output_root).resolve(), payload)


def _record(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-acwm-formal-visualization-export":
        raise AcwmTrainingSeedSummaryError(f"ACWM_TRAINING_SEED_RECEIPT_INVALID:{path}")
    if payload.get("state") != "ready":
        raise AcwmTrainingSeedSummaryError(f"ACWM_TRAINING_SEED_RECEIPT_NOT_READY:{path}")
    training_seed = payload.get("training_seed")
    eval_seed = payload.get("eval_seed")
    checkpoint_sha = payload.get("candidate_checkpoint_sha256")
    gate = payload.get("official_quality_gate")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 1
        or isinstance(eval_seed, bool)
        or not isinstance(eval_seed, int)
        or eval_seed < 1
        or not isinstance(checkpoint_sha, str)
        or len(checkpoint_sha) != 64
        or not isinstance(gate, Mapping)
    ):
        raise AcwmTrainingSeedSummaryError(f"ACWM_TRAINING_SEED_RECEIPT_FIELDS_INVALID:{path}")
    delta = gate.get("delta_candidate_minus_baseline")
    if not isinstance(delta, Mapping):
        raise AcwmTrainingSeedSummaryError(f"ACWM_TRAINING_SEED_DELTA_INVALID:{path}")
    deltas: dict[str, float] = {}
    for metric in METRICS:
        value = delta.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AcwmTrainingSeedSummaryError(f"ACWM_TRAINING_SEED_DELTA_INVALID:{path}:{metric}")
        deltas[metric] = float(value)
    return {
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "environment": str(payload.get("environment") or ""),
        "primitive": str(payload.get("primitive") or ""),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": str(payload.get("candidate_checkpoint") or ""),
        "pass": gate.get("pass") is True,
        "delta": deltas,
        "manifest_path": str(path),
    }


def _metric_statistics(
    records: Sequence[Mapping[str, object]],
    metric: str,
    training_seeds: Sequence[int],
    eval_seeds: Sequence[int],
) -> dict[str, object]:
    values = {
        (int(row["training_seed"]), int(row["eval_seed"])): float(row["delta"][metric])  # type: ignore[index]
        for row in records
    }
    flat = list(values.values())
    grand = statistics.fmean(flat)
    training_means = {
        str(seed): statistics.fmean(values[(seed, eval_seed)] for eval_seed in eval_seeds)
        for seed in training_seeds
    }
    eval_means = {
        str(seed): statistics.fmean(values[(training_seed, seed)] for training_seed in training_seeds)
        for seed in eval_seeds
    }
    ss_training = len(eval_seeds) * sum((value - grand) ** 2 for value in training_means.values())
    ss_eval = len(training_seeds) * sum((value - grand) ** 2 for value in eval_means.values())
    ss_residual = sum(
        (
            values[(training_seed, eval_seed)]
            - training_means[str(training_seed)]
            - eval_means[str(eval_seed)]
            + grand
        )
        ** 2
        for training_seed in training_seeds
        for eval_seed in eval_seeds
    )
    ss_total = sum((value - grand) ** 2 for value in flat)
    return {
        "mean": grand,
        "sample_std": statistics.stdev(flat),
        "min": min(flat),
        "max": max(flat),
        "training_seed_means": training_means,
        "eval_seed_means": eval_means,
        "sum_squares": {
            "training_seed": ss_training,
            "eval_seed": ss_eval,
            "residual_interaction": ss_residual,
            "total": ss_total,
        },
        "variance_share": {
            "training_seed": ss_training / ss_total if ss_total > 0.0 else 0.0,
            "eval_seed": ss_eval / ss_total if ss_total > 0.0 else 0.0,
            "residual_interaction": ss_residual / ss_total if ss_total > 0.0 else 0.0,
        },
    }


def _write_bundle(destination: Path, payload: dict[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedSummaryError("ACWM_TRAINING_SEED_SUMMARY_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "summary.json", payload)
        public_summary = {
            key: value
            for key, value in payload.items()
            if key not in {"records"}
        }
        public_summary["records"] = [
            {key: value for key, value in row.items() if key not in {"checkpoint_path", "manifest_path"}}
            for row in payload["records"]  # type: ignore[union-attr]
        ]
        _write_json(temporary / "public-summary.json", public_summary)
        records = payload["records"]
        assert isinstance(records, list)
        fields = [
            "training_seed", "eval_seed", "checkpoint_sha256", "pass",
            "delta_psnr", "delta_ssim", "delta_mse", "delta_masked_mse", "manifest_path",
        ]
        with (temporary / "factorial-cells.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                assert isinstance(row, Mapping)
                delta = row["delta"]
                assert isinstance(delta, Mapping)
                writer.writerow(
                    {
                        "training_seed": row["training_seed"],
                        "eval_seed": row["eval_seed"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "pass": row["pass"],
                        **{f"delta_{metric}": delta[metric] for metric in METRICS},
                        "manifest_path": row["manifest_path"],
                    }
                )
        public_fields = [field for field in fields if field != "manifest_path"]
        with (temporary / "public-factorial-cells.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=public_fields)
            writer.writeheader()
            for row in records:
                assert isinstance(row, Mapping)
                delta = row["delta"]
                assert isinstance(delta, Mapping)
                writer.writerow(
                    {
                        "training_seed": row["training_seed"],
                        "eval_seed": row["eval_seed"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "pass": row["pass"],
                        **{f"delta_{metric}": delta[metric] for metric in METRICS},
                    }
                )
        lines = [
            "# ACWM repair-training seed replication",
            "",
            f"Target: `{payload['environment']} + {payload['primitive']}`",
            f"Official gate pass rate: `{payload['official_gate_pass_count']}/{payload['receipt_count']}`",
            "",
            "| Training seed | Eval seed | Pass | Delta PSNR | Delta SSIM | Delta MSE | Delta masked-MSE |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
        for row in records:
            assert isinstance(row, Mapping)
            delta = row["delta"]
            assert isinstance(delta, Mapping)
            lines.append(
                f"| {row['training_seed']} | {row['eval_seed']} | {row['pass']} | "
                f"{float(delta['psnr']):+.4f} | {float(delta['ssim']):+.6f} | "
                f"{float(delta['mse']):+.6f} | {float(delta['masked_mse']):+.6f} |"
            )
        (temporary / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-training-seed-replication-summary-manifest",
            "state": "ready",
            "summary_path": str(destination / "summary.json"),
            "table_path": str(destination / "factorial-cells.csv"),
            "public_summary_path": str(destination / "public-summary.json"),
            "public_table_path": str(destination / "public-factorial-cells.csv"),
            "receipt_count": payload["receipt_count"],
            "official_gate_pass_count": payload["official_gate_pass_count"],
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_summary(manifests=args.manifest, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
