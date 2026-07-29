#!/usr/bin/env python3
"""Aggregate a checkpoint ladder of ACWM training-seed factorial summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


class AcwmTrainingSeedStabilitySummaryError(RuntimeError):
    """The checkpoint stability evidence is incomplete or inconsistent."""


def build_stability_summary(
    *,
    checkpoint_summaries: Mapping[int, Path],
    output_root: Path,
) -> dict[str, object]:
    if len(checkpoint_summaries) < 2:
        raise AcwmTrainingSeedStabilitySummaryError("ACWM_STABILITY_SUMMARIES_TOO_FEW")
    loaded: dict[int, dict[str, object]] = {}
    for step, path in sorted(checkpoint_summaries.items()):
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise AcwmTrainingSeedStabilitySummaryError("ACWM_STABILITY_STEP_INVALID")
        payload = _load_json(Path(path).resolve(strict=True))
        if (
            payload.get("artifact_type") != "verdiwm-acwm-training-seed-replication-summary"
            or payload.get("state") != "ready"
        ):
            raise AcwmTrainingSeedStabilitySummaryError(
                f"ACWM_STABILITY_SUMMARY_INVALID:{step}"
            )
        loaded[step] = payload
    reference = loaded[min(loaded)]
    identity = (
        reference.get("environment"),
        reference.get("primitive"),
        reference.get("training_seeds"),
        reference.get("eval_seeds"),
        reference.get("factorial_shape"),
    )
    for step, payload in loaded.items():
        observed = (
            payload.get("environment"),
            payload.get("primitive"),
            payload.get("training_seeds"),
            payload.get("eval_seeds"),
            payload.get("factorial_shape"),
        )
        if observed != identity:
            raise AcwmTrainingSeedStabilitySummaryError(
                f"ACWM_STABILITY_FACTORIAL_MISMATCH:{step}"
            )

    training_seeds = _int_list(reference.get("training_seeds"), "ACWM_STABILITY_TRAINING_SEEDS_INVALID")
    eval_seeds = _int_list(reference.get("eval_seeds"), "ACWM_STABILITY_EVAL_SEEDS_INVALID")
    expected_cells = len(training_seeds) * len(eval_seeds)
    checkpoint_records: list[dict[str, object]] = []
    by_step_seed: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for step, payload in loaded.items():
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != expected_cells:
            raise AcwmTrainingSeedStabilitySummaryError(
                f"ACWM_STABILITY_FACTORIAL_INCOMPLETE:{step}"
            )
        typed_records = [record for record in records if isinstance(record, Mapping)]
        if len(typed_records) != expected_cells:
            raise AcwmTrainingSeedStabilitySummaryError(
                f"ACWM_STABILITY_FACTORIAL_INCOMPLETE:{step}"
            )
        for training_seed in training_seeds:
            seed_records = [
                record for record in typed_records if record.get("training_seed") == training_seed
            ]
            if {record.get("eval_seed") for record in seed_records} != set(eval_seeds):
                raise AcwmTrainingSeedStabilitySummaryError(
                    f"ACWM_STABILITY_SEED_MATRIX_INCOMPLETE:{step}:{training_seed}"
                )
            by_step_seed[(step, training_seed)] = seed_records
        metric_summary = _mapping(payload.get("metric_summary"), "ACWM_STABILITY_METRICS_INVALID")
        psnr = _mapping(metric_summary.get("psnr"), "ACWM_STABILITY_PSNR_INVALID")
        failing_cells = [
            {
                "training_seed": int(record["training_seed"]),
                "eval_seed": int(record["eval_seed"]),
                "delta": dict(_mapping(record.get("delta"), "ACWM_STABILITY_DELTA_INVALID")),
            }
            for record in typed_records
            if record.get("pass") is not True
        ]
        checkpoint_records.append(
            {
                "checkpoint_step": step,
                "all_cells_pass": payload.get("all_cells_pass") is True,
                "pass_count": int(payload.get("official_gate_pass_count", 0)),
                "cell_count": expected_cells,
                "pass_rate": float(payload.get("official_gate_pass_rate", 0.0)),
                "psnr_mean": _finite_float(psnr.get("mean"), "ACWM_STABILITY_PSNR_INVALID"),
                "psnr_min": _finite_float(psnr.get("min"), "ACWM_STABILITY_PSNR_INVALID"),
                "psnr_max": _finite_float(psnr.get("max"), "ACWM_STABILITY_PSNR_INVALID"),
                "psnr_sample_std": _finite_float(
                    psnr.get("sample_std"), "ACWM_STABILITY_PSNR_INVALID"
                ),
                "failing_cells": failing_cells,
            }
        )

    per_training_seed: list[dict[str, object]] = []
    selected_checkpoints: list[dict[str, object]] = []
    for training_seed in training_seeds:
        candidates: list[dict[str, object]] = []
        for step in sorted(loaded):
            records = by_step_seed[(step, training_seed)]
            deltas = [
                _finite_float(
                    _mapping(record.get("delta"), "ACWM_STABILITY_DELTA_INVALID").get("psnr"),
                    "ACWM_STABILITY_PSNR_INVALID",
                )
                for record in records
            ]
            checkpoint_shas = {str(record.get("checkpoint_sha256") or "") for record in records}
            checkpoint_paths = {str(record.get("checkpoint_path") or "") for record in records}
            if len(checkpoint_shas) != 1 or "" in checkpoint_shas or len(checkpoint_paths) != 1:
                raise AcwmTrainingSeedStabilitySummaryError(
                    f"ACWM_STABILITY_CHECKPOINT_IDENTITY_INVALID:{step}:{training_seed}"
                )
            candidates.append(
                {
                    "checkpoint_step": step,
                    "all_eval_seeds_pass": all(record.get("pass") is True for record in records),
                    "mean_delta_psnr": mean(deltas),
                    "min_delta_psnr": min(deltas),
                    "checkpoint_sha256": next(iter(checkpoint_shas)),
                    "checkpoint_path": next(iter(checkpoint_paths)),
                }
            )
        eligible = [candidate for candidate in candidates if candidate["all_eval_seeds_pass"]]
        if not eligible:
            raise AcwmTrainingSeedStabilitySummaryError(
                f"ACWM_STABILITY_NO_PASSING_CHECKPOINT:{training_seed}"
            )
        selected = max(
            eligible,
            key=lambda item: (float(item["mean_delta_psnr"]), -int(item["checkpoint_step"])),
        )
        per_training_seed.append(
            {
                "training_seed": training_seed,
                "checkpoint_candidates": candidates,
                "selected_checkpoint_step": selected["checkpoint_step"],
                "selection_reason": "all_eval_seeds_pass_then_max_mean_psnr_tie_earlier_step",
            }
        )
        selected_checkpoints.append(
            {
                "training_seed": training_seed,
                "checkpoint_step": selected["checkpoint_step"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "checkpoint_path": selected["checkpoint_path"],
                "mean_delta_psnr": selected["mean_delta_psnr"],
                "min_delta_psnr": selected["min_delta_psnr"],
            }
        )

    globally_eligible = [record for record in checkpoint_records if record["all_cells_pass"]]
    if not globally_eligible:
        raise AcwmTrainingSeedStabilitySummaryError("ACWM_STABILITY_NO_GLOBAL_PASSING_CHECKPOINT")
    global_selected = max(
        globally_eligible,
        key=lambda item: (float(item["psnr_mean"]), -int(item["checkpoint_step"])),
    )
    max_step_record = checkpoint_records[-1]
    summary = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary",
        "state": "ready",
        "environment": reference["environment"],
        "primitive": reference["primitive"],
        "training_seeds": training_seeds,
        "eval_seeds": eval_seeds,
        "factorial_shape": [len(training_seeds), len(eval_seeds)],
        "checkpoint_steps": sorted(loaded),
        "checkpoint_records": checkpoint_records,
        "per_training_seed": per_training_seed,
        "selected_checkpoints": selected_checkpoints,
        "global_selected_checkpoint_step": global_selected["checkpoint_step"],
        "global_selection_reason": "all_factorial_cells_pass_then_max_mean_psnr_tie_earlier_step",
        "stability_verdict": (
            "stable_through_max_checkpoint"
            if max_step_record["all_cells_pass"]
            else "earlier_checkpoint_retained_after_max_checkpoint_regression"
        ),
        "claim_boundary": (
            "Independent repair-fine-tuning checkpoint stability only. Selection requires all shared "
            "evaluation seeds to pass the frozen official gate and does not establish cross-backbone transfer."
        ),
    }
    return _write_bundle(Path(output_root).resolve(), summary)


def _write_bundle(destination: Path, summary: dict[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedStabilitySummaryError("ACWM_STABILITY_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "summary.json", summary)
        public = json.loads(json.dumps(summary))
        for selected in public["selected_checkpoints"]:
            selected.pop("checkpoint_path", None)
        for seed_record in public["per_training_seed"]:
            for candidate in seed_record["checkpoint_candidates"]:
                candidate.pop("checkpoint_path", None)
        _write_json(temporary / "public-summary.json", public)
        _write_json(
            temporary / "selected-checkpoints.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-acwm-selected-checkpoints",
                "state": "ready",
                "selection_rule": "all_eval_seeds_pass_then_max_mean_psnr_tie_earlier_step",
                "records": summary["selected_checkpoints"],
            },
        )
        with (temporary / "checkpoint-ladder.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "checkpoint_step",
                "pass_count",
                "cell_count",
                "pass_rate",
                "all_cells_pass",
                "psnr_mean",
                "psnr_min",
                "psnr_max",
                "psnr_sample_std",
                "failing_cell_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in summary["checkpoint_records"]:
                assert isinstance(record, Mapping)
                writer.writerow(
                    {
                        **{field: record.get(field, "") for field in fields},
                        "failing_cell_count": len(record.get("failing_cells", [])),
                    }
                )
        _write_markdown(temporary / "README.md", summary)
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-training-seed-checkpoint-stability-summary-manifest",
            "state": "ready",
            "summary_path": str(destination / "summary.json"),
            "public_summary_path": str(destination / "public-summary.json"),
            "selected_checkpoints_path": str(destination / "selected-checkpoints.json"),
            "checkpoint_ladder_path": str(destination / "checkpoint-ladder.csv"),
            "checkpoint_steps": summary["checkpoint_steps"],
            "global_selected_checkpoint_step": summary["global_selected_checkpoint_step"],
            "stability_verdict": summary["stability_verdict"],
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


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# ACWM training-seed checkpoint stability",
        "",
        f"Target: `{summary['environment']} + {summary['primitive']}`",
        f"Verdict: `{summary['stability_verdict']}`",
        f"Global selected step: `{summary['global_selected_checkpoint_step']}`",
        "",
        "| Step | Pass | Mean PSNR delta | Min PSNR delta | Failures |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in summary["checkpoint_records"]:
        assert isinstance(record, Mapping)
        lines.append(
            f"| {record['checkpoint_step']} | {record['pass_count']}/{record['cell_count']} | "
            f"{float(record['psnr_mean']):+.4f} | {float(record['psnr_min']):+.4f} | "
            f"{len(record['failing_cells'])} |"
        )
    lines.extend(["", "## Per-training-seed selection", ""])
    for record in summary["selected_checkpoints"]:
        assert isinstance(record, Mapping)
        lines.append(
            f"- seed `{record['training_seed']}` -> step `{record['checkpoint_step']}` "
            f"(mean PSNR `{float(record['mean_delta_psnr']):+.4f}`)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_checkpoint_summary(value: str) -> tuple[int, Path]:
    step_raw, separator, path_raw = value.partition("=")
    if not separator or not path_raw:
        raise argparse.ArgumentTypeError("expected STEP=PATH")
    try:
        step = int(step_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint step must be an integer") from exc
    return step, Path(path_raw)


def _mapping(value: object, error: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AcwmTrainingSeedStabilitySummaryError(error)
    return value


def _int_list(value: object, error: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise AcwmTrainingSeedStabilitySummaryError(error)
    return list(value)


def _finite_float(value: object, error: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcwmTrainingSeedStabilitySummaryError(error)
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise AcwmTrainingSeedStabilitySummaryError(error)
    return result


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcwmTrainingSeedStabilitySummaryError(f"ACWM_STABILITY_JSON_INVALID:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-summary",
        action="append",
        type=_parse_checkpoint_summary,
        required=True,
        metavar="STEP=PATH",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint_summaries: dict[int, Path] = {}
    for step, path in args.checkpoint_summary:
        if step in checkpoint_summaries:
            parser.error(f"duplicate checkpoint step: {step}")
        checkpoint_summaries[step] = path
    manifest = build_stability_summary(
        checkpoint_summaries=checkpoint_summaries,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
