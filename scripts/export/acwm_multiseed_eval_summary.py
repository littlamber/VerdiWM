#!/usr/bin/env python3
"""Export an ACWM frozen-checkpoint eval-seed replication summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


METRICS = ("psnr", "ssim", "mse", "masked_mse")


class AcwmMultiseedSummaryError(ValueError):
    """Source receipts do not form a valid eval-seed replication set."""


def export_acwm_multiseed_eval_summary(
    *,
    receipt_paths: Sequence[Path],
    output_root: Path,
    expected_seeds_per_cell: int = 3,
) -> dict[str, object]:
    if expected_seeds_per_cell < 2:
        raise AcwmMultiseedSummaryError("ACWM_MULTISEED_EXPECTED_SEEDS_TOO_SMALL")

    records = [_load_receipt(Path(path)) for path in receipt_paths]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["environment"]), str(record["primitive"]))].append(record)
    if not grouped:
        raise AcwmMultiseedSummaryError("ACWM_MULTISEED_EMPTY")

    rows: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for identity in sorted(grouped):
        cell_records = sorted(grouped[identity], key=lambda row: int(row["eval_seed"]))
        seeds = [int(row["eval_seed"]) for row in cell_records]
        if len(seeds) != expected_seeds_per_cell or len(set(seeds)) != len(seeds):
            raise AcwmMultiseedSummaryError(
                f"ACWM_MULTISEED_INCOMPLETE:{identity}:expected={expected_seeds_per_cell}:actual={seeds}"
            )
        _require_single_value(cell_records, "candidate_checkpoint_sha256", identity)
        _require_single_value(cell_records, "baseline_checkpoint_sha256", identity)
        pass_count = sum(bool(record["pass"]) for record in cell_records)
        pass_rate = pass_count / len(cell_records)
        if pass_count == len(cell_records):
            stability = "eval_seed_robust"
        elif pass_count >= 2:
            stability = "eval_seed_sensitive_positive"
        elif pass_count == 1:
            stability = "eval_seed_fragile"
        else:
            stability = "no_replication_pass"
        aggregate = {
            metric: {
                "mean_delta": statistics.fmean(float(record[f"delta_{metric}"]) for record in cell_records),
                "population_std_delta": statistics.pstdev(
                    float(record[f"delta_{metric}"]) for record in cell_records
                ),
            }
            for metric in METRICS
        }
        cells.append(
            {
                "environment": identity[0],
                "primitive": identity[1],
                "eval_seeds": seeds,
                "pass_count": pass_count,
                "seed_count": len(cell_records),
                "pass_rate": pass_rate,
                "stability": stability,
                "candidate_checkpoint_sha256": cell_records[0]["candidate_checkpoint_sha256"],
                "aggregates": aggregate,
            }
        )
        rows.extend(cell_records)

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmMultiseedSummaryError("ACWM_MULTISEED_OUTPUT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "tables").mkdir()
        (temporary / "figures").mkdir()
        (temporary / "videos").mkdir()
        _write_seed_csv(temporary / "tables/eval-seed-results.csv", rows)
        _write_cell_csv(temporary / "tables/cell-stability.csv", cells)
        video_rows = _copy_representative_videos(temporary / "videos", rows)
        _write_svg(temporary / "figures/eval-seed-replication.svg", cells)
        summary: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-eval-seed-replication-summary",
            "state": "ready",
            "cell_count": len(cells),
            "receipt_count": len(rows),
            "expected_seeds_per_cell": expected_seeds_per_cell,
            "cells": cells,
            "videos": video_rows,
            "claim_boundary": (
                "Frozen-checkpoint robustness across evaluation randomness only. "
                "This is not independent training-seed evidence and does not establish cross-backbone transfer."
            ),
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "README.md").write_text(_readme(summary), encoding="utf-8")
        _write_manifest(temporary)
        os.replace(temporary, destination)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_receipt(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "wmloop-acwm-formal-visualization-export":
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_WRONG_ARTIFACT:{source}")
    gate = payload.get("official_quality_gate")
    if not isinstance(gate, dict) or gate.get("state") not in {"pass", "fail"}:
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_UNSETTLED_GATE:{source}")
    if gate.get("protocol") != "official_acwm_eval_py" or float(payload.get("steps", 0)) != 50:
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_PROTOCOL_MISMATCH:{source}")
    deltas = gate.get("delta_candidate_minus_baseline")
    if not isinstance(deltas, dict) or any(metric not in deltas for metric in METRICS):
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_DELTA_MISSING:{source}")
    required = (
        "environment",
        "primitive",
        "eval_seed",
        "candidate_checkpoint_sha256",
        "baseline_checkpoint_sha256",
    )
    if any(payload.get(name) is None for name in required):
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_IDENTITY_MISSING:{source}")
    selected = payload.get("hard_case_visualization", {}).get("selected", [])
    if not selected or not selected[0].get("selected_video_path"):
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_VIDEO_MISSING:{source}")
    video = Path(str(selected[0]["selected_video_path"])).resolve(strict=True)
    return {
        "environment": payload["environment"],
        "primitive": payload["primitive"],
        "eval_seed": int(payload["eval_seed"]),
        "pass": gate.get("pass") is True,
        "delta_psnr": float(deltas["psnr"]),
        "delta_ssim": float(deltas["ssim"]),
        "delta_mse": float(deltas["mse"]),
        "delta_masked_mse": float(deltas["masked_mse"]),
        "candidate_checkpoint_sha256": payload["candidate_checkpoint_sha256"],
        "baseline_checkpoint_sha256": payload["baseline_checkpoint_sha256"],
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "representative_video": str(video),
        "representative_video_sha256": _sha256(video),
    }


def _require_single_value(
    records: Sequence[dict[str, object]], key: str, identity: tuple[str, str]
) -> None:
    values = {str(record[key]) for record in records}
    if len(values) != 1:
        raise AcwmMultiseedSummaryError(f"ACWM_MULTISEED_CHECKPOINT_MISMATCH:{identity}:{key}")


def _copy_representative_videos(root: Path, rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    copied = []
    for row in rows:
        name = f"{row['environment']}-{row['primitive']}-seed-{row['eval_seed']}.mp4"
        target = root / name
        shutil.copy2(str(row["representative_video"]), target)
        digest = _sha256(target)
        if digest != row["representative_video_sha256"]:
            raise AcwmMultiseedSummaryError("ACWM_MULTISEED_VIDEO_COPY_SHA_MISMATCH")
        copied.append(
            {
                "environment": row["environment"],
                "primitive": row["primitive"],
                "eval_seed": row["eval_seed"],
                "official_gate_pass": row["pass"],
                "layout": "GT|baseline prediction|primitive prediction",
                "path": f"videos/{name}",
                "sha256": digest,
            }
        )
    return copied


def _write_seed_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "environment",
        "primitive",
        "eval_seed",
        "pass",
        "delta_psnr",
        "delta_ssim",
        "delta_mse",
        "delta_masked_mse",
        "candidate_checkpoint_sha256",
        "source_manifest_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_cell_csv(path: Path, cells: Sequence[dict[str, object]]) -> None:
    fields = (
        "environment",
        "primitive",
        "pass_count",
        "seed_count",
        "pass_rate",
        "stability",
        "mean_delta_psnr",
        "std_delta_psnr",
        "mean_delta_ssim",
        "mean_delta_mse",
        "mean_delta_masked_mse",
        "candidate_checkpoint_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            aggregate = cell["aggregates"]
            writer.writerow(
                {
                    **{field: cell[field] for field in fields if field in cell},
                    "mean_delta_psnr": aggregate["psnr"]["mean_delta"],
                    "std_delta_psnr": aggregate["psnr"]["population_std_delta"],
                    "mean_delta_ssim": aggregate["ssim"]["mean_delta"],
                    "mean_delta_mse": aggregate["mse"]["mean_delta"],
                    "mean_delta_masked_mse": aggregate["masked_mse"]["mean_delta"],
                }
            )


def _write_svg(path: Path, cells: Sequence[dict[str, object]]) -> None:
    width = 980
    height = 126 + 82 * len(cells)
    rows = []
    colors = {"eval_seed_robust": "#18864B", "eval_seed_sensitive_positive": "#E38B20"}
    for index, cell in enumerate(cells):
        y = 90 + index * 82
        rate = float(cell["pass_rate"])
        color = colors.get(str(cell["stability"]), "#C23B3B")
        rows.extend(
            [
                f'<text x="28" y="{y + 21}" font-size="16">{cell["environment"]} + {cell["primitive"]}</text>',
                f'<rect x="470" y="{y}" width="{360 * rate:.2f}" height="28" fill="{color}"/>',
                f'<rect x="470" y="{y}" width="360" height="28" fill="none" stroke="#444"/>',
                f'<text x="842" y="{y + 20}" font-size="15">{cell["pass_count"]}/{cell["seed_count"]}</text>',
                f'<text x="470" y="{y + 52}" font-size="13" fill="#555">mean delta PSNR {cell["aggregates"]["psnr"]["mean_delta"]:+.3f} dB | {cell["stability"]}</text>',
            ]
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="28" y="36" font-size="24" font-weight="600">ACWM frozen-checkpoint eval-seed replication</text>',
            '<text x="28" y="63" font-size="14" fill="#555">Official 50-step four-metric gate; not independent training-seed evidence</text>',
            *rows,
            '</svg>',
        ]
    )
    path.write_text(svg + "\n", encoding="utf-8")


def _readme(summary: dict[str, object]) -> str:
    robust = sum(cell["stability"] == "eval_seed_robust" for cell in summary["cells"])
    return (
        "# ACWM Eval-Seed Replication\n\n"
        f"This bundle contains `{summary['receipt_count']}` settled official-gate receipts "
        f"over `{summary['cell_count']}` environment/primitive cells. `{robust}` cells pass "
        "all frozen evaluation seeds.\n\n"
        "The videos use `GT | baseline prediction | primitive prediction`. This bundle measures "
        "evaluation-randomness robustness of fixed checkpoints; independent training-seed "
        "replication remains separate work.\n"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-seeds-per-cell", type=int, default=3)
    args = parser.parse_args(argv)
    summary = export_acwm_multiseed_eval_summary(
        receipt_paths=args.receipt,
        output_root=args.output_root,
        expected_seeds_per_cell=args.expected_seeds_per_cell,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
