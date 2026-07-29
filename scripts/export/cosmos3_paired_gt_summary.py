#!/usr/bin/env python3
"""Export a self-contained Cosmos3 paired-GT split summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from wmloop.evaluate.adapters.cosmos3_predictive import evaluate_cosmos3_prediction_receipt
from wmloop.evaluate.cosmos3_paired_gt import sha256_file


METRICS = (
    "rollout_video_psnr",
    "rollout_video_l1",
    "final_frame_mae",
    "temporal_difference_mae",
)


class Cosmos3PairedSummaryError(ValueError):
    """Paired receipts are incomplete, mutable, or outside one frozen split."""


def export_cosmos3_paired_gt_summary(
    *,
    receipt_roots: Sequence[Path],
    split_path: Path,
    split_name: str,
    output_root: Path,
    include_videos: bool = True,
) -> dict[str, object]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    expected = {(int(row["sample_index"]), int(row["seed"])) for row in split[split_name]}
    rows: list[dict[str, object]] = []
    sources: dict[tuple[int, int], tuple[Path, dict[str, object]]] = {}
    for raw_root in receipt_roots:
        root = Path(raw_root).resolve(strict=True)
        receipt_path = root / "prediction-receipt.json"
        evidence = evaluate_cosmos3_prediction_receipt(
            receipt_path=receipt_path,
            heldout_split_path=split_path,
            split_name=split_name,
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity = (int(receipt["sample_index"]), int(receipt["seed"]))
        if identity in sources:
            raise Cosmos3PairedSummaryError("COSMOS3_SUMMARY_DUPLICATE_IDENTITY")
        _verify_self_contained_receipt(root, receipt)
        sources[identity] = (root, receipt)
        row: dict[str, object] = {
            "sample_index": identity[0],
            "seed": identity[1],
            "conditioning_frame_mae": receipt["frame_alignment"]["conditioning_frame_mae"],
            "rollout_sha256": receipt["sha256"]["rollout"],
        }
        row.update({name: evidence["metrics"][name] for name in METRICS})
        rows.append(row)
    if set(sources) != expected:
        raise Cosmos3PairedSummaryError(
            f"COSMOS3_SUMMARY_SPLIT_INCOMPLETE:expected={sorted(expected)}:actual={sorted(sources)}"
        )
    rows.sort(key=lambda row: int(row["sample_index"]))

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3PairedSummaryError("COSMOS3_SUMMARY_OUTPUT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "tables").mkdir()
        (temporary / "figures").mkdir()
        _write_csv(temporary / "tables/dev-metrics.csv", rows)
        _write_svg(temporary / "figures/dev-paired-gt-metrics.svg", rows)
        video_rows: list[dict[str, object]] = []
        if include_videos:
            (temporary / "videos").mkdir()
            for row in rows:
                identity = (int(row["sample_index"]), int(row["seed"]))
                source_root, receipt = sources[identity]
                video_name = f"sample-{identity[0]}-seed-{identity[1]}-gt-prediction.mp4"
                _write_pair_video(
                    gt_path=source_root / str(receipt["ground_truth_ref"]),
                    rollout_path=source_root / str(receipt["rollout_ref"]),
                    output_path=temporary / "videos" / video_name,
                )
                video_rows.append(
                    {
                        "sample_index": identity[0],
                        "seed": identity[1],
                        "layout": "GT|Cosmos3-Nano prediction",
                        "path": f"videos/{video_name}",
                    }
                )
        aggregates = {
            name: {
                "mean": statistics.fmean(float(row[name]) for row in rows),
                "population_std": statistics.pstdev(float(row[name]) for row in rows),
            }
            for name in METRICS
        }
        summary: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-paired-gt-split-summary",
            "state": "ready",
            "split_id": split["split_id"],
            "split": split_name,
            "receipt_count": len(rows),
            "identities": [
                {"sample_index": row["sample_index"], "seed": row["seed"]} for row in rows
            ],
            "aggregates": aggregates,
            "videos": video_rows,
            "claim_boundary": "Baseline predictive quality on the complete frozen dev split; no primitive benefit or transfer claim.",
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "README.md").write_text(_readme(summary), encoding="utf-8")
        _write_manifest(temporary)
        os.replace(temporary, destination)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_self_contained_receipt(root: Path, receipt: dict[str, object]) -> None:
    references = {
        "action_input": "action_ref",
        "conditioning": "conditioning_ref",
        "ground_truth": "ground_truth_ref",
        "rollout": "rollout_ref",
    }
    for digest_name, ref_name in references.items():
        relative = Path(str(receipt[ref_name]))
        if relative.is_absolute() or ".." in relative.parts:
            raise Cosmos3PairedSummaryError("COSMOS3_SUMMARY_NONPORTABLE_REFERENCE")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise Cosmos3PairedSummaryError("COSMOS3_SUMMARY_REFERENCE_ESCAPE")
        if sha256_file(path) != receipt["sha256"][digest_name]:
            raise Cosmos3PairedSummaryError(f"COSMOS3_SUMMARY_SHA_MISMATCH:{digest_name}")


def _write_pair_video(*, gt_path: Path, rollout_path: Path, output_path: Path) -> None:
    gt = np.load(gt_path, allow_pickle=False)
    pred = iio.imread(rollout_path)
    if gt.shape[0] != pred.shape[0] or gt.shape[-1] != 3 or pred.shape[-1] != 3:
        raise Cosmos3PairedSummaryError("COSMOS3_SUMMARY_VIDEO_ALIGNMENT_INVALID")
    gt = gt[:, : pred.shape[1], : pred.shape[2], :]
    frames = []
    for gt_frame, pred_frame in zip(gt, pred, strict=True):
        canvas = Image.new("RGB", (pred.shape[2] * 2, pred.shape[1] + 32), "white")
        canvas.paste(Image.fromarray(gt_frame), (0, 32))
        canvas.paste(Image.fromarray(pred_frame), (pred.shape[2], 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 9), "GT", fill="black")
        draw.text((pred.shape[2] + 12, 9), "Cosmos3-Nano prediction", fill="black")
        frames.append(np.asarray(canvas))
    iio.imwrite(output_path, np.stack(frames), fps=15, codec="libx264", pixelformat="yuv420p")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = ["sample_index", "seed", *METRICS, "conditioning_frame_mae", "rollout_sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, rows: Sequence[dict[str, object]]) -> None:
    width, height = 920, 390
    max_psnr = max(float(row["rollout_video_psnr"]) for row in rows) * 1.1
    bars = []
    for index, row in enumerate(rows):
        y = 92 + index * 72
        bar_width = 560 * float(row["rollout_video_psnr"]) / max_psnr
        bars.extend(
            [
                f'<text x="28" y="{y + 21}" font-size="16">sample {row["sample_index"]}</text>',
                f'<rect x="150" y="{y}" width="{bar_width:.2f}" height="30" fill="#1976D2"/>',
                f'<text x="{160 + bar_width:.2f}" y="{y + 21}" font-size="16">{float(row["rollout_video_psnr"]):.3f} dB</text>',
                f'<text x="150" y="{y + 50}" font-size="13" fill="#555">L1 {float(row["rollout_video_l1"]):.4f}  final MAE {float(row["final_frame_mae"]):.4f}  temporal MAE {float(row["temporal_difference_mae"]):.4f}</text>',
            ]
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="28" y="38" font-size="24" font-weight="600">Cosmos3-Nano paired-GT dev baseline</text>',
            '<text x="28" y="66" font-size="14" fill="#555">Frozen DROID windows; future 16 frames only; higher PSNR is better</text>',
            *bars,
            '</svg>',
        ]
    )
    path.write_text(svg + "\n", encoding="utf-8")


def _readme(summary: dict[str, object]) -> str:
    mean = summary["aggregates"]["rollout_video_psnr"]["mean"]
    return (
        "# Cosmos3-Nano Paired-GT Dev Baseline\n\n"
        f"Complete frozen dev coverage: `{summary['receipt_count']}` receipts. "
        f"Mean future-frame PSNR: `{mean:.4f}` dB.\n\n"
        "The CSV and SVG are derived from self-contained, SHA-verified receipts. "
        "Videos are aligned `GT | Cosmos3-Nano prediction` views. This is baseline "
        "quality evidence, not a primitive or transfer result.\n"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", type=Path, action="append", required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--split-name", choices=("dev", "accept"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args(argv)
    summary = export_cosmos3_paired_gt_summary(
        receipt_roots=args.receipt_root,
        split_path=args.split_path,
        split_name=args.split_name,
        output_root=args.output_root,
        include_videos=not args.no_videos,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
