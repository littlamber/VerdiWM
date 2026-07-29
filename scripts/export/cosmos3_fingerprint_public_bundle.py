#!/usr/bin/env python3
"""Export a path-sanitized Cosmos3 target-local fingerprint bundle."""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Cosmos3FingerprintPublicBundleError(ValueError):
    """A fitted Cosmos3 fingerprint cannot be published safely."""


def export_cosmos3_fingerprint_public_bundle(
    *,
    fingerprint_root: Path,
    shard_manifests: Sequence[Path],
    output_root: Path,
    include_videos: bool = True,
) -> dict[str, object]:
    fingerprint_dir = Path(fingerprint_root).resolve(strict=True)
    report = _load_mapping(fingerprint_dir / "target-local-fingerprint.json")
    campaign = _load_mapping(fingerprint_dir / "input-campaign.json")
    measurements = _load_jsonl(fingerprint_dir / "measurements.jsonl")
    if (
        report.get("artifact_type") != "verdiwm-cosmos3-target-local-fingerprint"
        or report.get("state") != "ready"
        or campaign.get("campaign_id") != report.get("campaign_id")
        or len(measurements) != int(report.get("measurement_count", -1))
    ):
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_FINGERPRINT_INVALID")

    receipt_records = _load_receipt_records(shard_manifests, report)
    measurement_by_key = {
        (float(row["dose"]), int(row["sample_index"]), int(row["seed"])): row
        for row in measurements
    }
    if len(measurement_by_key) != len(measurements) or set(measurement_by_key) != set(receipt_records):
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_MEASUREMENT_RECEIPT_MISMATCH")

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_OUTPUT_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        outcome_names = [str(value) for value in report["chart"]["outcome_names"]]
        probe = campaign.get("probe")
        if not isinstance(probe, Mapping):
            raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_PROBE_INVALID")
        probe_id = str(probe.get("probe_id", ""))
        dose_unit = str(probe.get("dose_unit", ""))
        probe_label = _probe_label(probe_id)
        metric_rows = _metric_rows(receipt_records)
        aggregate_rows = _aggregate_rows(metric_rows)
        _write_csv(temporary / "tables/dose-metrics.csv", metric_rows)
        _write_csv(temporary / "tables/dose-response.csv", aggregate_rows)
        _write_svg(
            temporary / "figures/dose-response.svg",
            aggregate_rows,
            outcome_names,
            probe_label=probe_label,
        )

        sanitized_measurements = []
        for row in measurements:
            sanitized_measurements.append(dict(row))
        (temporary / "measurements.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in sanitized_measurements
            ),
            encoding="utf-8",
        )
        (temporary / "target-local-fingerprint.json").write_text(
            _pretty_json(report), encoding="utf-8"
        )
        (temporary / "input-campaign.json").write_text(_pretty_json(campaign), encoding="utf-8")

        videos = []
        if include_videos:
            identities = sorted({(key[1], key[2]) for key in receipt_records})
            doses = sorted({key[0] for key in receipt_records})
            selected = _selected_video_doses(doses)
            for sample_index, seed in identities:
                relative = f"videos/sample-{sample_index}-seed-{seed}-dose-response.mp4"
                _write_response_video(
                    sources={
                        dose: receipt_records[(dose, sample_index, seed)]["receipt_root"]
                        for dose in selected
                    },
                    output_path=temporary / relative,
                    probe_label=probe_label,
                )
                videos.append(
                    {
                        "sample_index": sample_index,
                        "seed": seed,
                        "layout": "|".join(
                            ["GT", *(f"{probe_label} {dose:+.4f}" for dose in selected)]
                        ),
                        "path": relative,
                    }
                )

        locality = report["locality_admission"]
        bundle: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-target-local-fingerprint-public-bundle",
            "state": "ready",
            "campaign_id": report["campaign_id"],
            "protocol": report["protocol"],
            "split": report["split"],
            "measurement_count": len(measurements),
            "repeat_count": report["repeat_count"],
            "probe_id": probe_id,
            "dose_unit": dose_unit,
            "locality_admission_state": locality["state"],
            "cross_backbone_transfer_eligible": locality["cross_backbone_transfer_eligible"],
            "videos": videos,
            "claim_boundary": report["claim_boundary"],
        }
        (temporary / "bundle.json").write_text(_pretty_json(bundle), encoding="utf-8")
        (temporary / "README.md").write_text(_readme(bundle), encoding="utf-8")
        _assert_public_text(temporary)
        _write_manifest(temporary)
        os.replace(temporary, destination)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_receipt_records(
    shard_manifests: Sequence[Path], report: Mapping[str, Any]
) -> dict[tuple[float, int, int], dict[str, object]]:
    records: dict[tuple[float, int, int], dict[str, object]] = {}
    for manifest_path in shard_manifests:
        shard = _load_mapping(Path(manifest_path))
        if (
            shard.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign-shard"
            or shard.get("state") != "ready"
            or shard.get("campaign_id") != report["campaign_id"]
            or shard.get("protocol") != report["protocol"]
        ):
            raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_SHARD_INVALID")
        for row in shard.get("records", []):
            audit = row.get("gpu_exclusivity_audit")
            if (
                not isinstance(audit, Mapping)
                or audit.get("state") != "ready"
                or audit.get("foreign_pid_events") != []
            ):
                raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_GPU_AUDIT_INVALID")
            receipt_path = Path(str(row["receipt_ref"])).resolve(strict=True)
            if _sha256(receipt_path) != row["receipt_sha256"]:
                raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_RECEIPT_SHA_MISMATCH")
            key = (float(row["dose"]), int(row["sample_index"]), int(row["seed"]))
            if key in records:
                raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_RECEIPT_DUPLICATE")
            records[key] = {
                "receipt_root": receipt_path.parent,
                "receipt": _load_mapping(receipt_path),
                "gpu_uuid": audit["gpu_uuid"],
            }
    return records


def _metric_rows(
    records: Mapping[tuple[float, int, int], Mapping[str, object]]
) -> list[dict[str, object]]:
    baselines = {
        (sample, seed): record["receipt"]["metrics"]
        for (dose, sample, seed), record in records.items()
        if dose == 0.0
    }
    rows = []
    for (dose, sample, seed), record in sorted(records.items()):
        metrics = record["receipt"]["metrics"]
        baseline = baselines.get((sample, seed))
        if baseline is None:
            raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_BASELINE_MISSING")
        row = {
            "dose": dose,
            "sample_index": sample,
            "seed": seed,
            "gpu_uuid": record["gpu_uuid"],
        }
        for name, value in metrics.items():
            row[name] = float(value)
            row[f"delta_{name}"] = float(value) - float(baseline[name])
        rows.append(row)
    return rows


def _selected_video_doses(doses: Sequence[float]) -> tuple[float, float, float]:
    ordered = tuple(sorted(set(float(value) for value in doses)))
    if len(ordered) < 3 or 0.0 not in ordered:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_VIDEO_DOSE_MISSING")
    if ordered[0] < 0.0 < ordered[-1]:
        selected = (ordered[0], 0.0, ordered[-1])
    else:
        selected = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
    if len(set(selected)) != 3:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_VIDEO_DOSE_DUPLICATE")
    return selected


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    oriented = {
        "rollout_video_psnr": 1.0,
        "rollout_video_l1": -1.0,
        "final_frame_mae": -1.0,
        "temporal_difference_mae": -1.0,
    }
    for row in rows:
        for metric, sign in oriented.items():
            grouped[(float(row["dose"]), metric)].append(float(row[metric]) * sign)
    result = []
    for (dose, metric), values in sorted(grouped.items()):
        result.append(
            {
                "dose": dose,
                "outcome": metric if oriented[metric] > 0 else f"negative_{metric}",
                "mean": statistics.fmean(values),
                "population_std": statistics.pstdev(values),
                "repeat_count": len(values),
            }
        )
    return result


def _write_response_video(
    *, sources: Mapping[float, Path], output_path: Path, probe_label: str
) -> None:
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw

    zero = sources[0.0]
    zero_receipt = _load_mapping(zero / "prediction-receipt.json")
    gt = np.load(zero / str(zero_receipt["ground_truth_ref"]), allow_pickle=False)
    predictions = {}
    for dose, root in sources.items():
        receipt = _load_mapping(root / "prediction-receipt.json")
        predictions[dose] = iio.imread(root / str(receipt["rollout_ref"]))
    reference = predictions[0.0]
    if any(value.shape != reference.shape for value in predictions.values()) or gt.shape[0] != reference.shape[0]:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_VIDEO_ALIGNMENT_INVALID")
    gt = gt[:, : reference.shape[1], : reference.shape[2], :]
    labels = ["GT", *[f"{probe_label} {dose:+.4f}" for dose in sorted(sources)]]
    frames = []
    for frame_index in range(reference.shape[0]):
        images = [gt[frame_index], *[predictions[dose][frame_index] for dose in sorted(sources)]]
        canvas = Image.new("RGB", (reference.shape[2] * 4, reference.shape[1] + 32), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = index * reference.shape[2]
            canvas.paste(Image.fromarray(image), (x, 32))
            draw.text((x + 12, 9), label, fill="black")
        frames.append(np.asarray(canvas))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.stack(frames), fps=15, codec="libx264", pixelformat="yuv420p")


def _write_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    outcome_names: Sequence[str],
    *,
    probe_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_outcome: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_outcome[str(row["outcome"])].append(row)
    width, height = 1000, 620
    colors = ("#1976D2", "#C62828", "#2E7D32", "#6A1B9A")
    panels = []
    for index, outcome in enumerate(outcome_names):
        values = sorted(by_outcome[outcome], key=lambda row: float(row["dose"]))
        if not values:
            raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_OUTCOME_MISSING")
        col, row_index = index % 2, index // 2
        x0, y0 = 70 + col * 480, 105 + row_index * 245
        panel_w, panel_h = 390, 155
        means = [float(value["mean"]) for value in values]
        low, high = min(means), max(means)
        padding = max((high - low) * 0.15, 1e-9)
        low, high = low - padding, high + padding
        points = []
        for point_index, value in enumerate(values):
            x = x0 + point_index * panel_w / max(1, len(values) - 1)
            y = y0 + panel_h - (float(value["mean"]) - low) * panel_h / (high - low)
            points.append((x, y, value))
        color = colors[index % len(colors)]
        panels.extend(
            [
                f'<text x="{x0}" y="{y0 - 22}" font-size="16" font-weight="600">{outcome}</text>',
                f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#999"/>',
                f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#999"/>',
                f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)}" fill="none" stroke="{color}" stroke-width="3"/>',
            ]
        )
        for x, y, value in points:
            panels.extend(
                [
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>',
                    f'<text x="{x - 18:.2f}" y="{y0 + panel_h + 21}" font-size="12">{float(value["dose"]):+.2f}</text>',
                ]
            )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="40" y="42" font-size="24" font-weight="600">Cosmos3 target-local {probe_label} response</text>',
            '<text x="40" y="70" font-size="14" fill="#555">Paired frozen DROID windows; every plotted outcome is oriented so higher is better</text>',
            *panels,
            '</svg>',
        ]
    )
    path.write_text(svg + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_TABLE_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _readme(bundle: Mapping[str, object]) -> str:
    return (
        "# Cosmos3 Target-Local Fingerprint\n\n"
        f"This bundle contains `{bundle['measurement_count']}` paired-dose measurements "
        f"on the frozen `{bundle['split']}` split. Locality admission is "
        f"`{bundle['locality_admission_state']}`.\n\n"
        f"The chart measures a target-local response to reversible `{bundle['probe_id']}` "
        f"doses in `{bundle['dose_unit']}` units. "
        "It is not by itself evidence of model improvement or cross-backbone transfer.\n"
    )


def _probe_label(probe_id: str) -> str:
    labels = {
        "action_conditioning_scale": "action scale",
        "action_embedding_temporal_mix": "temporal action mix",
    }
    try:
        return labels[probe_id]
    except KeyError as exc:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_PROBE_INVALID") from exc


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3FingerprintPublicBundleError(f"COSMOS3_PUBLIC_JSON_INVALID:{path.name}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = [_load_mapping_from_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_MEASUREMENTS_EMPTY")
    return rows


def _load_mapping_from_line(line: str) -> Mapping[str, Any]:
    payload = json.loads(line)
    if not isinstance(payload, Mapping):
        raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_MEASUREMENT_INVALID")
    return payload


def _assert_public_text(root: Path) -> None:
    prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8")
            if any(prefix in text for prefix in prefixes):
                raise Cosmos3FingerprintPublicBundleError("COSMOS3_PUBLIC_LOCAL_PATH_LEAK")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args(argv)
    bundle = export_cosmos3_fingerprint_public_bundle(
        fingerprint_root=args.fingerprint_root,
        shard_manifests=args.shard_manifest,
        output_root=args.output_root,
        include_videos=not args.no_videos,
    )
    print(json.dumps(bundle, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
