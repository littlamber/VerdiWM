#!/usr/bin/env python3
"""Bundle four validated ACWM hard-case videos into one auditable handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence


class FourSceneBundleError(RuntimeError):
    """A source result was not eligible for the four-scene bundle."""


def run_bundle(
    *,
    output_root: Path,
    gate_manifests: Sequence[Path],
    showcase_manifests: Sequence[Path] | None = None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise FourSceneBundleError("FOUR_SCENE_BUNDLE_OUTPUT_EXISTS")
    if len(gate_manifests) != 4:
        raise FourSceneBundleError("FOUR_SCENE_BUNDLE_REQUIRES_FOUR_GATES")
    if showcase_manifests is not None and len(showcase_manifests) != 4:
        raise FourSceneBundleError("FOUR_SCENE_BUNDLE_REQUIRES_FOUR_SHOWCASES")

    gate_paths = [Path(path).resolve() for path in gate_manifests]
    showcase_by_environment: dict[str, Path] = {}
    if showcase_manifests is not None:
        for path in showcase_manifests:
            resolved = Path(path).resolve()
            payload = _load_json(resolved)
            environment = str(payload.get("environment") or "")
            if not environment or environment in showcase_by_environment:
                raise FourSceneBundleError("FOUR_SCENE_BUNDLE_SHOWCASE_ENVIRONMENTS_INVALID")
            showcase_by_environment[environment] = resolved

    records = []
    for gate_path in gate_paths:
        gate_payload = _load_json(gate_path)
        environment = str(gate_payload.get("environment") or "")
        showcase_path = showcase_by_environment.get(environment) if showcase_manifests is not None else None
        if showcase_manifests is not None and showcase_path is None:
            raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_SHOWCASE_MISSING:{environment}")
        records.append(_source_record(gate_path, showcase_path=showcase_path, gate_payload=gate_payload))
    environments = [str(record["environment"]) for record in records]
    if len(set(environments)) != 4:
        raise FourSceneBundleError("FOUR_SCENE_BUNDLE_ENVIRONMENTS_NOT_UNIQUE")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        video_root = temporary / "videos"
        video_root.mkdir(mode=0o700, parents=True)
        bundled: list[dict[str, object]] = []
        for index, record in enumerate(records, start=1):
            source = Path(str(record["source_video_path"]))
            target = video_root / f"{index:02d}_{record['environment']}_{record['primitive']}_gt_baseline_ours.mp4"
            shutil.copy2(source, target)
            bundled.append(
                {
                    **record,
                    "video_path": str(destination / "videos" / target.name),
                    "video_sha256": _sha256(target),
                    "video_size_bytes": target.stat().st_size,
                }
            )

        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-four-scene-visible-uplift-bundle",
            "state": "ready",
            "scene_count": 4,
            "environments": environments,
            "records": bundled,
            "selection_rule": (
                "one baseline-only hardest case from each supplementary held-out showcase pool"
                if showcase_manifests is not None
                else "one baseline-only hardest case from each frozen official gate"
            ),
            "claim_boundary": (
                "Each numeric row comes from its supplied frozen official 50-step PSNR/SSIM/MSE/masked-MSE gate. "
                "Videos are qualitative evidence selected only by baseline error against GT; supplementary showcase "
                "metrics do not replace the frozen gate."
            ),
        }
        _write_json(temporary / "manifest.json", report)
        _write_csv(temporary / "metrics.csv", bundled)
        (temporary / "README.md").write_text(_markdown(bundled), encoding="utf-8")
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_record(
    path: Path,
    *,
    showcase_path: Path | None = None,
    gate_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_MANIFEST_INVALID:{path}")
    payload = dict(gate_payload) if gate_payload is not None else _load_json(path)
    gate = payload.get("official_quality_gate")
    if not isinstance(gate, Mapping) or gate.get("pass") is not True:
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_GATE_NOT_PASSING:{path}")
    showcase_payload = _load_json(showcase_path) if showcase_path is not None else payload
    if (
        str(showcase_payload.get("environment") or "") != str(payload.get("environment") or "")
        or str(showcase_payload.get("primitive") or "") != str(payload.get("primitive") or "")
    ):
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_SHOWCASE_MISMATCH:{showcase_path}")
    hard_case = showcase_payload.get("hard_case_visualization")
    if not isinstance(hard_case, Mapping) or "baseline-only" not in str(hard_case.get("selection_rule") or ""):
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_HARD_CASE_INVALID:{showcase_path or path}")
    selected = hard_case.get("selected")
    if not isinstance(selected, list) or not selected or not isinstance(selected[0], Mapping):
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_HARD_CASE_INVALID:{showcase_path or path}")
    video = Path(str(selected[0].get("selected_video_path") or ""))
    if video.is_symlink() or not video.is_file():
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_VIDEO_INVALID:{video}")
    delta = gate.get("delta_candidate_minus_baseline")
    if not isinstance(delta, Mapping):
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_DELTA_INVALID:{path}")
    return {
        "environment": str(payload.get("environment") or ""),
        "primitive": str(payload.get("primitive") or ""),
        "seed": int(payload.get("seed") or 0),
        "source_manifest": str(path),
        "source_manifest_sha256": _sha256(path),
        "source_showcase_manifest": str(showcase_path or path),
        "source_showcase_manifest_sha256": _sha256(showcase_path or path),
        "source_video_path": str(video),
        "sample_index": int(selected[0].get("sample_index") or 0),
        "baseline_video_mse": float(selected[0]["baseline_video_mse"]),
        "candidate_video_mse": float(selected[0]["candidate_video_mse"]),
        "psnr_delta": float(delta["psnr"]),
        "ssim_delta": float(delta["ssim"]),
        "mse_delta": float(delta["mse"]),
        "masked_mse_delta": float(delta["masked_mse"]),
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FourSceneBundleError(f"FOUR_SCENE_BUNDLE_MANIFEST_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "environment",
        "primitive",
        "seed",
        "sample_index",
        "psnr_delta",
        "ssim_delta",
        "mse_delta",
        "masked_mse_delta",
        "baseline_video_mse",
        "candidate_video_mse",
        "video_path",
        "source_manifest",
        "source_showcase_manifest",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def _markdown(records: list[dict[str, object]]) -> str:
    lines = [
        "# ACWM Four-Scene Visible Uplift Bundle",
        "",
        "All four rows passed the official 50-step quality gate. The displayed case for each environment was selected only by baseline error against GT.",
        "",
        "| Environment | Primitive | PSNR delta | SSIM delta | MSE delta | Masked-MSE delta | Video |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['environment']} | {record['primitive']} | {float(record['psnr_delta']):+.4f} | "
            f"{float(record['ssim_delta']):+.6f} | {float(record['mse_delta']):+.8f} | "
            f"{float(record['masked_mse_delta']):+.8f} | {Path(str(record['video_path'])).name} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--showcase-manifest", type=Path, action="append")
    args = parser.parse_args(argv)
    report = run_bundle(
        output_root=args.output_root,
        gate_manifests=args.gate_manifest,
        showcase_manifests=args.showcase_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
