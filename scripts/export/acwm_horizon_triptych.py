#!/usr/bin/env python3
"""Export aligned GT|Baseline|Ours videos from paired ACWM horizon probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class HorizonTriptychError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HorizonTriptychError(f"HORIZON_TRIPTYCH_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_videos(manifest: Mapping[str, object]) -> dict[int, Path]:
    records = manifest.get("trajectory_results")
    if not isinstance(records, list):
        return {}
    videos: dict[int, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        index = record.get("trajectory_index")
        video = record.get("rollout_video_path")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(video, str):
            path = Path(video).resolve()
            if path.is_file() and not path.is_symlink():
                videos[index] = path
    return videos


def _font() -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), 17) if path.is_file() else ImageFont.load_default()


def export_horizon_triptychs(
    *,
    baseline_manifest: Path,
    candidate_manifest: Path,
    output_root: Path,
    fps: int = 10,
    max_gt_panel_mse: float = 1e-4,
) -> dict[str, object]:
    baseline_path = baseline_manifest.resolve(strict=True)
    candidate_path = candidate_manifest.resolve(strict=True)
    destination = output_root.resolve()
    if destination.exists() or destination.is_symlink():
        raise HorizonTriptychError(f"HORIZON_TRIPTYCH_OUTPUT_EXISTS:{destination}")
    if fps < 1 or max_gt_panel_mse < 0:
        raise HorizonTriptychError("HORIZON_TRIPTYCH_ARGUMENT_INVALID")
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("state") != "ready":
            raise HorizonTriptychError(f"HORIZON_TRIPTYCH_SOURCE_NOT_READY:{name}")
    for field in ("environment", "split"):
        if baseline.get(field) != candidate.get(field):
            raise HorizonTriptychError(f"HORIZON_TRIPTYCH_SOURCE_MISMATCH:{field}")
    baseline_videos = _trajectory_videos(baseline)
    candidate_videos = _trajectory_videos(candidate)
    trajectory_indices = sorted(set(baseline_videos) & set(candidate_videos))
    if not trajectory_indices:
        raise HorizonTriptychError("HORIZON_TRIPTYCH_NO_PAIRED_TRAJECTORIES")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700, parents=True)
    rows: list[dict[str, object]] = []
    try:
        video_root = temporary / "videos"
        video_root.mkdir(mode=0o700)
        font = _font()
        for trajectory_index in trajectory_indices:
            baseline_video = baseline_videos[trajectory_index]
            candidate_video = candidate_videos[trajectory_index]
            baseline_frames = imageio.mimread(baseline_video)
            candidate_frames = imageio.mimread(candidate_video)
            frame_count = min(len(baseline_frames), len(candidate_frames))
            if frame_count < 1:
                continue
            output_frames: list[np.ndarray] = []
            gt_squared_error = 0.0
            gt_value_count = 0
            panel_width = 0
            panel_height = 0
            for baseline_frame, candidate_frame in zip(
                baseline_frames[:frame_count], candidate_frames[:frame_count]
            ):
                baseline_array = np.asarray(baseline_frame)[:, :, :3]
                candidate_array = np.asarray(candidate_frame)[:, :, :3]
                height = min(baseline_array.shape[0], candidate_array.shape[0])
                width = min(baseline_array.shape[1], candidate_array.shape[1])
                half = width // 2
                if height < 1 or half < 1:
                    raise HorizonTriptychError("HORIZON_TRIPTYCH_FRAME_SHAPE_INVALID")
                baseline_gt = baseline_array[:height, :half]
                candidate_gt = candidate_array[:height, :half]
                baseline_prediction = baseline_array[:height, half : half * 2]
                candidate_prediction = candidate_array[:height, half : half * 2]
                difference = (
                    baseline_gt.astype(np.float32) - candidate_gt.astype(np.float32)
                ) / 255.0
                gt_squared_error += float(np.square(difference).sum())
                gt_value_count += int(difference.size)
                triptych = np.concatenate(
                    [baseline_gt, baseline_prediction, candidate_prediction], axis=1
                )
                header_height = 30
                canvas = np.zeros((height + header_height, triptych.shape[1], 3), dtype=np.uint8)
                canvas[:header_height] = 20
                canvas[header_height:] = triptych
                image = Image.fromarray(canvas)
                draw = ImageDraw.Draw(image)
                for panel_index, label in enumerate(("GT", "Baseline", "Ours")):
                    box = draw.textbbox((0, 0), label, font=font)
                    text_width = box[2] - box[0]
                    x = panel_index * half + max(0, (half - text_width) // 2)
                    draw.text((x, 5), label, fill=(255, 255, 255), font=font)
                output_frames.append(np.asarray(image))
                panel_width = half
                panel_height = height
            gt_panel_mse = gt_squared_error / max(1, gt_value_count)
            if gt_panel_mse > max_gt_panel_mse:
                raise HorizonTriptychError(
                    f"HORIZON_TRIPTYCH_GT_ALIGNMENT_FAILED:{trajectory_index}:{gt_panel_mse}"
                )
            filename = f"trajectory_{trajectory_index:05d}_gt_baseline_ours.mp4"
            temporary_video = video_root / filename
            imageio.mimsave(temporary_video, output_frames, fps=fps, macro_block_size=1)
            rows.append(
                {
                    "trajectory_index": trajectory_index,
                    "layout": "labeled_GT|baseline_prediction|ours_prediction",
                    "frame_count": frame_count,
                    "fps": fps,
                    "panel_width": panel_width,
                    "panel_height": panel_height,
                    "gt_panel_mse_between_sources": gt_panel_mse,
                    "baseline_video_path": str(baseline_video),
                    "candidate_video_path": str(candidate_video),
                    "paired_video_path": str(destination / "videos" / filename),
                }
            )
        if not rows:
            raise HorizonTriptychError("HORIZON_TRIPTYCH_NO_OUTPUTS")
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-horizon-triptych",
            "state": "ready",
            "environment": baseline.get("environment"),
            "split": baseline.get("split"),
            "layout": "labeled_GT|baseline_prediction|ours_prediction",
            "paired_video_count": len(rows),
            "claim_boundary": (
                "Visualization only. Numeric long-horizon claims remain governed by the paired "
                "horizon effect profile and frozen metric manifests."
            ),
            "source": {
                "baseline_manifest": str(baseline_path),
                "baseline_manifest_sha256": _sha256(baseline_path),
                "candidate_manifest": str(candidate_path),
                "candidate_manifest_sha256": _sha256(candidate_path),
            },
            "paired_videos": rows,
        }
        (temporary / "horizon-triptych.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-horizon-triptych-manifest",
            "state": "ready",
            "environment": baseline.get("environment"),
            "paired_video_count": len(rows),
            "report_path": str(destination / "horizon-triptych.json"),
            "video_root": str(destination / "videos"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-gt-panel-mse", type=float, default=1e-4)
    args = parser.parse_args(argv)
    manifest = export_horizon_triptychs(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        output_root=args.output_root,
        fps=args.fps,
        max_gt_panel_mse=args.max_gt_panel_mse,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
