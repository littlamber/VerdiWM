#!/usr/bin/env python3
"""Render baseline-only ROI zooms for an auditable ACWM video bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence


class BaselineRoiShowcaseError(RuntimeError):
    """The source bundle cannot be rendered under the baseline-only contract."""


def run_bundle(
    *,
    source_bundle: Path,
    output_root: Path,
    runtime_python: Path,
    error_scale: float = 10.0,
    roi_fraction: float = 0.5,
) -> dict[str, object]:
    source_bundle = Path(source_bundle).resolve()
    destination = Path(output_root).resolve()
    runtime_python = Path(runtime_python).resolve()
    if destination.exists() or destination.is_symlink():
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_OUTPUT_EXISTS")
    if error_scale <= 0:
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_SCALE_INVALID")
    if not 0.1 <= roi_fraction <= 1.0:
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_FRACTION_INVALID")
    manifest_path = source_bundle / "manifest.json"
    if not manifest_path.is_file() or runtime_python.is_symlink() or not runtime_python.is_file():
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_SOURCE_INVALID")
    source = _load_json(manifest_path)
    records = source.get("records")
    if source.get("state") != "ready" or not isinstance(records, list) or not records:
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_SOURCE_NOT_READY")
    if any(not isinstance(record, Mapping) for record in records):
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_RECORD_INVALID")
    selection_rule = str(source.get("selection_rule") or "")
    if "baseline-only" not in selection_rule:
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_SELECTION_NOT_BASELINE_ONLY")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        videos = temporary / "videos"
        videos.mkdir(mode=0o700, parents=True)
        bundled: list[dict[str, object]] = []
        for record in records:
            source_video = Path(str(record.get("video_path") or ""))
            if source_video.is_symlink() or not source_video.is_file():
                raise BaselineRoiShowcaseError(f"BASELINE_ROI_SHOWCASE_VIDEO_INVALID:{source_video}")
            target = videos / f"{source_video.stem}_baseline_roi_x{_format_number(error_scale)}.mp4"
            render = _render(
                source_video,
                target,
                runtime_python=runtime_python,
                error_scale=error_scale,
                roi_fraction=roi_fraction,
            )
            probe = _probe_video(
                target,
                expected_frames=int(render["frame_count"]),
                expected_fps=float(render["fps"]),
            )
            bundled.append(
                {
                    "environment": str(record.get("environment") or ""),
                    "primitive": str(record.get("primitive") or ""),
                    "seed": int(record.get("seed") or 0),
                    "sample_index": int(record.get("sample_index") or 0),
                    "psnr_delta": float(record.get("psnr_delta") or 0.0),
                    "ssim_delta": float(record.get("ssim_delta") or 0.0),
                    "mse_delta": float(record.get("mse_delta") or 0.0),
                    "masked_mse_delta": float(record.get("masked_mse_delta") or 0.0),
                    "baseline_video_mse": float(record.get("baseline_video_mse") or 0.0),
                    "candidate_video_mse": float(record.get("candidate_video_mse") or 0.0),
                    "roi": render["roi"],
                    "frame_count": render["frame_count"],
                    "fps": render["fps"],
                    "source_video_path": str(source_video),
                    "source_video_sha256": _sha256(source_video),
                    "video_path": str(destination / "videos" / target.name),
                    "video_sha256": _sha256(target),
                    "video_size_bytes": target.stat().st_size,
                    "video_probe": probe,
                }
            )
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-baseline-roi-showcase-bundle",
            "state": "ready",
            "source_bundle": str(source_bundle),
            "source_bundle_sha256": _sha256(manifest_path),
            "scene_count": len(bundled),
            "records": bundled,
            "error_scale": error_scale,
            "roi_fraction": roi_fraction,
            "layout": "full GT/Baseline/Ours/errors above; baseline-only max-error ROI zooms below",
            "selection_rule": (
                "The source sample and fixed ROI are selected only from GT-vs-baseline error; "
                "candidate pixels and candidate gains are not used for selection."
            ),
            "candidate_used_for_sample_selection": False,
            "candidate_used_for_roi_selection": False,
            "claim_boundary": (
                "Qualitative visibility aid only. Numeric claims remain governed by the source frozen official gates; "
                "ROI zoom and labeled error amplification are not additional metrics."
            ),
        }
        _write_json(temporary / "manifest.json", report)
        (temporary / "README.md").write_text(_markdown(report), encoding="utf-8")
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _render(
    source: Path,
    target: Path,
    *,
    runtime_python: Path,
    error_scale: float,
    roi_fraction: float,
) -> dict[str, object]:
    spec = target.parent / f".{target.stem}.spec.json"
    result = target.parent / f".{target.stem}.result.json"
    _write_json(
        spec,
        {
            "source": str(source),
            "target": str(target),
            "error_scale": error_scale,
            "roi_fraction": roi_fraction,
        },
    )
    helper = r'''
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def max_sum_window(error, crop_h, crop_w):
    integral = np.pad(error, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sums = (
        integral[crop_h:, crop_w:]
        - integral[:-crop_h, crop_w:]
        - integral[crop_h:, :-crop_w]
        + integral[:-crop_h, :-crop_w]
    )
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(x), int(y), int(crop_w), int(crop_h)


def title(panel, label, font, rectangle=None):
    image = Image.fromarray(panel)
    if rectangle is not None:
        draw = ImageDraw.Draw(image)
        x, y, w, h = rectangle
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(255, 220, 0), width=3)
    panel = np.asarray(image)
    header = np.full((30, panel.shape[1], 3), 20, dtype=np.uint8)
    image = Image.fromarray(np.concatenate([header, panel], axis=0))
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), label, font=font)
    x = max(4, (panel.shape[1] - (box[2] - box[0])) // 2)
    draw.text((x, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(spec["source"])
target = Path(spec["target"])
scale = float(spec["error_scale"])
fraction = float(spec["roi_fraction"])
reader = imageio.get_reader(source)
metadata = reader.get_meta_data()
frames = [np.asarray(frame) for frame in reader]
reader.close()
if not frames:
    raise RuntimeError("BASELINE_ROI_SHOWCASE_EMPTY_VIDEO")

triples = []
baseline_errors = []
for frame in frames:
    if frame.shape[0] > 40:
        frame = frame[30:]
    height, width = frame.shape[:2]
    panel_width = width // 3
    gt = frame[:, :panel_width, :3]
    baseline = frame[:, panel_width:2 * panel_width, :3]
    ours = frame[:, 2 * panel_width:3 * panel_width, :3]
    triples.append((gt, baseline, ours))
    baseline_errors.append(np.abs(gt.astype(np.float32) - baseline.astype(np.float32)).mean(axis=2))

height, panel_width = triples[0][0].shape[:2]
crop_h = min(height, max(48, int(round(height * fraction))))
crop_w = min(panel_width, max(48, int(round(panel_width * fraction))))
mean_error = np.mean(np.stack(baseline_errors, axis=0), axis=0)
roi = max_sum_window(mean_error, crop_h, crop_w)
x, y, crop_w, crop_h = roi
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
output = []
for gt, baseline, ours in triples:
    base_error = np.clip(np.abs(gt.astype(np.float32) - baseline.astype(np.float32)) * scale, 0, 255).astype(np.uint8)
    ours_error = np.clip(np.abs(gt.astype(np.float32) - ours.astype(np.float32)) * scale, 0, 255).astype(np.uint8)
    top_panels = [gt, baseline, ours, base_error, ours_error]
    top_labels = ["GT full", "Baseline full", "Ours full", f"Baseline error x{scale:g}", f"Ours error x{scale:g}"]
    top = np.concatenate(
        [title(panel, label, font, roi if index < 3 else None) for index, (panel, label) in enumerate(zip(top_panels, top_labels))],
        axis=1,
    )
    roi_panels = []
    for panel in top_panels:
        crop = panel[y:y + crop_h, x:x + crop_w]
        roi_panels.append(np.asarray(Image.fromarray(crop).resize((panel_width, height), Image.Resampling.NEAREST)))
    roi_labels = ["GT ROI", "Baseline ROI", "Ours ROI", f"Baseline ROI error x{scale:g}", f"Ours ROI error x{scale:g}"]
    bottom = np.concatenate([title(panel, label, font) for panel, label in zip(roi_panels, roi_labels)], axis=1)
    output.append(np.concatenate([top, bottom], axis=0))

fps = float(metadata.get("fps") or 10.0)
imageio.mimsave(target, output, fps=fps, macro_block_size=2)
Path(sys.argv[2]).write_text(
    json.dumps({"frame_count": len(output), "fps": fps, "roi": {"x": x, "y": y, "width": crop_w, "height": crop_h}}),
    encoding="utf-8",
)
'''
    completed = subprocess.run(
        [str(runtime_python), "-c", helper, str(spec), str(result)],
        check=False,
        capture_output=True,
        text=True,
    )
    spec.unlink(missing_ok=True)
    if completed.returncode != 0:
        result.unlink(missing_ok=True)
        raise BaselineRoiShowcaseError(f"BASELINE_ROI_SHOWCASE_RENDER_FAILED:{completed.stderr[-1500:]}")
    rendered = _load_json(result)
    result.unlink(missing_ok=True)
    return rendered


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaselineRoiShowcaseError(f"BASELINE_ROI_SHOWCASE_JSON_INVALID:{path}")
    return value


def _probe_video(path: Path, *, expected_frames: int, expected_fps: float) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BaselineRoiShowcaseError(f"BASELINE_ROI_SHOWCASE_PROBE_FAILED:{completed.stderr[-1000:]}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_PROBE_INVALID")
    stream = streams[0]
    try:
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
        frame_count = int(stream["nb_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(stream["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_PROBE_INVALID") from error
    if (
        stream.get("codec_name") != "h264"
        or frame_count != expected_frames
        or abs(fps - expected_fps) > 1e-6
        or width <= 0
        or height <= 0
        or duration <= 0
    ):
        raise BaselineRoiShowcaseError("BASELINE_ROI_SHOWCASE_PROBE_CONTRACT_FAILED")
    return {
        "codec_name": "h264",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "contract_pass": True,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# ACWM Baseline-Only ROI Showcase",
        "",
        "The source sample and zoom region are selected only from GT-vs-baseline error. Candidate output is never used for selection.",
        "",
        "| Environment | Primitive | PSNR delta | SSIM delta | ROI | Video |",
        "|---|---|---:|---:|---|---|",
    ]
    for record in report["records"]:  # type: ignore[index]
        roi = record["roi"]
        lines.append(
            f"| {record['environment']} | {record['primitive']} | {float(record['psnr_delta']):+.4f} | "
            f"{float(record['ssim_delta']):+.6f} | `{roi}` | {Path(str(record['video_path'])).name} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable)),
    )
    parser.add_argument("--error-scale", type=float, default=10.0)
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = run_bundle(
        source_bundle=args.source_bundle,
        output_root=args.output_root,
        runtime_python=args.runtime_python,
        error_scale=args.error_scale,
        roi_fraction=args.roi_fraction,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
