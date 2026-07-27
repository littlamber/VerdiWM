#!/usr/bin/env python3
"""Add baseline-blind amplified error panels to a validated ACWM video bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


class ErrorShowcaseError(RuntimeError):
    """A source bundle is not eligible for qualitative error visualization."""


def run_bundle(*, source_bundle: Path, output_root: Path, runtime_python: Path, error_scale: float = 5.0) -> dict[str, object]:
    source_bundle = Path(source_bundle).resolve()
    destination = Path(output_root).resolve()
    runtime_python = Path(runtime_python).resolve()
    if destination.exists() or destination.is_symlink():
        raise ErrorShowcaseError("ERROR_SHOWCASE_OUTPUT_EXISTS")
    if error_scale <= 0:
        raise ErrorShowcaseError("ERROR_SHOWCASE_SCALE_INVALID")
    manifest_path = source_bundle / "manifest.json"
    if not manifest_path.is_file() or runtime_python.is_symlink() or not runtime_python.is_file():
        raise ErrorShowcaseError("ERROR_SHOWCASE_SOURCE_INVALID")
    source = _load_json(manifest_path)
    records = source.get("records")
    if source.get("state") != "ready" or not isinstance(records, list) or not records:
        raise ErrorShowcaseError("ERROR_SHOWCASE_SOURCE_NOT_READY")
    if any(not isinstance(record, Mapping) for record in records):
        raise ErrorShowcaseError("ERROR_SHOWCASE_RECORD_INVALID")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        videos = temporary / "videos"
        videos.mkdir(mode=0o700, parents=True)
        bundled: list[dict[str, object]] = []
        for record in records:
            source_video = Path(str(record.get("video_path") or ""))
            if source_video.is_symlink() or not source_video.is_file():
                raise ErrorShowcaseError(f"ERROR_SHOWCASE_VIDEO_INVALID:{source_video}")
            target = videos / f"{source_video.stem}_error_x{_format_scale(error_scale)}.mp4"
            _render(source_video, target, runtime_python=runtime_python, error_scale=error_scale)
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
                    "source_video_path": str(source_video),
                    "source_video_sha256": _sha256(source_video),
                    "video_path": str(destination / "videos" / target.name),
                    "video_sha256": _sha256(target),
                    "video_size_bytes": target.stat().st_size,
                }
            )
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-error-showcase-bundle",
            "state": "ready",
            "source_bundle": str(source_bundle),
            "source_bundle_sha256": _sha256(manifest_path),
            "scene_count": len(bundled),
            "records": bundled,
            "error_scale": error_scale,
            "layout": "GT|Baseline|Ours|Baseline error x scale|Ours error x scale",
            "selection_rule": "Inherited from the source bundle; source cases were selected by baseline-only error.",
            "claim_boundary": (
                "Qualitative error visualization only. Numeric claims remain governed by the source official gates; "
                "amplification is not an additional metric or a candidate-dependent sample selector."
            ),
        }
        (temporary / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "README.md").write_text(_markdown(report), encoding="utf-8")
        os.replace(temporary, destination)
        return report
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _render(source: Path, target: Path, *, runtime_python: Path, error_scale: float) -> None:
    spec = target.parent / f".{target.stem}.spec.json"
    result = target.parent / f".{target.stem}.result.json"
    spec.write_text(
        json.dumps({"source": str(source), "target": str(target), "error_scale": error_scale}), encoding="utf-8"
    )
    helper = r'''
import json
import sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(spec["source"])
target = Path(spec["target"])
scale = float(spec["error_scale"])
frames = imageio.mimread(source)
if not frames:
    raise RuntimeError("ERROR_SHOWCASE_EMPTY_VIDEO")
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
output = []
for frame in frames:
    array = np.asarray(frame)
    # The formal GT|baseline|ours source already has a 30px title strip.
    if array.shape[0] > 40:
        array = array[30:]
    height, width = array.shape[:2]
    panel_width = width // 3
    gt = array[:, :panel_width, :3]
    baseline = array[:, panel_width:2 * panel_width, :3]
    ours = array[:, 2 * panel_width:3 * panel_width, :3]
    base_error = np.clip(np.abs(gt.astype(np.float32) - baseline.astype(np.float32)) * scale, 0, 255).astype(np.uint8)
    ours_error = np.clip(np.abs(gt.astype(np.float32) - ours.astype(np.float32)) * scale, 0, 255).astype(np.uint8)
    panels = [gt, baseline, ours, base_error, ours_error]
    canvas = np.concatenate(panels, axis=1)
    header = np.zeros((30, canvas.shape[1], 3), dtype=np.uint8)
    header[:] = 20
    image = Image.fromarray(np.concatenate([header, canvas], axis=0))
    draw = ImageDraw.Draw(image)
    labels = ("GT", "Official baseline", "Primitive model", f"Baseline error x{scale:g}", f"Primitive error x{scale:g}")
    for index, label in enumerate(labels):
        box = draw.textbbox((0, 0), label, font=font)
        x = index * panel_width + max(4, (panel_width - (box[2] - box[0])) // 2)
        draw.text((x, 6), label, fill=(255, 255, 255), font=font)
    output.append(np.asarray(image))
imageio.mimsave(target, output, fps=10)
Path(sys.argv[2]).write_text(json.dumps({"frame_count": len(output)}), encoding="utf-8")
'''
    completed = subprocess.run([str(runtime_python), "-c", helper, str(spec), str(result)], check=False, capture_output=True, text=True)
    spec.unlink(missing_ok=True)
    result.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise ErrorShowcaseError(f"ERROR_SHOWCASE_RENDER_FAILED:{completed.stderr[-1000:]}")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ErrorShowcaseError("ERROR_SHOWCASE_JSON_INVALID")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_scale(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# ACWM Error Showcase Bundle",
        "",
        "Layout: `GT | Official baseline | Primitive model | Baseline error x5 | Primitive error x5`.",
        "The source cases were selected using baseline-only error; error amplification is qualitative only.",
        "",
        "| Environment | Primitive | PSNR delta | SSIM delta | Video |",
        "|---|---|---:|---:|---|",
    ]
    for record in report["records"]:  # type: ignore[index]
        lines.append(
            f"| {record['environment']} | {record['primitive']} | {float(record['psnr_delta']):+.4f} | "
            f"{float(record['ssim_delta']):+.6f} | {Path(str(record['video_path'])).name} |"
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
    parser.add_argument("--error-scale", type=float, default=5.0)
    args = parser.parse_args(argv)
    report = run_bundle(
        source_bundle=args.source_bundle,
        output_root=args.output_root,
        runtime_python=args.runtime_python,
        error_scale=args.error_scale,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
