"""Small, deterministic paired ACWM/ground-truth visualization artifacts.

The visualization is an observation aid, never a quality metric.  Keeping it
in the control plane makes the requirement model-independent: any runner that
publishes a generated video and its paired ground truth can use the same
receipt and retention contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PairedVisualizationError(ValueError):
    """The paired videos cannot produce a trustworthy comparison artifact."""


def create_paired_visualization(
    *,
    generated_video: Path,
    ground_truth_video: Path,
    output_root: Path,
    metadata: Mapping[str, Any] | None = None,
    fps: float = 5.0,
) -> dict[str, object]:
    """Write a side-by-side MP4, contact sheet, and machine-readable manifest."""

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    generated = Path(generated_video).expanduser().resolve(strict=True)
    ground_truth = Path(ground_truth_video).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
            raise PairedVisualizationError("PAIRED_VISUALIZATION_OUTPUT_UNBOUND")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if generated == ground_truth:
        raise PairedVisualizationError("PAIRED_VISUALIZATION_INPUTS_IDENTICAL")

    generated_frames = _read_video(imageio, generated)
    ground_truth_frames = _read_video(imageio, ground_truth)
    if generated_frames.shape != ground_truth_frames.shape:
        raise PairedVisualizationError(
            "PAIRED_VISUALIZATION_VIDEO_SHAPE_MISMATCH:"
            f"{tuple(generated_frames.shape)}:{tuple(ground_truth_frames.shape)}"
        )
    if generated_frames.shape[0] < 1 or float(fps) <= 0:
        raise PairedVisualizationError("PAIRED_VISUALIZATION_VIDEO_INVALID")

    separator_frame = np.full(
        (generated_frames.shape[1], 4, generated_frames.shape[3]),
        255,
        dtype=np.uint8,
    )
    separator = np.repeat(separator_frame[None, ...], generated_frames.shape[0], axis=0)
    side_by_side = np.concatenate(
        (generated_frames, separator, ground_truth_frames), axis=2
    )
    comparison_path = destination / "acwm_gt_comparison.mp4"
    imageio.mimsave(
        str(comparison_path),
        list(side_by_side),
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
    )

    indexes = sorted({0, generated_frames.shape[0] // 2, generated_frames.shape[0] - 1})
    rows = []
    for index in indexes:
        pair = np.concatenate(
            (generated_frames[index], separator_frame, ground_truth_frames[index]), axis=1
        )
        image = Image.fromarray(pair)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
        draw.text((6, 4), f"ACWM   |   GT   frame={index}", fill=(255, 255, 255))
        rows.append(np.asarray(image))
    contact_sheet = np.concatenate(rows, axis=0)
    contact_path = destination / "acwm_gt_contact_sheet.png"
    Image.fromarray(contact_sheet).save(contact_path)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-paired-acwm-gt-visualization",
        "state": "ready",
        "inputs": {
            "generated_video": str(generated),
            "ground_truth_video": str(ground_truth),
            "generated_sha256": _sha256(generated),
            "ground_truth_sha256": _sha256(ground_truth),
            "frames": int(generated_frames.shape[0]),
            "fps": float(fps),
            "frame_shape": list(generated_frames.shape[1:]),
        },
        "artifacts": {
            "comparison_video": str(comparison_path),
            "comparison_video_sha256": _sha256(comparison_path),
            "contact_sheet": str(contact_path),
            "contact_sheet_sha256": _sha256(contact_path),
        },
        "metadata": dict(metadata or {}),
        "claim_boundary": (
            "This side-by-side artifact is for human inspection only. It is not a "
            "metric, a promotion decision, or evidence that ACWM is superior to GT."
        ),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _read_video(imageio: Any, path: Path):
    import numpy as np

    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise PairedVisualizationError(f"PAIRED_VISUALIZATION_FRAME_INVALID:{path}")
            frames.append(array)
    finally:
        reader.close()
    if not frames:
        raise PairedVisualizationError(f"PAIRED_VISUALIZATION_VIDEO_EMPTY:{path}")
    return np.stack(frames, axis=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
