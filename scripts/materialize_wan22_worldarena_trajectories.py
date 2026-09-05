#!/usr/bin/env python3
"""Extract official WorldArena gripper trajectories for a prepared dataset.

The trajectory metric is defined in image space as ``(T, 2, 2)``.  This
command deliberately delegates detection and post-processing to the pinned
WorldArena implementation; it never projects robot state or fabricates
coordinates.  Both GT and generated videos are processed with the same
detector and a provenance receipt is written next to the prepared dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_info(path: Path, expected_frames: int) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"WORLD_ARENA_TRAJECTORY_MISSING:{path}")
    trajectory = np.load(path)
    if trajectory.shape != (expected_frames, 2, 2):
        raise ValueError(
            f"WORLD_ARENA_TRAJECTORY_SHAPE_INVALID:{path}:{trajectory.shape}"
        )
    return {
        "path": str(path),
        "shape": list(trajectory.shape),
        "dtype": str(trajectory.dtype),
        "sha256": _sha256(path),
        "valid_points": int(np.sum(np.all(trajectory != -1, axis=2))),
    }


def _source_revision(worldarena_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(worldarena_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def materialize(
    *,
    prepared_root: Path,
    worldarena_root: Path,
    sam3_model_root: Path,
    task_id: str = "droid",
    force_reprocess: bool = False,
    expected_frames: int = 150,
) -> dict[str, Any]:
    prepared_root = prepared_root.expanduser().resolve(strict=True)
    worldarena_root = worldarena_root.expanduser().resolve(strict=True)
    sam3_model_root = sam3_model_root.expanduser().resolve(strict=True)
    checkpoint = sam3_model_root / "sam3.pt"
    bpe = sam3_model_root / "bpe_simple_vocab_16e6.txt.gz"
    missing = [str(path) for path in (checkpoint, bpe) if not path.is_file()]
    if missing:
        raise ValueError("SAM3_CHECKPOINT_FILES_MISSING:" + ",".join(missing))

    video_quality = worldarena_root / "video_quality"
    if not (video_quality / "processing" / "detection_tracking.py").is_file():
        raise ValueError("WORLD_ARENA_DETECTION_TRACKING_MISSING")
    # WorldArena keeps the pinned detector as a vendored checkout below
    # ``video_quality/processing/sam3`` rather than installing it globally.
    # Add both the processing package and its vendored project root so the
    # official module imports resolve in a clean runtime.
    sys.path.insert(0, str(video_quality))
    sys.path.insert(0, str(video_quality / "processing"))
    sys.path.insert(0, str(video_quality / "processing" / "sam3"))
    from processing.detection_tracking import GripperDetector, process_video_with_tracking

    detector = GripperDetector(model_path=str(sam3_model_root))
    generated_root = prepared_root / "generated_dataset" / task_id
    gt_root = prepared_root / "gt_dataset" / task_id
    if not generated_root.is_dir() or not gt_root.is_dir():
        raise ValueError("WORLD_ARENA_PREPARED_DATA_ROOT_INVALID")

    rows: list[dict[str, Any]] = []
    for episode in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        gt_video = episode / "video"
        if not gt_video.is_dir():
            raise ValueError(f"WORLD_ARENA_GT_VIDEO_MISSING:{gt_video}")
        process_video_with_tracking(
            input_path=str(gt_video),
            output_path=str(episode),
            detector=detector,
            data_type="gt",
            force_reprocess=force_reprocess,
        )
        rows.append(
            {
                "kind": "gt",
                "episode_id": episode.name,
                "trajectory": _trajectory_info(episode / "traj" / "traj.npy", expected_frames),
            }
        )
        generated_episode = generated_root / episode.name
        if not generated_episode.is_dir():
            raise ValueError(f"WORLD_ARENA_GENERATED_EPISODE_MISSING:{generated_episode}")
        for gid_root in sorted(path for path in generated_episode.iterdir() if path.is_dir()):
            video = gid_root / "video"
            if not video.is_dir():
                raise ValueError(f"WORLD_ARENA_GENERATED_VIDEO_MISSING:{video}")
            process_video_with_tracking(
                input_path=str(video),
                output_path=str(generated_episode),
                detector=detector,
                gid=gid_root.name,
                data_type="val",
                force_reprocess=force_reprocess,
            )
            rows.append(
                {
                    "kind": "generated",
                    "episode_id": episode.name,
                    "gid": gid_root.name,
                    "trajectory": _trajectory_info(
                        gid_root / "traj" / "traj.npy", expected_frames
                    ),
                }
            )

    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-worldarena-trajectory-receipt",
        "state": "ready",
        "detector": {
            "name": "WorldArena official detection_tracking.py",
            "worldarena_root": str(worldarena_root),
            "worldarena_revision": _source_revision(worldarena_root),
            "sam3_model_root": str(sam3_model_root),
            "sam3_checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            },
            "bpe": {
                "path": str(bpe),
                "bytes": bpe.stat().st_size,
                "sha256": _sha256(bpe),
            },
        },
        "prepared_root": str(prepared_root),
        "task_id": task_id,
        "expected_frames": expected_frames,
        "rows": rows,
        "claim_boundary": "Trajectories are official image-space detector outputs; metric quality and model promotion remain separate decisions.",
    }
    destination = prepared_root / "trajectory_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--worldarena-root", type=Path, required=True)
    parser.add_argument("--sam3-model-root", type=Path, required=True)
    parser.add_argument("--task-id", default="droid")
    parser.add_argument("--force-reprocess", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = materialize(
            prepared_root=args.prepared_root,
            worldarena_root=args.worldarena_root,
            sam3_model_root=args.sam3_model_root,
            task_id=args.task_id,
            force_reprocess=args.force_reprocess,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
