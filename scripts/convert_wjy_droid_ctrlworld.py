#!/usr/bin/env python3
"""Create a small, immutable WAN2.2-DROID dataset from WJY's DROID export.

The source export contains three encoded camera streams and the original DROID
state/action fields.  This converter keeps the wrist camera (camera ``2``),
projects the recorded Cartesian action to the adapter's 7D action contract,
and projects observation joint/cartesian/gripper values to the 14D proprio
contract.  It never synthesizes image-space trajectories; those remain a
separate WorldArena preprocessing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ConversionError(ValueError):
    """The WJY source cannot satisfy the WAN2.2-DROID data contract."""


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_matrix(value: object, width: int, field: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConversionError(f"DROID_FIELD_INVALID:{field}")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ConversionError(f"DROID_FIELD_SHAPE_INVALID:{field}")
        normalized = [float(item) for item in row]
        if not all(math.isfinite(item) for item in normalized):
            raise ConversionError(f"DROID_FIELD_NONFINITE:{field}")
        rows.append(normalized)
    return rows


def _scalar_series(value: object, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConversionError(f"DROID_FIELD_INVALID:{field}")
    result: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 1:
                raise ConversionError(f"DROID_FIELD_SHAPE_INVALID:{field}")
            item = item[0]
        number = float(item)
        if not math.isfinite(number):
            raise ConversionError(f"DROID_FIELD_NONFINITE:{field}")
        result.append(number)
    return result


def _load_source_annotation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"DROID_ANNOTATION_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise ConversionError(f"DROID_ANNOTATION_INVALID:{path}")
    split = str(payload.get("split", ""))
    if split not in {"train", "val"}:
        raise ConversionError(f"DROID_SPLIT_INVALID:{path}")
    return payload


def _source_video_and_latent(root: Path, payload: Mapping[str, Any], camera: int) -> tuple[Path, Path]:
    episode_id = str(payload.get("episode_id") or "").strip()
    videos = payload.get("videos")
    latents = payload.get("latent_videos")
    try:
        video_rel = str(videos[camera]["video_path"])
        latent_rel = str(latents[camera]["latent_video_path"])
    except (IndexError, KeyError, TypeError) as exc:
        raise ConversionError(f"DROID_CAMERA_ASSET_METADATA_MISSING:{episode_id}") from exc
    video = (root / video_rel).resolve()
    latent = (root / latent_rel).resolve()
    if not video.is_file() or video.is_symlink():
        raise ConversionError(f"DROID_VIDEO_MISSING:{episode_id}:{video}")
    if not latent.is_file() or latent.is_symlink():
        raise ConversionError(f"DROID_LATENT_MISSING:{episode_id}:{latent}")
    return video, latent


def _project(payload: Mapping[str, Any], *, source_root: Path, camera: int) -> tuple[dict[str, Any], Path, Path]:
    episode_id = str(payload.get("episode_id") or "").strip()
    if not episode_id:
        raise ConversionError("DROID_EPISODE_ID_MISSING")
    if int(payload.get("frame_stride", 0)) != 3 or int(payload.get("processed_fps", 0)) != 5:
        raise ConversionError(f"DROID_SAMPLING_CONTRACT_INVALID:{episode_id}")
    actions_cart = _finite_matrix(payload.get("action.cartesian_position"), 6, "action.cartesian_position")
    actions_gripper = _scalar_series(payload.get("action.gripper_position"), "action.gripper_position")
    obs_joint = _finite_matrix(payload.get("observation.state.joint_position"), 7, "observation.state.joint_position")
    obs_cart = _finite_matrix(payload.get("observation.state.cartesian_position"), 6, "observation.state.cartesian_position")
    obs_gripper = _scalar_series(payload.get("observation.state.gripper_position"), "observation.state.gripper_position")
    raw_length = len(actions_cart)
    if any(len(series) != raw_length for series in (actions_gripper, obs_joint, obs_cart, obs_gripper)):
        raise ConversionError(f"DROID_STATE_ACTION_ALIGNMENT_INVALID:{episode_id}")
    indices = list(range(0, raw_length, 3))
    if len(indices) < 152:
        raise ConversionError(f"DROID_EPISODE_TOO_SHORT:{episode_id}:{len(indices)}")
    action = [[*actions_cart[index], actions_gripper[index]] for index in indices]
    proprio = [
        [*obs_joint[index], *obs_cart[index], obs_gripper[index]]
        for index in indices
    ]
    video, latent = _source_video_and_latent(source_root, payload, camera)
    projected = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-converted-annotation",
        "episode_id": episode_id,
        "split": str(payload["split"]),
        "source_shard": str(payload.get("source_shard", "")),
        "source_record_index": int(payload.get("source_record_index", -1)),
        "source_record_sha256": str(payload.get("source_record_sha256", "")),
        "source_annotation_sha256": _sha256(source_root / "annotation" / str(payload["split"]) / f"{episode_id}.json"),
        "source_camera": camera,
        "source_camera_field": (
            "steps/observation/exterior_image_1_left",
            "steps/observation/exterior_image_2_left",
            "steps/observation/wrist_image_left",
        )[camera],
        "source_fps": 15,
        "processed_fps": 5,
        "frame_stride": 3,
        "raw_frame_count": raw_length,
        "video_length": len(action),
        "action_dim": 7,
        "proprio_dim": 14,
        "action": action,
        "proprio": proprio,
        "instruction": str((payload.get("texts") or ["DROID robot action-conditioned future prediction"])[0]),
        "video_path": str(video.relative_to(source_root)),
        "latent_path": str(latent.relative_to(source_root)),
        "projection": {
            "action": "[action.cartesian_position(6), action.gripper_position(1)]",
            "proprio": "[observation.state.joint_position(7), observation.state.cartesian_position(6), observation.state.gripper_position(1)]",
            "sampling": "raw arrays sampled at indices 0,3,6,...; no padding or interpolation",
        },
        "claim_boundary": "This is a conditioning-data conversion. It does not provide WorldArena image-space gripper trajectories.",
    }
    return projected, video, latent


def _copy_episode(
    *,
    source_root: Path,
    output_root: Path,
    payload: Mapping[str, Any],
    camera: int,
) -> dict[str, Any]:
    projected, video, latent = _project(payload, source_root=source_root, camera=camera)
    split = str(projected["split"])
    episode_id = str(projected["episode_id"])
    episode_root = output_root / "videos" / split / episode_id
    episode_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _materialize_file(video, episode_root / "wrist.mp4")
    latent_root = output_root / "latent_videos" / split / episode_id
    latent_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _materialize_file(latent, latent_root / "wrist.pt")
    annotation = dict(projected)
    annotation["video_path"] = f"videos/{split}/{episode_id}/wrist.mp4"
    annotation["latent_path"] = f"latent_videos/{split}/{episode_id}/wrist.pt"
    annotation_path = output_root / "annotation" / split / f"{episode_id}.json"
    annotation_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    annotation_path.write_bytes(_canonical(annotation))
    return {
        "sample_id": f"{episode_id}:000000:150",
        "episode_id": episode_id,
        "split": split,
        "start_frame": 0,
        "horizon_frames": 150,
        "source_frame_count": int(projected["video_length"]),
        "rollout_source_frames_required": 152,
        "video_path": annotation["video_path"],
        "latent_path": annotation["latent_path"],
        "annotation_path": str(annotation_path.relative_to(output_root)),
        "action_dim": 7,
        "proprio_dim": 14,
        "fps": 5,
        "instruction": annotation["instruction"],
        "latent_source": "raw_video_wan22_vae",
        "source_record_sha256": annotation["source_record_sha256"],
    }


def _materialize_file(source: Path, destination: Path) -> None:
    """Materialize without duplicating large latent blobs when possible."""

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _manifest(split: str, root: Path, records: list[dict[str, Any]], source_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(_canonical(records)).hexdigest()
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-sample-manifest",
        "split": split,
        "data_root": str(root),
        "source_data_root": str(source_root),
        "horizon_frames": 150,
        "rollout_source_frames_required": 152,
        "window_stride": 30,
        "record_count": len(records),
        "episode_count": len({row["episode_id"] for row in records}),
        "excluded_short_or_invalid_episodes": 0,
        "precomputed_latent_compatible_records": 0,
        "raw_video_reencode_records": len(records),
        "manifest_sha256": digest,
        "records": records,
        "claim_boundary": "Conditioning manifest only; WorldArena trajectory_accuracy requires separately extracted image-space traj/traj.npy.",
    }


def convert(*, source_root: Path, output_root: Path, train_episodes: int, val_episodes: int, camera: int = 2) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir())):
        raise ConversionError("DROID_OUTPUT_ROOT_MUST_BE_EMPTY")
    if train_episodes < 1 or val_episodes < 1 or camera not in {0, 1, 2}:
        raise ConversionError("DROID_CONVERSION_ARGUMENT_INVALID")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    try:
        records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
        requested = {"train": train_episodes, "val": val_episodes}
        for split in ("train", "val"):
            annotations = sorted((source_root / "annotation" / split).glob("*.json"))
            for path in annotations:
                if len(records_by_split[split]) >= requested[split]:
                    break
                payload = _load_source_annotation(path)
                if payload.get("split") != split:
                    continue
                try:
                    record = _copy_episode(source_root=source_root, output_root=staging, payload=payload, camera=camera)
                except ConversionError:
                    continue
                records_by_split[split].append(record)
            if len(records_by_split[split]) != requested[split]:
                raise ConversionError(f"DROID_NOT_ENOUGH_VALID_EPISODES:{split}:{len(records_by_split[split])}:{requested[split]}")
        for split, records in records_by_split.items():
            manifest_path = staging / f"{split}.json"
            manifest_path.write_bytes(_canonical(_manifest(split, output_root, records, source_root)))
        summary = {
            "schema_version": 1,
            "artifact_type": "verdiwm-wan22-droid-wjy-conversion-receipt",
            "state": "ready",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "camera": camera,
            "camera_field": "steps/observation/wrist_image_left",
            "train_manifest": str(output_root / "train.json"),
            "validation_manifest": str(output_root / "val.json"),
            "train_episode_count": len(records_by_split["train"]),
            "validation_episode_count": len(records_by_split["val"]),
            "trajectory_accuracy": {
                "state": "not_materialized",
                "reason": "DROID conversion has no image-space gripper trajectory; SAM3/official WorldArena preprocessing remains required.",
            },
            "claim_boundary": "This receipt proves only deterministic conditioning-data conversion and provenance.",
        }
        (staging / "conversion_receipt.json").write_bytes(_canonical(summary))
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return {**summary, "output_root": str(output_root)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--val-episodes", type=int, default=3)
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = convert(
            source_root=args.source_root,
            output_root=args.output_root,
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            camera=args.camera,
        )
    except ConversionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
