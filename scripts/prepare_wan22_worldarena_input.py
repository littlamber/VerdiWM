#!/usr/bin/env python3
"""Materialize one WAN2.2 rollout into the pinned WorldArena frame layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _frames(video: Path) -> list[Any]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def _copy_frames(video: Path, destination: Path) -> int:
    import imageio.v2 as imageio

    frames = _frames(video)
    if len(frames) != 150:
        raise ValueError(f"WAN22_WORLD_ARENA_VIDEO_INVALID:{video}:{len(frames)}")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        imageio.imwrite(str(destination / f"frame_{index:05d}.png"), frame)
    return len(frames)


def _bind_evaluator_paths(config: dict[str, Any], *, worldarena_root: Path, asset_root: Path, sea_raft_config: Path) -> None:
    video_quality = worldarena_root / "video_quality"
    config["ckpt"] = {
        "action_following": str(asset_root / "clip" / "ViT-B-32.pt"),
        "background_consistency": {
            "clip": str(asset_root / "clip" / "ViT-B-32.pt"),
            "raft": str(asset_root / "raft" / "raft-things.pth"),
        },
        "photometric_smoothness": {
            "cfg": str(sea_raft_config),
            "model": str(asset_root / "sea-raft" / "Tartan-C-T-TSKH-spring540x960-M.pth"),
        },
        "motion_smoothness": {"model": str(asset_root / "vfimamba" / "model.pkl")},
        "subject_consistency": {
            "repo": str(asset_root / "dino" / "facebookresearch_dino"),
            "weight": str(asset_root / "dino" / "dino_vitbase16_pretrain.pth"),
            "model": "dino_vitb16",
            "raft": str(asset_root / "raft" / "raft-things.pth"),
        },
    }
    required = [
        video_quality / "evaluate.py",
        sea_raft_config,
        asset_root / "clip" / "ViT-B-32.pt",
        asset_root / "raft" / "raft-things.pth",
        asset_root / "vfimamba" / "model.pkl",
        asset_root / "sea-raft" / "Tartan-C-T-TSKH-spring540x960-M.pth",
        asset_root / "dino" / "facebookresearch_dino",
        asset_root / "dino" / "dino_vitbase16_pretrain.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("WAN22_WORLD_ARENA_RUNTIME_BINDING_MISSING:" + ",".join(missing))


def materialize(
    *, run_root: Path, output_root: Path, config_template: Path,
    worldarena_root: Path | None = None, asset_root: Path | None = None,
    sea_raft_config: Path | None = None, task_id: str = "droid", gid: str = "1",
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    input_info = json.loads((run_root / "worldarena_input.json").read_text(encoding="utf-8"))
    episode_id = str(input_info["episode_id"])
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise ValueError("WAN22_WORLD_ARENA_OUTPUT_UNBOUND")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_template = config_template.expanduser().resolve(strict=True)
    generated_dir = output_root / "generated_dataset" / task_id / episode_id / str(gid) / "video"
    gt_dir = output_root / "gt_dataset" / task_id / episode_id / "video"
    generated_count = _copy_frames(run_root / "generated_150f.mp4", generated_dir)
    gt_count = _copy_frames(run_root / "ground_truth_150f.mp4", gt_dir)
    first_frame = run_root / "first_frame.png"
    prompt_dir = output_root / "gt_dataset" / task_id / episode_id / "prompt"
    prompt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy2(first_frame, prompt_dir / "init_frame.png")
    (prompt_dir / "prompt.txt").write_text("DROID robot action-conditioned future prediction", encoding="utf-8")
    config = yaml.safe_load(config_template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("WAN22_WORLD_ARENA_CONFIG_INVALID")
    runtime_bindings = (worldarena_root, asset_root, sea_raft_config)
    if any(runtime_bindings) and not all(runtime_bindings):
        raise ValueError("WAN22_WORLD_ARENA_RUNTIME_BINDING_INCOMPLETE")
    if all(runtime_bindings):
        _bind_evaluator_paths(
            config,
            worldarena_root=worldarena_root.expanduser().resolve(strict=True),
            asset_root=asset_root.expanduser().resolve(strict=True),
            sea_raft_config=sea_raft_config.expanduser().resolve(strict=True),
        )
    data = config.setdefault("data", {})
    data["gt_path"] = str(output_root / "gt_dataset")
    data["val_base"] = str(output_root / "generated_dataset")
    action_data = config.setdefault("data_action_following", {})
    action_data["gt_path"] = str(output_root / "gt_dataset")
    action_data["val_base"] = str(output_root / "generated_dataset")
    config["save_path"] = str(output_root / "output")
    config["save_path_action_following"] = str(output_root / "output_action_following")
    config_path = output_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = [{
        "gt_path": str(gt_dir),
        "image": str(prompt_dir / "init_frame.png"),
        "prompt": ["DROID robot action-conditioned future prediction"],
    }]
    summary_path = output_root / "worldarena_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"state": "ready", "episode_id": episode_id, "generated_frames": generated_count, "ground_truth_frames": gt_count, "root": str(output_root), "config": str(config_path)}


def materialize_many(*, run_roots: list[Path], output_root: Path, config_template: Path, task_id: str = "droid", gids: list[str] | None = None) -> dict[str, Any]:
    """Materialize multiple generated gids for one episode without overwriting an existing root."""

    if not run_roots:
        raise ValueError("WAN22_WORLD_ARENA_RUN_ROOTS_EMPTY")
    gids = gids or [str(index + 1) for index in range(len(run_roots))]
    if len(gids) != len(run_roots) or len(set(gids)) != len(gids):
        raise ValueError("WAN22_WORLD_ARENA_GID_BINDING_INVALID")
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError("WAN22_WORLD_ARENA_OUTPUT_UNBOUND")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    infos = []
    for run_root, gid in zip(run_roots, gids):
        root = run_root.expanduser().resolve(strict=True)
        info = json.loads((root / "worldarena_input.json").read_text(encoding="utf-8"))
        infos.append((root, str(info["episode_id"]), gid))
    episode_ids = {episode for _root, episode, _gid in infos}
    if len(episode_ids) != 1:
        raise ValueError("WAN22_WORLD_ARENA_ACTION_GIDS_MUST_SHARE_EPISODE")
    episode_id = infos[0][1]
    for root, _episode, gid in infos:
        _copy_frames(root / "generated_150f.mp4", output_root / "generated_dataset" / task_id / episode_id / gid / "video")
    first_root = infos[0][0]
    gt_dir = output_root / "gt_dataset" / task_id / episode_id / "video"
    _copy_frames(first_root / "ground_truth_150f.mp4", gt_dir)
    prompt_dir = output_root / "gt_dataset" / task_id / episode_id / "prompt"
    prompt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy2(first_root / "first_frame.png", prompt_dir / "init_frame.png")
    (prompt_dir / "prompt.txt").write_text("DROID robot action-conditioned future prediction", encoding="utf-8")
    config_template = config_template.expanduser().resolve(strict=True)
    config = yaml.safe_load(config_template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("WAN22_WORLD_ARENA_CONFIG_INVALID")
    config.setdefault("data", {}).update({"gt_path": str(output_root / "gt_dataset"), "val_base": str(output_root / "generated_dataset")})
    config.setdefault("data_action_following", {}).update({"gt_path": str(output_root / "gt_dataset"), "val_base": str(output_root / "generated_dataset")})
    config["save_path"] = str(output_root / "output")
    config["save_path_action_following"] = str(output_root / "output_action_following")
    config_path = output_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = [{"gt_path": str(gt_dir), "image": str(prompt_dir / "init_frame.png"), "prompt": ["DROID robot action-conditioned future prediction"]}]
    (output_root / "worldarena_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"state": "ready", "episode_id": episode_id, "generated_gid_count": len(gids), "generated_frames_each": 150, "root": str(output_root), "config": str(config_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-template", type=Path, required=True)
    parser.add_argument("--worldarena-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--sea-raft-config", type=Path)
    parser.add_argument("--task-id", default="droid")
    parser.add_argument("--gid", nargs="+", default=None)
    args = parser.parse_args(argv)
    try:
        if len(args.run_root) == 1:
            result = materialize(
                run_root=args.run_root[0], output_root=args.output_root,
                config_template=args.config_template,
                worldarena_root=args.worldarena_root, asset_root=args.asset_root,
                sea_raft_config=args.sea_raft_config,
                task_id=args.task_id, gid=(args.gid or ["1"])[0],
            )
        else:
            result = materialize_many(run_roots=args.run_root, output_root=args.output_root, config_template=args.config_template, task_id=args.task_id, gids=args.gid)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
