import json
from pathlib import Path

from scripts.convert_wjy_droid_ctrlworld import convert


def _annotation(root: Path, split: str, episode: str, length: int = 156) -> None:
    payload = {
        "episode_id": episode,
        "split": split,
        "frame_stride": 3,
        "processed_fps": 5,
        "raw_frame_count": length * 3,
        "action.cartesian_position": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * (length * 3),
        "action.gripper_position": [[0.0]] * (length * 3),
        "observation.state.joint_position": [[0.0] * 7] * (length * 3),
        "observation.state.cartesian_position": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * (length * 3),
        "observation.state.gripper_position": [[0.0]] * (length * 3),
        "texts": ["test instruction"],
        "videos": [{"video_path": f"videos/{split}/{episode}/{camera}.mp4"} for camera in range(3)],
        "latent_videos": [{"latent_video_path": f"latent_videos/{split}/{episode}/{camera}.pt"} for camera in range(3)],
    }
    path = root / "annotation" / split / f"{episode}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    for camera in range(3):
        video = root / "videos" / split / episode / f"{camera}.mp4"
        latent = root / "latent_videos" / split / episode / f"{camera}.pt"
        video.parent.mkdir(parents=True, exist_ok=True)
        latent.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        latent.write_bytes(b"latent")


def test_conversion_projects_droid_fields_without_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _annotation(source, "train", "train-episode")
    _annotation(source, "val", "val-episode")
    output = tmp_path / "output"

    receipt = convert(
        source_root=source,
        output_root=output,
        train_episodes=1,
        val_episodes=1,
    )

    assert receipt["state"] == "ready"
    assert receipt["trajectory_accuracy"]["state"] == "not_materialized"
    train = json.loads((output / "train.json").read_text(encoding="utf-8"))
    row = train["records"][0]
    assert (output / row["video_path"]).is_file()
    assert (output / row["latent_path"]).is_file()
    annotation = json.loads((output / row["annotation_path"]).read_text(encoding="utf-8"))
    assert len(annotation["action"]) == 156
    assert len(annotation["action"][0]) == 7
    assert len(annotation["proprio"][0]) == 14
