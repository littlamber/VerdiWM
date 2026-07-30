#!/usr/bin/env python3
"""Run the official cookbook DROID LeRobot forward-dynamics example.

This is a command-line version of the DROID section in
``cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb``.
It uses the checked-in ``DROIDLeRobotDataset`` sample directly and writes the
same action JSONL specs that the notebook builds.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from cosmos_framework.data.vfm.action.datasets import DROIDLeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--config-file",
        required=True,
        help="Local config that uses the bundled tokenizer and local Wan2.2 VAE.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "verdiwm",
        help="Fallback root for Hugging Face, XDG, and uv caches when their environment variables are unset.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-probe",
        choices=(
            "action_conditioning_scale",
            "action_dimension_anisotropy",
            "action_embedding_temporal_mix",
            "action_translation_scale",
        ),
        default="action_conditioning_scale",
        help="VerdiWM inference-only action probe materialized before forward dynamics.",
    )
    parser.add_argument(
        "--action-dose",
        type=float,
        default=0.0,
        help="VerdiWM relative action-conditioning dose in [-0.1, 0.1].",
    )
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "29631"))
    parser.add_argument("--skip-run", action="store_true", help="Only build input specs and initial frame.")
    return parser.parse_args()


def write_chunk_specs(args: argparse.Namespace, input_dir: Path) -> list[dict]:
    dataset = DROIDLeRobotDataset(
        root=str(args.dataset_root),
        chunk_length=args.chunk_length,
        mode="forward_dynamics",
    )
    chunk_starts = [args.start_index + i * args.chunk_length for i in range(args.num_chunks)]
    if chunk_starts[-1] >= len(dataset):
        raise ValueError(f"last chunk start {chunk_starts[-1]} exceeds dataset length {len(dataset)}")

    input_dir.mkdir(parents=True, exist_ok=True)
    initial_vision_path = input_dir / "robotics_droid_autoregressive_input_chunk_00.png"
    records: list[dict] = []

    for chunk_idx, sample_idx in enumerate(chunk_starts):
        sample = dataset[sample_idx]
        action = sample["action"].cpu()
        if int(action.shape[0]) != args.chunk_length:
            raise ValueError(f"unexpected action length {tuple(action.shape)} for chunk {chunk_idx}")

        source_action_path = input_dir / f"robotics_droid_action_chunk_{chunk_idx:02d}.source.json"
        source_action_path.write_text(json.dumps(action.tolist()))
        action_path = input_dir / f"robotics_droid_action_chunk_{chunk_idx:02d}.json"
        from wmloop.primitives.adapters.cosmos3_hooks import materialize_action_json

        hook_receipt = materialize_action_json(
            source=source_action_path,
            destination=action_path,
            mode=args.action_probe,
            dose=args.action_dose,
        )
        hook_receipt.update({"chunk_index": chunk_idx, "sample_index": sample_idx})
        (input_dir / f"robotics_droid_action_chunk_{chunk_idx:02d}.hook.json").write_text(
            json.dumps(hook_receipt, indent=2, sort_keys=True) + "\n"
        )

        if chunk_idx == 0:
            first_frame = sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(first_frame).save(initial_vision_path)
            vision_path = initial_vision_path
        else:
            vision_path = input_dir / f"robotics_droid_autoregressive_input_chunk_{chunk_idx:02d}.png"

        records.append(
            {
                "action_chunk_size": args.chunk_length,
                "action_path": str(action_path),
                "domain_name": "droid_lerobot",
                "fps": int(sample["conditioning_fps"]),
                "image_size": args.image_size,
                "view_point": sample["viewpoint"],
                "model_mode": "forward_dynamics",
                "name": f"robotics_action_cond_chunk_{chunk_idx:02d}",
                "prompt": sample["ai_caption"],
                "seed": args.seed,
                "vision_path": str(vision_path),
            }
        )

    plan_path = input_dir / "action_forward_dynamics_robotics_custom.jsonl"
    plan_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(f"loaded DROID samples from: {args.dataset_root}", flush=True)
    print(f"chunk starts: {chunk_starts}", flush=True)
    print(f"wrote initial frame: {initial_vision_path}", flush=True)
    print(f"wrote planned spec: {plan_path}", flush=True)
    return records


def run_chunks(args: argparse.Namespace, records: list[dict], input_dir: Path, output_dir: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cache_root = args.cache_root.expanduser().resolve()
    current_vision_path = Path(records[0]["vision_path"])
    actual_records: list[dict] = []

    for chunk_idx, base_record in enumerate(records):
        record = dict(base_record)
        record["vision_path"] = str(current_vision_path)
        records[chunk_idx]["vision_path"] = str(current_vision_path)

        chunk_input_path = input_dir / f"action_forward_dynamics_robotics_chunk_{chunk_idx:02d}.jsonl"
        chunk_input_path.write_text(json.dumps(record) + "\n")
        actual_records.append(record)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        env["MASTER_ADDR"] = env.get("MASTER_ADDR", socket.gethostbyname("localhost"))
        env["MASTER_PORT"] = str(int(args.master_port) + chunk_idx)
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault("HF_XET_CACHE", str(cache_root / "hf-xet"))
        env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
        env.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))

        print(f"running chunk {chunk_idx}: {record['name']}", flush=True)
        print(f"conditioning image: {current_vision_path}", flush=True)
        subprocess.run(
            [
                str(args.repo_root / ".venv" / "bin" / "python"),
                "-m",
                "cosmos_framework.scripts.inference",
                "--parallelism-preset=latency",
                "--no-guardrails",
                "-i",
                str(chunk_input_path),
                "-o",
                str(output_dir),
                "--checkpoint-path",
                args.checkpoint_path,
                "--config-file",
                args.config_file,
                "--video-save-quality",
                "8",
                "--image_size",
                str(args.image_size),
                "--seed",
                str(record["seed"]),
                "--benchmark",
            ],
            cwd=str(args.repo_root),
            env=env,
            check=True,
        )

        output_video = output_dir / record["name"] / "vision.mp4"
        if not output_video.exists():
            raise FileNotFoundError(f"missing generated video: {output_video}")

        if chunk_idx + 1 < len(records):
            next_vision_path = input_dir / f"robotics_droid_autoregressive_input_chunk_{chunk_idx + 1:02d}.png"
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output_video),
                    "-vf",
                    rf"select=eq(n\,{record['action_chunk_size']})",
                    "-frames:v",
                    "1",
                    str(next_vision_path),
                ],
                check=True,
            )
            current_vision_path = next_vision_path

    plan_path = input_dir / "action_forward_dynamics_robotics_custom.jsonl"
    plan_path.write_text("".join(json.dumps(record) + "\n" for record in actual_records))
    print(f"wrote autoregressive run spec: {plan_path}", flush=True)
    print(f"completed chunks: {[record['name'] for record in actual_records]}", flush=True)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.repo_root / "outputs" / f"official_droid_lerobot_fd_{stamp}"
    args.output_dir = args.output_dir.resolve()
    input_dir = args.output_dir / "inputs"

    records = write_chunk_specs(args, input_dir)
    if not args.skip_run:
        run_chunks(args, records, input_dir, args.output_dir)
    print(f"output dir: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
