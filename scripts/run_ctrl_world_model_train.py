#!/usr/bin/env python3
"""Translate a VerdiWM model-run manifest into Ctrl-World training.

This wrapper is the only process allowed to invoke the external training
script. It writes a receipt in the fresh campaign root and never edits the
external source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class TrainWrapperError(RuntimeError):
    pass


def run(manifest_path: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    source = manifest["source"]
    assets = source["asset_bindings"]
    model_root = Path(str(source["model_root"])).resolve()
    runtime = Path(str(manifest["runtime"]["python"])).expanduser()
    external = model_root / "scripts" / "train_wm.py"
    if not external.is_file() or not runtime.is_file() or not os.access(runtime.resolve(), os.X_OK):
        raise TrainWrapperError("CTRL_WORLD_TRAIN_RUNTIME_MISSING")
    output = Path(str(manifest["output"]["root"])).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700, exist_ok=True)
    scale = manifest["training"]["scale_plan"]
    updates = scale["updates"]
    parallelism = scale["parallelism"]
    steps = int(updates["planned_steps"])
    if steps < 1:
        raise TrainWrapperError("CTRL_WORLD_TRAIN_STEPS_INVALID")
    bindings = {str(key): str(value) for key, value in assets.items()}
    required = ("--dataset_root_path", "--dataset_meta_info_path", "--ckpt_path", "--svd_model_path", "--clip_model_path")
    missing = [key for key in required if key not in bindings]
    if missing:
        raise TrainWrapperError("CTRL_WORLD_TRAIN_ASSET_MISSING:" + ",".join(missing))
    command = [
        str(runtime), str(external),
        "--dataset_root_path", bindings["--dataset_root_path"],
        "--dataset_meta_info_path", bindings["--dataset_meta_info_path"],
        "--ckpt_path", bindings["--ckpt_path"],
        "--svd_model_path", bindings["--svd_model_path"],
        "--clip_model_path", bindings["--clip_model_path"],
        "--output_dir", str(checkpoint_dir),
        "--max_train_steps", str(steps),
        "--checkpointing_steps", str(steps),
        "--validation_steps", str(steps + 1),
        "--disable_validation",
        "--train_batch_size", str(int(parallelism["batch_size"])),
        "--gradient_accumulation_steps", str(int(parallelism["gradient_accumulation"])),
    ]
    env = dict(os.environ)
    env["CTRL_WORLD_ENABLE_TRACKING"] = "0"
    env["WANDB_MODE"] = "offline"
    env["SWANLAB_MODE"] = "offline"
    completed = subprocess.run(command, cwd=str(model_root), env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = str(completed.stderr or "")[-4000:].replace("\x00", "")
        raise TrainWrapperError(
            f"CTRL_WORLD_TRAIN_FAILED:returncode={completed.returncode}:stderr={stderr_tail}"
        )
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoints:
        raise TrainWrapperError("CTRL_WORLD_CHECKPOINT_NOT_WRITTEN")
    checkpoint = checkpoints[-1].resolve()
    receipt_path = Path(str(manifest["training"]["checkpoint_receipt"])).resolve()
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-checkpoint-receipt",
        "state": "complete",
        "model_run_id": manifest["model_run_id"],
        "checkpoint_path": str(checkpoint),
        "planned_steps": steps,
        "returncode": completed.returncode,
    }
    _write(receipt_path, receipt)
    return receipt


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainWrapperError("CTRL_WORLD_MODEL_RUN_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise TrainWrapperError("CTRL_WORLD_MODEL_RUN_MANIFEST_INVALID")
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
