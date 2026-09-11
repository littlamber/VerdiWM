#!/usr/bin/env python3
"""Execute the RoboCoach WAN2.2 trainer from a VerdiWM model-run manifest.

The wrapper keeps the external checkout read-only and makes the model-run
manifest the only interface consumed by the adapter.  It performs no work at
import time; callers decide when a GPU lease has been granted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


class Wan22TrainError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Wan22TrainError("WAN22_MODEL_RUN_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise Wan22TrainError("WAN22_MODEL_RUN_MANIFEST_INVALID")
    return value


def run(manifest_path: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise Wan22TrainError("WAN22_MODEL_RUN_SOURCE_INVALID")
    source_root = Path(str(source.get("source_root") or source.get("model_root"))).expanduser().resolve()
    model_root = Path(str(source.get("model_root"))).expanduser().resolve()
    runtime = Path(str(manifest.get("runtime", {}).get("python"))).expanduser().absolute()
    script = source_root / "scripts" / "training" / "train_world_model.py"
    if not source_root.is_dir() or not script.is_file():
        raise Wan22TrainError("WAN22_TRAIN_ENTRYPOINT_MISSING")
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise Wan22TrainError("WAN22_TRAIN_RUNTIME_INVALID")
    bindings = source.get("asset_bindings")
    if not isinstance(bindings, dict):
        raise Wan22TrainError("WAN22_TRAIN_ASSET_BINDINGS_INVALID")
    checkpoint = bindings.get("--checkpoint") or model_root
    output = Path(str(manifest.get("output", {}).get("root"))).expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    # RoboCoach keeps its optimizer and checkpoint layout in the YAML config.
    # Deployments can select a project-specific config without changing the
    # adapter by setting VERDIWM_WAN22_TRAIN_CONFIG.
    config = os.environ.get("VERDIWM_WAN22_TRAIN_CONFIG", "configs/training/sa_wm_h5f3.yaml")
    command = [str(runtime), str(script), "--config", config, "--init_from", str(checkpoint)]
    if os.environ.get("VERDIWM_EXECUTE_WAN22", "0") != "1":
        raise Wan22TrainError(
            "WAN22_TRAIN_EXECUTION_DISABLED:set VERDIWM_EXECUTE_WAN22=1 after a GPU lease"
        )
    completed = subprocess.run(command, cwd=str(source_root), capture_output=True, text=True, check=False)
    (output / "train.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output / "train.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise Wan22TrainError(f"WAN22_TRAIN_FAILED:returncode={completed.returncode}")
    receipt_path = Path(str(manifest["training"]["checkpoint_receipt"])).expanduser().resolve()
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-checkpoint-receipt",
        "state": "complete",
        "model_run_id": manifest.get("model_run_id"),
        "source_root": str(source_root),
        "returncode": completed.returncode,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
