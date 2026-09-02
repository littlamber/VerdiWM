#!/usr/bin/env python3
"""Run the pinned WorldArena evaluator against one WAN2.2-DROID run.

The evaluator checkout and its assets stay external to VerdiWM. This adapter
only validates the 150-frame handoff, launches the explicitly selected
dimensions, and writes a receipt with raw and per-video metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_run(run_root: Path) -> dict[str, Any]:
    required = ["generated_150f.mp4", "ground_truth_150f.mp4", "worldarena_input.json", "worldarena_summary.json"]
    missing = [name for name in required if not (run_root / name).is_file()]
    if missing:
        raise ValueError(f"WAN22_WORLD_ARENA_RUN_ARTIFACT_MISSING:{','.join(missing)}")
    reader = imageio.get_reader(str(run_root / "generated_150f.mp4"))
    generated_frames = reader.count_frames()
    metadata = reader.get_meta_data()
    reader.close()
    if generated_frames != 150 or float(metadata.get("fps", 0)) != 5.0:
        raise ValueError(f"WAN22_WORLD_ARENA_VIDEO_INVALID:{generated_frames}:{metadata.get('fps')}")
    return {"generated_frames": generated_frames, "fps": float(metadata["fps"]), "video_sha256": _sha256(run_root / "generated_150f.mp4")}


def _validate_assets(manifest_path: Path, asset_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for asset in manifest.get("assets", []):
        path = asset_root / str(asset["local_path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"WORLD_ARENA_ASSET_MISSING:{asset['id']}:{path}")
        digest = _sha256(path)
        size = path.stat().st_size
        if digest != str(asset["sha256"]) or size != int(asset["bytes"]):
            raise ValueError(f"WORLD_ARENA_ASSET_MISMATCH:{asset['id']}:{path}")
        row: dict[str, Any] = {"id": asset["id"], "path": str(path), "bytes": size, "sha256": digest}
        if asset.get("derived_torch_path"):
            derived = asset_root / str(asset["derived_torch_path"])
            if not derived.is_file() or _sha256(derived) != str(asset["derived_sha256"]):
                raise ValueError(f"WORLD_ARENA_DERIVED_ASSET_MISMATCH:{asset['id']}:{derived}")
            row["derived"] = {"path": str(derived), "bytes": derived.stat().st_size, "sha256": _sha256(derived)}
        rows.append(row)
    return {"manifest": str(manifest_path), "asset_root": str(asset_root), "assets": rows}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric, value in payload.items():
        if not isinstance(value, list) or len(value) < 2:
            metrics[metric] = {"raw": value}
            continue
        aggregate, rows = value[0], value[1]
        metrics[metric] = {"aggregate_reported": aggregate, "per_video": rows}
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worldarena-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True, help="WorldArena data root containing gt_dataset/generated_dataset")
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    worldarena_root = args.worldarena_root.expanduser().resolve()
    prepared_root = args.prepared_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    video_info = _validate_run(run_root)
    if bool(args.asset_manifest) != bool(args.asset_root):
        raise ValueError("WORLD_ARENA_ASSET_BINDING_INCOMPLETE")
    asset_info = _validate_assets(args.asset_manifest.expanduser().resolve(), args.asset_root.expanduser().resolve()) if args.asset_manifest else None
    if not (worldarena_root / "video_quality" / "evaluate.py").is_file():
        raise ValueError("WORLD_ARENA_EVALUATOR_ENTRYPOINT_MISSING")
    if not prepared_root.joinpath("gt_dataset").is_dir() or not prepared_root.joinpath("generated_dataset").is_dir():
        raise ValueError("WORLD_ARENA_PREPARED_DATA_ROOT_INVALID")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worldarena_root / "video_quality") + os.pathsep + env.get("PYTHONPATH", "")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    command = [
        str(args.runtime_python.expanduser().absolute()),
        str(worldarena_root / "video_quality" / "evaluate.py"),
        "--dimension", *args.dimensions,
        "--config", str(config),
        "--overwrite",
    ]
    completed = subprocess.run(command, cwd=str(worldarena_root / "video_quality"), env=env, capture_output=True, text=True)
    (run_root / "worldarena.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_root / "worldarena.stderr.log").write_text(completed.stderr, encoding="utf-8")

    output_root = config.parent / "output"
    result_path = output_root / "generated_results.json"
    parsed: dict[str, Any] | None = None
    if result_path.is_file():
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
        shutil.copy2(result_path, run_root / "worldarena_result.json")
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-worldarena-metrics-receipt",
        "state": "evaluated_partial" if completed.returncode == 0 else "evaluator_failed",
        "command": command,
        "returncode": completed.returncode,
        "cuda_visible_devices": args.cuda_visible_devices,
        "video": video_info,
        "assets": asset_info,
        "dimensions_requested": args.dimensions,
        "metrics": _extract_metrics(parsed or {}),
        "raw_result": str(run_root / "worldarena_result.json") if parsed is not None else None,
        "claim_boundary": "A single generated 150-frame rollout and partial dimensions do not establish a promoted 30-second method; frozen episode-disjoint multi-seed confirmation is still required.",
    }
    (run_root / "worldarena_metrics_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluator_receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-evaluator-receipt",
        "evaluator_id": "wan22-droid-worldarena-30s-v1",
        "state": "evaluated_partial" if completed.returncode == 0 else "evaluator_failed",
        "command_bound": True,
        "dimensions": args.dimensions,
        "metrics_receipt": str(run_root / "worldarena_metrics_receipt.json"),
        "returncode": completed.returncode,
        "claim_boundary": receipt["claim_boundary"],
    }
    (run_root / "evaluator_receipt.json").write_text(json.dumps(evaluator_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if completed.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
