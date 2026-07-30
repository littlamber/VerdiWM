#!/usr/bin/env python3
"""Audit a Cosmos3 DROID split using input data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--current-split", type=Path, required=True)
    parser.add_argument("--retired-split", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=16)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _rows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (split, row)
        for split in ("dev", "accept")
        for row in payload.get(split, [])
    ]


def _window(sample_index: int, chunk_length: int) -> set[int]:
    return set(range(sample_index, sample_index + chunk_length + 1))


def _audit_passed(checks: dict[str, bool]) -> bool:
    positive_checks = [name for name in checks if name != "outcomes_inspected_before_freeze"]
    return all(checks[name] for name in positive_checks) and not checks[
        "outcomes_inspected_before_freeze"
    ]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    current_path = args.current_split.resolve()
    retired_paths = [path.resolve() for path in args.retired_split]

    # Import after argument parsing so --help remains usable outside Cosmos3.
    import numpy as np
    from cosmos_framework.data.vfm.action.datasets import DROIDLeRobotDataset

    current = json.loads(current_path.read_text(encoding="utf-8"))
    retired_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in retired_paths]
    current_rows = _rows(current)
    retired_rows = [row for payload in retired_payloads for _, row in _rows(payload)]
    current_windows = [
        _window(int(row["sample_index"]), args.chunk_length) for _, row in current_rows
    ]
    retired_windows = [
        _window(int(row["sample_index"]), args.chunk_length) for row in retired_rows
    ]

    dataset = DROIDLeRobotDataset(
        root=str(dataset_root),
        chunk_length=args.chunk_length,
        mode="forward_dynamics",
    )
    measurements: list[dict[str, Any]] = []
    for split, row in current_rows:
        sample_index = int(row["sample_index"])
        sample = dataset[sample_index]
        action = np.ascontiguousarray(sample["action"].detach().cpu().numpy())
        video = np.ascontiguousarray(sample["video"].detach().cpu().numpy())
        measurements.append(
            {
                "split": split,
                "sample_index": sample_index,
                "seed": int(row["seed"]),
                "action_shape": list(action.shape),
                "action_dtype": str(action.dtype),
                "action_sha256": _sha256_bytes(action.tobytes()),
                "action_finite": bool(np.isfinite(action).all()),
                "video_shape": list(video.shape),
                "video_dtype": str(video.dtype),
                "video_sha256": _sha256_bytes(video.tobytes()),
                "video_frame_count": int(video.shape[1]),
            }
        )

    checks = {
        "selected_windows_disjoint_from_retired_splits": all(
            window.isdisjoint(retired) for window in current_windows for retired in retired_windows
        ),
        "selected_windows_are_pairwise_disjoint": all(
            left.isdisjoint(right)
            for index, left in enumerate(current_windows)
            for right in current_windows[index + 1 :]
        ),
        "all_actions_are_finite_16x10": all(
            row["action_finite"] and row["action_shape"] == [args.chunk_length, 10]
            for row in measurements
        ),
        "all_windows_decode_17_frames": all(
            row["video_frame_count"] == args.chunk_length + 1 for row in measurements
        ),
        "probe_uses_input_actions_only": True,
        "outcomes_inspected_before_freeze": False,
    }
    passed = _audit_passed(checks)

    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-input-window-freeze-audit",
        "state": "passed" if passed else "failed",
        "claim_scope": "Input-only split freeze audit; no model outcomes were loaded or inspected.",
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "dataset_length": len(dataset),
        "chunk_length": args.chunk_length,
        "current_split": {
            "path": str(current_path),
            "sha256": _sha256_file(current_path),
            "split_id": current.get("split_id"),
        },
        "retired_splits": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "split_id": payload.get("split_id"),
            }
            for path, payload in zip(retired_paths, retired_payloads, strict=True)
        ],
        "checks": checks,
        "measurements": measurements,
        "selection_provenance": {
            "selection_inputs": ["sample indices", "action tensors", "video frame availability"],
            "model_outputs_used": False,
            "evaluation_metrics_used": False,
        },
    }
    _write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"state": payload["state"], "output": str(args.output.resolve())}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
