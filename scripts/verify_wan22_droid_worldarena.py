#!/usr/bin/env python3
"""Fail-closed completeness verifier for frozen WAN2.2-DROID WorldArena evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("subject_consistency", "background_consistency", "photometric_smoothness")


def verify(receipt_paths: list[Path], *, required_metrics: tuple[str, ...] = REQUIRED_METRICS, expected_seeds: tuple[int, ...] = (4101, 4202, 4303)) -> dict[str, Any]:
    rows = []
    blockers = []
    seen_seeds: set[int] = set()
    seen_episodes: set[str] = set()
    for path in receipt_paths:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"RECEIPT_INVALID:{path}")
            continue
        if receipt.get("artifact_type") != "verdiwm-wan22-droid-worldarena-metrics-receipt":
            blockers.append(f"RECEIPT_TYPE_INVALID:{path}")
        if receipt.get("state") != "evaluated_partial" or int(receipt.get("returncode", -1)) != 0:
            blockers.append(f"EVALUATION_NOT_SUCCESSFUL:{path}")
        metrics = receipt.get("metrics")
        if not isinstance(metrics, dict) or any(metric not in metrics for metric in required_metrics):
            blockers.append(f"REQUIRED_METRICS_MISSING:{path}")
        video = receipt.get("video") if isinstance(receipt.get("video"), dict) else {}
        if video.get("generated_frames") != 150 or float(video.get("fps", 0)) != 5.0:
            blockers.append(f"VIDEO_CONTRACT_INVALID:{path}")
        run_root = path.parent / "training_receipt.json"
        if run_root.is_file():
            training = json.loads(run_root.read_text(encoding="utf-8"))
            seed = training.get("seed")
            episode = training.get("sample_id", "").split(":", 1)[0]
            if isinstance(seed, int):
                seen_seeds.add(seed)
            if episode:
                seen_episodes.add(episode)
            rows.append({"receipt": str(path), "seed": seed, "episode_id": episode})
        else:
            blockers.append(f"TRAINING_RECEIPT_MISSING:{path}")
    expected = set(expected_seeds)
    if seen_seeds != expected:
        blockers.append(f"SEED_SET_INVALID:expected={sorted(expected)}:observed={sorted(seen_seeds)}")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-worldarena-frozen-verifier-receipt",
        "state": "verified" if not blockers else "blocked",
        "receipt_count": len(receipt_paths),
        "expected_seed_count": len(expected),
        "observed_seeds": sorted(seen_seeds),
        "observed_episode_count": len(seen_episodes),
        "rows": rows,
        "blockers": sorted(set(blockers)),
        "claim_boundary": "Completeness verification does not establish a quality threshold; metric thresholds must be frozen separately before promotion.",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = verify([path.expanduser().resolve(strict=True) for path in args.receipts])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["state"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
