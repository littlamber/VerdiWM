#!/usr/bin/env python3
"""Fail-closed completeness verifier for frozen WAN2.2-DROID WorldArena evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("subject_consistency", "background_consistency", "photometric_smoothness")


def verify(
    receipt_paths: list[Path], *,
    required_metrics: tuple[str, ...] = REQUIRED_METRICS,
    expected_seeds: tuple[int, ...] = (4101, 4202, 4303),
    expected_panel_size: int = 1,
) -> dict[str, Any]:
    if expected_panel_size < 1:
        raise ValueError("EXPECTED_VALIDATION_PANEL_SIZE_INVALID")
    rows = []
    blockers = []
    seen_seeds: set[int] = set()
    seen_episodes: set[str] = set()
    seen_pairs: set[tuple[int, str]] = set()
    panel_by_seed: dict[int, set[str]] = {}
    for path in receipt_paths:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"RECEIPT_INVALID:{path}")
            continue
        if receipt.get("artifact_type") != "verdiwm-wan22-droid-worldarena-metrics-receipt":
            blockers.append(f"RECEIPT_TYPE_INVALID:{path}")
        try:
            returncode = int(receipt.get("returncode", -1))
            fps = float(
                (receipt.get("video") if isinstance(receipt.get("video"), dict) else {}).get(
                    "fps", 0
                )
            )
        except (TypeError, ValueError):
            blockers.append(f"RECEIPT_NUMERIC_FIELDS_INVALID:{path}")
            returncode = -1
            fps = 0.0
        if receipt.get("state") != "evaluated_partial" or returncode != 0:
            blockers.append(f"EVALUATION_NOT_SUCCESSFUL:{path}")
        metrics = receipt.get("metrics")
        if not isinstance(metrics, dict) or any(metric not in metrics for metric in required_metrics):
            blockers.append(f"REQUIRED_METRICS_MISSING:{path}")
        video = receipt.get("video") if isinstance(receipt.get("video"), dict) else {}
        if video.get("generated_frames") != 150 or fps != 5.0:
            blockers.append(f"VIDEO_CONTRACT_INVALID:{path}")
        video_sample_id = str(video.get("sample_id") or "")
        video_episode_id = str(video.get("episode_id") or "")
        run_root = path.parent / "training_receipt.json"
        if not run_root.is_file():
            run_root = path.parent.parent / "training_receipt.json"
        if run_root.is_file():
            try:
                training = json.loads(run_root.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blockers.append(f"TRAINING_RECEIPT_INVALID:{run_root}")
                continue
            if not isinstance(training, dict):
                blockers.append(f"TRAINING_RECEIPT_INVALID:{run_root}")
                continue
            seed = training.get("seed")
            sample_id = video_sample_id or str(training.get("sample_id", ""))
            episode = video_episode_id or str(training.get("sample_id", "")).split(":", 1)[0]
            if isinstance(seed, int):
                seen_seeds.add(seed)
            if episode:
                seen_episodes.add(episode)
            if isinstance(seed, int):
                panel_by_seed.setdefault(seed, set()).add(sample_id or episode)
                pair = (seed, sample_id or episode)
                if pair in seen_pairs:
                    blockers.append(f"DUPLICATE_SEED_PANEL_MEMBER:{seed}:{sample_id or episode}")
                seen_pairs.add(pair)
            rows.append({"receipt": str(path), "seed": seed, "sample_id": sample_id, "episode_id": episode})
        else:
            blockers.append(f"TRAINING_RECEIPT_MISSING:{path}")
    expected = set(expected_seeds)
    if seen_seeds != expected:
        blockers.append(f"SEED_SET_INVALID:expected={sorted(expected)}:observed={sorted(seen_seeds)}")
    for seed in sorted(expected):
        observed_count = len(panel_by_seed.get(seed, set()))
        if observed_count != expected_panel_size:
            blockers.append(
                f"VALIDATION_PANEL_SIZE_INVALID:seed={seed}:expected={expected_panel_size}:observed={observed_count}"
            )
    panel_sets = [panel_by_seed[seed] for seed in sorted(expected) if seed in panel_by_seed]
    if panel_sets and any(panel != panel_sets[0] for panel in panel_sets[1:]):
        blockers.append("VALIDATION_PANEL_IDENTITY_MISMATCH_ACROSS_SEEDS")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-worldarena-frozen-verifier-receipt",
        "state": "verified" if not blockers else "blocked",
        "receipt_count": len(receipt_paths),
        "expected_seed_count": len(expected),
        "expected_panel_size": expected_panel_size,
        "observed_panel_sizes": {str(seed): len(panel_by_seed.get(seed, set())) for seed in sorted(seen_seeds)},
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
    parser.add_argument("--expected-panel-size", type=int, default=1)
    args = parser.parse_args(argv)
    result = verify(
        [path.expanduser().resolve(strict=True) for path in args.receipts],
        expected_panel_size=args.expected_panel_size,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["state"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
