#!/usr/bin/env python3
"""Export a path-free public bundle for ACWM training-seed horizon evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path


class TrainingSeedHorizonPublicBundleError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingSeedHorizonPublicBundleError(f"PUBLIC_HORIZON_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> None:
    rows = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.sha256"
    ]
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_training_seed_horizon_public_bundle(
    *, summary_root: Path, profile_roots: Sequence[Path], triptych_roots: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_OUTPUT_EXISTS")
    summary_dir = Path(summary_root).resolve(strict=True)
    summary = _json(summary_dir / "summary.json")
    if (
        summary.get("artifact_type")
        != "verdiwm-acwm-training-seed-horizon-stability-summary"
        or summary.get("state") != "ready"
    ):
        raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_SUMMARY_INVALID")
    profiles: dict[int, dict[str, object]] = {}
    for root in profile_roots:
        profile = _json(Path(root).resolve(strict=True) / "horizon-effect-profile.json")
        seed = profile.get("training_seed")
        if (
            profile.get("artifact_type") != "wmloop-acwm-horizon-effect-profile"
            or profile.get("state") != "ready"
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed in profiles
        ):
            raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_PROFILE_INVALID")
        profiles[seed] = profile
    expected = {int(value) for value in summary.get("training_seeds", [])}
    if set(profiles) != expected:
        raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_PROFILE_COVERAGE_MISMATCH")

    videos: list[tuple[int, int, Path]] = []
    for root in triptych_roots:
        report = _json(Path(root).resolve(strict=True) / "horizon-triptych.json")
        source = report.get("source")
        match = re.search(r"-ts(\d+)-", str(source.get("candidate_manifest") if isinstance(source, Mapping) else ""))
        rows = report.get("paired_videos")
        if report.get("state") != "ready" or match is None or not isinstance(rows, list):
            raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_TRIPTYCH_INVALID")
        seed = int(match.group(1))
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("trajectory_index"), int):
                raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_TRIPTYCH_INVALID")
            path = Path(str(row.get("paired_video_path") or "")).resolve(strict=True)
            videos.append((seed, int(row["trajectory_index"]), path))
    if {seed for seed, _, _ in videos} != expected:
        raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_VIDEO_COVERAGE_MISMATCH")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700, parents=True)
    try:
        public_summary = json.loads(json.dumps(summary))
        (temporary / "summary.json").write_text(
            json.dumps(public_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(summary_dir / "horizon-summary.csv", temporary / "horizon-summary.csv")
        profile_dir = temporary / "profiles"
        video_dir = temporary / "videos"
        profile_dir.mkdir()
        video_dir.mkdir()
        for seed, profile in sorted(profiles.items()):
            compact = {
                "schema_version": 1,
                "artifact_type": "verdiwm-public-training-seed-horizon-profile",
                "state": "ready",
                "environment": profile["environment"],
                "primitive": profile["primitive"],
                "training_seed": seed,
                "horizons": profile["horizons"],
                "effect_classification": profile["effect_classification"],
                "horizon_effects": profile["horizon_effects"],
                "per_frame_effect": profile["per_frame_effect"],
                "effective_training_window": profile["transfer_prior"]["effective_training_window"],
                "claim_boundary": profile["claim_boundary"],
            }
            (profile_dir / f"training_seed_{seed}.json").write_text(
                json.dumps(compact, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        video_index = []
        for seed, trajectory, source in sorted(videos):
            name = f"training_seed_{seed}_trajectory_{trajectory:05d}_gt_baseline_ours.mp4"
            target = video_dir / name
            shutil.copy2(source, target)
            video_index.append(
                {
                    "training_seed": seed,
                    "trajectory_index": trajectory,
                    "layout": "GT|Baseline|Ours",
                    "path": f"videos/{name}",
                    "sha256": _sha256(target),
                }
            )
        (temporary / "video-index.json").write_text(
            json.dumps({"schema_version": 1, "state": "ready", "videos": video_index},
                       ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# ACWM training-seed horizon stability", "",
            "Selected checkpoints are evaluated on the same three held-out trajectories at 16/32/48 frames.",
            "", f"- Target: `{summary['environment']} + {summary['primitive']}`",
            f"- Verdict: `{summary['stability_verdict']}`",
            f"- Max-horizon strict passes: `{summary['max_horizon_pass_count']}/{summary['max_horizon_cell_count']}`",
            "", "The videos use the fixed `GT|Baseline|Ours` layout. Numeric claims are governed by",
            "`summary.json` and the per-seed profiles, not by cherry-picked video appearance.",
            "", "## Claim boundary", "", str(summary["claim_boundary"]), "",
        ]
        (temporary / "README.md").write_text("\n".join(lines), encoding="utf-8")
        _manifest(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_training_seed_horizon_public_bundle(destination)


def validate_training_seed_horizon_public_bundle(root: Path) -> dict[str, object]:
    bundle = Path(root).resolve(strict=True)
    lines = (bundle / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, separator, relative = line.partition("  ")
        path = bundle / relative
        if not separator or not path.is_file() or _sha256(path) != digest:
            raise TrainingSeedHorizonPublicBundleError(f"PUBLIC_HORIZON_SHA_MISMATCH:{relative}")
    summary = _json(bundle / "summary.json")
    videos = _json(bundle / "video-index.json").get("videos")
    if not isinstance(videos, list) or len(videos) < 1:
        raise TrainingSeedHorizonPublicBundleError("PUBLIC_HORIZON_VIDEO_INDEX_INVALID")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-training-seed-horizon-bundle-validation",
        "state": "ready",
        "manifest_entry_count": len(lines),
        "training_seed_count": len(summary.get("training_seeds", [])),
        "video_count": len(videos),
        "stability_verdict": summary.get("stability_verdict"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("export", "validate"), default="export")
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--profile-root", action="append", type=Path, default=[])
    parser.add_argument("--triptych-root", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_training_seed_horizon_public_bundle(args.output_root)
    else:
        if args.summary_root is None:
            parser.error("export requires --summary-root")
        result = export_training_seed_horizon_public_bundle(
            summary_root=args.summary_root, profile_roots=args.profile_root,
            triptych_roots=args.triptych_root, output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
