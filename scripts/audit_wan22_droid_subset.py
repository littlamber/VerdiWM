"""Read-only audit for the processed DROID wrist subset used by WAN2.2 ACWM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, *, horizon_frames: int = 150) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("DROID_SUBSET_ROOT_INVALID")
    report: dict[str, object] = {
        "artifact_type": "verdiwm-wan22-droid-subset-audit",
        "root": str(root),
        "horizon_frames": horizon_frames,
        "splits": {},
    }
    for split in ("train", "val"):
        annotations = sorted((root / "annotation" / split).glob("*.json"))
        if not annotations:
            raise ValueError(f"DROID_ANNOTATION_MISSING:{split}")
        lengths: list[int] = []
        paired = 0
        latent_paired = 0
        missing: list[str] = []
        for path in annotations:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("split") != split:
                raise ValueError(f"DROID_SPLIT_MISMATCH:{path.name}")
            action = payload.get("action")
            proprio = payload.get("proprio")
            if not isinstance(action, list) or not action or len(action[0]) != 7:
                raise ValueError(f"DROID_ACTION_SCHEMA_INVALID:{path.name}")
            if not isinstance(proprio, list) or not proprio or len(proprio[0]) != 14:
                raise ValueError(f"DROID_PROPRIO_SCHEMA_INVALID:{path.name}")
            length = int(payload["video_length"])
            if length != len(action) or length != len(proprio):
                raise ValueError(f"DROID_TEMPORAL_LENGTH_MISMATCH:{path.name}")
            lengths.append(length)
            episode = path.stem
            video = root / "videos" / split / episode / "wrist.mp4"
            latent = root / "latent_videos" / split / episode / "wrist.pt"
            if video.is_file() and not video.is_symlink():
                paired += 1
            else:
                missing.append(f"video:{episode}")
            if latent.is_file() and not latent.is_symlink():
                latent_paired += 1
            else:
                missing.append(f"latent:{episode}")
        report["splits"][split] = {
            "episodes": len(annotations),
            "min_frames": min(lengths),
            "median_frames": sorted(lengths)[len(lengths) // 2],
            "max_frames": max(lengths),
            "episodes_at_target": sum(length >= horizon_frames for length in lengths),
            "video_pairs": paired,
            "latent_pairs": latent_paired,
            "missing_count": len(missing),
            "missing_sample": missing[:20],
            "annotation_manifest_sha256": _sha256_bytes(annotations),
        }
    return report


def _sha256_bytes(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon-frames", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, horizon_frames=args.horizon_frames)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
