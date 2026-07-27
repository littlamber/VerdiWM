#!/usr/bin/env python3
"""Generate real pour-water diagnostic probes from an h200 rollout manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.diagnose.pour_water_video_adapter import build_video_diagnostic_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_video_diagnostic_bundle(
        horizon_manifest=args.horizon_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
