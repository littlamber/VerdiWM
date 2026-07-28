#!/usr/bin/env python3
"""Export an audited ACWM fingerprint atlas and selector projections."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.fingerprint_atlas import build_fingerprint_atlas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-locality-residual", type=float, default=0.5)
    parser.add_argument("--path-calibration-policy", type=Path)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_fingerprint_atlas(
        campaign_path=args.campaign,
        campaign_output_root=args.campaign_output_root,
        output_root=args.output_root,
        maximum_locality_residual=args.maximum_locality_residual,
        path_calibration_policy=args.path_calibration_policy,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
