#!/usr/bin/env python3
"""Compose ACWM selector projections from independently calibrated probe atlases."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.selector_projection_compose import compose_selector_projections


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-projection", type=Path, required=True)
    parser.add_argument("--primary-path", action="append", required=True)
    parser.add_argument("--extension-projection", type=Path, action="append", default=[])
    parser.add_argument("--extension-path", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if len(args.extension_projection) != len(args.extension_path):
        parser.error("each --extension-projection requires one --extension-path")
    manifest = compose_selector_projections(
        primary_projection=args.primary_projection,
        primary_path_order=args.primary_path,
        extension_projections=[
            (path, [probe_path])
            for path, probe_path in zip(args.extension_projection, args.extension_path, strict=True)
        ],
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
