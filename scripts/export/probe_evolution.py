#!/usr/bin/env python3
"""Record a counterexample-driven diagnostic-probe successor campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.probe_evolution import build_probe_evolution_proposal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-fingerprint", type=Path, action="append", required=True)
    parser.add_argument("--successor-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_probe_evolution_proposal(
        failed_fingerprints=args.failed_fingerprint,
        successor_campaign=args.successor_campaign,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
