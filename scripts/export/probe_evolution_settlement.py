#!/usr/bin/env python3
"""Settle a counterexample-driven diagnostic-probe successor campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.probe_evolution import settle_probe_evolution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--successor-fingerprint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = settle_probe_evolution(
        proposal_path=args.proposal,
        successor_fingerprint=args.successor_fingerprint,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
