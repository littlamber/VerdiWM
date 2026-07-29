#!/usr/bin/env python3
"""Fit a Cosmos3 target-local IRG chart from campaign shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.cosmos3_fingerprint import fit_cosmos3_fingerprint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = fit_cosmos3_fingerprint(
        campaign_path=args.campaign,
        shard_manifests=args.shard_manifest,
        split_path=args.split_path,
        protocol=args.protocol,
        output_root=args.output_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
