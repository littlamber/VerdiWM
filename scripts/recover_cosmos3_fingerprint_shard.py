#!/usr/bin/env python3
"""Recover verified completed records from an interrupted Cosmos3 shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmloop.experiments.cosmos3_shard_recovery import recover_cosmos3_shard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--doses", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = recover_cosmos3_shard(
        campaign_path=args.campaign,
        source_manifest_path=args.source_manifest,
        split_path=args.split_path,
        protocol=args.protocol,
        doses=args.doses,
        output_path=args.output,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
