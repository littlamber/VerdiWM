#!/usr/bin/env python3
"""Merge complete Ctrl-World fingerprint receipt shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.ctrl_world_receipt_merge import merge_ctrl_world_receipt_indexes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--heldout-split", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--receipt-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = merge_ctrl_world_receipt_indexes(
        campaign_path=args.campaign,
        heldout_split_path=args.heldout_split,
        protocol=args.protocol,
        receipt_index_paths=args.receipt_index,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
