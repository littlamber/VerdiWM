#!/usr/bin/env python3
"""Evaluate paired Ctrl-World task receipts into a target-local IRG chart."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.ctrl_world_fingerprint import evaluate_ctrl_world_fingerprint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--receipt-index", type=Path, required=True)
    parser.add_argument("--heldout-split", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = evaluate_ctrl_world_fingerprint(
        campaign_path=args.campaign,
        receipt_index_path=args.receipt_index,
        heldout_split_path=args.heldout_split,
        protocol=args.protocol,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
