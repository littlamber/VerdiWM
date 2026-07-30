#!/usr/bin/env python3
"""Audit ACWM source-effect evidence before certificate recovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.source_effect_audit import build_source_effect_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-label-index", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--additional-receipt-root",
        type=Path,
        action="append",
        default=[],
        help="Additional directory containing archived official-gate receipt directories.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--minimum-independent-seeds", type=int, default=3)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_source_effect_audit(
        effect_label_index_path=args.effect_label_index,
        reports_root=args.reports_root,
        additional_receipt_roots=args.additional_receipt_root,
        output_root=args.output_root,
        primitive=args.primitive,
        minimum_independent_seeds=args.minimum_independent_seeds,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
