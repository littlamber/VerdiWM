#!/usr/bin/env python3
"""Index settled ACWM official-gate outcomes for selector supervision."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.effect_labels import build_effect_label_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment", action="append", required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_effect_label_index(
        reports_root=args.reports_root,
        output_root=args.output_root,
        expected_environments=args.environment,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
