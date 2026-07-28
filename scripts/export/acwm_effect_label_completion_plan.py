#!/usr/bin/env python3
"""Build the minimum ACWM-Phys queue needed to complete selector labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.effect_labels import build_effect_label_completion_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-label-index", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--candidate-frontier", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidates-per-environment", type=int, default=3)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_effect_label_completion_plan(
        effect_label_index_path=args.effect_label_index,
        screen_summary_path=args.screen_summary,
        candidate_frontier_path=args.candidate_frontier,
        output_root=args.output_root,
        candidates_per_environment=args.candidates_per_environment,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
