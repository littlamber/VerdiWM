#!/usr/bin/env python3
"""Replay ACWM selector choices on settled leave-one-environment-out labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.selector_replay import run_selector_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--effect-label-index", type=Path, required=True)
    parser.add_argument("--primitive-probe-affinity", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_selector_replay(
        plan_path=args.plan,
        projection_path=args.projections,
        effect_label_index=args.effect_label_index,
        output_root=args.output_root,
        primitive_probe_affinity=args.primitive_probe_affinity,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
