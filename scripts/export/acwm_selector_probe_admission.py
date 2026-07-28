#!/usr/bin/env python3
"""Admit or reject an ACWM selector probe from held-out replay evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.selector_probe_admission import evaluate_selector_probe_admission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-replay-root", type=Path, required=True)
    parser.add_argument("--candidate-replay-root", type=Path, required=True)
    parser.add_argument("--candidate-atlas-manifest", type=Path, required=True)
    parser.add_argument("--candidate-affinity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = evaluate_selector_probe_admission(
        baseline_replay_root=args.baseline_replay_root,
        candidate_replay_root=args.candidate_replay_root,
        candidate_atlas_manifest=args.candidate_atlas_manifest,
        candidate_affinity=args.candidate_affinity,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
