#!/usr/bin/env python3
"""Reject a redundant ACWM successor probe before an eight-environment pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.probe_smoke_redundancy import evaluate_probe_smoke_redundancy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-campaign-root", type=Path, required=True)
    parser.add_argument("--candidate-campaign-root", type=Path, required=True)
    parser.add_argument("--environment", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.1)
    parser.add_argument("--maximum-locality-residual", type=float, default=0.5)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = evaluate_probe_smoke_redundancy(
        reference_campaign_root=args.reference_campaign_root,
        candidate_campaign_root=args.candidate_campaign_root,
        environments=args.environment,
        output_root=args.output_root,
        minimum_cosine_similarity=args.minimum_cosine_similarity,
        maximum_relative_l2=args.maximum_relative_l2,
        maximum_locality_residual=args.maximum_locality_residual,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
