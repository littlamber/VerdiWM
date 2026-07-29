#!/usr/bin/env python3
"""Build a dev-only directional Cosmos3 probe-evolution receipt."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.cosmos3_directional_probe import (
    build_cosmos3_directional_probe_evolution,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fingerprint-root", type=Path, required=True)
    parser.add_argument("--successor-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_cosmos3_directional_probe_evolution(
        source_fingerprint_root=args.source_fingerprint_root,
        successor_campaign_path=args.successor_campaign,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
