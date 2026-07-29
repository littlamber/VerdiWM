#!/usr/bin/env python3
"""Settle a dev-selected Cosmos3 directional probe on accept evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.cosmos3_directional_settlement import (
    settle_cosmos3_fingerprint_pair,
    settle_cosmos3_directional_probe,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    dev = parser.add_mutually_exclusive_group(required=True)
    dev.add_argument("--dev-selection-root", type=Path)
    dev.add_argument("--dev-fingerprint-root", type=Path)
    parser.add_argument("--accept-fingerprint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.dev_fingerprint_root is not None:
        manifest = settle_cosmos3_fingerprint_pair(
            dev_fingerprint_root=args.dev_fingerprint_root,
            accept_fingerprint_root=args.accept_fingerprint_root,
            output_root=args.output_root,
        )
    else:
        manifest = settle_cosmos3_directional_probe(
            dev_selection_root=args.dev_selection_root,
            accept_fingerprint_root=args.accept_fingerprint_root,
            output_root=args.output_root,
        )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
