#!/usr/bin/env python3
"""Settle one or more Ctrl-World target-local fingerprint radii."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.ctrl_world_fingerprint_settlement import settle_ctrl_world_fingerprints


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-root", type=Path, action="append", required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = settle_ctrl_world_fingerprints(
        fingerprint_roots=args.fingerprint_root,
        protocol=args.protocol,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
