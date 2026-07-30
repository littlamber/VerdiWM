#!/usr/bin/env python3
"""Run the preregistered ACWM-Phys S4 random probe expansion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wmloop.experiments.random_probe_expansion import run_random_probe_expansion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_random_probe_expansion(
        config_path=args.config,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
