#!/usr/bin/env python3
"""Reclassify an immutable mechanism atlas against expanded references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wmloop.retrieve.mechanism_discovery import (
    MechanismDiscoveryError,
    recompare_mechanism_atlas,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-atlas", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--reference-atlas", type=Path, action="append", default=[])
    parser.add_argument("--reference-profiles", type=Path, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = recompare_mechanism_atlas(
            input_atlas=args.input_atlas,
            output_root=args.output_root,
            repo_root=args.repo_root,
            reference_atlas_paths=args.reference_atlas,
            reference_profiles_paths=args.reference_profiles,
        )
    except MechanismDiscoveryError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
