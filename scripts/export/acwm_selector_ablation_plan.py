#!/usr/bin/env python3
"""Export the matched ACWM selector-ablation plan and evidence checklist."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.selector_ablation import build_selector_ablation_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fingerprint-atlas-root", type=Path, required=True)
    parser.add_argument("--effect-label-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_selector_ablation_plan(
        config_path=args.config,
        fingerprint_atlas_root=args.fingerprint_atlas_root,
        effect_label_index=args.effect_label_index,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
