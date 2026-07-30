#!/usr/bin/env python3
"""Execute one frozen ACWM source-effect repair job."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.experiments.source_effect_repair import (
    execute_source_effect_repair_job,
    load_source_effect_repair_plan,
    settle_source_effect_repair,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.preflight:
        plan = load_source_effect_repair_plan(config_path=args.config)
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.settle:
        if args.output_root is None:
            parser.error("--output-root is required with --settle")
        manifest = settle_source_effect_repair(
            config_path=args.config,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if not args.job_id or args.gpu_index is None:
        parser.error("--job-id and --gpu-index are required without --preflight")
    manifest = execute_source_effect_repair_job(
        config_path=args.config,
        job_id=args.job_id,
        gpu_index=args.gpu_index,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
