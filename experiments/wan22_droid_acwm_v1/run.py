#!/usr/bin/env python3
"""Prepare and validate the WAN2.2-DROID ACWM execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.wan22_droid import (  # noqa: E402
    Wan22DroidError,
    build_sample_manifest,
    validate_contract,
    write_sample_manifest,
)


def _dump(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--horizon-frames", type=int, default=150)
    prepare.add_argument("--stride", type=int, default=30)
    conformance = commands.add_parser("conformance")
    conformance.add_argument("--train-manifest", type=Path, required=True)
    conformance.add_argument("--validation-manifest", type=Path, required=True)
    conformance.add_argument("--model", type=Path, required=True)
    conformance.add_argument("--source", type=Path, required=True)
    conformance.add_argument("--evaluator-contract", type=Path, required=True)
    conformance.add_argument("--adapter", type=Path, required=True)
    conformance.add_argument("--horizon-frames", type=int, default=150)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = args.output_root.expanduser().resolve()
            train = write_sample_manifest(args.data_root, "train", output / "train.json", horizon_frames=args.horizon_frames, stride=args.stride)
            val = write_sample_manifest(args.data_root, "val", output / "val.json", horizon_frames=args.horizon_frames, stride=args.stride)
            _dump({"state": "ready", "train": train, "validation": val, "output_root": str(output)})
            return 0
        report = validate_contract(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            model=args.model,
            source=args.source,
            evaluator_contract=args.evaluator_contract,
            adapter=args.adapter,
            horizon_frames=args.horizon_frames,
        )
        _dump(report)
        return 0 if report["state"] == "ready_for_execution" else 2
    except Wan22DroidError as exc:
        _dump({"state": "blocked", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
