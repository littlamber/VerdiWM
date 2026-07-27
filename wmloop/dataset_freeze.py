"""Create or verify the immutable ACWM dataset and held-out artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmloop.freeze import (
    freeze_acwm_dataset,
    make_acwm_heldout_protocol,
    verify_acwm_dataset_freeze,
    verify_acwm_heldout_protocol,
    write_acwm_dataset_freeze,
    write_acwm_heldout_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="hash a complete dataset and write its held-out protocol")
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--seed", type=int, default=20260721)
    create.add_argument("--dev-ratio", type=float, default=0.7)
    create.add_argument("--scope", choices=("full", "evaluation"), default="full")
    verify = commands.add_parser("verify", help="verify existing immutable artifacts against local data")
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--dataset-freeze", type=Path, required=True)
    verify.add_argument("--heldout-protocol", type=Path, required=True)
    verify.add_argument("--scope", choices=("full", "evaluation"), default="full")
    args = parser.parse_args(argv)
    required_splits = ("ind_train", "ind_test", "ood_test") if args.scope == "full" else ("ind_test", "ood_test")
    if args.command == "create":
        dataset_freeze = freeze_acwm_dataset(args.data_root, required_splits=required_splits)
        protocol = make_acwm_heldout_protocol(dataset_freeze, seed=args.seed, dev_ratio=args.dev_ratio)
        write_acwm_dataset_freeze(args.output_dir / "dataset-freeze.json", dataset_freeze)
        write_acwm_heldout_protocol(args.output_dir / "heldout-protocol.json", protocol)
        print(json.dumps({"ready": True, "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
        return 0
    dataset_freeze = json.loads(args.dataset_freeze.read_text(encoding="utf-8"))
    protocol = json.loads(args.heldout_protocol.read_text(encoding="utf-8"))
    verify_acwm_dataset_freeze(args.data_root, dataset_freeze, required_splits=required_splits)
    verify_acwm_heldout_protocol(dataset_freeze, protocol)
    print(json.dumps({"ready": True, "verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
