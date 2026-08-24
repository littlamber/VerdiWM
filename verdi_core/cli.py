"""CLI for the clean release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.fixture_world import FixtureWorldAdapter

from .loop import run_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verdi")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    demo = commands.add_parser("demo")
    demo.add_argument("--state-root", type=Path, required=True)
    graph = commands.add_parser("graph")
    graph.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps({"state": "ready", "kernel": "model-agnostic", "adapter_contract": "v1"}, sort_keys=True))
        return 0
    if args.command == "demo":
        print(json.dumps(run_loop(FixtureWorldAdapter(), state_root=args.state_root), indent=2, sort_keys=True))
        return 0
    path = args.state_root / "knowledge" / "knowledge.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"state": "ready", "record_count": len(records), "outcomes": [r["outcome"] for r in records]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
