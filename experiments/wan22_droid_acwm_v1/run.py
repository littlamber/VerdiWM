#!/usr/bin/env python3
"""Prepare and validate the WAN2.2-DROID ACWM execution contract."""

from __future__ import annotations

import argparse
import json
import subprocess
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
    closed_loop = commands.add_parser("closed-loop")
    for argument in ("train_manifest", "validation_manifest"):
        closed_loop.add_argument(f"--{argument.replace('_', '-')}", type=Path, required=True)
    closed_loop.add_argument("--model", type=Path, required=True)
    closed_loop.add_argument("--source", type=Path, required=True)
    closed_loop.add_argument("--adapter", type=Path, required=True)
    closed_loop.add_argument("--evaluator-contract", type=Path, required=True)
    closed_loop.add_argument("--runtime-python", type=Path, required=True)
    closed_loop.add_argument("--runner", type=Path, help="WAN2.2-DROID runner owned by the caller")
    closed_loop.add_argument("--output-root", type=Path, required=True)
    closed_loop.add_argument("--execute", action="store_true", help="allow the explicitly bound runner to use GPU")
    closed_loop.add_argument("--horizon-frames", type=int, default=150)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = args.output_root.expanduser().resolve()
            train = write_sample_manifest(args.data_root, "train", output / "train.json", horizon_frames=args.horizon_frames, stride=args.stride)
            val = write_sample_manifest(args.data_root, "val", output / "val.json", horizon_frames=args.horizon_frames, stride=args.stride)
            _dump({"state": "ready", "train": train, "validation": val, "output_root": str(output)})
            return 0
        if args.command == "closed-loop":
            report = validate_contract(
                train_manifest=args.train_manifest,
                validation_manifest=args.validation_manifest,
                model=args.model,
                source=args.source,
                evaluator_contract=args.evaluator_contract,
                adapter=args.adapter,
                horizon_frames=args.horizon_frames,
            )
            blockers = list(report["blockers"])
            runtime = args.runtime_python.expanduser().resolve()
            if not runtime.is_file() or not runtime.exists():
                blockers.append("RUNTIME_PYTHON_INVALID")
            else:
                try:
                    probe = subprocess.run(
                        [str(runtime), "-c", "import torch; assert torch.cuda.is_available()"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    probe = None
                if probe is None or probe.returncode != 0:
                    blockers.append("RUNTIME_TORCH_CUDA_UNAVAILABLE")
            if args.runner is None:
                blockers.append("WAN22_DROID_RUNNER_REQUIRED")
            elif not args.runner.expanduser().resolve().is_file():
                blockers.append("WAN22_DROID_RUNNER_INVALID")
            if not args.execute:
                blockers.append("GPU_EXECUTION_NOT_ARMED")
            receipt = {
                "schema_version": 1,
                "artifact_type": "verdiwm-wan22-droid-closed-loop-receipt",
                "state": "blocked" if blockers else "ready_to_launch",
                "budget_gpu_hours": 40,
                "conformance": report,
                "runtime_python": str(runtime),
                "runner": str(args.runner.expanduser().resolve()) if args.runner else None,
                "output_root": str(args.output_root.expanduser().resolve()),
                "stages": ["train", "rollout_150f", "worldarena_frozen_eval", "promotion"],
                "blockers": sorted(set(blockers)),
                "claim_boundary": "A ready-to-launch receipt does not prove the 30-second target; promotion requires frozen WorldArena evidence.",
            }
            _dump(receipt)
            return 0 if receipt["state"] == "ready_to_launch" else 2
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
