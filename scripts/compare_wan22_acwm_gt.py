#!/usr/bin/env python3
"""Create a side-by-side ACWM/GT video and contact sheet for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmloop.execute.paired_visualization import create_paired_visualization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--training-steps", type=int)
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve(strict=True)
    output_root = (args.output_root or run_root / "acwm-gt-visualization").expanduser().resolve()
    result = create_paired_visualization(
        generated_video=run_root / "generated_150f.mp4",
        ground_truth_video=run_root / "ground_truth_150f.mp4",
        output_root=output_root,
        metadata={
            "seed": args.seed,
            "model_id": args.model_id,
            "training_steps": args.training_steps,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
