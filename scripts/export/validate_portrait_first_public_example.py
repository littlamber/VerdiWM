#!/usr/bin/env python3
"""Validate the CPU-only public portrait-first control-plane example."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class PortraitFirstPublicExampleError(RuntimeError):
    """The public example is missing, malformed, or changed its contract."""


def validate_portrait_first_public_example(example_root: Path) -> dict[str, object]:
    root = Path(example_root).resolve(strict=True)
    runner = root / "run.py"
    if not runner.is_file() or runner.is_symlink():
        raise PortraitFirstPublicExampleError("PORTRAIT_FIRST_PUBLIC_RUNNER_MISSING")
    completed = subprocess.run(
        [sys.executable, str(runner)],
        cwd=root.parents[1],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
        suffix = detail[0][:240] if detail else "no-output"
        raise PortraitFirstPublicExampleError(f"PORTRAIT_FIRST_PUBLIC_RUNNER_FAILED:{suffix}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PortraitFirstPublicExampleError("PORTRAIT_FIRST_PUBLIC_OUTPUT_INVALID") from exc
    expected = {
        "artifact_type": "verdiwm-public-portrait-first-demo",
        "state": "ready",
        "readiness_state": "ready_for_gap_planning",
        "gap_plan_state": "ready_for_portfolio",
        "requirement_classification": "satisfied",
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise PortraitFirstPublicExampleError("PORTRAIT_FIRST_PUBLIC_CONTRACT_INVALID")
    authority = payload.get("authority")
    if authority != {
        "gpu_authority": False,
        "module_manufacturing_authority": False,
        "promotion_authority": False,
    }:
        raise PortraitFirstPublicExampleError("PORTRAIT_FIRST_PUBLIC_AUTHORITY_INVALID")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_portrait_first_public_example(args.example_root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
