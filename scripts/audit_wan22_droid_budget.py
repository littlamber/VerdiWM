#!/usr/bin/env python3
"""Audit WAN2.2-DROID runner receipts against the immutable 40 GPU-hour cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(root: Path, *, cap_gpu_hours: float = 40.0) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if cap_gpu_hours <= 0 or cap_gpu_hours > 40.0:
        raise ValueError("WAN22_DROID_BUDGET_CAP_INVALID")
    rows = []
    blockers = []
    seen: set[Path] = set()
    for path in sorted(root.glob("**/training_receipt.json")):
        if path in seen or path.is_symlink():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"RECEIPT_INVALID:{path}")
            continue
        if payload.get("artifact_type") != "verdiwm-wan22-droid-training-receipt":
            continue
        value = float(payload.get("gpu_hours", -1.0))
        budget = float(payload.get("budget_gpu_hours", -1.0))
        if value < 0 or value > budget or budget <= 0 or budget > cap_gpu_hours:
            blockers.append(f"RECEIPT_GPU_BUDGET_INVALID:{path}")
        rows.append({"path": str(path), "gpu_hours": value, "seed": payload.get("seed"), "conditioning_mode": payload.get("conditioning_mode")})
    total = sum(float(row["gpu_hours"]) for row in rows)
    if total > cap_gpu_hours:
        blockers.append(f"TOTAL_GPU_BUDGET_EXCEEDED:{total:.9f}>{cap_gpu_hours:.9f}")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-budget-receipt",
        "state": "within_budget" if not blockers else "blocked",
        "cap_gpu_hours": cap_gpu_hours,
        "consumed_gpu_hours": total,
        "remaining_gpu_hours": max(0.0, cap_gpu_hours - total),
        "receipt_count": len(rows),
        "rows": rows,
        "blockers": sorted(set(blockers)),
        "claim_boundary": "Budget compliance does not establish model quality or promotion eligibility.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["state"] == "within_budget" else 2


if __name__ == "__main__":
    raise SystemExit(main())
