#!/usr/bin/env python3
"""Reject WAN2.2 training until its budget and checkpoint handoff are bound.

The external trainer does not yet consume the model-run scale plan or isolated
output root. Its checkpoint layout also has no validated handoff to evaluation.
An execution opt-in cannot authorize this incomplete bridge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


class Wan22TrainError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Wan22TrainError("WAN22_MODEL_RUN_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise Wan22TrainError("WAN22_MODEL_RUN_MANIFEST_INVALID")
    return value


def run(manifest_path: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise Wan22TrainError("WAN22_MODEL_RUN_SOURCE_INVALID")
    if os.environ.get("VERDIWM_EXECUTE_WAN22", "0") != "1":
        raise Wan22TrainError(
            "WAN22_TRAIN_EXECUTION_DISABLED:explicit execution opt-in is required; "
            "the training bridge also requires budget, output and checkpoint bindings"
        )
    raise Wan22TrainError(
        "WAN22_TRAIN_BUDGET_OUTPUT_BINDING_UNSUPPORTED:"
        "training.scale_plan budgets and support are not mapped to the external trainer; "
        "output.root is not bound to its checkpoint directory; "
        "checkpoint_receipt does not identify a trained checkpoint and evaluation "
        "still selects the initial checkpoint"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
