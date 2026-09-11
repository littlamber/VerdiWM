#!/usr/bin/env python3
"""Execute RoboCoach's frozen SA-WM evaluator from a VerdiWM run manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


class Wan22EvalError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Wan22EvalError("WAN22_MODEL_RUN_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise Wan22EvalError("WAN22_MODEL_RUN_MANIFEST_INVALID")
    return value


def run(manifest_path: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise Wan22EvalError("WAN22_EVAL_SOURCE_INVALID")
    source_root = Path(str(source.get("source_root") or source.get("model_root"))).expanduser().resolve()
    runtime = Path(str(manifest.get("runtime", {}).get("python"))).expanduser().absolute()
    script = source_root / "scripts" / "evaluation" / "run_sa_wm_eval_manifest.py"
    if not source_root.is_dir() or not script.is_file():
        raise Wan22EvalError("WAN22_EVAL_ENTRYPOINT_MISSING")
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise Wan22EvalError("WAN22_EVAL_RUNTIME_INVALID")
    bindings = source.get("asset_bindings")
    if not isinstance(bindings, dict):
        raise Wan22EvalError("WAN22_EVAL_ASSET_BINDINGS_INVALID")
    model = bindings.get("--checkpoint")
    data = bindings.get("--data-root")
    if not model or not data:
        raise Wan22EvalError("WAN22_EVAL_MODEL_OR_DATA_MISSING")
    output = Path(str(manifest.get("output", {}).get("root"))).expanduser().resolve() / "evaluation"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    eval_manifest = os.environ.get(
        "VERDIWM_WAN22_EVAL_MANIFEST",
        str(source_root / "configs" / "evaluation" / "sa_wm_eval_long_v1.json"),
    )
    config = os.environ.get("VERDIWM_WAN22_EVAL_CONFIG", "configs/training/sa_wm_pretrain_v6.yaml")
    command = [
        str(runtime), str(script), "--manifest", eval_manifest, "--tier", "dev",
        "--config", config, "--checkpoint", str(model), "--name", str(manifest.get("model_run_id", "verdi")),
        "--out-root", str(output), "--gpus", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    ]
    if os.environ.get("VERDIWM_EXECUTE_WAN22", "0") != "1":
        raise Wan22EvalError(
            "WAN22_EVAL_EXECUTION_DISABLED:set VERDIWM_EXECUTE_WAN22=1 after a GPU lease"
        )
    completed = subprocess.run(command, cwd=str(source_root), capture_output=True, text=True, check=False)
    (output / "eval.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output / "eval.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise Wan22EvalError(f"WAN22_EVAL_FAILED:returncode={completed.returncode}")
    receipt_path = Path(str(manifest["evaluation"]["evidence_receipt"])).expanduser().resolve()
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-heldout-evidence-receipt",
        "state": "complete",
        "model_run_id": manifest.get("model_run_id"),
        "evaluator_contract": manifest.get("evaluation", {}).get("evaluator_contract"),
        "evidence_source": "paired_ground_truth_rollout",
        "output_root": str(output),
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
