#!/usr/bin/env python3
"""Execute RoboCoach's frozen SA-WM evaluator from a VerdiWM run manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_VERDIWM_ROOT = Path(__file__).resolve().parents[1]
if str(_VERDIWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERDIWM_ROOT))
from scripts.evaluate_psnr_ssim_smoke import PsnrSsimError, evaluate_video_rollouts


class Wan22EvalError(RuntimeError):
    pass


_PSNR_SSIM_EVALUATOR_ID = "wan22-droid-psnr-ssim-smoke-v1"


def _evaluator_contract(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Wan22EvalError("WAN22_EVAL_CONTRACT_INVALID") from exc
    if not isinstance(value, dict):
        raise Wan22EvalError("WAN22_EVAL_CONTRACT_INVALID")
    return value


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
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise Wan22EvalError("WAN22_EVAL_CONTRACT_INVALID")
    contract_path = Path(str(evaluation.get("evaluator_contract", ""))).expanduser().resolve()
    contract = _evaluator_contract(contract_path)
    evaluator_id = contract.get("evaluator_id")
    if evaluator_id != _PSNR_SSIM_EVALUATOR_ID or contract.get("metrics") != ["psnr", "ssim"]:
        raise Wan22EvalError(
            "WAN22_EVAL_CONTRACT_UNSUPPORTED:"
            f"expected={_PSNR_SSIM_EVALUATOR_ID}:received={evaluator_id}"
        )
    if os.environ.get("VERDIWM_EXECUTE_WAN22", "0") != "1":
        raise Wan22EvalError(
            "WAN22_EVAL_EXECUTION_DISABLED:set VERDIWM_EXECUTE_WAN22=1 after a GPU lease"
        )
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
    completed = subprocess.run(command, cwd=str(source_root), capture_output=True, text=True, check=False)
    (output / "eval.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output / "eval.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise Wan22EvalError(f"WAN22_EVAL_FAILED:returncode={completed.returncode}")
    rollout_root = output / str(manifest.get("model_run_id", "verdi")) / "rollouts"
    measurement_root = output / "psnr-ssim"
    baseline_value = os.environ.get("VERDIWM_WAN22_BASELINE_RECEIPT", "").strip()
    try:
        measurement = evaluate_video_rollouts(
            rollout_root=rollout_root,
            output_root=measurement_root,
            contract_path=contract_path,
            baseline_receipt=Path(baseline_value) if baseline_value else None,
        )
    except PsnrSsimError as exc:
        raise Wan22EvalError(f"WAN22_EVAL_MEASUREMENT_FAILED:{exc}") from exc

    receipt_path = Path(str(evaluation["evidence_receipt"])).expanduser().resolve()
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-heldout-evidence-receipt",
        "state": "complete",
        "model_run_id": manifest.get("model_run_id"),
        "evaluator_contract": str(contract_path),
        "evaluator_id": _PSNR_SSIM_EVALUATOR_ID,
        "evidence_source": "paired_ground_truth_rollout",
        "output_root": str(output),
        "rollout_root": str(rollout_root),
        "measurement_root": str(measurement_root),
        "measurement_receipt": measurement["receipt_path"],
        "measurement_receipt_sha256": _sha256(Path(str(measurement["receipt_path"]))),
        "sample_count": measurement["sample_count"],
        "metrics": measurement["metrics"],
        "verdict": measurement["verdict"],
        "claim_boundary": contract.get(
            "claim_boundary",
            "Paired PSNR/SSIM evidence only; this does not establish minute-level consistency or transfer validity.",
        ),
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
