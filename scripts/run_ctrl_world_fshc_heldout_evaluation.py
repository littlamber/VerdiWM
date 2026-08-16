#!/usr/bin/env python3
"""Run a frozen Ctrl-World FSHC held-out evaluation phase across GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


class FSHCHeldoutLaunchError(RuntimeError):
    """The frozen held-out evaluation frame is invalid."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FSHCHeldoutLaunchError(f"FSHC_HELDOUT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise FSHCHeldoutLaunchError(f"FSHC_HELDOUT_JSON_INVALID:{path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = plan.get("models")
    if not isinstance(rows, list) or not rows:
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_MODELS_INVALID")
    models: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise FSHCHeldoutLaunchError("FSHC_HELDOUT_MODELS_INVALID")
        model_id = str(row["id"])
        if model_id in models:
            raise FSHCHeldoutLaunchError("FSHC_HELDOUT_MODEL_DUPLICATE")
        models[model_id] = row
    return models


def run(args: argparse.Namespace) -> dict[str, object]:
    plan_path = Path(args.plan).resolve(strict=True)
    plan = _load_json(plan_path)
    if plan.get("artifact_type") != "ctrl-world-fshc-heldout-evaluation-plan":
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PLAN_INVALID")
    phases = plan.get("phases")
    dependencies = plan.get("dependencies")
    if not isinstance(phases, Mapping) or not isinstance(dependencies, Mapping):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PLAN_INVALID")
    phase = phases.get(args.phase)
    if not isinstance(phase, Mapping):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PHASE_INVALID")
    models = _model_map(plan)
    model_ids = phase.get("models")
    gpu_assignments = phase.get("gpu_assignments")
    doses = phase.get("doses")
    if (
        not isinstance(model_ids, list)
        or not model_ids
        or not isinstance(gpu_assignments, Mapping)
        or not isinstance(doses, list)
        or len(doses) < 3
    ):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PHASE_INVALID")
    if any(model_id not in models for model_id in model_ids):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PHASE_MODEL_UNKNOWN")
    gpus = [int(gpu_assignments[model_id]) for model_id in model_ids]
    if len(set(gpus)) != len(gpus):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_PHASE_GPU_COLLISION")

    evaluator = Path(str(dependencies["evaluator"])).resolve(strict=True)
    expected_evaluator_hash = str(dependencies.get("evaluator_sha256", ""))
    if expected_evaluator_hash and _sha256(evaluator) != expected_evaluator_hash:
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_EVALUATOR_HASH_MISMATCH")
    runtime_python = Path(str(dependencies["runtime_python"]))
    if not runtime_python.is_absolute() or not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise FSHCHeldoutLaunchError("FSHC_HELDOUT_RUNTIME_PYTHON_INVALID")
    preflight = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "import diffusers, mediapy, torch; print(torch.__version__, diffusers.__version__)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if preflight.returncode != 0:
        raise FSHCHeldoutLaunchError(
            f"FSHC_HELDOUT_RUNTIME_IMPORT_FAILED:{preflight.stdout.strip()}"
        )
    output_root = Path(str(plan["output_root"])).resolve()
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    phase_root = output_root / args.phase
    phase_root.mkdir(mode=0o700, exist_ok=False)
    status_path = output_root / f"{args.phase}-status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ctrl-world-fshc-heldout-phase-status",
        "state": "running",
        "experiment_id": plan["experiment_id"],
        "phase": args.phase,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "started_at_unix": time.time(),
        "jobs": {model_id: {"state": "pending"} for model_id in model_ids},
    }
    _atomic_json(status_path, status)
    lock = threading.Lock()

    def execute(model_id: str) -> tuple[str, int]:
        model = models[model_id]
        gpu = int(gpu_assignments[model_id])
        job_root = phase_root / model_id
        log_path = output_root / "logs" / f"{args.phase}-{model_id}.log"
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [
            str(runtime_python),
            "-u",
            str(evaluator),
            "--campaign-id",
            f"{plan['experiment_id']}:{args.phase}:{model_id}",
            "--probe-id",
            str(phase["probe_id"]),
            "--ctrl-world-root",
            str(dependencies["ctrl_world_root"]),
            "--dataset-root",
            str(dependencies["dataset_root"]),
            "--data-stat",
            str(dependencies["data_stat"]),
            "--svd-model-path",
            str(dependencies["svd_model_path"]),
            "--clip-model-path",
            str(dependencies["clip_model_path"]),
            "--ckpt-path",
            str(model["checkpoint"]),
            "--contexts-json",
            str(plan["contexts_json"]),
            "--doses",
            *[str(value) for value in doses],
            "--interact-num",
            str(phase["interact_num"]),
            "--num-inference-steps",
            str(phase["num_inference_steps"]),
            "--zero-reference-mode",
            str(phase["zero_reference_mode"]),
            "--output-root",
            str(job_root),
        ]
        if phase.get("fshc_dose_mode") is not None:
            command.extend(["--fshc-dose-mode", str(phase["fshc_dose_mode"])])
        if bool(model.get("enable_signed_history_correction")):
            command.append("--enable-signed-history-correction")
        if bool(model.get("unsigned_history_gate")):
            command.append("--unsigned-history-gate")
        if bool(model.get("enable_multiscale_history_adapter")):
            command.append("--enable-multiscale-history-adapter")
        if bool(model.get("multiscale_history_always_on")):
            command.append("--multiscale-history-always-on")
        child_env = dict(os.environ)
        child_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        launch = {
            "state": "running",
            "physical_gpu": gpu,
            "command": command,
            "checkpoint": str(Path(str(model["checkpoint"])).resolve(strict=True)),
            "log_path": str(log_path),
            "started_at_unix": time.time(),
        }
        with lock:
            status["jobs"][model_id] = launch
            _atomic_json(status_path, status)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = job_root / "result.json"
        ready = False
        if process.returncode == 0 and result_path.is_file():
            ready = _load_json(result_path).get("state") == "ready"
        receipt = {
            **launch,
            "schema_version": 1,
            "artifact_type": "ctrl-world-fshc-heldout-job-receipt",
            "state": "completed" if ready else "failed",
            "return_code": process.returncode,
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path) if result_path.is_file() else None,
            "completed_at_unix": time.time(),
        }
        _atomic_json(output_root / "receipts" / f"{args.phase}-{model_id}.json", receipt)
        with lock:
            status["jobs"][model_id] = receipt
            _atomic_json(status_path, status)
        return model_id, 0 if ready else max(process.returncode, 1)

    return_codes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(model_ids)) as executor:
        futures = [executor.submit(execute, model_id) for model_id in model_ids]
        for future in as_completed(futures):
            model_id, return_code = future.result()
            return_codes[model_id] = return_code
    status["state"] = "completed" if all(value == 0 for value in return_codes.values()) else "failed"
    status["return_codes"] = return_codes
    status["completed_at_unix"] = time.time()
    _atomic_json(status_path, status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("action_sensitivity", "routing"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
