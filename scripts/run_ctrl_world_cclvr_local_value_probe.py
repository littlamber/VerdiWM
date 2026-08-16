#!/usr/bin/env python3
"""Launch the frozen CCLVR interaction-local counterfactual probe on eight GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


class CCLVRProbeLaunchError(RuntimeError):
    """The frozen local-value campaign or one of its GPU jobs is invalid."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRProbeLaunchError(f"CCLVR_PROBE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRProbeLaunchError(f"CCLVR_PROBE_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return (ROOT / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _executable(value: object) -> Path:
    path = Path(str(value))
    path = ROOT / path if not path.is_absolute() else path
    path = path.absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CCLVRProbeLaunchError("CCLVR_PROBE_RUNTIME_INVALID")
    return path


def _validated_plan(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, Path, Path]:
    plan = _load(path)
    checkpoint = plan.get("checkpoint")
    contexts = plan.get("contexts")
    dependencies = plan.get("dependencies")
    protocol = plan.get("protocol")
    execution = plan.get("execution")
    if (
        plan.get("artifact_type") != "verdiwm-ctrl-world-cclvr-local-value-campaign"
        or plan.get("state") != "frozen_before_execution"
        or not isinstance(checkpoint, Mapping)
        or not isinstance(contexts, Mapping)
        or not isinstance(dependencies, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(execution, Mapping)
    ):
        raise CCLVRProbeLaunchError("CCLVR_PROBE_PLAN_INVALID")
    checkpoint_path = _resolve(checkpoint.get("path"))
    contexts_path = _resolve(contexts.get("path"))
    evaluator_path = _resolve(dependencies.get("evaluator"))
    launcher_path = _resolve(dependencies.get("launcher"))
    for expected, resolved, error in (
        (checkpoint.get("sha256"), checkpoint_path, "CCLVR_PROBE_CHECKPOINT_HASH_MISMATCH"),
        (contexts.get("sha256"), contexts_path, "CCLVR_PROBE_CONTEXTS_HASH_MISMATCH"),
        (dependencies.get("evaluator_sha256"), evaluator_path, "CCLVR_PROBE_EVALUATOR_HASH_MISMATCH"),
        (dependencies.get("launcher_sha256"), launcher_path, "CCLVR_PROBE_LAUNCHER_HASH_MISMATCH"),
    ):
        if not isinstance(expected, str) or _sha256(resolved) != expected:
            raise CCLVRProbeLaunchError(error)
    context_payload = _load(contexts_path)
    rows = context_payload.get("contexts")
    assignments = execution.get("gpu_assignments")
    if (
        context_payload.get("artifact_type") != "verdiwm-ctrl-world-local-context-set"
        or not isinstance(rows, list)
        or len(rows) != int(contexts.get("context_count", -1))
        or not isinstance(assignments, list)
        or len(assignments) != int(execution.get("gpus", -1))
    ):
        raise CCLVRProbeLaunchError("CCLVR_PROBE_EXECUTION_FRAME_INVALID")
    expected_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("context_id"), str):
            raise CCLVRProbeLaunchError("CCLVR_PROBE_CONTEXTS_INVALID")
        if str(row.get("episode_id")) == "1799":
            raise CCLVRProbeLaunchError("CCLVR_PROBE_PROMOTION_EPISODE_FORBIDDEN")
        expected_ids.add(str(row["context_id"]))
    assigned_ids: list[str] = []
    gpus: list[int] = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or not isinstance(assignment.get("context_ids"), list):
            raise CCLVRProbeLaunchError("CCLVR_PROBE_GPU_ASSIGNMENT_INVALID")
        gpus.append(int(assignment.get("gpu", -1)))
        assigned_ids.extend(str(value) for value in assignment["context_ids"])
    if len(set(gpus)) != len(gpus) or set(assigned_ids) != expected_ids or len(assigned_ids) != len(expected_ids):
        raise CCLVRProbeLaunchError("CCLVR_PROBE_GPU_ASSIGNMENT_INVALID")
    return plan, context_payload, checkpoint_path, contexts_path, evaluator_path


def run(*, plan_path: Path, output_root: Path) -> dict[str, object]:
    plan_path = plan_path.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CCLVRProbeLaunchError("CCLVR_PROBE_OUTPUT_EXISTS")
    plan, _contexts, checkpoint, contexts_path, evaluator = _validated_plan(plan_path)
    dependencies = plan["dependencies"]
    protocol = plan["protocol"]
    execution = plan["execution"]
    runtime = _executable(dependencies["runtime_python"])
    preflight = subprocess.run(
        [str(runtime), "-c", "import diffusers, mediapy, torch; assert torch.cuda.is_available()"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if preflight.returncode != 0:
        raise CCLVRProbeLaunchError(f"CCLVR_PROBE_RUNTIME_IMPORT_FAILED:{preflight.stdout.strip()}")

    output_root.mkdir(mode=0o700, parents=True)
    status_path = output_root / "probe-status.json"
    assignments = execution["gpu_assignments"]
    status: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-probe-status",
        "state": "running",
        "campaign_id": plan["campaign_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "started_at_unix": time.time(),
        "jobs": {f"gpu-{int(row['gpu'])}": {"state": "pending"} for row in assignments},
    }
    _atomic_json(status_path, status)
    lock = threading.Lock()

    def execute(assignment: Mapping[str, Any]) -> tuple[str, int]:
        gpu = int(assignment["gpu"])
        job_id = f"gpu-{gpu}"
        context_ids = [str(value) for value in assignment["context_ids"]]
        job_root = output_root / "shards" / job_id
        log_path = output_root / "logs" / f"{job_id}.log"
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [
            str(runtime),
            "-u",
            str(evaluator),
            "--campaign-id",
            str(plan["campaign_id"]),
            "--probe-id",
            "fshc_interaction_local_gain",
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
            str(checkpoint),
            "--contexts-json",
            str(contexts_path),
            "--context-ids",
            *context_ids,
            "--doses",
            *[str(value) for value in protocol["doses"]],
            "--target-interactions",
            *[str(value) for value in protocol["target_interactions"]],
            "--interact-num",
            str(protocol["interact_num"]),
            "--num-inference-steps",
            str(protocol["num_inference_steps"]),
            "--fshc-dose-mode",
            "normalized_mechanism",
            "--zero-reference-mode",
            str(protocol["zero_reference_mode"]),
            "--enable-multiscale-history-adapter",
            "--output-root",
            str(job_root),
        ]
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
            "context_ids": context_ids,
            "command": command,
            "log_path": str(log_path),
            "started_at_unix": time.time(),
        }
        with lock:
            status["jobs"][job_id] = launch
            _atomic_json(status_path, status)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = job_root / "result.json"
        ready = process.returncode == 0 and result_path.is_file() and _load(result_path).get("state") == "ready"
        receipt = {
            **launch,
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-cclvr-probe-job-receipt",
            "state": "completed" if ready else "failed",
            "return_code": process.returncode,
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path) if result_path.is_file() else None,
            "completed_at_unix": time.time(),
        }
        _atomic_json(output_root / "receipts" / f"{job_id}.json", receipt)
        with lock:
            status["jobs"][job_id] = receipt
            _atomic_json(status_path, status)
        return job_id, 0 if ready else max(process.returncode, 1)

    return_codes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [executor.submit(execute, row) for row in assignments]
        for future in as_completed(futures):
            job_id, return_code = future.result()
            return_codes[job_id] = return_code
    status["state"] = "completed" if all(value == 0 for value in return_codes.values()) else "failed"
    status["return_codes"] = return_codes
    status["completed_at_unix"] = time.time()
    _atomic_json(status_path, status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(plan_path=args.plan, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CCLVRProbeLaunchError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        raise SystemExit(2) from exc
