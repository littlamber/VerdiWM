"""Receipt-first execution for compiled model-run manifests."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from wmloop.control.model_run import ModelRunError, validate_model_run
from wmloop.execute.model_adapters import ModelLaunchAdapterError, preflight_model_adapter
from wmloop.execute.gpu_lease import GpuLease, GpuLeaseError, GpuLeaseManager


class ModelRunnerError(RuntimeError):
    """A bounded model run failed before producing authoritative evidence."""


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


def execute_model_run(
    manifest: Mapping[str, object],
    *,
    lease_manager: GpuLeaseManager | None = None,
    process_runner: ProcessRunner | None = None,
    lease_wait_seconds: float = 0.0,
) -> dict[str, object]:
    """Run train then held-out evaluation under the manifest's GPU lease."""

    try:
        validate_model_run(manifest)
    except ModelRunError as exc:
        raise ModelRunnerError(f"MODEL_RUN_MANIFEST_INVALID:{exc}") from exc
    try:
        preflight_model_adapter(manifest)
    except ModelLaunchAdapterError as exc:
        raise ModelRunnerError(str(exc)) from exc
    output = Path(str(manifest["output"]["root"])).resolve()  # type: ignore[index]
    if output.exists():
        if not output.is_dir():
            raise ModelRunnerError("MODEL_RUN_OUTPUT_NOT_FRESH")
        existing = [path for path in output.iterdir() if path.name != "model-run.json"]
        if existing:
            raise ModelRunnerError("MODEL_RUN_OUTPUT_NOT_FRESH")
    else:
        output.mkdir(mode=0o700, parents=True)
    allocation = manifest["resource_binding"]["allocation"]  # type: ignore[index]
    indices = _lease_indices(allocation)  # type: ignore[arg-type]
    world_size = int(manifest["training"]["scale_plan"]["parallelism"]["world_size"])  # type: ignore[index]
    if len(indices) < world_size:
        raise ModelRunnerError("MODEL_RUN_RESOURCE_CAPACITY_INVALID")
    manager = lease_manager or GpuLeaseManager()
    try:
        leases = manager.acquire_many(indices, world_size, wait_seconds=lease_wait_seconds)
    except GpuLeaseError as exc:
        raise ModelRunnerError(str(exc)) from exc
    runner = process_runner or subprocess.run
    try:
        source = manifest["source"]  # type: ignore[index]
        generated_adapter = isinstance(source, Mapping) and source.get("adapter_root") is not None
        env = _generated_adapter_environment(output) if generated_adapter else dict(os.environ)
        env.update(_lease_environment(leases))
        cwd = Path(str(manifest["source"]["model_root"])).resolve()  # type: ignore[index]
        runtime = str(manifest["runtime"]["python"])  # type: ignore[index]
        train = _command(runtime, manifest["runtime"]["train_command"])  # type: ignore[index]
        evaluate = _command(runtime, manifest["evaluation"]["command"])  # type: ignore[index]
        started = time.time()
        train_result = _run(runner, train, cwd=cwd, env=env)
        train_record = _stage_record("train", train, train_result)
        if train_result.returncode != 0:
            raise ModelRunnerError("MODEL_RUN_TRAIN_FAILED")
        checkpoint = _receipt(Path(str(manifest["training"]["checkpoint_receipt"])))  # type: ignore[index]
        evaluate_result = _run(runner, evaluate, cwd=cwd, env=env)
        evaluate_record = _stage_record("evaluate", evaluate, evaluate_result)
        if evaluate_result.returncode != 0:
            raise ModelRunnerError("MODEL_RUN_EVALUATE_FAILED")
        evidence = _receipt(Path(str(manifest["evaluation"]["evidence_receipt"])))  # type: ignore[index]
        result = {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-run-execution-receipt",
            "state": "completed",
            "model_run_id": manifest["model_run_id"],
            "checkpoint_receipt": str(checkpoint),
            "evidence_receipt": str(evidence),
            "stages": [train_record, evaluate_record],
            "leased_gpus": [lease.to_document() for lease in leases],
            "elapsed_seconds": round(time.time() - started, 6),
            "claim_boundary": "Execution provenance only; held-out evidence remains subject to the frozen evaluator and promotion policy.",
        }
        _write_json(output / "execution-receipt.json", result)
        return result
    finally:
        for lease in leases:
            lease.release()


def _lease_indices(allocation: Mapping[str, object]) -> list[int]:
    roles = allocation.get("roles")
    if not isinstance(roles, list):
        raise ModelRunnerError("MODEL_RUN_RESOURCE_ALLOCATION_INVALID")
    matches = [row for row in roles if isinstance(row, Mapping) and row.get("role") == "autonomous_candidate_evaluation"]
    if len(matches) != 1 or not isinstance(matches[0].get("gpu_indices"), list):
        raise ModelRunnerError("MODEL_RUN_RESOURCE_ROLE_INVALID")
    indices = matches[0]["gpu_indices"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices):
        raise ModelRunnerError("MODEL_RUN_RESOURCE_INDICES_INVALID")
    return list(indices)


def _lease_environment(leases: Sequence[GpuLease]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(lease.index) for lease in leases),
        "VERDIWM_PHYSICAL_GPU_INDEXES": ",".join(str(lease.index) for lease in leases),
        "VERDIWM_PHYSICAL_GPU_UUIDS": ",".join(lease.uuid for lease in leases),
    }


def _generated_adapter_environment(output: Path) -> dict[str, str]:
    home = output / "adapter-home"
    scratch = output / "adapter-tmp"
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _command(runtime: str, command: object) -> list[str]:
    if not isinstance(command, list) or not command or any(not isinstance(token, str) or not token for token in command):
        raise ModelRunnerError("MODEL_RUN_COMMAND_INVALID")
    return [runtime, *command]


def _run(runner: ProcessRunner, command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    try:
        return runner(list(command), cwd=str(cwd), env=dict(env), capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ModelRunnerError("MODEL_RUN_PROCESS_START_FAILED") from exc


def _stage_record(stage: str, command: Sequence[str], result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return {"stage": stage, "command": list(command), "returncode": int(result.returncode), "stdout_tail": str(result.stdout or "")[-2000:], "stderr_tail": str(result.stderr or "")[-2000:]}


def _receipt(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID") from exc
    if not isinstance(value, Mapping):
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID")
    return resolved


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
