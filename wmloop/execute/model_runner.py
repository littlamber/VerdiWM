"""Receipt-first execution for compiled model-run manifests."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

from wmloop.control.model_run import ModelRunError, validate_model_run
from wmloop.execute import gpu_lease
from wmloop.execute.model_adapters import (
    ModelLaunchAdapterError,
    preflight_model_adapter,
)
from wmloop.execute.gpu_lease import GpuLease, GpuLeaseCancelled, GpuLeaseError, GpuLeaseManager
from wmloop.process_lifecycle import LocalCommandSignals, LocalProcessTree, retire_process_group
from wmloop.storage import atomic_write, checked_path, sha256_file
from wmloop.contracts import ContractValidationError, validate_document


class ModelRunnerError(RuntimeError):
    """A bounded model run failed before producing authoritative evidence."""


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
_STAGE_OUTPUT_EXCERPT_CHARS = 2000
_TORCH_ELASTIC_DIAGNOSTIC_CHARS = 1200
_TORCH_ELASTIC_DIAGNOSTIC_FILES = 16


class _StageCancelled(Exception):
    def __init__(self, result: subprocess.CompletedProcess[str] | None = None) -> None:
        self.result = result


def execute_model_run(
    manifest: Mapping[str, object],
    *,
    lease_manager: GpuLeaseManager | None = None,
    process_runner: ProcessRunner | None = None,
    lease_wait_seconds: float = 0.0,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> dict[str, object]:
    """Run train then held-out evaluation under the manifest's GPU lease.

    The built-in lease manager observes cancellation while waiting for GPUs.
    Custom managers retain their acquisition signature and must implement their
    own interruptible waits; cancellation is checked again before stage launch.
    """
    signal_context = (
        LocalCommandSignals()
        if threading.current_thread() is threading.main_thread()
        else nullcontext()
    )
    with signal_context as signals:
        return _execute_model_run(
            manifest,
            lease_manager=lease_manager,
            process_runner=process_runner,
            lease_wait_seconds=lease_wait_seconds,
            cancel_requested=lambda: cancel_requested() or (signals is not None and signals.stop_requested),
        )


def _execute_model_run(
    manifest: Mapping[str, object],
    *,
    lease_manager: GpuLeaseManager | None,
    process_runner: ProcessRunner | None,
    lease_wait_seconds: float,
    cancel_requested: Callable[[], bool],
) -> dict[str, object]:

    try:
        validate_model_run(manifest)
    except ModelRunError as exc:
        raise ModelRunnerError(f"MODEL_RUN_MANIFEST_INVALID:{exc}") from exc
    try:
        preflight_model_adapter(manifest)
    except ModelLaunchAdapterError as exc:
        raise ModelRunnerError(str(exc)) from exc
    output = Path(str(manifest["output"]["root"])).resolve()  # type: ignore[index]
    if output.exists() and not output.is_dir():
        raise ModelRunnerError("MODEL_RUN_OUTPUT_NOT_FRESH")
    if not output.exists():
        output.mkdir(mode=0o700, parents=True)
    allocation = manifest["resource_binding"]["allocation"]  # type: ignore[index]
    indices = _lease_indices(allocation)  # type: ignore[arg-type]
    world_size = int(manifest["training"]["scale_plan"]["parallelism"]["world_size"])  # type: ignore[index]
    if len(indices) < world_size:
        raise ModelRunnerError("MODEL_RUN_RESOURCE_CAPACITY_INVALID")
    manager = lease_manager or GpuLeaseManager()
    started = time.monotonic()
    runtime = str(manifest["runtime"]["python"])  # type: ignore[index]
    train = _command(runtime, manifest["runtime"]["train_command"])  # type: ignore[index]
    evaluate = _command(runtime, manifest["evaluation"]["command"])  # type: ignore[index]
    protocol_version = manifest["adapter"]["protocol_version"]  # type: ignore[index]
    protocol_v2 = protocol_version == 2
    protocol_v3 = protocol_version == 3
    if protocol_v3:
        from wmloop.execute.model_run_protocol_v3 import (
            execute_protocol_v3,
            validate_agibot_support_at_execution,
        )

        validate_agibot_support_at_execution(manifest)
        return execute_protocol_v3(
            manifest,
            output=output,
            manager=manager,
            indices=indices,
            world_size=world_size,
            train=train,
            evaluate=evaluate,
            process_runner=process_runner,
            lease_wait_seconds=lease_wait_seconds,
            cancel_requested=cancel_requested,
        )
    existing = [path for path in output.iterdir() if path.name != "model-run.json"]
    if existing:
        raise ModelRunnerError("MODEL_RUN_OUTPUT_NOT_FRESH")
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    try:
        if cancel_requested():
            raise GpuLeaseCancelled("GPU_LEASE_CANCELLED")
        if type(manager) is gpu_lease.GpuLeaseManager:
            leases = manager.acquire_many(
                indices, world_size, wait_seconds=lease_wait_seconds,
                cancel_requested=cancel_requested,
            )
        else:
            # Legacy/custom managers retain their acquisition call contract.
            # Their own blocking calls must implement cooperative cancellation;
            # _run checks again before launching a stage after they return.
            leases = manager.acquire_many(
                indices, world_size, wait_seconds=lease_wait_seconds,
            )
    except GpuLeaseCancelled:
        return _write_cancelled_receipt(
            output, manifest, stages=[_stage_record("train", train, None),
                                      _stage_record("evaluate", evaluate, None)],
            leases=[], elapsed_seconds=time.monotonic() - started,
        )
    except GpuLeaseError as exc:
        raise ModelRunnerError(str(exc)) from exc
    stages: list[dict[str, object]] = []
    try:
        source = manifest["source"]  # type: ignore[index]
        generated_adapter = (
            isinstance(source, Mapping) and source.get("adapter_root") is not None
        )
        env = (
            _generated_adapter_environment(output)
            if generated_adapter
            else dict(os.environ)
        )
        env.update(_lease_environment(leases))
        cwd = Path(str(manifest["source"]["model_root"])).resolve()  # type: ignore[index]
        execution_policy = manifest.get("execution_policy", {})
        timeout_seconds = float(
            execution_policy.get("stage_timeout_seconds", 24.0 * 3600.0)
            if isinstance(execution_policy, Mapping)
            else 24.0 * 3600.0
        )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ModelRunnerError("MODEL_RUN_TIMEOUT_INVALID")
        try:
            for stage, command in (("train", train), ("evaluate", evaluate)):
                if stage == "evaluate" and protocol_v2:
                    if checkpoint is None or checkpoint_sha256 is None:
                        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_HANDOFF_MISSING")
                    command = _bind_trained_checkpoint(command, checkpoint)
                    env["VERDIWM_TRAINED_CHECKPOINT"] = str(checkpoint)
                command, child_log_root = _configure_torch_elastic_logging(
                    command, output=output, stage=stage
                )
                try:
                    stage_result = _run(
                        process_runner, command, cwd=cwd, env=env,
                        timeout_seconds=timeout_seconds, cancel_requested=cancel_requested,
                    )
                except subprocess.TimeoutExpired as exc:
                    exc = _append_torch_elastic_timeout_diagnostics(exc, child_log_root)
                    _write_timeout_receipt(
                        output, manifest, stage, timeout_seconds, exc,
                        completed_stages=stages, leases=leases,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    raise ModelRunnerError(f"MODEL_RUN_{stage.upper()}_TIMEOUT") from exc
                except _StageCancelled as exc:
                    stages.append(_stage_record(stage, command, exc.result))
                    raise
                stage_result = _append_torch_elastic_failure_diagnostics(
                    stage_result, child_log_root
                )
                stages.append(_stage_record(stage, command, stage_result))
                if stage_result.returncode != 0:
                    raise ModelRunnerError(f"MODEL_RUN_{stage.upper()}_FAILED")
                if stage == "train":
                    checkpoint_receipt_path = Path(str(manifest["training"]["checkpoint_receipt"]))  # type: ignore[index]
                    if protocol_v2:
                        checkpoint, checkpoint_sha256 = _validate_checkpoint_handoff(
                            checkpoint_receipt_path, manifest, output
                        )
                    else:
                        _receipt(checkpoint_receipt_path)
        except _StageCancelled:
            return _write_cancelled_receipt(
                output, manifest, stages=stages, leases=leases,
                elapsed_seconds=time.monotonic() - started,
            )
        evidence_path = Path(str(manifest["evaluation"]["evidence_receipt"]))  # type: ignore[index]
        evidence = _receipt(evidence_path)
        if protocol_v2:
            _validate_evaluation_handoff(
                evidence, manifest, checkpoint, checkpoint_sha256
            )
        result = {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-run-execution-receipt",
            "state": "completed",
            "model_run_id": manifest["model_run_id"],
            "checkpoint_receipt": str(Path(str(manifest["training"]["checkpoint_receipt"]))),  # type: ignore[index]
            **({"trained_checkpoint_path": str(checkpoint), "trained_checkpoint_sha256": checkpoint_sha256} if protocol_v2 else {}),
            "evidence_receipt": str(evidence),
            "stages": stages,
            "leased_gpus": [lease.to_document() for lease in leases],
            "elapsed_seconds": round(time.monotonic() - started, 6),
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
    matches = [
        row
        for row in roles
        if isinstance(row, Mapping)
        and row.get("role") == "autonomous_candidate_evaluation"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("gpu_indices"), list):
        raise ModelRunnerError("MODEL_RUN_RESOURCE_ROLE_INVALID")
    indices = matches[0]["gpu_indices"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in indices
    ):
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
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ModelRunnerError("MODEL_RUN_COMMAND_INVALID")
    return [runtime, *command]


def _run(
    runner: ProcessRunner | None,
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    cancel_requested: Callable[[], bool],
    on_launch_attempt: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    if cancel_requested():
        raise _StageCancelled()
    try:
        if runner is not None:
            if on_launch_attempt is not None:
                on_launch_attempt()
            result = runner(
                list(command), cwd=str(cwd), env=dict(env), capture_output=True,
                text=True, check=False, timeout=timeout_seconds,
            )
            if cancel_requested():
                raise _StageCancelled(result)
            return result
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout, tempfile.TemporaryFile(
            mode="w+", encoding="utf-8"
        ) as stderr:
            if on_launch_attempt is not None:
                on_launch_attempt()
            process = subprocess.Popen(
                list(command), cwd=str(cwd), env=dict(env), stdout=stdout,
                stderr=stderr, text=True, start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds
            cancelled = False
            timed_out = False
            tree = None
            try:
                tree = LocalProcessTree(process)
                while tree.poll() is None:
                    tree.observe()
                    if cancel_requested():
                        cancelled = True
                        tree.terminate(grace_seconds=5.0)
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            finally:
                retire_process_group(process, tree=tree)
            stdout.seek(0)
            stderr.seek(0)
            result = subprocess.CompletedProcess(list(command), process.returncode, stdout.read(), stderr.read())
            if timed_out:
                raise subprocess.TimeoutExpired(list(command), timeout_seconds, output=result.stdout, stderr=result.stderr)
            if cancelled or cancel_requested():
                raise _StageCancelled(result)
            return result
    except OSError as exc:
        raise ModelRunnerError("MODEL_RUN_PROCESS_START_FAILED") from exc


def _stage_record(
    stage: str, command: Sequence[str], result: subprocess.CompletedProcess[str] | None
) -> dict[str, object]:
    failed = result is not None and result.returncode != 0
    return {
        "stage": stage,
        "command": list(command),
        "returncode": int(result.returncode) if result is not None else None,
        **_stage_output_fields(
            result.stdout if result is not None else "",
            result.stderr if result is not None else "",
            include_head=failed,
        ),
        **({"state": "not_started"} if result is None else {}),
    }


def _configure_torch_elastic_logging(
    command: Sequence[str], *, output: Path, stage: str
) -> tuple[list[str], Path | None]:
    """Enable bounded, run-local worker logs for torch.distributed.run.

    Torch Elastic's parent process often only reports ``ChildFailedError`` and
    ``error_file=N/A``. Capturing rank stderr gives the receipt a useful
    diagnostic without requiring changes to the model trainer.
    """

    configured = list(command)
    launcher_index: int | None = None
    insertion_index: int | None = None
    for index, token in enumerate(configured):
        if token == "-m" and index + 1 < len(configured) and configured[index + 1] in {
            "torch.distributed.run", "torch.distributed.launch"
        }:
            launcher_index = index + 1
            insertion_index = index + 2
            break
        if Path(token).name.lower() in {"torchrun", "torchrun.exe"}:
            launcher_index = index
            insertion_index = index + 1
            break
    if launcher_index is None or insertion_index is None:
        return configured, None

    options = configured[:]
    separator = options.index("--") if "--" in options else len(options)
    for index in range(insertion_index, separator):
        token = options[index]
        if token == "--nproc_per_node" and index + 1 < separator:
            insertion_index = index + 2
            break
        if token.startswith("--nproc_per_node="):
            insertion_index = index + 1
            break
    launcher_options = options[launcher_index + 1:separator]

    def has_option(*names: str) -> bool:
        return any(
            token == name or token.startswith(name + "=")
            for token in launcher_options
            for name in names
        )

    log_root: Path | None = output / "logs" / "torch-elastic" / stage
    try:
        log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        # Launch remains useful if diagnostic storage is unavailable.
        log_root = None

    additions: list[str] = []
    log_dir_names = ("--log-dir", "--log_dir")
    if not has_option(*log_dir_names) and log_root is not None:
        additions.extend(("--log-dir", str(log_root)))
    if not has_option("--tee") and not has_option("--redirects"):
        additions.extend(("--tee", "3"))
    configured[insertion_index:insertion_index] = additions

    if log_root is None:
        return configured, None
    # Respect an explicit log directory only when it remains under this run's
    # output root; otherwise do not inspect user-selected external files.
    if has_option(*log_dir_names):
        try:
            option_index = next(
                i for i, token in enumerate(launcher_options)
                if any(token == name or token.startswith(name + "=") for name in log_dir_names)
            )
            value = launcher_options[option_index]
            if value in log_dir_names and option_index + 1 < len(launcher_options):
                value = launcher_options[option_index + 1]
            elif any(value.startswith(name + "=") for name in log_dir_names):
                value = value.split("=", 1)[1]
            configured_root = Path(value).expanduser().resolve()
            output_root = output.resolve()
            if configured_root.is_relative_to(output_root):
                return configured, configured_root
        except (OSError, StopIteration, ValueError):
            pass
        return configured, None
    return configured, log_root


def _torch_elastic_failure_summaries(log_root: Path | None) -> list[str]:
    """Read only a small, rank-local tail from generated torch Elastic logs."""

    if log_root is None:
        return []
    try:
        root = log_root.resolve(strict=True)
        if not root.is_dir():
            return []
        candidates: list[Path] = []
        for path in root.rglob("*"):
            if len(candidates) >= _TORCH_ELASTIC_DIAGNOSTIC_FILES:
                break
            if path.name == "error.json" or path.name in {"stderr.log", "stderr"}:
                try:
                    if path.is_file() and path.resolve(strict=True).is_relative_to(root):
                        candidates.append(path)
                except OSError:
                    continue
        summaries: list[str] = []
        for path in sorted(candidates, key=lambda value: (value.name != "error.json", str(value))):
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16_000))
                    raw = stream.read(16_000)
                content = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            if path.name == "error.json":
                try:
                    document = json.loads(content)
                    values: list[str] = []

                    def collect(value: object) -> None:
                        if isinstance(value, str):
                            values.append(value)
                        elif isinstance(value, Mapping):
                            for nested in value.values():
                                collect(nested)
                        elif isinstance(value, list):
                            for nested in value:
                                collect(nested)

                    collect(document)
                    content = "\n".join(values)
                except json.JSONDecodeError:
                    pass
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            exception_lines = [
                line for line in lines
                if re.search(r"(?:[A-Za-z_][\w.]*Error|Exception|AssertionError|SystemExit):\s*\S", line)
            ]
            useful = exception_lines[-1] if exception_lines else next(
                (line for line in reversed(lines) if "error_file=N/A" not in line and "ChildFailedError" not in line),
                "",
            )
            if useful:
                rank = next((part for part in path.parts if part.isdigit()), None)
                label = f"rank {rank}" if rank is not None else "worker"
                summary = f"Torch Elastic {label}: {useful}"[:_TORCH_ELASTIC_DIAGNOSTIC_CHARS]
                if summary not in summaries:
                    summaries.append(summary)
        return summaries
    except OSError:
        return []


def _append_torch_elastic_failure_diagnostics(
    result: subprocess.CompletedProcess[str], log_root: Path | None
) -> subprocess.CompletedProcess[str]:
    if result.returncode == 0:
        return result
    summaries = _torch_elastic_failure_summaries(log_root)
    if not summaries:
        return result
    stderr = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr or "")
    stderr = (stderr + "\n" + "\n".join(summaries))[-_STAGE_OUTPUT_EXCERPT_CHARS:]
    return subprocess.CompletedProcess(result.args, result.returncode, result.stdout, stderr)


def _append_torch_elastic_timeout_diagnostics(
    error: subprocess.TimeoutExpired, log_root: Path | None
) -> subprocess.TimeoutExpired:
    summaries = _torch_elastic_failure_summaries(log_root)
    if not summaries:
        return error
    stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else str(error.stderr or "")
    stderr = (stderr + "\n" + "\n".join(summaries))[-_STAGE_OUTPUT_EXCERPT_CHARS:]
    return subprocess.TimeoutExpired(error.cmd, error.timeout, output=error.output, stderr=stderr)


def _stage_output_fields(
    stdout: object, stderr: object, *, include_head: bool
) -> dict[str, str]:
    """Return bounded stage output excerpts, retaining the legacy tail fields."""

    def text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")

    stdout_text = text(stdout)
    stderr_text = text(stderr)
    fields = {
        "stdout_tail": stdout_text[-_STAGE_OUTPUT_EXCERPT_CHARS:],
        "stderr_tail": stderr_text[-_STAGE_OUTPUT_EXCERPT_CHARS:],
    }
    if include_head:
        fields["stdout_head"] = stdout_text[:_STAGE_OUTPUT_EXCERPT_CHARS]
        fields["stderr_head"] = stderr_text[:_STAGE_OUTPUT_EXCERPT_CHARS]
    return fields


def _receipt(path: Path) -> Path:
    try:
        resolved = checked_path(path, code="MODEL_RUN_RECEIPT_INVALID", error=ModelRunnerError)
    except ModelRunnerError:
        raise
    if not resolved.is_file():
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID") from exc
    if not isinstance(value, Mapping):
        raise ModelRunnerError("MODEL_RUN_RECEIPT_INVALID")
    return resolved


def _validate_checkpoint_handoff(
    receipt_path: Path, manifest: Mapping[str, object], output_root: Path
) -> tuple[Path, str]:
    receipt_file = _receipt(receipt_path)
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise ModelRunnerError("MODEL_RUN_CHECKPOINT_RECEIPT_INVALID")
        validate_document(
            "model_checkpoint_receipt",
            receipt,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_RECEIPT_INVALID") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_RECEIPT_INVALID") from exc
    training = manifest["training"]  # type: ignore[index]
    plan = training["scale_plan"]
    replication = plan["replication"]
    dataset = plan["dataset"]
    expected = {
        "model_run_id": manifest["model_run_id"],
        "scale_plan_digest": training["scale_plan_digest"],
        "planned_steps": plan["updates"]["planned_steps"],
        "seed": replication["seeds"][0],
        "support_record_ids": dataset["selected_train_record_ids"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_RECEIPT_BINDING_MISMATCH")
    output = checked_path(output_root, code="MODEL_RUN_OUTPUT_INVALID", error=ModelRunnerError)
    raw_checkpoint = Path(str(receipt["checkpoint_path"]))
    checkpoint = checked_path(raw_checkpoint, code="MODEL_RUN_CHECKPOINT_INVALID", error=ModelRunnerError)
    try:
        checkpoint.relative_to(output)
    except ValueError as exc:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_OUTSIDE_OUTPUT") from exc
    if not checkpoint.is_file():
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_INVALID")
    actual_digest = sha256_file(checkpoint)
    if receipt.get("checkpoint_sha256") != actual_digest:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_DIGEST_MISMATCH")
    return checkpoint, actual_digest














def _bind_trained_checkpoint(command: Sequence[str], checkpoint: Path) -> list[str]:
    if command.count("{trained_checkpoint}") != 1:
        raise ModelRunnerError("MODEL_RUN_TRAINED_CHECKPOINT_PLACEHOLDER_REQUIRED")
    return [str(checkpoint) if token == "{trained_checkpoint}" else token for token in command]


def _validate_evaluation_handoff(
    evidence_path: Path,
    manifest: Mapping[str, object],
    checkpoint: Path | None,
    checkpoint_sha256: str | None,
) -> None:
    if checkpoint is None or checkpoint_sha256 is None:
        raise ModelRunnerError("MODEL_RUN_CHECKPOINT_HANDOFF_MISSING")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, Mapping):
            raise ModelRunnerError("MODEL_RUN_EVALUATION_RECEIPT_INVALID")
        validate_document(
            "model_evaluation_receipt",
            evidence,
            root=Path(__file__).resolve().parents[2],
        )
    except ContractValidationError as exc:
        raise ModelRunnerError("MODEL_RUN_EVALUATION_RECEIPT_INVALID") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRunnerError("MODEL_RUN_EVALUATION_RECEIPT_INVALID") from exc
    if (
        evidence.get("model_run_id") != manifest.get("model_run_id")
        or checked_path(Path(str(evidence.get("checkpoint_path"))), code="MODEL_RUN_EVALUATION_CHECKPOINT_INVALID", error=ModelRunnerError) != checkpoint
        or evidence.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ModelRunnerError("MODEL_RUN_EVALUATION_CHECKPOINT_MISMATCH")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_write(path, (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2) + "\n").encode("utf-8"))


def _write_cancelled_receipt(
    output: Path,
    manifest: Mapping[str, object],
    *,
    stages: Sequence[Mapping[str, object]],
    leases: Sequence[GpuLease],
    elapsed_seconds: float,
) -> dict[str, object]:
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-run-execution-receipt",
        "state": "cancelled",
        "cancelled": True,
        "model_run_id": manifest["model_run_id"],
        "stages": list(stages),
        "leased_gpus": [lease.to_document() for lease in leases],
        "elapsed_seconds": round(elapsed_seconds, 6),
        "claim_boundary": "Cancellation provenance only; elapsed work is retained and no interrupted output is accepted as scientific evidence or settled cost.",
    }
    _write_json(output / "execution-receipt.json", result)
    return result


def _write_timeout_receipt(
    output: Path,
    manifest: Mapping[str, object],
    stage: str,
    timeout_seconds: float,
    error: subprocess.TimeoutExpired,
    *,
    completed_stages: Sequence[Mapping[str, object]],
    leases: Sequence[GpuLease],
    elapsed_seconds: float,
) -> None:
    policy = manifest.get("execution_policy", {})
    name = (
        str(policy.get("timeout_receipt", "execution-timeout-receipt.json"))
        if isinstance(policy, Mapping)
        else "execution-timeout-receipt.json"
    )
    _write_json(
        output / name,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-run-timeout-receipt",
            "state": "timed_out",
            "model_run_id": manifest.get("model_run_id"),
            "stage": stage,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "completed_stages": list(completed_stages),
            "leased_gpus": [lease.to_document() for lease in leases],
            "command": (
                list(error.cmd)
                if isinstance(error.cmd, (list, tuple))
                else str(error.cmd)
            ),
            "claim_boundary": "The stage exceeded its wall-clock budget; no scientific evidence was accepted.",
        },
    )
