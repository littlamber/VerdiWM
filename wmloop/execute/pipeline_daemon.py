"""Durably run or resume one complete diagnostic-first pipeline.

The pipeline runner already persists every completed stage.  This daemon adds
bounded retries around that transaction so GPU contention during the probe or
candidate stages does not require an operator to relaunch it.  Resource
deferrals are recorded separately and never consume failure attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.autonomous_pipeline import (
    AutonomousPipelineOptions,
    run_autonomous_pipeline,
)
from wmloop.execute.gpu_lease import GpuLeaseError


class PipelineDaemonError(RuntimeError):
    """A pipeline daemon could not preserve its durable execution contract."""


PipelineRunner = Callable[[AutonomousPipelineOptions], Mapping[str, object]]
SleepFunction = Callable[[float], None]


@dataclass(frozen=True)
class PipelineDaemonOptions:
    """Configuration for one resumable full-pipeline daemon."""

    pipeline: AutonomousPipelineOptions
    state_root: Path
    poll_seconds: float = 60.0
    max_cycles: int = 1_440
    max_attempts: int = 3


def run_pipeline_daemon(
    options: PipelineDaemonOptions,
    *,
    pipeline_runner: PipelineRunner = run_autonomous_pipeline,
    sleeper: SleepFunction = time.sleep,
) -> dict[str, object]:
    """Run a pipeline until it settles, blocks, stops, or exhausts cycles."""

    _validate_options(options)
    destination = Path(options.state_root).expanduser().resolve()
    input_document = _input_document(
        options.pipeline,
        max_attempts=options.max_attempts,
    )
    input_hash = _sha256(_canonical_json(input_document))
    _bind_output(
        destination,
        input_document=input_document,
        input_hash=input_hash,
    )
    _acquire_daemon_lock(destination)
    stop_requested = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers: dict[int, Any] = {}
    state: dict[str, object] | None = None
    try:
        previous_handlers = _install_signal_handlers(request_stop)
        state = _load_state(
            destination,
            input_hash=input_hash,
            options=options,
        )
        if state.get("state") in {"completed", "blocked"}:
            return _manifest(state)
        state["state"] = "running"
        state["max_cycles"] = options.max_cycles
        state["poll_seconds"] = options.poll_seconds
        state["max_attempts"] = options.max_attempts
        _write_json(destination / "status.json", state)

        for cycle in range(int(state["cycle"]) + 1, options.max_cycles + 1):
            if stop_requested:
                state["state"] = "stopped"
                break
            cycle_record: dict[str, object] = {
                "cycle": cycle,
                "started_at": _utc_now(),
            }
            state["attempt_count"] = int(state["attempt_count"]) + 1
            try:
                pipeline_manifest = dict(pipeline_runner(options.pipeline))
                outcome = _record_pipeline_result(
                    state,
                    pipeline_manifest=pipeline_manifest,
                    pipeline_output=Path(options.pipeline.output_root).resolve(),
                )
                cycle_record.update(outcome)
            except Exception as exc:
                if _is_resource_deferral(exc):
                    state["deferral_count"] = int(state["deferral_count"]) + 1
                    state["last_outcome"] = "deferred"
                    state["last_deferral"] = str(exc)[:500]
                    state["last_error"] = None
                    cycle_record.update(
                        {
                            "outcome": "deferred",
                            "reason": str(exc)[:500],
                        }
                    )
                else:
                    state["error_count"] = int(state["error_count"]) + 1
                    error = {
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                    state["last_outcome"] = "error"
                    state["last_error"] = error
                    cycle_record.update({"outcome": "error", "error": error})
                    if int(state["error_count"]) >= options.max_attempts:
                        state["state"] = "blocked"
            state["cycle"] = cycle
            cycle_record["finished_at"] = _utc_now()
            _write_json(
                destination / "cycles" / f"cycle-{cycle:06d}.json",
                cycle_record,
            )
            if stop_requested and state.get("state") == "running":
                state["state"] = "stopped"
            _write_json(destination / "status.json", state)
            if state.get("state") in {"completed", "blocked", "stopped"}:
                break
            if cycle < options.max_cycles:
                sleeper(options.poll_seconds)
        else:
            if state.get("state") == "running":
                state["state"] = "exhausted"

        if state.get("state") == "running":
            state["state"] = "stopped" if stop_requested else "exhausted"
        _write_json(destination / "status.json", state)
        return _manifest(state)
    except Exception as exc:
        if state is not None:
            state["state"] = "interrupted"
            state["last_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            _write_json(destination / "status.json", state)
        raise
    finally:
        _restore_signal_handlers(previous_handlers)
        _release_daemon_lock(destination)


def _validate_options(options: PipelineDaemonOptions) -> None:
    if (
        options.max_cycles < 1
        or options.max_attempts < 1
        or not math.isfinite(options.poll_seconds)
        or options.poll_seconds < 0
    ):
        raise PipelineDaemonError("PIPELINE_DAEMON_ARGUMENT_INVALID")
    pipeline = options.pipeline
    state_root = Path(options.state_root).expanduser().resolve()
    repo = Path(pipeline.repo_root).expanduser().resolve()
    evaluator = Path(pipeline.evaluator_contract).expanduser().resolve()
    pipeline_output = Path(pipeline.output_root).expanduser().resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise PipelineDaemonError("PIPELINE_DAEMON_REPOSITORY_INVALID")
    if not evaluator.is_file() or evaluator.is_symlink():
        raise PipelineDaemonError("PIPELINE_DAEMON_EVALUATOR_INVALID")
    if pipeline.probe_contract is not None:
        probe = Path(pipeline.probe_contract).expanduser().resolve()
        if not probe.is_file() or probe.is_symlink():
            raise PipelineDaemonError("PIPELINE_DAEMON_PROBE_CONTRACT_INVALID")

    protected = [repo, evaluator, pipeline_output]
    optional_paths = (
        pipeline.runtime_python,
        pipeline.archive_db,
        pipeline.cas_root,
        pipeline.lock_root,
        pipeline.budget_db,
        pipeline.probe_contract,
        pipeline.retrieval_db,
    )
    protected.extend(
        Path(path).expanduser().resolve()
        for path in optional_paths
        if path is not None
    )
    protected.extend(
        Path(path).expanduser().resolve() for _, path in pipeline.asset_bindings
    )
    if any(_paths_overlap(state_root, path) for path in protected):
        raise PipelineDaemonError("PIPELINE_DAEMON_STATE_OVERLAP_INVALID")


def _input_document(
    options: AutonomousPipelineOptions,
    *,
    max_attempts: int,
) -> dict[str, object]:
    evaluator = Path(options.evaluator_contract).expanduser().resolve(strict=True)
    probe = (
        Path(options.probe_contract).expanduser().resolve(strict=True)
        if options.probe_contract is not None
        else None
    )
    assets = [
        {
            "parameter": parameter,
            "path": str(Path(path).expanduser().resolve()),
        }
        for parameter, path in options.asset_bindings
    ]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-pipeline-daemon-input",
        "repo_root": str(Path(options.repo_root).expanduser().resolve()),
        "pipeline_output_root": str(
            Path(options.output_root).expanduser().resolve()
        ),
        "evaluator_contract": str(evaluator),
        "evaluator_sha256": _sha256(evaluator.read_bytes()),
        "runtime_python": (
            str(Path(options.runtime_python).expanduser().absolute())
            if options.runtime_python is not None
            else None
        ),
        "asset_bindings": sorted(
            assets,
            key=lambda row: (row["parameter"], row["path"]),
        ),
        "probe_imports": options.probe_imports,
        "max_files": options.max_files,
        "conformance_timeout_seconds": options.conformance_timeout_seconds,
        "archive_db": _optional_path(options.archive_db),
        "cas_root": _optional_path(options.cas_root),
        "lock_root": str(Path(options.lock_root).expanduser().resolve()),
        "budget_db": _optional_path(options.budget_db),
        "budget_total_gpu_hours": options.budget_total_gpu_hours,
        **_nondefault_resource_policy(options),
        "probe_contract": str(probe) if probe is not None else None,
        "probe_contract_sha256": (
            _sha256(probe.read_bytes()) if probe is not None else None
        ),
        "retrieval_db": _optional_path(options.retrieval_db),
        "literature_query": options.literature_query,
        "literature_max_results": options.literature_max_results,
        "literature_timeout_seconds": options.literature_timeout_seconds,
        "max_attempts": max_attempts,
    }


def _nondefault_resource_policy(
    options: AutonomousPipelineOptions,
) -> dict[str, object]:
    if (
        options.budget_max_trial_gpu_hours == 120.0
        and options.budget_high_trial_limit == 2
        and options.budget_require_high_cost_approval
    ):
        return {}
    return {
        "resource_policy": {
            "max_trial_gpu_hours": options.budget_max_trial_gpu_hours,
            "high_trial_limit": options.budget_high_trial_limit,
            "require_high_cost_approval": (
                options.budget_require_high_cost_approval
            ),
        }
    }


def _optional_path(path: Path | None) -> str | None:
    return str(Path(path).expanduser().resolve()) if path is not None else None


def _bind_output(
    destination: Path,
    *,
    input_document: Mapping[str, object],
    input_hash: str,
) -> None:
    lock_path = destination / "input.lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise PipelineDaemonError("PIPELINE_DAEMON_STATE_ROOT_INVALID")
        lock = _load_json(lock_path)
        if lock.get("input_hash") != input_hash or lock.get("input") != input_document:
            raise PipelineDaemonError("PIPELINE_DAEMON_INPUT_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True)
    (destination / "cycles").mkdir(mode=0o700)
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-pipeline-daemon-input-lock",
            "input_hash": input_hash,
            "input": dict(input_document),
        },
    )


def _load_state(
    destination: Path,
    *,
    input_hash: str,
    options: PipelineDaemonOptions,
) -> dict[str, object]:
    path = destination / "status.json"
    if path.is_file() and not path.is_symlink():
        state = _load_json(path)
        if state.get("input_hash") != input_hash:
            raise PipelineDaemonError("PIPELINE_DAEMON_STATUS_INPUT_MISMATCH")
        return state
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-pipeline-daemon-manifest",
        "state": "running",
        "input_hash": input_hash,
        "pipeline_output_root": str(Path(options.pipeline.output_root).resolve()),
        "daemon_state_root": str(destination),
        "cycle": 0,
        "max_cycles": options.max_cycles,
        "poll_seconds": options.poll_seconds,
        "max_attempts": options.max_attempts,
        "attempt_count": 0,
        "error_count": 0,
        "deferral_count": 0,
        "last_outcome": None,
        "last_error": None,
        "last_deferral": None,
        "pipeline_manifest_path": None,
        "pipeline_state": None,
        "pipeline_verdict": None,
        "blocked_stage": None,
        "claim_boundary": (
            "The daemon retries one immutable, bounded pipeline; GPU capacity "
            "deferrals are not experiment failures or reusable evidence."
        ),
    }
    _write_json(path, state)
    return state


def _record_pipeline_result(
    state: dict[str, object],
    *,
    pipeline_manifest: Mapping[str, object],
    pipeline_output: Path,
) -> dict[str, object]:
    pipeline_state = pipeline_manifest.get("state")
    verdict = pipeline_manifest.get("verdict")
    if not isinstance(pipeline_state, str) or verdict not in {"PASS", "BLOCKED"}:
        raise PipelineDaemonError("PIPELINE_DAEMON_PIPELINE_RESULT_INVALID")
    outcome = "completed" if verdict == "PASS" else "blocked"
    state["state"] = outcome
    state["last_outcome"] = outcome
    state["last_error"] = None
    state["pipeline_manifest_path"] = str(
        pipeline_output / "pipeline-manifest.json"
    )
    state["pipeline_state"] = pipeline_state
    state["pipeline_verdict"] = verdict
    blocked_stage = pipeline_manifest.get("blocked_stage")
    state["blocked_stage"] = (
        blocked_stage if isinstance(blocked_stage, str) else None
    )
    return {
        "outcome": outcome,
        "pipeline_state": pipeline_state,
        "pipeline_verdict": verdict,
        "blocked_stage": state["blocked_stage"],
    }


def _is_resource_deferral(error: Exception) -> bool:
    return isinstance(error, GpuLeaseError) and str(error).startswith(
        "GPU_LEASE_UNAVAILABLE"
    )


def _acquire_daemon_lock(destination: Path) -> None:
    lock_path = destination / "daemon.lock"
    payload = json.dumps(
        {"pid": os.getpid(), "created_at": _utc_now()}, sort_keys=True
    ) + "\n"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _load_json(lock_path)
        pid = existing.get("pid")
        if isinstance(pid, int) and pid > 0 and _pid_alive(pid):
            raise PipelineDaemonError("PIPELINE_DAEMON_ALREADY_RUNNING")
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise PipelineDaemonError("PIPELINE_DAEMON_ALREADY_RUNNING") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _release_daemon_lock(destination: Path) -> None:
    path = destination / "daemon.lock"
    try:
        lock = _load_json(path)
        if lock.get("pid") == os.getpid():
            path.unlink()
    except (OSError, PipelineDaemonError):
        return


def _manifest(state: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(state)
    try:
        validate_document("pipeline_daemon_manifest", manifest)
    except ContractValidationError as exc:
        raise PipelineDaemonError("PIPELINE_DAEMON_MANIFEST_INVALID") from exc
    return manifest


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineDaemonError("PIPELINE_DAEMON_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise PipelineDaemonError("PIPELINE_DAEMON_JSON_INVALID")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _install_signal_handlers(handler: Callable[[int, object], None]) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_asset(value: str) -> tuple[str, Path]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise PipelineDaemonError("PIPELINE_DAEMON_ASSET_ARGUMENT_INVALID")
    return parameter.strip(), Path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--daemon-state-root", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--asset", action="append", default=[], metavar="PARAM=PATH")
    parser.add_argument("--no-import-probe", action="store_true")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--conformance-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    parser.add_argument(
        "--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases")
    )
    parser.add_argument("--budget-db", type=Path)
    parser.add_argument("--budget-total-gpu-hours", type=float)
    parser.add_argument("--budget-max-trial-gpu-hours", type=float, default=120.0)
    parser.add_argument("--budget-high-trial-limit", type=int, default=2)
    parser.add_argument("--auto-approve-high-cost", action="store_true")
    parser.add_argument("--probe-contract", type=Path)
    parser.add_argument("--retrieval-db", type=Path)
    parser.add_argument("--literature-query")
    parser.add_argument("--literature-max-results", type=int, default=8)
    parser.add_argument("--literature-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-cycles", type=int, default=1_440)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        manifest = run_pipeline_daemon(
            PipelineDaemonOptions(
                pipeline=AutonomousPipelineOptions(
                    repo_root=args.repo_root,
                    output_root=args.output_root,
                    evaluator_contract=args.evaluator_contract,
                    runtime_python=args.runtime_python,
                    asset_bindings=tuple(_parse_asset(value) for value in args.asset),
                    probe_imports=not args.no_import_probe,
                    max_files=args.max_files,
                    conformance_timeout_seconds=args.conformance_timeout_seconds,
                    archive_db=args.archive_db,
                    cas_root=args.cas_root,
                    lock_root=args.lock_root,
                    budget_db=args.budget_db,
                    budget_total_gpu_hours=args.budget_total_gpu_hours,
                    budget_max_trial_gpu_hours=args.budget_max_trial_gpu_hours,
                    budget_high_trial_limit=args.budget_high_trial_limit,
                    budget_require_high_cost_approval=(
                        not args.auto_approve_high_cost
                    ),
                    probe_contract=args.probe_contract,
                    retrieval_db=args.retrieval_db,
                    literature_query=args.literature_query,
                    literature_max_results=args.literature_max_results,
                    literature_timeout_seconds=args.literature_timeout_seconds,
                ),
                state_root=args.daemon_state_root,
                poll_seconds=args.poll_seconds,
                max_cycles=args.max_cycles,
                max_attempts=args.max_attempts,
            )
        )
    except PipelineDaemonError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0 if manifest.get("state") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
