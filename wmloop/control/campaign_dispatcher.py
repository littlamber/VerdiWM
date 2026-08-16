"""Consume Campaign API dispatch manifests through existing VerdiWM daemons."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore


class CampaignDispatchError(RuntimeError):
    """A dispatch manifest could not be admitted or settled."""


class CampaignProcessError(CampaignDispatchError):
    """A child process failed after producing a bounded execution receipt."""

    def __init__(self, code: str, result: Mapping[str, Any]):
        super().__init__(code)
        self.result = dict(result)


Runner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class DispatcherOptions:
    state_root: Path
    poll_seconds: float = 10.0
    max_cycles: int = 1
    max_parallel: int = 1
    campaign_ids: tuple[str, ...] = ()


def run_dispatcher(
    options: DispatcherOptions,
    *,
    runner: Runner | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if (
        options.max_cycles < 1
        or options.max_parallel < 1
        or options.poll_seconds < 0
        or any(not campaign_id for campaign_id in options.campaign_ids)
    ):
        raise CampaignDispatchError("DISPATCHER_OPTIONS_INVALID")
    root = Path(options.state_root).expanduser().resolve()
    store = CampaignStore(root)
    dispatch_root = store.root / "dispatch"
    pending = dispatch_root / "pending"
    running = dispatch_root / "running"
    completed = dispatch_root / "completed"
    failed = dispatch_root / "failed"
    cancelled = dispatch_root / "cancelled"
    for path in (pending, running, completed, failed, cancelled):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = dispatch_root / "dispatcher.lock"
    _acquire_lock(lock_path)
    settled: list[str] = []
    failed_ids: list[str] = []
    cancelled_ids: list[str] = []
    try:
        recovered = _recover_interrupted(
            store=store,
            running=running,
            failed=failed,
            cancelled=cancelled,
        )
        failed_ids.extend(recovered["failed"])
        cancelled_ids.extend(recovered["cancelled"])
        selected_ids = set(options.campaign_ids)
        for cycle in range(options.max_cycles):
            paths = sorted(
                path
                for path in pending.glob("*.json")
                if path.is_file()
                and not path.is_symlink()
                and (not selected_ids or path.stem in selected_ids)
            )
            if not paths:
                if cycle + 1 < options.max_cycles:
                    sleeper(options.poll_seconds)
                continue
            batch = paths[: options.max_parallel]
            with ThreadPoolExecutor(max_workers=options.max_parallel) as executor:
                outcomes = list(
                    executor.map(
                        lambda source: _dispatch_one(
                            source=source,
                            store=store,
                            running=running,
                            completed=completed,
                            failed=failed,
                            cancelled=cancelled,
                            runner=runner,
                        ),
                        batch,
                    )
                )
            for state, campaign_id in outcomes:
                if state == "completed":
                    settled.append(campaign_id)
                elif state == "cancelled":
                    cancelled_ids.append(campaign_id)
                else:
                    failed_ids.append(campaign_id)
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-campaign-dispatcher-manifest",
            "state": "completed",
            "settled_campaign_ids": settled,
            "failed_campaign_ids": failed_ids,
            "cancelled_campaign_ids": cancelled_ids,
            "pending_count": len(list(pending.glob("*.json"))),
        }
    finally:
        lock_path.unlink(missing_ok=True)


def _dispatch_one(
    *,
    source: Path,
    store: CampaignStore,
    running: Path,
    completed: Path,
    failed: Path,
    cancelled: Path,
    runner: Runner | None,
) -> tuple[str, str]:
    campaign_id = source.stem
    active = running / source.name
    dispatch: dict[str, Any] = {
        "campaign_id": campaign_id,
        "schema_version": 1,
    }
    try:
        if store.get(campaign_id).get("status") == "cancelled":
            try:
                dispatch = _load_dispatch(source)
            except CampaignDispatchError:
                dispatch = {"campaign_id": campaign_id, "schema_version": 1}
            dispatch["state"] = "cancelled"
            _write_json(cancelled / source.name, dispatch)
            source.unlink(missing_ok=True)
            store.record_dispatch_location(campaign_id, cancelled / source.name)
            return "cancelled", campaign_id
        source.replace(active)
        dispatch = _load_dispatch(active)
        store.record_dispatch_result(campaign_id, status="running")
        store.record_dispatch_location(campaign_id, active)
        if runner is None:
            result = dict(
                _run_subprocess(
                    dispatch["execution"],
                    cancel_requested=lambda: _cancel_requested(store, campaign_id),
                )
            )
        else:
            result = dict(runner(dispatch["execution"]))
        if result.get("cancelled") or _cancel_requested(store, campaign_id):
            if not _cancel_requested(store, campaign_id):
                store.cancel(campaign_id)
            dispatch["state"] = "cancelled"
            dispatch["result"] = result
            dispatch["cancelled_at"] = store.get(campaign_id).get(
                "cancellation_requested_at"
            )
            _write_json(cancelled / source.name, dispatch)
            active.unlink(missing_ok=True)
            store.record_dispatch_location(campaign_id, cancelled / source.name)
            return "cancelled", campaign_id
        store.record_dispatch_result(campaign_id, status="completed", result=result)
        dispatch["state"] = "completed"
        dispatch["result"] = result
        _write_json(completed / source.name, dispatch)
        active.unlink(missing_ok=True)
        store.record_dispatch_location(campaign_id, completed / source.name)
        return "completed", campaign_id
    except Exception as exc:
        if _cancel_requested(store, campaign_id):
            dispatch["state"] = "cancelled"
            dispatch["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            if isinstance(exc, CampaignProcessError):
                dispatch["result"] = exc.result
            _write_json(cancelled / source.name, dispatch)
            active.unlink(missing_ok=True)
            source.unlink(missing_ok=True)
            store.record_dispatch_location(campaign_id, cancelled / source.name)
            return "cancelled", campaign_id
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        try:
            store.record_dispatch_result(campaign_id, status="failed", error=error)
        except CampaignAPIError:
            pass
        if dispatch.get("artifact_type") != "verdiwm-campaign-dispatch":
            try:
                dispatch = _load_dispatch(active)
            except Exception:
                dispatch = {"campaign_id": campaign_id, "schema_version": 1}
        dispatch["state"] = "failed"
        dispatch["error"] = error
        if isinstance(exc, CampaignProcessError):
            dispatch["result"] = exc.result
        _write_json(failed / source.name, dispatch)
        active.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
        try:
            store.record_dispatch_location(campaign_id, failed / source.name)
        except CampaignAPIError:
            pass
        return "failed", campaign_id


def _cancel_requested(store: CampaignStore, campaign_id: str) -> bool:
    try:
        return store.get(campaign_id).get("status") == "cancelled"
    except CampaignAPIError:
        return False


def _recover_interrupted(
    *, store: CampaignStore, running: Path, failed: Path, cancelled: Path
) -> dict[str, list[str]]:
    recovered: dict[str, list[str]] = {"failed": [], "cancelled": []}
    for active in sorted(running.glob("*.json")):
        if active.is_symlink() or not active.is_file():
            continue
        campaign_id = active.stem
        if _cancel_requested(store, campaign_id):
            try:
                dispatch = json.loads(active.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                dispatch = {"campaign_id": campaign_id, "schema_version": 1}
            dispatch["state"] = "cancelled"
            dispatch["error"] = {
                "type": "DISPATCH_CANCELLED_DURING_INTERRUPTION",
                "message": "The prior dispatcher stopped while cancellation was pending.",
            }
            _write_json(cancelled / active.name, dispatch)
            active.unlink(missing_ok=True)
            store.record_dispatch_location(campaign_id, cancelled / active.name)
            recovered["cancelled"].append(campaign_id)
            continue
        error = {
            "type": "DISPATCH_INTERRUPTED",
            "message": "The previous dispatcher stopped after claiming this campaign. It was not relaunched automatically to prevent duplicate execution.",
        }
        try:
            store.record_dispatch_result(campaign_id, status="failed", error=error)
        except CampaignAPIError:
            pass
        try:
            dispatch = json.loads(active.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dispatch = {"campaign_id": campaign_id, "schema_version": 1}
        dispatch["state"] = "failed"
        dispatch["error"] = error
        _write_json(failed / active.name, dispatch)
        active.unlink(missing_ok=True)
        try:
            store.record_dispatch_location(campaign_id, failed / active.name)
        except CampaignAPIError:
            pass
        recovered["failed"].append(campaign_id)
    return recovered


def _acquire_lock(path: Path) -> None:
    payload = json.dumps({"pid": os.getpid()}, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
            pid = int(lock.get("pid", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            raise CampaignDispatchError("DISPATCHER_ALREADY_RUNNING")
        path.unlink(missing_ok=True)
        return _acquire_lock(path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_dispatch(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignDispatchError("DISPATCH_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "verdiwm-campaign-dispatch":
        raise CampaignDispatchError("DISPATCH_MANIFEST_CONTRACT_INVALID")
    if payload.get("state") != "pending" or not isinstance(payload.get("execution"), dict):
        raise CampaignDispatchError("DISPATCH_MANIFEST_NOT_PENDING")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_command(execution: Mapping[str, Any]) -> list[str]:
    kind = execution.get("kind")
    if kind == "pipeline":
        command = [
            sys.executable,
            "-m",
            "wmloop.execute.autonomous_pipeline",
            str(execution["repo_root"]),
            "--output-root",
            str(execution["output_root"]),
            "--evaluator-contract",
            str(execution["evaluator_contract"]),
            "--budget-total-gpu-hours",
            str(execution["budget_total_gpu_hours"]),
        ]
        _append_options(
            command,
            execution,
            {
                "runtime_python": "--runtime-python",
                "archive_db": "--archive-db",
                "cas_root": "--cas-root",
                "lock_root": "--lock-root",
                "budget_db": "--budget-db",
                "probe_contract": "--probe-contract",
                "retrieval_db": "--retrieval-db",
                "literature_query": "--literature-query",
                "literature_max_results": "--literature-max-results",
                "literature_timeout_seconds": "--literature-timeout-seconds",
                "candidate_catalog": "--candidate-catalog",
                "settlement_manifest": "--settlement-manifest",
                "max_files": "--max-files",
                "conformance_timeout_seconds": "--conformance-timeout-seconds",
                "budget_max_trial_gpu_hours": "--budget-max-trial-gpu-hours",
                "budget_high_trial_limit": "--budget-high-trial-limit",
            },
        )
        _append_assets(command, execution.get("asset_bindings"))
        if execution.get("probe_imports") is False:
            command.append("--no-import-probe")
        if execution.get("budget_require_high_cost_approval") is False:
            command.append("--auto-approve-high-cost")
    elif kind == "evolution":
        command = [
            sys.executable,
            "-m",
            "wmloop.execute.evolution_daemon",
            str(execution["repo_root"]),
            "--output-root",
            str(execution["output_root"]),
            "--state-root",
            str(execution["state_root"]),
            "--evaluator-contract",
            str(execution["evaluator_contract"]),
            "--total-budget-gpu-hours",
            str(execution["total_budget_gpu_hours"]),
        ]
        if execution.get("probe_contract"):
            command.extend(["--probe-contract", str(execution["probe_contract"])])
        _append_options(
            command,
            execution,
            {
                "runtime_python": "--runtime-python",
                "archive_db": "--archive-db",
                "cas_root": "--cas-root",
                "lock_root": "--lock-root",
                "budget_db": "--budget-db",
                "retrieval_db": "--retrieval-db",
                "literature_query": "--literature-query",
                "literature_max_results": "--literature-max-results",
                "literature_timeout_seconds": "--literature-timeout-seconds",
                "max_files": "--max-files",
                "conformance_timeout_seconds": "--conformance-timeout-seconds",
                "budget_max_trial_gpu_hours": "--budget-max-trial-gpu-hours",
                "budget_high_trial_limit": "--budget-high-trial-limit",
                "poll_seconds": "--poll-seconds",
                "max_iterations": "--max-iterations",
                "max_failures": "--max-failures",
                "max_no_information": "--max-no-information",
                "batch_size": "--batch-size",
                "inner_max_cycles": "--inner-max-cycles",
                "inner_max_attempts": "--inner-max-attempts",
            },
        )
        _append_assets(command, execution.get("asset_bindings"))
        if execution.get("probe_imports") is False:
            command.append("--no-import-probe")
        if execution.get("budget_require_high_cost_approval") is False:
            command.append("--auto-approve-high-cost")
    elif kind == "campaign_queue":
        command = [
            sys.executable,
            "-m",
            "wmloop.execute.campaign_daemon",
            "--output-root",
            str(execution["output_root"]),
            "--workspace-root",
            str(execution["workspace_root"]),
            "--archive-db",
            str(execution["archive_db"]),
            "--cas-root",
            str(execution["cas_root"]),
        ]
        for queue in execution["queue_paths"]:
            command.extend(["--queue", str(queue)])
        _append_options(
            command,
            execution,
            {
                "lock_root": "--lock-root",
                "budget_db": "--budget-db",
                "budget_total_gpu_hours": "--budget-total-gpu-hours",
                "budget_max_trial_gpu_hours": "--budget-max-trial-gpu-hours",
                "budget_high_trial_limit": "--budget-high-trial-limit",
                "poll_seconds": "--poll-seconds",
                "max_cycles": "--max-cycles",
                "max_parallel": "--max-parallel",
                "max_attempts_per_candidate": "--max-attempts-per-candidate",
                "retention_hours": "--retention-hours",
            },
        )
        if execution.get("budget_require_high_cost_approval") is False:
            command.append("--auto-approve-high-cost")
        if execution.get("cleanup") is False:
            command.append("--no-cleanup")
    else:
        raise CampaignDispatchError("EXECUTION_KIND_INVALID")
    return command


def _run_subprocess(
    execution: Mapping[str, Any],
    *,
    cancel_requested: Callable[[], bool] = lambda: False,
    poll_seconds: float = 0.2,
    terminate_grace_seconds: float = 5.0,
) -> Mapping[str, Any]:
    command = _build_command(execution)
    cancelled = False
    termination = None
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            if cancel_requested():
                cancelled = True
                termination = "SIGTERM"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=terminate_grace_seconds)
                except subprocess.TimeoutExpired:
                    termination = "SIGKILL"
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                break
            time.sleep(poll_seconds)
        returncode = process.wait()
        stdout.seek(0)
        stderr.seek(0)
        stdout_text = stdout.read()
        stderr_text = stderr.read()
    pipeline_manifest = _blocked_pipeline_manifest(
        execution, returncode=returncode, stdout=stdout_text
    )
    result = {
        "returncode": returncode,
        "stdout": stdout_text[-4000:],
        "stderr": stderr_text[-4000:],
        "command": command,
        "cancelled": cancelled,
        "termination": termination,
    }
    if pipeline_manifest is not None:
        result["pipeline_manifest"] = pipeline_manifest
        result["outcome"] = "blocked"
        return result
    if returncode != 0 and not cancelled:
        raise CampaignProcessError(
            f"DISPATCH_PROCESS_FAILED:{returncode}", result
        )
    return result


def _blocked_pipeline_manifest(
    execution: Mapping[str, Any], *, returncode: int, stdout: str
) -> dict[str, Any] | None:
    if execution.get("kind") != "pipeline" or returncode != 2:
        return None
    for line in reversed(stdout.splitlines()):
        try:
            manifest = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(manifest, dict)
            and manifest.get("artifact_type")
            == "verdiwm-autonomous-pipeline-manifest"
            and manifest.get("verdict") == "BLOCKED"
        ):
            return manifest
    return None


def _append_assets(command: list[str], value: object) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        rows = sorted((str(parameter), str(path)) for parameter, path in value.items())
    elif isinstance(value, list):
        rows = []
        for raw in value:
            if not isinstance(raw, Mapping):
                raise CampaignDispatchError("EXECUTION_ASSET_BINDINGS_INVALID")
            rows.append((str(raw.get("parameter")), str(raw.get("path"))))
        rows.sort()
    else:
        raise CampaignDispatchError("EXECUTION_ASSET_BINDINGS_INVALID")
    for parameter, path in rows:
        # PARAM intentionally starts with "--". Keep the value attached to the
        # option so argparse cannot mistake the parameter name for a new flag.
        command.append(f"--asset={parameter}={path}")


def _append_options(
    command: list[str],
    execution: Mapping[str, Any],
    options: Mapping[str, str],
) -> None:
    for field, flag in options.items():
        value = execution.get(field)
        if value is not None:
            command.extend([flag, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--campaign-id", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            run_dispatcher(
                DispatcherOptions(
                    state_root=args.state_root,
                    poll_seconds=args.poll_seconds,
                    max_cycles=args.max_cycles,
                    max_parallel=args.max_parallel,
                    campaign_ids=tuple(args.campaign_id),
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
