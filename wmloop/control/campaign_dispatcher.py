"""Consume Campaign API dispatch manifests through existing VerdiWM daemons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore


class CampaignDispatchError(RuntimeError):
    """A dispatch manifest could not be admitted or settled."""


Runner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class DispatcherOptions:
    state_root: Path
    poll_seconds: float = 10.0
    max_cycles: int = 1
    max_parallel: int = 1


def run_dispatcher(
    options: DispatcherOptions,
    *,
    runner: Runner | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if options.max_cycles < 1 or options.max_parallel != 1 or options.poll_seconds < 0:
        raise CampaignDispatchError("DISPATCHER_OPTIONS_INVALID")
    root = Path(options.state_root).expanduser().resolve()
    store = CampaignStore(root)
    dispatch_root = store.root / "dispatch"
    pending = dispatch_root / "pending"
    running = dispatch_root / "running"
    completed = dispatch_root / "completed"
    failed = dispatch_root / "failed"
    for path in (pending, running, completed, failed):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = dispatch_root / "dispatcher.lock"
    _acquire_lock(lock_path)
    execute = runner or _run_subprocess
    settled: list[str] = []
    failed_ids: list[str] = []
    try:
        failed_ids.extend(_recover_interrupted(store=store, running=running, failed=failed))
        for cycle in range(options.max_cycles):
            paths = sorted(path for path in pending.glob("*.json") if path.is_file() and not path.is_symlink())
            if not paths:
                if cycle + 1 < options.max_cycles:
                    sleeper(options.poll_seconds)
                continue
            for source in paths[: options.max_parallel]:
                campaign_id = source.stem
                active = running / source.name
                try:
                    source.replace(active)
                    dispatch = _load_dispatch(active)
                    store.record_dispatch_result(campaign_id, status="running")
                    result = dict(execute(dispatch["execution"]))
                    store.record_dispatch_result(campaign_id, status="completed", result=result)
                    dispatch["state"] = "completed"
                    dispatch["result"] = result
                    _write_json(completed / source.name, dispatch)
                    active.unlink(missing_ok=True)
                    settled.append(campaign_id)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                    try:
                        store.record_dispatch_result(campaign_id, status="failed", error=error)
                    except CampaignAPIError:
                        pass
                    try:
                        dispatch = _load_dispatch(active)
                    except Exception:
                        dispatch = {"campaign_id": campaign_id, "schema_version": 1}
                    dispatch["state"] = "failed"
                    dispatch["error"] = error
                    _write_json(failed / source.name, dispatch)
                    active.unlink(missing_ok=True)
                    failed_ids.append(campaign_id)
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-campaign-dispatcher-manifest",
            "state": "completed",
            "settled_campaign_ids": settled,
            "failed_campaign_ids": failed_ids,
            "pending_count": len(list(pending.glob("*.json"))),
        }
    finally:
        lock_path.unlink(missing_ok=True)


def _recover_interrupted(
    *, store: CampaignStore, running: Path, failed: Path
) -> list[str]:
    recovered: list[str] = []
    for active in sorted(running.glob("*.json")):
        if active.is_symlink() or not active.is_file():
            continue
        campaign_id = active.stem
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
        recovered.append(campaign_id)
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


def _run_subprocess(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = execution.get("kind")
    if kind == "evolution":
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
                "poll_seconds": "--poll-seconds",
                "max_iterations": "--max-iterations",
                "max_failures": "--max-failures",
                "max_no_information": "--max-no-information",
                "batch_size": "--batch-size",
                "inner_max_cycles": "--inner-max-cycles",
                "inner_max_attempts": "--inner-max-attempts",
            },
        )
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
    else:
        raise CampaignDispatchError("EXECUTION_KIND_INVALID")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result = {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "command": command,
    }
    if completed.returncode != 0:
        raise CampaignDispatchError(f"DISPATCH_PROCESS_FAILED:{completed.returncode}")
    return result


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
    args = parser.parse_args()
    print(
        json.dumps(
            run_dispatcher(
                DispatcherOptions(
                    state_root=args.state_root,
                    poll_seconds=args.poll_seconds,
                    max_cycles=args.max_cycles,
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
