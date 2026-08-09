"""Durable background execution for bounded VerdiWM candidate queues.

The daemon is a coordinator, not a second experiment executor.  Each selected
candidate receives an isolated queue projection and calls the existing
``run_selected_queue`` transaction boundary.  Workers share the budget ledger,
archive, CAS, and GPU lease namespace, so concurrent work remains bounded and
receipt-first.  A daemon restart resumes from its status file and settled
archive rather than replaying completed candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.auto_experiment import (
    AutoExperimentError,
    cleanup_auto_experiment_scratch,
)
from wmloop.execute.experiment_scheduler import (
    ExperimentSchedulerError,
    run_selected_queue,
)
from wmloop.execute.gpu_lease import GpuLeaseError


class CampaignDaemonError(RuntimeError):
    """A background campaign could not preserve its execution contract."""


QueueRunner = Callable[..., Mapping[str, object]]
SleepFunction = Callable[[float], None]


@dataclass(frozen=True)
class CampaignDaemonOptions:
    """Configuration for one resumable queue-draining daemon."""

    queue_paths: tuple[Path, ...]
    output_root: Path
    workspace_root: Path
    archive_db: Path
    cas_root: Path
    lock_root: Path = Path("/tmp/verdiwm-gpu-leases")
    budget_db: Path | None = None
    budget_total_gpu_hours: float | None = None
    budget_max_trial_gpu_hours: float = 120.0
    budget_high_trial_limit: int = 2
    budget_require_high_cost_approval: bool = True
    poll_seconds: float = 30.0
    max_cycles: int = 720
    max_parallel: int = 1
    max_attempts_per_candidate: int = 3
    retention_hours: float = 24.0
    cleanup_enabled: bool = True


def run_campaign_daemon(
    options: CampaignDaemonOptions,
    *,
    queue_runner: QueueRunner | None = None,
    sleeper: SleepFunction = time.sleep,
) -> dict[str, object]:
    """Drain queues until completion, stop signal, or bounded cycle limit."""

    queue_paths = tuple(Path(path).resolve(strict=True) for path in options.queue_paths)
    _validate_options(options, queue_paths=queue_paths)
    workspace = Path(options.workspace_root).resolve(strict=True)
    destination = Path(options.output_root).resolve()
    shared_budget = (
        Path(options.budget_db).resolve()
        if options.budget_db is not None
        else destination / "budget.db"
    )
    budget_total = _budget_total(options, queue_paths=queue_paths)
    input_hash = _input_hash(
        options,
        queue_paths=queue_paths,
        budget_total_gpu_hours=budget_total,
    )
    _bind_output(destination, input_hash=input_hash, queue_paths=queue_paths)
    _acquire_daemon_lock(destination)
    runner = queue_runner or run_selected_queue
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
            queue_paths=queue_paths,
            options=options,
        )
        state["state"] = "running"
        state["max_cycles"] = options.max_cycles
        state["poll_seconds"] = options.poll_seconds
        state["budget_total_gpu_hours"] = budget_total
        state.pop("last_error", None)
        _write_json(destination / "status.json", state)
        for cycle in range(int(state["cycle"]) + 1, options.max_cycles + 1):
            if stop_requested:
                state["state"] = "stopped"
                break
            pending = _pending_candidates(
                queue_paths=queue_paths,
                state=state,
                max_attempts=options.max_attempts_per_candidate,
            )
            if not pending:
                state["state"] = _terminal_state(state)
                state["cycle"] = cycle - 1
                break
            workers_root = destination / "workers"
            workers_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            cycle_record: dict[str, object] = {
                "cycle": cycle,
                "started_at": _utc_now(),
                "candidate_count": len(pending),
                "workers": [],
            }
            futures: dict[Future[Mapping[str, object]], tuple[str, str, Path]] = {}
            with ThreadPoolExecutor(max_workers=options.max_parallel) as executor:
                for queue_path, candidate_id in pending[: options.max_parallel]:
                    queue_worker_root = workers_root / _safe_id(
                        f"{queue_path.stem}-{_sha256(str(queue_path).encode('utf-8'))[:12]}"
                    )
                    worker_root = queue_worker_root / _safe_id(
                        f"{candidate_id}-{_sha256(candidate_id.encode('utf-8'))[:8]}"
                    )
                    worker_queue = _materialize_worker_queue(
                        queue_path=queue_path,
                        candidate_id=candidate_id,
                        worker_root=worker_root,
                    )
                    futures[executor.submit(
                        runner,
                        queue_path=worker_queue,
                        workspace_root=workspace,
                        archive_db=Path(options.archive_db).resolve(),
                        cas_root=Path(options.cas_root).resolve(),
                        lock_root=Path(options.lock_root).resolve(),
                        budget_db=shared_budget,
                        budget_total_gpu_hours=budget_total,
                        budget_max_trial_gpu_hours=options.budget_max_trial_gpu_hours,
                        budget_high_trial_limit=options.budget_high_trial_limit,
                        budget_require_high_cost_approval=(
                            options.budget_require_high_cost_approval
                        ),
                    )] = (f"{queue_path}::{candidate_id}", candidate_id, worker_queue)
                    state["launch_count"] = int(state["launch_count"]) + 1
                for future in as_completed(futures):
                    key, candidate_id, worker_queue = futures[future]
                    try:
                        result = dict(future.result())
                        candidate_states = result.get("candidate_states")
                        state_value = (
                            str(candidate_states.get(candidate_id))
                            if isinstance(candidate_states, Mapping)
                            else "error"
                        )
                        if state_value not in {"completed", "blocked"}:
                            state_value = "error"
                        _record_candidate_state(state, key, state_value)
                        cycle_record["workers"].append({
                            "key": key,
                            "state": state_value,
                            "worker_queue": str(worker_queue),
                        })
                    except Exception as exc:
                        if _is_resource_deferral(exc):
                            _record_candidate_deferred(
                                state,
                                key,
                                reason=str(exc)[:500],
                            )
                            worker_state = "deferred"
                        else:
                            _record_candidate_error(
                                state,
                                key,
                                error=f"{type(exc).__name__}:{str(exc)[:500]}",
                                max_attempts=options.max_attempts_per_candidate,
                            )
                            worker_state = "error"
                        cycle_record["workers"].append({
                            "key": key,
                            "state": worker_state,
                            "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                            "worker_queue": str(worker_queue),
                        })
            state["cycle"] = cycle
            state["error_count"] = sum(
                int(value.get("errors", 0))
                for value in state["candidate_states"].values()
                if isinstance(value, Mapping)
            )
            if options.cleanup_enabled:
                try:
                    state["cleanup"] = cleanup_auto_experiment_scratch(
                        run_root=destination,
                        older_than_hours=options.retention_hours,
                        apply=True,
                        archive_db=Path(options.archive_db),
                        cas_root=Path(options.cas_root),
                    )
                except Exception as exc:
                    state["cleanup"] = {
                        "state": "error",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                        "deleted_count": 0,
                    }
            cycle_record["finished_at"] = _utc_now()
            _write_json(destination / "cycles" / f"cycle-{cycle:06d}.json", cycle_record)
            _write_json(destination / "status.json", state)
            if stop_requested:
                state["state"] = "stopped"
                break
            if not _pending_candidates(
                queue_paths=queue_paths,
                state=state,
                max_attempts=options.max_attempts_per_candidate,
            ):
                state["state"] = _terminal_state(state)
                break
            if cycle < options.max_cycles:
                sleeper(options.poll_seconds)
        else:
            if state.get("state") == "running":
                state["state"] = "blocked"
        if state.get("state") == "running":
            state["state"] = "stopped" if stop_requested else "completed"
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


def _validate_options(options: CampaignDaemonOptions, *, queue_paths: Sequence[Path]) -> None:
    if not queue_paths or options.max_cycles < 1 or not 1 <= options.max_parallel <= 64:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_ARGUMENT_INVALID")
    if (
        options.max_attempts_per_candidate < 1
        or not math.isfinite(options.poll_seconds)
        or options.poll_seconds < 0
    ):
        raise CampaignDaemonError("CAMPAIGN_DAEMON_ARGUMENT_INVALID")
    if not math.isfinite(options.retention_hours) or options.retention_hours < 0:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_RETENTION_INVALID")
    if options.budget_total_gpu_hours is not None and (
        not math.isfinite(options.budget_total_gpu_hours)
        or options.budget_total_gpu_hours <= 0
    ):
        raise CampaignDaemonError("CAMPAIGN_DAEMON_BUDGET_TOTAL_INVALID")
    if (
        not math.isfinite(options.budget_max_trial_gpu_hours)
        or options.budget_max_trial_gpu_hours <= 0
        or options.budget_high_trial_limit < 0
    ):
        raise CampaignDaemonError("CAMPAIGN_DAEMON_RESOURCE_POLICY_INVALID")
    output = Path(options.output_root).resolve()
    for path in (
        options.workspace_root,
        options.archive_db,
        options.cas_root,
        options.lock_root,
        options.budget_db,
        *queue_paths,
    ):
        if path is not None and _paths_overlap(output, Path(path).resolve()):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_OUTPUT_OVERLAP_INVALID")


def _input_hash(
    options: CampaignDaemonOptions,
    *,
    queue_paths: Sequence[Path],
    budget_total_gpu_hours: float,
) -> str:
    payload = {
        "queues": [
            {"path": str(path), "sha256": _sha256(path.read_bytes())}
            for path in queue_paths
        ],
        "workspace_root": str(Path(options.workspace_root).resolve()),
        "archive_db": str(Path(options.archive_db).resolve()),
        "cas_root": str(Path(options.cas_root).resolve()),
        "lock_root": str(Path(options.lock_root).resolve()),
        "budget_db": str(Path(options.budget_db).resolve()) if options.budget_db else None,
        "budget_total_gpu_hours": budget_total_gpu_hours,
        **_nondefault_resource_policy(options),
        "max_parallel": options.max_parallel,
        "max_attempts_per_candidate": options.max_attempts_per_candidate,
        "retention_hours": options.retention_hours,
    }
    return _sha256(_canonical_json(payload))


def _nondefault_resource_policy(
    options: CampaignDaemonOptions,
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


def _budget_total(
    options: CampaignDaemonOptions, *, queue_paths: Sequence[Path]
) -> float:
    if options.budget_total_gpu_hours is not None:
        return float(options.budget_total_gpu_hours)
    total = 0.0
    for queue_path in queue_paths:
        value = _load_json(queue_path).get("total_budget_gpu_hours")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_BUDGET_INVALID")
        total += float(value)
    if not math.isfinite(total) or total <= 0:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_BUDGET_TOTAL_INVALID")
    return total


def _bind_output(
    destination: Path, *, input_hash: str, queue_paths: Sequence[Path]
) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise CampaignDaemonError("CAMPAIGN_DAEMON_OUTPUT_INVALID")
        lock = _load_json(destination / "input.lock.json")
        if lock.get("input_hash") != input_hash or lock.get("queue_paths") != [
            str(path) for path in queue_paths
        ]:
            raise CampaignDaemonError("CAMPAIGN_DAEMON_INPUT_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True)
    (destination / "cycles").mkdir(mode=0o700)
    _write_json(
        destination / "input.lock.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-campaign-daemon-input-lock",
            "input_hash": input_hash,
            "queue_paths": [str(path) for path in queue_paths],
        },
    )


def _acquire_daemon_lock(destination: Path) -> None:
    lock_path = destination / "daemon.lock"
    payload = (
        json.dumps({"pid": os.getpid(), "created_at": _utc_now()}, sort_keys=True)
        + "\n"
    )
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _load_json(lock_path)
        pid = existing.get("pid")
        if isinstance(pid, int) and pid > 0 and _pid_alive(pid):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_ALREADY_RUNNING")
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise CampaignDaemonError("CAMPAIGN_DAEMON_ALREADY_RUNNING") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _release_daemon_lock(destination: Path) -> None:
    path = destination / "daemon.lock"
    try:
        lock = _load_json(path)
        if lock.get("pid") == os.getpid():
            path.unlink()
    except (CampaignDaemonError, OSError):
        return


def _load_state(
    destination: Path,
    *,
    input_hash: str,
    queue_paths: Sequence[Path],
    options: CampaignDaemonOptions,
) -> dict[str, object]:
    path = destination / "status.json"
    if path.is_file() and not path.is_symlink():
        state = _load_json(path)
        if state.get("input_hash") != input_hash:
            raise CampaignDaemonError("CAMPAIGN_DAEMON_STATUS_INPUT_MISMATCH")
        return state
    state = {
        "schema_version": 1,
        "artifact_type": "verdiwm-campaign-daemon-manifest",
        "state": "running",
        "input_hash": input_hash,
        "queue_paths": [str(path) for path in queue_paths],
        "output_root": str(Path(options.output_root).resolve()),
        "cycle": 0,
        "max_cycles": options.max_cycles,
        "max_parallel": options.max_parallel,
        "poll_seconds": options.poll_seconds,
        "budget_total_gpu_hours": _budget_total(options, queue_paths=queue_paths),
        "candidate_states": {},
        "launch_count": 0,
        "error_count": 0,
        "cleanup": {"candidate_count": 0, "deleted_count": 0},
        "claim_boundary": (
            "The daemon coordinates bounded receipt-first workers; it does not "
            "change candidate contracts or turn failed trials into evidence."
        ),
    }
    _write_json(path, state)
    return state


def _pending_candidates(
    *,
    queue_paths: Sequence[Path],
    state: Mapping[str, object],
    max_attempts: int,
) -> list[tuple[Path, str]]:
    output: list[tuple[Path, str]] = []
    states = state.get("candidate_states", {})
    for queue_path in queue_paths:
        queue = _load_json(queue_path)
        selected = queue.get("selected")
        if not isinstance(selected, list):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
        seen: set[str] = set()
        for row in selected:
            if not isinstance(row, Mapping) or not isinstance(
                row.get("candidate_id"), str
            ):
                raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
            candidate_id = str(row["candidate_id"])
            if not candidate_id or candidate_id in seen:
                raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
            seen.add(candidate_id)
            key = f"{queue_path}::{candidate_id}"
            record = states.get(key) if isinstance(states, Mapping) else None
            if isinstance(record, Mapping) and record.get("state") in {
                "completed",
                "blocked",
            }:
                continue
            if (
                isinstance(record, Mapping)
                and int(record.get("errors", 0)) >= max_attempts
            ):
                continue
            output.append((queue_path, candidate_id))
    return output


def _materialize_worker_queue(
    *, queue_path: Path, candidate_id: str, worker_root: Path
) -> Path:
    queue = _load_json(queue_path)
    selected = queue.get("selected")
    if not isinstance(selected, list):
        raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
    row = next(
        (
            item
            for item in selected
            if isinstance(item, Mapping)
            and item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if row is None:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_CANDIDATE_NOT_FOUND")
    root = queue_path.parent.resolve()
    transformed = dict(row)
    stages = []
    source_stages = row.get("stages")
    if not isinstance(source_stages, list) or not source_stages:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
    for index, stage in enumerate(source_stages):
        if not isinstance(stage, Mapping):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_QUEUE_INVALID")
        stage_copy = dict(stage)
        if not isinstance(stage_copy.get("plan_path"), str):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_PLAN_PATH_INVALID")
        plan = Path(stage_copy["plan_path"])
        source_plan = (
            (root / plan).resolve() if not plan.is_absolute() else plan.resolve()
        )
        if (
            not source_plan.is_file()
            or source_plan.is_symlink()
            or not _is_inside(root, source_plan)
        ):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_PLAN_PATH_INVALID")
        if _sha256(source_plan.read_bytes()) != str(stage_copy.get("plan_sha256")):
            raise CampaignDaemonError("CAMPAIGN_DAEMON_PLAN_HASH_MISMATCH")
        target_plan = worker_root / "plans" / f"{index:02d}-{source_plan.name}"
        target_plan.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(source_plan, target_plan)
        stage_copy["plan_path"] = str(Path("plans") / target_plan.name)
        stages.append(stage_copy)
    transformed["stages"] = stages
    worker_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker_queue = worker_root / "queue.json"
    mini = dict(queue)
    mini["selected"] = [transformed]
    mini["deferred"] = []
    _write_json(worker_queue, mini)
    return worker_queue


def _record_candidate_state(
    state: dict[str, object], key: str, candidate_state: str
) -> None:
    states = state["candidate_states"]
    assert isinstance(states, dict)
    previous = states.get(key)
    errors = int(previous.get("errors", 0)) if isinstance(previous, Mapping) else 0
    deferrals = (
        int(previous.get("deferrals", 0)) if isinstance(previous, Mapping) else 0
    )
    record = {
        "state": candidate_state,
        "errors": errors,
        "updated_at": _utc_now(),
    }
    if deferrals:
        record["deferrals"] = deferrals
    states[key] = record


def _record_candidate_error(
    state: dict[str, object],
    key: str,
    *,
    error: str,
    max_attempts: int,
) -> None:
    states = state["candidate_states"]
    assert isinstance(states, dict)
    previous = states.get(key)
    errors = (
        int(previous.get("errors", 0)) if isinstance(previous, Mapping) else 0
    ) + 1
    deferrals = (
        int(previous.get("deferrals", 0)) if isinstance(previous, Mapping) else 0
    )
    states[key] = {
        "state": "blocked" if errors >= max_attempts else "error",
        "errors": errors,
        "deferrals": deferrals,
        "last_error": error,
        "updated_at": _utc_now(),
    }


def _record_candidate_deferred(
    state: dict[str, object], key: str, *, reason: str
) -> None:
    states = state["candidate_states"]
    assert isinstance(states, dict)
    previous = states.get(key)
    errors = int(previous.get("errors", 0)) if isinstance(previous, Mapping) else 0
    deferrals = (
        int(previous.get("deferrals", 0)) if isinstance(previous, Mapping) else 0
    ) + 1
    states[key] = {
        "state": "deferred",
        "errors": errors,
        "deferrals": deferrals,
        "last_deferral": reason,
        "updated_at": _utc_now(),
    }


def _is_resource_deferral(error: Exception) -> bool:
    return isinstance(error, GpuLeaseError) and str(error).startswith(
        "GPU_LEASE_UNAVAILABLE"
    )


def _terminal_state(state: Mapping[str, object]) -> str:
    states = state.get("candidate_states", {})
    if isinstance(states, Mapping) and any(
        isinstance(value, Mapping) and value.get("state") == "blocked"
        for value in states.values()
    ):
        return "blocked"
    return "completed"


def _manifest(state: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(state)
    try:
        validate_document("campaign_daemon_manifest", manifest)
    except ContractValidationError as exc:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_MANIFEST_INVALID") from exc
    return manifest


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignDaemonError("CAMPAIGN_DAEMON_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise CampaignDaemonError("CAMPAIGN_DAEMON_JSON_INVALID")
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in ".-_" else "-"
        for character in value
    )[:160]


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either path contains the other."""

    return first == second or first in second.parents or second in first.parents


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases"))
    parser.add_argument("--budget-db", type=Path)
    parser.add_argument("--budget-total-gpu-hours", type=float)
    parser.add_argument("--budget-max-trial-gpu-hours", type=float, default=120.0)
    parser.add_argument("--budget-high-trial-limit", type=int, default=2)
    parser.add_argument("--auto-approve-high-cost", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int, default=720)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--max-attempts-per-candidate", type=int, default=3)
    parser.add_argument("--retention-hours", type=float, default=24.0)
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = run_campaign_daemon(
            CampaignDaemonOptions(
                queue_paths=tuple(args.queue),
                output_root=args.output_root,
                workspace_root=args.workspace_root,
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
                poll_seconds=args.poll_seconds,
                max_cycles=args.max_cycles,
                max_parallel=args.max_parallel,
                max_attempts_per_candidate=args.max_attempts_per_candidate,
                retention_hours=args.retention_hours,
                cleanup_enabled=not args.no_cleanup,
            )
        )
    except (CampaignDaemonError, AutoExperimentError, ExperimentSchedulerError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
