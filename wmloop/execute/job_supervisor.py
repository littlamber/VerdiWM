"""Durable background execution for long model-training commands.

``submit`` starts a small worker in its own session and returns immediately.
The worker owns the actual child process, streams stdout/stderr to files,
updates a heartbeat, forwards termination signals, and writes a terminal
receipt.  A frontend, chat bridge, or SSH session may therefore disappear
without killing a submitted training job.

This is intentionally a process supervisor, not a cluster scheduler.  Multi-
node placement still belongs to Slurm/Kubernetes/other site infrastructure;
the recorded command and distributed launch metadata make that boundary
explicit and reproducible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.experiments.job_spec import JobSpec, JobSpecError, environment_digest


class JobSupervisorError(RuntimeError):
    """A background job could not be submitted or controlled safely."""


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "orphaned"})
RUNNING_STATES = frozenset({"submitted", "running", "cancelling", "resuming"})


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobSupervisorError(f"JOB_RECEIPT_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise JobSupervisorError(f"JOB_RECEIPT_OBJECT_REQUIRED:{path}")
    return value


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: object, signum: int) -> bool:
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    return True


def _append_event(root: Path, event: Mapping[str, object]) -> None:
    path = root / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _spec_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "spec.json", root / "environment.json", root / "status.json"


def submit_job(spec: JobSpec, *, inherit_environment: bool = True) -> dict[str, object]:
    """Persist and detach one job, returning immediately with its receipt."""

    normalized = spec.validate()
    root = normalized.job_root
    if root.exists() or root.is_symlink():
        raise JobSupervisorError("JOB_ROOT_ALREADY_EXISTS")
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    (root / "attempts").mkdir(mode=0o700)
    environment = dict(os.environ) if inherit_environment else {}
    environment.update({str(key): str(value) for key, value in normalized.environment.items()})
    persisted = JobSpec(
        command=normalized.command,
        cwd=normalized.cwd,
        job_root=root,
        environment=environment,
        output_root=normalized.output_root,
        timeout_seconds=normalized.timeout_seconds,
        metadata=normalized.metadata,
    ).validate()
    spec_path, environment_path, status_path = _spec_paths(root)
    _write_json(spec_path, persisted.to_document())
    _write_json(
        environment_path,
        {"schema_version": 1, "environment": environment},
    )
    status: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-background-job-receipt",
        "job_id": root.name,
        "state": "submitted",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "job_root": str(root),
        "spec_path": str(spec_path),
        "output_root": str(normalized.output_root) if normalized.output_root else None,
        "command": list(normalized.command),
        "cwd": str(normalized.cwd),
        "environment_keys": sorted(environment),
        "environment_sha256": environment_digest(environment),
        "attempt": 0,
        "worker_pid": None,
        "child_pid": None,
        "child_pgid": None,
        "heartbeat_at": None,
        "returncode": None,
        "failure": None,
        "claim_boundary": (
            "This receipt proves process lifecycle, logs, and exit status. It does not "
            "claim that the training result passed a scientific evaluator."
        ),
    }
    _write_json(status_path, status)
    _append_event(root, {"at": _utc_now(), "event": "submitted", "job_id": root.name})
    worker = subprocess.Popen(
        [sys.executable, "-m", "wmloop.execute.job_supervisor", "worker", "--job-root", str(root)],
        cwd=str(normalized.cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=environment,
    )
    status["worker_pid"] = worker.pid
    status["state"] = "running"
    status["updated_at"] = _utc_now()
    _write_json(status_path, status)
    return status


def load_status(job_root: Path, *, reconcile: bool = True) -> dict[str, object]:
    """Read status and mark a dead worker as orphaned instead of hiding it."""

    root = Path(job_root).expanduser().resolve()
    status_path = root / "status.json"
    status = _read_json(status_path)
    if reconcile and status.get("state") in RUNNING_STATES:
        worker_alive = _pid_alive(status.get("worker_pid"))
        child_alive = _pid_alive(status.get("child_pid"))
        if not worker_alive and not child_alive:
            status.update(
                {
                    "state": "orphaned",
                    "updated_at": _utc_now(),
                    "failure": {"code": "JOB_WORKER_NOT_ALIVE"},
                }
            )
            _write_json(status_path, status)
            _append_event(root, {"at": _utc_now(), "event": "orphaned"})
    return status


def cancel_job(job_root: Path, *, grace_seconds: float = 10.0) -> dict[str, object]:
    """Request cooperative cancellation and escalate after a bounded grace."""

    if grace_seconds < 0:
        raise JobSupervisorError("JOB_CANCEL_GRACE_INVALID")
    root = Path(job_root).expanduser().resolve()
    status = load_status(root)
    if status.get("state") in TERMINAL_STATES:
        return status
    status["state"] = "cancelling"
    status["updated_at"] = _utc_now()
    _write_json(root / "status.json", status)
    _append_event(root, {"at": _utc_now(), "event": "cancel_requested"})
    _signal_group(status.get("child_pgid"), signal.SIGTERM)
    worker_pid = status.get("worker_pid")
    if isinstance(worker_pid, int) and worker_pid > 1:
        try:
            os.kill(worker_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        current = load_status(root)
        if current.get("state") in TERMINAL_STATES:
            return current
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    current = load_status(root)
    _signal_group(current.get("child_pgid"), signal.SIGKILL)
    worker_pid = current.get("worker_pid")
    if isinstance(worker_pid, int) and worker_pid > 1:
        try:
            os.kill(worker_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    current.update(
        {
            "state": "cancelled",
            "updated_at": _utc_now(),
            "failure": {"code": "JOB_CANCELLED"},
        }
    )
    _write_json(root / "status.json", current)
    _append_event(root, {"at": _utc_now(), "event": "cancelled"})
    return current


def resume_job(job_root: Path) -> dict[str, object]:
    """Start a fresh attempt with the exact immutable command and environment."""

    root = Path(job_root).expanduser().resolve()
    status = load_status(root)
    if status.get("state") not in TERMINAL_STATES:
        raise JobSupervisorError("JOB_RESUME_REQUIRES_TERMINAL_STATE")
    spec_document = _read_json(root / "spec.json")
    environment_document = _read_json(root / "environment.json")
    environment = environment_document.get("environment")
    if not isinstance(environment, dict):
        raise JobSupervisorError("JOB_ENVIRONMENT_INVALID")
    spec = JobSpec.from_document(spec_document, environment={str(k): str(v) for k, v in environment.items()})
    if spec.job_root != root:
        raise JobSupervisorError("JOB_ROOT_BINDING_MISMATCH")
    if environment_digest(spec.environment) != status.get("environment_sha256"):
        raise JobSupervisorError("JOB_ENVIRONMENT_DIGEST_MISMATCH")
    status.update(
        {
            "state": "resuming",
            "updated_at": _utc_now(),
            "failure": None,
            "returncode": None,
            "heartbeat_at": None,
        }
    )
    _write_json(root / "status.json", status)
    _append_event(root, {"at": _utc_now(), "event": "resume_requested"})
    worker = subprocess.Popen(
        [sys.executable, "-m", "wmloop.execute.job_supervisor", "worker", "--job-root", str(root)],
        cwd=str(spec.cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=dict(spec.environment),
    )
    status["worker_pid"] = worker.pid
    status["state"] = "running"
    status["updated_at"] = _utc_now()
    _write_json(root / "status.json", status)
    return status


def tail_job(job_root: Path, *, lines: int = 40) -> str:
    if isinstance(lines, bool) or not isinstance(lines, int) or lines < 1:
        raise JobSupervisorError("JOB_TAIL_LINES_INVALID")
    root = Path(job_root).expanduser().resolve()
    attempts = sorted((root / "attempts").glob("attempt-*/stdout.log"))
    if not attempts:
        return ""
    return "".join(attempts[-1].read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])


def _worker(job_root: Path) -> int:
    root = Path(job_root).expanduser().resolve()
    spec_document = _read_json(root / "spec.json")
    environment_document = _read_json(root / "environment.json")
    environment = environment_document.get("environment")
    if not isinstance(environment, dict):
        raise JobSupervisorError("JOB_ENVIRONMENT_INVALID")
    spec = JobSpec.from_document(spec_document, environment={str(k): str(v) for k, v in environment.items()})
    status = _read_json(root / "status.json")
    attempt = int(status.get("attempt", 0)) + 1
    attempt_root = root / "attempts" / f"attempt-{attempt:06d}"
    attempt_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    stdout_path = attempt_root / "stdout.log"
    stderr_path = attempt_root / "stderr.log"
    start = time.monotonic()
    child: subprocess.Popen[str] | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        if child is not None:
            _signal_group(child.pid, signal.SIGTERM)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, request_stop)
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout, stderr_path.open("w", encoding="utf-8", buffering=1) as stderr:
            child = subprocess.Popen(
                list(spec.command),
                cwd=str(spec.cwd),
                env=dict(spec.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
                text=True,
            )
            status.update(
                {
                    "state": "running",
                    "attempt": attempt,
                    "attempt_root": str(attempt_root),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "child_pid": child.pid,
                    "child_pgid": child.pid,
                    "heartbeat_at": _utc_now(),
                    "updated_at": _utc_now(),
                }
            )
            _write_json(root / "status.json", status)
            _append_event(root, {"at": _utc_now(), "event": "started", "attempt": attempt, "pid": child.pid})
            timed_out = False
            while True:
                try:
                    returncode = child.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start
                    if spec.timeout_seconds is not None and elapsed >= spec.timeout_seconds:
                        timed_out = True
                        stop_requested = True
                        _signal_group(child.pid, signal.SIGTERM)
                        try:
                            returncode = child.wait(timeout=10.0)
                        except subprocess.TimeoutExpired:
                            _signal_group(child.pid, signal.SIGKILL)
                            returncode = child.wait(timeout=5.0)
                        break
                    status["heartbeat_at"] = _utc_now()
                    status["updated_at"] = status["heartbeat_at"]
                    _write_json(root / "status.json", status)
            state = "cancelled" if stop_requested else "completed" if returncode == 0 else "failed"
            failure = None
            if timed_out:
                state = "failed"
                failure = {"code": "JOB_TIMEOUT", "timeout_seconds": spec.timeout_seconds}
            elif returncode != 0:
                failure = {"code": "JOB_EXIT_NONZERO", "returncode": returncode}
            status.update(
                {
                    "state": state,
                    "returncode": returncode,
                    "failure": failure,
                    "child_pid": None,
                    "child_pgid": None,
                    "heartbeat_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            _write_json(root / "status.json", status)
            _append_event(root, {"at": _utc_now(), "event": state, "attempt": attempt, "returncode": returncode})
            return 0 if state == "completed" else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _parse_env(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise JobSupervisorError("JOB_ENVIRONMENT_OVERRIDE_INVALID")
        result[key] = item
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verdiwm-job")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="submit a detached long-running command")
    submit.add_argument("--job-root", type=Path, required=True)
    submit.add_argument("--cwd", type=Path, default=Path.cwd())
    submit.add_argument("--output-root", type=Path)
    submit.add_argument("--timeout-seconds", type=float)
    submit.add_argument("--env", action="append", default=[])
    submit.add_argument("--metadata", default="{}", help="JSON object stored as non-secret job metadata")
    submit.add_argument("job_command", nargs=argparse.REMAINDER)
    status = commands.add_parser("status")
    status.add_argument("--job-root", type=Path, required=True)
    tail = commands.add_parser("tail")
    tail.add_argument("--job-root", type=Path, required=True)
    tail.add_argument("--lines", type=int, default=40)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--job-root", type=Path, required=True)
    cancel.add_argument("--grace-seconds", type=float, default=10.0)
    resume = commands.add_parser("resume")
    resume.add_argument("--job-root", type=Path, required=True)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "submit":
            command = tuple(args.job_command)
            while command and command[0] == "--":
                command = command[1:]
            metadata = json.loads(args.metadata)
            if not isinstance(metadata, dict):
                raise JobSupervisorError("JOB_METADATA_INVALID")
            result = submit_job(JobSpec(command=command, cwd=args.cwd, job_root=args.job_root, environment=_parse_env(args.env), output_root=args.output_root, timeout_seconds=args.timeout_seconds, metadata=metadata))
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "status":
            print(json.dumps(load_status(args.job_root), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "tail":
            print(tail_job(args.job_root, lines=args.lines), end="")
            return 0
        if args.command == "cancel":
            print(json.dumps(cancel_job(args.job_root, grace_seconds=args.grace_seconds), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "worker":
            try:
                return _worker(args.job_root)
            except Exception as exc:
                root = Path(args.job_root).expanduser().resolve()
                try:
                    status = _read_json(root / "status.json")
                    status.update(
                        {
                            "state": "failed",
                            "updated_at": _utc_now(),
                            "finished_at": _utc_now(),
                            "failure": {
                                "code": "JOB_WORKER_EXCEPTION",
                                "type": type(exc).__name__,
                                "message": str(exc)[:500],
                            },
                        }
                    )
                    _write_json(root / "status.json", status)
                    _append_event(root, {"at": _utc_now(), "event": "failed", "worker_exception": str(exc)[:500]})
                except Exception:
                    pass
                return 1
        print(json.dumps(resume_job(args.job_root), ensure_ascii=False, sort_keys=True))
        return 0
    except (JobSupervisorError, JobSpecError, OSError, ValueError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
