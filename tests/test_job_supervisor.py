from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.execute.job_supervisor import (
    cancel_job,
    load_status,
    resume_job,
    submit_job,
    tail_job,
)
from wmloop.experiments.distributed_launch import (
    DistributedLaunchConfig,
    build_launch_command,
)
from wmloop.experiments.job_spec import JobSpec, effective_batch_size


def _wait_terminal(root: Path, timeout: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = load_status(root)
        if status["state"] in {"completed", "failed", "cancelled", "orphaned"}:
            return status
        time.sleep(0.05)
    raise AssertionError(load_status(root))


def test_job_streams_logs_and_records_terminal_receipt() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "job"
        result = submit_job(
            JobSpec(
                command=(
                    sys.executable,
                    "-c",
                    "import sys; print('hello'); print('oops', file=sys.stderr)",
                ),
                cwd=Path.cwd(),
                job_root=root,
            )
        )
        assert result["state"] == "running"
        status = _wait_terminal(root)
        assert status["state"] == "completed"
        assert "hello" in tail_job(root)
        attempt = Path(str(status["attempt_root"]))
        assert "oops" in (attempt / "stderr.log").read_text()
        events = (root / "events.jsonl").read_text().splitlines()
        assert any(json.loads(line)["event"] == "completed" for line in events)


def test_job_cancel_and_resume_use_same_immutable_spec() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "cancel-job"
        submit_job(
            JobSpec(
                command=(sys.executable, "-c", "import time; time.sleep(30)"),
                cwd=Path.cwd(),
                job_root=root,
            )
        )
        cancelled = cancel_job(root, grace_seconds=2)
        assert cancelled["state"] == "cancelled"
        resume_root = Path(temporary) / "resume-job"
        marker = Path(temporary) / "resume-marker"
        code = (
            "import pathlib,sys; p=pathlib.Path(" + repr(str(marker)) + "); "
            "sys.exit(1) if not p.exists() and p.write_text('1') else None"
        )
        # The first attempt fails and creates a marker; the second exits 0.
        submit_job(JobSpec(command=(sys.executable, "-c", code), cwd=Path.cwd(), job_root=resume_root))
        assert _wait_terminal(resume_root)["state"] == "failed"
        resumed = resume_job(resume_root)
        assert resumed["state"] == "running"
        status = _wait_terminal(resume_root)
        assert status["state"] == "completed"
        assert status["attempt"] == 2


def test_effective_batch_and_torchrun_command_are_explicit() -> None:
    assert effective_batch_size(8, 4, 8) == 256
    config = DistributedLaunchConfig(backend="torchrun", nproc_per_node=8)
    command = build_launch_command(("train.py", "--steps", "10"), config)
    assert command[:5] == ["torchrun", "--nnodes", "1", "--nproc_per_node", "8"]
    assert command[-3:] == ["train.py", "--steps", "10"]
