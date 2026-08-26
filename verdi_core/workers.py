"""Model-agnostic experiment workers.

Workers own execution and artifact production. They do not decide whether an
experiment is scientifically positive.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import urllib.request
from dataclasses import replace
from pathlib import Path
import uuid
from typing import Any, Protocol

from .adapter import ModelAdapter
from .autonomy import AutonomousRepairLoop
from .evidence import verify_artifacts


@dataclass(frozen=True)
class ExperimentTask:
    job_id: str
    hypothesis: dict[str, Any]
    split: str


class Worker(Protocol):
    def execute(self, task: ExperimentTask) -> dict[str, Any]: ...


class AdapterWorker:
    """Reference worker for adapters that expose intervene/evaluate hooks."""

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    def execute(self, task: ExperimentTask) -> dict[str, Any]:
        intervention = self.adapter.intervene(task.hypothesis)
        result = self.adapter.evaluate(intervention, task.split)
        return {"job_id": task.job_id, "intervention": intervention, "raw_result": result}


class RepairingWorker:
    """Retry a failed adapter task after an automatically verified patch."""

    def __init__(self, worker: Worker, repair_loop: AutonomousRepairLoop, *, repository: Path, allowed_paths: list[str], destination_root: Path, tests: list[list[str]] | None = None):
        self.worker = worker
        self.repair_loop = repair_loop
        self.repository = Path(repository)
        self.allowed_paths = list(allowed_paths)
        self.destination_root = Path(destination_root)
        self.tests = tests or []

    def execute(self, task: ExperimentTask) -> dict[str, Any]:
        try:
            return self.worker.execute(task)
        except Exception as failure:
            destination = self.destination_root / f"{task.job_id}-repair-{uuid.uuid4().hex[:10]}"
            repair = self.repair_loop.run(
                self.repository,
                objective=str(task.hypothesis.get("objective", task.hypothesis.get("title", "repair failed experiment"))),
                failure={"type": type(failure).__name__, "message": str(failure), "job_id": task.job_id},
                allowed_paths=self.allowed_paths,
                destination=destination,
                tests=self.tests,
                resume=lambda worktree, _receipt: self.worker.execute(replace(task, hypothesis={**task.hypothesis, "worktree": str(worktree)})),
            )
            if repair.get("state") != "approved" or "resumed" not in repair.get("execution", {}):
                raise RuntimeError(f"automatic repair did not resume task: {repair}") from failure
            return {"repair": repair, "resumed_result": repair["execution"]["resumed"]}


class LocalPythonWorker:
    """Runs a user-approved Python entrypoint in a subprocess boundary.

    The entrypoint receives one JSON object on stdin and must emit one JSON
    object on stdout. A caller should use a dedicated work directory and a
    restricted environment for untrusted generated code.
    """

    def __init__(self, command: list[str], *, timeout: float = 3600.0, cwd: str | None = None):
        self.command, self.timeout, self.cwd = command, timeout, cwd

    def execute(self, task: ExperimentTask) -> dict[str, Any]:
        payload = json.dumps({"job_id": task.job_id, "hypothesis": task.hypothesis, "split": task.split})
        completed = subprocess.run(self.command, input=payload, text=True, capture_output=True, timeout=self.timeout, cwd=self.cwd, check=False)
        if completed.returncode:
            raise RuntimeError(f"local worker failed ({completed.returncode}): {completed.stderr[-2000:]}")
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise ValueError("local worker output must be a JSON object")
        if "artifacts" in value:
            integrity = verify_artifacts(value["artifacts"])
            value["artifact_integrity"] = integrity
            if integrity["state"] != "complete":
                raise RuntimeError("worker artifacts failed integrity validation")
        return value


class DockerWorker(LocalPythonWorker):
    """Runs the same JSON worker contract inside a named Docker image."""

    def __init__(self, image: str, command: list[str], *, timeout: float = 3600.0, workdir: str | None = None):
        docker_command = ["docker", "run", "--rm", "--network", "none"]
        if workdir:
            docker_command += ["-v", f"{workdir}:/work:rw", "-w", "/work"]
        super().__init__([*docker_command, image, *command], timeout=timeout)


class HTTPWorker:
    """Calls a remote worker implementing the JSON task contract."""

    def __init__(self, endpoint: str, *, api_key: str | None = None, timeout: float = 3600.0):
        self.endpoint, self.api_key, self.timeout = endpoint, api_key, timeout

    def execute(self, task: ExperimentTask) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "User-Agent": "VerdiWM/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.endpoint, data=json.dumps({"job_id": task.job_id, "hypothesis": task.hypothesis, "split": task.split}).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("HTTP worker output must be a JSON object")
        return value
