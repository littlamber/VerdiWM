"""Model-agnostic experiment workers.

Workers own execution and artifact production. They do not decide whether an
experiment is scientifically positive.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import urllib.request
from typing import Any, Protocol

from .adapter import ModelAdapter


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
