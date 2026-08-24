"""Model-agnostic experiment workers.

Workers own execution and artifact production. They do not decide whether an
experiment is scientifically positive.
"""

from __future__ import annotations

from dataclasses import dataclass
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
