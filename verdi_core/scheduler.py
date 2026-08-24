"""Small scheduler contract; production workers can replace LocalScheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    hypothesis_id: str
    estimated_cost: float
    payload: dict[str, Any]


class LocalScheduler:
    def __init__(self, budget: float):
        self.remaining = budget

    def run(self, jobs: list[ExperimentJob], worker: Callable[[ExperimentJob], dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for job in jobs:
            if job.estimated_cost > self.remaining:
                results.append({"job_id": job.job_id, "state": "budget_exhausted"})
                continue
            self.remaining -= job.estimated_cost
            try:
                results.append({"job_id": job.job_id, "state": "settled", "result": worker(job)})
            except Exception as exc:  # worker isolation boundary
                results.append({"job_id": job.job_id, "state": "runtime_failed", "error": str(exc)})
        return results
