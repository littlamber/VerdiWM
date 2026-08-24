"""Small scheduler contract; production workers can replace LocalScheduler."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Any

from .storage import SQLiteState


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    hypothesis_id: str
    estimated_cost: float
    payload: dict[str, Any]


class LocalScheduler:
    def __init__(self, budget: float, *, state: SQLiteState | None = None):
        self.remaining = budget
        self.state = state

    def run(self, jobs: list[ExperimentJob], worker: Callable[[ExperimentJob], dict[str, Any]], *, retries: int = 0) -> list[dict[str, Any]]:
        results = []
        for job in jobs:
            if self.state:
                self.state.put_experiment(job.job_id, job.payload, "queued")
            if job.estimated_cost > self.remaining:
                results.append({"job_id": job.job_id, "state": "budget_exhausted"})
                if self.state:
                    self.state.put_experiment(job.job_id, job.payload, "budget_exhausted")
                continue
            self.remaining -= job.estimated_cost
            try:
                result = None
                last_error = None
                for attempt in range(retries + 1):
                    try:
                        result = worker(job)
                        break
                    except Exception as exc:
                        last_error = exc
                if result is None:
                    raise last_error or RuntimeError("worker returned no result")
                results.append({"job_id": job.job_id, "state": "settled", "result": result})
                if self.state:
                    self.state.put_experiment(job.job_id, {**job.payload, "result": result}, "settled")
            except Exception as exc:  # worker isolation boundary
                results.append({"job_id": job.job_id, "state": "runtime_failed", "error": str(exc)})
                if self.state:
                    self.state.put_experiment(job.job_id, {**job.payload, "error": str(exc)}, "runtime_failed")
        return results

    def resume(self, worker: Callable[[ExperimentJob], dict[str, Any]], *, retries: int = 0) -> list[dict[str, Any]]:
        """Re-run queued or runtime_failed jobs from SQLite after a restart."""
        if self.state is None:
            raise RuntimeError("resume requires SQLiteState")
        jobs = []
        for row in self.state.list_rows("experiments", limit=100000):
            if row["state"] not in {"queued", "runtime_failed"}:
                continue
            payload = json.loads(row["payload_json"])
            jobs.append(ExperimentJob(row["experiment_id"], str(row.get("hypothesis_id") or ""), float(payload.get("estimated_cost", 0.1)), payload))
        return self.run(jobs, worker, retries=retries)
