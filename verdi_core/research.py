"""End-to-end autonomous research composition root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import CapabilityIR, Evidence, Portrait, canonical_digest
from .ideas import AIJsonRoute, AutonomousResearchPlanner, CandidateIdea, DualRouteIdeator
from .knowledge import KnowledgeGraph
from .metrics import MetricAdvisor
from .probes import ProbeEvolution, ProbeRegistry
from .retrieval import RetrievalLedger, RetrievalRequest, OnlineRetriever
from .runtime import RuntimeBindings
from .scheduler import ExperimentJob, LocalScheduler
from .workers import AdapterWorker, ExperimentTask, Worker
from .evaluators import GenericEvaluator


class ResearchSystem:
    """All non-model modules wired around one user-supplied adapter."""

    def __init__(self, bindings: RuntimeBindings, *, retriever: OnlineRetriever | None = None, ideator: DualRouteIdeator | None = None, worker: Worker | None = None, evaluator: GenericEvaluator | None = None):
        self.bindings = bindings
        self.retriever = retriever
        self.ideator = ideator or (DualRouteIdeator((AIJsonRoute(bindings.ai, "paper_extractor"), AIJsonRoute(bindings.ai, "code_extractor"))) if bindings.ai else None)
        self.worker = worker or AdapterWorker(bindings.model)
        self.evaluator = evaluator or GenericEvaluator()
        self.metrics = MetricAdvisor(bindings.ai)
        self.probe_evolution = ProbeEvolution(bindings.ai)
        self.probes = ProbeRegistry()
        self.scheduler = LocalScheduler(float(bindings.options.get("budget", 1.0)))

    def run_cycle(self, *, objective: str, constraints: list[str] | None = None) -> dict[str, Any]:
        constraints = constraints or []
        report = self.bindings.model.inspect()
        capability = CapabilityIR(str(report["model_id"]), str(report["revision"]), tuple(sorted(map(str, report.get("capabilities", [])))), tuple(sorted(map(str, report.get("hooks", [])))), str(report["evaluator_id"]))
        portrait = Portrait(
            portrait_id="portrait-" + canonical_digest({"model": capability.model_id, "revision": capability.revision})[7:23],
            model_id=capability.model_id,
            capability_digest=canonical_digest(capability.__dict__),
            fingerprint_ids=(),
            readiness="ready_for_experiment",
        )
        metric_plan = self.metrics.select(objective, list(capability.capabilities), constraints)
        adequate, missing = self.metrics.adequate(metric_plan)

        search_plan = None
        documents: list[dict[str, Any]] = []
        if self.bindings.ai is not None:
            search_plan = AutonomousResearchPlanner(self.bindings.ai).plan(objective, portrait.__dict__)
            if self.retriever is not None:
                acquired = self.retriever.retrieve(list(search_plan.queries))
                RetrievalLedger(self.bindings.state_root).append(acquired)
                documents = [document.__dict__ for document in acquired]

        ideas: list[CandidateIdea] = []
        if self.ideator is not None and documents:
            ideas = self.ideator.extract(documents, {"objective": objective, "portrait": portrait.__dict__, "metrics": metric_plan.__dict__})

        jobs = [ExperimentJob("job-" + idea.idea_id, idea.idea_id, 0.1, {"idea": idea.__dict__, "portrait_id": portrait.portrait_id}) for idea in ideas if adequate]

        def worker(job: ExperimentJob) -> dict[str, Any]:
            payload = job.payload["idea"]
            payload = {**payload, "hypothesis_id": job.hypothesis_id}
            artifacts = self.worker.execute(ExperimentTask(job.job_id, payload, metric_plan.heldout_split))
            return self.evaluator.evaluate(artifacts, split=metric_plan.heldout_split, metrics=metric_plan.__dict__)

        scheduled = self.scheduler.run(jobs, worker)
        graph = KnowledgeGraph(Path(self.bindings.state_root) / "knowledge")
        evidence = []
        for item in scheduled:
            if item["state"] != "settled":
                continue
            result = item["result"]
            evidence.append(graph.append(Evidence(
                evidence_id="evidence-" + canonical_digest({"job": item["job_id"], "result": result})[7:23],
                experiment_id=item["job_id"], model_id=capability.model_id,
                hypothesis_id=str(item["job_id"]), outcome=str(result.get("outcome", "abstain")),
                delta=float(result.get("delta", 0.0)), protected_ok=bool(result.get("protected_ok", False)),
                verifier_digest=canonical_digest({"evaluator": self.evaluator.evaluator_id, "split": metric_plan.heldout_split}),
                claim_boundary="adapter-provided evidence; requires target-side review.",
            )))
        return {"state": "settled", "capability": capability.__dict__, "portrait": portrait.__dict__, "metrics": metric_plan.__dict__, "metrics_adequate": adequate, "metrics_missing": missing, "search_plan": search_plan.__dict__ if search_plan else None, "idea_count": len(ideas), "scheduled": scheduled, "evidence_count": len(evidence)}
