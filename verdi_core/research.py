"""End-to-end autonomous research composition root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import CapabilityIR, Evidence, Portrait, canonical_digest
from .ideas import AIJsonRoute, AutonomousResearchPlanner, CandidateIdea, DualRouteIdeator
from .knowledge import KnowledgeGraph
from .metrics import MetricAdvisor
from .probes import ProbeCampaign, ProbeEvolution, ProbeRegistry, ProbeSpec
from .retrieval import RetrievalLedger, RetrievalRequest, OnlineRetriever
from .runtime import RuntimeBindings
from .scheduler import ExperimentJob, LocalScheduler
from .storage import SQLiteState
from .workers import AdapterWorker, ExperimentTask, Worker
from .evaluators import GenericEvaluator
from .ingestion import DocumentIngestor


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
        self.probes = ProbeRegistry([ProbeSpec("action-sensitivity", "Does the model respond predictably to action changes?", ("action", "horizon"))])
        self.state = SQLiteState(Path(bindings.state_root) / "knowledge" / "knowledge.sqlite3")
        self.scheduler = LocalScheduler(float(bindings.options.get("budget", 1.0)), state=self.state)

    def run_cycle(self, *, objective: str, constraints: list[str] | None = None) -> dict[str, Any]:
        constraints = constraints or []
        report = self.bindings.model.inspect()
        capability = CapabilityIR(str(report["model_id"]), str(report["revision"]), tuple(sorted(map(str, report.get("capabilities", [])))), tuple(sorted(map(str, report.get("hooks", [])))), str(report["evaluator_id"]))
        probe_results = ProbeCampaign(self.probes, self.state).run(self.bindings.model)
        fingerprint_ids = tuple("fingerprint-" + canonical_digest(result)[7:23] for result in probe_results if result.get("status") == "evaluated")
        portrait = Portrait(
            portrait_id="portrait-" + canonical_digest({"model": capability.model_id, "revision": capability.revision})[7:23],
            model_id=capability.model_id,
            capability_digest=canonical_digest(capability.__dict__),
            fingerprint_ids=fingerprint_ids,
            readiness="ready_for_experiment",
        )
        run_id = "run-" + canonical_digest({"objective": objective, "portrait_id": portrait.portrait_id})[7:23]
        metric_plan = self.metrics.select(objective, list(capability.capabilities), constraints)
        adequate, missing = self.metrics.adequate(metric_plan)

        search_plan = None
        documents: list[dict[str, Any]] = []
        benchmark_review: dict[str, Any] = {"status": "not_run"}
        if self.bindings.ai is not None:
            search_plan = AutonomousResearchPlanner(self.bindings.ai).plan(objective, portrait.__dict__)
            if self.retriever is not None:
                acquired = self.retriever.retrieve(list(search_plan.queries))
                RetrievalLedger(self.bindings.state_root).append(acquired)
                ingest = DocumentIngestor(Path(self.bindings.state_root) / "retrieval" / "inbox")
                ingested = []
                for document in acquired:
                    if document.local_path and Path(document.local_path).is_file():
                        ingested.append(ingest.ingest_file(Path(document.local_path), source_url=document.url))
                ingested.extend(ingest.scan_inbox())
                unique_documents: dict[str, Any] = {}
                for document in ingested:
                    unique_documents[document.document_id] = document
                for document in unique_documents.values():
                    payload = document.__dict__
                    self.state.put_document(document.document_id, payload)
                    documents.append(payload)
                for document in acquired:
                    if document.status == "human_download" or not document.local_path:
                        documents.append(document.__dict__)
                benchmark_review = self.metrics.review_benchmarks(objective, documents[:20])

        ideas: list[CandidateIdea] = []
        if self.ideator is not None and documents:
            ideas = self.ideator.extract(documents, {"objective": objective, "portrait": portrait.__dict__, "metrics": metric_plan.__dict__})
            for idea in ideas:
                self.state.put_idea(idea.idea_id, idea.__dict__)

        jobs = [ExperimentJob("job-" + idea.idea_id, idea.idea_id, 0.1, {"idea": idea.__dict__, "portrait_id": portrait.portrait_id, "run_id": run_id}) for idea in ideas if adequate]
        for job in jobs:
            self.state.put_experiment(job.job_id, {"run_id": run_id, "hypothesis_id": job.hypothesis_id, "payload": job.payload})

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
        result = {"state": "settled", "run_id": run_id, "capability": capability.__dict__, "portrait": portrait.__dict__, "metrics": metric_plan.__dict__, "metrics_adequate": adequate, "metrics_missing": missing, "benchmark_review": benchmark_review, "search_plan": search_plan.__dict__ if search_plan else None, "idea_count": len(ideas), "scheduled": scheduled, "evidence_count": len(evidence)}
        self.state._put("runs", "run_id", {"run_id": run_id, "created_at": "runtime", "objective": objective, "state": "settled", "payload_json": json.dumps(result, sort_keys=True)})
        result["probes"] = probe_results
        return result
