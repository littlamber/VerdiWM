"""End-to-end autonomous research composition root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import CapabilityIR, Evidence, Portrait, canonical_digest
from .ideas import AIJsonRoute, AutonomousResearchPlanner, CandidateIdea, DualRouteIdeator
from .knowledge import KnowledgeGraph
from .knowledge_graph import project_model_portrait, project_metric_plan
from .metrics import MetricAdvisor, MetricPlan
from .benchmark_metrics import MetricCatalog, MetricCatalogDiscovery, WorldArenaMetricCatalog
from .probes import ProbeCampaign, ProbeEvolution, ProbeRegistry, ProbeSpec, public_probe_specs
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
        # Public semantic probes are attempted for every adapter.  Unsupported
        # hooks are recorded explicitly and may be materialized by an adapter
        # repair workflow later; they are never treated as healthy responses.
        self.probes = ProbeRegistry(public_probe_specs() + [ProbeSpec("action-sensitivity", "Does the model respond predictably to action changes?", ("action", "horizon"))])
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
        project_model_portrait(
            self.state,
            model_id=capability.model_id,
            revision=capability.revision,
            capabilities=capability.capabilities,
            hooks=capability.hooks,
            architecture_facets=tuple(str(value) for value in report.get("architecture_facets", report.get("architecture", [])) or []),
            probe_results=probe_results,
            portrait_id=portrait.portrait_id,
        )
        run_id = "run-" + canonical_digest({"objective": objective, "portrait_id": portrait.portrait_id})[7:23]
        available_signals = list(report.get("available_signals", report.get("signals", [])) or [])
        catalog: MetricCatalog = WorldArenaMetricCatalog.default()
        external_catalog = report.get("benchmark_metrics")
        if isinstance(external_catalog, list):
            try:
                adapter_catalog = MetricCatalog.from_records(external_catalog, catalog_id=str(report.get("benchmark_catalog_id", "adapter-benchmark-catalog")))
                for definition in adapter_catalog.all():
                    if catalog.get(definition.metric_id) is None:
                        catalog.add(definition)
            except (TypeError, ValueError):
                # An invalid adapter catalog is not silently treated as a
                # WorldArena evaluator; retain the public catalog and audit it.
                pass
        benchmark_plan, metric_plan = self._select_metric_plan(objective, report, available_signals, constraints, catalog)
        adequate, missing = self.metrics.adequate(metric_plan)

        search_plan = None
        documents: list[dict[str, Any]] = []
        benchmark_review: dict[str, Any] = {"status": "not_run"}
        if self.bindings.ai is not None:
            search_plan = AutonomousResearchPlanner(self.bindings.ai).plan(objective, portrait.__dict__)
            if self.retriever is not None:
                benchmark_query = "WorldArena benchmark official evaluator metrics"
                queries = list(search_plan.queries)
                if not any("worldarena" in query.lower() for query in queries):
                    queries.append(benchmark_query)
                acquired = self.retriever.retrieve(queries)
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
                discovered = MetricCatalogDiscovery(self.bindings.ai).discover(documents)
                if discovered.get("state") == "discovered":
                    try:
                        discovered_catalog = MetricCatalog.from_records(discovered.get("metrics", ()), catalog_id=str(discovered.get("catalog_id", "worldarena-discovered")))
                        for definition in discovered_catalog.all():
                            if catalog.get(definition.metric_id) is None:
                                catalog.add(definition)
                        benchmark_plan, metric_plan = self._select_metric_plan(objective, report, available_signals, constraints, catalog)
                        adequate, missing = self.metrics.adequate(metric_plan)
                        benchmark_review["metric_catalog_discovery"] = {key: value for key, value in discovered.items() if key != "metrics"}
                    except (TypeError, ValueError):
                        benchmark_review["metric_catalog_discovery"] = {"state": "abstain", "reason": "invalid discovered metric catalog"}

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
                claim_boundary=str(result.get("claim_boundary", "adapter-provided evidence; requires target-side review.")),
                metric_direction=str(result.get("metric_direction", getattr(metric_plan, "primary_direction", "maximize"))),
                ci95=tuple(result["ci95"]) if isinstance(result.get("ci95"), (list, tuple)) and len(result["ci95"]) == 2 else None,
                split=str(result.get("split", metric_plan.heldout_split)),
                artifact_digest=str(result.get("artifact_digest", result.get("artifact_integrity", {}).get("manifest_digest", ""))),
            )))
        result = {"state": "settled", "run_id": run_id, "capability": capability.__dict__, "portrait": portrait.__dict__, "metrics": metric_plan.__dict__, "benchmark_metric_plan": benchmark_plan, "metrics_adequate": adequate, "metrics_missing": missing, "benchmark_review": benchmark_review, "search_plan": search_plan.__dict__ if search_plan else None, "idea_count": len(ideas), "scheduled": scheduled, "evidence_count": len(evidence)}
        self.state._put("runs", "run_id", {"run_id": run_id, "created_at": "runtime", "objective": objective, "state": "settled", "payload_json": json.dumps(result, sort_keys=True)})
        result["probes"] = probe_results
        return result

    def _select_metric_plan(self, objective: str, report: dict[str, Any], available_signals: list[str], constraints: list[str], catalog: MetricCatalog) -> tuple[dict[str, Any], MetricPlan]:
        benchmark_plan = self.metrics.select_benchmark(objective, report, available_signals, catalog=catalog)
        if benchmark_plan.get("state") != "validated":
            return benchmark_plan, self.metrics.select(objective, list(report.get("capabilities", ())), constraints)
        definitions = tuple(dict(item) for item in benchmark_plan.get("definitions", ()) if isinstance(item, dict))
        definition_by_id = {str(item["metric_id"]): item for item in definitions}
        protected = tuple(str(item) for item in benchmark_plan.get("protected", ()))
        plan = MetricPlan(
            primary=str(benchmark_plan["primary"]),
            protected=protected,
            diagnostic=tuple(str(item) for item in benchmark_plan.get("diagnostic", ())),
            heldout_split=str(report.get("heldout_split", "heldout")),
            rationale=str(benchmark_plan.get("rationale", "")),
            practical_threshold=float(benchmark_plan["practical_threshold"]) if isinstance(benchmark_plan.get("practical_threshold"), (int, float)) else None,
            threshold_rationale=str(benchmark_plan.get("threshold_rationale", "")),
            primary_direction=str(definition_by_id[str(benchmark_plan["primary"])]["direction"]),
            protected_directions=tuple(str(definition_by_id[item]["direction"]) for item in protected),
            benchmark=str(definitions[0].get("benchmark", "")) if definitions else "",
            catalog_digest=str(benchmark_plan.get("catalog_digest", "")),
            metric_definitions=definitions,
            evaluation_order=tuple(str(item) for item in benchmark_plan.get("evaluation_order", ())),
            pilot_metrics=tuple(str(item) for item in benchmark_plan.get("evaluation_stages", {}).get("pilot_metrics", ())),
            promotion_metrics=tuple(str(item) for item in benchmark_plan.get("evaluation_stages", {}).get("promotion_metrics", ())),
            selection_state="validated",
        )
        project_metric_plan(self.state, model_id=str(report["model_id"]), plan=benchmark_plan)
        self.state.append_knowledge_record({"model_id": report["model_id"], "objective": objective, "metric_plan": benchmark_plan}, record_type="metric_selection", layer="L4", status="validated")
        return benchmark_plan, plan

    def replan(self, campaign: dict[str, Any], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Retrieve and ideate a fresh batch after a non-positive campaign.

        This callback is intentionally side-effect bounded: it only stages
        sources and ideas.  Training/evaluation remains the CampaignSupervisor
        responsibility, and an absent provider/retriever yields no guesses.
        """
        if self.bindings.ai is None or self.retriever is None or self.ideator is None:
            return []
        context = context or {}
        objective = str(campaign.get("objective", ""))
        portrait = campaign.get("portrait", {}) if isinstance(campaign.get("portrait"), dict) else {}
        plan = AutonomousResearchPlanner(self.bindings.ai).plan(objective, portrait)
        acquired = self.retriever.retrieve(list(plan.queries))
        ingest = DocumentIngestor(Path(self.bindings.state_root) / "retrieval" / "inbox")
        documents: dict[str, dict[str, Any]] = {}
        for item in acquired:
            if item.local_path and Path(item.local_path).is_file():
                doc = ingest.ingest_file(Path(item.local_path), source_url=item.url)
                documents[doc.document_id] = doc.__dict__
            elif item.status == "human_download":
                documents["remote:" + str(item.url)] = item.__dict__
        for doc in ingest.scan_inbox():
            documents[doc.document_id] = doc.__dict__
        for payload in documents.values():
            document_id = str(payload.get("document_id") or "remote-" + canonical_digest(payload)[7:23])
            payload.setdefault("document_id", document_id)
            self.state.put_document(document_id, payload)
        ideas = self.ideator.extract(list(documents.values()), {"objective": objective, "portrait": portrait, "round": context.get("round")})
        return [idea.__dict__ for idea in ideas]
