# Full Autonomous Loop

The clean system is a composition root around one user-provided model SDK. A
cycle runs the following stages:

1. **Bind and inspect.** `RuntimeBindings` is the single model binding point.
   The adapter reports capabilities, hooks, revision, and evaluator identity.
2. **Probe and portrait.** The probe registry executes available diagnostics and
   produces a fingerprint and a readiness portrait. Probe evolution can ask the
   configured AI provider for new probes when failures leave an unexplained gap.
3. **Select metrics.** `MetricAdvisor` asks the AI provider to choose primary,
   protected, diagnostic, and held-out metrics from available signals. The
   Kernel checks adequacy and never assumes one benchmark is authoritative.
4. **Plan retrieval.** `AutonomousResearchPlanner` chooses search queries and
   adjacent fields from the objective and portrait. It does not contain a fixed
   list of disciplines.
5. **Acquire sources.** `OnlineRetriever` calls any search backend, tries HTML,
   then stages PDFs. If a source cannot be fetched it emits `human_download`
   with the URL and inbox directory; a human can place the file there for the
   next cycle. The retrieval ledger records every outcome.
6. **Extract ideas twice.** Two independent `IdeaRoute` implementations read
   the staged documents. `DualRouteIdeator` normalizes, deduplicates, and keeps
   evidence bases and risks. One route may be a paper extractor and the other a
   code/reproduction extractor.
7. **Design and schedule.** Ideas become bounded jobs with baseline/null,
   candidate, protected metrics, held-out verification, and estimated cost.
   `LocalScheduler` is the reference scheduler; distributed workers can replace
   it without changing contracts.
8. **Execute through the adapter.** Only the adapter touches model runtime,
   data loaders, inference, training, or interventions. Unknown or unsupported
   interventions must return `abstain`, not be silently reinterpreted.
9. **Freeze and verify.** Results are classified as positive, null, harmful, or
   abstain using the selected held-out evaluator and protected metrics.
10. **Project knowledge.** Settled evidence is appended idempotently to the
    knowledge ledger with provenance, verifier digest, and claim boundary.
11. **Replan.** The next cycle retrieves by portrait/fingerprint similarity,
    failures, uncertainty, and information gain. New probes and ideas are
    proposed only where evidence shows a gap.

The fixture adapter demonstrates the wiring offline. It is not scientific
evidence for a real world model. Real network search, AI providers, evaluators,
and workers are injected at runtime and remain outside the Kernel dependency
set.
