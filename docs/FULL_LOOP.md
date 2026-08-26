# Full Autonomous Loop

> Status: this document describes the intended composition and the currently
> available provider interfaces. The CLI fixture path is fully runnable; the
> real web/AI/model path is an alpha integration and still requires a concrete
> search backend, document parser, model adapter, and domain evaluator.

The clean system is a composition root around one user-provided model SDK. A
cycle runs the following stages:

For restartable multi-idea campaigns, `CampaignSupervisor` persists the same
stage transitions and attempt receipts in the kernel SQLite state. The CLI
entry points are:

```text
verdi campaign run --state-root ... --run-id ... --model-id ... --objective ... --ideas ideas.json --runner module:function
verdi campaign autonomous-run --state-root ... --run-id ... --model-id ... --objective ... --ideas ideas.json --runner module:function --worktree-root ... --output-root ...
verdi campaign resume --state-root ... --run-id ... --model-id ... --runner module:function --watch
verdi campaign status --state-root ... --run-id ...
verdi campaign release-human --state-root ... --run-id ... --idea-id ... --labels labels.json
```

Each idea advances through static/environment/GPU smoke, short training,
replication, adaptive full training, and held-out evaluation. A positive result
must include the configured number of independent replicates; otherwise it is
settled as `abstain`. A full-train runner can return `continue_long_train` to
request another checkpoint interval, allowing the held-out controller to stop
at plateau or near-overfit while retaining the best checkpoint. Ideas waiting
for video labels pause independently and do not block unrelated ideas.

1. **Bind and inspect.** `RuntimeBindings` is the single model binding point.
   The adapter reports capabilities, hooks, revision, and evaluator identity.
2. **Probe and portrait.** The probe registry executes available diagnostics and
   produces a fingerprint and a readiness portrait. Probe evolution can ask the
   configured AI provider for new probes when failures leave an unexplained gap.
3. **Discover and select metrics.** `MetricCatalogDiscovery` turns pinned
   benchmark documents/code into provenance-tagged metric records. The AI then
   selects primary, protected, diagnostic, and held-out metrics from only the
   records compatible with the portrait, capabilities, and observed data. The
   Kernel rejects unavailable or non-ground-truth metrics as formal verdict
   sources, stages lower-cost pilots before expensive long-horizon checks, and
   never assumes one benchmark is authoritative.
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
   interventions must return `abstain`, not be silently reinterpreted. Environment
   preflight, deterministic seeds, retry receipts, and artifact hashes are
   recorded before promotion.
9. **Repair and continue.** If a failure is actionable, the configured AI may
   propose a scoped patch. The kernel applies it in a fresh detached worktree,
   runs tests against the patched worktree, and only then resumes the experiment.
10. **Adaptive long training.** An early held-out improvement promotes the
   candidate to continued training. Checkpoint evaluations keep the best
   held-out result and stop at a plateau, repeated train-improves/held-out-
   worsens pattern, or an explicit resource cap. Training loss alone never
   promotes or stops a candidate.
11. **Freeze and verify.** Results are classified as positive, null, harmful, or
   abstain using direction-aware primary metrics, protected metrics, paired
   replicates, and frozen uncertainty estimates.
12. **Project knowledge.** Settled evidence is appended idempotently to the
   knowledge ledger with provenance, verifier digest, and claim boundary.
13. **Replan.** The next cycle retrieves by portrait/fingerprint similarity,
    failures, uncertainty, and information gain. New probes and ideas are
    proposed only where evidence shows a gap.

## Layered knowledge and transfer routing

The knowledge projection is split into six stable layers: ontology (`L0`),
model portraits and probe fingerprints (`L1`), methods and sources (`L2`),
experiments and evidence (`L3`), transfer assessments (`L4`), and provenance
(`L5`). SQLite stores the local query projection; append-only records are the
community merge surface. `graph.json` and `transfer_index.json` are portable
exports, and `graph.html` is a dependency-free interactive viewer.

Transfer routing compares structured architecture facets independently from
probe-derived diagnostic dimensions. A diagnostic match can queue a bounded
target experiment even when backbones differ. Missing hooks trigger an
AI-authored adapter/plugin or isolated model-worktree materialization attempt;
only a failed materialization or failed conformance blocks execution. A graph
match is never target evidence: the target must still pass the frozen evaluator
and replication gate.

The fixture adapter demonstrates the wiring offline. It is not scientific
evidence for a real world model. Real network search, AI providers, evaluators,
and workers are injected at runtime and remain outside the Kernel dependency
set.

`autonomous-run` is the single-command composition path: the caller starts the
campaign, while the configured AI receives bounded tool actions for repair and
retry. Pass `--replanner module:function` when a research adapter is available
so an all-non-positive batch can trigger another retrieval/ideation round.

## Evaluator vs Worker

The **experiment worker** performs the expensive action: it may train a
candidate, run rollouts, render predictions, or apply an inference-time
intervention. It returns raw artifacts and measurements through the model
adapter. The **evaluator** is the frozen judge: it loads the held-out split,
computes the selected metrics, checks protected metrics, and emits the outcome
classification. A worker can be GPU/distributed while an evaluator can be CPU
or a separate service. Keeping them separate prevents a training process from
grading itself with mutable data or metrics.

All AI roles use one OpenAI-compatible provider. Configure
`VERDI_AI_BASE_URL`, `VERDI_AI_API_KEY`, and `VERDI_AI_MODEL`; planner,
`paper_extractor`, `code_extractor`, metric advisor, and probe evolver calls
share that endpoint while retaining distinct role prompts and audit fields.
