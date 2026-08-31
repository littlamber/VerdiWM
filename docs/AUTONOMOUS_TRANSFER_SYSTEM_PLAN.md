# Autonomous Transfer System Development Plan

Status: decision record v1

Date: 2026-08-17

## Purpose

This document records the target design for unattended, evidence-bound
cross-model method research. The system must discover research, formulate and
materialize hypotheses, run bounded experiments, retain useful negative
knowledge, and improve its measurement protocol without requiring routine
human intervention.

The binding implementation order and status checklist for this design are in
`docs/PORTRAIT_FIRST_AUTONOMOUS_RESEARCH_EXECUTION_PLAN.md`. That plan makes a
validated model portrait a fail-closed prerequisite for capability-gap
planning, experiment portfolio construction, and intervention-module
generation. This document remains the architectural decision record; the
execution plan is the operational source of truth for what is implemented
next.

It must do so without turning VerdiWM into a monolithic agent platform. The
design uses small, versioned contracts, the existing durable controller,
existing probe and fingerprint implementations, and immutable evidence.

## Decisions

1. The LLM is the cognitive layer. It may retrieve sources, formulate
   mechanisms, propose experiments, design interface extensions, generate
   isolated patches, and interpret evidence. It is never claim authority,
   GPU admission authority, or evaluator authority.
2. The Kernel owns frozen policy, resource admission, lifecycle receipts,
   promotion, revocation, and persistence. A screen is only a budget gate;
   frozen verification is the only source of a reusable scientific verdict.
3. Shared knowledge is semantic and content-addressed. Runtime paths,
   repository names, checkpoints, executables, and local manifests are
   execution bindings, not shared knowledge.
4. A method's mechanism is distinct from a particular implementation. A
   source-faithful implementation and a newly derived target implementation
   must be represented as different embodiments and must never be conflated.
5. Constitutional evolution may improve measurement only inside a frozen,
   pre-authorized transition policy. It may not relax the immutable evidence
   floor or rewrite historical conclusions.

## Current Baseline And Gaps

The following pieces already exist and should be reused rather than replaced:

- `experiments/ctrl_world_autonomous_transfer_v1/` provides durable discovery,
  SQLite recovery, eight physical GPU leases, screen/confirm/frozen-verifier
  sequencing, and a graph rebuild stage.
- Model Capability IR, Experiment IR, Evidence IR, and the workflow registry
  define a portable control-plane boundary. See
  `docs/INTERMEDIATE_REPRESENTATIONS.md`.
- `wmloop.geometry.portable_experience` rejects runtime bindings and requires
  content-addressed evidence for reusable experience. See
  `docs/TRANSFERABLE_EXPERIENCE.md`.
- The existing probe and fingerprint packages provide target measurements;
  `metric_adequacy.schema.json` and constitutional shadow proposals provide a
  starting point for measurement governance.
- Experiment-package rules already require an owned directory, manifest,
  entrypoint, focused tests, scale receipt, and external output root.

The current autonomous loop is not yet the complete system described here:

- Its local graph projection mainly records provenance, candidates,
  implementation, and terminal evidence. The portable projection now stages a
  source-grounded `MechanismContract` and a separate derived target
  `MethodEmbodiment` after each materialized frozen verdict. A transfer
  boundary is emitted only when independently validated source and target
  fingerprints plus the frozen evaluator binding are supplied; it is never
  inferred from local candidate provenance.
- `graph.json` includes local `input_root` and source paths for audit and
  rebuilding. It is a local audit projection, not a portable shared graph.
- Research proposals and materialized descriptors exist, but there is no
  unified mechanism contract that separates a transferable scientific idea
  from each implementation embodiment.
- A bounded, provider-neutral LLM task adapter and automatic module compiler
  now admit schema-valid module proposals through a trusted ABI registry. The
  first executable ABI is `history_selection_v1`; additional mechanism classes
  still require their own narrow ABI and fixed evaluator binding.
- Missing materializers are currently too coarse. The loop must distinguish an
  interface-extension opportunity, an architecture-bound method, missing data,
  and an operational failure.
- Constitutional change currently requires an external approval quorum. That
  is safe for the present system but does not yet fulfill fully unattended,
  policy-bounded metric evolution.

## Community Baselines And Delivery Roadmap

The project will use mature community protocols as replaceable infrastructure
baselines, not as a claim of scientific novelty. The selection rule is to
reuse a protocol when it solves a mechanical problem better than a local
implementation, while keeping VerdiWM-owned semantics for mechanism identity,
capability matching, evidence strength, transfer boundaries, and promotion.

| Problem | Baseline to study or adapt | VerdiWM-owned boundary |
| --- | --- | --- |
| Typed tool and plugin interfaces | MCP, OpenAPI, JSON Schema | No tool may grant GPU, verdict, or promotion authority |
| Durable execution and recovery | Temporal-style workflows, Dagster/Prefect patterns | Existing controller ledger, leases, receipts, and fail-closed resume |
| Isolated agent implementation | OpenHands/SWE-agent sandbox and patch patterns | Scientific evidence requires frozen evaluation, not just tests passing |
| Artifact and run lineage | MLflow/W&B/DVC/LakeFS practices | Archive/CAS and immutable evidence remain authoritative |
| Literature structure and provenance | arXiv/OpenAlex/Semantic Scholar metadata, GROBID/S2ORC-style parsing, W3C PROV/RO-Crate concepts | Source-grounded mechanism fields, anti-conditions, and target-side falsification |
| Scheduling and search budgets | Optuna/Ray Tune-style ranking and early stopping | Candidate admission, protected metrics, and promotion policy stay in the kernel |

The team must record an explicit compatibility decision before adding a new
dependency: protocol version, adopted subset, adapter contract, failure mode,
security boundary, and removal plan. A mature framework is never allowed to
replace the frozen evaluator, the append-only evidence ledger, or the target
verification boundary. This keeps the system modular and avoids building a
second generic agent framework inside VerdiWM.

The delivery order is intentionally incremental:

1. **Stabilize the evidence contract.** Finish the mechanism/embodiment,
   capability, fingerprint, source-locator, and negative-boundary schemas.
   Add canonical hashing, provenance validation, and fixtures for positive,
   negative, unsupported, and capability-missing sources.
2. **Build a measured source-grounded extraction lane.** Extend the current
   metadata/abstract intake to section-aware evidence records. Every extracted
   mechanism claim must point to source text, distinguish source assertion from
   system inference, list required components and anti-conditions, and emit a
   falsifiable target experiment. Keep extraction ranking-only until it passes
   an independently reviewed corpus of roughly 50--100 papers.
3. **Replace fixed retrieval branches behind the same intake contract.** Add
   failure-to-mechanism normalization, bounded adjacent-domain proposals,
   novelty and expected-information-gain scoring, query budgets, cooldowns,
   and durable rejection records. Compare the new lane with the existing
   failure-driven retriever on the same historical failures before enabling it
   for unattended discovery.
4. **Complete ACWM as a domain pack.** Keep Ctrl-World hooks, datasets,
   rollout metrics, and frozen evaluators outside the kernel. Add a conformance
   receipt and a reproducible domain-pack manifest so the same research loop
   can be instantiated without ACWM-specific branches in the controller.
5. **Prove portability on a second small domain.** Use a deliberately small,
   unlike-ACWM benchmark (for example sequence prediction or optimizer
   adaptation) to exercise onboarding, failure routing, candidate compilation,
   verification, and knowledge projection. Do not claim general RSI before the
   same contracts pass on this second domain.
6. **Add the meta-evaluation loop.** Freeze a benchmark of historical research
   failures and compare retrieval, mechanism extraction, proposal, verifier,
   and scheduler versions under equal budgets. Promote a replacement only when
   it improves useful-candidate yield, evidence quality, duplicate rate, or
   information gain without weakening protected checks. The new module is then
   itself recorded as a derived, versioned research artifact.

Each step has a fail-closed gate and a rollback point. No step may silently
   broaden the model, data, evaluator, or GPU authority of the current ACWM
   campaign. The goal is not to reach a universal RSI system in one release;
   it is to make each layer independently measurable until the ACWM loop can
   safely become a reusable RSI substrate.

## Knowledge Boundary

Two views are intentional and must remain separate.

| View | Contains | May contain local paths? | Authority |
| --- | --- | --- | --- |
| Execution ledger | Controller state, stage receipts, local artifact locations, process and GPU receipts | Yes | Authoritative for local execution and recovery |
| Portable knowledge | Semantic method and model identities, capability/fingerprint summaries, applicability, anti-conditions, protocol identities, CAS evidence references | No | Derived, shareable, and rebuildable |

The current Evidence Graph belongs to the first view because it includes local
source paths. A new Portable Knowledge Projection must be generated only from
validated portable records. It must reject path-like strings and ordinary URLs
in semantic fields, retain only `cas://`, `urn:`, or `sha256:` evidence
references, and use semantic identifiers plus schema versions as node keys.

This means a different checkout, machine, or archive layout can reuse the
knowledge after capability onboarding. It never means that a source-model
result proves a target-model result without the target-side verifier.

### Local-to-shared onboarding

The community graph never reads a user's local source tree. Onboarding is a
one-way local procedure:

```text
local model source and runtime
        -> local scanner and conformance receipt
        -> optional LLM capability-mapping draft
        -> deterministic Capability IR and probe fingerprint validation
        -> path-free portable record staging
        -> community graph projection
```

The LLM may inspect local code only through the local onboarding task and may
propose a semantic hook or capability mapping. It does not publish paths, code
snippets, repository identity, datasets, or checkpoints. A deterministic
conformance run must validate the mapping before its semantic Capability IR or
fingerprint can enter portable staging. Matching a method to a user's local
model is therefore performed locally from semantic capabilities and
fingerprints; the shared graph supplies reusable priors and boundaries.

The matching operation is deliberately a two-input local computation:

```text
portable community graph (mechanisms, capabilities, fingerprints, boundaries)
                              +
local onboarding receipt (the user's model capabilities and probe fingerprint)
                              |
                              v
                 LLM proposes a ranked correspondence
                              |
                              v
          deterministic capability/probe checks accept or reject it locally
```

The LLM may read and explain the portable graph, inspect the local model only
through the user's explicitly authorized onboarding surface, and produce a
candidate correspondence or experiment plan. It cannot copy local files into
the graph, infer a transfer result from textual similarity, or bypass the
frozen evaluator. The graph stores the resulting semantic boundary and CAS
evidence reference only; local paths remain in the execution ledger.

## Thin Architecture

```text
online sources + prior portable evidence
                |
                v
        LLM research tasks
                |
                v
    versioned contracts and static checks
                |
                v
capability/fingerprint match and materialization plan
                |
                v
isolated patch + tests + GPU controller
                |
                v
screen -> confirm -> frozen verifier
                |
                v
execution ledger -> portable evidence -> portable knowledge projection
                ^                                      |
                +----------- bounded next-task ranking-+
```

The only new reusable modules needed are narrowly scoped:

1. **Mechanism contract compiler**
   - Extends the existing research proposal rather than creating a second
     proposal system.
   - Captures the causal mechanism, required and optional capabilities,
     intervention operator, interface/data requirements, prohibited
     substitutions, falsification criterion, required ablations, and source
     evidence.
   - Produces a stable `mechanism_id` from the semantic body.

2. **Fingerprint and transfer-boundary projection**
   - Binds a probe protocol version, controlled intervention doses, context,
     horizon, split, seeds, response vector or digest, uncertainty, and
     diagnostic role.
   - Links a method mechanism to model capability regions, applicability and
     anti-conditions, not merely to a scalar score or a backbone name.
   - Keeps large vectors in content-addressed artifacts; graph nodes store
     semantic summaries and references only.

3. **Isolated materialization service**
   - Has two explicit modes: `source_faithful` and `derived_embodiment`.
   - Generates a patch in a disposable worktree, runs declared static,
     contract, leakage, and CPU smoke tests, and emits an implementation
     revision before GPU admission.
   - A derived embodiment is a new target method with its own claim boundary;
     it is never labeled a paper reproduction.

4. **Portable knowledge projector**
   - Transforms validated Evidence IR, portable experience, mechanism
     contracts, capability summaries, fingerprint summaries, and verifier
     bindings into a path-free graph and query index.
   - Treats the execution ledger and current Evidence Graph as inputs for
     audit, never as direct shared-knowledge exports.

5. **Metric adequacy and constitutional transition service**
   - Evaluates whether the protocol has discriminative power, seed stability,
     sensitivity to known regressions, anti-Goodhart coverage, and incremental
     information beyond the active metric set.
   - Runs new metrics and probes in shadow mode and manages a versioned
     promotion lifecycle without changing frozen historical evidence.

The existing controller remains an orchestrator. It must call these contracts;
it must not acquire research reasoning, graph logic, patch generation, or
metric policy branches of its own.

### First executable transfer loop

The first implementation slice is intentionally small and is now the canonical
path for the Ctrl-World masked intermediate action adapter:

```text
materialize
    -> fit_adapter (adapter-only; frozen Ctrl-World backbone)
    -> screen
    -> confirm
    -> independently frozen verifier
    -> path-free portable knowledge staging
```

`materialize` may emit a pending candidate skeleton, but a pending skeleton is
eligible only for integration smoke. Before screen or confirm, the controller
must produce an external `adapter-state.pt` and a training receipt binding the
candidate parameters, checkpoint digest, training split fingerprint, freeze
proof, trainable parameter scope, optimizer name/steps/learning rate, and state
digest. Any missing, stale, or drifted binding fails closed. Training artifacts
live under the state root's `adapter-training/` tree and never in the source
checkout. A frozen verifier, not the trainer or LLM, is the only authority that
can promote the result into reusable knowledge.

## Contract Requirements

### Mechanism and embodiment

Every admitted idea must contain a `MechanismContract` with:

- source and evidence identities;
- an explicit causal claim and a target-side falsification condition;
- required, optional, and forbidden capability/interface conditions;
- intervention semantics, allowed parameter domain, and protected metrics;
- required negative controls and ablations;
- a materialization class: `source_faithful`, `derived_embodiment`, or
  `architecture_bound`.

The method lifecycle is:

```text
discovered -> formalized -> capability_matched -> materialization_ready
-> patch_validated -> screened -> confirmed -> frozen
```

Terminal and deferred states are evidence, not discarded work:

- `requires_interface_extension`: a concrete interface contract is missing;
  schedule design and implementation of that extension.
- `architecture_bound`: preserve the mechanism and its architectural
  assumptions; do not create a proxy implementation.
- `missing_data_regime`: retain the experiment specification until a matching
  frozen dataset regime is available.
- `operational_failure`: retain operational receipts but create no scientific
  boundary.
- `verified_negative_boundary`: preserve the capability/fingerprint regime,
  regressed protected metric, and frozen evidence.

### Model capability and fingerprint

`verdiwm-model-capability-ir` remains the structural profile. A probe
fingerprint is an evidence-bound behavioral profile, not a replacement for the
Capability IR. Each transfer decision therefore requires both:

- structural compatibility: hooks, execution interfaces, data semantics, and
  backbone/component family; and
- behavioral compatibility: response under a versioned probe protocol,
  including context, dose, horizon, uncertainty, and replication status.

Backbone family may rank candidates. It may not authorize a transfer result.

### LLM tasks and credentials

The LLM receives bounded, schema-validated tasks, not unrestricted shell or
database authority. Initial task outputs are:

- source assessment and semantic mechanism proposal;
- a MechanismContract draft and discriminating experiment plan;
- an interface-extension specification or an isolated patch;
- a concise evidence interpretation and next-task ranking record.

The runtime validates every output before it enters the ledger. External text
is treated as data, never as executable instructions. The LLM may request a
new source or an experiment but cannot self-grant a GPU lease, promote a
result, alter a frozen split, or update the active constitution.

The adapter must use separately provisioned service credentials supplied at
runtime by the deployment. It must never read, copy, log, or persist an
interactive Codex session or bridge credential. Request metadata may record a
provider/model alias and prompt-template digest, but not secrets or hidden
reasoning traces.

The implemented module-generation path is:

```text
idea + work order + Model Capability IR
        -> schema-backed LLM module task
        -> automatic module spec
        -> trusted ABI and capability match
        -> static source policy and exact signature validation
        -> fixed module path, tests, parameters, evaluator, and budget
        -> isolated materialization receipt
        -> candidate compilation
        -> controller-owned GPU screen
```

The LLM does not name `v1`, `v2`, or `v3` candidates. Candidate identity is
derived from the idea and module-spec digest. It cannot choose a repository
path, test command, evaluator, split, metric, GPU budget, or promotion rule.
Unsupported proposals become one of `requires_interface_extension`,
`architecture_bound`, `missing_data_regime`, or `operational_failure`; the
controller does not silently substitute a nearby registered method.

`wmloop.execute.llm_task_adapter` invokes only a separately configured command
and passes credentials only from its explicit deployment allowlist. It rejects
all `CODEX_*`, `LARK_*`, and `LARKSUITE_*` credential inheritance. The optional
`wmloop/execute/json_llm_service_broker.py` forwards the task to an HTTPS JSON
service and expects the service to return the exact task-response contract.
Generated module checks run later in a separate environment with no service
credentials and with CUDA hidden.

Adding a new optimization surface is therefore a modular operation: register
one ABI with its exact callable signature, safe import surface, fixed CPU tests,
candidate parameter domain, evaluator binding, and capability aliases. The
durable controller, materialization transaction, screen/confirm sequence, and
frozen verifier are reused unchanged.

### Constitutional evolution

The Constitution has two layers:

- **Immutable core:** evidence provenance, held-out isolation, no proxy
  substitution, protected-metric non-regression, frozen-verifier-only verdicts,
  and append-only history. No automatic process can change these rules.
- **Evolvable shell:** diagnostic probes, candidate metrics, aggregation,
  thresholds, statistical-power requirements, and guard metrics.

For unattended operation, the immutable core must contain a pre-authorized
transition policy. The policy can permit narrowly defined automatic promotion
only after a candidate passes deterministic static checks, shadow evaluation,
historical calibration, a fresh held-out battery, and a canary. The new
constitution takes effect only for future work. It may re-evaluate old evidence
as a separate view but may never overwrite the verdict made under an earlier
constitution version.

Until that policy and its tests exist, the current external approval quorum
remains active. The system must not claim that constitutional promotion is
fully autonomous before then.

## Repository And Artifact Hygiene

The repository contains source, schemas, configuration, tests, and short
design notes. It must not accumulate generated logs, checkpoints, datasets,
or copied target repositories.

- Each experiment owns one small directory with an engineering manifest,
  README, entrypoint, focused tests, and scale plan. Reusable code moves into
  `wmloop/` only after a second consumer proves the same narrow contract.
- Generated implementations live in disposable worktrees and are represented
  by a patch digest and implementation revision. A target source tree is never
  edited in place.
- Outputs live under declared state roots. A durable receipt and content
  address are written before cleanup; terminal trials remain in Archive/CAS.
- The controller database, local receipts, and local audit graph are runtime
  state. They are excluded from source control and are reproducible from
  versioned code plus immutable artifacts.
- Every new persisted schema receives a schema version, canonical JSON, and
  focused validation tests. No generic agent framework, graph database,
  message broker, or global singleton is introduced for this plan.

## Development Sequence

### Phase 0: freeze the boundary

Document and test the distinction between local audit state and portable shared
knowledge. Add a release check that rejects exporting path-bearing local graph
records as portable knowledge.

Acceptance: a portable projection built from the same evidence is byte-stable
across two different checkout paths; injecting a path, repository name, or
checkpoint filename fails closed.

Implementation status: complete. `wmloop.experiments.portable_knowledge_graph`
builds a path-free graph only from explicitly staged validated documents. The
Ctrl-World controller now emits it beside the local audit graph and never scans
the full local state tree as a community export source.

### Phase 1: first-class mechanism, capability, and fingerprint knowledge

Add the MechanismContract schema/compiler and a probe-fingerprint summary
contract. Extend the autonomous-loop evidence emission and graph projection to
link mechanism, embodiment, Capability IR, probe protocol, fingerprint summary,
applicability, anti-condition, and verifier evidence.

Acceptance: given a verified positive and a verified negative fixture, queries
can identify the method mechanism, required capabilities, measured fingerprint
regime, protected-metric effect, and exact CAS evidence without reading a
runtime path.

Implementation status: the portable contracts and graph relations are complete
for mechanism, embodiment, probe-fingerprint summary, and transfer-boundary
records. After a materialized frozen verdict, the autonomous controller now
validates the bound source assessment and stages path-free mechanism and
embodiment records automatically. It stages a positive or negative transfer
boundary only when both verified fingerprints and the exact frozen policy
digest are explicitly present. The remaining work is to make capability
onboarding emit those validated fingerprints for each new local model; absent
that evidence the system deliberately retains only mechanism and embodiment.

### Phase 2: materialization classes and interface extensions

Replace the coarse materializer gap with the lifecycle states above. Add an
interface-extension contract and an isolated patch runner with manifest,
declared source surface, tests, negative controls, and worktree cleanup.

Acceptance: a missing target hook creates `requires_interface_extension`; a
passing derived implementation carries a new implementation revision and never
masquerades as source-faithful; an architecture-bound method consumes no GPU.

### Phase 3: bounded LLM research and code generation

Introduce one LLM adapter and four schema-validated task types. Give it
source snapshots, portable knowledge, diagnostics, and task budgets. Connect
accepted tasks to the existing proposal/compiler and isolated materializer.

Acceptance: the LLM can generate a source-grounded contract and an isolated
candidate patch; malformed or policy-violating outputs cannot launch code or
GPU work; no secret appears in configuration, receipt, log, or portable graph.

### Phase 3.5: failure-driven autonomous research-domain expansion

Extend retrieval beyond the current fixed failure-to-query branches without
turning the researcher into an unbounded web crawler. The existing seed tracks
(action conditioning, long-horizon memory, training/inference alignment, and
rollout stability) remain the stable starting vocabulary. A bounded domain
expander may propose adjacent mechanism domains when the current failure
evidence is not explained by those tracks or when repeated low-information-gain
cycles indicate a search gap.

The expander must operate in four explicit steps:

1. Normalize a failure into a mechanism-level gap, preserving the original
   failure signatures and protected metrics.
2. Retrieve a small, deterministic set of adjacent domains from the versioned
   mechanism ontology and generate auditable queries for each domain.
3. Score returned sources for mechanism evidence, target-capability fit,
   novelty against the portable knowledge index, and expected information gain.
4. Admit a new domain to the persistent search pool only after repeated,
   source-grounded evidence clears the same research-intake and materialization
   gates as seeded domains.

Domain expansion is advisory: it cannot alter the frozen evaluator, held-out
   split, GPU budget, or promotion policy. Each expansion attempt records its
   parent failure, ontology revision, query set, source assessments, rejected
   candidates, and a cooldown/expiry decision. The first implementation should
   cap new domains and queries per cycle, deduplicate by semantic mechanism
   identity, and retain negative results as reusable knowledge. Candidate
   areas include control and robotics, sequence/state-space modeling,
   uncertainty calibration, offline RL, causal representation learning, and
   optimization, but no area is admitted merely because it is adjacent by name.

Acceptance: a replay fixture with a seeded failure produces the same bounded
   expansion plan byte-for-byte; a genuinely new failure can add a domain
   without code or GPU authority; duplicate or low-evidence domains are
   rejected with durable receipts; and the resulting source records remain
   path-free and queryable by mechanism, capability, and anti-condition.

Implementation status: planned. The current controller already derives
failure-specific queries, but its expansion vocabulary is still static. This
phase is the planned replacement boundary for a future retrieval module; the
existing screen, confirm, and frozen-verifier loop remains unchanged.

### Phase 4: autonomous metric adequacy in shadow mode

Implement MetricAdequacyReport calculation and the immutable-core transition
policy. Start with diagnostic metric additions and probe selection. Promote no
primary or protected metric until shadow and held-out requirements are proven
on a calibration corpus.

Acceptance: the system detects a known metric blind spot, retains both old and
new reports, and refuses a candidate that improves the new score while failing
a preserved protected metric.

Implementation status: policy-bounded future-constitution authorization is
implemented. A v2 proposal carries static, shadow, historical-calibration,
fresh-heldout, and canary evidence references; a frozen transition policy can
authorize only diagnostic-to-primary/guard successor metrics while preserving
the immutable core. The authorization is not a verdict and cannot alter an
active constitution. Metric-adequacy report generation and calibration-corpus
integration are implemented as an evidence-bound, non-authoritative report;
collecting a real calibration corpus remains the next empirical work before
any policy is installed.

### Phase 5: unattended operation and knowledge quality audits

Enable continuous retrieval, bounded LLM planning, materialization, GPU
scheduling, verification, portable projection, and next-task ranking. Run a
periodic audit for duplicate candidates, stale capability records,
non-portable fields, failed cleanup, unverified promotion, and metric-policy
version drift.

Acceptance: after a controller restart, the system resumes without duplicate
GPU work; every terminal item is archived; every reusable item is path-free,
versioned, evidence-bound, and queryable by mechanism/capability/fingerprint
boundary.

## Explicit Non-Goals

- A result graph is not a general-purpose ontology or a replacement for the
  archive.
- An LLM is not allowed to silently alter a model, evaluator, budget, or data
  split.
- A similar backbone does not count as transfer verification.
- Interface innovation does not license uncontrolled edits to target code.
- Automated constitutional evolution does not mean the system can lower its
  own proof standard.

## Definition Of Completion

The system is complete enough for unattended scaled research when an external
source or prior failure can produce a mechanism contract; a new model can be
onboarded using Capability IR plus fingerprint evidence; the system can choose
an exact or clearly labeled derived materialization; all code runs in an
isolated, tested experiment package; a frozen verifier settles the outcome;
and the resulting portable knowledge can be transferred to another machine
without local-path bindings.

Completion of this control plane is not a claim that any particular method
transfers. Such claims remain individual frozen-verifier results.
