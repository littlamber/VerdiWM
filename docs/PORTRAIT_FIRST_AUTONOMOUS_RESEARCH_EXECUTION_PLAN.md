# Portrait-First Autonomous Research Execution Plan

Status: binding implementation baseline

Control-plane implementation: complete (WP1 through WP10)

Last verified: 2026-08-18

## Purpose

This document turns the autonomous-transfer architecture into a strict,
resumable implementation sequence. It exists so that later work does not lose
the system boundary or reorder critical steps when conversational context is
missing.

The central rule is:

> Observe the target before acting on it. Build and validate a model portrait,
> identify evidence-backed capability gaps, and only then reuse, compose, or
> manufacture intervention modules.

This plan refines `docs/AUTONOMOUS_TRANSFER_SYSTEM_PLAN.md`. If the two
documents appear inconsistent, stop and amend the decision record before
implementation. Do not silently choose a different architecture.

## Agreed System Boundary

VerdiWM is a lightweight, modular autonomous-research control plane. It is not
a monolithic general agent and not a collection of manually designed `v1`,
`v2`, and `v3` methods.

The plugin rule is:

> Every executable capability may be a plugin. Scientific and operational
> authority remains in the Kernel.

Plugins may provide scanners, probes, adapters, transforms, interventions,
training workflows, inference workflows, evidence projections, and bounded
planning policies. Generated code never owns evaluator selection, held-out
splits, metrics, GPU budgets, promotion, revocation, or constitutional policy.

The privileged Kernel remains small and owns:

- schema and semantic validation;
- immutable evaluator and split bindings;
- side-effect and authority admission;
- GPU leases and experiment budgets;
- durable lifecycle receipts and recovery;
- frozen verification and promotion;
- Archive/CAS writes, revocation, and portable projection admission.

An execution harness inspired by an "everything is a plugin" design can be a
transport and lifecycle layer. It is not the scientific architecture. The
scientific architecture is the combination of a model portrait, capability
graph, gap planner, module manufacturer, dependency composer, admission
Kernel, frozen verifier, and portable knowledge projection.

## End-To-End Mandatory Order

```text
User Intent
    -> Goal IR
    -> read-only model onboarding
    -> structural capability profile
    -> diagnostic probe coverage planning
    -> evidence-bound behavioral fingerprints
    -> Model Portrait + Portrait Readiness Receipt
    -> capability requirement graph
    -> capability gap detection
    -> reuse and composition of admitted modules
    -> manufacture only missing leaf capabilities
    -> experiment portfolio and resource admission
    -> screen -> confirm -> frozen verification
    -> local execution ledger
    -> path-free portable knowledge projection
    -> portrait and knowledge update
    -> next-task ranking
```

The order fails closed:

- Missing structural evidence schedules onboarding or conformance work.
- Missing behavioral evidence schedules an existing probe or a bounded probe
  extension.
- Missing interfaces produce an `InterfaceExtensionSpec`.
- Missing data produce `missing_data_regime`.
- Architecture mismatch produces `architecture_bound`.
- Operational failures produce receipts but no scientific boundary.
- No optimization module is generated merely because the LLM can write one.

## Model Portrait Contract

The portrait is a versioned aggregate over multiple evidence classes. It is
not a single checkpoint hash and not an LLM-written description.

### Portrait layers

1. **Identity layer**
   - semantic model family and content revision;
   - configuration, weight, and source-tree digests where disclosure is
     permitted;
   - dataset and task semantic identities;
   - parent portrait when the model was derived from an earlier state.

2. **Structural layer**
   - inputs, outputs, execution interfaces, topology classes, and tensor/data
     semantics;
   - available, unavailable, and unknown hooks;
   - trainable surfaces and frozen surfaces;
   - adapter, rollout, evaluation, and inference capabilities.

3. **Behavioral layer**
   - response under versioned controlled probes;
   - dose, context, horizon, split, seed, replication, and uncertainty;
   - sensitivity to action, history, noise, temporal alignment, and other
     admitted diagnostic dimensions;
   - response digests for large vectors stored in CAS.

4. **Failure layer**
   - horizon, task, action dimension, data regime, or intervention region in
     which protected behavior degrades;
   - verified negative boundaries and retained counterexamples;
   - distinctions among scientific failure, missing evidence, and operational
     failure.

5. **Operational layer**
   - measured memory, throughput, determinism, supported precision, and
     parallel execution constraints;
   - evidence-backed cost ranges used by the scheduler;
   - no hostnames, device numbers, process identifiers, or local paths in the
     portable portrait.

6. **Coverage layer**
   - known, unknown, conflicting, stale, and insufficiently replicated fields;
   - required probe coverage for the current Goal IR;
   - a readiness state and explicit blockers.

### Portrait readiness

A `PortraitReadinessReceipt` is goal-relative. A model does not need every
possible probe before any work can start, but it must cover the structural and
behavioral dimensions needed to distinguish the current hypotheses.

Permitted states are:

- `ready_for_gap_planning`;
- `requires_static_onboarding`;
- `requires_probe_coverage`;
- `requires_interface_extension`;
- `requires_evaluator_binding`;
- `conflicting_evidence`;
- `stale_portrait`.

Only `ready_for_gap_planning` may enter intervention planning. The receipt
binds the Goal IR digest, Capability IR digest, fingerprint summaries, coverage
policy, evidence references, and portrait digest.

### Portrait versioning

Portraits are append-only. Training, data changes, code changes, or a new
checkpoint produce a new portrait identity. A new portrait links to its parent
and to the admitted intervention that caused the transition. Historical
portraits and verdicts are never overwritten.

## Observation Before Intervention

The module manufacturer has two distinct modes.

### Observation module manufacturing

This mode fills a portrait blind spot. It may propose or generate:

- a read-only scanner or connector;
- a data-semantic adapter needed to run an existing probe;
- a diagnostic probe implementation under an admitted probe ABI;
- a hook extension with conformance tests;
- a response summarizer whose output remains non-authoritative until bound to
  the admitted protocol.

Generated observation modules may not redefine the protected metrics or
promote their own probe. Novel probes and metrics begin in shadow mode.

### Intervention module manufacturing

This mode acts on an evidence-backed hypothesis after portrait readiness. It
may generate a bounded implementation of an admitted ABI or produce an
`InterfaceExtensionSpec` when no ABI fits. It may not select its evaluator,
split, repository path, tests, parameter budget, or promotion policy.

The manufacturer always tries, in order:

1. reuse an already admitted module;
2. compose compatible admitted modules;
3. instantiate a registered ABI;
4. request a narrow interface extension;
5. retain an explicit unsupported or architecture-bound result.

It manufactures only missing leaf capabilities, not an entire replacement
workflow.

## Capability And Plugin Protocol

The existing automatic-module ABI registry will evolve into a
capability-oriented registry. Each executable module declares:

- stable module and ABI identity;
- typed input and output ports;
- semantic `provides` capabilities;
- semantic `requires` capabilities;
- model-family constraints only where behavior truly depends on them;
- side-effect class and authority level;
- deterministic or measured cost model;
- allowed imports and execution surface;
- fixed conformance and negative tests;
- evaluator and admission-suite bindings owned by the Kernel;
- portability, license, and content-addressed implementation metadata.

Authority levels are:

- **L0:** pure transforms and summarizers; automatic admission is possible
  after fixed tests.
- **L1:** read-only scanners, connectors, hooks, and diagnostic adapters;
  sandbox and conformance receipts are required.
- **L2:** training, writes, target mutation, processes, and GPU execution;
  controller-owned work orders and leases are required.
- **L3:** evaluators, protected metrics, promotion, revocation, and
  constitution; generated code receives no direct L3 authority.

The dependency composer resolves compatible ABI versions, verifies the full
capability closure, rejects cycles or unsatisfied ports, and emits a
`ModuleCompositionReceipt`. No dynamically generated module is imported into
the controller process during admission.

## Gap Planning And Automatic Hypothesis Design

The planner consumes four inputs:

```text
Goal IR
+ Model Portrait
+ portable community knowledge
+ local budget and authority policy
```

It emits a `CapabilityRequirementGraph` whose nodes are required observations,
interfaces, data regimes, interventions, evaluations, and evidence conditions.
The gap detector marks every node as satisfied, reusable, composable,
manufacturable, interface-extension-required, data-blocked, or
architecture-bound.

After dependency closure, the hypothesis planner emits an
`ExperimentPortfolio`. Candidates are identified by semantic content digests,
not by LLM-selected version numbers. Every candidate contains:

- causal mechanism and expected portrait change;
- structural and behavioral applicability conditions;
- falsification criterion;
- negative controls and discriminating ablations;
- protected metrics and frozen evaluator binding;
- expected cost, information gain, and selection reason;
- required modules and unresolved risks;
- artifact and cleanup policy.

The planner must prefer experiments that distinguish competing mechanisms. It
must not produce cosmetic parameter variants merely to fill GPU slots.

## Resource Portfolio Policy

GPU use is a portfolio decision, not a default preference for distributed
training.

For the current eight-GPU Ctrl-World campaign, the initial policy is:

- reserve two physical GPUs for continued DROID-to-Ctrl-World conversion;
- use up to six GPUs for independent, diverse, bounded hypothesis screens;
- include required baselines, negative controls, and replications in the
  admitted portfolio rather than treating all six slots as novel methods;
- after screen evidence, reallocate GPUs to confirmation or distributed
  training only when the scale plan shows that it is more informative than
  continuing independent experiments.

This allocation belongs in a versioned campaign budget receipt. It must not be
hard-coded into the generic Kernel. Physical GPU identity, observed activity,
exit status, artifacts, and cleanup remain receipt-bound.

## Portable Community Knowledge

The community graph stores semantic, evidence-bound knowledge. It never reads
a user's local source tree and never treats a local audit graph as a community
export.

### Portable node types

- `ModelCapabilityProfile`;
- `ModelPortrait` and `ProbeFingerprintSummary`;
- `MechanismContract`;
- `MethodEmbodiment`;
- portable `ModuleManifest` and ABI identity;
- `EvidenceRecord` and frozen verifier binding;
- `ApplicabilityBoundary` and verified negative boundary;
- `ProtocolContract` for probes, metrics, and evaluators;
- `TransformationContract` for semantic dataset conversions;
- revocation, deprecation, and supersession records.

### Required graph relations

```text
portrait --has_capability--> capability
portrait --has_fingerprint--> fingerprint
portrait --derived_from--> prior_portrait
portrait --changed_by--> admitted_embodiment
mechanism --requires/forbids--> capability
embodiment --implements--> mechanism
module --provides/requires--> capability
evidence --evaluates--> embodiment_in_portrait_context
verdict --establishes--> applicability_or_negative_boundary
fingerprint --measured_by--> probe_protocol
transform --maps--> source_and_target_data_semantics
```

### Local-only information

The following remain in the execution ledger or CAS sidecars and are rejected
from semantic graph fields:

- absolute or relative runtime paths and repository layout;
- checkpoint filenames and local manifest locations;
- environment variables, credentials, hostnames, GPU numbers, and process
  identifiers;
- raw user conversations and hidden reasoning traces;
- unlicensed code, data, or weights;
- unverified LLM claims and context-free leaderboard scores.

Large response vectors, code bundles, logs, and evidence packages remain in
content-addressed storage. Portable records use only `cas://`, `urn:`, or
`sha256:` references plus license, provenance, schema, and verification
metadata.

Evidence from model A is a prior for model B, never a verdict for model B.
Target-side portrait matching, bounded experimentation, and frozen verification
are always required before a target or transfer boundary is promoted.

## Work Packages And Strict Execution Order

Status values are `complete`, `in_progress`, `pending`, and `blocked`. A work
package may move to `complete` only after its acceptance criteria and focused
tests pass.

### WP0: Freeze the architecture and execution baseline

Status: complete

Deliverables:

- this binding execution plan;
- link from the architecture decision record;
- repository rule requiring future work to read and follow the plan.

Acceptance:

- the portrait-first gate, plugin/Kernel authority boundary, community-memory
  boundary, and eight-GPU campaign policy are recorded explicitly;
- later implementation can identify the first uncompleted dependency without
  relying on conversational context.

### WP1: Model Portrait and readiness contracts

Status: complete

Deliverables:

- `model_portrait.schema.json`;
- `portrait_readiness_receipt.schema.json`;
- a deterministic portrait builder that combines Model Capability IR,
  probe-fingerprint summaries, operational evidence, and coverage state;
- canonical identity, tamper detection, append-only parent linkage, and
  portability validation.

Acceptance:

- the same semantic evidence under two checkout paths produces the same
  portrait digest;
- path-bearing semantic fields fail closed;
- missing or conflicting evidence produces an explicit non-ready state;
- a model revision cannot reuse a stale portrait silently.

Tests:

- builder and schema unit tests;
- tamper, path-independence, stale-revision, conflicting-evidence, and parent
  transition tests.

Implementation evidence:

- `wmloop.control.model_portrait` deterministically builds and validates
  path-free portraits plus goal-relative readiness receipts;
- `model_portrait.schema.json` and
  `portrait_readiness_receipt.schema.json` are included in the wheel;
- focused Capability IR, fingerprint, portrait, and portable-knowledge tests
  pass;
- `scripts/ci/check_control_plane.sh` passes with 477 tests and 5 declared
  environment-dependent skips.

### WP2: Portrait-readiness gate in the autonomous controller

Status: complete; depends on WP1

Deliverables:

- Goal IR to required-portrait-coverage compilation;
- a controller transition that requires a bound readiness receipt before gap
  planning or intervention-module generation;
- durable states for onboarding, probe coverage, interface, evaluator, stale,
  and conflicting-evidence blockers;
- restart and idempotency support.

Acceptance:

- the current direct `idea -> automatic module task` path is impossible when
  the portrait is not ready;
- blocked work schedules observation tasks without allocating an intervention
  GPU lease;
- restart does not duplicate onboarding or probe work.

Tests:

- success, each blocker, resume, tamper, and no-GPU-on-not-ready integration
  tests.

Implementation evidence:

- portrait-enabled discoveries enter `pending_portrait`; only a
  `ready_for_gap_planning` receipt transitions to `pending_materialization`;
- every non-ready state produces one immutable observation work order and
  remains at `pending_observation` with no materialization or GPU claim;
- automatic module generation is rejected at config load when no portrait gate
  is bound;
- portrait files are hash-bound, schema-validated, and rechecked at execution;
- restart returns interrupted portrait work to `pending_portrait` without
  duplicating source or observation work;
- focused controller/module regressions pass, and
  `scripts/ci/check_control_plane.sh` passes with 488 tests and 5 declared
  environment-dependent skips.

### WP3: Adaptive observation and interface-extension loop

Status: complete; depends on WP2

Deliverables:

- `AdaptiveProbePlan` or equivalent bounded observation plan;
- `InterfaceExtensionSpec` with exact semantic hook, typed ports, authority,
  conformance tests, and negative tests;
- observation-module ABIs for at least one read-only adapter and one diagnostic
  probe extension;
- shadow-only handling for novel probes and metrics.

Acceptance:

- an uncovered portrait dimension selects an existing probe first;
- a missing hook creates an interface-extension work order instead of a proxy
  optimization method;
- generated observation code cannot change the active evaluator or protected
  metrics;
- architecture-bound and data-blocked states consume no intervention GPU.

Implementation evidence:

- `wmloop.control.adaptive_observation` converts readiness blockers into
  deterministic read-only, admitted-probe, shadow-probe, interface-extension,
  evaluator, data, or architecture tasks;
- the observation ABI registry includes an admitted read-only model-surface
  adapter, an admitted action-dose-response probe, and a generic shadow-only
  diagnostic-probe template;
- existing exact probe protocols are selected before the shadow manufacturer;
- generated interface-extension specs expose typed read-only observation
  surfaces and explicitly deny source, active-metric, evaluator, verdict, and
  promotion authority;
- the durable controller compiles `pending_observation` exactly once and
  transitions to probe execution, shadow admission, interface extension, or a
  durable non-executable blocker without materialization or intervention GPU;
- focused probe/controller regressions pass, and
  `scripts/ci/check_control_plane.sh` passes with 489 tests and 5 declared
  environment-dependent skips.

### WP4: Capability-oriented plugin registry and composer

Status: complete; depends on WP1

Deliverables:

- registry fields for typed ports, `provides`, `requires`, side effects,
  authority, admission suite, cost, portability, and license;
- migration of `history_selection_v1` without changing its scientific claim;
- dependency resolver and `ModuleCompositionReceipt`;
- compatibility, cycle, authority, and capability-closure checks.

Acceptance:

- existing admitted history-selection generation still passes unchanged
  evaluator bindings and fixed tests;
- compatible modules compose deterministically;
- missing ports, cycles, ABI drift, or authority escalation fail closed.

Implementation evidence:

- the production automatic-module registry uses the versioned v2 contract with
  stable module and ABI versions, content digests, typed ports, semantic
  capabilities, dependency constraints, authority, side effects, admission
  suite, cost, portability, license, and implementation metadata;
- the legacy v1 registry contract remains loadable by the existing automatic
  module compiler, while `history_selection_v1` is migrated to v2 without
  changing its symbol, repository destinations, fixed test template, frozen
  evaluator, parameter domain, import policy, or GPU-hour estimate;
- `wmloop.control.module_composition` selects the highest compatible semantic
  version deterministically, resolves the complete dependency closure, checks
  typed port contracts and external bindings, rejects cycles and unsatisfied
  capabilities, and enforces the caller's maximum L0-L2 authority;
- registry, ABI, and optional composition locks reject drift before execution,
  and the path-free `ModuleCompositionReceipt` records the exact closure,
  costs, authority, and side-effect boundary without importing generated code;
- focused automatic-generation and composition tests pass with 19 tests,
  related portrait, observation, controller, and materialization regressions
  pass with 49 tests, and `scripts/ci/check_control_plane.sh` passes with 508
  tests and 5 declared environment-dependent skips.

### WP5: Capability requirement graph and gap planner

Status: complete; depends on WP2 and WP4

Deliverables:

- Goal IR plus portrait compiler into a capability requirement DAG;
- exact gap classification and reuse-first resolution;
- portable-knowledge retrieval as ranking input only;
- a gap-plan receipt binding all inputs and decisions.

Acceptance:

- satisfied capabilities are reused;
- only missing leaf capabilities reach the manufacturer;
- similarity never substitutes for structural or behavioral compatibility;
- unsupported ideas produce the correct durable gap state.

Implementation evidence:

- versioned, path-free contracts now define Goal IR, the capability requirement
  DAG, and the capability gap plan receipt with content-derived identities and
  digests bound to the exact portrait, ABI registry, graph, and goal;
- `wmloop.control.capability_gap_planner` classifies requirements as satisfied,
  reusable, composable, manufacturable, interface-extension-required,
  data-blocked, or architecture-bound after exact model-family, capability,
  hook, interface, data, typed-port, ABI, and authority checks;
- admitted capabilities are reused first, compatible admitted modules compose
  deterministically, and only unresolved dependency leaves produce
  manufacturing requests; portable community knowledge contributes ranking
  priors but cannot override structural or behavioral incompatibility;
- the autonomous controller now advances a ready portrait through durable gap
  planning before either portfolio compilation, materialization, interface
  extension, data binding, or architecture termination, and gap planning has
  no GPU, evaluator, code-generation, or promotion authority;
- materialization requires the hash-bound gap receipt and requirement graph,
  revalidates their semantic binding and exact manufacturing closure, and
  rejects ordinary tampering as well as forged receipts with recomputed
  content hashes; interrupted `running_gap_planning` work resumes idempotently;
- focused planner, composer, and controller tests pass with 48 tests, related
  portrait, observation, generation, materialization, and controller
  regressions pass with 82 tests, and `scripts/ci/check_control_plane.sh`
  passes with 522 tests and 5 declared environment-dependent skips.

### WP6: Automatic experiment portfolio generation

Status: complete; depends on WP5

Deliverables:

- schema and compiler for evidence-bound experiment portfolios;
- diversity and information-gain ranking across competing mechanisms;
- automatic baseline, negative-control, ablation, and replication allocation;
- content-derived candidate identities with no human-style version naming.

Acceptance:

- the planner produces discriminating experiments rather than cosmetic
  parameter variants;
- every portfolio entry states its hypothesis, falsification, cost,
  dependencies, protected metrics, evaluator, and artifact policy;
- malformed LLM output cannot influence admission.

Implementation evidence:

- versioned, path-free contracts now define the only accepted LLM-shaped
  hypothesis batch and the compiled `ExperimentPortfolio`; the draft contract
  excludes evaluator, metric, path, GPU, promotion, and version-selection
  authority;
- `wmloop.control.experiment_portfolio` validates content-derived mechanism
  contracts, exact goal/portrait/requirement-graph/gap-plan bindings, and
  required module capabilities before ranking any candidate;
- candidates receive semantic digest identities rather than LLM-selected
  version names, and the deterministic greedy ranking combines information
  gain, uncertainty, explicit competition coverage, portrait-change diversity,
  and bounded cost while rejecting cosmetic mechanism variants;
- every selected mechanism automatically receives one shared frozen baseline,
  a no-op negative control, all declared discriminating ablations, and
  deterministic replications; every entry binds its hypothesis,
  falsification, dependencies, cost, protected metrics, portrait evaluator,
  held-out protocol, unresolved risks, archive policy, and cleanup policy;
- the autonomous controller now executes `pending_portfolio` as a durable,
  restart-safe stage before module manufacture or resource admission; budget
  blocks, malformed batches, receipt drift, and portfolio tampering start no
  materialization or GPU work;
- focused composer, gap-planner, portfolio, and controller tests pass with 59
  tests, related proposal, IR, scheduling, portrait, observation, generation,
  and materialization regressions pass with 132 tests, and
  `scripts/ci/check_control_plane.sh` passes with 533 tests and 5 declared
  environment-dependent skips.

### WP7: Module manufacturer integration

Status: complete; depends on WP3, WP4, WP5, and WP6

Deliverables:

- reuse/compose/manufacture/interface-extension decision flow;
- generated module work orders bound to portrait, gap plan, and portfolio;
- isolated implementation, fixed checks, admission receipt, and cleanup;
- explicit observation versus intervention manufacturing modes.

Acceptance:

- generated code cannot choose paths, evaluators, splits, metrics, budgets, or
  promotion rules;
- an admitted module has content-addressed identity and full dependency
  closure;
- unsupported work never silently falls back to a nearby method.

Implementation evidence:

- a versioned, path-free `ModuleManufacturingWorkOrder` binds every
  intervention request to the exact Goal IR, Model Portrait, capability
  requirement graph, gap plan, experiment portfolio, selected entries,
  target ABI registry digest, transitive upstream dependency closure, frozen
  evaluator, protected metrics, and archive and cleanup policies;
- observation manufacturing uses a separate shadow-only work-order mode that
  can request only the declared adaptive probe task and observation ABI, and
  has no intervention, evaluator, metric, verdict, GPU, or promotion
  authority;
- unbound ABIs, unsupported observation tasks, ambiguous autonomous
  intervention requests, and attempts to route observation work through the
  intervention generator fail closed with explicit interface-extension or
  validation errors;
- automatic generation exposes only the ordered work order's exact ABI to the
  LLM; its input lock, candidate identity, admission receipt, materialization
  plan, CAS archive, and terminal receipt all carry the work-order identity,
  digest, mode, request, dependencies, evaluator, metrics, and portfolio
  bindings;
- the autonomous controller now manufactures intervention modules only from
  admitted portfolio work orders and records shadow manufacturing work orders
  for planned observations, while legacy direct APIs remain backward
  compatible and terminal materialization resume remains idempotent;
- focused manufacturer and integration tests pass with 60 tests, related
  WP3-WP7 regressions pass with 101 tests, and
  `scripts/ci/check_control_plane.sh` passes with 541 tests and 5 declared
  environment-dependent skips.

### WP8: Portfolio-aware eight-GPU scheduling

Status: complete; depends on WP6 and WP7

Deliverables:

- versioned resource-portfolio receipt;
- current two-GPU DROID conversion reservation and up-to-six-GPU experiment
  allocation outside the generic Kernel;
- screen-to-confirm reallocation policy based on evidence and scale plans;
- durable restart, physical-GPU, activity, artifact, and cleanup receipts.

Acceptance:

- independent hypotheses can run concurrently without oversubscribing a
  physical GPU;
- distributed training is selected only with an admitted scale rationale;
- controller restart creates no duplicate trial or conversion work;
- failed jobs release resources and retain terminal evidence.

Implementation evidence:

- a versioned `ResourcePortfolioReceipt` binds the exact experiment portfolio,
  selected candidate and trial, config digest, experiment and DROID scale-plan
  digests, full baseline/control/ablation/replication queue, archive policy,
  cleanup policy, and restricted GPU-only authority;
- campaign policy remains outside the generic `GpuLeaseManager`: the current
  Ctrl-World v1 resource compiler requires the complete physical inventory
  `0-7`, admits independent experiment work only on `0-5`, and reserves `6-7`
  exclusively for resumable DROID conversion; even a recomputed receipt cannot
  reassign the reserved role;
- the durable controller now advances materialized work through
  `pending_resource_admission` before screen and accepted screen evidence
  through `pending_resource_reallocation` before confirm, while draining
  pending screens before confirmation work and recovering claimed resource
  stages without duplicate work;
- confirmation receipts bind the exact accepted screen evidence; distributed
  confirmation additionally requires a ready confirm-stage training-scale
  plan with an exact world-size match and an explicit information-gain value
  greater than independent-screen opportunity cost; the current one-GPU
  executor rejects admitted multi-GPU work until a distributed executor is
  bound;
- each launched stage hash-checks its resource receipt before lease
  acquisition, uses only admitted experiment GPUs, and records physical GPU
  identity, observed activity, exit status, artifacts, release state, and
  cleanup policy; campaign failures still write terminal execution and release
  receipts, and release failures are recorded rather than reported as success;
- focused resource, deployment, and controller tests pass with 46 tests,
  related campaign, DROID, scale, portfolio, and manufacturing regressions pass
  with 78 tests, and `scripts/ci/check_control_plane.sh` passes with 552 tests
  and 5 declared environment-dependent skips.

### WP9: Portrait-aware portable knowledge projection

Status: complete; depends on WP1 and may be developed before WP8

Deliverables:

- portable Model Portrait nodes and transition edges;
- portable module, protocol, transform, evidence, applicability, negative,
  revocation, and supersession projections;
- one-way local-to-community staging and quality audit;
- exact matching and bounded similarity queries over semantic fields.

Acceptance:

- a graph rebuilt on another checkout is byte-stable;
- no local path, machine identity, credential, private artifact, or unlicensed
  content enters semantic nodes;
- model A evidence is exposed only as a prior for model B;
- every promoted relation has frozen evidence and CAS references.

Implementation evidence:

- the existing `portable_knowledge_graph` remains the single community
  projection framework and now accepts Model Capability IR, Model Portrait,
  append-only portrait transitions, path-free module-composition receipts,
  Evidence IR, probe/metric/evaluator protocol contracts, semantic data
  transformation contracts, and revocation/deprecation/supersession records;
- projection rejects unsupported or path-bound documents, deterministic node
  or edge conflicts, non-redistributable content without a declared license,
  and community-promoted relations that are not both frozen and bound to an
  exact `cas://sha256/` reference; large artifacts and local execution records
  remain outside the semantic graph;
- exact filters and bounded Jaccard-style similarity operate only on an
  explicit semantic-field allowlist; a result from another portrait is marked
  `prior_only`, so similarity cannot create target verdict authority;
- portrait and reusable module-composition records are staged one way from the
  durable controller into content-digest-named semantic files; retries and
  restart replay deduplicate identical documents, and graph rebuilds never
  inspect arbitrary source trees or import local receipt locations;
- graph, manifest, and versioned quality-audit outputs are byte-identical
  across output roots and input ordering; the audit binds the exact input
  document set and graph digest and rejects path leakage or graph/document
  divergence;
- focused portable-knowledge and autonomous-controller tests pass with 55
  tests, related portrait, IR, composition, planning, manufacturing, resource,
  and public-release regressions pass with 129 tests, and
  `scripts/ci/check_control_plane.sh` passes with 563 tests and 5 declared
  environment-dependent skips.

### WP10: Closed-loop replanning and quality audits

Status: complete; depends on WP2 through WP9

Deliverables:

- portrait updates after admitted interventions;
- residual, counterexample, uncertainty, and information-gain based next-task
  ranking;
- duplicate-candidate, stale-portrait, protocol-drift, cleanup, authority, and
  non-portability audits;
- explicit stop conditions for success, exhausted budget, insufficient
  evidence, architecture mismatch, and unresolved policy gaps.

Acceptance:

- the system can resume after restart and choose the next observation or
  intervention without human-designed method versions;
- all terminal work is archived;
- only frozen, path-free, evidence-bound records affect reusable community
  knowledge;
- the loop stops rather than manufacturing meaningless work when expected
  information gain is below policy threshold.

Implementation evidence:

- the durable controller routes verified terminal work and non-executable
  architecture, data, evaluator, interface, resource, and budget blockers
  through one `pending_replan` stage; retries remain within that stage and
  terminate as `replan_failure` after the declared operational retry budget;
- next-task decisions validate the exact quality-audit identity, rank only
  currently available observation/intervention tasks, derive bounded residual,
  counterexample, uncertainty, and information-gain signals from frozen
  transfer evidence, and emit schema-bound stop reasons for every declared
  policy boundary;
- admitted interventions create deterministic child portraits, invalidate old
  fingerprints and operational observations, bind verified transitions to the
  frozen evaluator and verdict, and use a candidate-bound durable marker so a
  restart cannot derive the same transition twice; the mutable local active
  pointer never replaces the append-only portrait and transition history;
- terminal inputs and work receipts are copied into a local SHA-256 CAS with
  append-only per-work archive generations; identical retries reuse the same
  receipt, changed later rounds create a new receipt, imported terminal
  evidence is queued once, and byte conflicts fail closed;
- quality audits cover duplicate candidates, stale portraits, protocol drift,
  cleanup and missing archives, unauthorized stage authority, and
  non-portable knowledge; only validated portrait/transition documents are
  staged into the path-free community knowledge input;
- 29 focused closed-loop tests pass, 92 related portrait, verifier, portable
  knowledge, and controller regressions pass, and
  `scripts/ci/check_control_plane.sh` passes with 595 tests and 5 declared
  environment-dependent skips.

## Existing Foundation To Reuse

The following are implemented foundations, not reasons to skip the pending
work packages:

- Model Capability IR, onboarding, conformance, and local execution bindings;
- probe and fingerprint implementations plus portable fingerprint summaries;
- MechanismContract, MethodEmbodiment, transfer-boundary, and portable graph
  projection;
- durable Ctrl-World controller, physical GPU leases, restart recovery, and
  screen/confirm/frozen-verifier stages;
- schema-bound provider-neutral LLM tasks;
- automatic generated-module admission through `history_selection_v1`;
- content-addressed portable evidence and path rejection;
- metric-adequacy reports and policy-bounded constitutional proposals.

The current control-plane path now exercises portrait-first planning,
automatic capability-gap decomposition, deterministic module composition,
evidence-bound experiment portfolios, bounded module manufacturing, resource
admission, frozen verification, append-only portrait transitions, and
path-free community projection in focused and controller integration tests.
Those tests establish the control-plane contract; they do not establish any
particular model-quality improvement or cross-model transfer claim.

## Current Verification Snapshot

The latest repository audit on 2026-08-18 recorded:

- `bash scripts/ci/check_control_plane.sh`: `595 passed, 5 skipped`;
- focused WP1-WP10 regression set: `153 passed`;
- `git diff --check`: passed;
- wheel build and public-release validation: passed.

The five skips are declared environment-dependent checks: missing `torch`, an
unmaterialized live Cosmos3 CPU smoke, and a missing ffprobe/video fixture.
They do not represent failed control-plane assertions. Live model, GPU, and
community-scale operation remain empirical validation work, and every model
improvement or transfer statement still requires its own frozen verifier.

## Execution Discipline

For every future implementation turn in this program:

1. Read this document and identify the first uncompleted dependency relevant
   to the request.
2. Inspect the current worktree and preserve unrelated user changes and active
   runtime sessions.
3. Implement the smallest complete work-package slice; do not create a second
   orchestration framework.
4. Add focused tests proportional to the authority and blast radius.
5. Run focused tests, `bash scripts/ci/check_control_plane.sh`, and
   `git diff --check` before marking a package complete.
6. Update the work-package status and record any design amendment with its
   evidence.
7. Report scientific claims only from frozen-verifier evidence. Report control
   plane completion separately from model-quality results.

Do not skip a dependency because a later feature is easier to demonstrate. Do
not mark a work package complete from schemas or import success alone. Do not
rewrite historical evidence or convert a screen result into a reusable claim.

## Definition Of Program Completion

This program is complete when a new user can provide a model, data regime,
goal, and budget, and the system can autonomously:

1. onboard and portray the model;
2. identify and fill observation gaps;
3. compile a capability requirement graph;
4. retrieve reusable community priors without trusting them as target proof;
5. compose or manufacture only missing bounded modules;
6. construct and schedule a discriminating experiment portfolio;
7. settle results through frozen verification;
8. update append-only portraits and path-free community knowledge; and
9. select the next task or stop for an evidence-backed reason.

Completion of this program is a control-plane claim. Each model-improvement or
cross-model-transfer claim still requires its own frozen evidence.
