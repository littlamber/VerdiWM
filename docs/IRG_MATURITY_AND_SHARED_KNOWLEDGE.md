# IRG maturity roadmap and shared knowledge architecture

This document records the intended maturation path for the first three VerdiWM
contributions and clarifies how they relate to the evidence chain and knowledge
graph. It is a repository-level memory of the design position; it does not
claim that the empirical milestones below have already been achieved.

## Current position

The first three contributions are implemented as bounded mechanisms:

1. **IRG + Model Portrait** measures intervention responses and binds them to a
   model-conditioned diagnostic artifact.
2. **Counterexample-driven probe evolution** detects nearby IRGs with opposing
   method effects and emits a proposal-only work order for a discriminating
   measurement.
3. **IRG-guided cross-domain discovery** turns supported IRG hotspots into
   policy-versioned research queries and stages external methods as untrusted,
   typed candidates.

Their current limitation is empirical, not primarily structural. We still need
frozen comparisons showing that IRG predicts method effects, evolved probes
reduce prediction regret or cost, and IRG-guided retrieval finds better
mechanisms than ordinary retrieval baselines.

## Maturity gates

### IRG predictive validity

Compare static metrics, generic embeddings, a fixed probe bank, and IRG across
held-out models, methods, and episodes. Report sign accuracy, effect-size error,
ranking regret, calibration, and cross-model LOBO transfer. A bottleneck must
remain a local failure signature unless dose and horizon sweeps support a
plateau or saturation claim.

### Probe-evolution utility

Compare fixed probes, random additions, LLM-only proposals, and collision-guided
CPBE under the same budget. Every successor probe records its parent collision,
predicted information gain, realized gain, regret reduction, locality, and
redundancy. The process stops when additional probes do not improve held-out
discrimination. Probe design and final evaluation use separate splits.

### Retrieval utility

Compare LLM-only, keyword, ordinary RAG, failure-signature RAG, and IRG-guided
retrieval. Evaluate compile rate, verified success rate, GPU cost per useful
candidate, mechanism coverage, negative-result quality, and citation
provenance. A retrieved paper is not a mechanism until it has a target failure,
hook, bounded intervention, prediction, invariant, and anti-condition.

## Evidence chain and knowledge graph

Yes: the evidence chain and knowledge graph are a central part of this design,
not an unrelated reporting feature. The relationship is:

```text
probe measurement
  -> IRG / Model Portrait
  -> failure signature and bottleneck hypothesis
  -> literature/mechanism candidate
  -> compiled intervention
  -> paired trial and frozen verdict
  -> Archive + CAS + Evidence Graph
  -> effect memory / transferable experience
  -> future IRG routing and probe acquisition
```

The **Archive** and **CAS** are authoritative for immutable receipts and bytes.
The **Evidence Graph** is a deterministic, rebuildable provenance projection.
**EffectMemory** and **transferable experience** are derived routing views that
retain positive, null, harmful, and abstained outcomes. A graph edge does not
become verified merely because a process completed; settlement and verifier
state are required.

The current implementation therefore already supports the first form of the
shared knowledge idea:

- local experiments deposit content-addressed evidence;
- settled effects can become context-local or licensed portable experience;
- compatible users or model families can retrieve those derived records;
- negative and failed evidence remains available for routing and collision
  analysis;
- runtime paths, private checkpoints, and machine-specific bindings are
  excluded from portable knowledge.

## What “shared” means today versus the target

Today, “shared” means that multiple runs can point at the same Archive/CAS and
retrieval index, or exchange path-free portable experience records. The graph is
still a local projection and the repository does not yet provide a hosted,
multi-tenant federation service.

The target multi-user system should have three layers:

1. **Private evidence space**: raw videos, checkpoints, datasets, logs, and
   sensitive receipts remain under the user's control.
2. **Publishable evidence capsule**: only content-addressed, schema-validated,
   license-compatible, path-free records are exported. The capsule contains
   provenance, protocol hashes, model family/capability class, IRG coordinates,
   intervention semantics, effect estimates, uncertainty, and negative
   boundaries.
3. **Shared knowledge graph**: a server or replicated store indexes capsules by
   semantic identities and evidence references. It serves priors and candidate
   routing, never direct scientific verdicts.

This gives the intended community loop:

```text
user experiment
  -> private raw evidence
  -> local verification
  -> publishable capsule
  -> shared graph / effect memory
  -> another user's IRG-compatible retrieval
  -> target-side revalidation
```

Shared records must remain priors. A `licensed_prior` can schedule a bounded
target-side reuse experiment, but cannot replace target validation. This is the
same reason a similar IRG does not automatically imply a transferable effect.

## Required federation safeguards

Before enabling public deposition, the service needs:

- signed submitter and license metadata;
- content-addressed artifact and protocol hashes;
- schema/version negotiation;
- privacy and redaction checks for paths, identifiers, and raw media;
- duplicate and replay detection;
- source/reproduction confidence and conflict tracking;
- quarantine for unverified or disputed capsules;
- append-only revisions rather than destructive edits;
- explicit separation of local audit graph, portable knowledge graph, and
  execution ledger;
- target-side confirmation before any shared prior affects promotion.

Conflicting reports should be represented as competing evidence, not averaged
away. This is especially important for IRG collisions: a collision is useful
shared knowledge because it tells future users where the current diagnostic basis
is insufficient.

## Recommended implementation sequence

1. Finish the three predictive-utility benchmark suites above.
2. Add a canonical `EvidenceCapsule` schema with redaction and license checks.
3. Add signed export/import and a conflict-aware shared graph projection.
4. Add federated retrieval that returns priors plus their evidence boundaries.
5. Measure whether community deposition improves candidate hit rate and lowers
   GPU cost without increasing false transfer.

Until step 5 is measured, the correct claim is **shared-evidence-ready research
platform**, not a proven community knowledge network.
