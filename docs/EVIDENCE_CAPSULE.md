# Evidence Capsule

VerdiWM keeps the Archive, CAS, frozen evaluators, and full Evidence Graph as
authoritative or audit surfaces. The normal diagnose-to-compile loop does not
need to carry every graph node or every retrieval row, however. The
`verdiwm-evidence-capsule` projection is the small runtime surface for that
decision.

## Contract

There are two deliberately separate capsule surfaces. The existing
`wmloop.retrieve.evidence_capsule.build_evidence_capsule` is an internal,
bounded runtime routing projection. The exchange CLI in
`wmloop.evidence_capsule` emits the standardized v1 portable capsule used for
sharing between machines or installations.

The exchange projection contains:

- model/capability and IRG/probe/failure semantics when present;
- intervention semantics, evaluator/protocol hashes, effects, uncertainty,
  protected metrics, anti-conditions, and provenance references;
- one of `verified`, `exploratory`, `null`, `harmful`, `abstained`, or
  `disputed`;
- an explicit claim boundary saying that the capsule is a prior, not evidence
  authority.

Export removes absolute paths, checkpoint/dataset/runtime bindings, command
and environment data, and raw media. `validate` re-checks those constraints,
the v1 schema, and 64-character hashes. `import` stores a content-addressed
local index and writes a `verdiwm-evidence-capsule-import-receipt` whose
`routing_authority` is always `prior_only`. Import never creates a trial or a
target-side verdict.

The command-line surface is:

```text
verdiwm-evidence-capsule export --source receipt.json --output capsule.json
verdiwm-evidence-capsule validate --capsule capsule.json
verdiwm-evidence-capsule import --capsule capsule.json --destination-root index/
```

The selected rows are copied only after `retrieve_probe_experiences` has
revalidated settlement and CAS hashes. Invalid or unbound rows fail closed.
The full records remain available in the retrieval index and the Evidence Graph
can still be rebuilt on demand.

## What is borrowed from DeepSeek Harness

DeepSeek Harness uses plugin seams, an append-only session log, and derived
projections. VerdiWM applies the same separation at a narrower boundary:

```text
Archive/CAS receipts -> retrieval index -> evidence capsule -> compiler route
                                      \-> Evidence Graph (audit projection)
```

This is not a rewrite into a general plugin framework. VerdiWM's scientific
authority remains receipt-first and fail-closed; a capsule cannot promote a
candidate, change a verifier, or substitute for IRG locality or held-out
confirmation.
