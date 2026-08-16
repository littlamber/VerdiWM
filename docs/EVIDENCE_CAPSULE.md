# Evidence Capsule

VerdiWM keeps the Archive, CAS, frozen evaluators, and full Evidence Graph as
authoritative or audit surfaces. The normal diagnose-to-compile loop does not
need to carry every graph node or every retrieval row, however. The
`verdiwm-evidence-capsule` projection is the small runtime surface for that
decision.

## Contract

`wmloop.retrieve.evidence_capsule.build_evidence_capsule` emits one immutable
JSON projection with:

- a route: `no_diagnostic`, `cold_start`, or `reuse_settled`;
- the diagnostic query fields and normalized failure signatures;
- at most three selected settled matches by default;
- only scalar ranking fields and receipt/CAS/Archive references;
- an explicit claim boundary saying that the capsule is not evidence authority.

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
