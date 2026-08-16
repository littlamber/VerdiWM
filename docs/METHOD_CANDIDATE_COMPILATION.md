# Method Candidate Compilation

This document defines the persisted contract between diagnosis, retrieval,
adapter capability declaration, historical evidence, and the experiment
scheduler. Compilation is an admission step. It does not establish scientific
quality or grant promotion authority.

## Inputs

The compiler joins four typed inputs:

| Input | Role | Authority |
| --- | --- | --- |
| Diagnostic probe | Declares observed failure signatures | May route a matching recipe; cannot create one |
| Literature method staging | Adds typed method and primitive matches | Ranking only, or a materialization gap |
| Imported settlement records | Retains terminal verdicts and claim boundaries | Negative constraint unless promotion was explicitly authorized |
| Adapter method catalog | Binds a method to hooks, files, hashes, budget, and an executable candidate template | Scheduling authority after all checks pass |

The adapter profile resolves `candidate_catalog` and the first valid
`settlement_manifest_candidates` entry. Onboarding then materializes catalog
placeholders from admitted repository, runtime, asset, and state-root bindings.
Unbound placeholders fail closed.

## Compilation States

A catalog recipe becomes `compiled` only when its diagnostic route matches, its
candidate ID is new, no declared historical candidate was rejected without
promotion, and every required regular file exists with the declared SHA-256.
The materialized candidate is appended to the scheduler batch and retains its
source, mechanism hypothesis, required hooks, budget, applicability conditions,
failure boundaries, and matched literature IDs.

Anything that cannot enter the queue remains explicit in `capability_gaps`.
Stable blocker codes distinguish duplicate IDs, diagnostic mismatch,
historically rejected candidates, missing or drifted adapter files, methods
without a materialized primitive, and known primitives without an adapter
recipe. A catalog can therefore be `ready` while also reporting gaps; it is
`blocked` when it compiles no candidate.

## Evidence And Authority Boundaries

An imported `NOT_PROMOTED` settlement is never a positive prior. Repeating an
exact candidate named by `historical_candidate_ids` is blocked unless the
record carries explicit promotion authority. Other settlement records remain
visible as constraints and provenance, rather than being discarded.

Literature staging cannot supply commands, files, hooks, or GPU authority. A
known primitive can improve the bounded retrieval prior only after the adapter
catalog supplies a complete recipe. An unknown method is recorded as
`PRIMITIVE_MATERIALIZATION_REQUIRED`; a known but unsupported primitive is
recorded as `ADAPTER_RECIPE_MISSING`.

Compilation authorizes only the declared scheduler stage under its existing
GPU budget and metric gates. It does not authorize training, broaden an
exact-context result, convert exploratory evidence into a transfer claim, or
override the evaluator-specific promotion process.

## Durable Artifacts

For an autonomous run, the compiler writes:

```text
compiled/
  candidate-batch.json
  manifest.json
  method-candidates/manifest.json
  queue/queue.json
```

`method-candidates/manifest.json` is the durable compilation report. Its
canonical SHA-256 is stored both in the candidate batch's
`method_candidate_compilation` binding and in `compiled/manifest.json`. The
top-level `pipeline-manifest.json` projects the report for inspection, while
the compiled batch and queue remain the scheduler's authoritative inputs.

The onboarding compilation input hash covers the admitted report and receipt,
scheduler template, diagnostic probe, literature artifacts, candidate catalog,
the settlement manifest plus every imported record, and retrieval context. This
binds candidate selection to the complete evidence set, not only to a path.

## Resume Rules

Rerunning an identical command may resume only when the compilation input hash
matches. Resume rechecks the candidate batch SHA-256 and the persisted method
candidate report SHA-256 before rebuilding or reading the queue. Any changed
catalog, settlement record, diagnostic result, literature record, required-file
hash, or materialization input requires a new compiled output root; mutation of
an existing durable artifact is rejected.

Required adapter files are checked again during fresh compilation. A missing
file or SHA-256 mismatch becomes a capability gap and prevents that recipe
from entering the scheduler. This makes source drift visible before GPU work is
admitted.
