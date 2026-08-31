# Mechanism relations

VerdiWM now stores pairwise mechanism knowledge as a first-class portable
artifact: `verdiwm-mechanism-relation`. This is separate from an individual
`EffectRecord`, because a relationship needs its own counterfactuals and
evidence.

## What is recorded

Each relation identifies a source and target mechanism, the composition
operator (`parallel`, `sequential`, `gated`, or `conditional`), applicability
conditions, anti-conditions, effect uncertainty, required ablations, evidence
references, and a verification state. Supported relation types are
`positive_synergy`, `antagonism`, `redundancy`, `conditional_compatibility`,
`sequential_dependency`, and `substitution`.

The interaction contrast is deterministic:

```text
combined - source - target + baseline
```

`classify_interaction()` reports `positive_synergy`, `antagonism`, or
`redundancy`; results whose absolute contrast is no larger than uncertainty
are returned as `abstained`.

## Admission boundary

Relations in `candidate` or `screened` state are ranking knowledge only. A
`confirmed` or `rejected` relation must have all validity gates passing and
must name its required ablations. The evidence references must be portable
(`cas://`, `urn:`, or `sha256:`); local paths are rejected. This means a
composition receipt alone cannot be promoted to a synergy claim.

Use `propose_mechanism_relation()` to turn a four-cell comparison into a
non-authoritative candidate; it never bypasses evaluator admission. Use
`EffectMemory.add_relation()` and `query_relations()` for local storage and
lookup. `write_relation_jsonl()` persists a deterministic derived view. Passing
the same relation artifact to `build_portable_knowledge_graph()` adds typed
source/target, condition, ablation, and evidence edges to the portable graph.

For automatic experiment planning, call
`wmloop.control.experiment_portfolio.build_relation_hypothesis_batch()` (or
`compile_relation_experiment_portfolio()`). The adapter creates one A+B
composition candidate. The normal portfolio then schedules the shared
baseline, composition test, no-op control, and component-removal ablations
(A-only and B-only), preserving the existing budget and frozen-evaluator
admission rules.
