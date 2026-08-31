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

After those four cells settle, call
`wmloop.geometry.mechanism_relations.settle_mechanism_relation()` or
`EffectMemory.settle_relation()`. The adapter checks that all records share a
comparable context, computes the interaction contrast, aggregates uncertainty
and gates, and emits a `confirmed`, `rejected`, `candidate`, or `abstained`
relation artifact. That artifact can then be passed to
`build_portable_knowledge_graph()` for durable relationship edges.

For actual execution, `wmloop.control.mechanism_composition` provides a
registry-bound compiler. `bind_executable_mechanism()` binds a semantic method
to one reviewed primitive and its validated parameters;
`compile_mechanism_composition()` derives four executable cells, including an
empty baseline and automatic A-only/B-only ablations. A single generic
executor can consume those cells and return `EffectRecord` objects via
`execute_mechanism_composition()`. This keeps model-specific code behind one
backend boundary instead of requiring a separate runner for every method pair.

`discover_mechanism_compositions()` closes the candidate-selection step. It
scans confirmed effects and executable bindings, requires comparable goal and
outcome contexts, rejects registry conflicts, excludes pairs with an existing
relation, and ranks compatible pairs using settled lower bounds, uncertainty,
mechanism-family diversity, and hook diversity. Its output already contains
the four-cell execution plan.

To make a deposited method reusable by discovery, include an optional
`executable_binding` in its `verdiwm-method-embodiment` record. It names the
registered `primitive`, validated `params`, and implementation revision. The
binding is semantic metadata only; runtime source paths remain excluded.

## First-contact model bootstrap

The same adapter boundary is available from `CampaignStore.create()`. A caller
that may encounter a new model family can include an `executor_bootstrap`
object with a trusted `base_profile_path`, a configured bounded `llm_adapter`,
an optional external `repair_output_root`, and `max_attempts` (1-5). The API
first resolves the normal profile; only a repairable interface error can enter
the bounded repair loop. A repaired profile is conformance-compiled before the
campaign is recorded, and the resulting bootstrap manifest is included in the
campaign revision for audit and replay.

Bootstrap never edits the model or evaluator, schedules GPUs, or grants
promotion authority. Missing trust inputs, unsupported model capabilities, or
failed conformance checks produce an explicit `EXECUTOR_BOOTSTRAP_BLOCKED`
error. This is intentional: a first-contact model can become executable by
learning an adapter, but the system must not claim readiness when the missing
piece is scientific rather than an interface contract.
