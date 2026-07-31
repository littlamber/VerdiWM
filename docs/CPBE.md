# Counterexample-Guided Probe Basis Expansion

CPBE turns a repair collision into a bounded diagnostic search problem. It does
not allow a language model to write an arbitrary probe and route its output
directly into a verdict. Every candidate is first represented in a typed Probe
DSL, ranked by an evidence-conditioned acquisition function, and settled under
the same frozen stage gates.

## Why basis expansion

A repair collision occurs when two contexts are close under the current IRG
coordinates but a common primitive has confidence-separated opposing effects.
The current diagnostic basis aliases a mechanism that matters for repair.
CPBE searches for a new local response direction that separates that
counterexample while remaining measurable, nonredundant, and affordable.

## Probe DSL

A `ProbeProgram` binds:

```text
signal source
x model hook
x spatial mask
x temporal basis
x contrast operator
x dose schedule
x aggregation
```

The descriptor also binds required capabilities, invariants, reversibility,
diagnostic-only scope, lineage, and estimated canary cost. A retrieval result
or LLM proposal outside the frozen grammar fails before code generation.

## Candidate sources

CPBE merges four candidate sources:

1. **Residual expansion** mutates the highest-weight unexplained DSL axes.
2. **Structured mutation** enumerates one-component changes from the current basis.
3. **Atlas retrieval** imports compatible programs from settled effect memory.
4. **LLM hypothesis generation** proposes mechanism-grounded DSL programs.

Semantic duplicates are removed before scoring. The source affects lineage,
not admission authority: all candidates use the same capability and evidence
gates.

## Evidence-conditioned acquisition

Historical trials are weighted by backbone family, capability class, failure
signature, primitive, and Probe DSL similarity. A Beta posterior estimates
locality, nonredundancy, and collision-resolution probabilities. Weighted
effect observations estimate regret and accepted-coverage gains.

The v1 acquisition function combines:

```text
LCB(expected regret reduction + coverage gain + collision resolution)
+ residual alignment
+ uncertainty bonus
- nonlocality risk
- structural redundancy
```

and divides positive utility terms by estimated GPU cost. The surrogate is
deliberately small-data and inspectable. A community registry can later replace
the weighted estimator with a contextual bandit without changing the Probe DSL
or settlement contract.

## Successive halving

Selected probes advance through:

```text
static -> offline -> canary -> expanded
```

- `static`: descriptor, hook, capability, invariant, and no-verdict checks.
- `offline`: fixture execution and schema-valid diagnostic output.
- `canary`: measured locality, empirical redundancy, and collision separation.
- `expanded`: selector regret reduction or accepted-coverage gain.

Missing or out-of-order receipts fail closed. Probe admission only expands the
diagnostic basis. A separate frozen selector and primitive confirmation are
still required for model-quality or transfer claims.

## CLI

Create a plan:

```bash
verdiwm-cpbe plan \
  --request cpbe-request.json \
  --history probe-trials.jsonl \
  --output-root results/cpbe-plan
```

Settle stage receipts:

```bash
verdiwm-cpbe settle \
  --plan results/cpbe-plan/cpbe-plan.json \
  --receipts cpbe-stage-receipts.jsonl \
  --output-root results/cpbe-settlement
```

The request and stage-receipt interfaces are defined by
`configs/schemas/cpbe_request.schema.json` and
`configs/schemas/cpbe_stage_receipt.schema.json`. Historical surrogate inputs
use `configs/schemas/cpbe_history_trial.schema.json`; synthetic fixture records
are excluded automatically from live plans.

## Claim boundary

CPBE v1 implements deterministic candidate synthesis, evidence-conditioned
ranking, work-order generation, and fail-closed settlement. It has unit-level
algorithm validation. It has not yet established nonzero accepted evolved
coverage, reduced held-out selector regret, or cross-backbone repair benefit.
