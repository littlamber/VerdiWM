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
- multi-axis intervention complexity
```

and divides positive utility terms by estimated GPU cost. The surrogate is
deliberately small-data and inspectable. A community registry can later replace
the weighted estimator with a contextual bandit without changing the Probe DSL
or settlement contract.
The complexity term prevents a broad LLM hypothesis from winning merely by
changing many high-residual axes at once. Single-axis counterfactuals remain
preferred because their canary outcome is easier to attribute and reuse.

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
For non-synthetic plans, CLI settlement also requires each stage receipt to
bind local evidence artifacts by relative path, byte size, and SHA-256; a bare
`passed=true` receipt is rejected.

## ACWM-Phys replay status

The checked-in r29/r30 evidence is a negative, reproducible self-evolution
trace. The r29 live settlement tested three terminal candidates and admitted
zero. Its expanded counterexample was retained in r30: locality failed for
`push_rope`, the probe was nonredundant, collision resolution was false,
regret reduction was `-0.007786748614631023`, coverage gain was `0`, and the
measured cost was `0.1386963263888889` GPU-hours.

The counterexample learner converts that result into measured search history.
It discounts the edited temporal axis, reallocates credit toward aggregation,
signal source, and spatial locality, rejects unsupported horizon-weighted
aggregation when horizon-indexed outcomes are absent, removes multi-axis LLM
hypotheses, and restricts the next round to the failed candidate's direct
parent plus one canary. The axis-credit update is a heuristic proposal prior;
it is explicitly not an empirical causal attribution.

The r30 materializer generated `cpbe_residual_33b1d8a8f5` from the
`action_temporal_alignment_phase` parent with a single aggregation edit to
`source_sign_margin`. Its source-sign projection is fit on source
environments only and records `target_label_used_for_fit=false`. The valid
canary passed locality and nonredundancy, but did not separate the target
collision (`collision_separation=-0.006469471739052768`), so it was eliminated
at canary and the formal r30 settlement again admitted zero candidates.

These runs establish that the implementation can learn from a measured
counterexample, materialize a typed candidate, execute it, and abstain. They
do not establish a positive evolved probe, model-quality improvement, or
cross-backbone transfer. The candidate with a mismatched parent/reference
lineage is retained only in raw campaign evidence and is excluded from all
claims and the public source tree.

## CLI

Create a plan:

```bash
verdiwm-cpbe plan \
  --request cpbe-request.json \
  --history probe-trials.jsonl \
  --output-root results/cpbe-plan
```

For ACWM replay evidence, first convert a still-open frozen work order and its
measured locality/redundancy receipt into CPBE inputs:

```bash
verdiwm-acwm-cpbe-bootstrap \
  --template cpbe-live-template.json \
  --selector-replay selector-replay.json \
  --redundancy-report probe-smoke-redundancy.json \
  --environment-manifest environment-manifest.json \
  --output-root results/acwm-cpbe-bootstrap
```

The adapter marks the source probe as an unresolved historical trial. It uses
an explicit counterexample-axis heuristic only to initialize the residual
prior; it never fabricates collision resolution, regret reduction, or accepted
coverage.

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
are excluded automatically from live plans. Outcomes that were not measured
must be `null`; the surrogate excludes them from the corresponding posterior
instead of learning that missing evidence is a failure or a zero gain.

## Claim boundary

CPBE v1 implements deterministic candidate synthesis, evidence-conditioned
ranking, work-order generation, and fail-closed settlement. It has unit-level
algorithm validation. It has not yet established nonzero accepted evolved
coverage, reduced held-out selector regret, or cross-backbone repair benefit.
