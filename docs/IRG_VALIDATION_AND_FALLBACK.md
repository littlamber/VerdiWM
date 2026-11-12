# IRG Validation and Fallback

IRG is a diagnostic response representation. It is useful only when it
improves the decision of which repair to try under a fixed budget. A large
response is not, by itself, a defect, a cause, or evidence that a repair will
work.

## Required evaluation

Run the selector replay on a frozen outer split with the target labels hidden
from the selector. The benchmark must include at least:

```text
random selector
environment or metadata selector
static baseline features
raw intervention response
IRG response coordinates
```

The split used to choose probes, outcome weights, distance versions, and
thresholds is an inner split. A backbone, task regime, or repair primitive in
the outer split cannot be used for those choices. Report per-target and
aggregate values for top-1 success, selection regret, sign accuracy, ranking
correlation, negative transfer, abstention, calibration, and probe cost.

The command-level entry point is:

```bash
verdi research evaluate-irg \
  --plan selector-plan.json \
  --projections selector-projections.jsonl \
  --labels settled-effect-labels.json \
  --affinity primitive-probe-affinity.json \
  --output results/irg-replay
```

The result is reportable only when the manifest says the comparison is ready.
An incomplete probe basis remains an evidence gap and must produce an
abstention or a work order for a successor probe.

## Decision policy

Use the following policy when the outer replay is complete:

1. If IRG lowers selection regret and its calibration interval excludes the
   raw-response baseline, it may be used as a routing prior for the next
   target-side trial.
2. If IRG and raw response are statistically indistinguishable, keep IRG for
   diagnosis and use the simpler raw-response selector for routing. Do not
   claim an IRG selection improvement.
3. If IRG is worse than raw response, disable IRG routing for that task and
   retain the failed replay as negative evidence. The system should not
   silently tune the distance until the outer split is regenerated.
4. If support or locality is incomplete, abstain and request a measurement or
   a counterexample-driven successor probe. Missing coordinates are not zero
   response.

## Fallback: contextual repair-effect prediction

The fallback is not a more complicated distance. It predicts the outcome that
the user actually cares about: the effect of a concrete repair under a target
context. The training label is the settled paired effect of

```text
(target context, repair primitive, dose, schedule) -> metric delta
```

Features are grouped by provenance:

```text
target baseline metrics and failure rates
raw intervention response and uncertainty
IRG signed coordinates, support, and locality
capability and hook compatibility
data/task context overlap
repair primitive and dose metadata
```

Fit a small, inspectable model first, such as ridge regression with a fixed
feature schema and an isotonic calibration layer. Fit and tune it only on the
inner source environments. For every outer target, predict every candidate
repair, choose the highest lower confidence bound, and abstain when the
prediction interval overlaps zero or the target is outside training support.

This changes the scientific question from “are two response vectors close?” to
“does this model predict repair effect better than the available baselines?”
The evaluation must compare the contextual predictor against raw-response
nearest-neighbor, metadata, random, and IRG distance selectors under the same
source frames and budget.

## Three meanings of an IRG axis

The implementation should keep these meanings separate in reports and APIs:

```text
sensitivity   response to an intervention
vulnerability baseline failure or protected-metric risk in that direction
repairability held-out evidence that a named repair improves the target
```

Sensitivity can be high while vulnerability is low. Vulnerability can be high
while repairability is unknown. Only the third quantity can justify a repair
ranking, and it must come from independent target-side labels.

## When to evolve the probe basis

Probe evolution is justified only by a reproducible collision: nearby charts
with compatible support and opposite, confidence-separated repair effects.
The successor probe must be evaluated on a fresh outer context and must show a
downstream reduction in repair-selection regret or a measured increase in
accepted coverage. A lower locality residual alone is not sufficient evidence
that the new probe is useful.

If the contextual predictor remains no better than raw response after adding
task context and capability compatibility, the honest result is that the
current probe family does not predict repair transfer. Keep IRG as an
explanatory diagnostic and redirect effort to task-specific causal probes,
better effect labels, or a different decision objective.
