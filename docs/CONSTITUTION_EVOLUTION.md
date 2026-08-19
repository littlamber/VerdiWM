# Constitutional Evolution

VerdiWM separates model improvement, optimizer improvement, and constitutional
change:

- L0 may change model parameters, training data, or inference behavior.
- L1 may change probes, primitives, compilers, routing, or memory.
- L2 may propose changes to goals, verifiers, protected metrics, or allowed
  actions.

The current agent may propose an L2 candidate, but it cannot authorize one.
`wmloop.control.constitution_proposal` records the candidate as a
content-addressed contract and keeps it outside verdict authority.

## Shadow lifecycle

```text
candidate -> shadow -> probation -> approved
       \-> rejected                 approved -> revoked
```

The proposal must keep the parent protected metrics (`append_only`). New
metrics are diagnostic until approval; a candidate cannot add a primary or
guard verdict role, and it cannot promote a metric before approval.

`shadow -> probation` requires incremental predictive value, anti-Goodhart
checks, and a frozen regression battery. `probation -> approved` additionally
requires independent validation, a canary, and an external approval quorum.
The proposal verdict always reports `verdict_authority: false`; a separately
frozen constitutional manifest remains the only source of claim authority.

This is intentionally a small control-plane seam. It does not modify the
dispatcher or replace an existing evaluator. Probe runs, ablations,
negative controls, and historical Evidence IR remain the sources of evidence
for the gates.

## Policy-Bounded Automatic Transition

`wmloop.control.constitution_transition` adds the unattended path without
weakening that boundary. A version-two proposal may request automatic promotion
of one or more *diagnostic* metrics only after attaching content-addressed
receipts for static checks, shadow evaluation, historical calibration, a fresh
held-out battery, and a canary. Each proposed metric also needs a matching
`wmloop-metric-adequacy-report` which has passed discrimination, stability,
anti-Goodhart, incremental-information, protected-metric, and fresh-heldout
gates. It cannot directly become `approved`.

An installed `wmloop-constitution-transition-policy` is a path listed in the
successor constitutional config's `frozen_code` and is therefore part of that
new freeze without modifying historical configs. It fixes the parent
constitution identity, the complete protected-metric set, allowable successor
roles, promotion limit, and the only legal change set:

- goal and verifier stay unchanged;
- protected metrics are append-only;
- allowed actions stay unchanged;
- only future work may use a successor constitution;
- historical verdicts remain append-only.

When all checks pass, the module writes a deterministic transition receipt.
The receipt has `verdict_authority: false` and requires a separately frozen
successor constitution before any new metric can participate in future
verdicts. Existing constitutional configs intentionally have no transition
policy until a calibrated policy is explicitly added and frozen.
