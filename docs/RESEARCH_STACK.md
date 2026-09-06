# Research-stack integration policy

VerdiWM incorporates ideas from open research-agent and experiment-design
projects through bounded adapters, not by importing their autonomous runners.
The authoritative inventory is
`configs/research/research_stack_v1.json`; validate it with:

```bash
uv run verdiwm-research-stack-audit
```

The adopted patterns are:

- AI Scientist-v2-style proposal search, constrained to proposal authority;
- Darwin Gödel Machine-style isolated successor staging, with no direct writes
  to the active checkout;
- Ax/BoTorch-style acquisition for diagnostic probe ranking, while CPBE's
  capability-checked DSL remains authoritative;
- Agent Laboratory-style staged research planning and cumulative evidence
  retrieval;
- PromptWizard-style critique/refinement for probe work orders;
- Hydra/Ray-style explicit configuration and durable scheduling, subordinate to
  VerdiWM budget leases and receipt settlement.

GPT Researcher is retained as a reference for cited web retrieval only. The
provider layer remains data-only: retrieved code is not executed and external
records cannot bypass compilation, frozen evaluators, or artifact gates.

This is an integration record, not a claim that any external project or
borrowed strategy improves model quality. Scientific authority remains in the
constitution, paired verifier, archive, and release gates.

## Scope boundary

The stack manifest deliberately exposes only `advisory`, `proposal_only`, and
`ranking_only` authorities. There is no `execute` authority for external
components. Any future self-improvement work must be staged in an isolated
successor campaign and accepted by an independent frozen benchmark before it
can become the next system version.
