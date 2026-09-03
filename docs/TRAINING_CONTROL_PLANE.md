# Training control plane

The experiment designer must not let a model runner decide whether a probe is
"good enough" to count as training. VerdiWM therefore emits a model-neutral
training scale plan and, when requested, a complete stage ladder:

```text
probe (runtime-only) -> screen (diagnostic) -> pilot (formal) -> confirm (formal)
```

`verdiwm plan-training-ladder` writes every upgrade as a receipt. Each formal
stage requires long training, episode-balanced sampling, a declared update
budget, a validation manifest, and episode-disjoint train/validation IDs.
Missing episode IDs, overlapping splits, insufficient episode diversity, and
under-sized formal budgets fail closed. A failed screen is retained as
diagnostic evidence but has `screen_failure_veto: false`; only the frozen
formal evaluator can make a quality decision.

The runner contract is model-independent. A backend may translate it to
native flags, but the control plane supplies immutable values for stage, mode,
steps, record limit, sampler, seed count, and scale-plan digest through the
`VERDIWM_TRAINING_*` environment variables.

Prompts remain useful for proposing an idea or implementing an adapter inside
an isolated worktree. They are not a sufficient safety mechanism for training
scale, data separation, GPU budgets, frozen evaluators, or promotion. Those
boundaries belong in schemas, immutable input locks, stage state machines,
budget ledgers, runner contracts, and receipt-first settlement.
