# Clean Architecture

```text
Goal -> Adapter.inspect -> probes/fingerprint -> portrait readiness
     -> bounded experiment portfolio -> frozen verifier
     -> append-only evidence ledger -> portable knowledge projection
```

The Kernel owns authority: schemas, budgets, lifecycle receipts, split and
verifier bindings, promotion, and durable state. Adapters own model-specific
runtime details. Probe results describe behavior; they do not decide promotion.
Knowledge records preserve positive, null, harmful, and abstained outcomes with
their claim boundaries. Transferable knowledge is a projection of settled
evidence, not a training recipe copied between models.

The reference fixture is deliberately CPU-only. Real adapters can implement
the same protocol at L0-L3 without changing Kernel code.

## Engineering agent boundary

`verdi_core.engineering` provides the model-facing engineering loop as a
backend-neutral tool protocol. An OpenAI-compatible model (for example
GPT-5.6) returns one structured action at a time; the Kernel validates and
executes it inside the run's isolated worktree/output roots. Supported actions
cover file inspection, patch application, tests/commands, worktree creation,
and artifact collection. AI and tool receipts are redacted and append-only.

Adapters may compose this loop with `autonomous_campaign(...)`. The adapter
still owns model-specific commands and evaluator semantics; the Kernel owns
path, network-write, privilege, GPU, timeout, budget, and evidence authority.
Codex is an optional future backend, not a clean-kernel dependency.
