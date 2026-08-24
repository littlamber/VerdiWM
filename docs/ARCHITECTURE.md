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

