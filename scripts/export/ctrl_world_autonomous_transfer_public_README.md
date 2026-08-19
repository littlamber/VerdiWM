# Portrait-First Autonomous Transfer Controller

This package is the durable controller for VerdiWM's portrait-first research
loop. It receives a deployment-specific model connector, data regime, frozen
evaluator, resource policy, and state root at runtime. None of those runtime
bindings are included in the public release.

The controller enforces this order:

```text
Goal IR -> Model Capability IR -> Model Portrait -> readiness receipt
-> capability gap plan -> experiment portfolio -> module work order
-> resource admission -> screen -> confirm -> frozen verification
-> portable knowledge projection -> replan or stop
```

Use the public CPU example before binding a real model:

```bash
uv run python examples/portrait_first_minimal_loop_v1/run.py
verdiwm-ctrl-world-autonomous-transfer --help
```

For a real deployment, create a new local configuration and state root outside
this checkout. Bind model paths, datasets, credentials, evaluator code, and
GPU inventory only in that local deployment configuration. The controller will
reject a missing portrait, unbound evaluator, unavailable capability, or
unadmitted resource request rather than substituting a nearby implementation.

The public package is a control-plane release. It does not include model
weights, datasets, private artifacts, or a claim that a particular intervention
improves a target model.
