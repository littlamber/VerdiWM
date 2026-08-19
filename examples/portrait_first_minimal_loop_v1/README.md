# Portrait-First Minimal Loop

This CPU-only example demonstrates the planning half of the public VerdiWM
control plane. It constructs a read-only onboarding report, a Model Capability
IR, a behavioral fingerprint, a Model Portrait, a goal-relative readiness
receipt, and a capability-gap plan. It does not load a model, reserve a GPU,
generate code, or make a scientific claim.

Run it from the repository root:

```bash
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

The result is canonical JSON. The successful state sequence is:

```text
model capability -> portrait -> ready_for_gap_planning -> ready_for_portfolio
```

The example exists to verify that a fresh public checkout contains the schemas,
plugin registry, and control-plane APIs required to begin a real deployment.
