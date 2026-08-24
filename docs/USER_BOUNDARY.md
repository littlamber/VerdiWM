# User Boundary

Users do **not** need to implement the scheduler, worker, evaluator, retrieval
system, idea generator, probe registry, or knowledge graph.

The only model-specific boundary is a thin Adapter manifest and runtime
binding. It tells VerdiWM how to load or call the model, how to submit an
intervention, and what raw outputs are available. A user may provide an SDK
path, an HTTP endpoint, or a small wrapper; the system can generate a wrapper
from the SDK documentation and run the adapter contract tests before admission.

The only domain-specific boundary is data/evaluator metadata: where the
observations and labels live, which split is held out, and which signals are
observable. The AI metric advisor can propose metric implementations and the
generic evaluator can run them, but it cannot invent unavailable ground truth
or credentials. A human must approve generated code and provide missing access
when the domain has no machine-readable oracle.

The intended onboarding is therefore:

```text
SDK/API + data manifest + objective
    -> AI-generated adapter/evaluator draft
    -> sandbox contract tests and metric adequacy checks
    -> human approval for permissions and scientific assumptions
    -> autonomous research cycles
```

This keeps model-specific code outside the Kernel while avoiding a requirement
that users hand-build experiment infrastructure.
