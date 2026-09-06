# VerdiWM research position

VerdiWM is an evidence-constrained, model-agnostic research agent for world
models. Its central contribution is the combination of four boundaries:

1. Interventional Repair Geometry (IRG) binds measured intervention responses
   to a model portrait so retrieval can target a diagnosed bottleneck.
2. Counterexample collisions produce new diagnostic probe proposals instead of
   silently extrapolating beyond the current geometry.
3. Cross-domain literature providers stage methods as untrusted, typed evidence
   that must pass capability and falsifiability checks before execution.
4. A constitution and independent paired verifier separate exploration from
   scientific claims, retaining negative and harmful evidence.

The project should currently be described as **evidence-grounded research
self-improvement**, or **RSI-adjacent**, rather than strict recursive
self-improvement. The system can improve its probe basis, retrieval routing,
and experiment portfolio, but it does not yet autonomously deploy a stronger
successor of its own research kernel and prove recursive gains.

The following claims remain empirical work, not implementation claims:

- IRG-guided retrieval must beat static embeddings and random retrieval;
- collision-driven probe evolution must improve method-effect prediction or
  reduce GPU cost to a fixed evidence target;
- repeated closed-loop rounds must improve the research policy itself;
- any successor system must pass an independent frozen benchmark before
  replacing the current kernel.

External projects are used as bounded design references. Their autonomous
runners are not imported into the execution path, and no external component
has authority to execute code, alter held-out data, modify frozen evaluators,
or promote a model result.
