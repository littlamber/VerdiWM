# Contributing

Keep the Kernel model-agnostic and dependency-light. Model runtimes belong in
external adapters; do not add model names, host paths, weights, datasets, or
GPU launch assumptions to `verdi_core`.

Before opening a change, run `scripts/release_preflight.sh`. New adapters must
pass the `ModelAdapter` contract tests without editing Kernel code. Changes to
evidence semantics must include a test for positive, null, harmful, and
abstained outcomes and document the claim boundary.
