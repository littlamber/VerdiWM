# Contributing

Contributions should preserve VerdiWM's claim boundary.

1. Open an issue describing the failure mechanism, intended intervention, and
   falsifiable expected effect.
2. Keep changes narrow and add an offline harness or unit test.
3. Do not modify frozen evaluators, held-out splits, acceptance thresholds, or
   existing evidence to make a candidate pass.
4. For a new primitive, declare its semantic descriptor, required capability,
   hook, dose, invariants, and intent-to-code receipt.
5. For a new backbone, add capability, goal, split, evaluator, hook, receipt,
   and regression-harness surfaces before claiming closed-loop support.
6. Run `bash scripts/ci/check_control_plane.sh` before submitting a change.

Generated code is reviewed under the same standard as handwritten code. If an
implementation cannot preserve the declared method intent, it must fail closed
or return to proposal rather than silently substitute a weaker method.

Do not commit model weights, datasets, private logs, credentials, machine-local
paths, or generated `results/` and `runs/` trees.

