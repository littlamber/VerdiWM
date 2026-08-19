# Ctrl-World ACWM Guidance Calibration v1

This package runs one bounded, source-preserving ACWM calibration batch. It
uses the frozen dual-surface contract and never reads or reports WAM policy,
task-success, task-progress, or safety signals.

The batch assigns one immutable guidance-scale candidate to each of eight
physical GPUs. Every worker writes a model measurement, physical-GPU runtime
receipt, terminal settlement, and a normalized knowledge record under the
external output root. A screen acceptance is only eligible for a later
held-out confirm; it is not a promotion claim.

Run `run.py --help` for the explicit asset and output-root arguments. The
checked-in batch is `configs/experiments/ctrl_world_acwm_guidance_batch_v1.json`.

`run.py --autoloop` is the bounded autonomous mode. It reads prior screen and
confirm settlements, creates a new batch only when its configured evidence
trigger matches, and runs confirmation only for screen admissions. `--watch`
keeps the controller alive for newly frozen `inbox/*.json` screen batches; it
does not execute unstructured instructions or mutate model source.

`run.py --research-scan` is the separate discovery intake. It reads bounded
arXiv and GitHub metadata, rejects instruction-like source content, and writes
source-linked ideas plus materialization work orders. It does not run a GPU or
modify Ctrl-World. A research work order needs an isolated implementation
receipt and an immutable ACWM batch before it may enter the ordinary
`screen -> confirm -> frozen verifier` route. This separation keeps external
inspiration useful without treating a paper or project as execution authority.
