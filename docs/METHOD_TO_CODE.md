# Method-to-Code Map

This document distinguishes implemented interfaces from empirical claims that
still require experiments.

## Open implementation and pairwise studies

The open-method path accepts candidate-local implementation and training files
without a fixed primitive family. Its runnable generation bridge is
`wmloop.control.open_method_generation.generate_open_method`: it invokes the
configured `run_llm_task` adapter, checks the response digest, target portrait,
source provenance and component identities, then calls the isolated proposal
compiler. Provider failures and rejected proposals retain local receipts. The
bridge does not execute generated code or certify its scientific correctness.

`build_open_method_request` accepts source evidence, the target portrait,
probe fingerprints, failure context, a kernel-owned `target_portrait_binding`,
and optionally two validated `component_methods`. It binds the complete input
and prompt to the request identity. Components are hypotheses unless separately
supported by target-side evidence; their supplied source digests identify the
evidence used, not independent verification of a paper's claims.

New ready proposals require `implementation_validation` in Method IR. Each
check names a declared test, observable and failure condition. All methods
declare hook execution, no future leakage and an ablation effect; training
methods also declare optimizer binding, actual parameter updates and
train/inference parity; stateful methods declare state lifecycle checks.
Referenced implementation files must exist in the proposed bundle. Compilation
emits `implementation-check-plan.json` with `state=declared_not_executed`.
These checks must subsequently run on the target. A declared check, import
success, decreasing loss or an LLM assertion cannot establish implementation
fidelity or improvement. Older Method IR without this field remains readable,
but cannot be compiled as a new ready proposal.

For A+B, `composition` binds two normalized Method IR IDs and describes
complementarity, the predicted joint effect, conflict resolution and
anti-conditions. `compile_open_method_study` compiles four isolated bundles:

| Role | Comparison purpose |
| --- | --- |
| `baseline` | Unmodified target control, represented by an explicit proposal |
| `source_only` | A's effect and removal of B |
| `target_only` | B's effect and removal of A |
| `combined` | Joint effect and removal of neither |

The kernel supplies a common checkpoint digest, train/selection/confirmation
split digests, verifier digest, paired seeds, per-arm cost estimates and total
GPU budget. Arms must agree on portrait and metric bindings; incomplete,
inconsistent or over-budget studies do not publish a partial study. The first
version supports pairs only. Method IDs must differ, but semantic identity
cannot be proved by hashing: target-side checks must verify that baseline has
no intervention and that each ablation changes only its intended mechanism.

`study.json` binds every compiled file's hash and requires contrasts A−base,
B−base, AB−base, AB−A, AB−B, and AB−A−B+base. Effects must first be oriented so
larger means better. Improvement over both individual methods and positive
additive interaction are distinct claims. Both need paired uncertainty and
independent confirmation, with protected metrics and training/inference costs.
Distinct split hashes do not prove disjoint episodes, and a cost estimate is
not a GPU lease or an enforced runtime limit. Runtime validation and the
existing budget ledger remain necessary.

Before target-side experiments, `verdiwm-calibrate-method` executes the declared
implementation tests in the isolated candidate workspace and writes a bounded
calibration receipt. A failed check is retained as `state=failed` and cannot be
promoted or treated as a model result. This local calibration intentionally has
no GPU, evaluator, network or source-write authority.

Deployment-facing commands (the request files are internal research artifacts):

```bash
uv run verdiwm-generate-method \
  --request /workspace/research/open-method-request.json \
  --adapter /workspace/research/llm-adapter.json \
  --base-revision /workspace/research/base-revision.json \
  --output /workspace/research/generated-a

uv run verdiwm-open-method-study \
  --request /workspace/research/study-request.json \
  --output /workspace/research/study-ab

uv run verdiwm-run-open-method-study \
  --study /workspace/research/study-ab \
  --checkpoint /workspace/assets/checkpoint.pt \
  --train-split /workspace/assets/train.json \
  --selection-split /workspace/assets/selection.json \
  --confirmation-split /workspace/assets/confirmation.json \
  --verifier /workspace/research/verifier/verifier.json \
  --runtime-python /workspace/model/.venv/bin/python \
  --output /workspace/research/executions/study-ab \
  --gpus 0
```

Use the model's own runtime when its dependencies are separate from VerdiWM;
the command defaults to the current Python only when no runtime is supplied.
Use fresh output directories outside the VerdiWM checkout. The first request
is produced by `build_open_method_request`; the adapter uses the existing
trusted LLM adapter configuration. The second request contains `proposals`
with exactly the four roles above, `base_revision`, `target_portrait_binding`,
`experiment_binding` (the five `*_digest` fields above), `seeds`,
`estimated_gpu_hours_per_seed` (one positive value per role), and
`budget_gpu_hours`. Module entrypoints are also available through
`python -m wmloop.control.open_method_generation` and
`python -m wmloop.control.open_method_study`.

These entrypoints connect provider output to real candidate files and complete
study compilation. `verdiwm-run-open-method-study` then executes the declared
checks, bounded candidate operations, frozen verifier, paired contrasts and
local CAS evidence archive. The default literature pipeline still needs an integration
that selects open candidates, executes their target checks, submits the four
arms to the scheduler and deposits settled results. Existing registered
primitive compositions retain their separate execution and settlement path in
`mechanism_composition.py`; the new study does not bypass that verifier or
promote an open method on compilation alone.

## Implementation map

| Method object | Code surface | Current evidence |
|---|---|---|
| Goal compiler and constitutional contract | `wmloop/control/user_intent_compiler.py`, `wmloop/constitution.py`, `configs/schemas/` | Implemented and contract tested |
| Verdict/diagnostic probe separation | `wmloop/diagnose/probe_registry.py`, `wmloop/diagnose/probes/` | Implemented; ACWM adapters available |
| Canonical base intervention-probe contract | `configs/probes/irg_base_v1.json`, `configs/schemas/irg_base_probe_registry.schema.json` | Four semantic families frozen; only action scaling is measured in the current ACWM atlas |
| Typed semantic intervention | `wmloop/geometry/types.py`, `wmloop/primitives/` | Implemented and unit tested |
| Intent-to-code materialization gate | `wmloop/propose/primitive_materialization_prompt.py`, `wmloop/verify/primitive_materialization_gate.py` | Implemented; 17 ACWM primitives materialized |
| Local response chart | `wmloop/geometry/irg.py` | Central/one-sided secants implemented and unit tested |
| IRG metric and response coordinates | `wmloop/geometry/irg.py`, `wmloop/geometry/assets.py` | Implemented and unit tested; eight ACWM-Phys assets included |
| Model-conditioned IRG binding | `wmloop/geometry/model_irg.py`, `configs/schemas/model_irg.schema.json` | Binds a measured response vector to one Model Portrait, expands coordinate-level diagnoses, attaches method effects and collision/evolution evidence, and exposes ranking-only IRG distance/collision queries |
| IRG-guided cross-domain discovery | `wmloop/retrieve/irg_guided_discovery.py` | Converts supported IRG sensitivity hotspots into explicit local-bottleneck hypotheses, cross-domain research lenses, and shadow-only `DiscoveryRequest` records; global ceilings still require dose/horizon sweeps |
| Joint-frame probe calibration | `wmloop/experiments/joint_fingerprint.py`, `scripts/run_acwm_joint_fingerprint_*.py` | 600-condition ACWM-Phys pilot complete; eight full-covariance assets included |
| Adaptive locality-radius settlement | `wmloop/experiments/ctrl_world_fingerprint_settlement.py`, `scripts/export/ctrl_world_fingerprint_settlement.py` | Wide Ctrl-World radius rejected; radius 0.025 admitted on the frozen pilot split |
| Progressive-fidelity validation | `scripts/export/acwm_screen_summary.py`, `wmloop/verify/` | Operational on ACWM-Phys |
| Probe information/collision evidence export | `wmloop/experiments/probe_information.py`, `wmloop/experiments/random_probe_expansion.py`, `wmloop/experiments/collision_labels.py` | S4 retains all three conditions, 80 preregistered random-subset replays, four redundancy-smoke comparisons, and eight independently frozen collision cases. It remains partial because the evolved certificate accepted zero folds, so post-evolution collision rate is undefined |
| Transfer certificate | `wmloop/geometry/transfer.py`, `wmloop/experiments/cosmos3_directional_settlement.py` | Six fail-closed terms implemented; Cosmos3 dev/accept split reversal correctly abstained before LOBO |
| Intervention-Effect Memory | `wmloop/geometry/memory.py`, `wmloop/archive/` | Positive/null/harmful/interaction records implemented |
| Repair-collision discovery | `wmloop/geometry/evolution.py` | Implemented and unit tested; online atlas evolution pending |
| Counterexample-driven probe evolution | `wmloop/experiments/probe_evolution.py`, `scripts/export/probe_evolution.py`, `scripts/export/probe_evolution_settlement.py` | Cosmos3 scale counterexamples produced and evaluated a novel temporal-mix probe; the 15-cell successor correctly settled as abstained |
| Counterexample-Guided Probe Basis Expansion | `wmloop/experiments/cpbe.py`, `wmloop/experiments/acwm_cpbe_bootstrap.py`, `wmloop/experiments/acwm_cpbe_canary.py`, `wmloop/experiments/cpbe_counterexample.py`, `wmloop/experiments/cpbe_materializer.py`, `configs/schemas/cpbe_request.schema.json`, `configs/schemas/cpbe_stage_receipt.schema.json` | Probe DSL, four-source synthesis, frozen ACWM evidence adapter, target-label-free source-sign projection, counterexample learner, deterministic materializer, evidence-conditioned acquisition, capability filtering, direct-parent canary preparation, and hash-bound successive-halving settlement implemented and unit tested; r29/r30 admitted zero candidates |
| Closed-loop orchestration | `wmloop/orchestrator.py`, `scripts/export/acwm_autoloop_daemon.py` | Operational ACWM search loop |
| Cross-backbone LOBO protocol | `wmloop/experiments/spec.py`, `wmloop/experiments/lobo.py` | CPU planner implemented; Ctrl-World chart is settled, while Cosmos3 LOBO is blocked by three frozen locality abstentions |
| Universal diagnostic-first onboarding | `wmloop/diagnose/probe_campaign.py`, `wmloop/retrieve/index.py`, `wmloop/execute/autonomous_pipeline.py` | Declarative GPU probe receipt precedes experience retrieval and candidate compilation; current probe is excluded from retrieval and empty indexes settle as `cold_start` |
| Online cold-start method discovery | `wmloop/retrieve/literature.py`, `wmloop/retrieve/method_staging.py`, `wmloop/propose/prior_library.py`, `configs/schemas/literature_method_candidate.schema.json` | Bounded arXiv lookup stages data-only records; strict synthesis maps registered methods to ranking-only evidence and unknown methods to prompt-compatible next-version work orders with no command or GPU authority |
| Settled stage ledger and paper tables | `wmloop/experiments/ledger.py`, `wmloop/experiments/report.py` | Contract implemented; no cross-backbone quality claim until confirm receipts are supplied |
| Ctrl-World ACWM pilot adapters | `wmloop/evaluate/adapters/ctrl_world_predictive.py`, `wmloop/experiments/ctrl_world_fingerprint.py`, `wmloop/primitives/adapters/ctrl_world_hooks.py` | Paired predictive receipt projection, reversible action-embedding dose, frozen ACWM constitution, and fail-closed downstream-success exclusion are complete; the pilot chart is measured and settled, while paper-split transfer receipts are pending |
| Ctrl-World ACWM two-surface scale gate | `wmloop/control/acwm_dual_evaluation.py`, `configs/experiments/ctrl_world_acwm_dual_evaluation_v1.json` | Exactly two paired-ground-truth surfaces use Pareto admission: primary long-horizon prediction must improve and action/trajectory protections may not regress. It excludes WAM/task-success inputs and remains non-authoritative until frozen verification. |
| Public minimal-loop proof | `examples/acwm_minimal_loop_cloth_next_forcing_v2/` | Integrity checked; not independent-seed replication |

## Intervention descriptor

An intervention is not identified by a method name alone. Its descriptor binds
the transformation, hook type, scope, dose unit, schedule, preconditions,
invariants, prediction, capability requirements, reversibility, and whether it
is inference-only. `compile_intervention` returns a receipt rather than silently
substituting an easier implementation.

## Interventional Repair Geometry

Probe terminology is layer-qualified. `configs/probes/acwm_v1.json` registers
passive outcome and verdict diagnostics; its four entries are coordinates of
the measured outcome vector, not columns of `J`. The canonical base
intervention bank is `configs/probes/irg_base_v1.json` and contains action
scaling, controlled context retention, first-frame anchoring strength, and
sampler-noise stress. An instantiated atlas may split these families by dose
polarity or add counterexample-driven successor paths. The seven columns in
the ACWM-Phys v1 joint frame are such admitted paths, not seven base probes.

For local intervention doses `d` and goal outcomes `m`, VerdiWM estimates a
response Jacobian by paired central differences when both dose signs are
available, otherwise by a one-sided secant. Outcome weights induce the local
metric `G = J^T W J`. The current implementation also records response
coordinates, paired-seed covariance, locality residuals, support masks, and
source hashes. Canonical per-environment assets serialize these as `J_X`,
`G_X`, `r_X`, and `Sigma_X` through a stable symbol table.

IRG is intended to support mechanism-aware routing. The immutable
`examples/acwm_unified_irg_assets_v1` snapshot records the discovery that the
original atlas mixed parallel and autoregressive baseline frames. The corrected
`examples/acwm_joint_irg_assets_v2` campaign reruns all seven semantic
directions in autoregressive mode, with one no-hook baseline for each
environment and seed. All eight resulting assets contain observed cross-path
covariance in one baseline-compatible block.

This makes the ACWM reference geometry complete, but it does not establish
alignment across arbitrary backbones. A target-backbone chart, semantic hook
compilation, held-out calibration, and effect confirmation remain necessary.

The checked-in `examples/ctrl_world_target_local_irg_v1` bundle demonstrates
the target-chart step. A radius-0.1 campaign violated the locality threshold,
so `ctrl_world_fingerprint_settlement` retained the failed chart and selected
the widest passing recalibration, radius 0.025. This converts nonlinearity into
an explicit abstention-and-remeasure path instead of silently fitting one global
Jacobian. It still does not provide a transferred repair effect.

## Transfer certificate

A source intervention is licensed on a target only when all terms pass:

1. semantic compilation succeeds on the target capability profile;
2. source/target support overlaps;
3. effective sample size is sufficient;
4. aligned chart error is below threshold;
5. effect signs agree;
6. the calibrated lower confidence bound exceeds the goal threshold.

Any failed term yields `status=abstain` with explicit reasons. The certificate
is designed to prevent similarity-only transfer claims.

## Atlas evolution

The effect memory preserves negative information. When nearby atlas points
have statistically confident opposing effects for the same primitive, VerdiWM
records a repair collision. Candidate probes are then ranked by a lower
confidence bound on nested regret reduction per unit cost, subject to
calibration and frozen regression checks.

The Cosmos3 instance exercises this path without granting itself a positive
result. Wide and narrow `action_conditioning_scale` charts failed locality at
`0.9510` and `45.6476`. Their counterexample record proposed a novel reversible
`action_embedding_temporal_mix` direction that preserves each action
dimension's temporal mean. The successor was materialized and evaluated over
15 paired cells, then failed the unchanged `0.5` gate with residual `2.0649`.
`probe_evolution_settlement` retains all three failures and returns
`settled_abstained`. This is evidence for an executable self-evolution loop and
its safety boundary, not evidence for model repair or cross-backbone transfer.

A second branch decomposed the narrow scale probe by polarity using dev data
only. The positive one-sided doses `[0, 0.0125, 0.025]` passed the unchanged
locality gate on dev (`0.3004`) and independently on accept (`0.3238`). This is
not sufficient for transfer: the unit-Frobenius Jacobian alignment error was
`1.999998`, above the frozen `0.5` threshold, because the dominant response
reversed sign. `cosmos3_directional_settlement` therefore returns
`settled_abstained`. The public
`examples/cosmos3_directional_probe_split_reversal_v1` bundle retains both
local charts, all accept dose-response videos, and the final certificate
counterexample.

The latest branch replaces diagonal contrast scaling with a mean- and
energy-preserving orthogonal rotation over four frozen action-dimension pairs.
It passes the dev4 locality gate (`0.0065`) but fails independent accept4
locality (`0.5672`) and reverses the normalized Jacobian direction
(`1.9998 > 0.5`). The
`examples/cosmos3_action_dimension_interaction_split_reversal_v4` bundle
retains the complete certificate counterexample. The result demonstrates
fail-closed probe evolution; it does not license LOBO or establish repair.

## ACWM reference instance

The public examples project ACWM runs into diagnosis, typed materialized
interventions, progressive-fidelity gates, long-horizon effect profiles,
routing memory, and measured IRG assets. The corrected eight-environment bundle
is valid for within-instance routing and full cross-path covariance analysis.
A cross-backbone certificate remains an unestablished empirical claim until a
compatible target chart and held-out effect receipts exist.

## Cross-backbone experiment control plane

The paper-facing evidence inventory is frozen in
`configs/experiments/verdiwm_iclr_evidence_matrix_v1.json`. Export the
reviewable Markdown, CSV, and LaTeX tables with:

```bash
python scripts/export/verdiwm_paper_experiment_matrix.py \
  --config configs/experiments/verdiwm_iclr_evidence_matrix_v1.json \
  --output-root results/reports/verdiwm-iclr-evidence-matrix-r1
```

The full selector ablation belongs on all eight ACWM-Phys environments because
that is the reference-instance mechanism test. It is not sufficient for a
cross-backbone claim. The IRG, raw-response, static-probe, and label selectors
must also be compared on held-out target trials from at least two external
backbone families. The current minimum targets are Ctrl-World and Cosmos3;
WAM is the recommended additional stress target.

Target charts are a hard prerequisite, not a box-checking step. The Cosmos3
temporal-mix successor remains unsupported under the frozen locality gate, and
the locally admitted positive-scale branch fails held-out alignment. The
planner must not manufacture a warm-start LOBO arm from either branch. The next
executable paper experiment is a new pre-registered diagnostic axis or an
independent external backbone whose target chart and held-out alignment both
pass admission.

The checked-in LOBO pilot specification is
`configs/experiments/three_backbone_lobo_pilot_v1.json`. It is deliberately a
protocol artifact rather than a fabricated result. Generate the deterministic
trial plan with:

```bash
uv run verdiwm-experiment-plan \
  --spec configs/experiments/three_backbone_lobo_pilot_v1.json \
  --output-root results/reports/three-backbone-lobo-plan-r1 \
  --archive-db results/archive.db \
  --cas-root results
```

The three arms have different meanings: `warm_start` may consume source
Effect Memory and chooses one of `environment_label`, `static_probe`,
`raw_response`, or `irg`; `cold_start` uses only target-local diagnosis; and
`random_search` samples the target-compatible registry uniformly. The latter
is not an alias for `shuffled_prior`.

Each executed stage must emit a settled
`verdiwm-experiment-stage-receipt`. Produce paper tables with:

```bash
uv run verdiwm-experiment-report \
  --spec configs/experiments/three_backbone_lobo_pilot_v1.json \
  --receipt-dir results/experiments/three_backbone_lobo_pilot_v1/receipts \
  --output-root results/reports/three-backbone-lobo-report-r1 \
  --archive-db results/archive.db \
  --cas-root results
```

Only a settled `confirm` receipt can establish a formal positive. Missing
confirm receipts, screen/gate positives, and unsettled jobs are not promoted
into the paper result tables. The report exports deterministic CSV and LaTeX
tables for hit rate, negative transfer, abstention, coverage/risk, and stage
GPU cost; values remain incomplete until real receipts exist.
