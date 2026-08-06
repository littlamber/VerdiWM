# Method-to-Code Map

This document distinguishes implemented interfaces from empirical claims that
still require experiments.

| Method object | Code surface | Current evidence |
|---|---|---|
| Goal compiler and constitutional contract | `wmloop/control/user_intent_compiler.py`, `wmloop/constitution.py`, `configs/schemas/` | Implemented and contract tested |
| Verdict/diagnostic probe separation | `wmloop/diagnose/probe_registry.py`, `wmloop/diagnose/probes/` | Implemented; ACWM adapters available |
| Canonical base intervention-probe contract | `configs/probes/irg_base_v1.json`, `configs/schemas/irg_base_probe_registry.schema.json` | Four semantic families frozen; only action scaling is measured in the current ACWM atlas |
| Typed semantic intervention | `wmloop/geometry/types.py`, `wmloop/primitives/` | Implemented and unit tested |
| Intent-to-code materialization gate | `wmloop/propose/primitive_materialization_prompt.py`, `wmloop/verify/primitive_materialization_gate.py` | Implemented; 17 ACWM primitives materialized |
| Local response chart | `wmloop/geometry/irg.py` | Central/one-sided secants implemented and unit tested |
| IRG metric and response coordinates | `wmloop/geometry/irg.py`, `wmloop/geometry/assets.py` | Implemented and unit tested; eight ACWM-Phys assets included |
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
| Optional Ctrl-World downstream/WAM packet | `wmloop/evaluate/adapters/ctrl_world.py`, `configs/goal/ctrl_world_g2_action_success_pilot_v1.yaml` | Retained as a separate stress protocol and excluded from ACWM LOBO verdicts |
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
