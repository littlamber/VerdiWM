# Method-to-Code Map

This document distinguishes implemented interfaces from empirical claims that
still require experiments.

| Method object | Code surface | Current evidence |
|---|---|---|
| Goal compiler and constitutional contract | `wmloop/control/user_intent_compiler.py`, `wmloop/constitution.py`, `configs/schemas/` | Implemented and contract tested |
| Verdict/diagnostic probe separation | `wmloop/diagnose/probe_registry.py`, `wmloop/diagnose/probes/` | Implemented; ACWM adapters available |
| Typed semantic intervention | `wmloop/geometry/types.py`, `wmloop/primitives/` | Implemented and unit tested |
| Intent-to-code materialization gate | `wmloop/propose/primitive_materialization_prompt.py`, `wmloop/verify/primitive_materialization_gate.py` | Implemented; 17 ACWM primitives materialized |
| Local response chart | `wmloop/geometry/irg.py` | Central/one-sided secants implemented and unit tested |
| IRG metric and response coordinates | `wmloop/geometry/irg.py`, `wmloop/geometry/assets.py` | Implemented and unit tested; eight ACWM-Phys assets included |
| Progressive-fidelity validation | `scripts/export/acwm_screen_summary.py`, `wmloop/verify/` | Operational on ACWM-Phys |
| Transfer certificate | `wmloop/geometry/transfer.py` | Six fail-closed terms implemented; calibration across backbones pending |
| Intervention-Effect Memory | `wmloop/geometry/memory.py`, `wmloop/archive/` | Positive/null/harmful/interaction records implemented |
| Repair-collision discovery | `wmloop/geometry/evolution.py` | Implemented and unit tested; online atlas evolution pending |
| Closed-loop orchestration | `wmloop/orchestrator.py`, `scripts/export/acwm_autoloop_daemon.py` | Operational ACWM search loop |
| Cross-backbone LOBO protocol | `wmloop/experiments/spec.py`, `wmloop/experiments/lobo.py` | CPU planner implemented; Ctrl-World is ready and the current pilot remains blocked on Cosmos3 plus settled target receipts |
| Settled stage ledger and paper tables | `wmloop/experiments/ledger.py`, `wmloop/experiments/report.py` | Contract implemented; no cross-backbone quality claim until confirm receipts are supplied |
| Ctrl-World ACWM pilot adapters | `wmloop/evaluate/adapters/ctrl_world_predictive.py`, `wmloop/experiments/ctrl_world_fingerprint.py`, `wmloop/primitives/adapters/ctrl_world_hooks.py` | Paired predictive receipt projection, reversible action-embedding dose, frozen ACWM constitution, and fail-closed downstream-success exclusion are complete; measured chart and transfer receipts are pending |
| Optional Ctrl-World downstream/WAM packet | `wmloop/evaluate/adapters/ctrl_world.py`, `configs/goal/ctrl_world_g2_action_success_pilot_v1.yaml` | Retained as a separate stress protocol and excluded from ACWM LOBO verdicts |
| Public minimal-loop proof | `examples/acwm_minimal_loop_cloth_next_forcing_v2/` | Integrity checked; not independent-seed replication |

## Intervention descriptor

An intervention is not identified by a method name alone. Its descriptor binds
the transformation, hook type, scope, dose unit, schedule, preconditions,
invariants, prediction, capability requirements, reversibility, and whether it
is inference-only. `compile_intervention` returns a receipt rather than silently
substituting an easier implementation.

## Interventional Repair Geometry

For local intervention doses `d` and goal outcomes `m`, VerdiWM estimates a
response Jacobian by paired central differences when both dose signs are
available, otherwise by a one-sided secant. Outcome weights induce the local
metric `G = J^T W J`. The current implementation also records response
coordinates, paired-seed covariance, locality residuals, support masks, and
source hashes. Canonical per-environment assets serialize these as `J_X`,
`G_X`, `r_X`, and `Sigma_X` through a stable symbol table.

IRG is intended to support mechanism-aware routing. The checked-in
`examples/acwm_unified_irg_assets_v1` bundle composes seven semantic probe
directions across all eight ACWM-Phys environments. It does not establish that
charts are aligned across arbitrary backbones. The contributing campaigns have
two incompatible zero-dose baseline frames, so cross-group covariance is
marked unobserved and every current asset abstains from transfer licensing.
Jointly paired chart data and held-out calibration are still required.

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

## ACWM reference instance

The public examples project ACWM runs into diagnosis, typed materialized
interventions, progressive-fidelity gates, long-horizon effect profiles,
routing memory, and measured IRG assets. The eight-environment asset bundle is
valid for within-instance routing. It carries an explicit transfer abstention,
so a cross-backbone certificate remains an unestablished empirical claim.

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
