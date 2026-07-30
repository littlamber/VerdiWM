# VerdiWM

VerdiWM is an evidence-grounded research loop for diagnosing and repairing
world models. It turns a user goal into a frozen evaluation contract, probes a
model for failure mechanisms, compiles bounded intervention primitives into
runtime changes, validates them at progressive fidelity, and retains positive,
null, and harmful effects for later transfer.

![VerdiWM overview](figures/verdiwm_overview.png)

The repository is an early research release. It contains a working control
plane, typed intervention and evidence contracts, ACWM-Phys, Ctrl-World, and
Cosmos3 ACWM adapters, 17 materialized intervention primitives, an
Interventional Repair Geometry (IRG) implementation, eight joint-frame
ACWM-Phys IRG assets, development and paper-split Ctrl-World charts, one
Cosmos3 forward-dynamics instantiation bundle, six settled Cosmos3
target-local charts, three held-out directional split settlements, three
counterexample-driven probe-evolution settlements, and one integrity-checked
operational closed-loop example.
The release also includes a receipt-derived progressive-fidelity cost audit;
its modest savings are retained as a system limitation rather than hidden.

## What is implemented

| Surface | Release status |
|---|---|
| Goal, probe, proposal, trial, verdict, and evidence schemas | Implemented |
| Fail-closed intervention compilation and runtime receipts | Implemented |
| Diagnosis, proposal, budgeted execution, independent verification, archive | Implemented |
| Per-environment horizon ladder and frozen verifier checks | Implemented |
| IRG local response charts, composed assets, and distances | Implemented, unit tested; eight joint-frame ACWM-Phys assets included |
| Transfer certificate with explicit abstention | Implemented, unit tested |
| Intervention-Effect Memory and counterexample discovery | Implemented, unit tested |
| ACWM-Phys operational minimal loop | Included as a public evidence bundle |
| Ctrl-World ACWM predictive-quality protocol | Dev chart admitted; independent paper split completed and correctly abstained |
| Cosmos3-Nano ACWM forward-dynamics instance | GPU runtime and paired-GT evidence are complete; held-out directional probes have correctly abstained, so transfer and LOBO remain prohibited |
| Counterexample-driven diagnostic-probe evolution | Global scale, temporal mix, directional scale, translation-only scale, action-dimension anisotropy, and off-diagonal action coupling produced retained counterexamples; the fresh dev4 successor passes locality but reverses on accept4 |
| Progressive-fidelity efficiency | Receipt-derived audit complete; current 512-step screen saves only 6.28% projected GPU hours and remains an optimization target |
| Multi-seed causal replication of the bundled effect | Not established |
| Cross-backbone IRG alignment and calibrated transfer | Research work in progress |
| Autonomous atlas evolution validated on new backbones | Research work in progress |

This distinction is intentional: code availability is not treated as empirical
validation.

## Core loop

1. **Compile intent.** A user objective is converted into a versioned goal,
   metric roles, held-out protocol, resource budget, and immutable verifier.
2. **Diagnose.** Verdict probes measure success; diagnostic probes localize
   likely mechanisms without being allowed to decide acceptance.
3. **Compile an intervention.** A semantic primitive is admitted only when the
   target backbone exposes the required hook and all invariants are checked.
4. **Test progressively.** Cheap screens precede the frozen official gate and
   checkpoint/seed confirmation.
5. **Settle evidence.** Every verdict is tied to a receipt. Missing evidence,
   protocol drift, or failed gates yields rejection or abstention.
6. **Retain effects.** Confirmed, null, harmful, and interaction effects remain
   context-local in memory and become transfer priors only under a certificate.
7. **Evolve the atlas.** Nearby contexts with opposing repair effects create
   counterexamples that prioritize new diagnostic directions.

See [Architecture](docs/ARCHITECTURE.md) and
[Method to code](docs/METHOD_TO_CODE.md) for the detailed mapping. To bring up
a different model family, follow the fail-closed
[backbone instantiation guide](docs/BACKBONE_INSTANTIATION.md).

## Quick start

VerdiWM's CPU control plane requires Python 3.10.

```bash
python -m pip install uv
uv sync --all-groups
bash scripts/ci/check_control_plane.sh
```

Validate the checked-in public evidence bundle:

```bash
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
```

Run the IRG tests directly:

```bash
uv run pytest -q tests/test_verdiwm_geometry.py tests/test_acwm_unified_irg_assets.py
```

GPU execution is adapter-specific. ACWM-Phys data and checkpoints are not
redistributed here; follow the upstream projects and then pass their roots to
the corresponding execution command. See
[Reproducibility](docs/REPRODUCIBILITY.md).

## Minimal closed-loop evidence

[`examples/acwm_minimal_loop_cloth_next_forcing_v2`](examples/acwm_minimal_loop_cloth_next_forcing_v2)
contains a diagnosis, an intervention receipt, a 512-step screen, a frozen
official 50-step gate, checkpoint confirmation, a long-horizon effect profile,
an experience-map entry, and a three-column video.

For `cloth_move + next_forcing` it records:

- 512-step screen AUC delta: `+11.9055`
- official 50-step PSNR delta: `+0.79 dB`
- selected checkpoint PSNR delta: `+0.94 dB`
- long-horizon PSNR deltas: `+1.5883`, `+1.3677`, and `+1.0753 dB` at horizons 16, 32, and 48

The initial and confirmation evaluations share evaluation seed `2802`.
Therefore this bundle proves an operational minimal loop and a local paired
effect, not a paper-level replicated causal effect. The upstream cloth
checkpoint also has a published-step metadata inconsistency, so this result is
restricted to paired deltas against the exact frozen checkpoint.

## Public experience snapshot

[`examples/acwm_experience_atlas_v1`](examples/acwm_experience_atlas_v1)
retains the current ACWM-Phys exploration ledger in a path-safe form. It
contains 284 completed screens, 874 per-horizon measurements, 22 deduplicated
experience records from 25 source maps, and four official-gated
`GT | Baseline | Repair` showcase cases. Positive, null, and harmful screens
are all retained so failed interventions are not silently rediscovered.

Screen labels are triage evidence, not paper results. The four showcase cases
carry separate official-gate and confirmation evidence; experience-atlas
records remain context-local routing priors unless causal credit is explicitly
established.

## Method evidence maps

[`examples/acwm_method_evidence_maps_v1`](examples/acwm_method_evidence_maps_v1)
links the current environment-by-primitive official-gate matrix to the
active-r20 environment-by-probe Jacobians, the primitive-to-probe mechanism
contract, and a descriptive IRG projection. Figures are provided as PNG, SVG,
and PDF; source tables are provided as CSV, Markdown, and LaTeX. Crossed probe
cells fail the frozen locality threshold and are excluded from routing. Probe
responses and the PCA are diagnostic pilot evidence, not repair-quality or
cross-backbone transfer claims.

## Unified IRG assets

[`examples/acwm_unified_irg_assets_v1`](examples/acwm_unified_irg_assets_v1)
is the immutable audit snapshot that exposed a mixed-mode protocol error: two
source probes used parallel generation while four used autoregressive
generation. It retains the resulting covariance gaps and abstention decisions.

[`examples/acwm_joint_irg_assets_v2`](examples/acwm_joint_irg_assets_v2) is the
corrected joint-frame bundle. Six source probes, expanded into seven semantic
paths, were rerun in autoregressive mode against one no-hook baseline per
environment and seed. Its 600 measurements include only 24 canonical baselines.
Every asset stores the locality-masked Jacobian `J_X`, repair metric `G_X`,
response coordinate `r_X`, full paired-seed covariance, support mask,
checkpoint identity, and source hashes.

All eight v2 assets are routing-ready and have one observed covariance block.
This closes the within-ACWM cross-path covariance gap. It does not establish a
cross-backbone repair effect: a target still needs a compatible measured chart,
a held-out transfer certificate, and confirmed target effects.

## Ctrl-World target-local chart

[`examples/ctrl_world_target_local_irg_v1`](examples/ctrl_world_target_local_irg_v1)
contains 30 paired predictive-quality measurements over two locality radii on
the frozen three-episode pilot split. The wide action-embedding scale campaign
failed transfer admission with residual `1.2573`. The automatically narrowed
campaign passed with residual `0.3566` at radius `0.025`, so the settlement
selected that chart and preserved the wide failure as negative evidence.

The bundle exports both response curves and the resulting `J_X`, `G_X`, `r_X`,
and covariance diagonal in CSV and JSON form. The independent zero-dose rerun
reproduced PSNR and pixel L1 exactly; the three reward-derived diagnostics had a
maximum absolute drift of `4.68e-4`, within the declared `1e-3` audit tolerance.
This is locality calibration, not evidence that an intervention improves
Ctrl-World or transfers from ACWM-Phys; those claims still require paper-split
selector and effect receipts.

[`examples/ctrl_world_paper_split_abstention_v1`](examples/ctrl_world_paper_split_abstention_v1)
contains the independent `stack` paper-split result. All 15 paired receipts are
complete, but the radius-`0.025` chart has locality residual `1.1063`, above the
frozen `0.5` threshold. The certificate therefore returns
`settled_abstained`. This retained negative-transfer case is evidence that the
admission gate changes behavior; it is not a failed result to hide or relabel.

## Cosmos3 forward-dynamics instance

[`examples/cosmos3_forward_dynamics_instance_v1`](examples/cosmos3_forward_dynamics_instance_v1)
records a path-safe CPU bring-up and one official GPU execution of
Cosmos3-Nano as an ACWM `forward_dynamics` backbone. It validates the official
`16 x 10` DROID action
contract, frozen cookbook-sample identity, first-frame conditioning, H1-H5
anchors, zero-dose byte identity, and one runtime-ready
`action_dimension_balancing` binding. Three more primitives are explicitly
mapped for materialization; the other 13 remain blocked rather than being
declared available by convention.
The GPU runtime summary records the frozen `16 x 10` action window, 95 physical
GPU samples with peak memory of 36,722 MiB, and a decodable 17-frame output.

[`examples/cosmos3_paired_gt_dev_v1`](examples/cosmos3_paired_gt_dev_v1)
adds three self-contained paired-GT dev receipts, an SVG/CSV summary, and
aligned `GT | prediction` videos. Mean future-frame PSNR is `21.2759` dB.
This is baseline predictive-quality evidence only; it does not establish a
primitive benefit or transfer result.

[`examples/cosmos3_target_local_irg_wide_v1`](examples/cosmos3_target_local_irg_wide_v1)
adds the complete 15-cell action-conditioning-scale response chart over three
dev identities and five paired doses. The public bundle includes metric tables,
an SVG chart, and aligned `GT | -0.10 | zero | +0.10` videos. Its frozen
locality residual is `0.9510`, above the `0.5` admission threshold, so the chart
correctly abstains from cross-backbone transfer.

[`examples/cosmos3_target_local_irg_narrow_v1`](examples/cosmos3_target_local_irg_narrow_v1)
contains the completed pre-registered scale follow-up at doses
`[-0.025, -0.0125, 0, 0.0125, 0.025]`. Its residual is `45.6476`, so shrinking
the radius did not produce an identifiable local direction relative to repeat
variation.

Those two counterexamples retire global action scale as the next diagnostic
direction and trigger the mean-preserving `action_embedding_temporal_mix`
successor. The concrete hook replaces each action input by
`a_t + d * (mean_time(a) - a_t)`, preserving every action dimension's temporal
mean and exact zero-dose bytes. The complete successor bundle is
[`examples/cosmos3_target_local_irg_temporal_mix_v1`](examples/cosmos3_target_local_irg_temporal_mix_v1).
It contains 15 paired cells, three response videos, and residual `2.0649` under
the unchanged `0.5` threshold. The settlement is therefore
`settled_abstained`: probe evolution executed end to end, but Cosmos3 transfer
and LOBO remain prohibited. None of these charts is model-improvement evidence.

[`examples/cosmos3_translation_narrow_split_reversal_v2`](examples/cosmos3_translation_narrow_split_reversal_v2)
records the subsequent translation-only probe on a second split frozen before
outcomes were inspected. The dose path changes action columns `0:3` only and
preserves rotation and gripper values exactly. It is locally linear on dev2
with residual `0.1032`, but fails locality on independent accept2 with residual
`2.4468`; its normalized Jacobian alignment error is `1.9998` against the
frozen `0.5` limit. The result is `settled_abstained` and is retained as a
certificate counterexample. It does not license LOBO, cross-backbone transfer,
or a model-improvement claim.

[`examples/cosmos3_action_dimension_anisotropy_counterexample_v3`](examples/cosmos3_action_dimension_anisotropy_counterexample_v3)
records the next counterexample-driven successor on the independently frozen
dev3 split. Its signed, mean-preserving per-dimension contrast doses produce a
locality residual of `0.7386`, above the unchanged `0.5` threshold. The system
therefore settles `settled_abstained` without inspecting accept3. This is
evidence that probe evolution and the pre-acceptance safety gate execute as
specified; it is not model-improvement or cross-backbone-transfer evidence.

[`examples/cosmos3_action_dimension_interaction_split_reversal_v4`](examples/cosmos3_action_dimension_interaction_split_reversal_v4)
records the structurally different off-diagonal successor on fresh dev4 and
accept4 windows. Signed orthogonal rotations couple action dimensions
`(0,3)`, `(1,4)`, `(2,5)`, and `(6,7)` while preserving temporal means,
centered action energy, and dimensions 8/9. Dev4 passes locality with residual
`0.0065`, but accept4 fails at `0.5672` and reverses the dominant response;
the normalized Jacobian alignment error is `1.9998`. The frozen certificate
therefore returns `settled_abstained` and prohibits LOBO. This is diagnostic
split-instability evidence, not a model-improvement result.

[`examples/acwm_eval_seed_replication_v1`](examples/acwm_eval_seed_replication_v1)
adds 12 official 50-step gate receipts over four fixed checkpoint cells.
`cloth_move + self_forcing_finetune` and
`robot_arm + self_forcing_finetune` pass all three evaluation seeds; the
`push_rope` and `pour_water` cells pass two of three and remain seed-sensitive.
This tests frozen-checkpoint evaluation robustness, not independent training
seed replication.

[`examples/acwm_training_seed_replication_cloth_self_forcing_v1`](examples/acwm_training_seed_replication_cloth_self_forcing_v1)
separates repair-training randomness from evaluation randomness for
`cloth_move + self_forcing_finetune`. Three independently fine-tuned 512-step
checkpoints are each evaluated under the same three frozen evaluation seeds.
All nine official four-metric gates pass; mean PSNR delta is `+1.0867` dB
with a range of `+0.69` to `+1.67` dB. This is independent repair-fine-tuning
evidence, not independent base-model pretraining or cross-backbone transfer.

## Progressive-fidelity cost audit

[`examples/acwm_progressive_fidelity_efficiency_v1`](examples/acwm_progressive_fidelity_efficiency_v1)
reuses 21 settled 512-step screen-to-official-gate pairs and six independent
800/1000 confirmation ladders. Against the frozen 512 gate, the cheap screen
has positive recall `0.75` and false-rejection rate `0.20`; screen-to-confirm
Spearman correlation is `0.4058`. The measured-cost confirm-all projection
shows only `6.28%` GPU-hour reduction, demonstrating that the current 512-step
screen is still too expensive. The bundle reports this as an optimization
target, not as a strong efficiency claim.

The instance remains `pilot_draft` and formal launch is false. See
[Cosmos3 forward dynamics](docs/COSMOS3_FORWARD_DYNAMICS.md) for the binding
and promotion sequence.

## Repository layout

```text
wmloop/                    control, diagnosis, execution, verification, archive
wmloop/geometry/           IRG, transfer certificates, effect memory, evolution
wmloop/primitives/         typed and materialized intervention registry
configs/                   versioned goals, probes, schemas, and adapter packets
scripts/export/            evidence, visualization, and release tooling
tests/                     CPU contract and integration harnesses
examples/                  small public evidence bundles
docs/                      architecture and reproducibility notes
ops/                       restricted container runtime assets
```

## Backbone adapters

ACWM-Phys is the reference instance, not the system boundary. A new backbone
must declare capabilities and provide goal, data/split, evaluator, hook,
receipt, and archive adapters. Agent-generated code is staged behind an
intent-to-code contract and frozen regression harness; implementation
convenience is not allowed to silently weaken the requested intervention.

Ctrl-World and Cosmos3 are instantiated here as action-conditioned world
models. Their paper-facing verdict uses paired predictive quality,
action-conditioning consistency, and long-horizon stability. The separately retained action-success
packet is an optional downstream/WAM extension and is not used by the ACWM LOBO
experiment.

## Citation

Paper citation metadata will be added with the public manuscript. Until then,
cite the repository revision and the immutable evidence manifest used in your
experiment.

## License

VerdiWM source code is released under the [Apache License 2.0](LICENSE).
Upstream datasets, checkpoints, model repositories, and generated evidence are
not relicensed by this repository; their original terms still apply.
