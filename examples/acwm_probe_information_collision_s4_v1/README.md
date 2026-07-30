# ACWM-Phys S4 Probe Information and Collision

This path-free snapshot contains the three frozen S4 conditions:

- fixed minimal probe set;
- preregistered deterministic random probe expansion;
- latest completed counterexample-gated probe evolution.

The random condition runs 80 CPU-only replays over 16 randomization seeds and
probe-set sizes 1, 2, 4, 6, and 8. The eight-path condition has selection
regret `0.125` and Gram condition number `431.0082`; one-path subsets have
lower average sign error on this small label set but much worse regret and a
mean Gram condition number of `79731.1816`.

Eight fixed-selector top-1 cases were labeled using target-local settled
official-gate signs: six collisions and two non-collisions. Counterexample
evolution then tested four chronological successors without changing the
certificate. Horizon-weighted teacher recovery produced 120 measurements and
passed locality in 7/8 environments, but was redundant with uniform teacher
recovery. Action-dimension interaction was non-redundant but failed locality in
both requested targets, so it was not expanded. Motion-event phase curvature
passed its frozen smoke gate and produced a complete 120-measurement atlas with
7/8 local environments; it recovered the `stack_cube` motion-region coverage
gap but not `cloth_move`.

Stage 5 tested a quadratic history-age recovery successor under a protocol-
matched 10-step pilot. The `reacher` response was non-redundant but failed
locality (`4.6426`); the locality-admitted `push_cube` response (`0.0358`) was
redundant with linear horizon recovery (cosine `0.999999997`, relative L2
`0.0221`). The frozen rule therefore rejected eight-environment expansion.

Stage 6 then materialized `multi_chunk_exposure_stability` for the supervised
prefix-window proxy actually implemented by `next_forcing`; it does not claim
generated-rollout training. The protocol-matched gate admitted `cloth_move`
(locality `0.0744`, cosine `0.99953`, relative L2 `0.14778`) as local and
non-redundant against latest-state feedback, while `reacher` failed locality
(`0.7691`). The resulting eight-environment atlas has 120 complete measurements
and 7/8 local environments.

Stage 7 materialized `action_temporal_alignment_phase` for the actual
`inv_dyn_reward_finetune` hook, which regresses time-aligned actions from
adjacent latent deltas. Its first 15 measurements are retained as a protocol
failure only: the candidate dose grid did not match the reference grid, so no
effect value entered a decision. Stage 8 froze the matched grid before rerun.
On `push_sand`, the matched phase response is local (`0.3352`) and
non-redundant with action-event alignment (cosine `-0.9999999`, relative L2
`1.6160`). The expanded atlas is complete for all eight environments and local
in 7/8; `push_rope` is excluded as nonlocal (`1.9803`).

The latest r29 replay removes the `push_sand` inverse-dynamics work order and
reduces unresolved probe work orders from five to four, but still accepts zero
folds. Twelve IRG cells are stopped by the unchanged
transfer certificate and twelve by incomplete target-local probe coverage. The
remaining `push_sand` fold is blocked by the nonlocal
`action_dimension_anisotropy` path. The
safety alert remains F1 `0.8571`; all four pre-certificate comparable choices
are collisions, including the existing `pour_water` regression. Therefore the
post-evolution collision rate is undefined, not zero, and S4 remains partial.

## Claim Boundary

This is diagnostic-system evidence on frozen historical ACWM-Phys receipts. It
does not establish model improvement, selector superiority, or cross-backbone
transfer. The alert F1 is not a success claim because it is achieved with zero
accepted coverage. The probe campaigns demonstrate fail-closed evolution:
redundant or nonlocal successors are retained as counterexamples, and local
coverage alone does not override the transfer certificate.
