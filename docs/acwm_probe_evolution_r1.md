# ACWM-Phys Probe Evolution Audit

This audit records an exploratory eight-environment calibration cycle. It does
not claim selector superiority or model-quality improvement.

## Admitted Diagnostic Axes

All campaigns use five paired doses, three fixed seeds, the same checkpoint and
input trajectory within each pair, and a frozen locality residual threshold of
`0.5`.

| Probe axis | Local environments | Coverage |
|---|---|---:|
| `action_embedding_temporal_mix` | push_cube, stack_cube, cloth_move, pour_water, robot_arm | 5/8 |
| `motion_region_scale` | push_cube, push_rope, cloth_move, push_sand, pour_water, robot_arm | 6/8 |
| `self_rollout_temporal_mix` | push_cube, stack_cube, cloth_move, push_sand, robot_arm | 5/8 |

The earlier `motion_history_scale` axis was rejected as an independent
coordinate because constant scaling of cumulative temporal increments is
algebraically identical to scaling history deviation around the first latent.
Its raw measurements remain audit evidence but do not support primitive
affinity or selector claims.

## Transfer Certificate

IRG candidate selection is fail-closed when any candidate lacks its required
local mechanism path. A selected candidate additionally requires:

- at least two non-leaking source environments with settled effect labels;
- distance-weighted positive probability of at least `0.75`;
- no use of the held-out target effect label in the certificate.

The certificate rejected a single-source selector divergence that chose
`motion_region_reweight` for `robot_arm`. The settled held-out label later shows
that this rejection prevented a negative transfer. No 512-step confirmation was
launched from that rejected choice.

## Claim Boundary

Probe evolution increased mechanism-path coverage and produced auditable
abstentions. The current replay is still partial, and no positive IRG selector
superiority claim is made. Remaining work requires additional non-leaking
settled effect labels and independently frozen distance-support calibration.
