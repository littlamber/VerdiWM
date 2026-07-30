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
official-gate signs: six collisions and two non-collisions. The latest
chronological evolved replay cumulatively composes existing successor probes
and adds an eight-environment, 120-measurement horizon-weighted teacher-recovery
campaign. The new axis passes locality in 7/8 environments, but is redundant
with uniform teacher recovery on the two prespecified collision environments.

The evolved safety alert has F1 `0.8571`, but it still accepts zero folds.
All three pre-certificate comparable choices are collisions, including a new
`pour_water` regression. Therefore the post-evolution collision rate is
undefined, not zero, and S4 remains partial.

## Claim Boundary

This is diagnostic-system evidence on frozen historical ACWM-Phys receipts. It
does not establish model improvement, selector superiority, or cross-backbone
transfer. The alert F1 is not a success claim because it is achieved with zero
accepted coverage. The horizon-recovery campaign is retained as a negative
probe-evolution result, not as evidence of improvement.
