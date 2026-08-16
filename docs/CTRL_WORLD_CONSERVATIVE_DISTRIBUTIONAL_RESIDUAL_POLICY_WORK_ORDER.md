# Ctrl-World Conservative Distributional Residual Policy

## Status And Boundary

This document freezes a trainable mechanism hypothesis and its falsification
protocol. It is not an implementation, a trained checkpoint, or a promotion
decision. The canonical machine-readable work order is:

```text
configs/primitives/ctrl_world_conservative_distributional_residual_policy_v1.json
```

The previous CCLVR screen is a settled negative reference. Its D4 cell passed
mean L1 and counterfactual policy value, but failed terminal interaction L1,
horizon slope, policy Brier, and harmful-routing gates. Episode `1799` is
forbidden from all future training, labels, calibration, method selection, and
promotion.

## Retrieved Mechanism Evidence

The broad failure-specific atlas is v11; v12 and v13 are targeted,
evidence-bound supplements. All three are shadow-only research artifacts and
contain no execution authority.

| Source | Transferable mechanism | Deliberate boundary |
|---|---|---|
| `1710.10044v1` | Quantile regression represents a return distribution instead of one mean value. | We use supervised paired rollout returns, not a Bellman operator. |
| `1612.01474v3` | Parallel independently initialized predictors provide calibrated predictive uncertainty. | The first screen uses lightweight value heads, not five full U-Nets. |
| `1712.06924v5` | SPIBB falls back to the data-collection baseline under high uncertainty. | Zero correction is our exact baseline; no theorem is claimed for video rollouts. |
| `2006.04779v3` | Conservative value learning limits unsupported overestimation under distribution shift. | Only the anti-overestimation principle transfers; no offline-RL objective is assumed. |
| `2212.13629v1` | Quantile risk control targets high-loss events instead of only expected loss. | It is a development calibration/certification layer, not the trainable core. |
| `2506.07469v1` | Treatment-effect intervals should execute only when the sign is informative. | Same-context, same-seed paired outcomes are required to make intervals useful. |
| `2502.01459v2` | Deferral can be local to a sequence token or suffix. | Ctrl-World uses one decision per interaction boundary. |
| `2407.01392v4` | Per-position and variable-horizon credit avoids a whole-rollout average. | The official backbone remains frozen in the first screen. |

The retrieved evidence supports a structural candidate, not a publication
novelty claim. Missing-source reports are retained in each atlas directory;
v13 currently reports zero missing sources.

## Mechanism

For interaction `i`, arm `a` in `{negative, zero, positive}`, and objective
`m` in `{mean, terminal, slope}`, define the paired benefit target:

```text
Delta[i, a, m] = error[i, zero, m] - error[i, a, m]
```

The local value module has five independently initialized bootstrap heads. Each
head predicts five quantiles of `Delta` for all three arms and all three
objectives. Bootstrap masks are grouped by development context, so three seeds
from one context cannot masquerade as three independent contexts.

At inference, a nonzero arm is eligible only when its calibrated lower bound is
positive for mean benefit and nonnegative for terminal and horizon benefit. The
best eligible arm is selected. Any unsupported or uncertain state selects the
zero arm, which is an exact official-checkpoint identity. There is no coverage
dual and no target execution-rate constraint.

The adapter is trained, not merely routed: eligible nonzero current-adapter
rollouts receive terminal and horizon tail gradients through the multiscale
side residual. Unsupported or harmful arms train the value heads and a
conservative overestimation margin but cannot force a residual update. Paired
arm targets are regenerated at steps `0, 16, 32, 48` so labels do not remain
bound to an obsolete adapter checkpoint.

## Ablation And Gates

The screen keeps the official baseline (`A0`) and failed CCLVR (`D4`) visible.
It then compares point values without coverage (`E1`), one quantile head
(`E2`), bootstrap quantiles with zero fallback (`E3`), risk-gated adapter
updates (`E4`), and the complete refreshed tail-risk mechanism (`E5`). A
context-shuffled bootstrap control (`E6`) tests whether grouped uncertainty is
real.

Every row uses eight ranks, BF16, per-device batch size `2`, global batch `16`,
and 64 optimization steps. A one-step memory smoke must stay below 90 percent
of physical memory on every rank. The baseline CCLVR run used this resource
setting successfully.

Promotion requires every gate: exact zero identity, unchanged frozen tensors,
strict improvement on mean L1, terminal interaction L1, horizon slope, policy
Brier, harmful-routing rate, positive counterfactual policy value, paired seed
wins, and action response. It also requires E5 to beat E3 on both terminal and
horizon fixed-arm quality, proving a trained residual benefit beyond a routing
calibration effect. No minimum coverage is imposed. A 512-step confirmation is
unauthorized until all gates pass on a newly frozen promotion-only episode.

See the JSON work order for hashes, exact thresholds, data-admission counts,
loss terms, and implementation order.
