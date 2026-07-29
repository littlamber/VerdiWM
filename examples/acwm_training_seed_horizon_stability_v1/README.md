# ACWM training-seed horizon stability

Selected checkpoints are evaluated on the same three held-out trajectories at 16/32/48 frames.

- Target: `cloth_move + self_forcing_finetune`
- Verdict: `training_seed_sensitive_long_horizon_effect`
- Max-horizon strict passes: `2/3`

The videos use the fixed `GT|Baseline|Ours` layout. Numeric claims are governed by
`summary.json` and the per-seed profiles, not by cherry-picked video appearance.

## Claim boundary

Paired autoregressive long-horizon evidence across independent repair-training seeds. It updates scoped routing priors but does not establish cross-backbone transfer or causal credit.
