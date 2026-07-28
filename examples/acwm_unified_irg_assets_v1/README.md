# ACWM-Phys Unified IRG Assets

This bundle materializes one canonical IRG asset for each ACWM-Phys environment.
Every asset stores raw and locality-masked `J_X`, `G_X`, and `r_X`, paired-seed
response covariance, support masks, and source hashes.

- Environments: 8
- Probe paths: 7
- Routing-ready environments: 8
- Transfer-abstaining environments: 8

The current source campaigns share outcomes, weights, seeds, and checkpoints,
but use more than one zero-dose baseline frame. Covariance is measured only
within compatible groups; cross-group entries are explicitly unobserved and
zero-filled. These assets support audited routing, not a cross-backbone transfer
claim.
