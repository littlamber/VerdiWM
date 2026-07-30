# ACWM source-effect audit and repair

This example separates two evidence levels that must not be conflated:

1. A checkpoint group can be positive across the preregistered evaluation
   seeds.
2. An environment remains conservatively mixed when the archive also contains
   negative, conflicting, or evaluation-seed-sensitive checkpoints.

The archive-aware audit deduplicates copied receipt directories by receipt ID
and reports checkpoint-level training-seed effects. It does not modify the
frozen transfer certificate, admit a probe, or establish cross-backbone
transfer.

## Current checkpoint groups

| Environment | Primitive | Eval-stable positive training seeds | Conservative environment class |
|---|---|---|---|
| `cloth_move` | `self_forcing_finetune` | 2805, 4101, 4202, 4303 | `same_protocol_reproduction_conflict` |
| `robot_arm` | `self_forcing_finetune` | 3311 | `eval_seed_sensitive` |

`robot_arm/3311` passed the unchanged official quality gate for all three
preregistered evaluation seeds:

| Eval seed | Delta PSNR | Delta SSIM | Delta MSE | Delta masked-MSE |
|---:|---:|---:|---:|---:|
| 101 | +3.01 | +0.0167 | -0.001091 | -0.007334 |
| 202 | +0.56 | +0.0069 | -0.000216 | -0.000080 |
| 303 | +1.01 | +0.0088 | -0.000359 | -0.001384 |

An independently trained `robot_arm/3322` checkpoint passed the initial
official gate but did not replicate across the preregistered evaluation seeds:

| Eval seed | Gate pass | Delta PSNR | Delta SSIM | Delta MSE | Delta masked-MSE |
|---:|---|---:|---:|---:|---:|
| 101 | yes | +3.09 | +0.0187 | -0.001145 | -0.006997 |
| 202 | no | +0.69 | +0.0088 | -0.000272 | +0.000267 |
| 303 | no | +0.05 | +0.0059 | -0.000011 | +0.002919 |

This counterexample is retained as training-seed sensitivity evidence. It is
not counted as stable-positive source support.

See `source-effect-audit.md` and `tables/environment-audit.csv` for the
archive-wide classification. The two `source-effect-repair-settlement-*`
reports and matching `tables/repair-groups-*` files preserve the preregistered
`robot_arm/3311` and `robot_arm/3322` outcomes. Absolute local paths and raw
checkpoints are intentionally excluded from this public view.
