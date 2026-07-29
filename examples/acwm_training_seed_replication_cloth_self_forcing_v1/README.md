# ACWM repair-training seed replication

Target: `cloth_move + self_forcing_finetune`
Official gate pass rate: `9/9`

This bundle separates three independent repair-training seeds from three
shared evaluation seeds. The candidate checkpoint SHA is retained per row in
`factorial-cells.csv`; aggregate statistics and the two-way variance
decomposition are in `summary.json`. Absolute local paths are intentionally
excluded from this public view.

Raw gate receipts, logs, triptych videos, screen artifacts, and checkpoint
identity manifests are archived in ModelScope repository
`littlamberr/VerdiWM-Evidence` under
`runs/verdiwm_acwm_cloth_trainseed_replication_20260729_r1`.

The claim is limited to independent repair fine-tuning on this ACWM-Phys cell.
It is not independent base-model pretraining or cross-backbone transfer.

| Training seed | Eval seed | Pass | Delta PSNR | Delta SSIM | Delta MSE | Delta masked-MSE |
|---:|---:|---|---:|---:|---:|---:|
| 4101 | 1101 | True | +1.0600 | +0.007700 | -0.003314 | -0.016618 |
| 4101 | 2202 | True | +0.8200 | +0.004300 | -0.002226 | -0.012786 |
| 4101 | 3303 | True | +1.0300 | +0.007800 | -0.002835 | -0.011984 |
| 4202 | 1101 | True | +1.6700 | +0.011700 | -0.004817 | -0.022875 |
| 4202 | 2202 | True | +1.3400 | +0.006700 | -0.003602 | -0.018206 |
| 4202 | 3303 | True | +1.1200 | +0.008400 | -0.002985 | -0.012454 |
| 4303 | 1101 | True | +1.1800 | +0.008300 | -0.003579 | -0.013329 |
| 4303 | 2202 | True | +0.8700 | +0.002600 | -0.002424 | -0.010725 |
| 4303 | 3303 | True | +0.6900 | +0.004800 | -0.002100 | -0.005187 |
