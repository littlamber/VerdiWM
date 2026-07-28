# Pour Water + Inverse-Dynamics Reward Fine-Tuning

This case passed the frozen official ACWM-Phys 50-step quality gate at the
retained 512-step checkpoint:

- PSNR: `+1.36 dB`
- SSIM: `+0.0063`
- MSE: `-0.000307`
- masked-MSE: `-0.002532`

`paired_gt_baseline_ours.mp4` is laid out as `GT | Baseline | Ours`. It is the
baseline-blind hardest sample among the three exported held-out videos, selected
by descending baseline RGB video MSE against GT. Candidate quality was not used
for sample selection.

The aggregate official gate is the claim-governing evidence. Human inspection
found the qualitative difference real but subtle, so this video is a
quantitative fallback example rather than a visually strong project-page case.
This single retained-checkpoint result does not establish cross-seed,
long-training, or cross-backbone robustness.
