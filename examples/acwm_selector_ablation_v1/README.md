# VerdiWM ACWM Selector Ablation Evidence

The 8-environment evidence matrix is complete. Ranking and sign metrics are public evidence; top-1 selector superiority is not claimed when selection_discrimination_ready is false.

- Environments: `8`
- Evaluated replay cells: `96`
- Formal comparison ready: `true`
- Selection discrimination ready: `false`

| Selector | Top-1 positive | Negative selection | Sign accuracy | Kendall tau | Regret |
|---|---:|---:|---:|---:|---:|
| environment_label | 0.3750 | 0.6250 | 0.4792 | 0.4167 | 0.1250 |
| static_probe | 0.3750 | 0.6250 | 0.4167 | 0.4167 | 0.1250 |
| raw_response | 0.3750 | 0.6250 | 0.4792 | 0.4167 | 0.1250 |
| irg | 0.3750 | 0.6250 | 0.5417 | 0.4167 | 0.1250 |
