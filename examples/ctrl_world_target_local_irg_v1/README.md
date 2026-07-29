# Ctrl-World Target-Local IRG Calibration

This example records a paired-dose ACWM predictive-quality calibration on the frozen Ctrl-World pilot split.
It is a target-local response-chart asset, not evidence of model improvement or completed cross-backbone transfer.

- Settlement: `settled_admitted`
- Selected campaign: `ctrl_world_irg_calibration_narrow_pilot_v1`
- Transfer-eligible chart available: `true`

The wide and narrow candidates preserve `J_X`, the covariance metric, response coordinate, and locality residual.
`tables/dose-response.csv` contains repeat-level aggregate response curves; `tables/chart-summary.csv` is paper-table ready.
