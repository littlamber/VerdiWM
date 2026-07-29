# ACWM Progressive-Fidelity Efficiency

This public snapshot reuses settled ACWM-Phys receipts. It contains 21 paired
512-step screen-to-official-gate candidates and six candidates with independent
800/1000 checkpoint confirmation.

The measured results are:

- positive recall against the frozen 512 gate: `0.75`;
- false-rejection rate: `0.20`;
- screen-to-confirm Spearman correlation: `0.405839725`;
- projected GPU-hour reduction versus confirming every candidate: `0.062751783`.

The confirm-all condition is a measured-cost projection: exact candidate cost
is used where available, then an environment median, then a global median. It
is not presented as an observed full rerun. The modest cost reduction is a
negative system result: the current 512-step screen remains too expensive.

Model-quality claims remain governed by the frozen official gate and
checkpoint-ladder receipts. This bundle is efficiency evidence only.
