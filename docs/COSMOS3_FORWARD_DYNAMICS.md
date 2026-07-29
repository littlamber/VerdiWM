# Cosmos3 Forward-Dynamics Instance

This packet instantiates Cosmos3-Nano as an action-conditioned world model,
not as a policy or WAM. The model consumes one conditioning frame and a DROID
action chunk and predicts future video. Downstream task success and policy
action accuracy are forbidden verdict sources for this instance.

## Included contracts

- `configs/goal/cosmos3_forward_dynamics_predictive_pilot_v1.yaml` declares
  paired predictive metrics and a fixed 16-frame horizon.
- `configs/goal/cosmos3_forward_dynamics_split_v1.json` freezes separate dev
  and accept sample starts and seeds.
- `configs/backbones/cosmos3_droid_lerobot_dataset_freeze_v1.json` records every
  official cookbook-sample file by SHA-256 and size.
- `configs/probes/cosmos3_acwm_forward_dynamics_v1.json` separates four verdict
  probes from action-sensitivity and autoregressive-drift diagnostics.
- `wmloop/evaluate/adapters/cosmos3_predictive.py` rejects policy-mode,
  downstream-success, non-finite, or out-of-split receipts.
- `wmloop/evaluate/cosmos3_paired_gt.py` enforces 17-frame alignment, excludes
  the conditioning frame from verdict metrics, and checks the official
  top-left content crop before measuring future-frame error.
- `wmloop/primitives/adapters/cosmos3_hooks.py` audits H1-H5, implements a
  reversible action-conditioning dose, and materializes bounded per-dimension
  action balancing with a receipt.

The public instance remains `pilot_draft`. A local instance may be
closed-loop wired while still being ineligible for a formal launch.

## CPU smoke

After binding the external paths, run:

```bash
python -m wmloop.control.cosmos3_forward_dynamics_smoke \
  --cosmos3-root /path/to/cosmos/packages/cosmos3 \
  --runtime-python /path/to/cosmos/packages/cosmos3/.venv/bin/python \
  --runner-path /path/to/cosmos/packages/cosmos3/tools/run_official_droid_lerobot_fd.py \
  --dataset-root /path/to/droid_lerobot_example \
  --dataset-freeze configs/backbones/cosmos3_droid_lerobot_dataset_freeze_v1.json \
  --split configs/goal/cosmos3_forward_dynamics_split_v1.json \
  --split-name dev \
  --checkpoint-path /path/to/Cosmos3-Nano \
  --config-file /path/to/Cosmos3-Nano/config.local_all.json \
  --output-root outputs/cosmos3-forward-dynamics-smoke
```

This command validates the frozen dataset, official `16 x 10` action contract,
forward-dynamics mode, first-frame conditioning, H1-H5 anchors, zero-dose byte
identity, and action-dimension-balancing materialization. It deliberately uses
`--skip-run`: no prediction is generated and no model-quality claim follows.

## Paired-GT dev baseline

[`examples/cosmos3_paired_gt_dev_v1`](../examples/cosmos3_paired_gt_dev_v1)
contains complete self-contained receipts for the three frozen dev windows,
plus CSV, SVG, and aligned `GT | prediction` videos. Mean future-frame metrics
are PSNR `21.2759` dB, L1 `0.04335`, final-frame MAE `0.06272`, and temporal
difference MAE `0.02310`. These are baseline quality measurements, not a
primitive benefit or transfer result.

## Promotion boundary

The next evidence levels are separate:

1. real GPU inference smoke with the frozen checkpoint and physical GPU receipt;
2. paired GT predictive receipts on dev windows (complete);
3. a target-local dose chart and locality check;
4. independent accept-window settlement;
5. held-out selector and effect confirmation.

Failure at any level yields rejection or abstention. It must not be relabeled
as transfer success.
