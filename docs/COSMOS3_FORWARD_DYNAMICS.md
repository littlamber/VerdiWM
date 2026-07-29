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
- `wmloop/primitives/adapters/cosmos3_hooks.py` audits H1-H5, implements a
  reversible action-conditioning dose, a mean-preserving temporal-action-mix
  diagnostic, and bounded per-dimension action balancing with receipts. Both
  diagnostic paths copy the source bytes exactly at zero dose.

The public instance remains `pilot_draft`. A local instance may be
closed-loop wired while still being ineligible for a formal launch.

## CPU smoke

After binding the external paths, run:

```bash
python -m wmloop.control.cosmos3_forward_dynamics_smoke \
  --cosmos3-root /path/to/cosmos/packages/cosmos3 \
  --runtime-python /path/to/cosmos/packages/cosmos3/.venv/bin/python \
  --runner-path scripts/integrations/run_cosmos3_droid_lerobot_fd.py \
  --dataset-root /path/to/droid_lerobot_example \
  --dataset-freeze configs/backbones/cosmos3_droid_lerobot_dataset_freeze_v1.json \
  --split configs/goal/cosmos3_forward_dynamics_split_v1.json \
  --split-name dev \
  --checkpoint-path /path/to/Cosmos3-Nano \
  --config-file /path/to/Cosmos3-Nano/config.local_assets.json \
  --output-root outputs/cosmos3-forward-dynamics-smoke
```

This command validates the frozen dataset, official `16 x 10` action contract,
forward-dynamics mode, first-frame conditioning, H1-H5 anchors, zero-dose byte
identity, and action-dimension-balancing materialization. It deliberately uses
`--skip-run`: no prediction is generated and no model-quality claim follows.

## GPU runtime receipt

The reference instance has also completed one official GPU execution on the
frozen dev window. `wmloop.control.cosmos3_gpu_runtime_receipt` fails closed
unless the official runner reports success, the output video is decodable, the
action tensor has a consistent shape, and a physical GPU UUID has nonzero
memory samples during the run. The path-safe public summary records 95 active
samples, peak memory of 36,722 MiB, a 25.09-second generation call, and a
17-frame H.264 output. Raw checkpoints and machine-local paths are excluded.

That receipt advances runtime readiness only. Separate paired-GT dev receipts
and target-local charts are now available, but the one-run bring-up artifact is
still not itself evidence of predictive quality, primitive benefit, or
transfer.

## Promotion boundary

The next evidence levels are separate:

1. paired GT predictive receipts on dev windows;
2. a target-local dose chart and locality check;
3. independent accept-window settlement;
4. held-out selector and effect confirmation.

Failure at any level yields rejection or abstention. It must not be relabeled
as transfer success.

## Counterexample-driven probe evolution

The wide and narrow `action_conditioning_scale` charts failed the same frozen
locality threshold. The successor campaign at
`configs/experiments/cosmos3_irg_calibration_temporal_mix_dev_v1.json` changes
only the diagnostic direction: for every action dimension it applies
`a_t + d * (mean_time(a) - a_t)`. This transformation preserves the temporal
mean, is reversible over the admitted dose range, and leaves the checkpoint,
split, seeds, evaluator, verdict outcomes, and zero-dose input unchanged.

`scripts/run_cosmos3_fingerprint_campaign.py` passes the campaign probe ID to
both the upstream runner and paired-GT evaluator. A mismatch between the
configured probe and the receipt fails closed. This successor remains
ineligible for transfer until a complete chart passes the pre-registered
locality gate.

The complete dev settlement contains 15 paired cells over three frozen windows
and five doses. Every admitted receipt has a clean physical-GPU exclusivity
audit. The successor residual is `2.0649139795`, above the unchanged `0.5`
threshold, so the result is `settled_abstained`. The wide (`0.9510`), narrow
(`45.6476`), and temporal-mix (`2.0649`) failures remain linked as one
counterexample lineage. Cosmos3 LOBO is therefore blocked; changing the radius
or threshold after observing these results would require a new campaign.
