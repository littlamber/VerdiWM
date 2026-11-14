# Agibot Engineering Diagnostic Quickstart

The built-in `wan22-agibot-engineering-diagnostic-v1` profile checks the
training, checkpoint saving, independent reload, validation, budget settlement,
and runtime receipt/CAS retention for Wan 2.2 5B initialization weights on
Agibot arm-slot data. This entrypoint does not write an RSI settled-trial
Archive record or model-effect memory. It runs at most eight planned optimizer
updates on a train/validation split and cannot establish model improvement or
formal promotion.

Run the following commands from the VerdiWM repository root. Install the
declared control-plane dependencies and verify the CLI first:

```bash
python -m pip install uv==0.12.10
uv sync --locked
uv run verdi doctor
```

The VerdiWM control plane requires Python 3.12. The training Python is a
separate environment that you provide, with PyTorch and CUDA compatible with
the native training code. `verdi doctor` checks only the control plane; it does
not check PyTorch or CUDA in the training environment. We recommend selecting
the interpreter explicitly and checking it before launch:

```bash
export VERDIWM_AGIBOT_PYTHON=/path/to/training-python
"$VERDIWM_AGIBOT_PYTHON" -c '
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
'
```

`CUDA available` should be `True`. If `VERDIWM_AGIBOT_PYTHON` is unset, the
profile searches for `.venv/bin/python` in the source directory and then the
model directory; confirm that the selected interpreter meets these requirements.

## Prepare the inputs

Provide these five inputs on your machine:

1. The Wan 2.2 Diffusers base directory, including `config.json`, model weight
   files, and `Wan2.2_VAE.pth` (or provide the VAE separately).
2. The target Wan 2.2 5B action-checkpoint directory, including
   `dit_model.safetensors` and `meta.json`. Verdi passes this directory to the
   native trainer as its warm-start input.
3. The native EtherWM source directory, including
   `scripts/training/train_world_model.py` and the required `sa_wm` files.
4. The Agibot data root, including `manifest.json`, `train.index.jsonl`,
   `val.index.jsonl`, `stats/`, and `shards/`.
5. A native YAML configuration compatible with the model's training code.

Pass local paths through environment variables. The commands and repository
profile do not contain paths tied to a developer machine:

```bash
export WAN5B_CHECKPOINT=/path/to/wan-2.2-5b-action-checkpoint
export VERDIWM_AGIBOT_BASE_MODEL=/path/to/wan-2.2-diffusers-base
export ETHERWM_SOURCE=/path/to/etherwm-source
export AGIBOT_DATA=/path/to/agibot-data
export AGIBOT_TRAINING_CONFIG=/path/to/training_config.yaml
# Keep campaign state and outputs outside the repository checkout.
export VERDIWM_STATE_ROOT="$HOME/.local/state/verdiwm/agibot-state"
export VERDIWM_CAMPAIGN_ID="agibot-wan5b-$(date -u +%Y%m%d-%H%M%S)"
```

The VAE defaults to `VERDIWM_AGIBOT_BASE_MODEL/Wan2.2_VAE.pth`; set
`VERDIWM_AGIBOT_VAE_CHECKPOINT` when it is stored separately. Before GPU admission,
Verdi checks the Diffusers config, weight shards, VAE file, and required warm-start
metadata. Before taking a GPU lease, the selected training interpreter also reads
the native YAML, constructs the native `WMTrainer` DiT architecture on meta
tensors, and compares its keys and shapes with the warm-start safetensors
header. It does not read full weight tensors or use the GPU. This is a
pre-admission check; the native strict checkpoint loader still validates and
decides whether to accept the weights when training starts. When the checkpoint
restores the `history_reference` branch, this preflight also scans its tensors
on CPU in bounded chunks. It rejects non-finite values anywhere in that branch
and all-zero encoder/QKV weights, using the same dead-weight rules as the native
training restore check; zero biases and zero output-projection weights remain
valid. A checkpoint with the entire branch absent is an allowed warm-start
case, while a partially restored branch is rejected. The scan does not load the
full checkpoint into memory.

## Start the diagnostic

```bash
uv run verdi run \
  --model "$WAN5B_CHECKPOINT" \
  --source "$ETHERWM_SOURCE" \
  --dataset-root "$AGIBOT_DATA" \
  --training-config "$AGIBOT_TRAINING_CONFIG" \
  --world-size 2 \
  --goal "Diagnose the Agibot arm-slot training and validation path" \
  --budget 2gpu-hour \
  --campaign-id "$VERDIWM_CAMPAIGN_ID" \
  --state-root "$VERDIWM_STATE_ROOT"
```

When `--adapter-profile` is omitted, `--training-config` selects the built-in
Agibot v3 profile. If you specify `--adapter-profile`, Verdi uses that profile.
The tested loader settings are 5 history frames, 3 prediction frames, an 8-frame
sampling stride, sparse history, dilation `[1, 2]`, a 0.15 history-collapse
probability, validation dilation `2`, disabled generated history,
`arm_slot_eef_pose`, and body conditioning. The validation panel is fixed at 128
samples. If your YAML uses another loader, copy the profile, update
`agibot_engineering_diagnostic.support_loader`, and pass the copy with
`--adapter-profile`. Do not change only the YAML while keeping the old frozen
data-support contract.

Verdi derives a campaign-private configuration from the local YAML and binds
`world_model.checkpoint` to the Diffusers base and `world_model.vae_checkpoint` to
the VAE; it passes the target action checkpoint separately through `--init_from`.
The derived config also binds the dataset root and output directory without
modifying the original YAML. The built-in profile caps training at 40
minutes and checkpoint validation at 20 minutes. Their sum is multiplied by
the planned GPU count and reserved as the ledger's worst-case budget. The
native configuration that enables FSDP must use at least two GPUs, declared
with `--world-size`; the example uses two GPUs and `--budget 2gpu-hour`. The
reserved budget scales with GPU count, so add budget when using more GPUs or
longer stage limits. If the frozen estimate exceeds the declared budget, Verdi
blocks the campaign before enqueueing and reports both values. A copied profile can change the limits under
`agibot_engineering_diagnostic.stage_timeouts_seconds`. Before launching GPU
work, Verdi checks the data indexes, train/validation episode separation,
shard references, and native source binding. A failed binding stops the run
with an error code.

`--world-size` applies only to the auto-generated Agibot scale plan and must
be used with `--training-config`. If you provide `--training-scale-plan`, its
parallelism is authoritative and `--world-size` cannot be combined with it.

If preflight or the native trainer reports that the restored
`history_reference` encoder/QKV weights are non-finite or dead, the checkpoint
is not usable for this run. The failed diagnostic's native trainer recommended
starting from the H1 `checkpoint-134750` checkpoint. Point `WAN5B_CHECKPOINT`
at that checkpoint directory only after it passes the GPU-free preflight,
assign a new campaign ID, and rerun the command above:

```bash
export WAN5B_CHECKPOINT=/path/to/etherwm/checkpoint-134750
export VERDIWM_CAMPAIGN_ID="agibot-wan5b-retry-$(date -u +%Y%m%d-%H%M%S)"
# Run the same `uv run verdi run` command shown above.
```

Keep the failed campaign, its ID, and its receipts for traceability; do not
reuse the old ID or delete its evidence. If the recommended checkpoint also
fails preflight, stop and investigate its source or use another checkpoint
whose architecture and restored history-reference weights pass. For other
configuration or asset failures, correct the local YAML or model assets and
also use a new campaign ID. See the [run lifecycle guide](RUN_LIFECYCLE.md)
for failure recovery, status, and cancellation behavior.

The training process's working directory, `HOME`, temporary directory, and
common model caches are placed under this campaign's output directory. Its
environment preserves the selected Conda, PyTorch, CUDA, and NCCL runtime
variables while removing model-service credentials and proxy settings from the
shell environment; W&B, SwanLab, Hugging Face downloads, and telemetry are
disabled. The manifest's `network_required: false` means this diagnostic does
not need network access. The current local-process execution does not provide
an OS-level network or filesystem sandbox, however, so it cannot prevent
native training code from making network requests or writing through absolute
paths elsewhere that the user can access. Run this diagnostic only with
trusted native source. Use an external container or equivalent OS sandbox if
untrusted code must be isolated.

## Find the result

Use the same state root and campaign ID to check progress or the final status:

```bash
uv run verdi status "$VERDIWM_CAMPAIGN_ID" --state-root "$VERDIWM_STATE_ROOT"
```

The run artifacts are under
`$VERDIWM_STATE_ROOT/runs/$VERDIWM_CAMPAIGN_ID`. List the manifest, stage
receipts, checkpoint, and trace files with:

```bash
find "$VERDIWM_STATE_ROOT/runs/$VERDIWM_CAMPAIGN_ID" -maxdepth 4 -type f -print
```

Run manifests and receipts may record local absolute paths for the model, data,
source, Python interpreter, and run directory. These are useful for local
traceability, but inspect and remove private paths before public sharing. Do
not upload the entire state root, run directory, or raw receipts to GitHub or
a public bundle as-is.

Review the terminal output, campaign record from `verdi status`, and the run's
`model-run.json`, `execution-receipt.json`, `receipts/`, and Agibot trace. A
`completed` status plus valid checkpoint and diagnostic receipt bindings means
the engineering path closed successfully. This profile only supports
engineering diagnostics; it does not show that the model improved over its
initial weights.

The frozen primary-effect target is at least a 2 dB PSNR increase. Use the same
held-out samples and rollout/decode settings for baseline and candidate, run at
least three paired seeds, average validation samples within each seed, and
calculate a two-sided 95% Student-t confidence interval over the paired
candidate-minus-baseline seed differences. The target passes only when the
interval's lower bound is strictly greater than 2 dB. SSIM is reported only;
LPIPS is not measured. The current engineering profile does not connect paired
rollouts to a formal verdict, settlement, or memory update, so its successful
completion cannot pass this PSNR target or establish reusable effect memory.
