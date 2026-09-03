# WAN2.2-DROID ACWM v1

This is the executable contract for **first frame + DROID action/proprio
sequence -> 30 second future video**.  It is intentionally model-decoupled:
the WAN2.2 weights, source checkout, adapter implementation, and WorldArena
evaluator are external immutable inputs.

The package owns manifest generation and admission checks.  It does not claim
that the base `Wan2.2-TI2V-5B` checkpoint is action-conditioned, and it never
silently imports the Wan2.1 `embodydrive-srr` adapter.  A GPU run is admitted
only after the manifests and an explicit WAN2.2 adapter command are bound.

Example:

```bash
python experiments/wan22_droid_acwm_v1/run.py prepare \
  --data-root /share/project/zhiwei/wjy/data/droid_wrist_192x320 \
  --output-root /share/project/zhiwei/nsq/wan22-droid-manifests

python experiments/wan22_droid_acwm_v1/run.py conformance \
  --train-manifest /path/train.json --validation-manifest /path/val.json \
  --model /share/project/zhiwei/nsq/model/Wan2.2-TI2V-5B \
  --source /path/to/Wan2.2-source \
  --evaluator-contract configs/evaluators/wan22_droid_worldarena_v1.json \
  --adapter experiments/wan22_droid_acwm_v1/wan22_droid_adapter.py
```

The resulting report is a readiness receipt, not a quality result. Quality
requires generated 150-frame rollouts and the frozen WorldArena verifier. The
external runner emits `generated_150f.mp4`, `ground_truth_150f.mp4`,
`droid_conditioning.npz`, `worldarena_summary.json`, and an evaluator receipt;
these are the stable hand-off contract for a separately installed WorldArena
environment.

The external runner exposes the scale-plan arms through `--conditioning-mode`
(`visual_anchor_only`, `action`, `action_proprio`, `action_proprio_history`, or
`action_proprio_ema`) and binds randomness through `--seed`. The EMA arm is a
causal exponential state over action and proprioception; `--history-decay`
controls the retained-history factor (default `0.8`, with larger values retaining
older state longer). The train
manifest is used only for adapter optimization; the validation manifest is
required for every published rollout and is checked for episode disjointness.
Formal long runs resolve a frozen validation panel before loading the GPU: pass
`--validation-sample-indices i j k` (or let the runner select the first three
episode-diverse records). Every seed is evaluated on the same panel; a single
held-out clip is diagnostic only and cannot support a formal idea decision.
The runner distinguishes `--training-mode probe` (one train window) from
`--training-mode long` (iterates over train-manifest windows, optionally bounded
by `--train-record-limit`; use `0` for all records). Long mode defaults to the
deterministic `episode_balanced` sampler: it shuffles episodes by seed, takes one
window per episode before taking a second, and rotates the 45-frame offset inside
each 150-frame record. `sequential` remains available as an explicit control.
Receipts include actual window, episode, chunk-offset, and optimization-update
coverage; `training_progress.json` is updated atomically during a run.
Rollout anchoring is an orthogonal policy selected by `--anchor-policy`.
`initial_reference_blend` causally tethers each later chunk boundary to the
first observed latent with `--anchor-refresh-strength`; it never reads a future
validation frame.
Sampling diversity is independently controlled with `--branch-count` and
`--branch-selection`. `terminal_reference_consistency` selects among sampled
chunks using only the first observed latent and the preceding generated latent;
the per-branch scores and selected index are retained in each chunk receipt.

The full gate is available as `closed-loop`. It executes rather than merely
emitting a launch plan: the default is 512 global optimization updates over
up to 256 train windows for each of three seeds, followed by a 150-frame
rollout on the frozen validation panel, four valid WorldArena dimensions, and
multi-seed evidence verification. It
requires explicit runner, adapter, evaluator, asset, GPU, and `--execute`
bindings; without them it returns a blocked receipt and spends zero GPU hours.

Every successful runner invocation also writes
`acwm-gt-visualization/manifest.json`, a side-by-side
`acwm_gt_comparison.mp4`, and an `acwm_gt_contact_sheet.png`. These are
inspection artifacts only; they do not replace the frozen metrics.

The generic control plane can materialize the complete upgrade path before a
model is launched:

```bash
verdiwm plan-training-ladder \
  --train-manifest /path/train.json --val-manifest /path/val.json \
  --current-stage probe --target-stage confirm \
  --output /path/idea/training-stage-ladder.json
```

`probe` is runtime-only, `screen` is diagnostic, and `pilot`/`confirm` are
formal training stages. A screen failure does not veto the formal path, but a
missing episode identity, train/validation overlap, or insufficient formal
coverage blocks execution. One idea should own one output root; seeds,
attempts, and WorldArena staging belong below that root rather than becoming
top-level idea names.

```bash
python experiments/wan22_droid_acwm_v1/run.py closed-loop \
  --train-manifest /path/train.json --validation-manifest /path/val.json \
  --model /path/Wan2.2-TI2V-5B --source /path/WorldArena/embodied_task \
  --adapter experiments/wan22_droid_acwm_v1/wan22_droid_adapter.py \
  --evaluator-contract configs/evaluators/wan22_droid_worldarena_v1.json \
  --runtime-python /path/to/python --runner /path/to/wan22_droid_runner.py \
  --output-root /path/to/run --cuda-visible-devices 0 \
  --worldarena-root /path/to/WorldArena \
  --worldarena-config-template configs/evaluators/wan22_droid_worldarena_config_template.yaml \
  --worldarena-asset-manifest configs/evaluators/worldarena_assets_v1.json \
  --worldarena-asset-root /path/to/worldarena/assets \
  --prior-budget-receipt /path/to/current-budget-receipt.json --execute
```

The closed-loop receipt uses `admitted`, `running`, `completed`, `failed`, or
`blocked`; it never uses `ready_to_launch` as a substitute for execution.
`trajectory_accuracy` and `action_following` are excluded from the default
single-trajectory evaluation because the required policy/multi-GID inputs are
not present, and are never synthesized from visual metrics.
