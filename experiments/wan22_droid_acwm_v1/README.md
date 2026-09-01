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
  --source /share/project/zhiwei/nsq/WorldArena/embodied_task \
  --evaluator-contract configs/evaluators/wan22_droid_worldarena_v1.json
```

The resulting report is a readiness receipt, not a quality result.  Quality
requires generated 150-frame rollouts and the frozen WorldArena verifier.
