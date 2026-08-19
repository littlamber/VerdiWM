# Experiment Engineering

VerdiWM separates a scientific hypothesis from the repository that executes
it. New experiments should keep a small owned directory with:

```text
experiment-id/
  experiment.json       # engineering manifest
  README.md             # scope, ownership, and failure modes
  run.py                # one reproducible entrypoint
  test_run.py           # CPU smoke and contract tests
  scale-plan.json       # generated training-scale receipt
  outputs/              # never used as source code
```

The engineering manifest is intentionally narrower than a paper protocol. It
records the source entrypoint, test path, Git revision, dataset freeze and
held-out split, evaluation horizons/seeds, required artifacts, and an offline
reproduction command. `verdiwm lint-experiment` checks these fields and the
owned source surface. A dirty checkout is rejected unless the manifest opts in
to `allow_with_receipt`.

## Runtime boundary

The campaign runtime does not contain legacy/modern protocol branches. It
orchestrates an execution contract and records lifecycle receipts. Repository
lint, scale planning, proposal compilation, and verification remain versioned
capabilities that validate their own inputs and outputs before composition.
Obsolete launch implementations are removed from the capability surface rather
than carried as compatibility modes; historical results belong in Archive/CAS,
not in runtime control flow.

## Training scale

Do not choose epochs in a shell script. Generate a scale receipt from the
frozen sample manifests:

```bash
verdiwm plan-training \
  --train-manifest /path/to/train_sample.json \
  --val-manifest /path/to/val_sample.json \
  --stage screen \
  --batch-size 1 \
  --output scale-plan.json
```

The planner derives selected sample count, episode diversity, effective batch
size, steps per epoch, planned updates, and checkpoint evaluation steps. The
stage policy is deliberately progressive: smoke, screen, pilot, and confirm
use increasing data fractions and update caps. Pilot and confirm require
multiple independent episodes; repeated windows from one episode cannot be
used as dataset-level evidence. Missing validation or insufficient episode
diversity blocks admission instead of silently increasing epochs.

Every checkpoint in the plan is evaluated on held-out rollouts. Two consecutive
held-out regressions request a stop, and extending the budget requires a new
versioned plan. The receipt is a resource decision, not a scientific result;
only the frozen evaluator can promote a model effect.

### Adapter-only training receipts

Methods that add a side adapter to a frozen backbone use a separate fit stage.
The fit command writes its state and receipt to an external artifact root. The
receipt must bind the candidate id and parameters, the exact training split and
backbone checkpoint, a freeze proof showing zero trainable backbone parameters,
the complete adapter parameter scope, and a positive optimizer step count. A
candidate without that binding is an integration-smoke artifact and cannot
enter screen, confirmation, or verified evidence. Reusing a bound batch is
allowed only after re-validating every digest; changing the candidate or state
requires a new fit receipt.

The resulting evidence chain is therefore:

```text
materialization receipt -> adapter training receipt -> screen/confirm receipts
                         -> frozen verifier -> portable knowledge record
```

The first Ctrl-World implementation is
`experiments/ctrl_world_masked_intermediate_adapter_v1/`; its `train_adapter.py`
owns only adapter updates and its `evaluate.py` checks state keys, shapes,
finiteness, and checkpoint binding before rollout.

For the current Ctrl-World subset, the planner sees 559 train windows from 7
episodes and 97 validation windows from 2 episodes. A screen is runnable, but
the pilot stage is correctly blocked until the training split contains at
least 8 episodes. This is the kind of training common sense that must live in
the control plane rather than in an individual researcher's memory.
