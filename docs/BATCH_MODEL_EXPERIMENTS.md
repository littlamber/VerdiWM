# Batch model experiments

`verdiwm batch-plan` is the static admission boundary for running one research
objective over several heterogeneous model instances. It compiles a request
into a durable `plan.json`; it does not import model code, run an evaluator,
allocate a GPU, or create a verdict.

The request lives outside model repositories and contains one row per model:

```json
{
  "schema_version": 1,
  "artifact_type": "verdiwm-model-batch-request",
  "batch_id": "long-horizon-cross-model-v1",
  "goal": "improve long-horizon action-conditioned prediction",
  "budget": "1gpu-hour",
  "target_metrics": ["runtime_ready"],
  "models": [
    {
      "model_id": "ctrl-world",
      "model": "/models/ctrl-world",
      "data": "/datasets/droid-heldout",
      "adapter": "ctrl-world-predictive-v2",
      "adapter_profile": "/contracts/ctrl-world-profile.json",
      "runtime_python": "/envs/ctrl-world/bin/python",
      "evaluator_contract": "/contracts/ctrl-world-evaluator.json",
      "dataset_freeze": "/contracts/droid-freeze.json",
      "source_revision": "git:0123456789abcdef",
      "assets": {"--ckpt_path": "/models/ctrl-world/checkpoint.pt"}
    }
  ]
}
```

Compile it before dispatch:

```bash
verdiwm batch-plan \
  --manifest batch-request.json \
  --output-root ./.verdiwm/batches/long-horizon-cross-model-v1
```

The plan records one stable digest per model, resolved local bindings,
per-model blockers, and the sum of declared GPU-hour budgets. Missing optional
bindings are retained as explicit `null` values; supplied but missing files
become blockers. The batch is dispatchable only when every model row is ready.
Repeating the command with the same request and plan is resumable; changing the
request requires a new output root.

Each executable row must name a frozen `evaluator_contract`. This is the
target-side evidence boundary, so a row without one is marked blocked even if
its model and data paths exist. When `adapter_profile` is supplied, its
evaluator binding must match the requested contract byte-for-byte; a mismatch
is blocked before a campaign record is created.

Once the plan is ready, materialize it into ordinary campaigns with:

```bash
verdiwm batch-run \
  --plan ./.verdiwm/batches/long-horizon-cross-model-v1/plan.json \
  --max-parallel 2
```

Use `--queue-only` when a scheduler or daemon should start the work later. The
command writes `execution.json` next to the plan. Each ready row receives a
deterministic campaign id, while rows that still have blockers remain visible
in the same manifest. Every generated pipeline points at the same batch-level
`budget.db`, Archive database, and CAS root; therefore splitting one batch into
several campaigns cannot multiply the available budget. Campaign records and
settlement receipts remain the source of truth for status and evidence.

Long-running batches can be inspected without dispatching more work:

```bash
verdiwm batch status \
  --execution ./.verdiwm/batches/long-horizon-cross-model-v1/execution.json
```

The status command validates the execution digest and refreshes each campaign
from CampaignStore. It does not queue, cancel, evaluate, settle, or create a
model-quality claim. `verdiwm batch-status --execution ...` remains available
as a script-friendly alias.

For a first-contact workflow, the short commands are:

```bash
verdiwm setup --model /models/my-model --data /datasets/heldout \
  --goal "improve long-horizon prediction"
verdiwm check
verdiwm diagnose
verdiwm run
```

`verdiwm check` combines local installation checks with model readiness and
returns a structured report containing completed checks, blockers, an action for
each blocker, and the next command to run. The older `init`, `check-model`, and
`guide-model` commands remain available for advanced or scripted use.

The plan is an execution input, not evidence. Later dispatch must create normal
campaign records and use the existing adapter compiler, budget ledger, GPU
lease, scheduler, receipt settlement, frozen verifier, Archive, and CAS. A
batch plan cannot establish an IRG result, a transfer claim, or a model-quality
improvement by itself.
