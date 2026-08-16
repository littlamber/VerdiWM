# Automatic GPU Experiments

This document defines the admission and settlement contract for the generic
VerdiWM GPU runner. It is a control-plane interface, not a license to launch
an arbitrary training campaign.

## Boundary

One plan describes one pre-registered trial. A plan must contain an objective,
hypothesis, selection reason, falsification criterion, stage, command, working
directory, GPU allowlist, time and GPU-hour budgets, metric gates, declared
artifacts, environment, and cleanup policy. The runner never invents random
trials or expands a plan after admission.

The four stages are ordered by cost and evidence strength:

| Stage | Purpose | Allowed conclusion |
| --- | --- | --- |
| `smoke` | prove runtime, CUDA, identity, and artifact wiring | execution is real and auditable |
| `screen` | cheap candidate triage | exploratory routing evidence |
| `gate` | frozen evaluator on a declared split | gate result under that protocol |
| `confirm` | held-out or multi-seed confirmation | result eligible for a backbone-specific promotion review |

A generic `PASS` remains `exploratory=True` in the archive. It cannot update
Intervention-Effect Memory or become a transfer prior until a frozen,
backbone-specific verifier promotes it.

## Candidate queues and progressive fidelity

For an external repository, `verdiwm-run` is the preferred orchestration entry
point. It executes the following fail-closed sequence:

```text
input lock -> onboarding -> CPU conformance -> candidate compilation
           -> budgeted GPU scheduler -> CAS/archive settlement -> cleanup
```

The run root contains `onboarding/`, `conformance/`, `compiled/`, `cas/`,
`archive.db`, and `pipeline-manifest.json`. A top-level `PASS` requires every
selected candidate to settle `PASS`; onboarding or conformance blockers stop
before GPU scheduling. An interruption is durable and rerunning the identical
command resumes from existing hashes and receipts.

For a bounded set of hypotheses, use a candidate batch rather than creating
plans ad hoc. The batch schema is
`configs/schemas/auto_experiment_candidate_batch.schema.json`, and the checked-in
CUDA canary is
`configs/smoke/auto_experiment_candidate_batch_cuda_v1.json`.

`verdiwm-auto-scheduler plan` validates every candidate's hypothesis, selection
reason, falsification criterion, stage ladder, plan contract, and candidate-local
budget. It computes the transparent score

```
expected_gain_weight * expected_gain
+ uncertainty_weight * uncertainty
+ information_gain_weight * information_gain
+ novelty_weight * novelty
- cost_weight * screen_gpu_hours
```

Candidates are sorted by score with candidate ID as the deterministic tie-break.
The greedy planning pass charges only screen estimates against
`total_budget_gpu_hours` and applies `max_selected_candidates`; deferred
candidates are retained with a machine-readable reason. During execution, the
same campaign ceiling is enforced by one shared ledger across every admitted
stage. Planning and execution are different accounting views, not independent
budget pools, and every individual ladder must fit inside the ceiling.

`verdiwm-auto-scheduler run` verifies each generated plan's SHA-256 before
launch, then executes selected candidates sequentially. `gate` is never started
unless `screen` settled `PASS`, and `confirm` is never started unless `gate`
settled `PASS`. A `VOID` blocks the remaining ladder but remains a durable
exploratory record. `execution.json` is resumable and locks the shared budget
database path, so changing the path cannot reset the campaign budget.

```bash
uv run verdiwm-auto-scheduler plan \
  --batch configs/smoke/auto_experiment_candidate_batch_cuda_v1.json \
  --workspace-root /path/to/VerdiWM \
  --output-root /path/to/verdiwm-runs/auto-scheduler-cuda-plan-v1

uv run verdiwm-auto-scheduler run \
  --queue /path/to/verdiwm-runs/auto-scheduler-cuda-plan-v1/queue.json \
  --workspace-root /path/to/VerdiWM \
  --archive-db /path/to/verdiwm-runs/archive.db \
  --cas-root /path/to/verdiwm-runs/auto-scheduler-cuda-plan-v1/cas \
  --lock-root /tmp/verdiwm-gpu-leases
```

This queue is an execution and resource-allocation mechanism. It does not
create optimization-memory entries, assert model-quality gains, or authorize a
world-model training campaign.

## Admission and execution

`verdiwm-auto-experiment run` performs these checks in order:

1. Validate the JSON plan against `configs/schemas/auto_experiment_plan.schema.json` and reject short rationale, unsafe paths, invalid gates, or an estimate over the declared budget.
2. Reserve one trial in the campaign-shared SQLite budget ledger. The ledger
   locks its total budget policy on first use, so changing an output directory
   or later plan cannot reset or enlarge it. High-cost trials require explicit
   `human_approved` and are capped by policy.
3. Acquire a host-local `fcntl` lease from the allowlisted physical GPU indices. A fresh `nvidia-smi` snapshot must show low memory/utilization and no compute application.
4. Run a read-only GPU exclusivity audit. The child receives `CUDA_VISIBLE_DEVICES` plus the physical GPU UUID and index. Inside the child, logical `cuda:0` is the leased physical device.
5. Execute synchronously with a hard timeout while recording stdout, stderr, and an `nvidia-smi` sampling curve.
6. Verify the declared result: schema, CUDA device, exact physical UUID, metric gates, and non-zero activity during execution.

The runner does not infer GPU use from `torch.cuda.is_available()`. A valid
runtime result must be joined to a physical UUID and a sampling curve with an
active during-execution sample.

By default the ledger is stored under `<output-parent>/budgets/` using a hash
of `campaign_id`. All run roots for that campaign must share the same output
parent, or operators must pass the same explicit `--budget-db` path.

## Settlement protocol

Every admitted attempt reaches one of two terminal outcomes:

- `PASS`: all result, metric, identity, and activity checks pass.
- `VOID`: command failure, timeout, missing artifact, malformed output, wrong
  GPU, failed metric gate, or missing activity.

Both outcomes retain their rationale, source state, execution metadata, cost,
verdict, logs, sampling curve, and artifact references. Settlement follows a
receipt-first order:

1. Put declared artifacts, support evidence, context, and verdict into CAS.
2. Put the immutable receipt core into CAS and derive its `cas://sha256/...` reference.
3. Write the local receipt and verdict projection.
4. Settle the budget with the same fencing token.
5. Publish the settled-trial row to Archive.
6. Mark the scratch directory settled and apply only the plan's cleanup policy.

If the process stops after Archive publication, rerunning the same plan
rebuilds the local manifest and finishes the pending projection. A stale worker
cannot settle a takeover because the budget fencing token changes.

## Artifact retention and cleanup

The default cleanup policy is `retain`. `archive_then_delete` is permitted only
for a trial's own `output/scratch/<trial>/attempt-*` directory. Immediate
cleanup is allowed after settlement; periodic cleanup is conservative and
defaults to dry-run.

Periodic cleanup independently checks the receipt in CAS, the settled trial in
Archive, and every declared/support/context/verdict reference in CAS. Missing
proof retains the directory and reports a reason. It never deletes source
trees, raw datasets, checkpoints, or another run's output.

## Smoke command

Run the checked-in CUDA canary only after selecting an actually idle GPU from
the plan allowlist:

```bash
uv run verdiwm-auto-experiment run \
  --plan configs/smoke/auto_experiment_cuda_smoke_v1.json \
  --workspace-root /path/to/VerdiWM \
  --output-root /path/to/verdiwm-runs/auto-experiment-cuda-smoke-v1
```

The smoke workload performs a bounded matrix multiplication, writes
`result.json`, and records the physical UUID observed by both `nvidia-smi` and
PyTorch. Its output belongs outside the source checkout. Inspect
`manifest.json`, `receipts/`, `verdicts/`, the Archive database, and the CAS
directory before treating the control plane as ready for model experiments.

## Operational rules

- Keep one plan, one trial, and one output root. Reusing an output root with a
  different plan is rejected by `plan.lock.json`.
- Keep raw datasets and model checkpoints outside run roots.
- Do not interpret `VOID` as a missing result; it is a retained negative
  control-plane observation.
- Do not promote generic receipts into reusable effect memory. Promotion needs
  the target backbone's frozen evaluator, held-out split, and review contract.
- Run cleanup in dry-run first:

```bash
uv run verdiwm-auto-experiment cleanup \
  --run-root /path/to/verdiwm-runs/auto-experiment-cuda-smoke-v1 \
  --older-than-hours 168
```
