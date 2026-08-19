# Universal Model Onboarding

VerdiWM onboards external world-model repositories through a read-only discovery and contract boundary. Common Python projects do not need a handwritten VerdiWM adapter. The generated connector is declarative and lives outside the model source tree.

## Quick Start

The normal user path is one resumable command from source discovery through a
settled bounded experiment:

```bash
verdiwm-run /path/to/model \
  --output-root /path/to/runs/model-smoke-v1 \
  --runtime-python /path/to/environment/bin/python \
  --evaluator-contract /path/to/evaluator.json \
  --asset=--ckpt_path=/path/to/checkpoint.pt
```

`pipeline-input.lock.json` binds the source tree, evaluator, runtime path,
assets, budgets, and output stores. Repeating the same command resumes atomic
stages and already settled trials. Changing any locked input requires a new run
root; it never silently reuses old evidence.

For the paper-aligned diagnostic-first path, add a declarative probe contract:

```bash
verdiwm-run /path/to/model \
  --output-root /path/to/runs/model-campaign-v1 \
  --runtime-python /path/to/environment/bin/python \
  --evaluator-contract /path/to/evaluator.json \
  --probe-contract /path/to/diagnostic-probe.json \
  --retrieval-db /path/to/shared/verdiwm-retrieval.db \
  --archive-db /path/to/shared/verdiwm-archive.db \
  --cas-root /path/to/shared/verdiwm-cas \
  --literature-query "action conditioned world model long horizon drift"
```

The fixed order is conformance, diagnostic probe, receipt-bound experience
retrieval, optional read-only literature staging, typed method staging, then
candidate compilation and `screen -> gate -> confirm`. The declared probe result
(`result.json` by default, or `diagnostic_result_path`) must include `probe_id`,
`model_family`, `runtime_capability`, and non-empty `failure_signatures`. Retrieval
changes candidate order only. Literature is stored as `shadow_only` data and
cannot provide a command, modify source code, bypass a typed candidate
contract, or enter formal verdict evidence. Registered method matches are
`ranking_only`; unknown methods produce guarded work orders under
`literature-methods/work-orders/` and prompt packets under
`literature-method-prompts/`. Each work order binds the future isolated
`AgentRepairSession` to the current source revision and registry digest, and
requires a new registry version plus all admission receipts. An empty compatible index is
recorded as `cold_start`, so the declared batch remains runnable.

For video-producing models, a probe can keep the model repository untouched and
declare a generic paired-video workload. The command runs the external
rollout, then `wmloop.diagnose.external_video_probe` measures a declared
vertical or horizontal reference/prediction layout and emits
`diagnostic_result_path` (for example, `diagnostic-output.json`). Its
pre-registered thresholds produce receipt-bound signatures such as
`short_horizon_observed`, `horizon_drift`, and `paired_rollout_error_high`.
The utility is not a model adapter: the repository contributes only the
declared command, artifact glob, layout, and thresholds.

Once a diagnostic probe is present, candidate compilation is fail-closed for
routing. Every candidate must declare `retrieval_keys.failure_signatures` and
overlap at least one observed signature. Candidates with no matching route
remain in the immutable batch with `routing_admission.state=blocked`; they are
not materialized into GPU plans, charged budget, or allowed to create an
experiment receipt.

The method staging transaction can also be inspected separately:

```bash
verdiwm-literature-method-stage /path/to/literature/manifest.json \
  --repo-root /path/to/model \
  --output-root /path/to/literature-methods \
  --failure-signature train_infer_mismatch \
  --model-family ctrl-world
```

When `--retrieval-db` is shared, keep its archive and CAS shared as well. The
runner derives sibling defaults when those flags are omitted; this is required
for cross-campaign receipt verification.

Unless `--budget-total-gpu-hours` is supplied, the full runner derives one
shared ceiling by adding the probe's declared `estimated_gpu_hours` to the
candidate batch's `total_budget_gpu_hours`. The probe and all later candidate
stages settle against the same budget database. This prevents a diagnostic
probe and its follow-up screen from each assuming that it owns the full
campaign budget. For the checked-in Ctrl-World predictive contracts, the bound
is `0.08 + 0.08 = 0.16 GPU-hours`.

For discovery only, run:

```bash
verdiwm-onboard /path/to/model
```

The default sidecar is a sibling directory named `<model>.verdiwm-instance`. An explicit destination is also supported, but it must remain outside the model repository:

```bash
verdiwm-onboard /path/to/model \
  --output-root /path/to/onboarding/model-instance \
  --runtime-python /path/to/environment/bin/python
```

The command performs no dependency installation, model import, training, inference, rollout, or GPU allocation. Dependency imports and `pip check` run in bounded subprocesses with CUDA hidden. A `pip check` platform-metadata warning (for example, decord's unsupported-wheel notice) is recorded as `warning` only when every reported line is that known warning class; import failures and real dependency conflicts remain blocking.

## Sidecar Contract

```text
<model>.verdiwm-instance/
  manifest.json
  model_manifest.json
  runtime_lock.json
  asset_manifest.json
  capability_report.json
  model-capability-ir.json
  evaluator_contract.json
  generated_connector/
    connector.json
  conformance_report.json
  onboarding-report.json
  onboarding-report.md
```

`manifest.json` is the stable machine entrypoint. It binds both the file hash
and semantic digest of `model-capability-ir.json`. The Capability IR contains
semantic capabilities, content revision, interfaces, asset classes, and frozen
evaluator identity; it excludes checkout, runtime, checkpoint, and dataset
paths. `optimization_launch_allowed` remains `false` until a separate
conformance runner verifies those bindings and produces a passing receipt.

## Admission States

| State | Meaning | Scheduler action |
|:--|:--|:--|
| `blocked` | A runtime, source revision, checkpoint, evaluation entrypoint, or evaluator contract is missing or invalid. | Reject. |
| `binding_required` | Discovery completed but a declarative binding is incomplete. | Reject. |
| `ready_for_conformance_smoke` | Runtime and evaluator bindings are complete. Model execution has not been authorized. | Run bounded conformance only. |

The scanner fails closed with stable blocker codes such as `RUNTIME_UNREADY`, `SOURCE_REVISION_UNBOUND`, `CHECKPOINT_MISSING`, `MODEL_ASSET_BINDING_REQUIRED`, `EVALUATION_ENTRYPOINT_MISSING`, and `EVALUATOR_CONTRACT_REQUIRED`. Input-like CLI flags are matched to discovered sidecar assets, so missing model, policy, checkpoint, and dataset paths are rejected before conformance or GPU admission.

## Evaluator Contract

An evaluator contract is an external frozen JSON file. Keeping it outside the imported repository prevents onboarding from rewriting upstream model code or silently changing verdict semantics.

```json
{
  "evaluator_id": "task_success_v1",
  "command": ["{python}", "{repo_root}/scripts/eval.py", "--ckpt_path", "{asset:--ckpt_path}"],
  "input_artifacts": ["checkpoint", "frozen_split"],
  "output_artifacts": ["evaluation-receipt.json"],
  "metrics": ["success_rate"],
  "verifier": "task_success_receipt_v1",
  "conformance_imports": ["torch"],
  "entrypoint_probe": "help",
  "scheduler_template": "/path/to/candidate-batch-template.json"
}
```

`entrypoint_probe` defaults to `help`. Set it to `skip` only when importing the
entrypoint itself performs heavyweight model construction or other side effects
that make `--help` unsafe. In that case conformance still runs the declared
imports and records the deferred probe; a bounded runtime smoke must execute the
real command before any screen or training queue is admitted.

Bind it with:

```bash
verdiwm-onboard /path/to/model \
  --runtime-python /path/to/environment/bin/python \
  --evaluator-contract /path/to/contracts/task-success-v1.json
```

Discovery evidence is not a performance claim. Only a passing frozen evaluator receipt may enter reusable optimization memory.

Required evaluator assets are fingerprinted during onboarding, checked before
and after CPU conformance, embedded in the PASS receipt, and recomputed before
compilation and every scheduler admission. A changed checkpoint, model
dependency, or dataset is rejected before GPU allocation.

## Ctrl-World Replay Example

The checked-in declarative contract and candidate template onboard Ctrl-World
without a handwritten Python adapter:

```bash
uv run verdiwm-run /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-universal-loop-v3 \
  --runtime-python /path/to/ctrl-world-env/bin/python \
  --evaluator-contract configs/onboarding/ctrl_world_replay_evaluator_v1.json \
  --asset=--svd_model_path=/path/to/models/stable-video-diffusion-img2vid \
  --asset=--clip_model_path=/path/to/models/clip-vit-base-patch32 \
  --asset=--ckpt_path=/path/to/Ctrl-World/checkpoint-10000.pt \
  --asset=--dataset_root_path=/path/to/Ctrl-World/dataset_example \
  --asset=--dataset_meta_info_path=/path/to/Ctrl-World/dataset_meta_info \
  --no-import-probe
```

`--no-import-probe` skips the onboarding metadata probe only. The isolated
conformance stage still performs the declared real imports and evaluator help
check (unless the contract explicitly sets `entrypoint_probe` to `skip`) before
it can authorize compilation. A deferred probe never counts as model-quality
evidence; the subsequent runtime smoke must produce the physical-GPU and
artifact receipt.

## Running the full pipeline in the background

Use `verdiwm-run-daemon` when the diagnostic probe may need to wait for GPU
capacity. It invokes the same resumable pipeline transaction as `verdiwm-run`;
it does not introduce a second evaluator or experiment path.

```bash
nohup uv run verdiwm-run-daemon /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-predictive-v1 \
  --daemon-state-root /path/to/verdiwm-state/ctrl-world-predictive-v1/daemon \
  --runtime-python /path/to/ctrl-world-env/bin/python \
  --evaluator-contract /path/to/VerdiWM/configs/onboarding/ctrl_world_predictive_probe_evaluator_v1.json \
  --probe-contract /path/to/VerdiWM/configs/probes/ctrl_world_predictive_diagnostic_v1.json \
  --retrieval-db /path/to/verdiwm-state/ctrl-world-predictive-v1/retrieval.db \
  --archive-db /path/to/verdiwm-state/ctrl-world-predictive-v1/archive.db \
  --cas-root /path/to/verdiwm-state/ctrl-world-predictive-v1/cas \
  --budget-db /path/to/verdiwm-state/ctrl-world-predictive-v1/budget.db \
  --asset=--svd_model_path=/path/to/models/stable-video-diffusion-img2vid \
  --asset=--clip_model_path=/path/to/models/clip-vit-base-patch32 \
  --asset=--ckpt_path=/path/to/Ctrl-World/checkpoint-10000.pt \
  --asset=--dataset_root_path=/path/to/Ctrl-World/dataset_example \
  --asset=--dataset_meta_info_path=/path/to/Ctrl-World/dataset_meta_info \
  --no-import-probe --poll-seconds 60 --max-cycles 1440 --max-attempts 3 \
  > /path/to/verdiwm-state/ctrl-world-predictive-v1/daemon.log 2>&1 &
```

The daemon's `status.json` separates `deferral_count` from `error_count`.
`GPU_LEASE_UNAVAILABLE` waits do not consume `--max-attempts`, create receipts,
or charge budget. Other failures use the finite attempt bound. `SIGTERM` and
`SIGINT` stop after the current pipeline call, and the identical command resumes
from the pipeline's persisted stage, Archive, CAS, and budget ledger. If the
cycle limit is reached while capacity remains unavailable, the state is
`exhausted`; increasing only `--max-cycles` resumes the same immutable run.
Changing the model, evaluator, probe, assets, or retry policy requires a new
daemon state root.

When no explicit `--literature-query` is supplied, a successful probe with no
compatible receipt-bound experience automatically derives a network literature
query from its model family and failure signatures. A matched experience index
skips that cold-start query.

## Continuous immutable evolution

For unattended iteration, use `verdiwm-evolution-daemon`. It materializes a new
`iteration-XXXXXX/inputs/` directory before every pipeline call. The generated
probe, evaluator, campaign, candidate, and trial identities are never reused;
the shared Archive/CAS/retrieval index and one global budget ledger remain the
authoritative memory and resource boundary.

```bash
nohup uv run verdiwm-evolution-daemon /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-evolution \
  --state-root /path/to/verdiwm-state/ctrl-world-evolution \
  --evaluator-contract /path/to/VerdiWM/configs/onboarding/ctrl_world_predictive_probe_evaluator_v1.json \
  --probe-contract /path/to/VerdiWM/configs/probes/ctrl_world_predictive_diagnostic_v1.json \
  --retrieval-db /path/to/verdiwm-state/ctrl-world-evolution/retrieval.db \
  --archive-db /path/to/verdiwm-state/ctrl-world-evolution/archive.db \
  --cas-root /path/to/verdiwm-state/ctrl-world-evolution/cas \
  --budget-db /path/to/verdiwm-state/ctrl-world-evolution/budget.db \
  --total-budget-gpu-hours 24 \
  --budget-max-trial-gpu-hours 240 \
  --budget-high-trial-limit 8 --auto-approve-high-cost \
  --runtime-python /path/to/ctrl-world-env/bin/python \
  --asset=--svd_model_path=/path/to/models/stable-video-diffusion-img2vid \
  --asset=--clip_model_path=/path/to/models/clip-vit-base-patch32 \
  --asset=--ckpt_path=/path/to/Ctrl-World/checkpoint-10000.pt \
  --asset=--dataset_root_path=/path/to/Ctrl-World/dataset_example \
  --asset=--dataset_meta_info_path=/path/to/Ctrl-World/dataset_meta_info \
  --no-import-probe --poll-seconds 60 --max-iterations 0 \
  --max-failures 3 --max-no-information 3 \
  > /path/to/verdiwm-state/ctrl-world-evolution/daemon.log 2>&1 &
```

`--max-iterations 0` means long-running, not unbounded resource usage: the
global GPU-hour ledger, consecutive-failure limit, and no-new-information limit
are hard stop conditions. A stopped or exhausted controller resumes the current
immutable iteration with the same command and state root. Use a new state root
when changing contracts, assets, retry policy, or the global budget.

The resource flags are admission controls, not scientific quality gates. A
method estimated at 80 GPU-hours is classified as `high` and can run when the
declared policy allows it; its quality promotion still depends only on the
declared confirm-stage metrics. Set `--budget-max-trial-gpu-hours` above the
longest stage and use a sufficiently large `--total-budget-gpu-hours`. The
`--auto-approve-high-cost` flag is an explicit operator policy choice for
unattended campaigns, not evidence that the method is scientifically valid.

## Moving an admitted queue to the background

After conformance and candidate compilation have produced
`compiled/queue/queue.json`, the queue can be drained by the generic campaign
daemon. No model-family-specific background adapter is required: model-specific
behavior stays in the declarative evaluator contract and generated experiment
plans.

```bash
nohup uv run verdiwm-campaign-daemon \
  --queue /path/to/pipeline/compiled/queue/queue.json \
  --output-root /path/to/runs/campaigns/model-v1 \
  --workspace-root /path/to/model \
  --archive-db /path/to/verdiwm-state/archive.db \
  --cas-root /path/to/verdiwm-state/cas \
  --budget-db /path/to/verdiwm-state/budgets/model-v1.db \
  --budget-total-gpu-hours 0.03 \
  --lock-root /tmp/verdiwm-gpu-leases \
  --max-parallel 1 --max-attempts-per-candidate 3 \
  --poll-seconds 60 --max-cycles 1440 \
  > /path/to/logs/model-v1.log 2>&1 &
```

The log path, Archive, CAS, and an explicitly supplied budget ledger must be
outside the daemon output root. Inspect `status.json` for the current campaign
state and `cycles/cycle-*.json` for per-worker outcomes. Sending `SIGTERM` or
`SIGINT` stops admission of new work after in-flight workers return and persists
`state=stopped`; rerunning the identical command resumes unfinished candidates.
Changing a queue or durable input requires a new daemon output root so evidence
from different immutable inputs cannot be mixed.

When all allowed GPUs are occupied, the candidate appears as `deferred` rather
than `blocked` inside `candidate_states`. Deferral does not create an experiment
receipt, charge the GPU-hour budget, or consume `--max-attempts-per-candidate`.
The daemon checks again after `--poll-seconds` until capacity appears or
`--max-cycles` provides the campaign-level stop bound.
