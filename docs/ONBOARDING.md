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
candidate compilation and `screen -> gate -> confirm`. A probe `result.json` must include
`probe_id`, `model_family`, `runtime_capability`, and non-empty
`failure_signatures` in addition to the generic CUDA result contract. Retrieval
changes candidate order only. Literature is stored as `shadow_only` data and
cannot provide a command, modify source code, bypass a typed candidate
contract, or enter formal verdict evidence. Registered method matches are
`ranking_only`; unknown methods produce guarded work orders under
`literature-methods/work-orders/` and prompt packets under
`literature-method-prompts/`. Each work order binds the future isolated
`AgentRepairSession` to the current source revision and registry digest, and
requires a new registry version plus all admission receipts. An empty compatible index is
recorded as `cold_start`, so the declared batch remains runnable.

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

The command performs no dependency installation, model import, training, inference, rollout, or GPU allocation. Dependency imports and `pip check` run in bounded subprocesses with CUDA hidden.

## Sidecar Contract

```text
<model>.verdiwm-instance/
  manifest.json
  model_manifest.json
  runtime_lock.json
  asset_manifest.json
  capability_report.json
  evaluator_contract.json
  generated_connector/
    connector.json
  conformance_report.json
  onboarding-report.json
  onboarding-report.md
```

`manifest.json` is the stable machine entrypoint. `optimization_launch_allowed` remains `false` until a separate conformance runner produces a passing receipt.

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
  "scheduler_template": "/path/to/candidate-batch-template.json"
}
```

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
uv run verdiwm-run /share/project/hywu/wjy/Ctrl-World \
  --output-root /share/project/hywu/wjy/verdiwm-runs/ctrl-world-universal-loop-v3 \
  --runtime-python /root/miniconda3/envs/ctrl-world/bin/python3.11 \
  --evaluator-contract configs/onboarding/ctrl_world_replay_evaluator_v1.json \
  --asset=--svd_model_path=/share/project/hywu/kyy/models/stable-video-diffusion-img2vid \
  --asset=--clip_model_path=/share/project/hywu/kyy/models/clip-vit-base-patch32 \
  --asset=--ckpt_path=/share/project/hywu/wjy/Ctrl-World/checkpoint-10000.pt \
  --asset=--dataset_root_path=/share/project/hywu/wjy/Ctrl-World/dataset_example \
  --asset=--dataset_meta_info_path=/share/project/hywu/wjy/Ctrl-World/dataset_meta_info \
  --no-import-probe
```

`--no-import-probe` skips the onboarding metadata probe only. The isolated
conformance stage still performs the declared real imports and evaluator help
check before it can authorize compilation.

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
