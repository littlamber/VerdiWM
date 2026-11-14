# Universal Model Onboarding

VerdiWM onboards external world-model repositories through a read-only discovery and contract boundary. Common Python projects do not need a handwritten VerdiWM adapter. The generated connector is declarative and lives outside the model source tree.

## Quick Start

When a model family has no matching adapter profile, generate a plain-language
request for an operator instead of asking them to construct internal JSON:

```bash
verdiwm bootstrap-template \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve long-horizon prediction"
```

The template explains the two approvals in ordinary terms: a trusted evaluator
baseline and a bounded adapter-generation provider. It contains no credentials.
The operator fills the approval fields or routes the request to an administrator;
the system keeps secrets in the provider environment and never asks users to
paste them into campaign state.

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

依赖冷启动较慢时，`verdiwm-onboard` 与顶层 `verdiwm-run` 都可显式设置
`--runtime-timeout-seconds 60`。该参数分别限制每次依赖 import 和 `pip check`，
默认 30 秒；只接受大于 0 且不超过 300 的有限秒数。超时仍会阻塞接入，并回收本次
探测的进程组。报告的 `runtime.probe_timeout_seconds` 记录实际设置；顶层运行会把
该设置绑定到不可变输入，调整设置须使用新的输出目录。此参数与后续真实 CPU
conformance 的 `--conformance-timeout-seconds` 独立，不跳过 import 检查。

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

使用 `research plan --adapter-profile /path/to/contracts/profile.json` 时，外部
profile 的 `evaluator_contract`、`probe_contract`、`candidate_catalog` 和
`constitution_freeze` 相对该 profile 所在目录解析；冻结清单的 `entries.path`
相对同一资源根目录。内置 profile 保留安装根目录下的相对路径，外部生成的旧
overlay 在本地资源缺失时也兼容这些内置绑定。schema 始终来自安装包，用户无需
向 `site-packages` 写入配置；符号链接、越界路径或无效的本地冻结证据会明确报错。

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

### 从已有评测配置生成待确认契约

`verdi evaluator discover --source-root /path/to/model-source` 只读扫描
`configs/evaluation/` 和 `configs/evaluators/`，报告候选、路径问题与扫描上限。
普通 rollout 配置可以作为候选，但其中的 FPS、horizon 或指标名称不能替代实际的
评测命令与 verifier。

```bash
verdi evaluator materialize \
  --candidate /path/to/model-source/configs/evaluation/eval.json \
  --source-root /path/to/model-source \
  --model-root /path/to/checkpoint \
  --data-root /path/to/dataset \
  --output-root /path/to/state/evaluator-review
```

返回的 receipt 用 `state` 说明下一步：

| 状态 | 含义与下一步 |
|:--|:--|
| `binding_required` | 按 `blockers` 中的字段与动作补齐 command、输入输出、metrics、verifier 或入口文件等实际绑定；`--confirm` 不能跳过缺失项。 |
| `ready_for_confirmation` | 声明的契约完整且入口存在。审阅候选与路径改写，再用新输出目录加 `--confirm`。 |
| `frozen` | 生成已明确确认的 JSON 契约，可交给 onboarding 与 conformance；仍需真实目标证据才能证明优化效果。 |

前两种状态的输出是审阅文件，原始配置位于 `candidate_manifest`，不能作为可执行
契约传给 onboarding。确认时重新指定原始候选，使用例如
`--output-root /path/to/state/evaluator-frozen --confirm`；已有输出目录拒绝覆盖。
YAML 候选统一生成 JSON 输出，receipt 保留原文件与输出文件的 SHA-256。

路径改写只使用候选明确声明的 source/model/data 根目录并保留相对后缀。无法唯一
映射的输入会报告 blocker；工具不会猜测 checkpoint，也不会补造 command、指标或
verifier。所有状态的 `optimization_launch_allowed` 都为 `false`：契约确认之后
还必须通过 CPU conformance、执行准入和目标侧验证。

## Ctrl-World Replay Example

The checked-in declarative contract and candidate template onboard Ctrl-World
without a handwritten Python adapter:

```bash
uv run verdiwm-run /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-universal-loop-v3 \
  --runtime-python /path/to/ctrl-world-env/bin/python \
  --evaluator-contract configs/evaluators/ctrl_world_predictive_probe_evaluator_v2.json \
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
mkdir -p /path/to/verdiwm-state/ctrl-world-predictive-v1
nohup uv run verdiwm-run-daemon /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-predictive-v1 \
  --daemon-state-root /path/to/verdiwm-state/ctrl-world-predictive-v1/daemon \
  --runtime-python /path/to/ctrl-world-env/bin/python \
  --evaluator-contract /path/to/VerdiWM/configs/evaluators/ctrl_world_predictive_probe_evaluator_v2.json \
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
`SIGINT` prevent another cycle and persist the stop after the current pipeline
call returns. Local subprocesses can respond to cancellation inside that call;
the current work is not guaranteed to finish. The identical command resumes
from the pipeline's persisted stage, Archive, CAS, and budget ledger. An admitted
trial without a settled receipt can still require cost reconciliation. If the
cycle limit is reached while capacity remains unavailable, the state is
`exhausted`; increasing only `--max-cycles` resumes the same immutable run.
Changing the model, evaluator, probe, assets, or retry policy requires a new
daemon state root.

普通研究任务每轮都会在诊断和经验检索之后准备联网查询。查询保留用户目标，
使用任务领域、当前轮失败征象与可用机制线索；不会把内部轮换执行策略当成
论文关键词。`retrieval/literature-query-plan.json` 保存本轮查询及诊断、经验
来源，`literature/` 下的 manifest、请求记录和论文记录用于复核真实联网结果。
查询和文献只提供候选线索，模型收益仍须由冻结验证器确认。

直接使用单轮 pipeline、未提供 `--literature-query` 时，可以使用当前诊断与
模型领域中的明确检索词准备查询；已有匹配经验时也会纳入查询上下文。
普通多轮研究入口会传入研究目标，因而每轮都准备文献检索。
相同输入恢复已有产物不等于新联网，具体网络、缓存和恢复状态以本轮请求记录为准。
中文目标保留原文；缺少可检索任务语义时会记录跳过原因，用户可在研究目标中
补充实际的英文领域关键词。查询放宽和无域跳过的边界见
[每轮联网检索与经验反馈](RUN_LIFECYCLE.md#每轮联网检索与经验反馈)。

## Continuous immutable evolution

普通用户可直接在现有终端设置多轮，无需另找后台控制器：

```text
verdi> /plan "提升长程一致性" --rounds 2 --budget 2gpu-hours --stop-on-success
verdi> /run
```

`/run --rounds 2` 也会先持久化并预览派生计划，确认一次后执行；无参数时回车
沿用原轮数。自然语言“跑两轮”只准备计划，不启动 GPU。多轮任务统一通过
campaign 的 `/status` 和 `/cancel ID` 管理，所有轮次共用总预算。
`--stop-on-success` 依赖 evaluator 的 `frozen_goal`（相对路径 goal v2）与本轮
confirm 的正式 CAS 判定证据；无此契约或证据时继续受轮数、预算等条件约束，
pipeline 通过或诊断重复都不算目标达成。
详见[研究运行与停止说明](RUN_LIFECYCLE.md)。

Both this public path and the compatible standalone `verdiwm-evolution-daemon`
reuse the same evolution controller. It materializes a new
`iteration-XXXXXX/inputs/` directory before every pipeline call. The generated
probe, evaluator, campaign, candidate, and trial identities are never reused;
the shared Archive/CAS/retrieval index and one global budget ledger remain the
authoritative memory and resource boundary.

For existing advanced scripts that launch the standalone command directly,
set `--max-iterations 2` for at most two research iterations;
`--max-iterations 0` explicitly removes its round limit. The normal
`research plan/run --rounds N` path always requires a positive finite limit and
disables stopping solely because diagnostic signatures repeat.

The command below is a resource-binding template, not a ready-to-run model
example. Supply a confirmed evaluator whose `scheduler_template` points to a
valid `auto_experiment_candidate_batch`, with `screen`, `gate`, and `confirm`
stages in each candidate. The bundled
`ctrl_world_predictive_probe_evaluator_v2.json` is a diagnostic evaluator and
does not provide this scheduler contract; using it directly here fails with
`EVOLUTION_EVALUATOR_TEMPLATE_INVALID`. Do not change a frozen evaluator just
to bypass that check. Prepare and review a new campaign contract instead.

```bash
mkdir -p /path/to/verdiwm-state/ctrl-world-evolution
nohup uv run verdiwm-evolution-daemon /path/to/Ctrl-World \
  --output-root /path/to/verdiwm-runs/ctrl-world-evolution \
  --state-root /path/to/verdiwm-state/ctrl-world-evolution \
  --evaluator-contract /path/to/confirmed-evaluator-with-scheduler.json \
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
  --no-import-probe --poll-seconds 60 --max-iterations 2 \
  --max-failures 3 --max-no-information 3 \
  > /path/to/verdiwm-state/ctrl-world-evolution/daemon.log 2>&1 &
```

`--max-iterations 0` means long-running, not unbounded resource usage: the
global GPU-hour ledger, consecutive-failure limit, and no-new-information limit
are hard stop conditions. With the same command and state root, a stopped or
recoverable exhausted controller resumes an unfinished immutable iteration; if
the last iteration completed before stopping, it starts the next one. Use a new
state root when changing contracts, assets, retry policy, the iteration limit,
or the global budget.
Stop signals propagate through the inner daemon, so a signal received during
the current pipeline call cannot silently start another research iteration.

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
