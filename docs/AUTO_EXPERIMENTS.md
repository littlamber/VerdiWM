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

The four stages are ordered by cost and evidence strength. `screen` is
optional for a formal-first campaign; a candidate may start at `gate` when the
declared training/evaluation plan is ready. `confirm` still requires a positive
formal gate.

| Stage | Purpose | Allowed conclusion |
| --- | --- | --- |
| `smoke` | prove runtime, CUDA, identity, and artifact wiring | execution is real and auditable |
| `screen` | cheap candidate triage/diagnostics | exploratory routing evidence; never a scientific veto |
| `gate` | frozen evaluator on a declared split | gate result under that protocol |
| `confirm` | held-out or multi-seed confirmation | result eligible for a backbone-specific promotion review |

A generic `PASS` remains `exploratory=True` in the archive. It cannot update
Intervention-Effect Memory or become a transfer prior until a frozen,
backbone-specific verifier promotes it.

通用结果的字段、有限值约束与自定义 workload 元数据边界见
[通用实验结果契约](AUTO_EXPERIMENT_RESULT.md)。schema 校验覆盖所有结果指标，
包括未进入本次 gate 的指标；结构损坏的结果保留原始证据并结算为 `VOID`。

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

接入阶段每个依赖 import 与 `pip check` 默认最多运行 30 秒，可通过
`--runtime-timeout-seconds` 设置大于 0 且不超过 300 的有限秒数。它与
`--conformance-timeout-seconds` 独立，并纳入顶层不可变输入绑定；修改后应使用新输出
目录。GPU admission 要求显存与利用率为有效有限数值，并成功检查 compute apps。
查询失败不会被视为空闲；多卡请求会遍历允许的 GPU，选足完整可用子集，数量不足或
获取锁中途失败时释放本次已经取得的所有 partial leases。

For a bounded set of hypotheses, use a candidate batch rather than creating
plans ad hoc. The batch schema is
`configs/schemas/auto_experiment_candidate_batch.schema.json`, and the checked-in
CUDA canary is
`configs/smoke/auto_experiment_candidate_batch_cuda_v1.json`.

`verdiwm-auto-scheduler plan` validates every candidate's hypothesis, selection
reason, falsification criterion, stage ladder, plan contract, and candidate-local
budget. Stage ladders may be `screen`, `screen -> gate`, `screen -> gate ->
confirm`, `gate`, or `gate -> confirm`. It computes the transparent score

```
expected_gain_weight * expected_gain
+ uncertainty_weight * uncertainty
+ information_gain_weight * information_gain
+ novelty_weight * novelty
- cost_weight * entry_stage_gpu_hours
```

Candidates are sorted by score with candidate ID as the deterministic tie-break.
The greedy planning pass charges the first admitted stage estimate against
`total_budget_gpu_hours` and applies `max_selected_candidates`; deferred
candidates are retained with a machine-readable reason. During execution, the
same campaign ceiling is enforced by one shared ledger across every admitted
stage. Planning and execution are different accounting views, not independent
budget pools, and every individual ladder must fit inside the ceiling.

`verdiwm-auto-scheduler run` verifies each generated plan's SHA-256 before
launch, then executes selected candidates sequentially. `screen` is advisory:
whether it settles `PASS`, `VOID`, or another non-positive outcome, the
candidate may still proceed to the frozen `gate`. A `confirm` is started only
after a settled positive `gate`; a non-passing formal stage blocks the remaining
formal ladder but remains a durable record. This prevents a low-fidelity proxy
from silently filtering an idea that could work under complete training and
frozen evaluation. `execution.json` is resumable and locks the shared budget
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

This queue is an execution and resource-allocation mechanism. Explicitly
declared diagnostic results can enter the retrieval index through the contract
below; scientific effect memory still requires its own frozen verification.

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

若进程在本地 receipt 写入后、预算提交前中断，恢复会读取 CAS 中的原 receipt，
核对计划、输出目录预留、预算库、原 fencing token 和实际成本，再幂等补结算。
预算已提交但 Archive 尚未发布时也按同一顺序恢复；已有相同结算不会重复计费。
不同 token、成本或 receipt 的重放会被拒绝。

若预算已经 admitted，但没有可核验的终态 receipt，运行成本仍然未知，恢复返回
`AUTO_EXPERIMENT_INCOMPLETE_ATTEMPT_REQUIRES_RECONCILIATION`，保留原预算预留，
需要人工核对执行情况。重复命令不会自动接管或重新启动这笔 GPU 实验。

`execution.json` 和 `pipeline-manifest.json` 都是派生视图。scheduler 在恢复时通过
runner 重建原 trial 的 verdict；把 JSON 中的 `VOID` 改成 `PASS` 不能获得晋级。
顶层终态绑定 journal 中的 manifest/execution CAS 快照，并对前置诊断探针和所有
已完成候选执行 `recover_only` 校验。receipt、declared artifact、support、verdict
或 failure context 的 CAS 对象缺失或损坏都会阻断恢复，且不会启动新 GPU 工作。
旧终态若缺少这些快照绑定，需要显式核对后迁移，不能仅凭旧 JSON 宣告恢复成功。

## 候选诊断结果自动积累

候选声明 `experiment_kind: "diagnostic"` 后，可在 stage 中显式指定
`diagnostic_result_path: "diagnostic.json"`，并把该相对路径列入 `artifacts`。
compiler/scheduler 会原样保留这一声明。通用 runner 的 `result_path` 继续指向通用
实验结果；`diagnostic.json` 必须满足
[诊断结果契约](DIAGNOSTIC_PROBE_RESULT.md)。所有配对观察依赖的 proof 文件也须
声明为 artifacts，才能从同一 receipt 的 CAS 引用复核。其他 experiment kind、
未声明 artifact 或不安全相对路径会在预算 admission 前被拒绝。

顶层 `verdiwm-run` 自动使用已绑定的 retrieval 数据库；独立运行 scheduler 时通过
`verdiwm-auto-scheduler run --retrieval-db /path/to/experience/index.db` 启用积累。
第一轮和第二轮需要共用同一 retrieval、Archive 与 CAS，并保持这些路径通过参数
配置。未声明 `diagnostic_result_path` 的旧计划保持原有执行行为。

每个 trial 结算后，scheduler 从权威 receipt/CAS 读取显式声明的诊断结果，复用
`index_probe_experience` 核验 schema、Archive 绑定、资产 fingerprint 和配对 proof；
不会扫描任意 JSON 来猜测结果。索引失败会阻断这次编排；恢复先复核已有 trial，
再重试索引。索引已提交后进程中断也可幂等重放，不会重复执行 trial 或重复计数。

正向、负向和无变化观察都被保留。配对观察的方向来自冻结指标与实际 control /
candidate 证据；执行 `PASS` 不能生成正向效果标签，`VOID` 不提供效果排序先验。
下一轮可用这些经验调整候选排序，目标模型仍须重新验证。直接调用通用 runner
只负责结算与归档；使用自动积累应选择顶层或 scheduler 入口。

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

Each candidate declares an `experiment_kind`. For `model_training` candidates,
`training_binding` is required; only genuinely evaluation-only, data-only, or
diagnostic candidates may omit it. The scheduler then rejects
probe mode, sequential formal sampling, under-sized update budgets, and
insufficient episode coverage before GPU launch. The binding is exported to the
runner as `VERDIWM_TRAINING_*`; model adapters translate those values to native
flags. A failed screen remains diagnostic routing evidence and does not veto a
declared formal stage. The generic scheduler cannot infer hidden training from
an arbitrary shell command, so the candidate kind and binding are part of the
admission contract rather than a prompt convention.
