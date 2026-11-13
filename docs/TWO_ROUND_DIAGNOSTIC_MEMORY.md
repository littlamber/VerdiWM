# 两轮诊断与经验复用

本流程检验同一个用户的两轮实验能否连接起来：第一轮保存真实配对测量，第二轮
先读取第一轮，再决定候选并重新验证。诊断经验只能影响候选排序；正式效果记忆
仍由目标模型的冻结 verifier、多 seed、held-out 和保护指标决定。

当前支持的配对 producer 是 Wan2.2 arm-slot 的 `history_sparse_dilation`。
其他模型仍可使用通用实验和诊断入口，但不能仅改 primitive 名称就生成这里的
配对经验。原生模型接入条件见 [有界推理说明](WAN_ARM_SLOT_INFERENCE.md)，
结果与权限约定见 [诊断结果契约](DIAGNOSTIC_PROBE_RESULT.md)。

## 目录和运行环境

模型源码、checkpoint、数据、模型 Python 和实验输出由用户提供。控制面使用
VerdiWM 自己的 Python 环境；模型 Python 通过参数传入，保留虚拟环境 launcher。
以下目录仅表示各自用途，可以放在用户机器的任意位置：

```text
experiment/
  inputs/                 # 用户的配置、selection、冻结协议与评价代码
  baseline/               # 已核验的 control receipt 与模型输出
  round-one/              # 新执行，不覆盖 baseline
  round-two/              # 检索后选择的新执行
  memory/
    archive.db            # 两轮共用的终态事实来源
    evidence/             # 两轮共用的 CAS
    retrieval.db          # 从 archive/CAS 派生的检索索引
  shared-budget.db        # 两轮共用，不因换输出目录而重建
```

配置文件和运行产物可以包含用户自己的实际路径；公开示例与源码不绑定这些路径。
配对 specification 中的证据路径相对本次 evidence root，归档后从 receipt 的
artifact refs 和 CAS 重建。原生模型资产只保存身份摘要，重放需要用户保留相同资产。

## 从公共顶层入口运行

已经完成模型接入时，可以由一个入口串联 onboarding、CPU conformance、GPU
diagnostic probe、检索、候选编译/执行、结算和诊断积累。先准备 evaluator contract、
它引用的 scheduler template，以及 diagnostic probe contract；其字段与完整通用示例
见 [自动实验](AUTO_EXPERIMENTS.md) 和 [模型接入](ONBOARDING.md)。下面使用环境
变量表示用户自行选择的路径，配置与运行目录都位于 checkout 之外：

```bash
uv run --offline --frozen python -m wmloop.execute.autonomous_pipeline "$MODEL_SOURCE" \
  --output-root "$EXPERIMENT_ROOT/round-one" \
  --evaluator-contract "$EXPERIMENT_ROOT/inputs/evaluator-contract.json" \
  --probe-contract "$EXPERIMENT_ROOT/inputs/probe-contract.json" \
  --runtime-python "$MODEL_PYTHON" \
  --runtime-timeout-seconds 60 \
  --conformance-timeout-seconds 120 \
  --asset=--checkpoint_path="$MODEL_CHECKPOINT" \
  --asset=--data-root="$MODEL_DATA" \
  --archive-db "$EXPERIMENT_ROOT/memory/archive.db" \
  --cas-root "$EXPERIMENT_ROOT/memory/evidence" \
  --retrieval-db "$EXPERIMENT_ROOT/memory/retrieval.db" \
  --lock-root "$EXPERIMENT_ROOT/gpu-locks" \
  --budget-db "$EXPERIMENT_ROOT/shared-budget.db" \
  --budget-total-gpu-hours 1
```

`--asset` 左侧名称必须匹配模型接入报告中的参数绑定；并非所有模型都叫
`--checkpoint_path` 或 `--data-root`。第一次外部安装按 README 使用联网的
`uv sync --locked`，缓存齐备后的实验才使用上面的离线命令。设置
`runtime-timeout-seconds` 只改变 CPU 依赖探测上限，不改变冻结评价器或 GPU
实验超时。每个 probe/stage 的 GPU、命令、超时和预留成本仍在自己的 contract 中声明。

在候选的 diagnostic stage 中声明 `"diagnostic_result_path": "diagnostic.json"`，
同时把 `diagnostic.json` 及其引用的所有证据放入 `artifacts`。顶层入口始终把共用的
retrieval database 传给 scheduler；候选终态落入 Archive/CAS 后，公共积累服务
读取权威 receipt 中该 artifact，复用同一索引校验写入检索库。它不扫描任意 JSON
猜测诊断类型，也不根据执行 `PASS` 推导正向效果。缺失或不合法的声明结果会报错，
不能静默宣布闭环成功。

第二轮使用新的 output root，并继续传入同一 budget、archive、CAS 和 retrieval
路径。保持同一轮命令不变再次运行则进入恢复：已有权威终态证据核验成功后补齐
结算与索引，不领取新 GPU、不增加已结算成本。若进程中断且尚无终态 receipt，
成本仍未知，需先显式 reconciliation；重复运行不会擅自退款或再执行一遍。

这要求模型专用 workload 已经正确连接原生命令、输入绑定、冻结测量与配对诊断。
Wan arm-slot 的公共构件见下文；当前没有为任意 checkpoint 自动生成这些适配的
功能。workload 只负责模型输入/输出，预算、GPU lease、恢复和最终判定复用外层
公共服务。不要在 workload 中重复实现调度器或依赖本地审计目录中的 helper。

## 第一轮：执行、测量、结算和索引

先固定 checkpoint、样本、seed、split、采样步数、control/candidate 的参数和
数值评价定义，再冻结协议与评价代码。协议应声明允许的候选范围、指标方向和
零差异容忍值。若历史结果已知，记录这个事实；不能将它描述为未见过的 held-out。
对 Wan rollout 的未来帧评价，还要明确是否去掉第 0 帧的 GT history anchor。

新用户可以直接冻结包内的 CPU 数值 recipe。准备一个 `comparison.json`，将
样本和 seed 替换成自己的 selection，并在生成候选之前运行 `freeze`：

```json
{
  "evaluator_id": "my-wan-future-screen-v1",
  "ranking_primary_metric": "psnr_db",
  "metric_tie_epsilon": 0.0,
  "comparison": {
    "primitive": "history_sparse_dilation",
    "baseline_history_rollout_dilation": 2,
    "allowed_candidate_history_rollout_dilations": [1, 3],
    "config_fields": [
      "world_model.history_rollout_dilation",
      "wm_training.history_rollout_dilation"
    ],
    "seed": 123,
    "split": "val",
    "inference_steps": 25,
    "max_chunks": 1,
    "episode_id": 0,
    "frame_now": 40,
    "identical_ground_truth_bytes_required": true,
    "other_config_semantics_must_match": true,
    "weights_fixed": true,
    "data_fixed": true
  }
}
```

```bash
uv run --offline --frozen python -m wmloop.evaluate.future_video_metrics freeze \
  --protocol /path/to/experiment/inputs/comparison.json \
  --output-root /path/to/experiment/inputs/frozen-future-v1

uv run --offline --frozen python -m wmloop.evaluate.future_video_metrics measure \
  --frozen-root /path/to/experiment/inputs/frozen-future-v1 \
  --prediction /path/to/experiment/round-one/rollouts/sample/sa_wm_rollout.mp4 \
  --ground-truth /path/to/experiment/round-one/rollouts/sample/gt.mp4 \
  --output /path/to/experiment/round-one/measurement.json
```

`freeze` 固定数值 recipe、指标方向、裁剪规则和诊断权限，保存 `protocol.json`、
`freeze-record.json` 和稳定数值模块 `future_video_recipe.py` 的源字节身份；
`measure` 只使用已安装的公开实现，
核对冻结文件后复算，不执行归档中的任意 Python。这个 recipe 要求恰好 13 帧、
512×768 RGB uint8，评价第 1 至 12 帧，SSIM 为 global RGB 定义。
分别对 control 和 candidate 运行 `measure`，每次使用新的输出文件。
冻结后保留冻结文件；数值 recipe 变更需要新的协议身份。CLI 或资源管理的修改
不会改变这个独立数值模块的身份。

模型执行使用 [通用 GPU runner](AUTO_EXPERIMENTS.md) 的 plan、共享预算和 GPU
lease。一个外层 runner 负责资源与最终结算；外部 workload 调用模型原生命令。
`wmloop.execute.wan_arm_slot.build_native_execution_evidence` 可以复核已经执行的
原生 manifest、输出和输入身份，生成过程证据。它不领取资源，也不创建终态
receipt。不要将自行管理 lease 的 `wan_arm_slot` CLI 再嵌入已持有 lease 的 runner。

冻结评价器分别测量 control 与 candidate。配对 producer 要求两侧 context、seed、
GT 和除干预参数之外的配置相同，并核对实际执行、配置、视频与测量摘要。
producer 计算方向统一的 `signed_delta`：大值更好的指标为 candidate − control，
小值更好的指标为 control − candidate。`positive`、`negative`、`null` 是本次
匹配测量的描述，权限固定为 `ranking_only`。

在外部 workload 已生成原生执行证据、冻结测量和单视频诊断结果后，用可安装的
module CLI 合成配对诊断，无需从本地审计目录复制 helper：

```bash
uv run --offline --frozen python -m wmloop.diagnose.paired_observation \
  --evidence-root /path/to/experiment/round-one/scratch \
  --specification /path/to/experiment/inputs/paired-specification.json \
  --diagnostic-result /path/to/experiment/round-one/scratch/single-video-diagnostic.json \
  --output /path/to/experiment/round-one/scratch/diagnostic.json
```

specification 的完整结构由
`configs/schemas/diagnostic_paired_observation.schema.json` 的 `$defs.specification`
定义。每侧声明 `config`、`selection`、`manifest`、`measurement`、`prediction`、
`ground_truth`，以及 `receipt` 或 `execution` 二选一；所有值均为 evidence root
以内的相对路径。顶层声明 `protocol`、`freeze_record`、`evaluator_files`、
`metrics` 的方向和 `null_tolerance`。不能只给一个指标数值而省略这些来源。

下面的例子将已核验的 baseline receipt 和本轮 native execution 证据放入同一个
evidence root；按实际文件位置修改相对路径，并沿用冻结协议中的容忍值：

```json
{
  "schema_version": 1,
  "primitive": "history_sparse_dilation",
  "protocol": "frozen/protocol.json",
  "freeze_record": "frozen/freeze-record.json",
  "evaluator_files": {
    "future_video_recipe.py": "frozen/future_video_recipe.py"
  },
  "metrics": {"psnr_db": "higher", "global_ssim_rgb": "higher"},
  "null_tolerance": 0.0,
  "control": {
    "receipt": "control/receipt.json",
    "config": "control/config.yaml",
    "selection": "control/selection.json",
    "manifest": "control/manifest.json",
    "measurement": "control/measurement.json",
    "prediction": "control/sa_wm_rollout.mp4",
    "ground_truth": "control/gt.mp4"
  },
  "candidate": {
    "execution": "candidate/native-execution.json",
    "config": "candidate/config.yaml",
    "selection": "candidate/selection.json",
    "manifest": "candidate/manifest.json",
    "measurement": "candidate/measurement.json",
    "prediction": "candidate/sa_wm_rollout.mp4",
    "ground_truth": "candidate/gt.mp4"
  }
}
```

将诊断结果、配对 specification 及其引用的全部证据声明在外层 plan 的 artifacts
中。外层 runner 的执行 `PASS` 只说明执行和门禁通过；测量负向仍可执行 `PASS`。
命令失败或证据损坏形成 `VOID`，保留成本和证据。不能把 `PASS` 当成模型收益。

通过顶层入口运行并声明 `diagnostic_result_path` 时，结算后的索引由 scheduler
自动完成。单独使用底层 runner、或对已有终态结果补建索引时，使用公开的
`wmloop.retrieve.index.index_probe_experience` 写入检索索引，
传入终态 receipt、诊断 result、同一 archive/CAS 和结果在 receipt 中的 artifact
路径。生产与检索两端都核对证据，索引不是事实来源。一个 trial 的多个 failure
signature 或多个指标不代表多个独立重复实验。

例如，已有 `receipt`、`result` 和用户配置的目录时，直接复用公共 API：

```python
from wmloop.archive.store import ContentAddressedStore
from wmloop.retrieve.index import (
    index_probe_experience,
    probe_asset_fingerprint,
    retrieve_probe_experiences,
)

asset_fingerprint = probe_asset_fingerprint(
    receipt, cas=ContentAddressedStore(memory_root / "evidence")
)

index_probe_experience(
    database_path=memory_root / "retrieval.db",
    result=result,
    receipt=receipt,
    archive_db=memory_root / "archive.db",
    cas_root=memory_root / "evidence",
    asset_fingerprint=asset_fingerprint,
    result_artifact_path="diagnostic.json",
)
matches = retrieve_probe_experiences(
    database_path=memory_root / "retrieval.db",
    model_family=result["model_family"],
    runtime_capability=result["runtime_capability"],
    failure_signatures=result["failure_signatures"],
    primitive=result["primitive"],
    asset_fingerprint=asset_fingerprint,
    archive_db=memory_root / "archive.db",
    cas_root=memory_root / "evidence",
)
```

## 第二轮：先检索，再选择和验证

使用 `retrieve_probe_experiences` 查询相同 model family、runtime capability、
failure signature 和 primitive。第二轮决策前保存查询输出与第一轮的 trial、
receipt/result 引用；用 `exclude_archive_trial_id` 排除当前 trial。
查询当前运行自身不能证明跨轮复用。

将相同候选集合交给公共配对经验排序服务，然后交给现有 scheduler 的
`plan_candidate_batch`。比较空检索上下文和真实检索上下文时，必须保持候选、
非记忆评分、预算与 tie-break 一致。这样才能看出经验是否真正影响了选择。

候选在 `retrieval_keys.paired_query` 中显式声明所需观察。query 包含 primitive、
metric 名称和方向、protocol/context 摘要、seed、control/candidate 参数；其结构
与 observation 的 `query` 相同。第二轮可以复用第一轮的 context/protocol 身份，
但 candidate 参数必须写实际准备执行的值。公共排序入口从 archive/CAS 重核
保存的 matches，而不信任用户编辑的 JSON 投影：

```bash
uv run --offline --frozen python -m wmloop.retrieve.paired_prior \
  --batch /path/to/experiment/inputs/round-two-candidates.json \
  --matches /path/to/experiment/round-two/retrieved-matches.json \
  --archive-db /path/to/experiment/memory/archive.db \
  --cas-root /path/to/experiment/memory/evidence \
  --output /path/to/experiment/round-two/ranked-candidates.json

uv run --offline --frozen python -m wmloop.execute.experiment_scheduler plan \
  --batch /path/to/experiment/round-two/ranked-candidates.json \
  --workspace-root /path/to/VerdiWM \
  --output-root /path/to/experiment/round-two/queue
```

`retrieved-matches.json` 为 `[row.to_dict() for row in matches]` 的 JSON 数组，或
含 `matches` 数组的对象。输出的 `paired_observation_retrieval` 记录 applied/abstain、
原因、匹配 trial 和结果计数；`retrieval_matches` 保留 receipt/result 引用。
`paired_observation_prior` 是有界排序政策值，与测量的 `signed_delta` 分开；
scheduler 按 `retrieval_weight` 加入评分，不能将它解释为效果大小或成功概率。

只有 metric 名称和方向、冻结协议、context、seed 以及 control/candidate 参数
完全匹配的配对观察能影响该动作的先验。一次 dose 1 的负向观察不能给未测过的
dose 3 赋予负向效果；不同协议或 context 也不能静默套用。缺少有效观察时保持
中性，并保留没有使用经验的理由。历史只有执行状态的记录仍可用于查看来源，
不能升级成有符号收益。

执行第二轮实际选中的候选，重复冻结测量、producer、外层验证、结算和索引。
使用顶层入口时，让 scheduler 自动积累新结果，不另写一次性 finalizer 代替这一阶段。
在新的进程中再次查询，核对两轮均可从 archive/CAS 取回，第一轮原始字节没有
被第二轮覆盖。第二轮无论正向、负向或零差异都应积累；不只保存改善的样本。

## 检查闭环是否成立

完成两轮后应能回答以下问题，并提供对应文件或不可变引用：

1. 两轮哪些模型执行是新运行，哪些 baseline 是核验后的缓存？
2. 第一轮是否已经 settled，测量的方向、动作和冻结协议是什么？
3. 第二轮决策前是否检索到第一轮？空记忆和有记忆的同一批候选如何排序？
4. 选中候选是否真实执行，第二轮结果是否再次归档并能在新进程中取回？
5. 不兼容、缺失或篡改证据是否保持中性或明确拒绝？
6. 两轮是否沿用同一预算和终态事实来源，失败与未知成本是否保留？

这些证据可以确认诊断与记忆工程链路。一个 context/seed 的测量不生成正式
EffectMemory、transfer license 或质量提升结论；完整模型优化仍须通过原有的
onboarding/conformance、目标验证、held-out、多 seed 和保护指标门禁。
