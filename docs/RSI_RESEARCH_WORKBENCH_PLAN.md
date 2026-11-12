# VerdiWM RSI Research Workbench Plan

本文档把 VerdiWM 的长期目标转成可执行的系统路线：用户提供模型、数据、
研究目标和预算，系统自动发现可用能力，提出并筛选 repair，执行冻结评估，
保存正向、无效和有害证据，并用这些证据改善下一轮选择。

目标不是让某个 IRG 距离在所有任务上都胜出。目标是让整个研究闭环在同一预算
下更快找到有效改进，并且在无法预测时安全 abstain。IRG 是可替换的诊断特征源；
如果 raw-response 或其他 predictor 更好，路由器应自动降级而不是继续包装失败的
几何假设。

## 1. 可验证的系统命题

VerdiWM 的核心命题应写成：

> 在相同候选池、评估器、随机种子和实验预算下，VerdiWM 的上下文化候选选择器
> 能降低首次有效 repair 的成本、selection regret 和负迁移率；加入已结算经验
> 后，这些指标在新的 outer target 上继续改善。

这个命题比“IRG 表示了模型缺陷”更窄，也更容易被证伪。每个组件都必须回答
一个独立问题：

| 组件 | 必须证明的事情 |
|---|---|
| Model Adapter | 新模型的入口、权重、数据和可修改 hook 能被可靠识别 |
| Evaluator | 目标指标、切分、阈值和 verifier 能被冻结并复现 |
| Failure Analyzer | 能把指标失败转成可检查的 failure state 和机制假设 |
| Probe Provider | 干预响应能增加诊断信息，而不是只增加响应维度 |
| Repair Effect Predictor | 能预测具体 primitive、dose、schedule 的目标侧效果 |
| Policy Router | 能在候选之间选择、screen、confirm 或 abstain |
| Experience Store | 正向、无效和有害结果能被正确检索并改善后续选择 |

## 2. 系统闭环

每次研究都应经过下列有状态对象，而不是在一个大 JSON 中混合假设、预测和
结论：

```text
UserGoal
  -> ModelPortrait
  -> CapabilityProfile
  -> EvaluatorContract
  -> FailureState
  -> MechanismHypothesis
  -> RepairCandidate
  -> RepairPrediction
  -> ScreenTrial
  -> GateTrial
  -> ConfirmTrial
  -> SettledEvidence
  -> ExperienceEntry
  -> next-round policy
```

推荐的最小字段如下：

```json
{
  "repair_candidate": {
    "primitive": "self_forcing_finetune",
    "dose": 0.1,
    "schedule": "late_rollout_weighted",
    "required_hooks": ["H3"],
    "context_digest": "..."
  },
  "prediction": {
    "expected_delta": 0.018,
    "interval": [-0.004, 0.041],
    "support_state": "partial",
    "decision": "screen",
    "predictor_id": "ridge-v1"
  },
  "evidence": {
    "state": "settled",
    "effect_class": "positive",
    "evaluator_digest": "...",
    "split_id": "outer-target-03",
    "provenance": ["screen/123", "gate/456"]
  }
}
```

`sensitivity`、`vulnerability` 和 `repairability` 必须分开：probe 响应只能说明
模型对干预敏感；failure state 才能说明某个方向存在风险；只有独立 target-side
repair label 才能说明命名的 repair 可行。

## 3. 真正值得做的创新机制

创新应来自“如何减少昂贵试错并保持证据边界”，而不是给已有距离换名字。以下
机制可以组合成 VerdiWM 的独到贡献，每个机制都有可单独验收的指标。

### 3.1 Support-aware repair routing

预测器同时输出效果区间和训练支持状态。候选超出 source support、hook 不兼容、
区间跨过零或校准失败时，系统不强行排序，而是生成低成本 screen 或 abstain。

```text
high LCB + in-support       -> confirm candidate
positive mean, interval > 0 -> screen candidate
interval crosses zero       -> low-budget screen / defer
out of support              -> abstain and request probe or label
incompatible hook           -> reject before execution
```

验收：在相同预算下，负迁移率不升高，且 abstention precision 高于固定阈值策略。

### 3.2 Contextual repair-effect prediction

不要直接把 IRG 距离当作 repair 效果。用固定 schema 的小模型预测：

```text
(target context, failure state, intervention response, uncertainty,
 capability compatibility, primitive, dose, schedule)
    -> metric delta and interval
```

第一版使用标准化 ridge、bootstrap 或 conformal interval，训练只使用 inner source
环境。IRG 作为结构化特征源，与 raw response、baseline metrics、task context 和
capability features 并列。outer target 的 repair labels 只能用于最终评估。

验收：与 random、metadata、raw-response 和 IRG-nearest-neighbor 在完全相同候选池
和预算下比较 top-1、regret、首次有效成本、校准和 harmful selection。

### 3.3 Evidence-conditioned active probing

下一条 probe 不由响应幅度决定，而由预期信息收益决定：

```text
acquisition = expected regret reduction + expected coverage gain
              - execution cost - nonlocality penalty - redundancy penalty
```

只有出现可复现的近邻冲突，才允许生成 successor probe。successor 必须在新的
outer context 上减少 repair-selection regret；仅仅降低 locality residual 不算成功。

### 3.4 Failure-aware experience memory

经验条目必须包含 `positive`、`null`、`harmful` 三种 effect class，以及适用范围、
支持区域、评估器摘要、split、代码版本和 provenance。检索器同时返回支持证据和
反例证据，路由器根据二者更新预测区间，而不是只累积成功案例。

验收：加入完整 memory 后，新的 outer target 上首次有效成本下降；只加入正例时若
负迁移上升，应被实验明确显示出来。

### 3.5 Budgeted screen-gate-confirm policy

把每个 repair 的成本写入策略，而不是只优化离线排序：

```text
cheap diagnostic -> bounded screen -> frozen gate -> independent confirm
```

策略优化目标可以是：

```text
expected utility = probability of useful repair * gain
                   - screen cost - confirmation cost
                   - risk penalty for harmful transfer
```

这使系统真正优化“更快找到有效改进”，而不是优化一个与用户目标脱节的相似度。

## 4. 模块边界

建议在 codebase 中固定以下协议。任何 predictor、IRG 版本或 LLM proposer 都通过
这些协议接入，避免单一方法绑死整个系统：

```text
ModelAdapter
Evaluator
FailureAnalyzer
ProbeProvider
RepairGenerator
RepairEffectPredictor
Selector
ExperimentExecutor
EvidenceStore
ExperienceRetriever
PolicyRouter
```

其中：

- `ProbeProvider` 只产生诊断特征和 coverage receipt；
- `RepairEffectPredictor` 只预测候选效果和不确定性；
- `Selector` 只在固定候选和预算下排序或 abstain；
- `ExperimentExecutor` 只执行已经通过 capability 和 evaluator 门禁的候选；
- `EvidenceStore` 保存不可变 receipt，不得把 screen 结果升级为 settled verdict；
- `PolicyRouter` 根据 replay 结果选择 IRG、raw-response、contextual predictor 或
  random fallback。

IRG 失败时，系统仍可运行：

```text
IRG unavailable or worse than raw-response
    -> keep IRG as diagnostic evidence
    -> route with raw-response or contextual predictor
    -> record negative evidence
    -> schedule probe/predictor improvement only on a new outer split
```

## 5. 分阶段路线图

### Phase A: RSI-0，自动接入

目标：让陌生模型进入系统的成本可测量地下降。

交付：

1. `ModelPortrait` 和 `CapabilityProfile` 的统一 schema；
2. evaluator discovery、runtime check 和 blocker receipt；
3. 一条只读 onboarding 命令，能生成研究计划但不启动 GPU；
4. 5 个不同结构的模型接入 replay。

门槛：计划生成成功率至少 90%，非法输入 fail closed，接入时间和人工 baseline
都必须记录。

### Phase B: RSI-1，候选决策

目标：在一个任务族内证明系统能减少无效实验。

交付：

1. contextual repair-effect predictor；
2. random、metadata、raw-response、IRG 和 predictor 五路对照；
3. outer-split leakage guard；
4. support-aware routing 和自动降级；
5. screen-gate-confirm budget policy。

门槛：相对 random 降低 selection regret 和首次有效成本；相对 raw-response 至少
在一个预先冻结的 outer task 上有稳定收益；不能以调参后的同一 outer split 作为证据。

### Phase C: RSI-2，经验驱动迭代

目标：证明保存经验会让下一轮研究变快。

交付：

1. positive/null/harmful experience entries；
2. 检索、预测、执行和结算的 provenance 链；
3. memory-on 与 memory-off 对照；
4. counterexample-driven probe evolution；
5. 新模型族和新任务上的迁移评估。

门槛：加入经验后 outer target 的首次有效成本下降，负迁移率不增加，且失败经验
不会被错误提升为成功经验。

## 6. 最小可行实验协议

先不要大规模 GPU。使用一个任务族、6 个 source environment、2 个 outer target、
4 个 repair primitive 和多个 dose/schedule，冻结以下内容：

```text
source/target split
candidate pool
evaluator and thresholds
random seeds
screen and confirm budgets
feature schema
predictor hyperparameters
```

每个 outer fold 都输出：

```text
source digest
target id
training fold digest
feature schema digest
candidate pool digest
predictor artifact digest
selection result
```

比较指标：

| 指标 | 目的 |
|---|---|
| top-1 positive | 第一次选择是否有效 |
| selection regret | 是否错过已有正例 |
| first-positive cost | 找到有效修复花了多少预算 |
| negative transfer | 是否选择了有害修复 |
| abstention precision | 拒绝是否真的减少风险 |
| interval coverage | 不确定性是否可信 |
| accepted coverage | 有多少目标仍能做出有证据的选择 |
| probe cost | 诊断本身消耗了多少预算 |

任何模型或 predictor 只有在 outer replay 完成后才能进入 routing；局部响应更大、
locality residual 更低或代码执行成功，都不能单独构成有效性结论。

## 7. 当前仓库的下一步实现顺序

1. 保留现有 IRG replay、random expectation 和自动 routing。
2. 已新增 `RepairCandidate`、`RepairExample`、`RepairPrediction` 和
   `TrainingScope` 的 schema。
3. 已实现 CPU-only ridge predictor，输出模型 artifact、feature schema digest、
   training fold digest 和 calibration receipt。
4. 将 predictor 接入 selector replay，增加 outer target label leakage 检查。
5. 把 positive/null/harmful 经验写入现有 evidence/archive 接口。
6. 实现 memory-on/off 和 predictor-on/off 的 matched-budget replay。
7. 只有当 regret 或首次有效成本在 outer target 上改善后，才投入真实 GPU probe
   evolution 和跨 backbone 实验。

## 8. 失败时仍然有价值的结果

如果 IRG、contextual predictor 和 probe evolution 都没有超过 raw-response，系统仍
应输出可复用结论：

- 哪些 response feature 与 repairability 无关；
- 哪些机制需要 task-specific causal probe；
- 哪些 context 缺失导致无法预测；
- 哪些区域应该 abstain；
- 哪些经验不能跨模型迁移。

这保证 VerdiWM 的价值来自可验证的研究效率、失败边界和持续改进能力，而不是依赖
某个永远不能失败的单一算法。
