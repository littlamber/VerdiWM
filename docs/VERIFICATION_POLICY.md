# 正式验收策略与历史协议兼容

`wmloop.verify.judge` 只有在只读评估、held-out/扩展 horizon、diff audit、
动作跟随门禁、完整指标及重复观测都通过后才返回 `ACCEPT`。任何已声明保护指标
的退化都会保留在 `negative_metrics` 中，主指标改善不能抵消它。动作跟随失败时，
仍将可计入收益的 `delta_m_ver` 清零，同时保留实际观测到的负向指标。

重复观测门禁至少需要三个观测值，使用样本标准差计算标准误和双侧 95% Student-t
区间。只有主 utility 的下界严格大于预声明 `minimum_effect` 才通过；等于阈值不
通过。表内自由度向下取整，超过表上限仍使用最后一项，区间因此偏保守。此计算
依赖独立配对差值和近似正态的均值抽样分布；同一实验单位的重复帧必须先聚合。
统计公式不证明数据独立、control/candidate 的真实配对或多重比较校正。

## 新协议的声明与绑定

新目标使用 `configs/schemas/goal_spec_v2.schema.json`，并显式声明
`schema_version: 2`、`artifact_type: wmloop-goal-spec` 和 `claim_boundary`。
`eval_protocol.acceptance_policy` 必须包含：

```json
{
  "metric_directions": {
    "primary_utility": "maximize",
    "protected_error": "minimize"
  },
  "minimum_effect": 0.5,
  "confidence_interval": "student_t_two_sided_95"
}
```

其中 primary metric 名称来自目标已有的 `primary_objective`，必须出现在方向表。
方向表还必须列出每个正式保护指标。执行端提供原始 candidate-minus-control
差值，judge 按方向转换成改善为正的值；保护指标只允许不退化，缺少任何声明指标
都不能接受。`minimum_effect` 使用该主指标的原始单位。`seeds_accept` 至少包含
三个互不相同的 uint32 seed，必须与实际重复观测的 seed 集合完全相同。主指标的
汇总差值也必须等于这些重复观测的均值。

控制面在 proposal/provider 和 executor 运行前调用
`verification_policy_from_goal`，得到包含完整目标 digest 的不可变策略。
`ResearchLoop` 固定第一次编译的策略，后续轮次继续核对完整目标；外部传入策略
也必须等于该目标实际编译出的规则。执行端不能通过修改目标、关闭动作跟随门禁
或降低阈值改变验收。动作跟随阈值始终取冻结目标的 `action_following_gate.tau_af`。

`verification_policy.schema.json` 定义策略快照，`verdict_v2.schema.json` 定义新判定。
判定记录 Student-t 区间、样本数、阈值、实际 seed、策略快照及策略 digest。
训练评估 smoke 在执行前将完整目标和策略放入 CAS，receipt 引用这两个真实对象。
该 smoke 目前只支持单一 primary metric 加独立动作跟随测量；声明其他保护指标或
重复数与冻结 seed 数不一致时，在 GPU 执行前报
`M3_TRAINING_EVAL_SMOKE_POLICY_EVIDENCE_UNSUPPORTED`，不会补造证据。

## 历史兼容与 campaign 版本边界

原 `goal_spec.schema.json` 和 `verdict.schema.json` 的字节保持不变。
`validate_document` 对未带版本的历史目标/判定继续读取旧 schema，仅对显式 v2
文档选择新 schema；不支持的版本会报错。历史目标不能偷偷加入新策略字段。

无策略的历史证据接口仍可调用，新 judge 会执行保守的零阈值 Student-t 检查和
保护指标检查。其 `ACCEPT` 只表示兼容模式门禁通过；判定显式记录
`seed_binding=unverified_compatibility`、空策略及 claim boundary，不能声称验证了
预声明目标阈值或匹配 seed，也不能升级为论文所需的新协议正式准入。

归档和验收审计共用冻结绑定检查：新 v2 判定必须同时绑定完整策略快照及其
digest、预声明 seed 集合、统计阈值、样本数和 receipt 中的目标/策略 CAS 引用。
缺少绑定的兼容 `ACCEPT` 仍可保留为 exploratory 终态记录，但不更新正式收益
cell，也不能通过正式模型质量审计。v2 `INCONCLUSIVE` 同样只保留探索性记录。
完整绑定的 `REJECT` 可作为 null/负向结果结算，调度 credit 取主指标改善值与零
的较小值；保护指标退化时即使 utility 为正也不会获得正 credit。原 verdict 的
全部指标和负向事实保留在 CAS，不用调度 credit 覆盖观测。未带版本的历史
归档及报告保留原有读取语义，未重写或重新赋予新协议权威。

修改 judge 的行为属于 verifier 版本变化。历史 campaign 必须使用其冻结版本的
verifier 重放，不能在原 constitution ID 下改 hash 或重冻来接受新实现。新增
campaign 应使用新的目标/constitution ID，并在执行前冻结新目标、split、evaluator、
probe、新 schema、`wmloop/contracts.py`、`wmloop/verify/judge.py`、
`wmloop/verify/acceptance_policy.py`、`wmloop/verify/statistics.py` 及控制面策略编译入口。
这些文件构成新的验收权威依赖，单独冻结入口而遗漏 helper 不能声称完整锁定策略。
本次实现提供新协议接口，没有替换任何活动目标或历史 freeze。
