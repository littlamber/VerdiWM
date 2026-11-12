# VERDI 方法到 VerdiWM 工程映射

VERDI 方法稿（完整题目与引用状态见 `docs/CITATION.md`） 的方法部分将系统定义为四阶段闭环：

```text
Characterize → Retrieve → Validate → Accumulate
      ↑                             │
      └── nearby contradiction + held-out probe admission ──┘
```

在 VerdiWM 中，`Characterize` 生成 portrait-bound Optimization Fingerprint，`Retrieve` 通过 IRG 和已结算 archive 排候选并生成跨域检索请求，`Validate` 在冻结 verifier 下执行 paired target experiment，`Accumulate` 将 settled outcome、support、uncertainty、cost、lineage 和 implementation receipt 写入 archive。候选、检索和诊断都没有 verdict authority。

方法沉淀成三类稳定对象：

1. `Model IRG`：共享 inference-time probe 的响应向量、协方差、support mask、诊断轴和可用 hooks。它用于比较“相同干预是否产生相似响应”，不等价于架构相似度或全局能力。
2. `EffectMemory`/settled archive：保留 confirmed、null、rejected 和 interaction，并把 transfer certificate、适用条件和反条件作为派生投影。source 证据不可被 target 结果覆盖。
3. `Probe evolution`：由可信的 nearby opposing effects 触发，通过 CPBE 产生新 probe，必须以 held-out selector replay 改善和无风险回归作为准入条件。

跨域检索先从 IRG 热点生成带 `evidence_refs` 和 `ceiling_boundary` 的瓶颈假设，再生成 temporal、action、noise、context、appearance、contact 等 domain lens。文献/代码是 untrusted staging data；候选必须回到 target-side compile、screen 和 frozen confirmation，检索本身不写效果。

本轮工程加固包含：

- selector replay 要求 baseline/candidate 的 target、seed 完全配对，同一 target/operator 的 label 不变；覆盖缺失、seed 重复和非法行立即 fail closed；
- retrieval index 在写入和读取时同时核对 CAS 内容、receipt artifact reference 和 Archive SQLite settlement binding；
- `EffectRecord` 强制有限的 lower/mean 区间和真实布尔 validity gates，`EffectMemory` 采用拥有式快照，避免调用方事后修改证据；
- IRG collision 与 probe admission 阈值拒绝 NaN/Infinity，防止数值绕过门禁；
- 新增 CPU-only negative tests 覆盖篡改结果、缺失 receipt、跨 replay 不可比和不安全阈值。

详细权威约束见仓库根目录 [`AGENTS.md`](../AGENTS.md)；总体模块说明见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 实现范围与仍需验证的条件

`ArchiveStore` 保存终态运行账本，其中可能包含 VOID/执行失败；账本可追溯不等于
已确认的 operator outcome。`EffectRecord` 的字段校验、内存复制和引用保留，也
不独立证明某个引用包含 held-out confirmation。上游 verifier 和具体效果导入器
仍负责检查证据来源与确认状态。

现有 `evaluate_selector_probe_admission` 是 ACWM 的特定策略，含八环境测量和
locality 校准覆盖门槛。本轮修复其比较完整性；它不自动证明嵌套划分无泄漏、
完整的 calibration error 改善或论文所有 probe-growth 条件。推广到其他领域应
显式版本化配置与回放协议，不能把“八环境”作为通用框架要求。

`retrieve/index.py` 按 model family/runtime capability 匹配诊断经验，
`geometry/model_irg.py` 做跨骨干响应比较，`retrieve/irg_guided_discovery.py`
负责跨学科文献查询。这三条路径用途不同；本轮索引加固未更改跨骨干距离公式。
