# 实验产物规范（Artifact Convention）

本规范约束 VerdiWM 每轮研究循环产生的工件，目标是：**人看得懂、图谱投得清、机器可校验**。
它只增加约定，不改变任何验证权威——契约校验（contracts/schemas）仍是唯一强制门槛。

## 1. 命名

### 1.1 artifact_type

- 形如 `verdiwm-<域>-<名词>`，全小写 kebab-case，例如 `verdiwm-mechanism-relation`。
- 名词用**语义角色**命名（`settled-trial`、`transfer-certificate`），不用实现细节
  （禁止 `output2`、`final_v3` 这类名字）。
- 临时/中间产物以 `verdiwm-scratch-` 前缀开头，明示不得进入证据链。

### 1.2 身份字段（campaign_id / trial_id / relation_id / probe_id …）

- 用**人可读的语义 slug**：`<对象>-<限定词>[-vN]`，如 `acwm-cloth-long-horizon-a1`。
- 身份字段里**禁止内嵌裸哈希**。需要内容寻址时采用 `<名词>-<hash24>` 形态
  （如 `mechanism-relation-<sha256前24位>`），哈希永远放在尾部。
- 自动生成的探针/候选必须带语义前缀：`cpbe-residual-<hash8>` 而不是裸 `0f7e5f10f9`。
- 版本后缀只用 `-vN`，禁止 `-final`、`-new`、`-copy`。

### 1.3 文件名

- 单个工件：`<名词>[-<限定词>].json`（如 `screen-manifest.json`）。
- 追加型流水：`<名词>.jsonl`，只增不改。
- 同一目录下文件名即阅读顺序：用 `01-`、`02-` 前缀表达阶段，而不是依赖时间戳。

## 2. 内容

**模型身份红线**：任何引用模型的工件（`model_ref`）必须同时携带 `model_family` 或
`model_name` 语义字段。goal id（如 `g1_long_horizon_ladder_v1`）是目标协议名，**不是模型名**，
禁止拿它充当模型身份。只留一个 `cas://` 哈希的产物视为不合规——图谱和人都无法知道
"这是基于 ctrl-world 的哪个权重"。

每个工件至少包含：

| 字段 | 要求 |
|---|---|
| `artifact_type` | 必填，见 §1.1 |
| `schema_version` | 必填，整数，破坏性格式变更才递增 |
| `state` / `status` | 必填，取值来自封闭枚举（`ready`/`settled`/`blocked`…） |
| 身份字段 | 必填，见 §1.2 |
| `created_at` | 必填，UTC ISO-8601 |
| `claim_boundary` | 必填一段**人话**：这份产物能证明什么、不能证明什么 |
| 证据引用 | 只允许可移植引用（`cas://`、`urn:`、`sha256:`）；**禁止本地绝对路径进入证据字段**（本地路径放独立的 `source_path` 类展示字段） |

推荐另带：`goal_id`（它服务的目标）、`parents`（上游工件引用）、`summary`（一句话人话结论，
供 UI 直接展示）。

## 3. 图谱可读性规则

证据图谱是产物的投影，产物规范直接决定图谱是否看得懂：

1. **显示名来自语义字段**：UI 展示 `artifact_type` 的人话映射 + 序号
   （"指纹测量 #10"），不展示 `artifact_type:identity:ordinal` 原始 key。
2. **每轮循环的产物必须能回答四个问题**：改了什么（primitive/candidate）、
   在哪改（environment/backbone）、结果如何（verdict/outcome）、证据在哪（receipt/evidence_refs）。
   缺任一字段，图谱上就应该显示为"未分类"而不是静默省略。
3. **负证据与弃权也要落成工件**：`rejected_at_screen`、`operational_failure`、
   迁移弃权各自有独立 artifact_type，不允许只写进日志。
4. **跨组不可比性必须显式**：协方差未观测、基线不兼容这类情况，用 `claim_boundary`
   和弃权状态表达，不靠读者推断。

## 4. 生产者清单（每次循环结束自查）

- [ ] 新产物有语义化身份字段，无裸哈希、无 `-final`/`-new` 后缀
- [ ] `claim_boundary` 一句话说得清这份证据的边界
- [ ] 证据字段里没有本地路径
- [ ] 失败/弃权也落了工件
- [ ] 用 `verdiwm-evidence-graph` 重建图谱后，本轮产物能被正确分类

## 5. 机器校验：verdiwm-artifact-lint

`verdiwm-artifact-lint <目录>` 对产物做只读规范检查，退出码非零即存在 error 级问题：

| 级别 | 规则码 | 含义 |
|---|---|---|
| error | `MISSING_ARTIFACT_TYPE` | 产物没有 `artifact_type`（即使只是 `failure_report.json` 这类裸 JSON 也算产物） |
| error | `MODEL_REF_WITHOUT_NAME` | `model_ref` 是裸 `cas://`/`sha256:`/十六进制引用却没有 `model_family`/`model_name`（模型身份红线） |
| error | `IDENTITY_EMBEDS_BARE_HASH` / `IDENTITY_BAD_SUFFIX` | 身份字段内嵌裸哈希或用 `-final`/`-new` 后缀 |
| error | `LOCAL_PATH_IN_EVIDENCE` | 证据字段混入本地路径 |
| warning | `MISSING_CLAIM_BOUNDARY` / `MISSING_STATE` / `ARTIFACT_TYPE_NONCONVENTIONAL` | 可读性问题，不阻断投影 |

**error 级产物不会进入证据图谱的默认投影**（`/api/graph?clean=1`，前端默认视图）；
完整投影仍可通过 `/api/graph`（或前端关闭"合规视图"）查看。lint 只 advisory，
不改变任何验证权威。
