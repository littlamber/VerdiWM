# VerdiWM

[English](README.md)

VerdiWM 是面向世界模型的证据约束研究循环。它把用户目标编译成冻结的评测契约，诊断模型的失效机制，将有边界的干预原语编译到目标模型，按递进成本验证候选，并沉淀成功、无效和有害经验。

项目面向受约束的递归自改进（RSI）基础设施，但不宣称已经实现无限制的自我进化。候选生成不能绕过预算、独立验证、CAS/Archive 溯源、迁移证书或晋升门禁。

## 安装与自检

CPU 控制面要求 Python 3.10。默认安装不包含可选的 Torch 运行时测试，也不需要模型权重或 GPU。

```bash
python -m pip install uv
uv sync --group dev
uv run verdiwm doctor
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

`verdiwm doctor` 会检查 Python 版本、核心 schema、适配器配置和轻量机制本体；缺少或损坏任一必需项都会返回 `blocked`。portrait-first 示例会在 CPU 上依次验证 Goal IR、Model Capability IR、探针指纹、Model Portrait readiness 和能力缺口规划；它只验证控制面契约，不声明模型效果。

## 启动任务

用户入口只要求模型、数据、目标和预算。适配器配置负责解析评测器、诊断探针、运行时资产与控制面契约。

```bash
uv run verdiwm run \
  --model /path/to/model-checkout \
  --data /path/to/data \
  --goal "提升长时域动作条件预测" \
  --budget 8gpu-hour
```

任务支持查询、取消和隔离复现：

```bash
uv run verdiwm status CAMPAIGN_ID
uv run verdiwm cancel CAMPAIGN_ID
uv run verdiwm reproduce CAMPAIGN_ID
```

## 轻量核心

默认运行路径只保留下一步决策所需的信息：

1. 将目标编译为指标、held-out 协议和资源上限。
2. 诊断失效并生成有界 evidence capsule。
3. 优先复用已经结算且绑定 receipt/CAS 的经验。
4. 仅在冷启动时运行受预算约束的机制检索，默认使用 `light` 模式。
5. 依次执行 screen、官方 gate 和确认实验。
6. 将所有正向、无效和有害结果写入经验记忆。
7. 只有确认有效且迁移证书全部通过的记录才能成为 `licensed_prior`。

完整 Evidence Graph、IRG 资产和跨领域检索属于按需审计与科学分析层，不是正常循环的强制运行状态。

模型接入、实验编译和共享经验分别使用 Capability IR、Experiment IR 和
Evidence IR。共享 IR 只保留语义身份、内容哈希、能力、有效域和授权边界，
不携带仓库、checkpoint 或数据文件路径；本地路径只存在于可复现的 sidecar
与执行 receipt 中。插件由版本化 manifest 注册，新增无关插件不会改变既有
workflow 的能力摘要。详细边界见 `docs/INTERMEDIATE_REPRESENTATIONS.md`。

## 当前边界

- 控制面、适配器契约、渐进式实验调度、证据归档和失败闭合迁移已经实现。
- 轻量 Kernel、manifest 插件注册表和三类跨模型 IR 已实现并通过 CPU 篡改与路径独立性测试。
- ACWM-Phys 提供一个完整性校验过的最小闭环证据包。
- Ctrl-World 与 Cosmos3 包含局部响应图、反例和正确 abstain 的迁移结算。
- 跨模型自动迁移、新骨干上的自主 atlas 进化和多训练种子因果复现仍是研究工作，不作为已完成能力宣传。

完整实现状态与证据边界见 [英文 README](README.md)、[架构说明](docs/ARCHITECTURE.md)、[方法到代码映射](docs/METHOD_TO_CODE.md) 和 [可复现性说明](docs/REPRODUCIBILITY.md)。

### 两个执行端与 LLM API 接入

开放方法链路的第一个执行端本质上是一个外置 LLM API broker：

- `wmloop/execute/json_llm_service_broker.py` 适合任意受信任的 HTTPS JSON 网关；
- `wmloop/execute/openai_compatible_llm_broker.py` 适合 OpenAI-compatible 的
  Responses API 或 Chat Completions API；新手可以只填写 `--base-url`，broker
  会自动补上 `/v1/responses` 或 `/v1/chat/completions`。

部署时显式提供 endpoint、模型名和 token 环境变量即可接通，也可以使用权限为
`0600` 的 `--token-file` 传入一个独立认证文件，不需要为新方法
预先注册论文 ABI。两个 broker 都会限制请求/响应大小、拒绝缺少凭据、只允许
HTTPS（本地开发可用 localhost HTTP），并把结果原子写入既有 task-response
契约。

不能直接复用当前 Codex bridge 的“中转站”。Codex 会话使用的内部 endpoint、
session 状态和凭据不是本项目的 API，也不会被 broker 自动读取；不要从
`LARK_*`、`LARKSUITE_*` 环境、keychain 或 bridge 配置中抓取它们。只有在部署方
另外提供一个可访问的、兼容 OpenAI 协议的 endpoint 和 token 时，才可以使用同一
供应商或同一模型。

认证文件示例（不要把 Codex/bridge 的内部 auth 文件或 session token 直接复用）：

```bash
umask 077
printf '%s\n' '新建的服务 token' > ~/.config/verdiwm/llm.token
python wmloop/execute/openai_compatible_llm_broker.py request.json response.json \\
  --base-url https://ai.example \\
  --model gpt-5.5 \\
  --token-file ~/.config/verdiwm/llm.token
```

`--token-file` 会拒绝软链接、非普通文件、超过大小限制或权限不是 owner-only
的文件；token 不会写入任务响应或 receipt。环境变量和文件同时提供时，以文件为准。

要实现“配置一次后直接运行”，可以使用项目默认配置目录：

```toml
# ~/.config/verdiwm/config.toml
[llm]
base_url = "https://ai.example"
model = "gpt-5.5"
api_style = "responses"
reasoning_effort = "xhigh"
token_file = "auth"
```

将新建的服务 token 放到 `~/.config/verdiwm/auth`（权限 `0600`），之后适配器命令
只需使用 `verdiwm-llm-broker request.json response.json`。也可以通过
`VERDIWM_CONFIG=/path/to/config.toml` 指定配置位置。这个 `auth` 是 VerdiWM
自己的 bearer-token 文件，不是 Codex 的 `auth.json`，不会读取或解释后者。

第二个执行端是独立的 candidate sandbox broker，代码在
`wmloop/execute/candidate_sandbox_broker.py`，负责在 Docker/Podman 隔离 worker
中运行新方法的 `calibrate`、`train`、`infer`。它会只读挂载候选目录，单独挂载
输出目录，关闭网络、丢弃 Linux capabilities、启用 `no-new-privileges`，并且不
转发父进程环境或任何凭据。基础镜像见 `docker/candidate-sandbox/`；部署时应基于
它构建包含目标依赖的派生镜像，并使用 digest 固定镜像版本。

普通 shell 进程不能替代生产环境的 sandbox；没有 Docker/Podman、digest 固定的
镜像或内核授予的 GPU lease 时，系统应保留候选并阻断进入正式 screen/confirm。
部署配置只需要把 `candidate_sandbox_broker.py` 作为 `calibration_adapter`
命令，并显式传入 `--workspace-root`、固定的 `--image` 和 `--runtime docker`
（或 `podman`）；`credential_environment_keys` 必须为空。请求本身只能选择
`calibrate`、`train`、`infer` 和已验证的候选执行契约，不能选择镜像、宿主路径或
凭据。完整示例见 `docker/candidate-sandbox/README.md`。

没有 Docker 的用户可以把 adapter 指向
`wmloop/execute/candidate_process_broker.py`。这个一等 `worktree_process` 后端
支持 `calibrate`、`train`、`infer`，会复制候选工作区、清空凭据环境、按内核传入
GPU 设备并限制进程资源，也可以完成本地 screen/confirm。它与 container 后端共用
Method IR、执行契约和 receipt；区别只体现在 `assurance_level`。由于它不能强制隔离
网络和整个宿主文件系统，社区 promotion 前需由独立 container worker 复验，本地
开发和知识图谱消费不依赖 Docker。

## 系统效用验收

在继续扩大 GPU 实验前，先用公开证据运行 CPU 审计：

```bash
uv run verdiwm audit \
  --repo-root . \
  --output-root results/reports/system-utility-audit-v1
```

审计将可用性、渐进式成本收益、选择器质量、Ctrl-World 局部图、跨骨干运行时可移植性和正式迁移确认分开报告。`partial` 是有意的研究状态：它表示控制面可用，但尚未把局部或运行时证据误报成模型效果。下一轮受控实验计划是 `configs/experiments/ctrl_world_experience_utility_canary_v1.json`。

### 实验工程与训练规模

新的实验还必须维护独立的 engineering manifest、README、入口脚本、CPU
smoke 测试和可复现命令。用 `verdiwm lint-experiment` 检查源码 revision、
脏工作树策略、数据冻结、held-out 协议和必需产物；用
`verdiwm plan-training` 从 train/validation sample manifest 推导数据量、
episode 多样性、effective batch、steps/epoch、阶段更新上限和 checkpoint
评测阶梯。缺 validation 或 episode 多样性不足时计划会 fail closed，不能
靠随意增加 epoch 继续跑。详见 `docs/EXPERIMENT_ENGINEERING.md`。

## 发布

正式上传魔塔社区或 GitHub 前，必须从干净且全部已跟踪的 Git checkout 运行：

```bash
scripts/ci/release_preflight.sh --output-dir dist/verdiwm-release
```

上传生成的 `dist/verdiwm-release/repository/`，不要直接上传开发工作区，并保留其中的 `RELEASE_AUDIT.json` 与 `MANIFEST.sha256`。公开树可独立构建，但会排除本地部署配置、状态目录、模型权重、数据集、凭据和宿主机路径；这些绑定由实际部署在本地提供。完整步骤见 [发布清单](docs/RELEASE_CHECKLIST.md)。

## 训练配方研究与证据边界

公开世界模型训练数字已整理到
[`configs/retrieval/world_model_training_recipes_v1.json`](configs/retrieval/world_model_training_recipes_v1.json)，
说明与来源表见
[`docs/TRAINING_RECIPE_RESEARCH.md`](docs/TRAINING_RECIPE_RESEARCH.md)。这些条目默认是
`shadow_only` / `ranking_only`：可以检索和比较，不能直接变成 GPU 训练命令。

```bash
verdiwm training-recipes
verdiwm training-recipes --recipe-id genie_dynamics_pretrain_v1
```

只有经过目标 backbone 本地 screen、独立验证集和长程 rollout 评估，并明确写入
`local_validated` 或 `reusable_optimization_memory` 的 profile，才允许覆盖训练规模
规划。这样能区分论文明确披露、官方配置、推断数字和未披露字段，也避免把大实验室的
batch/step 直接套到 Ctrl-World 的小数据子集上。

## 许可证

VerdiWM 源代码采用 [Apache License 2.0](LICENSE)。外部数据集、模型权重和上游仓库仍遵循各自许可证。
