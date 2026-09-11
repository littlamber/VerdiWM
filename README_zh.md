# VerdiWM

VerdiWM 是一个面向世界模型的意图驱动研究工作台：用户提供模型、数据和
目标，系统负责解析适配器与评估契约，执行有边界的实验，并为每个决策保留
可追溯证据。

面向用户的 CLI 名称是 `verdi`；`verdiwm` 仍作为兼容入口保留。后文的完整
命令都可以把 `verdiwm` 替换成 `verdi`。

控制面不绑定某一个模型，也不包含模型权重、数据集、API 密钥或 GPU 运行时；
这些资产由实际部署的用户自行准备。

## 快速开始

环境要求：Python 3.10 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/littlamber/VerdiWM.git
cd VerdiWM
python -m pip install uv
uv sync --group dev
uv run verdiwm doctor
```

直接运行 `verdi` 会进入一个轻量的交互会话。输入 `/` 并回车打开命令面板；在
支持 raw TTY 的真实终端中，面板会实时响应字符筛选、上下方向键、Tab、Enter
和 Esc。输入 `/res`、`/sta` 等前缀可以筛选命令；还会启用 Python 标准库的
历史记录和 Tab 补全（可用 `NO_COLOR=1` 关闭颜色）：

```text
verdi> /
verdi> /start
模型目录 [/path/to/model]:
数据目录 [/path/to/data]:
研究目标（例如：提升分钟级长程一致性）:
verdi> /plan "提升分钟级长程一致性"
verdi> /run --plan ./.verdiwm/research-plan.json --confirm
verdi> /status
verdi> /exit
```

首次输入 `/start` 或 `/setup` 时，如果当前目录有 `model/` 和 `data/`，向导会
自动填入路径；用户只需确认路径并输入一句目标。项目配置生成后，`/start` 会继续
生成研究计划。常用命令支持短写法，例如 `/plan "目标"` 等价于显式的
`/plan --goal "目标"`，`/run PLAN` 会补全为 `--plan PLAN`。

命令执行时会先显示“执行 …”，随后将状态、计划路径、campaign 和 blocker 等
关键字段整理成可读结果卡，不要求用户阅读原始 JSON。交互会话中的自然语言只会
生成下一步计划提示，不会静默导入模型或启动 GPU。`/run` 仍要求显式 `--confirm`，
并继续执行研究计划、评测器和证据门禁。脚本、
CI 和管道环境不会进入会话，继续使用普通命令帮助；`verdiwm` 兼容入口也保持
原有行为。需要强制进入会话时可运行 `verdi chat`（或 `verdi shell`）。

`doctor` 会检查已安装的包、schema、适配器配置和轻量运行时契约。仓库内的
控制面示例不需要 GPU 或模型权重：

```bash
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

这些示例验证的是编排契约，不代表任何模型质量结论。

开放方法生成和 A/B 组合实验已有独立入口：`verdiwm-generate-method` 可调用
已配置的 LLM 适配器，将方法实现编译成隔离代码包；`verdiwm-open-method-study`
编译 baseline、A、B、A+B 四组方案并绑定种子、数据、评测与预算估计。
它们目前交付待校准候选，不会仅凭代码生成成功声称提点；四臂开放方法研究仍需
显式绑定目标画像、三份独立切分和冻结 verifier。实现检查、接口与使用边界见
[Method-to-Code](docs/METHOD_TO_CODE.md#open-implementation-and-pairwise-studies)。

当四组候选、冻结 verifier、checkpoint 和三份 episode manifest 已经准备好后，
可以用 `verdiwm-run-open-method-study` 自动执行本地校准、配对 seed 的训练/推理、
冻结评测、interaction 计算和本地 evidence CAS 归档。该命令需要显式提供 GPU
编号和一次确认过的输出目录；如果模型依赖独立的 Python 环境，可用
`--runtime-python /path/to/model/.venv/bin/python` 指定，避免候选进程误用控制面环境。
没有 GPU、split 重叠、文件 digest 漂移或校准失败
时会安全阻断并保留回执。它不会直接发布到社区或提升模型版本。
生成候选后可先运行 `uv run verdiwm-calibrate-method --compilation ... --output ...`，
执行声明的实现检查并保存通过或失败回执；失败检查不会被当作模型效果。

## 第一次使用自己的模型

你需要准备四项信息：模型代码目录、模型权重文件、数据集路径，以及一句
研究目标。权重通常不放进 VerdiWM 仓库，也不会被上传。

如果目录采用默认名称，可以直接运行简化入口：

```bash
uv run verdiwm setup --goal "提升长时域预测稳定性"
```

系统会生成 `verdiwm.toml`。目录名称不同则显式指定：

```bash
uv run verdiwm setup \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升长时域预测稳定性"
```

然后先做只读接入检查：

```bash
uv run verdiwm check
```

对完全陌生的模型，生成一份给用户或 Codex 使用的接入问卷：

```bash
uv run verdiwm guide-model --output ./.verdiwm/onboarding-questions.json
```

问卷会根据模型目录实际内容列出入口、权重、运行环境和评测方法等问题。
Codex 可以读源码并起草适配器和配置，但评测含义、指标阈值和 GPU 启动仍需
用户确认。不要把 API key 写入问卷或项目文件。

如果已有冻结评测契约和模型 Python 环境，可以在初始化时一并绑定：

```bash
uv run verdiwm setup \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升长时域预测稳定性" \
  --evaluator-contract /path/to/evaluator.json \
  --runtime-python /path/to/model/.venv/bin/python
```

### 一次确认的研究闭环入口

如果不想先手写项目配置，可以直接让系统把模型、数据、目标、运行环境和
评测器编译成一份可审阅计划。计划阶段只读扫描文件并计算摘要，不会导入模型、
联网调用研究服务或占用 GPU：

```bash
uv run verdiwm research plan \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升分钟级长程交互一致性" \
  --budget 4gpu-hours \
  --mode hybrid \
  --output ./.verdiwm/research-plan.json
```

检查计划中的 `state`、`blockers`、`evaluator_contract`、`input_digest` 和阶段
顺序。只有计划处于 `ready` 或 `ready_with_deferred_discovery` 时才可以确认：

```bash
uv run verdiwm research run \
  --plan ./.verdiwm/research-plan.json \
  --confirm
```

确认命令会再次校验模型、数据、适配器资源、运行 Python 和评测器的内容摘要，
然后复用 CampaignStore 创建不可变 revision，按当前适配器和证据门禁推进
onboarding、IRG/诊断、证据检索、候选物化、配对筛选、独立确认和知识沉淀阶段；
无法证明的阶段会停在对应门禁。任何输入在计划生成后被
替换都会停止并要求重新生成计划。检索、生成代码、接口校准或单次 loss 下降都
不会自动形成“有效提升”结论；开放方法只有在目标侧四臂 study、冻结 verifier
和 held-out confirmation 完成后才具备科学结论权。

## 运行自己的项目

在模型和数据集旁边创建 `verdiwm.toml`：

```toml
[project]
model = "./model"
data = "./data"              # 也会发现 ./dataset
budget = "1gpu-hour"
state_root = "./.verdiwm/state"
```

确认检查结果中没有阻断项后，再启动任务。模型权重作为 asset 传入；例如：

```bash
uv run verdiwm check
uv run verdiwm run \
  --goal "提升长时域动作条件预测" \
  --target-metrics runtime_ready \
  --asset=--ckpt_path=/path/to/checkpoint.pt
```

`check` 或 `run` 如果提示缺少评测入口、评测契约、运行环境或权重，
这是正常的安全阻断：系统会告诉你要补什么，不会猜测成功标准，也不会在
未确认评测方法前占用 GPU。已有适配器的模型通常只需补齐路径；完全新模型
需要按问卷回答运行和评测信息，确认后才能生成可启动的隔离配置。

没有项目文件时，系统会发现约定目录 `model/` 与 `data/`（或 `dataset/`）。
运行器会选择明确匹配的适配器配置，解析 evaluator 已声明的指标；接口需要
调整时会自动生成隔离的 adapter overlay。未知指标、适配器歧义、科学资产缺失
或协议漂移都会安全阻断，并给出诊断信息。

### 批量接入多个异构模型

把每个模型、数据、适配器和冻结评测写入一个批次请求，先编译静态计划：

```bash
uv run verdiwm batch plan \
  --manifest batch-request.json \
  --output-root ./.verdiwm/batches/my-batch
```

检查通过后，创建并排队各模型的独立 campaign：

```bash
uv run verdiwm batch run \
  --plan ./.verdiwm/batches/my-batch/plan.json \
  --max-parallel 2
uv run verdiwm batch status \
  --execution ./.verdiwm/batches/my-batch/execution.json
```

也可以使用兼容脚本的 `batch-plan` 和 `batch-run` 命令。批次会共享一个
预算账本、Archive 和 CAS；每个模型仍保留独立 campaign、revision 和评测
receipt。任何缺少冻结 evaluator 或发生文件漂移的模型都会单独显示为阻断项，
不会被静默跳过或当作成功。

### 发布社区知识包

实验完成后，可以先从多个本地产物目录只读整理已经验证的模型画像、Capability
IR、IRG 和 evidence records，再签名发布到社区 registry：

```bash
uv run verdiwm community export \
  --source-root ./local-artifacts \
  --source-root ./.verdiwm/semantic-records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./.verdiwm/community-export
```

导出器只扫描 JSON 语义记录：无关的运行时文件会被统计并忽略，已识别但不符合
portable knowledge graph 契约的记录会阻断。它不会导入模型、占用 GPU、修改源
目录，也不会复制 `execution.json`、campaign、budget、Archive、CAS 或本地路径。
`export.json`、`graph.json` 和 `quality-audit.json` 是 staging 元数据；发布时
只使用 `records/` 目录：

```bash
uv run verdiwm community publish \
  --documents-dir ./.verdiwm/community-export/records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem
uv run verdiwm community verify \
  --bundle-root ./community-bundle \
  --public-key ./publisher-public.pem
```

发布包只包含无本地路径的语义记录、确定性知识图谱、quality audit、成员
SHA-256 和 Ed25519 签名。`execution.json` 只作为批次身份绑定，campaign、
预算数据库、模型路径和运行命令不会被打包。检索到的记录仍然只是目标侧
实验的 hypothesis；社区发布和验证不会替代冻结 evaluator 或 settlement
receipt。完整约定见 [社区 Bundle 文档](docs/COMMUNITY_BUNDLES.md)。

需要撤回或替换社区知识时，可以生成 append-only 生命周期记录：

```bash
uv run verdiwm community lifecycle \
  --action revocation \
  --subject-kind community_bundle \
  --subject-id verdiwm-bundle-0123456789abcdef01234567 \
  --reason "目标侧验证器发现结论无效" \
  --authority-ref cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --evidence-ref cas://sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --output ./lifecycle/revocation.json
```

该记录只包含内容地址和语义身份，不包含本地路径；同一路径已有不同内容时会安全阻断。

CI 和复现实验仍可使用显式参数：

```bash
uv run verdiwm run \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升长时域动作条件预测" \
  --budget 8gpu-hour \
  --mode hybrid
```

查看、取消或复现任务：

```bash
uv run verdiwm status CAMPAIGN_ID
uv run verdiwm cancel CAMPAIGN_ID
uv run verdiwm reproduce CAMPAIGN_ID
```

## 本地交互界面

启动 workbench：

```bash
uv run verdiwm-workbench --port 8765
```

默认会在 `state_root` 下发现已物化的 `graph.json` 以及历史运行目录；如果实验
产物保存在独立目录，可显式绑定证据根目录：

```bash
uv run verdiwm-workbench --port 8765 \
  --state-root ./.verdiwm/state \
  --evidence-root /path/to/verdiwm-runs
```

浏览器访问 <http://127.0.0.1:8765>。界面提供项目发现、快速开始/因果发现/
混合模式、任务控制、任务详情和交互式证据图谱。它只在本机运行，不会上传模型
或数据。

### Windows 策略拦截

部分 Windows 环境会以“应用程序控制策略已阻止此文件”为由拦截
`.venv\\Scripts\\*.exe` 启动器。可以改用 Python 模块入口：

```powershell
uv run python -m wmloop.cli doctor
uv run python -m wmloop.control.workbench --port 8765
```

如果连 `uv run python --version` 也被拦截，需要由管理员或 IT 在应用程序控制
策略中允许已安装的 Python/uv；这不是 VerdiWM 代码错误。从下载的 ZIP 解压时，
请先在 ZIP 文件属性中勾选“解除锁定”，再重新安装依赖。

## 主要能力

- 将自然语言目标编译为带类型的目标、指标、探针、试验、判定和证据契约。
- 适配器/profile 的发现、版本化解析与一致性检查。
- 渐进式评估、不可变运行回执、独立验证、取消和复现。
- 带 provenance 的证据图谱和效果记忆，保留正向、无效和有害结果。
- 面向重复实验的研究模式和本地 workbench。
- 为新模型族和 evaluator 提供受控扩展点。

架构与扩展边界见 [Architecture](docs/ARCHITECTURE.md)、[Onboarding](docs/ONBOARDING.md)
和 [Backbone instantiation](docs/BACKBONE_INSTANTIATION.md)；workbench 内置可用的
研究模式。

## 范围与版本状态

当前公开版本为 `1.0.3`（稳定版）。控制面、schema、CLI、示例、workbench、
机制自动组合和新模型首次接入流程均通过可复现发布门禁。部署提供可信基础
profile 和受约束修复 provider 后，新模型族可以自动生成并验证 adapter。系统
不会猜测科学资产或 evaluator 语义：编排成功不等于模型质量提升，任何质量结论
仍必须基于真实模型运行时、数据和冻结的验证协议。

发布检查、贡献和安全说明分别见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[SECURITY.md](SECURITY.md) 以及仓库中的 `RELEASE_AUDIT.json`。

## 许可证

VerdiWM 使用 [Apache License 2.0](LICENSE)。外部数据集、模型权重和上游项目
仍遵循各自许可证。
