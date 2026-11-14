# VerdiWM

工程和研究闭环约束见 [`AGENTS.md`](AGENTS.md)，论文方法到代码的映射见 [`docs/VERDI_METHOD_ENGINEERING.md`](docs/VERDI_METHOD_ENGINEERING.md)，项目目录地图见 [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)。

VerdiWM 是一个面向世界模型的意图驱动研究工作台：用户提供模型、数据和
目标，系统负责解析适配器与评估契约，执行有边界的实验，并为每个决策保留
可追溯证据。

面向用户的 CLI 名称是 `verdi`；`verdiwm` 仍作为兼容入口保留。后文的完整
命令都可以把 `verdiwm` 替换成 `verdi`。

控制面不绑定某一个模型，也不包含模型权重、数据集、API 密钥或 GPU 运行时；
这些资产由实际部署的用户自行准备。

## 快速开始

环境要求：Linux（或 WSL2）、Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/littlamber/VerdiWM.git
cd VerdiWM
python -m pip install uv==0.12.10
uv sync --locked --group dev
uv run verdiwm doctor
```

Wan 2.2 5B + Agibot 训练链路诊断请看[中文快速开始](docs/AGIBOT_ENGINEERING_DIAGNOSTIC_QUICKSTART.md)，
也提供[English version](docs/AGIBOT_ENGINEERING_DIAGNOSTIC_QUICKSTART.en.md)。

运行 `uv run verdi` 会进入终端工作台。首页以青色面板显示模型、数据、研究目标、
LLM 配置状态、接入阻塞项和一个推荐操作；未完成检查时不会显示已就绪。长路径
和中文会随终端宽度换行。配置损坏时显示具体错误，并保留会话供用户修复。

输入 `/` 并回车打开命令面板；在
支持 raw TTY 的真实终端中，面板会实时响应字符筛选、上下方向键、Tab、Enter
和 Esc，超出屏幕的命令会随选中项滚动。打开面板后输入 `res`、`sta` 或中文
关键词可以筛选；也可以直接输入完整命令，例如 `/status`。`/help` 按用途列出
全部命令，`/help plan` 展示单项说明。会话还提供 Python 标准库的历史记录和
Tab 补全。设置 `NO_COLOR=1` 或 `TERM=dumb` 时使用静态、无颜色的命令面板：

```text
verdi> /
verdi> /start
模型目录 [/path/to/model]:
源码目录 [/path/to/source]:
数据目录 [/path/to/data]:
研究目标（例如：提升分钟级长程一致性）:
verdi> /plan "提升分钟级长程一致性" --rounds 2 --stop-on-success
verdi> /run
  研究计划 …
  最多运行轮数 … [2]:
  确认创建并执行这个 campaign？ [y/N]: y
verdi> /status
verdi> /recent
verdi> /progress
verdi> /resume
verdi> /exit
```

首次输入 `/start` 或 `/setup` 时，如果当前目录有 `model/` 和 `data/`，向导会
自动填入路径；用户只需确认路径并输入一句目标。项目配置生成后，`/start` 会继续
生成研究计划。常用命令支持短写法，例如 `/plan "目标"` 等价于显式的
`/plan --goal "目标"`，`/run PLAN` 会补全为 `--plan PLAN`。

命令执行时会先显示“执行 …”，随后将状态、计划路径、campaign 和 blocker 等
关键字段整理成可读结果卡，并显示每个阻塞项的解决步骤、问卷内容和下一步，
不要求用户阅读原始 JSON。`/guide` 展示接入问卷；需要保存并填写时使用
`/guide --output .verdiwm/onboarding-questions.json`，结果会给出保存位置与后续命令。
默认路径可直接按 Enter 接受；Ctrl-D 或 Ctrl-C 可取消接入向导。
模型权重与源码分开存放时，可在“模型源码目录”一问直接指定源码，无需退出向导。
明确的轮数指令（例如“提升长程一致性，跑两轮”或“最多跑3轮，达到目标就停”）会准备可审阅
计划；只说轮数时保留已有研究目标。其他自然语言继续生成下一步计划提示。
这些输入不会启动 GPU；`/run` 会先让用户审阅并确认，
继续执行研究计划、评测器和证据门禁。脚本、
CI 和管道环境不会进入会话，继续使用普通命令帮助；`verdiwm` 兼容入口也保持
原有行为。需要强制进入会话时可运行 `verdi chat`（或 `verdi shell`）。

`/recent` 会显示默认计划、最近 campaign 和推荐下一步；`/resume` 会优先显示正在
排队或运行的任务进度，否则展示默认计划并要求确认；`/progress` 可复用当前会话最近的
campaign ID；`/cancel ID` 只取消明确指定的任务。长命令会显示存活状态。
前台调度 campaign 时，`Ctrl-C` 会请求取消，并在回收活动子进程后返回；查看状态时，
它只中断当前视图。研究记录会保留。
普通 CLI 和终端界面共用项目配置或 `VERDIWM_STATE_ROOT` 指定的任务库，切换工作
目录后仍能查询同一库中的任务。任务库损坏或不可读取时会显示错误及恢复建议，
不会把读取失败显示成没有任务。`/s`、`/p`、`/r` 分别是 `/status`、`/progress`、
`/research` 的别名，`/configure` 等价于 `/setup`。

直接使用 `/plan "目标" --rounds 2` 或 `/run --rounds 2` 即可设置最多研究两轮，
无需切换到独立 daemon。`/run` 无参数时也会提示选择轮数，回车保留计划值；修改
策略时先保存并预览派生计划，再确认一次。所有轮次共享 `--budget` 总预算，
统一通过 `/status` 和 `/cancel ID` 管理。`--stop-on-success` 可在冻结验证器
正式确认目标收益后提前停，不能用诊断重复或 pipeline 通过代替。未绑定正式目标
契约或未产出有效确认时，仍按轮数、预算等边界停止。旧的独立控制器继续兼容。
资源要求、轮数、自动停止和恢复见[研究运行与停止说明](docs/RUN_LIFECYCLE.md)。
流程完成不等于模型已经提升，效果仍由冻结验证器确认。

终端视图、键盘面板和结果呈现分别维护在 `wmloop/control/terminal_ui.py`、
`terminal_palette.py`、`terminal_results.py`；交互入口复用原有 CLI 服务。
界面布局参考了 DeepSeek Harness 生态中的
[dsh-tui](https://github.com/rbelem/dsh-tui) 与
[eDEX 风格界面](https://github.com/ph4310822/dsh-edex-ui)，实现使用 Python 标准库，
无额外终端框架或 Harness 运行时依赖。

`doctor` 会检查已安装的包、schema、适配器配置和轻量运行时契约。仓库内的
控制面示例不需要 GPU 或模型权重：

```bash
uv run verdi doctor
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

仓库有意只保留一个很小的 CPU onboarding 示例。它只验证编排契约，不加载模型
权重、不使用 GPU，也不代表任何模型质量结论。实验中的证据 bundle、媒体和论文
图表通过 [社区 Bundle 流程](docs/COMMUNITY_BUNDLES.md) 在仓库之外单独导出、版本化
和共享。

## 配置研究 LLM

VERDI 使用 LLM 作为受约束的研究助手：读取 IRG 和诊断结果、跨领域检索方法、
提出候选 idea，并在隔离目录中起草 adapter。它和你要评测的 Wan、JEPA 或其他
世界模型是两件事；模型权重、数据集和 GPU 运行环境仍保留在你的机器上。

推荐把配置和密钥分开保存。默认配置路径是
`~/.config/verdiwm/config.toml`，密钥文件必须是只有当前用户可读的普通文件：

```bash
mkdir -p ~/.config/verdiwm
cat > ~/.config/verdiwm/config.toml <<'EOF'
[llm]
base_url = "https://api.openai.com"
model = "gpt-4.1"
api_style = "responses"
token_file = "auth"
EOF
read -r -s -p 'LLM API key: ' VERDI_LLM_KEY
printf '\n'
printf '%s' "$VERDI_LLM_KEY" > ~/.config/verdiwm/auth
unset VERDI_LLM_KEY
chmod 600 ~/.config/verdiwm/auth
uv run verdi llm status
```

第三方 OpenAI-compatible 服务只需修改 `base_url` 和 `model`。如果服务只提供
Chat Completions API：

```toml
[llm]
base_url = "https://api.example.com"
model = "provider-model"
api_style = "chat_completions"
token_environment_key = "MY_VERDI_LLM_TOKEN"
```

```bash
read -r -s -p 'LLM API key: ' VERDI_LLM_KEY
printf '\n'
export MY_VERDI_LLM_TOKEN="$VERDI_LLM_KEY"
unset VERDI_LLM_KEY
uv run verdi llm status
```

使用环境变量时不要再设置 `token_file`。配置可以放在仓库外的任意位置，用
`VERDIWM_CONFIG=/path/to/config.toml` 指定；也可以把路径传给
`verdi llm status --config /path/to/config.toml`。需要研究 LLM 的命令可以显式
传 `--llm-config`：

```bash
uv run verdi run \
  --model /path/to/model --data /path/to/data \
  --goal "提升分钟级长程交互一致性" \
  --llm-config ~/.config/verdiwm/config.toml
uv run verdi repair-adapter ... --llm-config ~/.config/verdiwm/config.toml
```

本地 Ollama 或 vLLM 如果明确不需要认证，请显式写出：

```toml
[llm]
base_url = "http://127.0.0.1:11434"
model = "qwen2.5:32b"
api_style = "chat_completions"
auth_required = false
```

只有 `localhost`、`127.0.0.1` 和 `::1` 允许使用 HTTP；远程服务必须使用 HTTPS。
`uv run verdi llm status` 只读取本地配置和环境变量，不会联网，也不会显示 key，
只报告 provider、model、API 类型、凭据来源和权限问题。`uv run verdi llm` 会打印
同一份可复制教程；交互式 CLI 中输入 `/llm` 或 `/llm status` 也可以查看。

不要把真实 key 写入 `verdiwm.toml`、研究计划、问卷、prompt、receipt、evidence、
日志或 Git。检索和生成的 idea 仍然必须经过目标侧 evaluator、冻结 verifier 和
证据门禁；LLM 本身不能宣布模型提升。

开放方法生成和 A/B 组合实验已有独立入口：`verdiwm-generate-method` 可调用
已配置的 LLM 适配器，将方法实现编译成隔离代码包；`verdiwm-open-method-study`
编译 baseline、A、B、A+B 四组方案并绑定种子、数据、评测与预算估计。
它们目前交付待校准候选，不会仅凭代码生成成功声称提点；四臂开放方法研究仍需
显式绑定目标画像、三份独立切分和冻结 verifier。实现检查、接口与使用边界见
[Method-to-Code](docs/METHOD_TO_CODE.md#open-implementation-and-pairwise-studies)。

当四组候选、冻结 verifier、checkpoint 和三份 episode manifest 已经准备好后，
可以用 `verdiwm-run-open-method-study` 自动执行本地校准、配对 seed 的训练/推理、
冻结评测、interaction 计算和本地 evidence CAS 归档。该命令需要显式提供 GPU
编号和一次确认过的输出目录；研究请求须预先声明至少两个与选择阶段不重叠的确认种子。
该路径的确认观察目前仍是探索性证据：正式验证结算、可信归档和跨轮效果记忆
尚未接通。如果模型依赖独立的 Python 环境，可用
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
  --protected-metrics action_fidelity \
  --budget 4gpu-hours \
  --mode hybrid \
  --output ./.verdiwm/research-plan.json
```

`--protected-metrics` 是可选核对项：正式保护指标始终从 evaluator 绑定的
`frozen_goal` v2 验收策略派生，并显示在计划的 `protected_metrics` 中。显式传入时，
必须与冻结目标声明的完整非主指标集合一致；缺少冻结目标或集合不一致都会阻塞计划，
不会自动创建或修改 goal。它与 IRG 的 `--irg-protected-metric` 分开，后者只用于
IRG 检索约束，不决定 verifier 的正式判定。

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

继续前请同时查看 `state` 和 `next_step`，不能只看 `blockers`：即使阻断列表为空，
`needs_confirmation` 仍表示评测方法或运行入口尚待确认；只有到达
`ready_for_conformance` 后，才继续生成并审阅研究计划，再显式确认运行。模型权重作为
asset 传入；例如：

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

### 本地成熟度审计（PSNR/SSIM）

第一次接入真实模型之前，可以用无 GPU 的本地审计验证 VERDI 的控制面、成对评测、证据回执和知识沉淀是否闭环：

```bash
uv run verdi audit local-smoke \
  --repo-root /path/to/VerdiWM \
  --output-root /path/to/verdi-runs/local-maturity-audit
```

审计会生成合成的成对帧，使用冻结契约 `configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json` 计算 PSNR、SSIM，并验证完美预测、退化预测、输入错误、receipt digest、输入漂移、CAS 往返和源树不可变性。输出 `state: ready` 只表示 CPU 控制面和证据闭环通过；它不代表 Wan/DROID 真实训练成功、分钟级一致性提升、IRG 迁移预测有效或产生了科学结论。真实实验仍需用户确认 research plan，并在目标模型的冻结 evaluator 上运行。

已有 Wan2.2 两 slot × 10D arm-slot checkpoint 可以使用
[有界原生推理入口](docs/WAN_ARM_SLOT_INFERENCE.md)，由项目管理 GPU、共享预算、
超时和 CAS 证据。它保留模型原生 rollout 接口，与 7D action/14D proprio
训练适配器分别绑定；推理 receipt 只提供诊断证据，正式优化仍需完整 target 验证。

要检查第一轮结果能否被第二轮真正取回并用于候选选择，见
[两轮诊断与经验复用](docs/TWO_ROUND_DIAGNOSTIC_MEMORY.md)。流程保留正向、负向和
零差异观察，按匹配的动作与冻结协议提供排序先验，并要求第二轮重新执行验证。

评测器也可以单独使用。预测帧和 ground truth 帧放在两个目录中，文件名（不含扩展名）必须一一对应，支持 `.npy` 和常见图像格式：

```bash
uv run python scripts/evaluate_psnr_ssim_smoke.py \
  --predicted-dir ./predicted \
  --ground-truth-dir ./ground-truth \
  --output-root /path/to/verdi-runs/psnr-ssim \
  --contract configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json
```

加入 `--baseline-receipt` 后，评测器会在两个指标都改善时输出 `POSITIVE`，都下降时输出 `HARMFUL`，其余情况输出 `NULL`。回执可用 `--verify-receipt` 离线复核；shape 不匹配、缺帧、NaN/Inf、输入输出目录重叠或文件漂移都会 fail closed。

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

### 支持的平台

控制面支持 Linux 与 Python 3.12。Windows 用户请在 WSL2 内安装和运行；
原生 Windows 暂不支持，因为存储层使用 POSIX 文件锁与目录同步。
macOS 和其他 Python 版本尚未纳入 CI 验证。

Workbench 与 Campaign API 只允许监听本地回环地址，供可信的本机用户使用，
所有 API 均要求访问令牌，不提供多租户授权。安全边界见 [SECURITY.md](SECURITY.md)，
发布流程见 [开源发布规范](docs/RELEASING.md)。
界面源码位于 `wmloop/web/`，直接编辑 HTML/CSS/JavaScript，无需 Node 构建。

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

当前工作树为尚未发布的 `1.0.4rc1` 候选版本（Beta），已有公开标签为 `v1.0.3`。控制面、schema、CLI、示例、workbench、
机制自动组合和新模型首次接入流程均通过可复现发布门禁。部署提供可信基础
profile 和受约束修复 provider 后，新模型族可以自动生成并验证 adapter。系统
不会猜测科学资产或 evaluator 语义：编排成功不等于模型质量提升，任何质量结论
仍必须基于真实模型运行时、数据和冻结的验证协议。

发布检查、贡献和安全说明分别见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[SECURITY.md](SECURITY.md)、[Release Checklist](docs/RELEASE_CHECKLIST.md) 和
[社区 Bundle](docs/COMMUNITY_BUNDLES.md)。GitHub 源码仓库不包含自动生成的证据
manifest 或历史实验归档。

## 许可证

VerdiWM 使用 [Apache License 2.0](LICENSE)。外部数据集、模型权重和上游项目
仍遵循各自许可证。

## 候选版本说明

本工作树尚未发布，变更见 [CHANGELOG](CHANGELOG.md)，软件与论文引用状态见
[引用说明](docs/CITATION.md)，解释器迁移见 [Python 支持](docs/PYTHON_SUPPORT.md)。
本地 HTTP API 现在要求每次启动的访问令牌；终端显示仅当前用户可读的令牌文件路径，
读取后粘贴到工作台登录页。详见 [本地认证](docs/LOCAL_HTTP_AUTH.md)，不要把令牌放进 URL。
推荐通过 `uv sync --locked --group dev` 复现锁定环境。wheel 独立解析依赖另有检查；
离线结果只覆盖缓存可用版本，GitHub 在线工作流验证当时的解析结果。
