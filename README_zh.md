# VerdiWM

VerdiWM 是一个面向世界模型的意图驱动研究工作台：用户提供模型、数据和
目标，系统负责解析适配器与评估契约，执行有边界的实验，并为每个决策保留
可追溯证据。

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

`doctor` 会检查已安装的包、schema、适配器配置和轻量运行时契约。仓库内的
控制面示例不需要 GPU 或模型权重：

```bash
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

这些示例验证的是编排契约，不代表任何模型质量结论。

## 第一次使用自己的模型

你需要准备四项信息：模型代码目录、模型权重文件、数据集路径，以及一句
研究目标。权重通常不放进 VerdiWM 仓库，也不会被上传。

如果目录采用默认名称，可以直接运行：

```bash
uv run verdiwm init --goal "提升长时域预测稳定性"
```

系统会生成 `verdiwm.toml`。目录名称不同则显式指定：

```bash
uv run verdiwm init \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升长时域预测稳定性"
```

然后先做只读接入检查：

```bash
uv run verdiwm check-model
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
uv run verdiwm init \
  --model /path/to/model \
  --data /path/to/data \
  --goal "提升长时域预测稳定性" \
  --evaluator-contract /path/to/evaluator.json \
  --runtime-python /path/to/model/.venv/bin/python
```

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
uv run verdiwm check-model
uv run verdiwm run \
  --goal "提升长时域动作条件预测" \
  --target-metrics runtime_ready \
  --asset=--ckpt_path=/path/to/checkpoint.pt
```

`check-model` 或 `run` 如果提示缺少评测入口、评测契约、运行环境或权重，
这是正常的安全阻断：系统会告诉你要补什么，不会猜测成功标准，也不会在
未确认评测方法前占用 GPU。已有适配器的模型通常只需补齐路径；完全新模型
需要按问卷回答运行和评测信息，确认后才能生成可启动的隔离配置。

没有项目文件时，系统会发现约定目录 `model/` 与 `data/`（或 `dataset/`）。
运行器会选择明确匹配的适配器配置，解析 evaluator 已声明的指标；接口需要
调整时会自动生成隔离的 adapter overlay。未知指标、适配器歧义、科学资产缺失
或协议漂移都会安全阻断，并给出诊断信息。

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
