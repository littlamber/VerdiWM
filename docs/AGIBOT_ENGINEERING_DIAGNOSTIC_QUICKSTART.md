# Agibot 工程诊断快速开始

内置 `wan22-agibot-engineering-diagnostic-v1` profile 用于检查 Wan 2.2 5B
初始化权重在 Agibot arm-slot 数据上的训练、检查点保存、独立重载、验证、预算结算及
运行回执/CAS 留存链路。此入口不会写入 RSI settled-trial Archive 或模型效果记忆。
它是工程诊断：一次最多训练 8 个计划更新，只使用 train/val 分离的数据，并且结果
不能解释为模型效果提升或正式晋级结论。

以下命令从 VerdiWM 仓库根目录运行。先安装项目声明的控制端依赖，并确认命令可用：

```bash
python -m pip install uv==0.12.10
uv sync --locked
uv run verdi doctor
```

VerdiWM 控制端要求 Python 3.12。训练端 Python 是独立环境，由用户准备，并须包含与
原生训练代码兼容的 PyTorch 和 CUDA。`verdi doctor` 只检查控制端，不会检查训练端的
PyTorch 或 CUDA。建议显式指定本次要用的训练解释器，并在启动前做只读检查：

```bash
export VERDIWM_AGIBOT_PYTHON=/path/to/training-python
"$VERDIWM_AGIBOT_PYTHON" -c '
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
'
```

`CUDA available` 应为 `True`。若不设置 `VERDIWM_AGIBOT_PYTHON`，profile 会依次查找
源码目录和模型目录下的 `.venv/bin/python`；请确认实际选中的解释器也满足上述要求。

## 准备五项输入

在自己的机器上分别准备好：

1. Wan 2.2 Diffusers base 模型目录，含 `config.json`、模型权重及
   `Wan2.2_VAE.pth`。如果 VAE 单独存放，可用环境变量指定文件。
2. 用户的目标 Wan 2.2 5B action checkpoint 目录，含 `dit_model.safetensors` 和
   `meta.json`；该目录会作为 warm-start 参数传给原生训练入口。
3. 原生 EtherWM 代码目录，含 `scripts/training/train_world_model.py` 和 profile
   要求的 `sa_wm` 文件。
4. Agibot 数据根目录，含 `manifest.json`、`train.index.jsonl`、`val.index.jsonl`、
   `stats/` 和 `shards/`。
5. 与模型训练配置相符的 native YAML。

通过环境变量提供这些本机路径，命令和仓库 profile 本身不绑定任何开发机目录：

```bash
export WAN5B_CHECKPOINT=/path/to/wan-2.2-5b-action-checkpoint
export VERDIWM_AGIBOT_BASE_MODEL=/path/to/wan-2.2-diffusers-base
export ETHERWM_SOURCE=/path/to/etherwm-source
export AGIBOT_DATA=/path/to/agibot-data
export AGIBOT_TRAINING_CONFIG=/path/to/training_config.yaml
# 运行状态和产物放在仓库外，避免与源码目录重叠
export VERDIWM_STATE_ROOT="$HOME/.local/state/verdiwm/agibot-state"
export VERDIWM_CAMPAIGN_ID="agibot-wan5b-$(date -u +%Y%m%d-%H%M%S)"
```

默认从 `VERDIWM_AGIBOT_BASE_MODEL/Wan2.2_VAE.pth` 读取 VAE；若它单独存放，可额外设置
`VERDIWM_AGIBOT_VAE_CHECKPOINT`。控制端会检查 Diffusers 配置、权重分片、VAE 文件以及
warm-start checkpoint 的必需元数据，并在申请 GPU 前阻断缺失或明显不匹配的输入。
在 GPU lease 前，所选训练解释器还会读取 native YAML、以 meta 张量构造原生
`WMTrainer` 的 DiT 架构，并读取 warm-start safetensors header 中的键名和形状进行比较。
对于已恢复的 `history_reference` 分支，它还会在 CPU 上以有界分块检查所有参数是否有限，并按训练端相同规则拒绝全零的
encoder/QKV 权重；bias 和以零初始化的 output projection 权重仍然合法。整个检查不会把完整权重读入内存，也不会占用
GPU。完整缺失的 history-reference 分支属于原生支持的 warm-start 情形，因此不扫描 checkpoint 中不存在的权重；部分缺失会被拒绝。
这项 GPU 前预检与实际训练开始时的原生 checkpoint loader 和训练状态检查互相补充。

## 启动诊断

```bash
uv run verdi run \
  --model "$WAN5B_CHECKPOINT" \
  --source "$ETHERWM_SOURCE" \
  --dataset-root "$AGIBOT_DATA" \
  --training-config "$AGIBOT_TRAINING_CONFIG" \
  --world-size 2 \
  --goal "诊断 Agibot arm-slot 模型训练与验证链路" \
  --budget 2gpu-hour \
  --campaign-id "$VERDIWM_CAMPAIGN_ID" \
  --state-root "$VERDIWM_STATE_ROOT"
```

省略 `--adapter-profile` 时，`--training-config` 会自动选用内置 Agibot v3 profile；
显式传入 `--adapter-profile` 时则使用用户指定的 profile。v3 profile 的训练 loader
目前与随项目验证的 Wan 配置相符：5 帧历史、3 帧预测、8 帧采样步长、稀疏历史、
dilation `[1, 2]`、collapse 概率 `0.15`、验证 dilation `2`，关闭生成历史，并使用
`arm_slot_eef_pose` 与 body conditioning；验证固定面板为 128 个样本。使用不同 loader 的 YAML 时，复制 profile
并同步修改其中 `agibot_engineering_diagnostic.support_loader`，然后用
`--adapter-profile` 指定副本；不要只改 YAML 而让冻结的数据支持契约保留旧设置。

系统会从本机 YAML 派生 campaign 私有配置：`world_model.checkpoint` 绑定 Diffusers base，
`world_model.vae_checkpoint` 绑定 VAE，目标 action checkpoint 则通过 `--init_from` 单独
传给训练入口；派生配置还会绑定数据根目录和输出目录，原 YAML 不会被修改。内置 profile
将训练和检查点验证阶段分别限制为 40 分钟、
20 分钟；阶段上限之和会乘以计划使用的 GPU 数量，作为 ledger 的最坏情况预算预约。
因此启用 FSDP 的 YAML 必须通过 `--world-size` 声明至少 2 张 GPU；示例使用 2 卡，
对应 `--budget 2gpu-hour`。预算按阶段上限之和乘以计划 GPU 数量计算，增加 GPU 数量或
自定义 profile 提高阶段上限时，必须相应提高预算。若估算上限超过声明预算，系统会在
campaign 入队前阻断，并显示估算值与预算值；
复制 profile 后可在 `agibot_engineering_diagnostic.stage_timeouts_seconds` 中调整上限。
资源申请、阶段状态、训练 trace、检查点和诊断回执均写入本次 campaign。程序会在
启动 GPU 作业之前验证数据索引、train/val episode 是否隔离、shard 引用和原生源码
绑定；任一绑定失败都会给出错误码并停止。

`--world-size` 只用于自动生成的 Agibot scale plan，必须与 `--training-config` 一起使用；
若提供 `--training-scale-plan`，并行度以该显式计划为准，因此不能再传 `--world-size`。

如果预检或原生训练器报告恢复的 `history_reference` encoder/QKV 权重为非有限值或全零，
表示该检查点不适用于本次运行。失败诊断中的原生训练器建议从 H1 的
`checkpoint-134750` 开始。只有该检查点通过 GPU 前预检后，才将
`WAN5B_CHECKPOINT` 指向它，分配新的 campaign ID，并重跑上面的命令：

```bash
export WAN5B_CHECKPOINT=/path/to/etherwm/checkpoint-134750
export VERDIWM_CAMPAIGN_ID="agibot-wan5b-retry-$(date -u +%Y%m%d-%H%M%S)"
# 使用上面的 verdi run 命令重新运行
```

保留原失败 campaign、ID 和回执供追溯；不要复用旧 ID 或删除证据。如果建议的检查点
也没有通过预检，应暂停并调查其来源，或换用架构及恢复分支权重均通过检查的检查点。
其他配置或资产失败也应先修正输入，再使用新的 campaign ID。campaign 的失败恢复、状态和取消语义见
[运行生命周期说明](RUN_LIFECYCLE.md)。

训练进程的当前目录、HOME、临时目录和常用模型缓存都放在本次 campaign 输出目录中；
运行环境会保留所选 Conda/PyTorch/CUDA/NCCL 所需的运行变量，同时移除 shell 中的模型
服务凭证和代理设置，并关闭 W&B、SwanLab、Hugging Face 下载及遥测。manifest 中的
`network_required: false` 表示此诊断不需要联网；当前本机进程运行方式没有 OS 级网络或
文件系统沙箱，不能阻止任意原生训练代码主动联网，或用绝对路径写入其他可访问位置。
因此只对可信的原生源码运行此诊断；若需要强制隔离不可信代码，需在外部容器或等效 OS
沙箱中运行。

## 查看结果

查询任务时使用启动时相同的 state root 和 campaign ID：

```bash
uv run verdi status "$VERDIWM_CAMPAIGN_ID" --state-root "$VERDIWM_STATE_ROOT"
```

训练运行目录固定在 `$VERDIWM_STATE_ROOT/runs/$VERDIWM_CAMPAIGN_ID`。可列出本次运行的
清单、阶段回执、检查点和 trace 文件：

```bash
find "$VERDIWM_STATE_ROOT/runs/$VERDIWM_CAMPAIGN_ID" -maxdepth 4 -type f -print
```

运行 manifest 和 receipt 可能记录模型、数据、源码、Python 解释器及运行目录的本机绝对路径。
它们可用于本机追溯，但公开分享前请检查并清理私有路径；不要把整个 state root、运行目录或
原始回执直接上传到 GitHub 或公开 bundle。

结束后查看终端状态、`verdi status` 返回的 campaign 记录，以及该运行目录中的
`model-run.json`、`execution-receipt.json`、`receipts/` 与 Agibot trace。只有状态为
`completed` 且 checkpoint 与诊断回执通过绑定校验，才表示这次工程链路闭合。此 profile
只支持工程诊断，不会据此声称模型优于初始权重。当前已声明的研究目标是候选模型相对基线提高至少
2 dB PSNR：对相同 held-out 验证样本和 rollout/decode 设置，至少运行 3 个配对 seed，按 seed
计算 candidate-minus-baseline 平均 PSNR，并对 seed 差值计算双侧 95% Student-t 置信区间；只有下界严格大于
2 dB 才满足该主指标阈值。用户已指定当前只以 PSNR 为效果门槛；SSIM 仅报告，LPIPS 当前未测量。
当前 profile 没有把配对 rollout 测量接入正式 verdict、训练结算或记忆更新，因此这只是冻结的
测量目标，不能据工程诊断或 measurement receipt 宣称模型效果通过，也不能声称已形成可复用的效果记忆。
