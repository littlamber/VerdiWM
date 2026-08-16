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
```

`verdiwm doctor` 会检查 Python 版本、核心 schema、适配器配置和轻量机制本体；缺少或损坏任一必需项都会返回 `blocked`。

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

## 当前边界

- 控制面、适配器契约、渐进式实验调度、证据归档和失败闭合迁移已经实现。
- ACWM-Phys 提供一个完整性校验过的最小闭环证据包。
- Ctrl-World 与 Cosmos3 包含局部响应图、反例和正确 abstain 的迁移结算。
- 跨模型自动迁移、新骨干上的自主 atlas 进化和多训练种子因果复现仍是研究工作，不作为已完成能力宣传。

完整实现状态与证据边界见 [英文 README](README.md)、[架构说明](docs/ARCHITECTURE.md)、[方法到代码映射](docs/METHOD_TO_CODE.md) 和 [可复现性说明](docs/REPRODUCIBILITY.md)。

## 发布

正式上传魔塔社区或 GitHub 前，必须从干净且全部已跟踪的 Git checkout 运行：

```bash
scripts/ci/release_preflight.sh --output-dir dist/verdiwm-release
```

上传 `dist/verdiwm-release/repository/`，并保留其中的 `RELEASE_AUDIT.json` 与 `MANIFEST.sha256`。完整步骤见 [发布清单](docs/RELEASE_CHECKLIST.md)。

## 许可证

VerdiWM 源代码采用 [Apache License 2.0](LICENSE)。外部数据集、模型权重和上游仓库仍遵循各自许可证。
