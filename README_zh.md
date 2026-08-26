# VerdiWM Clean v0.1.0

VerdiWM 是一个轻量、模型无关的科研优化控制面：用户提供模型适配器、数据评估器、目标和预算，系统负责探针诊断、指纹与画像、批量干预验证，并把 positive、null、harmful、abstain 结果沉淀为可迁移证据。

本版本是可发布的 Kernel。它包含标准库内核、自动科研组合模块和 CPU fixture 适配器，用于验证控制面闭环，不代表任何真实模型效果。仓库不包含 Ctrl-World、模型权重、数据集、GPU 启动器、私有凭据或历史实验 bundle；这些内容应作为独立的适配器和 artifact 发布。

## 五分钟启动

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

`doctor` 检查安装和内核能力，`demo` 在无 GPU、无网络、无真实模型时运行完整契约闭环，`graph` 查看 fixture 产生的画像、指纹和证据。`state/` 是本地运行时目录，可在 smoke test 后删除。

要在无网络、无真实模型时验证检索、idea 抽取、调度、评测和知识投影的一体化组合：

```bash
.venv/bin/verdi cycle --offline \
  --state-root state/cycle \
  --objective "improve held-out quality"
```

发布前运行：

```bash
./scripts/release_preflight.sh
python -m build                 # 可选：构建 wheel/sdist
```

## 接入真实模型

用户不需要重写调度器、worker、评测器或知识图谱。提供一个实现 `ModelAdapter` 协议的薄适配器，以及数据/评测 manifest：

```text
模型 SDK 或 HTTP API + 数据 manifest + objective
    -> adapter 与 evaluator 草稿
    -> contract tests 和指标充分性检查
    -> 人工确认权限与假设
    -> 隔离、可恢复的 autonomous campaign
```

适配器通常实现 `inspect`、`probe`、`intervene`、`evaluate`，只声明自己确实支持的能力；不支持的探针会被记录为 unsupported，而不是静默跳过。缺少 hook 时，系统可以在隔离 worktree 中让 AI 尝试物化 adapter、probe 或 plugin，原始模型 checkout 和权重不会被覆盖。

统一 campaign 入口示例：

```bash
.venv/bin/verdi campaign autonomous-run \
  --state-root state/run --run-id run-001 --model-id my-world-v1 \
  --objective "improve held-out quality" --ideas ideas.json \
  --runner my_adapter:stage_runner \
  --replanner my_adapter:replan \
  --worktree-root state/worktrees --output-root state/artifacts
```

离线验收可将 runner 换成 `adapters.fixture_campaign:runner` 并增加 `--offline`；它会故意触发一次失败，验证 AI 修复收据、隔离重试和 replicated positive 停止门控。

可选的 AI 自主能力通过 `VERDI_AI_BASE_URL`、`VERDI_AI_API_KEY`、`VERDI_AI_MODEL` 配置；任意 OpenAI-compatible endpoint 均可使用，同一 provider 服务于规划、双路抽取、指标选择和探针进化。工程执行由受限的 `EngineeringAgent` 完成：它能读文件、创建 worktree、应用 patch、运行测试/训练、诊断失败并收集 artifact，但不能写原始 checkout、push、远程上传、提权或越界访问。

## 知识图谱与迁移

知识图谱不是只有结论的列表。系统将探针、模型画像、方法来源、实验、复现、指标、结论、迁移判断和哈希凭据按 L0-L5 分层保存。SQLite 是本地查询快照，append-only records 用于社区合并，`graph.json` 是交换格式，`graph.html` 是无需后端的搜索、筛选、拖动和点击追溯页面。日志、视频和 checkpoint 作为带哈希的 artifact 引用保存，不直接膨胀数据库。

```bash
.venv/bin/verdi graph-bundle \
  --state-root state/demo \
  --output-root state/demo-community-bundle
```

bundle 目录可作为一个社区 artifact 上传，包含 `knowledge.sqlite3`、`graph.json`、`knowledge.jsonl`、`transfer_index.json` 和 `graph.html`。大文件另行按版本上传。迁移查询只会生成排序后的候选，目标模型仍必须通过 conformance 和固定 held-out evaluator 的验证；完成训练不等于证明科学提升。

查询示例：

```bash
.venv/bin/verdi transfer --state-root state/demo \
  --target-model-id new-world \
  --diagnostic history_dependence \
  --architecture dit --capability rollout
```

## 发布边界

GitHub / ModelScope 发布 Kernel 源码和测试；模型适配器、知识图谱 bundle、权重、数据集、日志、视频和 checkpoint 使用独立、可版本化的 artifact。详见 [`PUBLISHING.md`](PUBLISHING.md)、[`docs/FULL_LOOP.md`](docs/FULL_LOOP.md) 和 [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)。真实科学结论必须依赖固定 held-out evaluator 和独立复现。
