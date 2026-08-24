# VerdiWM Clean v0.1.0

VerdiWM 是一个轻量、模型无关的科研优化控制面：用户提供模型适配器、数据评估器、目标和预算，系统负责探针诊断、指纹与画像、批量干预验证，并把 positive、null、harmful、abstain 结果沉淀为可迁移证据。

本版本包含标准库内核、自动科研组合模块和 CPU fixture 适配器，用于验证控制面闭环，不代表任何真实模型效果。用户不需要自己实现调度器、worker、评测器或知识图谱；只需提供模型 SDK/API、数据入口和目标，模型专属的薄适配器由系统生成或接入独立插件，并通过 `ModelAdapter` 协议验收。

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

要在无网络、无真实模型时验证完整的模型无关科研组合闭环：

```bash
.venv/bin/verdi cycle --offline --state-root state/cycle --objective quality
```

如需启用 AI 自主能力，只需配置 `VERDI_AI_BASE_URL`、
`VERDI_AI_API_KEY`、`VERDI_AI_MODEL`。所有规划、双路文献/代码抽取、指标
选择和探针进化都走同一个 OpenAI 兼容接口，只通过 role 区分任务。

设计原则：内核轻量化、模块化、模型解耦；探针用于诊断和画像，知识图谱只沉淀带边界和验证凭据的可迁移知识，不把单模型技巧伪装成通用结论。
