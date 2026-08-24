# VerdiWM Clean v0.1.0

VerdiWM 是一个轻量、模型无关的科研优化控制面：用户提供模型适配器、数据评估器、目标和预算，系统负责探针诊断、指纹与画像、批量干预验证，并把 positive、null、harmful、abstain 结果沉淀为可迁移证据。

本版本只包含标准库内核和 CPU fixture 适配器，用于验证控制面闭环，不代表任何真实模型效果。模型权重、数据集、GPU 运行时和领域适配器都应放在独立仓库，通过 `ModelAdapter` 协议接入。

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

设计原则：内核轻量化、模块化、模型解耦；探针用于诊断和画像，知识图谱只沉淀带边界和验证凭据的可迁移知识，不把单模型技巧伪装成通用结论。
