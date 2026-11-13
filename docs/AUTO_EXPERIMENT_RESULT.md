# 通用实验结果契约

`auto_experiment_result.schema.json` 定义
`artifact_type=verdiwm-auto-experiment-result` 的版本 1 边界。外部评估工作负载和
CUDA 冒烟工作负载在写入前校验；通用验证器在读取后独立校验。既有结果与 receipt
不需要重写，也不会因为补充 schema 而取得新的验证权威。

顶层只接受声明的字段。必需字段为 `schema_version`、`artifact_type`、`state`、
`device` 和 `metrics`；`device` 必须声明 CUDA 类型与非空 GPU UUID，已知设备
索引、内存和版本字段若出现则按各自类型校验。`metrics` 是非空的名称到有限数值
映射，布尔值、数字字符串、NaN、Infinity 和超出浮点范围的整数均不接受。校验
覆盖所有指标，包括没有用于本次 gate 的指标。

`workload` 是可选的扩展元数据对象，支持自定义字段和嵌套 JSON；内置工作负载
已有的命令摘要、日志路径、矩阵规模等字段保留其类型约束。动态字段的值必须是
JSON 值，所有层级的数值都必须有限。自定义工作负载无需伪装为内置工作负载，
也无需补充设备名称、物理索引或 `workload` 才能满足核心结果契约。

`state=invalid` 是可保留的失败产物；结构有效不代表本次运行通过。结构损坏的
结果在通用验证器中产生 `RESULT_CONTRACT_INVALID` 和 `VOID`，原始产物仍由
已声明的 artifact 清单进入 CAS。UUID、物理 GPU 活动、metric gate 与执行状态
仍需分别通过。schema 只约束结果结构，不能证明模型质量改善或替代冻结评估器。

通用 `screen` 的 `PASS` 仍是探索性筛选证据。通用 runner 在 SQLite 中保留
`exploratory=True`，不更新正式效果单元，也不授予迁移或模型质量声明权威。
