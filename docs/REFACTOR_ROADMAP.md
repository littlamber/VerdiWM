# VerdiWM 核心重构记录与维护边界

状态：核心六处单体的职责拆分已落地（2026-11-11），旧 CLI/import 兼容入口保留。
本文件记录代码结构和工程验证，不把 CPU 验证解释为真实模型效果或论文结果复现。
最终验证记录写入 `artifacts/refactor/validation_report.json`，运行产物不入 Git。

## 1. 公共基础能力已收敛

`wmloop.storage` 提供 canonical JSON、文件/字节 SHA-256、JSON 读取、原子写和锁。
运行时 digest、artifact bundle、多个 daemon、自动实验、onboarding、训练评估和
机制发现组件复用这些实现。历史私有函数只作为委托层保留；ASCII/UTF-8 和换行
契约必须通过测试保护。单个领域异常仍由调用层转换，storage 不依赖业务模块。

资源来自参数、项目配置或 `VERDIWM_*` 环境变量。GPU lock 默认通过系统临时目录
推导，配置默认值在调用/实例化时读取。pytest 已移除 `pythonpath` 设置，导入由
editable 安装提供，并有 checkout 外 `python -I` 的组件导入检查。

## 2. 已完成的职责拆分

| 兼容入口 | 内部实现 | 保留在入口的职责 |
|---|---|---|
| `wmloop/cli.py` | `control/commands/` | 命令装配、别名和错误呈现；parser 独立，run 接收显式依赖 |
| `wmloop/orchestrator_training_eval_smoke.py` | `execute/training_smoke/` | 训练/冻结评估的装配与 CLI |
| `wmloop/control/onboarding.py` | `control/onboarding_scan/` | 只读扫描与报告装配 |
| `wmloop/execute/autonomous_pipeline.py` | `execute/pipeline/` | 阶段顺序、恢复/中断异常边界、公开运行入口 |
| `wmloop/retrieve/mechanism_discovery.py` | `retrieve/discovery/` | 发现与重新比较事务装配 |
| `wmloop/experiments/portable_knowledge_graph.py` | `experiments/knowledge/` | 公开构建、审计、查询和导出 API |

训练评估将 planning、routing、runtime、executor、evidence、settlement、bundle 分开。
探针响应只负责构造验证输入，`verify` 保留判定权威。onboarding 将文件扫描、源码
解析、资产识别、运行时探测、evaluator 绑定和报告 IO 分开，禁止向外部源码写回。

机制发现将 request/query、来源获取、证据提取、profile 和比较分开，继续只产生
shadow ranking。知识图谱将通用图结构与 method/model/contract/transfer 投影分开，
保留 path-free 输出和内容身份。内置 extractor 的历史身份不因 Python 模块迁移改变。

pipeline 仍明确列出各阶段顺序，状态锁、输入预算、journal snapshot、catalog 合并
与 IRG/文献查询上下文下沉专用组件。失败快照和恢复语义由现有集成测试保护。
这不是把入口搬到另一个巨型 `runner.py`，也不引入动态 `globals()` 转发。

## 3. 防止结构退化的自动约束

- 布局审计限制根模块、domain 目录、脚本前缀和实验目录版本命名。
- 六个组件目录禁止反向 import 自己的兼容入口；组件内部 import 图不能成环。
- 六个组件目录单文件最多 800 行；旧文件大小仍以审计候选形式报告，不把所有
  历史实验驱动机械拆成细碎文件。
- `geometry/retrieve` 禁止导入 `execute/torch`。历史诊断运行入口不在该规则中，
  不能据此声称整个仓库已完全消除历史跨层调用。
- `control/geometry/retrieve/experiments` 的包级公开 API 惰性导出，组件导入不自动加载整条流程。
- 命令 JSON stdout 是兼容协议，由 `control/commands/io.py` 集中输出；普通日志走日志接口。

对应约束同时写入 `AGENTS.md`、`docs/code_conventions.md` 和 agent 代码生成策略。
新组件放入哪个目录、能够调用哪层、如何验证均有明确规则。

## 4. 验证与实际限制

验证命令：

```bash
uv run --offline --frozen pytest -q tests
uv run --offline --frozen python scripts/ci/check_repository_layout.py
uv run --offline --frozen python -m compileall -q wmloop scripts experiments tests
```

当前环境的离线构建缓存缺少构建依赖，普通 `uv run --offline --frozen` 在重新构建
editable 包时失败。按照 `AGENTS.md` 的既定例外，确认现有 editable 环境有效后，
使用相同命令加 `--no-sync` 完成代码验证；这不代表 `uv sync` 或 wheel 构建通过。
独立导入检查使用当前虚拟环境解释器的 `-I` 模式，不依赖当前工作目录。

2026-11-11 完成记录：全量回归 **684 passed、2 skipped**；布局/依赖审计和全仓
Python 编译通过，六个兼容入口的历史函数/类名均保留。两项跳过分别是当前环境
未安装 torch、真实 Cosmos3 CPU smoke 产物不存在。发布测试在独立临时 Git 仓库中
模拟新增文件入库，验证规范和组件进入源码副本，且公开 CPU 示例可运行；不修改
开发者实际 Git index，也不把未跟踪文件自动加入生产发布范围。

回归重点包括：run 的 confirm/dispatch 顺序、阻塞时不创建 campaign、自动修复
receipt 绑定、默认 schema 根目录、UTF-8 receipt 字节、环境变量的运行时解析、
IRG 跨域查询与缺失观察边界、组件独立导入、循环/反向依赖拒绝，以及已有 CAS、
selector、恢复和完整 CPU 回归测试。

## 5. 持续维护规则

历史 `scripts/export/` 和模型专属实验保留可复现路径；以后修改时将重复能力继续
下沉领域 API，不新增第二套 IO/资源解析/结算流程。大型版本化实验可以保留明确的
阶段装配，但不允许把公共能力复制到每个脚本。目录迁移不得删除旧 CLI alias、
artifact type、公开 example 或证据 fixture；需要改变语义时先提供版本化迁移。
