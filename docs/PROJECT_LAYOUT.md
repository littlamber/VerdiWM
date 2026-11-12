# VerdiWM 项目布局

VerdiWM 包含控制面、实验实现、模型适配器、导出脚本和历史兼容入口。目录按权限和生命周期分层，领域内部再按稳定职责拆分。

```text
VerdiWM/
├── wmloop/                 # 可安装的核心包，业务代码只能从这里复用能力
│   ├── storage.py          # canonical JSON、hash、原子文件和锁的唯一事实源
│   ├── contracts.py        # JSON Schema / contract 校验
│   ├── control/            # 意图、portrait、candidate、campaign 状态
│   │   ├── commands/       # CLI 参数声明、命令服务、配置边界
│   │   └── onboarding_scan/ # 只读源码、资产、runtime、evaluator 扫描
│   ├── diagnose/           # 诊断 probe、failure signature、probe admission
│   ├── geometry/           # IRG、效果区间、transfer、probe evolution 数学
│   ├── retrieve/           # archive index、IRG 检索、文献 staging
│   │   └── discovery/      # query、来源、证据提取、profile、比较
│   ├── propose/            # hypothesis、primitive/probe proposal
│   ├── primitives/         # 可执行 operator、hook、registry
│   ├── execute/            # worktree、budget、GPU lease、运行 receipt
│   │   ├── pipeline/       # 状态、预算、journal、catalog、检索上下文
│   │   └── training_smoke/ # 规划、执行、探针证据、结算和产物
│   ├── verify/             # 冻结 verifier、保护指标、正式结算
│   ├── archive/            # SQLite + CAS 事实库和公开投影
│   ├── evaluate/           # 评估器和结果视图
│   ├── experiments/        # 可复现实验编排和报告生成
│   │   └── knowledge/      # 可移植图结构与分类语义投影
│   └── web/                # 可选工作台 UI；训练/控制面不得依赖它
├── configs/                # schema、backbone、adapter、goal、probe、evaluator 配置
├── experiments/            # 大型、版本化、领域专属 campaign；目录名 `<domain>_<purpose>_vN`
├── scripts/                # 薄 CLI wrapper，不承载可复用业务逻辑
│   ├── ci/                 # check_*.py / check_*.sh
│   ├── export/             # 论文表格、bundle、settlement、public projection
│   └── integrations/       # 外部模型/机器人/数据连接器
├── tests/                  # pytest 单元/组件/契约测试
├── docs/                   # 正式架构、方法、运行协议和代码规范
├── examples/               # 可公开复现的最小输入与 fixture
└── ops/                    # docker、worldarena 等部署运维文件
```

`scripts/` 和 `experiments/` 保留历史路径兼容，但新增代码遵循这些前缀和用途：

| 路径 | 合法前缀 | 允许的内容 |
|---|---|---|
| `scripts/` | `run_`、`evaluate_`、`aggregate_`、`audit_`、`prepare_`、`freeze_`、`verify_` 等 | 参数解析、调用 `wmloop` API、输出 receipt |
| `scripts/export/` | `acwm_`、`cosmos3_`、`ctrl_world_`、`probe_`、`validate_`、`paper_` 等 | 只读汇总和发布产物 |
| `scripts/integrations/` | `run_` | 外部模型或机器人连接 |
| `scripts/ci/` | `check_` | 静态、布局、wheel 和控制面检查 |
| `experiments/` | `<domain>_<purpose>_vN` | 领域实验代码；公共逻辑必须下沉 `wmloop` |

根目录 `wmloop/*.py` 只允许兼容入口、协议和跨域基础设施。新的领域实现不能继续放在根目录。`wmloop.storage` 是文件和序列化共用层：新代码不得重新实现 canonical JSON、SHA-256、原子写和锁；领域错误码由调用层转换，底层不感知具体实验。

`scripts/ci/check_repository_layout.py` 会在 CI 中阻止新的未知根模块、未声明 domain 目录、非法脚本前缀和不符合版本命名的实验目录。上面六个组件目录还会检查反向 import 兼容入口、内部循环依赖和 800 行上限；`geometry/retrieve` 不能导入 `execute/torch`。其他历史大文件通过 `large_python_modules()` 持续报告。

`cli.py`、`orchestrator_training_eval_smoke.py`、`mechanism_discovery.py`、`onboarding.py`、`autonomous_pipeline.py` 和 `portable_knowledge_graph.py` 已完成职责拆分，保留原路径作为公开/兼容装配入口。`onboarding_scan/` 不命名为 `onboarding/`，避免遮蔽已有模块。新增功能直接放入对应组件，禁止往兼容入口回填公共实现。

拆分顺序固定为：先抽纯函数和 schema/IO，再抽阶段对象，最后才移动公开入口。每一步保留旧 import 和 CLI alias，并以窄测与全量 pytest 作为迁移门禁，禁止一次性目录重命名造成不可追踪的行为变化。

具体拆分顺序和不做事项见 [`REFACTOR_ROADMAP.md`](REFACTOR_ROADMAP.md)。
