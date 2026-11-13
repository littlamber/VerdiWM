# VerdiWM 代码组织与编写惯例

[`AGENTS.md`](../AGENTS.md) 是研究权威边界、离线依赖和资源约束的事实来源。
本文补充实现惯例；不把 FastWAM 的 `src/fastwam`、Hydra、Python 3.12、
O10 数据或日志依赖移植到本项目。

所有路径必须可移植到任意 GitHub checkout。模型、数据、运行时 Python、输出、
archive、CAS 和 GPU lock 通过参数、配置或 `VERDIWM_*` 环境变量提供；不得提交
`/cs`、`/mnt`、个人 home、开发机 venv 或私有挂载路径。公开 artifact 需要先做
路径无关投影，不能把 checkout 绝对路径写入社区 bundle。

## 1. 命名和层次

- 对外项目名保持 VerdiWM，论文方法名 VERDI，发行包 `verdiwm`，Python 包
  `wmloop`，用户 CLI `verdi` 与兼容入口 `verdiwm`。不要批量替换历史 artifact
  type、命令、引用或上游署名。
- CLI 只解析参数、调用已有 API 和呈现结果。数学计算进 `geometry/`，状态事务
  进 `control/campaign_repository.py`，模型执行进 `execute/`，最终判定进
  `verify/`，持久化进 `archive/`。不要把整条研究闭环堆到 `cli.py`。
- 新文件带 `from __future__ import annotations`；import 按标准库、第三方、
  本包分组。文件和函数 snake_case，类 PascalCase，类型注解使用内置泛型和
  `X | None`。无需为统一风格重写未触及的旧代码。
- 模型/数据实现经 adapter 接入；控制面 import 不应加载 torch 或 GPU。
  不新增模块级可变业务单例；配置由入口传入，禁止库函数改全局环境变量。
- 已拆分目录遵守 `AGENTS.md` 第 2.1 节：旧模块是兼容装配入口，内部组件不能
  反向导入入口或循环依赖。组件单文件上限 800 行；不能用长行、分号或把整类
  原封不动搬到新文件的方式规避职责拆分。parser 的长度来自声明式参数表，
  不得重新加入业务流程。
- `control`、`geometry`、`retrieve`、`experiments` 的包级导出使用 `__getattr__` 惰性加载。
  对外名称和对象身份保持兼容，内部直接从具体组件 import。

## 2. 配置和版本

- 保持当前 JSON/YAML + schema 的配置体系；不引入第二套 Hydra 装配机制。
- 配置解析后显式校验类型、形状、枚举、finite、范围和跨字段一致性。
  `bool("false")`、非空字符串 gate、`int(1.5)` 静默截断不能替代校验。
- 新功能默认关闭；启用缺少 hook、数据或输入时明确报错。不要用假响应、补零、
  stub、随机初始化或绕过 verifier 让功能看似可用。
- 改变坐标、距离、策略、probe basis 或 artifact 语义要显式版本化。
  历史 artifact 保持可读；兼容读取不能赋予新版本准入权限。
- 参数名、维度、probe/operator registry 是单点事实来源；跨模块 import，
  不复制另一份常量。不要为缺少来源的字段生成“合理默认值”。

## 3. 证据对象和副作用

- `dataclass(frozen=True)` 不冻结内部 dict/list；在归档/记忆边界防御性复制，
  或使用真正不可变结构。返回查询结果不得暴露可改写内部记录的引用。
- canonical hashing 使用现有工具，例如 `wmloop.storage.canonical_bytes`；
  禁止 NaN/Infinity。所有外部 JSON 先校验再消费，hash 只证明内容身份，
  receipt 的身份还必须与 authoritative archive 绑定。
- 文件写入复用 `wmloop/storage.py`、`wmloop/experiments/_artifacts.py` 和
  `ContentAddressedStore`。单次输出与历史证据分开；不覆盖旧 source artifact。
- 合并 JSON 工具时保留已有 receipt 的 ASCII/UTF-8、尾随换行和哈希语义。
  历史 UTF-8 使用 `canonical_json_bytes(..., ensure_ascii=False)`；通过非 ASCII
  fixture 验证字节兼容，禁止用统一格式化悄悄改变 evidence identity。
- 错误用模块自己的 `ValueError`/`RuntimeError` 子类和稳定错误码；上下文附在
  错误码后，例如 `SELECTOR_PROBE_SEED_COVERAGE_MISMATCH:cloth_move`。
  禁止裸 `assert` 作为运行期契约、裸 `except` 后返回成功或吞掉实现失败。
- 业务诊断使用 `logging.getLogger(__name__)` 或所在模块已有日志接口，
  不新增 loguru 依赖。现有 CLI stdout JSON 是机器协议，日志走 stderr；
  不把日志混进 JSON，也不机械改写旧命令的输出协议。

## 4. 脚本、测试和依赖

- 新 Python 脚本使用 argparse、`main()` 与 main guard；Shell 使用 bash、
  `set -euo pipefail`、解析仓库根路径、前置输入检查，并支持只打印计划的
  `DRY_RUN`。模型资源路径由配置给出，不硬编码开发机路径。
- 新脚本按用途命名 `check_*`、`smoke_test_*`、`prepare_*`、`convert_*` 或
  `run_*`；有界组件测试放 `tests/`，不要为改名而迁移所有历史脚本。
- 用 `uv run --offline --frozen ...` 运行；依赖增减同步 `pyproject.toml` 和
  `uv.lock`。保持 Python 3.12 兼容，不把宿主系统解释器当控制面环境。
- pytest 不设置 `pythonpath`，也不在测试里加 `sys.path`；组件导入需要在
  checkout 外通过 `python -I` 验证。环境变量默认值在实例化/调用时读取，
  不用函数默认实参提前计算可配置路径。
- 测试固定 seeds，使用 tiny CPU 数据、fake adapter 或真实临时 SQLite/CAS；
  输出写 `tmp_path`。测试需验证边界行为和失败路径，不只复述实现。
- GPU 冒烟明确硬件、模型、split 和预算；CPU 测试成功不代表质量改善、
  真实 hook 执行、迁移成功或论文结果复现。

## 5. 文档和评审

- 新仓库文档用中文，现有英文文档无需整篇翻译；不要新增双语 README。
- 正式规范、接口与 schema 入 Git；运行报告、checkpoint、日志不入 Git。
  本地工作计划和总结放 `docs/plans/`、`docs/summary/`、`docs/logs/` 并忽略。
- 变更说明写清：原问题、最终行为、接口/版本影响、实际测试和未验证部分。
  不把理论方法、接口就绪、CPU 回放和真实模型效果混成同一层结论。
- 修改约定时同步 `AGENTS.md` 和本文；方法边界还需同步
  `control/agent_engineering_policy.py`，使后续生成的 probe/operator prompt
  继承同一套约束。prompt 是指导，实际准入仍由代码门禁执行。
- 文件夹命名和 `scripts/` 前缀以 [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) 为准；
  新目录必须先归属一个权限平面，不能用 `misc/`、`common2/`、`new/` 或日期
  目录绕过边界。

## 6. 改动工作流与公共能力索引

可维护性要求以 `AGENTS.md` 第 1.1 节为准。开始修改时，先沿现有公共入口追到
行为所属服务，搜索对应的 schema、调用方和测试，确定修复点；提交说明写明
公共接口变化与复用选择即可，不必为小改动新建一套设计文档。

| 需要的能力 | 优先查看和复用 |
| --- | --- |
| 原始路径检查、canonical JSON、hash、原子写、文件锁 | `wmloop/storage.py` |
| 运行时 Python 路径、子进程取消和进程组回收 | `wmloop/runtime_env.py`、`wmloop/process_lifecycle.py`（`execute/process_lifecycle.py` 仅兼容导出） |
| GPU 准入与租约、预算预约和结算 | `wmloop/execute/gpu_lease.py`、`wmloop/execute/budget.py` |
| 实验执行、队列调度、pipeline 状态与恢复 | `wmloop/execute/auto_experiment.py`、`wmloop/execute/experiment_scheduler.py`、`wmloop/execute/pipeline/` |
| CAS 与终态归档、诊断积累、经验检索 | `wmloop/archive/store.py`、`wmloop/execute/diagnostic_memory.py`、`wmloop/retrieve/index.py` |
| 正式判定、JSON 契约、命令与服务边界 | `wmloop/verify/`、`configs/schemas/`、`wmloop/control/commands/`、`wmloop/control/local_http.py` |

索引说明能力归属，不授予跨层 import 权限；调用仍须遵守 `AGENTS.md` 第 2、3 节。
缺少合适接口时，在所属模块提供最小明确的 API，再让现有调用方复用。不要
跨模块调用私有 helper、重新拷贝实现，或用 `getattr` 猜函数签名来维持多套路径。

涉及执行或持久化时，先列出输入校验、资源获取、执行、证据落盘、结算、积累和
恢复各自的责任方。新增状态必须说明其终态、重试条件和清理范围；成本未知不能
当作未执行。正常入口和恢复入口共用权威证据校验，不能只在首次执行时检查。

评审重点是：相同规则是否只有一个实现、入口是否仍以装配为主、异常是否保留
根因、用户能否通过公共入口完成操作，以及修改后既有契约能否继续读取。类型和
模块拆分应让这些关系更容易理解；不以新增类、文件或抽象层数量衡量工程质量。

## 开源交付补充

发布约束以 `AGENTS.md` 第 7 节为准，操作步骤见 `docs/RELEASING.md`。
`wmloop` 实现不依赖仓库脚本；脚本引用可安装的业务服务。HTTP 公共边界复用
`control/local_http.py`，路径检查复用 `storage.checked_path`。前端直接维护
`wmloop/web/` 源码，第三方来源和许可同步 `THIRD_PARTY_NOTICES.md`。
构建依赖与 uv 版本显式固定，已安装 wheel 的验证必须在仓库目录之外运行。

第二轮交付约束：API 令牌和浏览器状态遵循 `docs/LOCAL_HTTP_AUTH.md`，公开报告
只保留白名单数值。库中生成的子进程命令也不得依赖仓库脚本或固定 `.venv`。
控制面解释器路径保留虚拟环境语义，不通过 resolve 改成环境外的底层解释器。

用户验收发现的契约同步纳入维护规则：doctor 从包元数据读取 Python 支持范围；
损坏配置和不存在的源码目录必须报错；默认计划通过独立历史服务版本化，显式
输出仍保护旧文件。调度执行先验证 SQLite 绑定，使用内存快照并复用内核文件锁。
验证命令返回的证据必须可在进程退出后复核，失败只能清理自己新建的产物。
有界 CPU 验证设置子进程超时；取消和构造失败同样执行资源清理。配置在入口
校验实际类型，不把数字或布尔值强转为指标字符串；交互空行与 EOF 分开处理。
并发写入时，冲突检查和原子提交共用同一锁；重复生成问卷要保留用户答案。
计划默认输出跟随项目配置所在目录，显式项目根优先；状态目录解析复用同一入口，
禁止各命令维护不同默认规则。Archive/CAS 输出同样先用 `checked_path` 检查原路径。
