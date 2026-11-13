# VerdiWM 工程与研究闭环规范

本文件是 VerdiWM 仓库的约束权威来源。它把 VERDI 方法稿（完整题目与引用状态见 `docs/CITATION.md`） 方法部分的四阶段闭环落实为代码边界：**Characterize → Retrieve → Validate → Accumulate**。新增代码、脚本、测试和文档必须保持这些边界；与本文件冲突时，以本文件为准。

代码组织和写作细则见 [`docs/code_conventions.md`](docs/code_conventions.md)。本文是实现目标与修改约束，不代表所有历史路径已经满足论文的全部证据要求。

目录地图和新文件命名规则见 [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)；CI 由 `scripts/ci/check_repository_layout.py` 执行静态布局审计。

大文件拆分和公共能力收敛顺序见 [`docs/REFACTOR_ROADMAP.md`](docs/REFACTOR_ROADMAP.md)。

## 1. 适用环境与修改纪律

- 控制面使用 Python 3.12 和仓库锁定的 `uv.lock`。依赖必须离线可解析；不要新增未经 `uv sync --offline --frozen` 验证的包。
- 仓库必须能在任意 GitHub checkout 中运行：禁止提交 `/cs/...`、`/mnt/...`、个人 home、开发机 venv、私有数据集或模型路径。模型、数据、运行时 Python、输出、archive、CAS 和 GPU lock 都必须由 CLI 参数、配置或 `VERDIWM_*` 环境变量提供。
- 默认临时目录通过标准库和可配置环境变量推导；GPU lock 默认使用系统临时目录并允许 `VERDIWM_LOCK_ROOT` 覆盖，不要把某台机器的挂载点或 home 写进 artifact。公开 bundle 必须去除 checkout 绝对路径。
- 运行测试使用 `uv run --offline --frozen pytest -q tests`。不要用 `PYTHONPATH`、`sys.path` 补丁或系统 Python 掩盖 editable 安装问题。
- 先读现有模块和契约，再写代码。保持小而可审查的改动；不要为了让测试通过修改冻结评估器、held-out split、目标阈值、验证器或历史证据。
- 任何可离线测试的行为都要有 `tests/test_*.py` 用例；真实 GPU、模型权重和网络检索只能作为独立 receipt，不得伪装成 CPU 验证。
- 代码和脚本不得用 `print()` 输出日志、进度或报告。优先使用项目现有结构化日志；CLI 若已有 stdout JSON 兼容契约，必须在对应测试中保留并说明例外。
- 输出使用临时目录加原子替换；拒绝已有目录、符号链接和路径穿越。不得向模型、数据或权重只读目录写入。

### 1.1 可维护性与改动准则

- **先找归属，再写实现。** 新增能力前先搜索同类服务、schema、错误码和测试，确定唯一负责该行为的模块；变更说明交代复用了什么、为何现有接口不足。公共能力索引见 `docs/code_conventions.md` 第 6 节。
- **在能力所属层修复。** 多个入口遇到同类问题时，修复共享服务并让调用方复用；不得在 CLI、adapter、脚本和 pipeline 中分别复制路径校验、超时、结算或索引逻辑。兼容包装只做参数转换和委托，不维护另一套状态机。
- **保持清楚的职责和依赖。** 装配入口负责阶段顺序，领域服务负责业务规则，adapter 负责模型命令与输入输出转换。新增分支使模块承担另一项独立职责时，按职责拆分；遵守第 2.1 节已有文件上限，不靠搬成另一个巨型文件或新增 `utils2` 规避。
- **抽象须有实际调用需求。** 不为假想的模型或未来功能引入插件体系、基类层级、万能 runner 或动态分发；先使用明确函数、已有协议和必要的参数。只有共享语义一致时才合并，不能仅因几行代码相似就把不同权威边界塞进同一个 helper。
- **数据在边界校验，依赖显式传入。** 外部 JSON 按 schema 解析；跨模块接口明确输入、输出、状态和副作用，适用时使用已有类型或 dataclass。避免不断扩张的上下文字典、成组布尔开关、隐藏全局状态和重复散落的字段推断；不要求为统一形式重写稳定旧代码。
- **错误暴露根因，不以回退掩盖失败。** 只在能补充领域错误、执行清理或实施明确恢复协议的边界捕获异常。禁止捕获所有异常后返回空结果、默认成功，或临时加一个跳过门禁的参数；可选能力缺失须有明确状态和调用方处理。
- **临时适配有边界，兼容有契约。** 正式用户路径必须调用安装包内的服务，不依赖本地一次性脚本、手工改数据库或补写产物。模型 adapter 不接管 GPU lease、预算或 verdict。替换实现时清理本次引入的废弃分支、重复配置和未使用 import；保留仍有用户的兼容入口与历史证据读取能力。
- **按行为验证，按证据交付。** 回归覆盖用户可见行为、失败路径和跨阶段契约，优先扩展已有 fixture；不堆砌只重复实现或断言文案的测试。交付前按第 6 节验证，并说明接口影响、恢复语义和未验证范围，不将局部单测通过写成整个闭环通过。

## 2. 目录职责

| 路径 | 职责 |
| --- | --- |
| `wmloop/control/` | 用户意图、模型 portrait、能力、候选编译、campaign 状态和策略；不执行模型训练 |
| `wmloop/diagnose/` | 诊断探针和失败签名；只能路由与解释，不能修改 verdict |
| `wmloop/experiments/` | fingerprint、selector replay、验证报告和实验产物编排；不得把未结算结果写入 archive |
| `wmloop/geometry/` | IRG、Model IRG、效果记忆、迁移证书和探针演化数学；只接受有限数值和明确支持集 |
| `wmloop/retrieve/` | 归档经验索引、IRG 跨域检索和文献 staging；检索结果永远是候选/先验，不是目标效果 |
| `wmloop/propose/` | 受限 primitive、探针 DSL 和方法候选；不得直接启动 GPU 或改活动 checkout |
| `wmloop/primitives/` | 可执行 operator 定义、hook 和 capability 校验 |
| `wmloop/execute/` | 隔离 worktree、预算、GPU lease、screen → gate → confirm 执行和 receipt |
| `wmloop/verify/` | 冻结验证器、保护指标和结算门禁；唯一拥有 formal verdict authority 的层 |
| `wmloop/archive/` | CAS 与 SQLite 的 append-oriented settled archive；只接收终态 receipt |
| `configs/schemas/` | JSON Schema；新增 artifact 必须同步 schema、验证器和测试 |
| `docs/` | 架构、方法到代码映射、实验协议；`runs/`、`results/`、`artifacts/` 是运行产物，不入 Git |

### 2.1 已拆分组件的维护边界

- `wmloop/cli.py` 只保留公共命令装配、兼容别名和错误呈现；参数声明在 `control/commands/parser.py`，业务命令按 project、run、research、community、training 等分组。命令服务通过参数接收外部依赖，禁止反向 import `wmloop.cli`。
- `control/onboarding.py` 保留扫描入口；内部实现属于 `control/onboarding_scan/` 的 files、source、assets、runtime、evaluator、bindings、readiness、reports、io。目录使用 `onboarding_scan`，避免与兼容模块 `onboarding.py` 同名遮蔽。
- `orchestrator_training_eval_smoke.py` 保留训练评估装配与 CLI；执行器、规划、诊断路由、探针证据、结算、产物分别在 `execute/training_smoke/`。冻结判定仍委托 `verify`。
- `execute/autonomous_pipeline.py` 保留阶段顺序、异常边界和兼容入口；输入预算、状态锁、journal、catalog 和检索上下文复用 `execute/pipeline/`，不能各自重写恢复协议。
- `retrieve/mechanism_discovery.py` 装配发现/重新比较流程；查询、文献来源、提取、profile 和比较分别属于 `retrieve/discovery/`。
- `experiments/portable_knowledge_graph.py` 保留公开构建/导出/查询 API；图结构和各类语义投影放在 `experiments/knowledge/`。
- 组件不能导入自己的兼容入口，也不能形成循环依赖；上述组件单文件不超过 800 行，达到上限时按职责拆分，禁止压缩语句绕过。`scripts/ci/check_repository_layout.py` 会检查这些规则，并禁止 `geometry/retrieve` 导入 `execute/torch`。
- `control`、`geometry`、`retrieve`、`experiments` 的包级公开 API 保持惰性导出；不要在 `__init__.py` 加回整条工作流的 eager import。
- 文件 hash、canonical JSON 和原子写复用 `wmloop.storage`。历史 UTF-8 receipt 显式传 `ensure_ascii=False`；默认 ASCII 编码不能机械替换旧字节格式。不得因移动类而改变 artifact 中稳定的内置 extractor 身份。
- 默认资源路径必须在调用或实例化时读取环境变量；dataclass 使用 `field(default_factory=...)`，普通函数使用 `None` 加运行时解析，不能在 import 时冻结 `VERDIWM_*`。
- 解释器路径使用 `runtime_env.runtime_python_path`，保留虚拟环境 launcher；不要用 `resolve()` 将其切换成系统解释器。Python 支持范围读取发行包的 `Requires-Python`，不能在 doctor 另写版本常量。
- 通用 CPU 子进程信号取消与进程组回收复用 `wmloop.process_lifecycle`；`execute/process_lifecycle.py` 仅兼容导出。检索与 CPU 评估不得经由共享工具间接加载 execute 或 torch。
- 默认研究计划通过 `control/research_plan_history.py` 保存 digest 版本并原子更新 latest；显式输出仍拒绝覆盖冲突。阻塞计划的界面建议必须指向解决阻塞项，不能建议直接执行。
- 文件的“检查冲突再写入”必须处于同一锁内；原子替换本身不阻止两个写入者覆盖彼此。接入问卷不得覆盖用户已填写的答案。CLI 状态目录统一由 `control/commands/inputs.resolve_state_root` 解析，显式参数优先于项目配置，不能通过与默认值相等来判断是否显式传参。

## 3. 四阶段闭环与权威边界

### Characterize：构造 Optimization Fingerprint

共享探针在推理时对行为输入或 inference setting 做小剂量、可逆干预；固定权重、目标和匹配的 context/seed。采集配对响应、协方差、支持掩码和 adapter hook，形成 `J_X / G_X / Σ_X / c_X` 及 `Model IRG`。探针描述模型响应，不能直接证明 operator 的收益。

当前入口包括 `wmloop/experiments/*fingerprint*.py`、`wmloop/geometry/model_irg.py` 和 `wmloop/diagnose/`。所有响应必须有限；不完整支持必须显式记录并在迁移时 abstain。verdict probe 与 diagnostic probe 分离，诊断坐标不得进入正式 utility/protected metric。

### Retrieve：IRG 引导的 operator 与跨域检索

`wmloop/geometry/model_irg.py` 只按兼容坐标、goal schema、support 和不确定性做排名。`wmloop/retrieve/irg_guided_discovery.py` 先把瓶颈变成带证据和 ceiling boundary 的跨域 lens，再交给 typed literature/mechanism staging。`wmloop/retrieve/index.py` 只能索引已经 settled 且 CAS 可复核的 receipt/result。

检索的输出是 ranking prior 或候选 work order。它不能写 target effect、发 transfer license、改变 verifier，也不能把论文摘要、LLM 判断、screen 或单次运行升级成 archive evidence。跨 backbone 的距离和邻居结果必须保留 `distance_version`、shared support、证据引用和 abstention 原因。

### Validate：冻结验证器下的 paired target experiments

候选必须先通过 compile、capability、support overlap、IRG alignment 和成本检查，再用相同初始化、数据 split、context、seed、verifier 的 matched control 做低成本 screen。正向候选需要 held-out confirmation、额外 seeds、保护指标全通过，且 utility gain 的 lower confidence bound 大于预声明阈值。utility 改善不能抵消 protected regression；null、harmful 和 interaction 也必须可结算。

验证权威只在 `wmloop/verify/` 和终态 receipt。`wmloop/execute/` 负责执行和收集证据，不能自行宣布效果；`wmloop/retrieve/`、LLM executor 和 proposal planner 都没有 verdict authority。

### Accumulate：沉淀、复用与探针升级

只有 `settlement_state=settled` 且 receipt、verdict、failure context、evaluator、hypothesis、implementation 都可由 CAS/hash 重建的记录，才能调用 `ArchiveStore.record_settled_trial`。`EffectMemory` 保留 `confirmed`、`null`、`rejected`、`interaction`；不得丢弃负向结果或只保留正向样本。

迁移先由 `wmloop/geometry/transfer.py` 生成证书；任何证书项失败都返回 `abstain`，不能降级成 licensed prior。跨域结果只能影响候选排序，目标实验仍必须重新验证。原始 source 记录不可被 target 结果覆盖；source→target 只追加新记录。

可靠的 nearby IRG opposing effects 才能触发探针候选。CPBE/`wmloop/experiments/probe_evolution.py` 的候选必须绑定 counterexample、supported hooks、固定 outcome schema 和 successor campaign；只有 held-out replay 改善 selection/regret 且风险不回归，才允许写 admission artifact。单个孤立失败、代码执行成功或校准下降都不能升级探针。

## 4. 数据、检索和证据不变量

- 所有 artifact 使用 `schema_version`、`artifact_type`、content-addressed refs 和 `claim_boundary`。schema 校验不接受额外未声明字段。
- 归档检索在写入和读取两端都要核对：result 内容、probe/model/runtime/failure signature/primitive/metric、receipt 的 stage/verdict/artifact ref，以及 SQLite trial/receipt 的三方绑定。
- selector probe admission 要求 baseline/candidate 的 target 集合、seed 集合完全相同，同一 target/operator 的 label 不变；否则返回稳定错误并不生成输出。改选另一 operator 可以改变其实际 label，必须使用同一冻结标签来源。
- 每个实验产物报告 shape、dtype、finite、support、seed、split、verifier/evaluator digest 和状态。`partial`、`blocked`、`abstain` 不得渲染成 `ready`。
- archive 是事实来源；JSON/CSV/Markdown 是只读派生视图。派生视图被篡改或 CAS 缺失时，检索必须过滤掉它。
- dispatcher claim 必须核对 JSON 队列与 SQLite 中已确认的执行绑定，并只执行核验后的内存快照；进程互斥复用 `storage.exclusive_file_lock`，不得删除锁文件处理崩溃。
- 验证报告返回的 CAS 引用、输入和 receipt 必须在命令结束后仍可核验；不得返回已删除临时目录中的证据。失败清理只作用于本次创建的输出。
- 有界 CPU 验证子进程必须有超时并回收进程；取消操作也清理本次未完成输出。HTTP 服务构造失败必须释放已绑定端口及凭据目录，不能依赖垃圾回收。资源失败路径须有故障注入测试。

## 5. 禁止事项

1. 用 retrieval 相似度、LLM 断言、screen、loss 下降或单 seed 替代 frozen verifier。
2. 修改活动 campaign 的目标、阈值、verdict probe、held-out split 或 evaluator 来制造通过。
3. 把诊断 probe 注入 utility/protected metric，或让 probe admission 改写历史效果。
4. 用 truthy 字符串代替布尔 gate、用 NaN/Infinity 绕过阈值、用缺失支持默认为支持。
5. 在未 settled 的 trial、未绑定 CAS 的 receipt 或未经 target replay 的 transfer 上更新 selector/archive。
6. 在多个模块重抄 probe coordinate、failure signature 或 operator capability；应从 canonical registry/schema 导入。
7. 在 `wmloop/diagnose`、`retrieve` 或 proposal 代码中直接 import GPU runtime、启动训练或写模型 checkout。
8. 提交 checkpoints、数据、私有路径、密钥、运行日志和生成的 `runs/`、`artifacts/`、`results/`。

## 6. 验证清单

- 改动 Python：`uv run --offline --frozen python -m py_compile ...`
- 改动闭环契约：相关窄测 + 负向门禁测试。
- 改动 schema/artifact：schema validation、CAS round-trip、篡改/缺失证据测试。
- 改动 probe/retrieval：完整 paired replay、support/seed 覆盖测试、跨域排序与 target-side abstain 测试。
- 完成前：`uv run --offline --frozen pytest -q tests`，并记录实际命令与失败原因；GPU/网络验证另附 receipt，不与 CPU 结果混写。
- 若未改依赖但离线构建缓存缺失，可在确认现有 editable 环境有效后用 `uv run --offline --frozen --no-sync ...` 验证代码，并报告构建限制；不能据此宣称 `uv sync` 或打包通过。


## 7. 开源发布与服务边界

- 发布流程见 `docs/RELEASING.md`。当前支持 Linux/WSL2、Python 3.12，发布状态为 Beta。未经测试的平台或 GPU 效果不得写成支持承诺。
- 首次外部安装允许联网但必须 `uv sync --locked`；开发验证保持离线。CI 固定 uv 版本、构建依赖版本，控制面不隐式安装 `torch-tests`。
- `wmloop/` 禁止导入 `scripts`；可复用服务必须放进安装包。wheel/sdist 与源码快照必须同时检查，并在 checkout 之外的新环境执行实际 schema、资源和服务验证。
- 路径信任校验复用 `wmloop.storage.checked_path`，在 resolve 前检查原路径和父目录；不能先 resolve 再声称拒绝了符号链接。它不替代进程隔离，也不承诺抵抗恶意并发替换目录。
- HTTP 服务共用 `control/local_http.py`，仅允许回环监听，保留 Host/Origin 校验、JSON framing/大小限制与读取超时。禁止换回无保护的 HTTPServer 或绕过构造器检查。
- 发布排除凭据文件、私有文档、运行产物与缺少来源的构建包。扫描所有有界文本，不依赖扩展名白名单；报告不得包含密钥原文，不得将模式扫描宣称为无秘密证明。
- 已跟踪的非公开文件必须阻断发布；默认导出遇到未跟踪公开源码必须报错；开发审阅通过显式 `--include-untracked` 生成完整快照，不修改真实 Git index，不静默漏文件。
- 当前文件审计不批准 Git 历史。交付使用不带历史的审阅快照；现有 GitHub 仓库已公开，不得据此重建或强推覆盖其历史。保留历史发布时运行全可达对象检查，未扫描/有问题对象必须阻断，不自动重写历史或推送。
- 工作台源码直接维护在 `wmloop/web/`。新增第三方前端或复制源码必须提供来源、依赖锁、构建步骤及许可文件，同步 `THIRD_PARTY_NOTICES.md`；不能只提交压缩 bundle。
- 所有 HTTP API（包括只读）必须验证每次启动随机生成的 bearer token；固定界面资源可公开。令牌只写入启动用户私有的临时文件，不写日志、URL 或 Cookie；浏览器仅用 sessionStorage 与 Authorization header。共享用户与 root 的剩余边界见 `docs/LOCAL_HTTP_AUTH.md`。
- 发布测试报告采用数值白名单重建，不能复制原始 pytest XML、主机名、stdout/stderr 或异常中的路径。对公开附件也执行审计。
- 子进程命令与直接 import 一样要遵守可安装边界。effect-label 队列用 `python -m wmloop.experiments.formal_visualization`，控制面 Python 和模型 Python 分开配置，禁止推断 checkout 的 `.venv`。
- 依赖安全升级允许先准备必要的官方发行物缓存，但必须在提交前完成 `uv sync --offline --frozen` 与完整测试；联网公告查询单独记录，失败或命中不得当成通过。依赖版本、解释器支持线变化须同步 CI 与文档。
- GitHub Actions 固定审核过的完整 SHA，使用只读 token、不持久化 checkout 凭据。远端保护规则必须实查；本地配置不等于远端生效。
