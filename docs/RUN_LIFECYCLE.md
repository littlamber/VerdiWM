# 研究轮数、停止与恢复

## 在普通终端选择轮数

普通 `/plan`、`/run` 和 `verdi research plan/run` 都支持 `--rounds N`。
N 必须是正整数，默认 1；它表示最多研究轮数，预算、失败或取消可能让任务提前
结束。无需另找 daemon 命令。一个多轮任务仍对应一个 campaign，统一查看状态
和请求取消。候选数、训练步数和后台轮询次数都不是研究闭环的轮数。

已完成项目接入后，可以直接使用：

```text
verdi> /plan "提升长程一致性" --rounds 2 --budget 2gpu-hours --stop-on-success
verdi> /run
  研究计划 … 最多 2 轮；总预算 2 GPUh；正式确认收益后提前停
  最多运行轮数 … [2]:
  确认创建并执行这个 campaign？ [y/N]: y
```

`/run` 无参数时会展示本会话最近的计划，尚无会话记录时使用项目默认计划。
轮数提示直接回车保留已计划值；填写其他正整数会先保存和展示派生计划，再询问
一次执行确认。也可以直接输入 `/run --rounds 3` 或 `/run --stop-on-success`。
这些交互操作都会先预览，拒绝确认时保留新计划，下一次 `/run` 可以继续，
无需手动复制路径。兼容旧操作：在轮数提示输入 `y` 表示确认刚刚展示的当前计划，
输入 `n` 取消。

终端还支持明确的自然语言指令，例如“提升长程一致性，跑两轮”、
“最多跑3轮，达到目标就停”或“run two rounds and stop on success”。
“达标就停”也可单独设置正式成功提前停止。它们只准备可审阅计划，不启动 GPU。
只指定轮数时沿用已有研究目标和计划绑定；尚无目标时会要求补充目标。
“跑两轮还是三轮”、非正整数等歧义输入不会修改目标或擅自选择。其他自由文本
继续作为研究目标提示；不宣称能理解任意自然语言的实验控制要求。

脚本使用相同的计划与执行服务：

```bash
verdi research plan --goal "improve consistency" --rounds 2 --budget 2gpu-hours
verdi research run --plan .verdiwm/research-plan.json
verdi research run --plan .verdiwm/research-plan.json --confirm
```

`research plan` 和不带 `--confirm` 的 `research run` 都不启动 GPU。
在 `research run` 中显式修改轮数或提前停止策略时，会持久化派生计划，
输出 `plan_path` 和策略；带 `--confirm` 表示同时授权本次明确给出的参数并执行
派生计划。希望分步审阅时，先不带确认预览，再使用返回的计划路径确认。
旧计划保持可读，并默认沿用原单轮行为；执行没有显式覆盖时保留计划中的策略。

## 跨轮总预算

`--budget` 是所有轮次共享的总 GPU 时预算，不是每轮重新发放。
多轮执行复用已有控制器和 pipeline，为每轮生成独立输入与输出，共享 Archive、
CAS、检索索引和全局预算账本。后续轮次可读取前轮已结算证据；未结算结果不能
当作经验。普通有界多轮不会仅因诊断签名重复而提前结束。

同一状态目录可用于后续独立 campaign。新控制器的 campaign、probe、candidate
和 evaluator 身份绑定冻结的控制器输入及轮数，避免复用模板时与知识库中的旧
trial 碰撞。相同控制器恢复时继续读取既有 `inputs/input.lock.json`，保持原 ID
和锁文件；不会重新命名或覆盖旧实验。知识复用仍需正常核验 Archive/CAS receipt。

## 每轮联网检索与经验反馈

普通研究任务的联网文献检索在当前轮诊断、已结算经验检索之后执行。查询保留
用户目标，以目标中的任务词和当前失败征象限定方向，再结合已有 IRG 检索视角和
经过 Archive/CAS 核验的经验。内部的种子轮换、推理预算等执行策略不作为论文
关键词；上一轮的旧征象不能代替本轮新诊断。

每轮的 `pipeline/retrieval/literature-query-plan.json` 保存查询计划：原始目标
摘要、当前诊断身份、查询词、IRG 视角和经验来源引用。经验中的正向、负向与
零差异观察保留在上下文中；只有相应机制和指标可作为检索线索，执行 `PASS`
不表示该机制已经改善模型。查询中的领域和诊断约束用明确的检索表达式组合，
避免仅靠同名词把视频长程问题检索成其他领域的无关论文。

联网结果位于同轮 `pipeline/literature/`。`manifest.json` 的 `delivery`
说明本次是请求检索还是恢复旧结果；`search-trace.json` 保留原请求的 URL、
HTTP 状态、开始与结束时间、响应哈希及失败原因，区分网络返回和缓存；
`records.json` 保留论文标题、摘要和来源链接。恢复旧结果不等于重新联网。
网络不可用或没有可执行文献候选时，流程保留可用的已配置候选，并明确记录降级。

每轮最多执行两个查询。优先查询结合目标、诊断和可用经验视角；只有服务成功
返回零条结果，才去掉附加视角，保留任务领域与当前诊断再查一次。网络错误不会
触发这一放宽；失败根因保留在请求记录中。

中文目标会保留原文；检索器不会自动翻译。目标缺少英文任务词时，可用模型领域
中的明确检索词辅助冷启动；只有内部适配器名称、无法确定任务领域时，查询计划会
记录跳过原因 `LITERATURE_TASK_TERMS_UNAVAILABLE`。此时可在研究目标中补充
`video prediction` 等实际任务关键词，再预览新计划；不要把内部部署名称当论文词。

是否更接近优化方向，需要结合返回论文的标题、摘要、可用机制和当前诊断审阅。
查询发生变化、返回新论文或检索排名靠前，都不能保证每轮相关性单调提升，也不能
代替目标模型的配对验证。未变化的诊断可能产生相同查询；这时仍可复用已结算
经验，不会为了让查询看起来不同而添加无关词。

## 正式达标停止

`--stop-on-success` 启用“正式确认目标收益后提前停止”；默认关闭，
`--no-stop-on-success` 可显式关闭。判断来自冻结 verifier 与本轮 confirm
绑定的正式 CAS 证据，不能用 pipeline `PASS`、screen 通过、诊断变化、loss
下降或 LLM 自评代替。evaluator 可通过相对路径 `frozen_goal` 绑定预先审阅的
goal v2 契约。没有正式目标契约或本轮未产出有效正式确认时，任务会继续受轮数、
预算、失败和用户取消等条件约束，不会伪报达标。

例如，已接入的 `eval/evaluator.json` 可以保留原有评测字段并增加以下绑定字段：

```json
{
  "scheduler_template": "candidate-batch.json",
  "frozen_goal": "goals/consistency-v2.yaml"
}
```

这是绑定片段，不是完整 evaluator 示例。两个相对路径都从 evaluator 文件所在
目录解析；上例的正式目标位于 `eval/goals/consistency-v2.yaml`。
目标需按 [正式验收策略](VERIFICATION_POLICY.md) 声明主指标、保护指标、
`minimum_effect`、`seeds_accept` 和动作跟随门禁，并在运行前冻结。用户的自由文本
研究目标用于规划；只有相应的冻结目标与本轮正式确认 CAS 证据才能决定达标停止。
开启 `--stop-on-success` 不会为仅能诊断的 evaluator 自动补齐训练、配对实验或
正式判定能力。计划预览会显示缺少 `frozen_goal` 的警告；结束后的结果卡会显示
停止原因、已完成轮数，以及正式目标确认状态和未确认原因。

`completed` 表示执行流程结束，不等于模型已经提升。最终效果仍须核对冻结
verifier、保护指标、held-out confirmation 和对应正式证据。

多轮要求已确认的 evaluator 能物化 `scheduler_template`，候选包含
`screen`、`gate`、`confirm` 阶段；缺少时计划会明确阻塞。
单独用于诊断的 evaluator 不一定具备这个契约。新用户仍须完成模型资源、
运行时和正式评估的接入，不能照抄其他模型配置或修改冻结评估器绕过门禁。
完整资源绑定见[接入文档](ONBOARDING.md#continuous-immutable-evolution)。

## 兼容的独立控制器

已有 `verdiwm-evolution-daemon` 继续支持高级独立运行，其参数如下：

| 参数 | 含义 |
| --- | --- |
| `--max-iterations 2` | 本次控制器最多进行两次研究迭代；其他停止条件可以更早触发 |
| `--max-iterations 0` | 不设轮数上限，仍受全局预算和其他停止条件约束 |
| `--total-budget-gpu-hours 2` | 所有轮次共享的 GPU 时预算，不是每轮重新获得两小时 |
| `--max-failures 3` | 连续失败达到三次后阻塞停止 |
| `--max-no-information 3` | 成功轮次的诊断签名连续三次与前一次相同后停止 |
| `--inner-max-cycles` | 单轮内部的轮询上限，不能用来设置研究轮数 |

独立控制器的 `--max-no-information` 按诊断签名是否重复判断，不表示正式收益
收敛；设为 `0` 可关闭。普通 `--rounds N` 入口已关闭这一诊断重复停止规则，
其正式成功提前停止由 `--stop-on-success` 单独控制。

## 停止、查看和恢复

一般用户使用 `verdi research run --confirm` 后无需另外启动 dispatcher。选择
`--queue-only` 只入队并返回后，可在需要独立 worker 时运行：

```bash
verdiwm-campaign-dispatcher --state-root "$VERDIWM_STATE_ROOT" \
  --max-cycles 1 --max-parallel 1
```

该公开命令的 `--state-root` 与 `verdi run --state-root` 含义相同：参数是
用户状态根目录，CampaignStore 位于其 `campaigns/` 子目录。不要把
`campaigns/` 子目录本身再传给这个公开命令。若未设置环境变量，可明确填写
相同的状态根路径，例如 `--state-root "$HOME/.local/state/verdiwm"`；提交任务、
查看状态和取消任务都要使用同一状态根目录。`--max-cycles` 是本次 worker
最多扫描待处理队列的次数，`--max-parallel` 是同时调度的 campaign 数；两者都
不是研究轮数，也不会延长 campaign 的冻结轮数或总预算。普通使用时优先由
`verdi research run` 直接调度，只有需要把入队与执行分开时才单独运行 worker。

通过 CampaignStore 调度的任务可在另一个终端请求取消：

```bash
verdi cancel CAMPAIGN_ID
verdi status CAMPAIGN_ID
```

使用了自定义状态目录时，两条命令均传入相同的 `--state-root STATE_ROOT`。
交互终端对应 `/cancel CAMPAIGN_ID` 和 `/status CAMPAIGN_ID`。
取消请求写入持久状态，执行器负责停止本次子进程并释放租约；查看状态时应同时
检查执行/取消 receipt，不能仅凭发出了取消命令就断言实验已经清理完毕。

前台调度 campaign 时，`Ctrl-C` 或 `SIGTERM` 也会请求取消活动任务。
执行器先请求子进程退出，必要时强制回收已跟踪的后代，再释放租约并返回；
仅查看状态时，`Ctrl-C` 不会取消其他进程正在执行的任务。

通过普通 `research run --rounds N` 或 `/run --rounds N` 启动的多轮任务已纳入
campaign 状态与取消接口。独立手动启动的 evolution daemon 不自动注册到
CampaignStore。直接在前台启动独立 daemon 时，
用 `Ctrl-C` 请求停止；在后台启动时向它的控制器 PID 发送 `SIGTERM`，并查看
对应 state root 的 `status.json`。停止请求会阻止新轮次；控制器在当前 pipeline
调用返回后记录停止。本地子进程可以在调用内响应取消并被回收，因此这不保证
当前轮次正常完成。已经预约预算、尚未结算 receipt 的工作仍需按账本核对成本，
不能把中断当作零消耗。
不要把 `/cancel` 当作任意独立后台进程的停止接口。

`/exit` 只退出交互终端，不代替对已排队任务或其他控制进程的取消。
`/resume` 对进行中的 campaign 显示进度；没有活动 campaign 时，它会回到
计划审阅和确认。当前计划已经对应终态任务时，首页会建议 `/resume`，后者展示
已有结果和恢复步骤，不再索要执行确认；新生成的计划不会被其他旧任务遮住。
它不会自动重启任意失败的训练或独立 daemon。

`blocked`、`failed`、`completed` 和 `cancelled` 均是本次 revision 的终态。
对相同 revision 再次确认 `research run` 会返回已有终态及下一步，`blocked` /
`failed` 仍返回非零退出码，不自动重新消耗实验预算。修复源码或策略使 revision
改变后，可使用原计划再次运行；没有活动调度 claim 时，系统创建新 revision，
并在 `revision_history` 中保存旧终态记录、调度快照和既有证据的引用。若修改了
计划绑定的模型、数据或 evaluator，先重新生成计划并审阅冻结输入；输入漂移仍
会阻断执行。希望相同输入下创建一次新尝试时，重新 `/plan` 再 `/run`。
不要尝试取消已经 `blocked` 的终态任务，也不要手工清理数据库或改写旧 lock。

独立 evolution 的 `stopped` 或可恢复 `exhausted` 状态，使用相同命令和 state
root 继续未完成的当前迭代；若停止前最近一轮已完成，则进入下一轮。
已经达到总轮数上限的 `completed` 控制器不会用相同 state root 自动开启额外轮次。
总轮数也属于冻结输入；更改它、evaluator、模型资产、重试策略或全局预算
需要新 state root。保留共享证据库以供后续检索。

旧版本曾把未显式传入的共享存储路径以 `None` 记录在输入锁中，使各轮默认存储
未必真正共享。修正后的控制器会在冻结前将 Archive、CAS、检索索引和预算账本
解析为明确的共享路径。因此，带这类旧 `None` 输入锁的任务不能承诺无损原地
升级或自动迁移；继续使用同一个 state root 可能报 `EVOLUTION_INPUT_MISMATCH`。
不要手工修改旧锁文件或覆盖原结果。请使用新的 state root 和 output root 启动新
任务，保留旧 Archive 与 CAS 的实际路径，并通过 `--archive-db`、`--cas-root` 等
参数显式绑定希望继续复用的证据库。旧记录仍须经过正常的证据读取和结算检查，
不会因路径复用被重新赋予正式达标权威。
